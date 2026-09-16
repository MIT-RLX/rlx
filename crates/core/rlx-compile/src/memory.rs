// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Memory planning — liveness analysis and buffer assignment.
//!
//! This is the XLA feature that no other Rust framework has. It computes
//! which intermediate tensors have non-overlapping lifetimes and assigns
//! them to the same memory, minimizing total arena size.
//!
//! The output is a [`MemoryPlan`] that tells the runtime exactly how
//! large the arena should be and where each tensor lives within it.

use rlx_ir::op::BinaryOp;
use rlx_ir::{Graph, NodeId, Op};
use std::collections::HashMap;

/// Extra bytes reserved after Input/Param/Constant slots so a kernel
/// that writes slightly past its logical tensor size cannot stomp the
/// next arena slot (e.g. small bias tensor adjacent to input_ids).
const BOUNDARY_TAIL_GUARD_BYTES: usize = 128;

fn boundary_min_slot_bytes(op: &rlx_ir::Op, alignment: usize) -> usize {
    if matches!(
        op,
        rlx_ir::Op::Input { .. } | rlx_ir::Op::Param { .. } | rlx_ir::Op::Constant { .. }
    ) {
        alignment.max(1)
    } else {
        0
    }
}

fn boundary_tail_guard(op: &rlx_ir::Op, alignment: usize) -> usize {
    if matches!(
        op,
        rlx_ir::Op::Input { .. } | rlx_ir::Op::Param { .. } | rlx_ir::Op::Constant { .. }
    ) {
        alignment.max(BOUNDARY_TAIL_GUARD_BYTES)
    } else {
        0
    }
}
/// Identify ops whose output is a *view* of an existing buffer — no
/// copy needed, no separate arena slot. Returns the parent input index
/// and the byte offset of the view within the parent.
///
/// Borrowed from MAX's "view-vs-copy" pattern.
/// The hard case (strided narrow on a non-outermost axis — e.g. BERT
/// QKV split) requires kernels that consume strided inputs and is
/// deferred. This function only catches the safely-elidable cases:
///
///   - **`Reshape`**: pure metadata; data layout is identical.
///   - **`Cast`** with `src dtype == dst dtype`: pure metadata.
///   - **`Narrow` on axis 0**: contiguous sub-slice of the parent;
///     offset = `start * size_of_inner_in_bytes`.
fn pure_view_offset(graph: &Graph, node: &rlx_ir::Node) -> Option<(NodeId, usize)> {
    match &node.op {
        Op::Reshape { .. } => Some((node.inputs[0], 0)),
        // KvAppend's output aliases `cache` (input 0) at offset 0 — the planner
        // gives it cache's slot (no copy). Unlike a pure view it still WRITES a
        // row, so `is_pure_view` (the backend Nop predicate) excludes it below.
        Op::KvAppend { .. } => Some((node.inputs[0], 0)),
        Op::Cast { to } => {
            let parent = graph.node(node.inputs[0]);
            if parent.shape.dtype() == *to {
                Some((node.inputs[0], 0))
            } else {
                None
            }
        }
        Op::Narrow {
            axis,
            start,
            len: _,
        } if *axis == 0 => {
            let parent = graph.node(node.inputs[0]);
            // inner = product of dims after axis 0
            let inner_elems: usize = (1..parent.shape.rank())
                .map(|i| parent.shape.dim(i).unwrap_static())
                .product();
            let dt_bytes = parent.shape.dtype().size_bytes();
            Some((node.inputs[0], start * inner_elems * dt_bytes))
        }
        _ => None,
    }
}

/// Public predicate for backends — true iff this op should compile to
/// a Nop because its output aliases a parent buffer (the memory
/// planner has already aliased its slot).
pub fn is_pure_view(graph: &Graph, node: &rlx_ir::Node) -> bool {
    // KvAppend aliases its parent's slot (so it's in `pure_view_offset`) but is
    // NOT a no-op — the backend must still encode the single-row write.
    !matches!(node.op, Op::KvAppend { .. }) && pure_view_offset(graph, node).is_some()
}

/// True iff this node is a bank transpose that a backend folding
/// `Transpose -> GroupedMatMul` will never execute.
///
/// `Op::GroupedMatMul` wants its expert bank as `[E, K, N]` while GGUF stores
/// banks as `[E, N, K]`, so a dense MoE layer emits `Transpose(bank, [0,2,1])`
/// ahead of the GEMM. A backend that can run the GEMM with B transposed skips
/// that node — but the plan is built first, so without this the arena still
/// reserves the transposed bank: a full second copy of the weights, per bank,
/// that is never written and never read. On GLM-5.3-Flash that is ~1.88 GB per
/// bank and ~5.6 GB per layer of dead arena, on nodes sized to hold one copy.
///
/// The condition is that EVERY reader folds. One transposed bank feeds all
/// `top_k` grouped matmuls of a layer, so requiring a single use rejects the
/// common case; but a reader that still needs the tensor materialized must keep
/// it, or it reads an unallocated buffer.
///
/// Shared with the backend deliberately: the planner and the compiler have to
/// agree exactly about which nodes vanish, and the only way to guarantee that
/// is one predicate. Gated by [`MemoryPlanOptions::elide_bank_transposes`], so
/// a backend that does not fold keeps its buffers.
pub fn is_elidable_bank_transpose(graph: &Graph, node: &rlx_ir::Node) -> bool {
    is_elidable_bank_transpose_gated(graph, node, false)
}

/// [`is_elidable_bank_transpose`], optionally requiring every reader to be a
/// SMALL-`m` grouped matmul.
///
/// Backends differ in where they can consume a transposed bank. A CPU
/// `sgemm_bt` handles any `m`; Metal's transposed kernel is a decode-path GEMV
/// (`m <= 4`) and its prefill kernel has no variant, so a transpose feeding a
/// prefill-size matmul there must stay materialized. `require_small_m` lets a
/// planner ask the question its backend will actually answer — the two have to
/// elide exactly the same set, or one drops a buffer the other writes.
pub fn is_elidable_bank_transpose_gated(
    graph: &Graph,
    node: &rlx_ir::Node,
    require_small_m: bool,
) -> bool {
    if !matches!(&node.op, Op::Transpose { perm } if perm.as_slice() == [0, 2, 1])
        || node.shape.rank() != 3
        || node.shape.dtype() != rlx_ir::DType::F32
    {
        return false;
    }
    let mut readers = 0usize;
    for n in graph.nodes() {
        for (slot, &i) in n.inputs.iter().enumerate() {
            if i != node.id {
                continue;
            }
            readers += 1;
            let folds = slot == 1
                && matches!(n.op, Op::GroupedMatMul)
                && n.shape.dtype() == rlx_ir::DType::F32;
            if !folds {
                return false;
            }
        }
    }
    if require_small_m {
        // Every reader must also be small enough for the decode-path kernel.
        for n in graph.nodes() {
            if !n.inputs.contains(&node.id) {
                continue;
            }
            let small = rlx_ir::shape::grouped_matmul_dims(
                &graph.node(n.inputs[0]).shape,
                &graph.node(n.inputs[1]).shape,
                Some(&n.shape),
            )
            .map(|gd| gd.m <= SMALL_M_GROUPED)
            .unwrap_or(false);
            if !small {
                return false;
            }
        }
    }
    // A transpose nothing reads is dead anyway; leave it to DCE rather than
    // claiming it here.
    readers > 0
}

/// The `m` below which a grouped matmul takes a decode-path GEMV. Mirrors
/// `rlx-metal`'s `encode_grouped_matmul` threshold; shared so the planner and
/// that encoder cannot disagree about which nodes vanish.
pub const SMALL_M_GROUPED: usize = 4;

/// True iff this node is a 2-D operand transpose that a backend folding
/// `Transpose -> MatMul` into GEMM trans-flags will never execute.
///
/// Matmul backward emits `Transpose(operand) -> MatMul` for `dA = g·Bᵀ` and
/// `dB = Aᵀ·g`, so a training graph carries one of these per weight. Folding it
/// into a `cblas` trans flag skips the copy, but the plan is built first, so the
/// arena still reserves a transposed copy of every weight it folds — never
/// written, never read.
///
/// Sole use, matching the fold: a transpose read by anything other than the one
/// matmul must stay materialized.
pub fn is_elidable_matmul_transpose(graph: &Graph, node: &rlx_ir::Node) -> bool {
    if !matches!(&node.op, Op::Transpose { perm } if perm.as_slice() == [1, 0])
        || node.shape.rank() != 2
    {
        return false;
    }
    let mut reader: Option<&rlx_ir::Node> = None;
    for n in graph.nodes() {
        for &i in &n.inputs {
            if i != node.id {
                continue;
            }
            if reader.is_some() {
                return false; // more than one use
            }
            reader = Some(n);
        }
    }
    let Some(mm) = reader else {
        return false;
    };
    // The fold only fires for a 2-D F32 matmul, so only then is the buffer dead.
    matches!(mm.op, Op::MatMul)
        && mm.shape.dtype() == rlx_ir::DType::F32
        && mm.inputs.len() >= 2
        && graph.node(mm.inputs[0]).shape.rank() == 2
        && graph.node(mm.inputs[1]).shape.rank() == 2
}

/// Either kind of transpose the CPU backend folds away — what the planner asks.
pub fn is_elidable_folded_transpose(graph: &Graph, node: &rlx_ir::Node) -> bool {
    is_elidable_folded_transpose_gated(graph, node, false)
}

/// [`is_elidable_folded_transpose`] with the small-`m` gate of
/// [`is_elidable_bank_transpose_gated`].
pub fn is_elidable_folded_transpose_gated(
    graph: &Graph,
    node: &rlx_ir::Node,
    require_small_m: bool,
) -> bool {
    is_elidable_bank_transpose_gated(graph, node, require_small_m)
        // The 2-D matmul fold has no shape gate on any backend that does it.
        || (!require_small_m && is_elidable_matmul_transpose(graph, node))
}

/// A buffer slot in the memory arena.
#[derive(Debug, Clone)]
pub struct BufferSlot {
    /// Offset in bytes from the start of the arena.
    pub offset: usize,
    /// Size in bytes.
    pub size: usize,
}

/// Complete memory plan for executing a graph.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Total arena size in bytes.
    pub arena_size: usize,
    /// Buffer assignment: NodeId → offset within arena.
    pub assignments: HashMap<NodeId, BufferSlot>,
    /// Node execution order (topological).
    ///
    /// "Schedule" here is the graph-level sense: the order ops run in. Not to
    /// be confused with [`rlx_ir::kernel_schedule::KernelSchedule`], which is
    /// the intra-kernel sense — roles, barriers and staging *inside* one op.
    /// This field orders the ops; that type describes how one of them drives
    /// the machine.
    pub schedule: Vec<NodeId>,
}

impl MemoryPlan {
    /// Sum of all assigned buffer sizes (i.e. how much memory the
    /// plan would use if every node had its own slot). Useful for
    /// reporting how much the liveness-aware sharing saved.
    pub fn total_unshared_bytes(&self) -> usize {
        self.assignments.values().map(|s| s.size).sum()
    }

    /// Bytes saved vs. naive "every node gets its own slot" — how
    /// much the liveness analysis bought you.
    pub fn bytes_saved(&self) -> usize {
        self.total_unshared_bytes().saturating_sub(self.arena_size)
    }

    /// Render the buffer plan as a one-line-per-node table for
    /// debugging — sorted by offset so adjacent buffers in memory
    /// are adjacent in the report (plan #87).
    ///
    /// The output is parseable: `<offset>\t<size>\t%<node_id>`. Pipe
    /// through `column -t` for human display, or grep / awk it for
    /// scripted analysis.
    pub fn report(&self) -> String {
        let mut rows: Vec<(usize, usize, NodeId)> = self
            .assignments
            .iter()
            .map(|(id, slot)| (slot.offset, slot.size, *id))
            .collect();
        rows.sort();
        let mut out = String::new();
        out.push_str(&format!(
            "# arena_size={} total_unshared={} saved={}\n",
            self.arena_size,
            self.total_unshared_bytes(),
            self.bytes_saved()
        ));
        out.push_str("# offset\tsize\tnode\n");
        for (off, sz, id) in rows {
            out.push_str(&format!("{off}\t{sz}\t{id}\n"));
        }
        out
    }
}

/// Collect view-node aliases for embedding in LIR.
pub fn collect_view_aliases(graph: &Graph) -> HashMap<NodeId, (NodeId, usize)> {
    let mut out = HashMap::new();
    for node in graph.nodes() {
        if pure_view_offset(graph, node).is_some() {
            let (root, off) = resolve_view_root(graph, node.id);
            out.insert(node.id, (root, off));
        }
    }
    out
}

/// Walk view chains until reaching a non-view ancestor. Returns the
/// root buffer-owning node and the cumulative byte offset from the root.
fn resolve_view_root(graph: &Graph, mut id: NodeId) -> (NodeId, usize) {
    let mut total_offset = 0usize;
    loop {
        let node = graph.node(id);
        match pure_view_offset(graph, node) {
            Some((parent, off)) => {
                total_offset += off;
                id = parent;
            }
            None => return (id, total_offset),
        }
    }
}

/// The nodes a 3-D convolution's epilogue fold absorbs, and the tensors it
/// reads through.
///
/// A backend can fold a convolution's consumers into its store (a LeakyReLU
/// epilogue) and its producers into its gather (a channel-wise `Concat`, a
/// nearest-neighbour upsample). That is arithmetically free, and it is a lie to
/// the memory planner unless the planner is told: folding a consumer moves the
/// convolution's **write earlier**, folding a producer extends a tensor's
/// **read later**, and slot reuse was computed for neither.
///
/// Measured cost of not telling it: rlx-metal's three conv3d folds were exact
/// at 64^3 and 128^3 and 17-19% of full scale wrong at 192^3, where the arena
/// is under enough pressure to recycle a slot. Correct arithmetic, recycled
/// memory, no error.
///
/// Shared with the backend deliberately, for the reason
/// [`is_elidable_bank_transpose`] gives: the planner and the compiler have to
/// agree about which nodes vanish. They need not agree *exactly* here, because
/// the directions differ — the planner is conservative (it extends liveness
/// for every candidate) and the backend is precise (it folds only when its own
/// checks also pass). Extending the life of a tensor that ends up materialised
/// wastes a little arena; folding one the planner did not extend corrupts.
#[derive(Debug, Default, Clone)]
pub struct Conv3dEpilogueFolds {
    /// `absorbed[n] = conv` — `n`'s buffer is written by `conv`, earlier in the
    /// schedule than `n`'s own step, so `n` must be live from `conv` onward.
    pub absorbed: HashMap<NodeId, NodeId>,
    /// `read_through[t] = conv` — `conv` gathers from `t` directly, after the
    /// node that nominally consumed `t` has run, so `t` must live to `conv`.
    pub read_through: HashMap<NodeId, NodeId>,
}

/// Recognise the graph-level shape of the folds. Backend-specific conditions
/// (arena residency, kernel selection) are *not* checked here — see the note on
/// [`Conv3dEpilogueFolds`] about which way the two sides may disagree.
pub fn conv3d_epilogue_folds(graph: &Graph) -> Conv3dEpilogueFolds {
    use rlx_ir::op::BinaryOp;
    let mut out = Conv3dEpilogueFolds::default();
    let mut uses: HashMap<NodeId, u32> = HashMap::new();
    for n in graph.nodes() {
        for &i in &n.inputs {
            *uses.entry(i).or_insert(0) += 1;
        }
    }
    let is_conv3d = |id: NodeId| {
        let n = graph.node(id);
        match &n.op {
            Op::Conv3d { .. } => true,
            Op::FusedConvBiasAct {
                activation: None,
                has_residual: false,
                ..
            } => n.shape.rank() == 5 && n.inputs.len() == 3,
            _ => false,
        }
    };
    let scalar_const = |id: NodeId| -> bool {
        matches!(&graph.node(id).op, Op::Constant { data } if data.len() == 4)
    };

    for node in graph.nodes() {
        // LeakyReLU `max(conv, alpha*conv)` absorbed into the conv's store.
        if matches!(node.op, Op::Binary(BinaryOp::Max)) && node.inputs.len() == 2 {
            for (ci, mi) in [(0usize, 1usize), (1, 0)] {
                let (c, m) = (node.inputs[ci], node.inputs[mi]);
                if !is_conv3d(c) {
                    continue;
                }
                let mn = graph.node(m);
                if !matches!(mn.op, Op::Binary(BinaryOp::Mul)) || mn.inputs.len() != 2 {
                    continue;
                }
                let k = if mn.inputs[0] == c {
                    mn.inputs[1]
                } else if mn.inputs[1] == c {
                    mn.inputs[0]
                } else {
                    continue;
                };
                if !scalar_const(k) || uses.get(&c) != Some(&2) || uses.get(&m) != Some(&1) {
                    continue;
                }
                if graph.outputs.contains(&c) || graph.outputs.contains(&m) {
                    continue;
                }
                out.absorbed.insert(node.id, c);
                out.absorbed.insert(m, c);
                break;
            }
        }
        // A channel-wise `Concat`, and a nearest upsample feeding it, read in
        // place by the convolution's gather.
        if is_conv3d(node.id) {
            let cat_id = node.inputs[0];
            let cat = graph.node(cat_id);
            if let Op::Concat { axis: 1 } = &cat.op
                && cat.inputs.len() == 2
                && uses.get(&cat_id) == Some(&1)
                && !graph.outputs.contains(&cat_id)
            {
                for &src in &cat.inputs {
                    out.read_through.insert(src, node.id);
                }
                // reshape -> Expand -> reshape is how `Graph::interpolate3d`
                // lowers a nearest integer upscale; the conv can index the
                // pre-expand tensor directly, so that must live to the conv too.
                let a = cat.inputs[0];
                if let Op::Reshape { .. } = &graph.node(a).op {
                    let e = graph.node(a).inputs[0];
                    if let Op::Expand { .. } = &graph.node(e).op {
                        let r = graph.node(e).inputs[0];
                        if let Op::Reshape { .. } = &graph.node(r).op {
                            out.read_through.insert(graph.node(r).inputs[0], node.id);
                        }
                    }
                }
            }
        }
    }
    out
}

/// Apply [`conv3d_epilogue_folds`] to the live ranges.
fn extend_conv3d_epilogue_fold_liveness(
    graph: &Graph,
    ranges: &mut HashMap<NodeId, (usize, usize)>,
) {
    let folds = conv3d_epilogue_folds(graph);
    if rlx_ir::env::flag("RLX_FOLD_LIVENESS_DEBUG") {
        eprintln!(
            "[fold-liveness] absorbed={} read_through={}",
            folds.absorbed.len(),
            folds.read_through.len()
        );
    }
    if folds.absorbed.is_empty() && folds.read_through.is_empty() {
        return;
    }
    let step_of: HashMap<NodeId, usize> = graph
        .nodes()
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id, i))
        .collect();
    // An absorbed node's buffer is written by its convolution, so it is live
    // from the convolution's step rather than its own.
    for (&absorbed, &conv) in &folds.absorbed {
        let Some(&cs) = step_of.get(&conv) else {
            continue;
        };
        if let Some(r) = ranges.get_mut(&absorbed) {
            r.0 = r.0.min(cs);
        }
    }
    // A read-through tensor must survive to the convolution that gathers it.
    for (&tensor, &conv) in &folds.read_through {
        let Some(&cs) = step_of.get(&conv) else {
            continue;
        };
        let (root, _) = resolve_view_root(graph, tensor);
        if let Some(r) = ranges.get_mut(&root) {
            r.1 = r.1.max(cs);
        }
        if root != tensor
            && let Some(r) = ranges.get_mut(&tensor)
        {
            r.1 = r.1.max(cs);
        }
    }
}

/// Compute the live range [birth, death] for each node's output buffer.
/// Birth = when the node produces its output.
/// Death = the last time any consumer reads it.
#[allow(dead_code)]
fn compute_live_ranges(graph: &Graph) -> HashMap<NodeId, (usize, usize)> {
    compute_live_ranges_opts(graph, true)
}

fn compute_live_ranges_opts(
    graph: &Graph,
    pin_output_ancestors: bool,
) -> HashMap<NodeId, (usize, usize)> {
    let mut ranges: HashMap<NodeId, (usize, usize)> = HashMap::new();

    for (step, node) in graph.nodes().iter().enumerate() {
        // Birth: this node's output is produced at this step
        ranges.entry(node.id).or_insert((step, step));

        // Extend death of all inputs to at least this step. For view
        // inputs, attribute the read to the *root* buffer so the
        // underlying allocation stays alive while any view of it is
        // still being read (#46 view-aliasing pattern).
        for &input in &node.inputs {
            let (root, _off) = resolve_view_root(graph, input);
            ranges.entry(root).and_modify(|r| r.1 = r.1.max(step));
            // Also track the view itself so we don't leave a dangling
            // entry; views inherit the root's range later in
            // plan_memory_aligned.
            if root != input {
                ranges.entry(input).and_modify(|r| r.1 = r.1.max(step));
            }
        }
    }

    // Extend death of output nodes to the end
    let last_step = graph.len();
    for &out in &graph.outputs {
        let (root, _off) = resolve_view_root(graph, out);
        ranges.entry(root).and_modify(|r| r.1 = last_step);
        if root != out {
            ranges.entry(out).and_modify(|r| r.1 = last_step);
        }
    }

    // All producers feeding graph outputs must stay live through the final
    // read-back (e.g. Cast f32→i64 feeding a boundary output). Without
    // this, a later epilogue tensor can reuse an ancestor slot while thunks
    // still run out of schedule order on overlapping paths.
    {
        let mut stack: Vec<NodeId> = graph.outputs.clone();
        let mut seen = std::collections::HashSet::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let (root, _) = resolve_view_root(graph, id);
            ranges.entry(root).and_modify(|r| r.1 = last_step);
            if root != id {
                ranges.entry(id).and_modify(|r| r.1 = last_step);
            }
            // Walking the full transitive ancestor DAG pins (almost) every node of
            // a deep feed-forward graph to the final step, which destroys slot reuse
            // — the HiFi-GAN decoder ballooned to a 5 GB arena (over wgpu's 4 GB bind
            // limit) purely from this. `pin_output_ancestors=false` keeps only the
            // read-back protection on the output nodes (and their view roots), which
            // is sufficient for in-order executors and drops that arena to ~0.12 GB.
            if pin_output_ancestors {
                for &input in &graph.node(id).inputs {
                    stack.push(input);
                }
            }
        }
    }

    // Params, Inputs, and Constants live for the ENTIRE execution.
    // Params/Inputs are pre-loaded externally; Constants are pre-loaded
    // by the runtime's compile step (see backend.rs::compile_inner). In
    // all three cases the slot must not be overwritten by intermediate
    // buffer sharing, otherwise iteration 2 of a training/inference
    // loop would read whatever the previous run scribbled into it.
    for node in graph.nodes() {
        if matches!(
            node.op,
            rlx_ir::Op::Param { .. } | rlx_ir::Op::Input { .. } | rlx_ir::Op::Constant { .. }
        ) {
            ranges.entry(node.id).and_modify(|r| {
                r.0 = 0;
                r.1 = last_step;
            });
        }
    }

    ranges
}

/// Keep packed `[B,S,3,H,D]` QKV parents alive through Attention. Without
/// this, liveness ends after the Narrow ops and the planner may reuse the
/// parent slot for the attention output while the CPU fused path (and
/// wgpu packed stride path) still read Q/K/V from that buffer.
fn extend_node_chain_liveness_to_end(
    graph: &Graph,
    ranges: &mut HashMap<NodeId, (usize, usize)>,
    start: NodeId,
    last_step: usize,
) {
    let mut stack = vec![start];
    let mut seen = std::collections::HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let (root, _) = resolve_view_root(graph, id);
        ranges.entry(root).and_modify(|r| r.1 = last_step);
        if root != id {
            ranges.entry(id).and_modify(|r| r.1 = last_step);
        }
        for &input in &graph.node(id).inputs {
            stack.push(input);
        }
    }
}

/// Pin Param/Constant packing subgraphs (Concat/Expand/Cast/…) through graph
/// end. wgpu marks those steps `static_once` and skips them on later `run()`s;
/// if their arena slots are reused by activations after the last consumer,
/// run 2+ reads clobbered weights (empty Conformer-CTC transcripts, etc.).
fn extend_static_weight_pack_liveness(graph: &Graph, ranges: &mut HashMap<NodeId, (usize, usize)>) {
    let last_step = graph.len();
    let mut memo: HashMap<NodeId, bool> = HashMap::new();
    for node in graph.nodes() {
        if !is_static_weight_tensor(graph, node.id, &mut memo) {
            continue;
        }
        // Params/Constants are already boundary-pinned; extend derived packs.
        if matches!(
            &node.op,
            Op::Param { .. } | Op::Constant { .. } | Op::Input { .. }
        ) {
            continue;
        }
        // Birth 0 as well as death `last_step`, i.e. live for the WHOLE graph.
        //
        // Extending only the death is not enough, and the reason is a
        // single-run vs across-runs mismatch. Liveness here describes ONE
        // execution, so the planner may hand a pack the slot of an activation
        // that died before the pack was born — legal within a run. But a
        // backend that skips re-materialising the pack runs everything else
        // again on the next `run()`, and that earlier activation is reborn into
        // the same bytes and clobbers it.
        //
        // Seen on wgpu/Vulkan: layer 1's fused-weight pack was given layer 0's
        // MatMul slot, so only 1 pack per graph was exclusively owned no matter
        // how deep the model — 2 steps marked at 1, 2 and 8 layers. The
        // backends' slot-exclusivity checks correctly refused to skip, which is
        // why this surfaced as a dead optimisation rather than wrong output.
        //
        // Making the range span the whole graph forces a private slot, which is
        // what "materialise once and never touch again" actually requires.
        ranges
            .entry(node.id)
            .and_modify(|r| *r = (0usize, last_step));
    }
}

/// Whether backends may skip re-materialising static weight packs after run 1.
///
/// Default ON. `RLX_STATIC_WEIGHT_PACK=0` opts out, uniformly across Metal,
/// wgpu, CUDA and ROCm — one switch for one behaviour, rather than a different
/// (or missing) knob per backend. Its purpose is bisecting: if a model produces
/// wrong output and a stale fused weight is a suspect, this turns the skip off
/// everywhere without a rebuild.
///
/// `RLX_QWEN3_BAKE_WEIGHTS` is honoured as a legacy alias on Metal, where it
/// was the original (model-specific) name for the same lever. The behaviour is
/// not qwen3-specific — it applies to any graph the weight-concat fusions
/// touch.
pub fn static_weight_pack_skip_enabled() -> bool {
    rlx_ir::env::flag_or("RLX_STATIC_WEIGHT_PACK", true)
}

/// True when `id`'s value is fixed after param/constant upload (no Inputs).
///
/// Public because backends need the SAME predicate the planner used. This is
/// what decides which packs `extend_static_weight_pack_liveness` pins to
/// graph end, and a backend that skips re-running a pack on later `run()`s is
/// relying on exactly that pin. A backend-local reimplementation can drift from
/// this one, and when it does the skip is armed for a pack the planner never
/// pinned — whose slot is then reused by an activation, so run 2+ reads
/// clobbered weights. `rlx-wgpu` carried a byte-identical private copy; sharing
/// this one removes the drift by construction rather than by vigilance.
///
/// A pin is still not a guarantee (see the `slot_is_exclusive` check in the
/// wgpu and Metal lowerings), so treat this as necessary, not sufficient.
pub fn is_static_weight_tensor(
    graph: &Graph,
    id: NodeId,
    memo: &mut HashMap<NodeId, bool>,
) -> bool {
    if let Some(&v) = memo.get(&id) {
        return v;
    }
    let node = graph.node(id);
    let v = match &node.op {
        Op::Param { .. } | Op::Constant { .. } => true,
        Op::Input { .. } => false,
        Op::Cast { .. }
        | Op::Reshape { .. }
        | Op::Transpose { .. }
        | Op::Narrow { .. }
        | Op::Expand { .. }
        | Op::Activation(_)
        | Op::Concat { .. } => {
            !node.inputs.is_empty()
                && node
                    .inputs
                    .iter()
                    .all(|&inp| is_static_weight_tensor(graph, inp, memo))
        }
        Op::Binary(_) | Op::Where | Op::Fma => node
            .inputs
            .iter()
            .all(|&inp| is_static_weight_tensor(graph, inp, memo)),
        _ => false,
    };
    memo.insert(id, v);
    v
}

/// Whether the active backend defers Expand/Concat/Transpose/Narrow to the host.
///
/// A device property, not a graph one, so it is set once by the backend at
/// device init (`rlx_wgpu` does this for discrete Vulkan/DX12) rather than
/// threaded through every planner entry point. `RLX_ARENA_PIN_HOST_STRUCTURE=1`
/// forces it on for debugging.
static PIN_HOST_STRUCTURE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Declare that this process's GPU backend runs structural ops on the host.
///
/// Their arena writes land at a later flush, so the planner must keep the slots
/// reserved; without this a subsequent GPU op takes the slot and the deferred
/// write corrupts it — nondeterministically, and only on backends that host
/// (Metal keeps these on GPU, which is why it looked Vulkan-specific).
pub fn set_pin_host_structure(on: bool) {
    PIN_HOST_STRUCTURE.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn pin_host_structure() -> bool {
    PIN_HOST_STRUCTURE.load(std::sync::atomic::Ordering::Relaxed)
        || rlx_ir::env::flag("RLX_ARENA_PIN_HOST_STRUCTURE")
}

/// Keep primary data inputs alive through graph end for `Op::Custom("onnx.*")`
/// thunks that read activations after parallel branches would otherwise reuse slots.
fn extend_custom_op_input_liveness(
    graph: &Graph,
    ranges: &mut HashMap<NodeId, (usize, usize)>,
    dequant_host_fallback: bool,
) {
    let last_step = graph.len();
    for node in graph.nodes() {
        let Op::Custom {
            name, num_inputs, ..
        } = &node.op
        else {
            continue;
        };
        if !name.starts_with("onnx.") {
            continue;
        }
        let n = (*num_inputs as usize).min(node.inputs.len());
        for &input in &node.inputs[..n] {
            extend_node_chain_liveness_to_end(graph, ranges, input, last_step);
        }
    }
    // Op::DequantMatMul / Op::DequantGroupedMatMul on Metal may fall back to a
    // deferred-host execution path (`RLX_METAL_DEQUANT_GPU_DISABLE=1`, or when
    // `dequant_scratch_off == 0`, or for schemes the GPU kernel doesn't support).
    // The deferred path runs at a `flush_deferred_host` sync point INSIDE a
    // later `e!()` macro invocation — by then the activation buffer may have
    // been reused by a subsequent GPU op because the planner sees the host op
    // as a normal-step consumer and considers the input free for reuse after
    // that step. Without this extension, attention output (last read by the
    // o_proj DequantMatMul) gets clobbered between attention's GPU dispatch
    // and the host o_proj flush, producing exact-zero downstream values
    // (task #50). The fix is conservative — extends only the direct
    // activation input (operand 0), not the whole ancestor chain — because
    // weights (operand 1+) are always Params and already pinned.
    // GATE: only needed when a dequant matmul can hit the deferred-host flush.
    // A backend that runs ALL dequant matmuls on-GPU passes
    // `dequant_host_fallback=false`; skipping this restores slot reuse (the
    // chain-walk otherwise pins nearly every activation in a packed prefill to
    // the graph end — qwen3.5 8K packed prefill: 61.9 GB pinned vs ~4 GB reused).
    // Same hazard, different ops: `rlx-wgpu` lowers Expand / Concat / Transpose
    // / Narrow to *host* steps on a discrete Vulkan/DX12 backend
    // (`wgpu_prefer_structure_host`), and those outputs "live only in the mirror
    // until a device-reading host step or GPU pass needs them" — i.e. the arena
    // write happens at a later flush. The planner sees them as ordinary
    // single-step consumers, so a subsequent GPU op takes the slot and the
    // deferred write lands on a buffer that now belongs to something else. It is
    // nondeterministic (it depends on when the flush falls) and Vulkan-only
    // (Metal keeps these on the GPU), which is what made it look like a family
    // of unrelated DSP/eig/pad bugs.
    //
    // Gated because the pin is expensive: only a backend that actually hosts
    // these ops should ask for it.
    if pin_host_structure() {
        for node in graph.nodes() {
            if matches!(
                &node.op,
                Op::Expand { .. } | Op::Concat { .. } | Op::Transpose { .. } | Op::Narrow { .. }
            ) {
                for &input in &node.inputs {
                    extend_node_chain_liveness_to_end(graph, ranges, input, last_step);
                }
                extend_node_chain_liveness_to_end(graph, ranges, node.id, last_step);
            }
        }
    }

    if dequant_host_fallback {
        for node in graph.nodes() {
            match &node.op {
                Op::DequantMatMul { .. } => {
                    if let Some(&x) = node.inputs.first() {
                        extend_node_chain_liveness_to_end(graph, ranges, x, last_step);
                    }
                }
                Op::DequantGroupedMatMul { .. } => {
                    if let Some(&x) = node.inputs.first() {
                        extend_node_chain_liveness_to_end(graph, ranges, x, last_step);
                    }
                }
                _ => {}
            }
        }
    }
}

/// Albert-style blocks reuse hidden buffers across many sequential Add/LN
/// stages; keep residual inputs alive through graph end when this graph uses
/// ONNX `QMatMul` thunks (marker for the bundled ONNX import path).
fn extend_bert_hidden_liveness(graph: &Graph, ranges: &mut HashMap<NodeId, (usize, usize)>) {
    let uses_onnx_qmatmul = graph.nodes().iter().any(|node| {
        matches!(
            &node.op,
            Op::Custom { name, .. } if name == "onnx.QMatMul" || name == "onnx.ActCopy"
        )
    });
    if !uses_onnx_qmatmul {
        return;
    }
    let last_step = graph.len();
    for node in graph.nodes() {
        match &node.op {
            Op::LayerNorm { .. } | Op::LayerNorm2d { .. } => {
                if let Some(&input) = node.inputs.first() {
                    extend_node_chain_liveness_to_end(graph, ranges, input, last_step);
                }
                ranges.entry(node.id).and_modify(|r| r.1 = last_step);
            }
            Op::Binary(BinaryOp::Add) => {
                for &input in &node.inputs {
                    extend_node_chain_liveness_to_end(graph, ranges, input, last_step);
                }
                ranges.entry(node.id).and_modify(|r| r.1 = last_step);
            }
            _ => {}
        }
    }
}

fn extend_onnx_duration_epilogue_liveness(
    graph: &Graph,
    ranges: &mut HashMap<NodeId, (usize, usize)>,
) {
    // Waveform-only graphs still contain duration-loop nodes in IR, but when
    // duration is not exported we can use normal slot reuse.
    if !graph_exports_onnx_duration(graph) {
        return;
    }
    let last_step = graph.len();
    for &out in &graph.outputs {
        extend_node_chain_liveness_to_end(graph, ranges, out, last_step);
    }
    for node in graph.nodes() {
        let keep = match &node.op {
            Op::Custom { name, .. }
                if name == "onnx.ConcatFromSequence" || name == "onnx.KittenConcatFromSequence" =>
            {
                true
            }
            Op::Expand { .. } => node.shape.dtype() == rlx_ir::DType::I64,
            Op::Cast { to, .. } => *to == rlx_ir::DType::I64,
            Op::Where => node.shape.dtype() == rlx_ir::DType::I64,
            Op::Binary(_) => node.shape.dtype() == rlx_ir::DType::I64,
            _ => node.shape.dtype() == rlx_ir::DType::I64 && node.shape.rank() <= 2,
        };
        if keep {
            extend_node_chain_liveness_to_end(graph, ranges, node.id, last_step);
            ranges.entry(node.id).and_modify(|r| r.1 = last_step);
        }
    }
}

fn graph_exports_onnx_duration(graph: &Graph) -> bool {
    graph
        .outputs
        .iter()
        .any(|&id| graph.node(id).shape.dtype() == rlx_ir::DType::I64)
}

#[allow(dead_code)]
fn graph_uses_onnx_duration_epilogue(graph: &Graph) -> bool {
    if graph.nodes().iter().any(|node| {
        matches!(
            &node.op,
            Op::Custom { name, .. }
                if name == "onnx.ConcatFromSequence"
                    || name == "onnx.KittenConcatFromSequence"
        )
    }) {
        return true;
    }
    graph_exports_onnx_duration(graph)
}

fn extend_packed_qkv_parent_liveness(graph: &Graph, ranges: &mut HashMap<NodeId, (usize, usize)>) {
    for (step, node) in graph.nodes().iter().enumerate() {
        let rlx_ir::Op::Attention { .. } = &node.op else {
            continue;
        };
        if node.inputs.len() < 3 {
            continue;
        }
        let Some((parent, _, _)) = rlx_ir::detect_packed_bshd_qkv_attention(
            graph,
            node.inputs[0],
            node.inputs[1],
            node.inputs[2],
        ) else {
            continue;
        };
        let (root, _) = resolve_view_root(graph, parent);
        ranges.entry(root).and_modify(|r| r.1 = r.1.max(step));
        if root != parent {
            ranges.entry(parent).and_modify(|r| r.1 = r.1.max(step));
        }
    }
}

/// Assign buffers using a greedy best-fit algorithm.
///
/// Sorts buffers by size (largest first), then for each buffer finds
/// the smallest free gap in the arena during its live interval.
/// This is a simplified version of XLA's GlobalDecreasingSizeBestFitHeap.
/// Controls which graph boundaries receive arena slots during planning.
///
/// Inference graphs use [`Self::inference`] (all boundaries allocated).
/// Backward graphs in a training pair use [`Self::backward_activations_only`]:
/// parameters borrow offsets from the forward plan via [`SharedWeightLayout`]
/// so weights are not stored twice in the activation arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPlanOptions {
    pub allocate_params: bool,
    pub allocate_inputs: bool,
    pub allocate_constants: bool,
    /// When true (or env `RLX_ARENA_NO_REUSE=1`), every tensor gets a unique arena slot.
    pub arena_no_reuse: bool,
    /// When true (default), pin the *entire* transitive ancestor DAG of the graph
    /// outputs to the final step. That's a conservative guard for out-of-order
    /// execution, but it destroys slot reuse on deep feed-forward graphs (the
    /// HiFi-GAN decoder hit a 5 GB arena). In-order executors (CPU, wgpu) can set
    /// this false: only the output nodes are pinned (read-back protection), which
    /// is sufficient and keeps the arena small.
    pub pin_output_ancestors: bool,
    /// When true (default), pin every `DequantMatMul`/`DequantGroupedMatMul`
    /// activation input (operand 0) live-to-end — guards the deferred-host dequant
    /// path (task #50: the host o_proj flush reads an activation a later GPU op
    /// would otherwise have reused → exact-zero downstream). A backend that runs
    /// ALL dequant matmuls on-GPU (no host flush) sets this false to restore slot
    /// reuse: without it, packed prefills pin nearly every activation to the graph
    /// end (qwen3.5 8K packed prefill: 61.9 GB pinned vs ~4 GB reused).
    pub dequant_host_fallback: bool,
    /// When true, do not reserve arena for transposes the backend will fold into
    /// GEMM trans-flags — see [`is_elidable_folded_transpose`].
    ///
    /// Off by default, because a backend that does NOT fold would then execute
    /// a transpose into a buffer that was never allocated. Only a planner
    /// dedicated to a folding backend turns it on.
    pub elide_bank_transposes: bool,
    /// When true, account for the conv3d epilogue folds a backend may apply —
    /// see [`conv3d_epilogue_folds`]. Off by default: a backend that does not
    /// fold gets the tighter, ordinary liveness.
    pub fold_conv3d_epilogue: bool,
    /// Restrict [`Self::elide_bank_transposes`] to transposes whose readers are
    /// all small-`m` grouped matmuls — for a backend (Metal) whose transposed
    /// kernel exists only on the decode path.
    pub elide_requires_small_m: bool,
}

impl MemoryPlanOptions {
    pub fn inference() -> Self {
        Self {
            allocate_params: true,
            allocate_inputs: true,
            allocate_constants: true,
            arena_no_reuse: rlx_ir::env::var("RLX_ARENA_NO_REUSE")
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            pin_output_ancestors: true,
            dequant_host_fallback: true,
            elide_bank_transposes: false,
            fold_conv3d_epilogue: false,
            elide_requires_small_m: false,
        }
    }

    /// Activations + inputs/constants only; params bound via [`SharedWeightLayout`].
    pub fn backward_activations_only() -> Self {
        Self {
            allocate_params: false,
            allocate_inputs: true,
            allocate_constants: true,
            arena_no_reuse: rlx_ir::env::var("RLX_ARENA_NO_REUSE")
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            pin_output_ancestors: true,
            dequant_host_fallback: true,
            elide_bank_transposes: false,
            fold_conv3d_epilogue: false,
            elide_requires_small_m: false,
        }
    }
}

impl Default for MemoryPlanOptions {
    fn default() -> Self {
        Self::inference()
    }
}

/// Persistent parameter slots extracted from a forward [`MemoryPlan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedWeightLayout {
    pub arena_size: usize,
    pub slots: Vec<WeightSlot>,
}

/// One named parameter and its byte range in the shared weight region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightSlot {
    pub name: String,
    pub forward_id: NodeId,
    pub offset: usize,
    pub size: usize,
}

impl SharedWeightLayout {
    /// Collect `Op::Param` slots from a forward memory plan (by param name).
    pub fn from_forward(graph: &Graph, plan: &MemoryPlan) -> Self {
        let mut slots = Vec::new();
        for node in graph.nodes() {
            if let rlx_ir::Op::Param { name } = &node.op {
                if let Some(slot) = plan.assignments.get(&node.id) {
                    slots.push(WeightSlot {
                        name: name.clone(),
                        forward_id: node.id,
                        offset: slot.offset,
                        size: slot.size,
                    });
                }
            }
        }
        slots.sort_by(|a, b| a.name.cmp(&b.name));
        let arena_size = slots.iter().map(|s| s.offset + s.size).max().unwrap_or(0);
        Self { arena_size, slots }
    }

    /// Map backward-graph `Op::Param` nodes to the forward weight offsets.
    pub fn apply_to_plan(&self, graph: &Graph, plan: &mut MemoryPlan) {
        let by_name: std::collections::HashMap<&str, &WeightSlot> =
            self.slots.iter().map(|s| (s.name.as_str(), s)).collect();
        for node in graph.nodes() {
            if let rlx_ir::Op::Param { name } = &node.op {
                let Some(slot) = by_name.get(name.as_str()) else {
                    continue;
                };
                plan.assignments.insert(
                    node.id,
                    BufferSlot {
                        offset: slot.offset,
                        size: slot.size,
                    },
                );
            }
        }
        plan.arena_size = plan.arena_size.max(self.arena_size);
    }
}

#[inline]
fn plans_boundary_buffer(op: &rlx_ir::Op, opts: MemoryPlanOptions) -> bool {
    match op {
        rlx_ir::Op::Param { .. } => opts.allocate_params,
        rlx_ir::Op::Input { .. } => opts.allocate_inputs,
        rlx_ir::Op::Constant { .. } => opts.allocate_constants,
        _ => true,
    }
}

/// Plan memory with default 64-byte alignment.
pub fn plan_memory(graph: &Graph) -> MemoryPlan {
    plan_memory_aligned(graph, 64)
}

/// Plan memory with custom alignment and boundary allocation policy.
pub fn plan_memory_with_options(
    graph: &Graph,
    alignment: usize,
    opts: MemoryPlanOptions,
) -> MemoryPlan {
    plan_memory_aligned_inner(graph, alignment, opts, None, ArenaWidthPolicy::Native)
}

/// Plan memory with custom alignment (inference defaults).
pub fn plan_memory_aligned(graph: &Graph, alignment: usize) -> MemoryPlan {
    plan_memory_aligned_inner(
        graph,
        alignment,
        MemoryPlanOptions::default(),
        None,
        ArenaWidthPolicy::Native,
    )
}

/// Liveness-aware planning with every slot sized as `num_elements * 4`
/// bytes (wgpu / uniform-f32 arenas). Reuses dead tensor slots so large
/// `[n, n]` pairwise graphs stay under WebGPU's 128 MiB binding cap.
///
/// When the graph has host indexing (`ScatterNd` / `Gather*`), keep the
/// output-ancestor pin — same rule as Metal. Unpinning lets later GPU ops
/// reuse slots that a mid-schedule CPU indexing thunk still needs to read,
/// which drifts long ODE chains (F5 DiT on a sharded >4 GiB arena).
pub fn plan_memory_f32_uniform(graph: &Graph, alignment: usize) -> MemoryPlan {
    let pin = graph_has_host_indexing(graph)
        || rlx_ir::env::var("RLX_PIN_OUTPUT_ANCESTORS").as_deref() == Some("1");
    let opts = MemoryPlanOptions {
        // Default off: deep feed-forward vocoders (HiFi-GAN) need reuse to
        // stay under wgpu's 4 GiB single-buffer / binding limits.
        pin_output_ancestors: pin,
        ..MemoryPlanOptions::default()
    };
    plan_memory_aligned_inner(graph, alignment, opts, None, ArenaWidthPolicy::F32Uniform)
}

/// **Native-width** sibling of [`plan_memory_f32_uniform`]: every slot is sized
/// at its true dtype byte width (bf16/f16 = 2 B, i8 = 1 B, …) instead of a uniform
/// 4 B. For backends that run low-precision activations + weights natively
/// (CPU/Metal/MLX/CUDA) — halves activation memory for a low-precision graph.
/// UNSAFE on a backend that widens bool/int activations to f32 at compute (use
/// [`plan_memory_hybrid`] there). Shares the same host-indexing pin rule.
pub fn plan_memory_native(graph: &Graph, alignment: usize) -> MemoryPlan {
    let opts = MemoryPlanOptions {
        pin_output_ancestors: graph_has_host_indexing(graph),
        ..MemoryPlanOptions::default()
    };
    plan_memory_aligned_inner(graph, alignment, opts, None, ArenaWidthPolicy::Native)
}

/// [`plan_memory_native`] for a **strictly in-order** backend (CPU).
///
/// Adds `dequant_host_fallback: false` on top of the native planner. That flag
/// exists for GPU backends whose `Op::DequantMatMul` can fall back to a
/// *deferred* host execution path flushed at a later sync point — there, the
/// planner must pin each dequant matmul's activation input to graph end or a
/// subsequent GPU op reuses the slot before the host flush reads it.
///
/// A CPU backend executes every thunk in schedule order and runs its dequant
/// matmuls natively (`Thunk::DequantMatMulGguf` and friends), so nothing is ever
/// deferred and the pin buys nothing — while costing a great deal. The
/// chain-walk pins nearly every activation in a packed prefill to the graph end,
/// destroying slot reuse: the gate's own note measures a qwen3.5 8K packed
/// prefill at 61.9 GB pinned vs ~4 GB reused.
///
/// Same relationship to [`plan_memory_native`] as that function has to
/// [`plan_memory_aligned`]: identical dtype widths, strictly less pinning.
pub fn plan_memory_native_in_order(graph: &Graph, alignment: usize) -> MemoryPlan {
    let opts = MemoryPlanOptions {
        pin_output_ancestors: graph_has_host_indexing(graph),
        dequant_host_fallback: false,
        // The CPU backend folds `Transpose -> GroupedMatMul`, and this planner
        // is its own — no other backend calls it.
        elide_bank_transposes: true,
        fold_conv3d_epilogue: false,
        ..MemoryPlanOptions::default()
    };
    plan_memory_aligned_inner(graph, alignment, opts, None, ArenaWidthPolicy::Native)
}

/// **Hybrid** sibling of [`plan_memory_f32_uniform`]: `Param` weights AND F16/BF16
/// activations keep their native (packed) width, while every other node stays
/// f32-uniform (so the bool/int widen-at-compute path is safe). The best-of-both
/// for a backend that runs f16/bf16 kernels natively but widens integer/bool
/// tensors to f32 — it packs the float low-precision tensors + weights (e.g. a
/// bf16 LM head + bf16 hidden states) without full [`plan_memory_native`]'s
/// integer-overrun risk. Same host-indexing pin rule as the f32-uniform planner.
pub fn plan_memory_hybrid(graph: &Graph, alignment: usize) -> MemoryPlan {
    let opts = MemoryPlanOptions {
        pin_output_ancestors: graph_has_host_indexing(graph),
        ..MemoryPlanOptions::default()
    };
    plan_memory_aligned_inner(graph, alignment, opts, None, ArenaWidthPolicy::Hybrid)
}

/// Same as [`plan_memory_f32_uniform`] but leaves `Op::Param` nodes UNassigned
/// (`allocate_params: false`) so the caller can park large packed weights in a
/// separate buffer. Used by wgpu to keep the activation arena under the 4 GiB
/// single-buffer cap for 27B-class packed GGUF models (Bonsai-27B Q1_0).
pub fn plan_memory_f32_uniform_no_params(graph: &Graph, alignment: usize) -> MemoryPlan {
    let pin = graph_has_host_indexing(graph);
    let opts = MemoryPlanOptions {
        pin_output_ancestors: pin,
        allocate_params: false,
        ..MemoryPlanOptions::default()
    };
    plan_memory_aligned_inner(graph, alignment, opts, None, ArenaWidthPolicy::F32Uniform)
}

/// True when the graph indexes with a **data-dependent** index tensor.
///
/// Such ops are hosted on several backends, and a hosted step's arena write is
/// deferred to the next flush — so the planner's "an operand dies at its last
/// consumer" rule stops describing when the buffer is actually free. Pinning
/// output ancestors is the conservative answer.
///
/// Plain `Op::Gather` belongs here and was missing. It is the *most common* of
/// these ops (every embedding table, every codebook lookup), and its index
/// tensor is as data-dependent as `GatherElements`'. A graph whose only host
/// indexing was a plain `Gather` planned as if it had none: rlx-peakflow's
/// encoder came back with activations that were not its own on wgpu, while
/// every op matched in isolation and `plan_check` called the plan clean —
/// `RLX_PIN_OUTPUT_ANCESTORS=1` was the difference, which is exactly the flag
/// this predicate sets.
fn graph_has_host_indexing(graph: &Graph) -> bool {
    graph.nodes().iter().any(|n| {
        matches!(
            &n.op,
            Op::Gather { .. }
                | Op::ScatterNd { .. }
                | Op::ScatterElements { .. }
                | Op::GatherNd { .. }
                | Op::GatherElements { .. }
                // Not indexing, but the same hazard, and the same remedy.
                // `LayerNormBackwardGamma` does not lower to one dispatch: wgpu
                // emits a multi-workgroup *partial* that writes per-chunk sums
                // into the arena's shared tail scratch zone, then a second pass
                // that reduces them into the real dgamma slot. That intermediate
                // is not a graph node, so liveness never accounts for it, and
                // under slot reuse the backward comes back wrong — not subtly:
                // on `rlx-sensorfm` the CPU gradients for the first slots are
                // ~1e-7 while wgpu returned 0.07-2.9, ~1e6x too large. AdamW
                // normalises magnitude, so those became full-size steps in
                // arbitrary directions and the model simply did not learn (loss
                // ratio 0.99 over 300 steps against 0.42 on CPU) while the
                // forward loss stayed bit-identical, which is what made it look
                // like a training-recipe problem rather than a backend one.
                | Op::LayerNormBackwardGamma { .. }
        )
    })
}

/// Plan backward activations, then alias params onto `weights`.
pub fn plan_memory_backward(
    graph: &Graph,
    alignment: usize,
    weights: &SharedWeightLayout,
) -> MemoryPlan {
    plan_memory_aligned_inner(
        graph,
        alignment,
        MemoryPlanOptions::backward_activations_only(),
        Some(weights),
        ArenaWidthPolicy::Native,
    )
}

/// How the arena planner sizes each node's slot — the width/packing strategy.
/// A backend picks the policy its kernels can consume; the planner then lays out
/// the arena accordingly. See [`plan_memory_f32_uniform`], [`plan_memory_native`],
/// [`plan_memory_hybrid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArenaWidthPolicy {
    /// Every activation slot is 4 B/elem (F32-bindable), EXCEPT non-F32 `Param`
    /// weights, which keep their native packed byte width. The classic
    /// wgpu/vulkan/rocm/cuda/oneapi f32-uniform arena. Bool/int activations widen
    /// to f32 at compute, so their slots must be f32-sized to avoid clobbering
    /// the neighboring slot (a real bug: a VITS bool mask → all-zero TTS output).
    #[default]
    F32Uniform,
    /// Every node at its native dtype byte width (bf16/f16 = 2 B, i8 = 1 B, …).
    /// For backends that run low-precision activations + weights natively
    /// (CPU/Metal/MLX/CUDA). Halves activation memory for low-precision graphs;
    /// UNSAFE on a backend that widens bool/int to f32 at compute.
    Native,
    /// Best-of-both: `Param` weights (any non-F32) AND F16/BF16 activations keep
    /// their native (packed) width; every OTHER node stays f32-uniform (so the
    /// bool/int widen-at-compute path is safe). Lets a mixed-precision graph pack
    /// its float low-precision tensors + weights without the integer-widening
    /// overrun of full `Native` — for backends that run f16/bf16 kernels natively
    /// but widen integer/bool tensors to f32.
    Hybrid,
}

#[inline]
fn node_slot_bytes(node: &rlx_ir::Node, policy: ArenaWidthPolicy) -> usize {
    // See ArenaWidthPolicy for the rationale. `native` = the tensor's true byte
    // width; `f32_wide` = the 4-B/elem width a slot needs if the backend binds /
    // widens it as f32 (never shrink a tensor already stored wider than 4 B).
    let native = node.shape.size_bytes().unwrap_or(0);
    let f32_wide = || native.max(node.shape.num_elements().unwrap_or(0) * 4);
    let is_param = matches!(node.op, rlx_ir::Op::Param { .. });
    let non_f32_param = is_param && node.shape.dtype() != rlx_ir::DType::F32;
    let low_prec_act =
        !is_param && matches!(node.shape.dtype(), rlx_ir::DType::F16 | rlx_ir::DType::BF16);
    match policy {
        ArenaWidthPolicy::Native => native,
        ArenaWidthPolicy::F32Uniform => {
            // packed / quantized WEIGHTS keep native (sub-4-B) width; all else f32.
            if non_f32_param { native } else { f32_wide() }
        }
        ArenaWidthPolicy::Hybrid => {
            // params + f16/bf16 activations packed native; everything else f32.
            if non_f32_param || low_prec_act {
                native
            } else {
                f32_wide()
            }
        }
    }
}

thread_local! {
    /// Per-thread switch for the planner's self-check (see `RLX_PLAN_VERIFY`).
    /// A thread-local rather than an env var so a test can enable it without
    /// racing every other test in the process.
    static VERIFY_PLAN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with the planner's candidate-scan self-check enabled on this thread.
#[cfg(test)]
fn with_plan_verify<R>(f: impl FnOnce() -> R) -> R {
    VERIFY_PLAN.with(|v| v.set(true));
    let r = f();
    VERIFY_PLAN.with(|v| v.set(false));
    r
}

fn plan_memory_aligned_inner(
    graph: &Graph,
    alignment: usize,
    opts: MemoryPlanOptions,
    weights: Option<&SharedWeightLayout>,
    width: ArenaWidthPolicy,
) -> MemoryPlan {
    let mut ranges = compute_live_ranges_opts(graph, opts.pin_output_ancestors);
    extend_packed_qkv_parent_liveness(graph, &mut ranges);
    extend_custom_op_input_liveness(graph, &mut ranges, opts.dequant_host_fallback);
    extend_bert_hidden_liveness(graph, &mut ranges);
    extend_onnx_duration_epilogue_liveness(graph, &mut ranges);
    extend_static_weight_pack_liveness(graph, &mut ranges);
    if opts.fold_conv3d_epilogue {
        extend_conv3d_epilogue_fold_liveness(graph, &mut ranges);
    }
    let mut opts = opts;
    if graph_exports_onnx_duration(graph) {
        opts.arena_no_reuse = true;
    }
    // Collect buffers that need allocation (skip inputs/params — external)
    struct BufInfo {
        id: NodeId,
        size: usize,
        birth: usize,
        death: usize,
    }

    let mut buffers: Vec<BufInfo> = Vec::new();
    for node in graph.nodes() {
        // Skip view nodes — they alias their parent's buffer (handled
        // in the post-pass below). Plan #46.
        if pure_view_offset(graph, node).is_some() {
            continue;
        }
        // Folded into its consumers' GEMMs, so never written or read.
        if opts.elide_bank_transposes
            && is_elidable_folded_transpose_gated(graph, node, opts.elide_requires_small_m)
        {
            continue;
        }
        let raw_size = node_slot_bytes(node, width);
        let size = if raw_size == 0 {
            boundary_min_slot_bytes(&node.op, alignment)
        } else {
            raw_size
        };
        if size > 0
            && let Some(&(birth, death)) = ranges.get(&node.id)
            && plans_boundary_buffer(&node.op, opts)
        {
            buffers.push(BufInfo {
                id: node.id,
                size,
                birth,
                death,
            });
        }
    }

    // Sort by size descending (largest first gets priority placement)
    buffers.sort_by_key(|b| std::cmp::Reverse(b.size));

    // Greedy first-fit allocation
    let mut assignments: HashMap<NodeId, BufferSlot> = HashMap::new();
    let mut arena_size: usize = 0;

    // Track allocated regions with their live ranges
    let mut placed: Vec<(usize, usize, usize, usize)> = Vec::new(); // (offset, size, birth, death)

    for buf in &buffers {
        let align = alignment;
        let node = graph.node(buf.id);
        let tail_guard = boundary_tail_guard(&node.op, align);
        let placement_size = buf.size + tail_guard;
        // Lowest offset at which this buffer fits without overlapping a buffer
        // that is live at the same time.
        //
        // **Why this is a sweep and not a candidate scan.** This used to build
        // a candidate offset per placed buffer, then test each candidate
        // against every placed buffer — O(i^2) work for buffer i, so O(N^3)
        // overall. A 12-layer Mamba training graph has N = 2378 buffers, and
        // planning it took 28 s, roughly 60% of total compile time; the graph
        // itself compiles in well under a second.
        //
        // Only buffers whose live range OVERLAPS this one can constrain it, so
        // collect those, sort by offset, and walk left to right taking the
        // first gap that fits. That is O(K log K) for the K time-overlapping
        // buffers instead of O(i^2) for all of them.
        //
        // **This picks the same offset as the scan did.** The scan took the
        // minimum conflict-free candidate, and its candidates were 0 and the
        // end of every placed buffer. Any valid offset lies in some gap between
        // time-overlapping buffers, and that gap's start is either 0 or the end
        // of a time-overlapping buffer — which the scan also had in its
        // candidate set, and which is <= the offset itself. So both take the
        // true minimum first fit; the scan just spent O(N^3) reaching it.
        let mut occupied: Vec<(usize, usize)> = placed
            .iter()
            .filter(|&&(_, _, p_birth, p_death)| buf.birth <= p_death && buf.death >= p_birth)
            .map(|&(p_off, p_size, _, _)| (p_off, p_off + p_size))
            .collect();
        occupied.sort_unstable();

        let mut cursor = 0usize;
        let mut best_offset: Option<usize> = None;
        for (start, end) in occupied {
            let aligned = (cursor + align - 1) & !(align - 1);
            if aligned + placement_size <= start {
                best_offset = Some(aligned);
                break;
            }
            cursor = cursor.max(end);
        }
        if best_offset.is_none() {
            // Past every time-overlapping buffer: the tail of that set is free.
            best_offset = Some((cursor + align - 1) & !(align - 1));
        }

        // RLX_PLAN_VERIFY=1 re-derives the offset with the original O(N^3)
        // candidate scan and asserts the sweep agrees. Off by default (it
        // restores the cubic cost); on in `plan_sweep_matches_candidate_scan`
        // and available for bisecting a suspected planning regression on a real
        // graph. This is what backs the equivalence claim above — the argument
        // is only an argument until something checks it.
        if VERIFY_PLAN.with(|v| v.get()) || rlx_ir::env::flag("RLX_PLAN_VERIFY") {
            let mut candidates = vec![0usize];
            for &(p_off, p_size, _, _) in &placed {
                candidates.push(p_off + p_size);
            }
            candidates.sort_unstable();
            candidates.dedup();
            let mut want: Option<usize> = None;
            for &cand in &candidates {
                let a = (cand + align - 1) & !(align - 1);
                let end = a + placement_size;
                let conflict = placed.iter().any(|&(p_off, p_size, p_birth, p_death)| {
                    a < p_off + p_size
                        && end > p_off
                        && buf.birth <= p_death
                        && buf.death >= p_birth
                });
                if !conflict && want.is_none_or(|w| a < w) {
                    want = Some(a);
                }
            }
            let want = want.unwrap_or_else(|| (arena_size + align - 1) & !(align - 1));
            assert_eq!(
                best_offset,
                Some(want),
                "plan sweep disagrees with the candidate scan for buffer {:?} \
                 (size {}, live {}..{})",
                buf.id,
                buf.size,
                buf.birth,
                buf.death
            );
        }

        let aligned = if opts.arena_no_reuse {
            (arena_size + align - 1) & !(align - 1)
        } else {
            best_offset.unwrap_or_else(|| {
                // No gap fit — append at arena tail.
                (arena_size + align - 1) & !(align - 1)
            })
        };
        assignments.insert(
            buf.id,
            BufferSlot {
                offset: aligned,
                size: buf.size,
            },
        );
        placed.push((aligned, placement_size, buf.birth, buf.death));
        arena_size = arena_size.max(aligned + placement_size);
    }

    // ── In-place safety pass ─────────────────────────────────
    // A node's output must never overlap the buffer of one of its own inputs:
    // an in-place permute/matmul/reduce reads and writes the same bytes and
    // corrupts (e.g. a Transpose whose output max exceeds its input's — only
    // possible if it clobbered unread source elements). The liveness overlap
    // check normally guarantees this (an input is live at the consumer's step,
    // so its slot can't be reused for the output), but a view-chain can
    // under-extend a root's death (reshape→transpose on wgpu) and slip an alias
    // through. Relocate any offending output to a fresh tail slot. This fires
    // ONLY on such a bug — correct planning never overlaps a live input — so it
    // costs no arena in the common case.
    if !opts.arena_no_reuse {
        let ids: Vec<NodeId> = buffers.iter().map(|b| b.id).collect();
        for id in ids {
            let node = graph.node(id);
            let Some(out) = assignments.get(&id).cloned() else {
                continue;
            };
            let out_size = node_slot_bytes(node, width).max(1);
            let out_end = out.offset + out_size;
            let mut overlaps_input = false;
            for &inp in &node.inputs {
                let (root, _off) = resolve_view_root(graph, inp);
                if root == id {
                    continue;
                }
                if let Some(rs) = assignments.get(&root) {
                    let r_size = node_slot_bytes(graph.node(root), width).max(1);
                    if out.offset < rs.offset + r_size && out_end > rs.offset {
                        overlaps_input = true;
                        break;
                    }
                }
            }
            if overlaps_input {
                let align = alignment;
                let aligned = (arena_size + align - 1) & !(align - 1);
                let guard = boundary_tail_guard(&node.op, align);
                assignments.insert(
                    id,
                    BufferSlot {
                        offset: aligned,
                        size: out.size,
                    },
                );
                arena_size = arena_size.max(aligned + out.size + guard);
            }
        }
    }

    // ── View aliasing pass (plan #46) ────────────────────────
    // Every view node points at its root buffer's slot, offset by the
    // accumulated view offset. The root has its own allocation above;
    // views just borrow its bytes. This is the post-pass — done after
    // root allocations are placed so we have offsets to point at.
    for node in graph.nodes() {
        if pure_view_offset(graph, node).is_some() {
            let (root, off) = resolve_view_root(graph, node.id);
            if let Some(root_slot) = assignments.get(&root).cloned() {
                let view_size = node_slot_bytes(node, width);
                assignments.insert(
                    node.id,
                    BufferSlot {
                        offset: root_slot.offset + off,
                        size: view_size,
                    },
                );
            }
        }
    }

    // ── Optional invariant self-check (RLX_MEM_VERIFY) ───────
    // The core planner invariant: no two buffers that are simultaneously live
    // may share overlapping arena bytes. If this ever fires, the allocator (not
    // the backend) is the culprit for a slot-reuse corruption. O(n²), so gated.
    if rlx_ir::env::flag("RLX_MEM_VERIFY") {
        let mut violations = 0usize;
        for (i, a) in buffers.iter().enumerate() {
            let Some(sa) = assignments.get(&a.id) else {
                continue;
            };
            let a_end = sa.offset + a.size.max(1);
            for b in &buffers[i + 1..] {
                let Some(sb) = assignments.get(&b.id) else {
                    continue;
                };
                let b_end = sb.offset + b.size.max(1);
                let mem = sa.offset < b_end && a_end > sb.offset;
                let time = a.birth <= b.death && a.death >= b.birth;
                if mem && time {
                    violations += 1;
                    if violations <= 20 {
                        eprintln!(
                            "[mem-verify] OVERLAP {:?} off={}..{} live[{},{}] <> {:?} off={}..{} live[{},{}]",
                            a.id,
                            sa.offset,
                            a_end,
                            a.birth,
                            a.death,
                            b.id,
                            sb.offset,
                            b_end,
                            b.birth,
                            b.death,
                        );
                    }
                }
            }
        }
        eprintln!(
            "[mem-verify] {violations} live+memory overlaps among {} real buffers",
            buffers.len()
        );

        // Second half of the invariant: no buffer may be READ after its
        // computed death. If this fires, a consumer (possibly via a view chain)
        // reads a root whose slot the planner already freed for reuse — the
        // classic "reused while still needed" corruption. Together with the
        // overlap check above, a clean pass here proves the *plan* is safe (so
        // any remaining corruption is in the backend's execution, not here).
        let mut read_after_death = 0usize;
        for (step, node) in graph.nodes().iter().enumerate() {
            for &input in &node.inputs {
                let (root, _off) = resolve_view_root(graph, input);
                if let Some(&(_b, d)) = ranges.get(&root) {
                    if d < step {
                        read_after_death += 1;
                        if read_after_death <= 20 {
                            eprintln!(
                                "[mem-verify] READ-AFTER-DEATH node {:?} step={step} reads {:?} (via {:?}) whose death={d}",
                                node.id, root, input,
                            );
                        }
                    }
                }
            }
        }
        eprintln!("[mem-verify] {read_after_death} reads-after-death");

        // Third check: a VIEW must fit entirely inside its root's slot. If
        // `off + view_size > root_size`, the view reads past the root into a
        // neighbouring buffer — reuse-aliased corruption the two checks above
        // miss (views aren't in `buffers`).
        let mut view_ovf = 0usize;
        for node in graph.nodes() {
            if pure_view_offset(graph, node).is_none() {
                continue;
            }
            let (root, off) = resolve_view_root(graph, node.id);
            let view_size = node_slot_bytes(node, width);
            let root_size = node_slot_bytes(graph.node(root), width);
            if off + view_size > root_size {
                view_ovf += 1;
                if view_ovf <= 20 {
                    eprintln!(
                        "[mem-verify] VIEW-OVERFLOW node {:?} off={off}+size={view_size} > root {:?} size={root_size} (by {})",
                        node.id,
                        root,
                        (off + view_size) - root_size,
                    );
                }
            }
        }
        eprintln!("[mem-verify] {view_ovf} view-past-root overflows");
    }

    let schedule = graph.topo_order().collect();

    let mut plan = MemoryPlan {
        arena_size,
        assignments,
        schedule,
    };
    if let Some(w) = weights {
        w.apply_to_plan(graph, &mut plan);
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::*;

    #[test]
    fn arena_width_policies_size_slots() {
        let mut g = Graph::new("mix");
        let x = g.input("x", Shape::new(&[64, 64], DType::F32));
        let wbf = g.param("wbf", Shape::new(&[64, 64], DType::BF16)); // bf16 weight
        let abf = g.add_node(
            Op::Cast { to: DType::BF16 },
            vec![x],
            Shape::new(&[64, 64], DType::BF16),
        ); // bf16 activation
        let abool = g.add_node(
            Op::Cast { to: DType::Bool },
            vec![x],
            Shape::new(&[64, 64], DType::Bool),
        ); // bool activation (widens to f32 at compute on f32-uniform backends)
        let ne = 64 * 64;
        use ArenaWidthPolicy::*;

        // A non-F32 Param weight stays PACKED (native, 2 B) under every policy.
        for p in [F32Uniform, Native, Hybrid] {
            assert_eq!(node_slot_bytes(g.node(wbf), p), ne * 2, "bf16 param {p:?}");
        }
        // A bf16 ACTIVATION: f32-uniform widens (4 B); native + hybrid pack (2 B).
        assert_eq!(node_slot_bytes(g.node(abf), F32Uniform), ne * 4);
        assert_eq!(node_slot_bytes(g.node(abf), Native), ne * 2);
        assert_eq!(node_slot_bytes(g.node(abf), Hybrid), ne * 2);
        // A bool ACTIVATION: hybrid keeps it f32-wide (safe for widen-at-compute),
        // only full Native shrinks it to 1 B/elem.
        assert_eq!(node_slot_bytes(g.node(abool), F32Uniform), ne * 4);
        assert_eq!(node_slot_bytes(g.node(abool), Native), ne);
        assert_eq!(node_slot_bytes(g.node(abool), Hybrid), ne * 4);
        // A plain f32 activation is 4 B/elem under every policy.
        for p in [F32Uniform, Native, Hybrid] {
            assert_eq!(node_slot_bytes(g.node(x), p), ne * 4, "f32 input {p:?}");
        }
    }

    #[test]
    fn non_overlapping_buffers_share_memory() {
        let mut g = Graph::new("test");
        let f = DType::F32;

        let x = g.input("x", Shape::new(&[100, 384], f)); // 153.6KB
        let w1 = g.param("w1", Shape::new(&[384, 384], f));
        let w2 = g.param("w2", Shape::new(&[384, 384], f));

        // mm1 is only used by mm2's input; after mm2, mm1 is dead
        let mm1 = g.matmul(x, w1, Shape::new(&[100, 384], f)); // 153.6KB, live [4, 5]
        let mm2 = g.matmul(mm1, w2, Shape::new(&[100, 384], f)); // 153.6KB, live [5, ∞]
        g.set_outputs(vec![mm2]);

        let plan = plan_memory(&g);
        println!("Arena size: {} bytes", plan.arena_size);
        for (id, slot) in &plan.assignments {
            if let Some((b, d)) = compute_live_ranges(&g).get(id) {
                println!(
                    "  {id}: offset={}, size={}, live=[{b}, {d}]",
                    slot.offset, slot.size
                );
            }
        }

        // Logical slot sizes omit 64-byte alignment gaps and param tail guards
        // (see `boundary_tail_guard`). Arena may be slightly larger than that sum
        // even when temporaries reuse gaps; cap slack at one guard per slot.
        let total_logical: usize = plan.assignments.values().map(|s| s.size).sum();
        let align_slack = plan.assignments.len() * BOUNDARY_TAIL_GUARD_BYTES;
        assert!(
            plan.arena_size <= total_logical + align_slack,
            "arena {} should be <= logical sum {} + slack {}",
            plan.arena_size,
            total_logical,
            align_slack
        );
    }

    #[test]
    fn plan_report_includes_savings() {
        // Plan #87: the public report() string surfaces enough info
        // for debug tooling — arena size, unshared total, saved
        // bytes, and a per-buffer table sorted by offset.
        let mut g = Graph::new("rep");
        let f = DType::F32;
        let x = g.input("x", Shape::new(&[16], f));
        let w = g.param("w", Shape::new(&[16, 16], f));
        let mm1 = g.matmul(x, w, Shape::new(&[1, 16], f));
        let mm2 = g.matmul(mm1, w, Shape::new(&[1, 16], f));
        g.set_outputs(vec![mm2]);

        let plan = plan_memory(&g);
        let r = plan.report();
        // Header carries the headline numbers.
        assert!(r.starts_with("# arena_size="));
        assert!(r.contains("total_unshared="));
        assert!(r.contains("saved="));
        // Body is parseable (offset\tsize\tnode), sorted ascending.
        let body: Vec<&str> = r.lines().filter(|l| !l.starts_with('#')).collect();
        assert!(!body.is_empty());
        // assignments map → at least mm1 + mm2 + x + w should appear.
        assert!(plan.assignments.contains_key(&mm1));
        assert!(plan.assignments.contains_key(&mm2));
    }

    #[test]
    fn view_ops_alias_parent_slot() {
        // Reshape, same-dtype Cast, and axis-0 Narrow should NOT get
        // their own arena slot — they alias the parent (#46).
        use rlx_ir::GraphExt;
        let mut g = Graph::new("views");
        let f = DType::F32;
        let x = g.input("x", Shape::new(&[8, 4], f)); // 128B
        let w = g.param("w", Shape::new(&[4, 4], f)); // 64B
        let mm = g.matmul(x, w, Shape::new(&[8, 4], f)); // 128B (root)
        let r = g.reshape_(mm, vec![32]); // VIEW (Reshape)
        let c = g.cast(r, DType::F32); // VIEW (same-dtype Cast)
        let n = g.narrow_(c, 0, 8, 16); // VIEW (axis-0 Narrow)
        g.set_outputs(vec![n]);

        let plan = plan_memory(&g);

        // All three view nodes should share mm's offset (with adjustment
        // for the narrow's start=8 → +8*4 = 32 bytes).
        let mm_off = plan.assignments[&mm].offset;
        assert_eq!(
            plan.assignments[&r].offset, mm_off,
            "reshape view should alias mm slot exactly"
        );
        assert_eq!(
            plan.assignments[&c].offset, mm_off,
            "same-dtype cast view should alias mm slot exactly"
        );
        assert_eq!(
            plan.assignments[&n].offset,
            mm_off + 32,
            "axis-0 narrow start=8 should alias mm slot + 8*4 bytes"
        );
        assert_eq!(
            plan.assignments[&n].size, 64,
            "narrow view's size is its own (16 f32 = 64B), not parent's"
        );
    }

    #[test]
    fn backward_plan_aliases_forward_param_slots() {
        let f = DType::F32;
        let mut fwd = Graph::new("fwd");
        let x = fwd.input("x", Shape::new(&[2, 4], f));
        let w = fwd.param("w", Shape::new(&[4, 4], f));
        let mm = fwd.matmul(x, w, Shape::new(&[2, 4], f));
        fwd.set_outputs(vec![mm]);
        let fwd_plan = plan_memory_aligned(&fwd, 64);
        let layout = SharedWeightLayout::from_forward(&fwd, &fwd_plan);

        let mut bwd = Graph::new("bwd_grad");
        let x2 = bwd.input("x", Shape::new(&[2, 4], f));
        let w2 = bwd.param("w", Shape::new(&[4, 4], f));
        let mm2 = bwd.matmul(x2, w2, Shape::new(&[2, 4], f));
        bwd.set_outputs(vec![mm2]);

        let bwd_plan = plan_memory_backward(&bwd, 64, &layout);
        let fwd_w_off = fwd_plan.assignments[&w].offset;
        let bwd_w_off = bwd_plan.assignments[&w2].offset;
        assert_eq!(bwd_w_off, fwd_w_off, "backward w must share forward offset");
        assert!(
            !bwd_plan.assignments.contains_key(&w2)
                || bwd_plan.assignments[&w2].offset == fwd_w_off
        );
    }

    #[test]
    fn overlapping_buffers_get_separate_memory() {
        let mut g = Graph::new("test");
        let f = DType::F32;

        let x = g.input("x", Shape::new(&[100, 384], f));
        let w = g.param("w", Shape::new(&[384, 384], f));

        let mm = g.matmul(x, w, Shape::new(&[100, 384], f));
        // Both mm and x are live at the same time (mm uses x)
        // x is also an output, so it stays live
        let add = g.binary(BinaryOp::Add, mm, x, Shape::new(&[100, 384], f));
        g.set_outputs(vec![add]);

        let plan = plan_memory(&g);
        let mm_slot = &plan.assignments[&mm];
        let add_slot = &plan.assignments[&add];

        // mm and add overlap in time, so they must not overlap in memory
        let mm_end = mm_slot.offset + mm_slot.size;
        let add_end = add_slot.offset + add_slot.size;
        let no_overlap = mm_end <= add_slot.offset || add_end <= mm_slot.offset;
        assert!(no_overlap, "overlapping buffers must have separate memory");
    }

    #[test]
    fn zero_length_inputs_get_arena_slots() {
        let mut g = Graph::new("empty_past");
        let f = DType::F32;
        let past = g.input("past_k", Shape::new(&[1, 0, 8], f));
        let x = g.input("x", Shape::new(&[1, 1, 8], f));
        let cat = g.concat(vec![past, x], 1, Shape::new(&[1, 1, 8], f));
        g.set_outputs(vec![cat]);

        let plan = plan_memory(&g);
        assert!(
            plan.assignments.contains_key(&past),
            "zero-length decode past input must have an arena slot"
        );
        assert!(plan.assignments[&past].size >= 64);
    }

    #[test]
    fn duration_export_forces_no_reuse_waveform_only_does_not() {
        let f = DType::F32;
        let mut wave_only = Graph::new("wave_only");
        let w = wave_only.input("wave", Shape::new(&[1024], f));
        wave_only.set_outputs(vec![w]);
        assert!(!graph_exports_onnx_duration(&wave_only));

        let mut dual = Graph::new("dual");
        let w2 = dual.input("wave", Shape::new(&[1024], f));
        let d = dual.input("dur", Shape::new(&[8], DType::I64));
        dual.set_outputs(vec![w2, d]);
        assert!(graph_exports_onnx_duration(&dual));
    }

    /// The sweep must place every buffer exactly where the original
    /// candidate-scan planner did.
    ///
    /// The scan was replaced because it was O(N^3) — 28 s on a 2378-node Mamba
    /// training graph, about 60% of that graph's total compile time. Equivalence
    /// was argued from the structure of the two searches; this runs both and
    /// compares, on a graph deep enough (200+ layers of differently-shaped
    /// buffers) that live ranges genuinely overlap and gaps genuinely open up.
    #[test]
    fn plan_sweep_matches_candidate_scan() {
        use rlx_ir::infer::GraphExt;
        let mut g = Graph::new("deep");
        let mut x = g.input("x", Shape::new(&[32, 64], DType::F32));
        // Varying widths so buffers differ in size and first-fit has to choose
        // between gaps rather than always appending at the tail.
        for i in 0..200 {
            let w = 32 + (i % 7) * 16;
            let p = g.param(format!("w{i}"), Shape::new(&[64, w], DType::F32));
            let m = g.mm(x, p);
            let r = g.relu(m);
            let p2 = g.param(format!("v{i}"), Shape::new(&[w, 64], DType::F32));
            x = g.mm(r, p2);
        }
        g.set_outputs(vec![x]);

        let plan =
            with_plan_verify(|| plan_memory_with_options(&g, 128, MemoryPlanOptions::inference()));
        assert!(
            plan.assignments.len() > 400,
            "expected a large plan, got {}",
            plan.assignments.len()
        );

        // Independently: no two buffers that are live at the same time may
        // overlap in the arena. The self-check above proves the sweep agrees
        // with the old planner; this proves the answer they agree on is sound.
        let ranges = compute_live_ranges_opts(&g, true);
        let placed: Vec<_> = plan
            .assignments
            .iter()
            .filter_map(|(id, slot)| ranges.get(id).map(|&(b, d)| (slot.offset, slot.size, b, d)))
            .collect();
        for (i, &(o1, s1, b1, d1)) in placed.iter().enumerate() {
            for &(o2, s2, b2, d2) in &placed[i + 1..] {
                let mem = o1 < o2 + s2 && o2 < o1 + s1;
                let time = b1 <= d2 && b2 <= d1;
                assert!(
                    !(mem && time),
                    "overlap: [{o1},{s1}) {b1}..{d1} vs [{o2},{s2}) {b2}..{d2}"
                );
            }
        }
    }
}
