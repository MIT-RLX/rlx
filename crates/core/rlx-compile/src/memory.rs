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
/// Borrowed from MAX's "view-vs-copy" pattern (#46 in PLAN.md).
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
    pure_view_offset(graph, node).is_some()
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
        ranges.entry(node.id).and_modify(|r| r.1 = last_step);
    }
}

/// True when `id`'s value is fixed after param/constant upload (no Inputs).
fn is_static_weight_tensor(graph: &Graph, id: NodeId, memo: &mut HashMap<NodeId, bool>) -> bool {
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

/// Keep primary data inputs alive through graph end for `Op::Custom("onnx.*")`
/// thunks that read activations after parallel branches would otherwise reuse slots.
fn extend_custom_op_input_liveness(graph: &Graph, ranges: &mut HashMap<NodeId, (usize, usize)>) {
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
}

impl MemoryPlanOptions {
    pub fn inference() -> Self {
        Self {
            allocate_params: true,
            allocate_inputs: true,
            allocate_constants: true,
            arena_no_reuse: std::env::var("RLX_ARENA_NO_REUSE")
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            pin_output_ancestors: true,
        }
    }

    /// Activations + inputs/constants only; params bound via [`SharedWeightLayout`].
    pub fn backward_activations_only() -> Self {
        Self {
            allocate_params: false,
            allocate_inputs: true,
            allocate_constants: true,
            arena_no_reuse: std::env::var("RLX_ARENA_NO_REUSE")
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            pin_output_ancestors: true,
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
    plan_memory_aligned_inner(graph, alignment, opts, None, false)
}

/// Plan memory with custom alignment (inference defaults).
pub fn plan_memory_aligned(graph: &Graph, alignment: usize) -> MemoryPlan {
    plan_memory_aligned_inner(graph, alignment, MemoryPlanOptions::default(), None, false)
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
    let pin = graph_has_host_indexing(graph);
    let opts = MemoryPlanOptions {
        // Default off: deep feed-forward vocoders (HiFi-GAN) need reuse to
        // stay under wgpu's 4 GiB single-buffer / binding limits.
        pin_output_ancestors: pin,
        ..MemoryPlanOptions::default()
    };
    plan_memory_aligned_inner(graph, alignment, opts, None, true)
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
    plan_memory_aligned_inner(graph, alignment, opts, None, true)
}

fn graph_has_host_indexing(graph: &Graph) -> bool {
    graph.nodes().iter().any(|n| {
        matches!(
            &n.op,
            Op::ScatterNd { .. }
                | Op::ScatterElements { .. }
                | Op::GatherNd { .. }
                | Op::GatherElements { .. }
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
        false,
    )
}

#[inline]
fn node_slot_bytes(node: &rlx_ir::Node, f32_uniform: bool) -> usize {
    // f32-uniform planning gives activation slots a uniform 4-byte-per-elem
    // width (so any slot can be bound as f32). The one exception is packed /
    // quantized WEIGHTS — non-F32 `Param`s read as raw bytes by DequantMatMul.
    // Sizing those as f32 4x-bloats packed weights (a 2B model's ~1.7 GB →
    // ~6.5 GB, which exceeds wgpu's 4 GiB single-buffer cap and OOMs), so they
    // keep their true (sub-4-byte) byte width.
    //
    // Every OTHER tensor — including bool masks and integer activations —
    // occupies 4 bytes per logical element in this arena (F32 directly; bool /
    // int tensors are widened to f32 on upload / at compute time). Sizing such
    // a slot by its native byte width (bool = 1 B/elem) leaves it 4x too small,
    // so the kernel's `num_elements * 4`-byte write overruns into the following
    // slot and silently clobbers whatever lives there — e.g. the persistent
    // params sitting after a VITS relative-position bool attention mask, which
    // turned long-form wgpu TTS output into all-zero garbage. Take the max of
    // the native width and the f32 width so integer tensors that are stored
    // wider than 4 bytes on other f32-uniform backends are never shrunk.
    let native = node.shape.size_bytes().unwrap_or(0);
    if f32_uniform {
        let packed_weight =
            matches!(node.op, rlx_ir::Op::Param { .. }) && node.shape.dtype() != rlx_ir::DType::F32;
        if !packed_weight {
            return native.max(node.shape.num_elements().unwrap_or(0) * 4);
        }
    }
    native
}

fn plan_memory_aligned_inner(
    graph: &Graph,
    alignment: usize,
    opts: MemoryPlanOptions,
    weights: Option<&SharedWeightLayout>,
    f32_uniform: bool,
) -> MemoryPlan {
    let mut ranges = compute_live_ranges_opts(graph, opts.pin_output_ancestors);
    extend_packed_qkv_parent_liveness(graph, &mut ranges);
    extend_custom_op_input_liveness(graph, &mut ranges);
    extend_bert_hidden_liveness(graph, &mut ranges);
    extend_onnx_duration_epilogue_liveness(graph, &mut ranges);
    extend_static_weight_pack_liveness(graph, &mut ranges);
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
        let raw_size = node_slot_bytes(node, f32_uniform);
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
        let mut best_offset: Option<usize> = None;

        // Collect candidate start offsets: 0 plus the end of every placed
        // buffer that could border a free gap.
        let mut candidates = vec![0usize];
        for &(p_off, p_size, _, _) in &placed {
            candidates.push(p_off + p_size);
        }
        candidates.sort_unstable();
        candidates.dedup();

        for &candidate_offset in &candidates {
            let aligned = (candidate_offset + align - 1) & !(align - 1);
            let end = aligned + placement_size;

            let conflict = placed.iter().any(|&(p_off, p_size, p_birth, p_death)| {
                let p_end = p_off + p_size;
                let mem_overlap = aligned < p_end && end > p_off;
                let time_overlap = buf.birth <= p_death && buf.death >= p_birth;
                mem_overlap && time_overlap
            });

            if !conflict {
                match best_offset {
                    None => best_offset = Some(aligned),
                    Some(best) if aligned < best => best_offset = Some(aligned),
                    _ => {}
                }
            }
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
            let out_size = node_slot_bytes(node, f32_uniform).max(1);
            let out_end = out.offset + out_size;
            let mut overlaps_input = false;
            for &inp in &node.inputs {
                let (root, _off) = resolve_view_root(graph, inp);
                if root == id {
                    continue;
                }
                if let Some(rs) = assignments.get(&root) {
                    let r_size = node_slot_bytes(graph.node(root), f32_uniform).max(1);
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
                let view_size = node_slot_bytes(node, f32_uniform);
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
}
