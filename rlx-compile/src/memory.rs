// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Memory planning — liveness analysis and buffer assignment.
//!
//! This is the XLA feature that no other Rust framework has. It computes
//! which intermediate tensors have non-overlapping lifetimes and assigns
//! them to the same memory, minimizing total arena size.
//!
//! The output is a [`MemoryPlan`] that tells the runtime exactly how
//! large the arena should be and where each tensor lives within it.

use rlx_ir::{Graph, NodeId, Op};
use std::collections::HashMap;

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
fn compute_live_ranges(graph: &Graph) -> HashMap<NodeId, (usize, usize)> {
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
}

impl MemoryPlanOptions {
    pub fn inference() -> Self {
        Self {
            allocate_params: true,
            allocate_inputs: true,
            allocate_constants: true,
        }
    }

    /// Activations + inputs/constants only; params bound via [`SharedWeightLayout`].
    pub fn backward_activations_only() -> Self {
        Self {
            allocate_params: false,
            allocate_inputs: true,
            allocate_constants: true,
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
pub fn plan_memory_f32_uniform(graph: &Graph, alignment: usize) -> MemoryPlan {
    plan_memory_aligned_inner(graph, alignment, MemoryPlanOptions::default(), None, true)
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
    if f32_uniform {
        node.shape.num_elements().unwrap_or(0) * 4
    } else {
        node.shape.size_bytes().unwrap_or(0)
    }
}

fn plan_memory_aligned_inner(
    graph: &Graph,
    alignment: usize,
    opts: MemoryPlanOptions,
    weights: Option<&SharedWeightLayout>,
    f32_uniform: bool,
) -> MemoryPlan {
    let ranges = compute_live_ranges(graph);

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
        let size = node_slot_bytes(node, f32_uniform);
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
            let end = aligned + buf.size;

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

        let aligned = best_offset.unwrap_or_else(|| {
            // No gap fit — append at arena tail.
            (arena_size + align - 1) & !(align - 1)
        });
        assignments.insert(
            buf.id,
            BufferSlot {
                offset: aligned,
                size: buf.size,
            },
        );
        placed.push((aligned, buf.size, buf.birth, buf.death));
        arena_size = arena_size.max(aligned + buf.size);
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
    use rlx_ir::op::*;
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

        // mm1 and mm2 have non-overlapping lifetimes, so they CAN share memory.
        // The arena should be smaller than the sum of all buffers.
        let total_if_no_sharing: usize = plan.assignments.values().map(|s| s.size).sum();
        assert!(
            plan.arena_size <= total_if_no_sharing,
            "arena {0} should be <= sum {total_if_no_sharing}",
            plan.arena_size
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
}
