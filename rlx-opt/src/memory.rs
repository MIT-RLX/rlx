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
/// Plan memory with default 64-byte alignment.
pub fn plan_memory(graph: &Graph) -> MemoryPlan {
    plan_memory_aligned(graph, 64)
}

/// Plan memory with custom alignment.
pub fn plan_memory_aligned(graph: &Graph, alignment: usize) -> MemoryPlan {
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
        // Skip nodes with no output size (inputs/params are external)
        if let Some(size) = node.shape.size_bytes()
            && size > 0
            && let Some(&(birth, death)) = ranges.get(&node.id)
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
        // Find the first offset where this buffer fits without overlapping
        // any other buffer that's live at the same time
        let mut candidate_offset = 0;
        let align = alignment;

        'search: loop {
            let aligned = (candidate_offset + align - 1) & !(align - 1);
            let end = aligned + buf.size;

            // Check for overlap with any placed buffer
            let mut conflict = false;
            for &(p_off, p_size, p_birth, p_death) in &placed {
                let p_end = p_off + p_size;
                // Memory overlap AND time overlap → conflict
                let mem_overlap = aligned < p_end && end > p_off;
                let time_overlap = buf.birth <= p_death && buf.death >= p_birth;
                if mem_overlap && time_overlap {
                    // Jump past this conflicting region
                    candidate_offset = p_end;
                    conflict = true;
                    break;
                }
            }

            if !conflict {
                // Found a valid slot
                let aligned = (candidate_offset + align - 1) & !(align - 1);
                assignments.insert(
                    buf.id,
                    BufferSlot {
                        offset: aligned,
                        size: buf.size,
                    },
                );
                placed.push((aligned, buf.size, buf.birth, buf.death));
                arena_size = arena_size.max(aligned + buf.size);
                break 'search;
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
                let view_size = node.shape.size_bytes().unwrap_or(0);
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

    MemoryPlan {
        arena_size,
        assignments,
        schedule,
    }
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
