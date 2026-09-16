// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::KvAppend` to primitives, so a model can use it **portably**.
//!
//! `Op::KvAppend` writes one row into a KV cache at `pos` and returns the
//! `[..pos+1]` prefix, aliasing the cache buffer — O(1) per decode step instead
//! of the O(context) full-cache copy `concat(past_kv, new_row)` performs.
//!
//! It is implemented natively on CPU, Metal, CUDA, ROCm, wgpu and Vulkan —
//! everywhere but MLX, whose immutable array API has no in-place row write.
//! When only Metal/CUDA/ROCm had it and there was no fallback, a graph
//! containing it simply could not run on CPU, wgpu, Vulkan or MLX — so a model
//! crate that advertised `--device cpu` could not use it at all, and stayed on
//! `Concat` on *every* backend including the three that had the fast path.
//!
//! That is not hypothetical. Profiling Carbon-500M (a stock Llama, 28 layers)
//! decoding on Metal:
//!
//! ```text
//! sgemm      113   17.88 ms   38.9%
//! concat     112   16.99 ms   36.9%   <- 4 per layer, the KV append
//! attention   28    2.76 ms    6.0%
//! ```
//!
//! `concat` costs as much as every matmul combined, and it grows with context
//! while the real work is one row. The native op existed the whole time; what
//! was missing was the fallback that makes it safe to emit.
//!
//! # What this lowering does
//!
//! Rebuilds the semantics from primitives every backend already has:
//!
//! * `pos == 0` → `narrow(row, axis, 0, len)` is the whole prefix, so the row
//!   *is* the result. No concat needed.
//! * `pos > 0` → `concat([narrow(cache, axis, 0, pos), row], axis)`.
//!
//! So the fallback is exactly the `concat` the op replaces — which is the
//! point: emitting `KvAppend` is free of portability risk, and the backends
//! with a native path get the O(1) write while the rest are no worse off than
//! they are today.
//!
//! # What it does NOT preserve
//!
//! The aliasing. Native `KvAppend` returns a view of `cache`'s buffer; this
//! returns a fresh tensor. Callers that rely on the write being visible through
//! the original `cache` handle — rather than through the returned value — will
//! not see it here. Every in-tree construction site uses the return value, and
//! a lowering that silently changed aliasing semantics would be worse than no
//! lowering, so this is stated rather than assumed.

use std::collections::HashMap;

use rlx_ir::{Graph, GraphExt, NodeId, Op, OpKind};

use crate::Pass;

/// Rebuild `KvAppend(cache, row)` from `narrow` + `concat`.
///
/// `axis` is the sequence axis; `pos` is the index the row is written at, so
/// the result is the `[..pos+1]` prefix.
pub fn lower_kv_append(
    g: &mut Graph,
    cache: NodeId,
    row: NodeId,
    axis: usize,
    pos: usize,
) -> NodeId {
    if pos == 0 {
        // The prefix is `[..1]`, which is the row itself. Concatenating a
        // zero-length narrow would be legal but wasteful, and some backends
        // reject a zero-extent operand outright.
        return row;
    }
    let past = g.narrow_(cache, axis, 0, pos);
    g.concat_(vec![past, row], axis)
}

/// True when the op's `[..pos+1]` output can actually alias the cache buffer.
///
/// The aliasing contract only holds when the prefix is CONTIGUOUS from the
/// start of the buffer, i.e. when nothing precedes `axis`. With `axis = 1` and
/// a `[batch, seq_cap, width]` cache and `batch > 1`, the prefix is `batch`
/// strided slices of `pos+1` rows each — not a contiguous range — so there is
/// no offset at which the output can alias the cache and still read back
/// correctly. A row write into the aliased buffer then returns the first
/// `batch*(pos+1)*width` elements of the cache, which is a different tensor.
///
/// Batch-1 decode (every case in the tree) has `outer == 1` and is fine, which
/// is why this went unnoticed: the defect needs `batch > 1` to appear at all.
pub fn kv_append_alias_is_contiguous(graph: &Graph, node: &rlx_ir::Node) -> bool {
    let Op::KvAppend { axis, .. } = &node.op else {
        return true;
    };
    graph.node(node.inputs[0]).shape.dims()[..*axis]
        .iter()
        .all(|d| d.unwrap_static() == 1)
}

/// Pass form: replace `Op::KvAppend` with its primitive expansion.
///
/// Fires for backends that do not claim `OpKind::KvAppend`, AND — on every
/// backend, native or not — for the shapes whose output cannot alias the cache
/// (see [`kv_append_alias_is_contiguous`]). A native row write is only
/// equivalent to the concat it replaces while that aliasing holds.
pub struct LowerKvAppend {
    /// `false` = only rewrite the non-aliasable shapes, leaving the native
    /// fast path in place for everything else.
    pub all: bool,
}

impl LowerKvAppend {
    /// Rewrite every `KvAppend` — for backends without native support.
    pub const ALL: Self = Self { all: true };
    /// Rewrite only what a native row write cannot express.
    pub const NON_ALIASABLE: Self = Self { all: false };
}

impl Pass for LowerKvAppend {
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::KvAppend]
    }

    fn name(&self) -> &str {
        "lower_kv_append"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::KvAppend { .. }))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let rewrite = matches!(node.op, Op::KvAppend { .. })
                && (self.all || !kv_append_alias_is_contiguous(&graph, node));
            let new_id = if let (true, &Op::KvAppend { axis, pos }) = (rewrite, &node.op) {
                let cache = id_map[&node.inputs[0]];
                let row = id_map[&node.inputs[1]];
                lower_kv_append(&mut new_graph, cache, row, axis, pos)
            } else {
                let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                new_graph.add_node(node.op.clone(), inputs, node.shape.clone())
            };
            id_map.insert(node.id, new_id);
        }

        let outputs: Vec<NodeId> = graph.outputs.iter().map(|o| id_map[o]).collect();
        new_graph.set_outputs(outputs);
        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    const F: DType = DType::F32;

    /// `[cache(1,H,S,D), row(1,H,1,D)] -> KvAppend(axis=2, pos)`.
    fn kv_graph(s: usize, pos: usize) -> Graph {
        let mut g = Graph::new("kv");
        let cache = g.input("cache", Shape::new(&[1, 2, s, 4], F));
        let row = g.input("row", Shape::new(&[1, 2, 1, 4], F));
        let out = g.add_node(
            Op::KvAppend { axis: 2, pos },
            vec![cache, row],
            Shape::new(&[1, 2, pos + 1, 4], F),
        );
        g.set_outputs(vec![out]);
        g
    }

    #[test]
    fn the_lowering_removes_every_kv_append() {
        let g = LowerKvAppend::ALL.run(kv_graph(8, 3));
        assert!(
            !g.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::KvAppend { .. })),
            "a KvAppend survived the lowering"
        );
    }

    /// The output shape must be the `[..pos+1]` prefix, or the graph is wrong
    /// in a way that only shows up as a shape error much later.
    #[test]
    fn the_lowered_shape_is_the_prefix() {
        for pos in [0usize, 1, 3, 7] {
            let g = LowerKvAppend::ALL.run(kv_graph(8, pos));
            let out = g.outputs[0];
            let dims: Vec<usize> = g
                .shape(out)
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            assert_eq!(
                dims,
                vec![1, 2, pos + 1, 4],
                "pos={pos}: wrong prefix shape"
            );
        }
    }

    /// `pos == 0` must not emit a concat with a zero-length operand. The prefix
    /// is the row itself, and a zero-extent narrow is both wasteful and refused
    /// outright by some backends.
    #[test]
    fn pos_zero_emits_no_concat() {
        let g = LowerKvAppend::ALL.run(kv_graph(8, 0));
        assert!(
            !g.nodes().iter().any(|n| matches!(n.op, Op::Concat { .. })),
            "pos=0 should be the row itself, not a concat"
        );
    }

    /// A graph with no `KvAppend` must come back untouched — the pass is in the
    /// default pipeline and rebuilding every graph node-for-node is the
    /// `fusion_pipeline_perf` trap.
    #[test]
    fn a_graph_without_kv_append_is_returned_unchanged() {
        let mut g = Graph::new("plain");
        let x = g.input("x", Shape::new(&[4], F));
        let y = g.relu(x);
        g.set_outputs(vec![y]);
        let before = g.nodes().len();
        let after = LowerKvAppend::ALL.run(g);
        assert_eq!(after.nodes().len(), before);
    }

    /// The trigger must name the kind, or the pass scans (and rebuilds) every
    /// graph for nothing.
    #[test]
    fn the_pass_declares_its_trigger() {
        assert_eq!(LowerKvAppend::ALL.trigger_kinds(), &[OpKind::KvAppend]);
    }
}
