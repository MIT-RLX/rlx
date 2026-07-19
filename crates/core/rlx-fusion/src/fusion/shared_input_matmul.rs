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

//! `shared_input_matmul` — extracted from the `fusion` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

// ── Helper: graph rewriter ──────────────────────────────────────────────

use crate::graph_rewrite::Rewriter;

// ── Pass 1: MatMul + Bias + Activation → FusedMatMulBiasAct ─────────────

use super::*;

/// Detects two MatMul nodes with the same input and concatenates their
/// weight matrices into a single larger MatMul.
///
/// Pattern:
///   %a = matmul(%x, %w1)
///   %b = matmul(%x, %w2)
/// Becomes:
///   %ab = matmul(%x, concat(%w1, %w2))
///   %a = narrow(%ab, ..., 0, n1)
///   %b = narrow(%ab, ..., n1, n2)
///
/// This saves one full input read (the shared input is read once instead
/// of twice). Critical for SwiGLU (fc11+fc12) and QKV fusion.
///
/// Groups larger than [`MAX_SHARED_INPUT_MATMULS`] (or whose concatenated
/// weights exceed [`MAX_SHARED_INPUT_WEIGHT_ELEMS`]) are left unfused.
/// F5 DiT otherwise packs ~23 AdaLN linears on the same time embed into one
/// ~0.5 GiB Concat weight; sharded wgpu cannot bind/stage that B correctly
/// (MatMul term collapses; AdaLN Gemm matches bias alone).
pub struct FuseSharedInputMatMul;

/// SwiGLU (2), QKV (3), and MoE shared-expert tails (4) stay fused.
const MAX_SHARED_INPUT_MATMULS: usize = 4;
/// Soft cap (~128 MiB f32) — keeps moderate packs; skips F5-scale AdaLN packs.
const MAX_SHARED_INPUT_WEIGHT_ELEMS: usize = 32 * 1024 * 1024;

impl Pass for FuseSharedInputMatMul {
    fn name(&self) -> &str {
        "fuse_shared_input_matmul"
    }

    fn run(&self, graph: Graph) -> Graph {
        struct FuseGroup {
            input_id: NodeId,
            matmul_ids: Vec<NodeId>,
        }

        let mut input_to_matmuls: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for node in graph.nodes() {
            if matches!(node.op, Op::MatMul) {
                input_to_matmuls
                    .entry(node.inputs[0])
                    .or_default()
                    .push(node.id);
            }
        }

        let mut groups: Vec<FuseGroup> = Vec::new();
        for (input_id, matmul_ids) in input_to_matmuls {
            if matmul_ids.len() < 2 || matmul_ids.len() > MAX_SHARED_INPUT_MATMULS {
                continue;
            }
            let first = graph.node(matmul_ids[0]);
            let w0 = graph.shape(first.inputs[1]);
            if w0.rank() != 2 {
                continue;
            }
            let compatible = matmul_ids.iter().all(|&id| {
                let m = graph.node(id);
                matches!(m.op, Op::MatMul)
                    && graph.shape(m.inputs[1]).rank() == 2
                    && graph.shape(m.inputs[1]).dim(0) == w0.dim(0)
            });
            if !compatible {
                continue;
            }
            let weight_elems: usize = matmul_ids
                .iter()
                .map(|&id| {
                    graph
                        .shape(graph.node(id).inputs[1])
                        .num_elements()
                        .unwrap_or(usize::MAX)
                })
                .fold(0usize, |acc, n| acc.saturating_add(n));
            if weight_elems > MAX_SHARED_INPUT_WEIGHT_ELEMS {
                continue;
            }
            groups.push(FuseGroup {
                input_id,
                matmul_ids,
            });
        }

        if groups.is_empty() {
            return graph;
        }

        let group_by_first: HashMap<NodeId, &FuseGroup> =
            groups.iter().map(|g| (g.matmul_ids[0], g)).collect();

        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for g in &groups {
            for &id in &g.matmul_ids[1..] {
                fused_away.insert(id, ());
            }
        }

        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            if let Some(group) = group_by_first.get(&node.id) {
                let matmuls: Vec<_> = group.matmul_ids.iter().map(|&id| graph.node(id)).collect();
                let weight_ids: Vec<NodeId> = matmuls.iter().map(|m| m.inputs[1]).collect();
                rw.ensure_mapped(&graph, std::slice::from_ref(&group.input_id));
                rw.ensure_mapped(&graph, &weight_ids);

                let w0_shape = graph.shape(weight_ids[0]);
                let k = w0_shape.dim(0).unwrap_static();
                let ns: Vec<usize> = weight_ids
                    .iter()
                    .map(|&w| graph.shape(w).dim(1).unwrap_static())
                    .collect();
                let combined_n: usize = ns.iter().sum();

                let concat_shape = Shape::new(&[k, combined_n], w0_shape.dtype());
                let concat_id = rw.add_fused(Op::Concat { axis: 1 }, &weight_ids, concat_shape);

                let out_rank = matmuls[0].shape.rank();
                let mut mm_dims: Vec<Dim> =
                    (0..out_rank).map(|i| matmuls[0].shape.dim(i)).collect();
                mm_dims[out_rank - 1] = Dim::Static(combined_n);
                let mm_shape = Shape::from_dims(&mm_dims, matmuls[0].shape.dtype());
                let mm_id = rw.new_graph.add_node(
                    Op::MatMul,
                    vec![rw.map(group.input_id), concat_id],
                    mm_shape,
                );

                let mut start = 0usize;
                for (mm, &n) in matmuls.iter().zip(&ns) {
                    let narrow = rw.new_graph.add_node(
                        Op::Narrow {
                            axis: out_rank - 1,
                            start,
                            len: n,
                        },
                        vec![mm_id],
                        mm.shape.clone(),
                    );
                    rw.replace(mm.id, narrow);
                    start += n;
                }
                continue;
            }

            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

// ── Pass 4: Detect SwiGLU pattern → FusedSwiGLU ────────────────────────
