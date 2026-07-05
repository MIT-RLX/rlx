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

//! `swiglu` — extracted from the `fusion` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

// ── Helper: graph rewriter ──────────────────────────────────────────────

use crate::graph_rewrite::Rewriter;

// ── Pass 1: MatMul + Bias + Activation → FusedMatMulBiasAct ─────────────

use super::*;

/// Detects the post-`FuseSharedInputMatMul` SwiGLU pattern and replaces it
/// with a single `Op::FusedSwiGLU` node consuming the concatenated matmul.
///
/// Pattern (after `FuseSharedInputMatMul` has fused fc11+fc12 into one mm):
///   %cat   = matmul(%x, concat(%fc11_w, %fc12_w))   ; shape [..., 2N]
///   %up    = narrow(%cat, axis=-1, 0, N)            ; shape [..., N]
///   %gate  = narrow(%cat, axis=-1, N, N)            ; shape [..., N]
///   %silu  = silu(%gate)
///   %out   = mul(%up, %silu)
///
/// Becomes:
///   %out   = fused_swiglu(%cat)
///
/// Saves three kernel launches (two narrows + silu + mul → one kernel) and
/// keeps up/gate resident in registers.
///
/// Single-use guard: only fuses when each intermediate (narrow, narrow, silu)
/// has exactly one consumer. The mul may have any number of consumers.
pub struct FuseSwiGLU;

impl Pass for FuseSwiGLU {
    fn name(&self) -> &str {
        "fuse_swiglu"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Scan for Mul nodes whose two inputs match the SwiGLU pattern.
        // Collect rewrites first, then rebuild.
        // up_narrow_id / silu_id / gate_narrow_id are kept for pattern-shape
        // self-documentation even though only the rewrite path reads
        // mul_id / cat_id / out_n.
        #[allow(dead_code)]
        struct Match {
            mul_id: NodeId,
            up_narrow_id: NodeId,
            silu_id: NodeId,
            gate_narrow_id: NodeId,
            cat_id: NodeId,
            out_n: usize,
            gate_first: bool,
        }

        let mut matches: Vec<Match> = Vec::new();
        let mut consumed: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            // Looking for: mul(narrow(cat, 0, n), silu(narrow(cat, n, n)))
            //   — or symmetrically with up/gate swapped.
            if !matches!(node.op, Op::Binary(BinaryOp::Mul)) {
                continue;
            }
            let lhs_id = node.inputs[0];
            let rhs_id = node.inputs[1];
            let lhs = graph.node(lhs_id);
            let rhs = graph.node(rhs_id);

            // Decide which side is silu(gate) — the silu branch.
            let (up_narrow, silu_id, silu_node) =
                if matches!(rhs.op, Op::Activation(Activation::Silu)) {
                    (lhs, rhs_id, rhs)
                } else if matches!(lhs.op, Op::Activation(Activation::Silu)) {
                    (rhs, lhs_id, lhs)
                } else {
                    continue;
                };

            // up side must be a Narrow.
            let (up_axis, up_start, up_len) = match &up_narrow.op {
                Op::Narrow { axis, start, len } => (*axis, *start, *len),
                _ => continue,
            };
            // silu input must be a Narrow.
            let gate_narrow_id = silu_node.inputs[0];
            let gate_narrow = graph.node(gate_narrow_id);
            let (g_axis, g_start, g_len) = match &gate_narrow.op {
                Op::Narrow { axis, start, len } => (*axis, *start, *len),
                _ => continue,
            };

            // Both narrows must come from the same source on the same axis,
            // covering the two halves: (0..N) and (N..2N).
            if up_narrow.inputs[0] != gate_narrow.inputs[0] {
                continue;
            }
            if up_axis != g_axis {
                continue;
            }
            if up_len != g_len {
                continue;
            }
            let n = up_len;
            // Canonical: up @ 0, gate @ N. Swapped (gate-first builders): gate @ 0, up @ N.
            let gate_first = up_start == n && g_start == 0;
            if !(gate_first || (up_start == 0 && g_start == n)) {
                continue;
            }

            // Single-use checks: narrows feed only into silu+mul, silu feeds
            // only into mul. The cat itself can have arbitrary other users.
            if graph.use_count(up_narrow.id) != 1 {
                continue;
            }
            if graph.use_count(gate_narrow_id) != 1 {
                continue;
            }
            if graph.use_count(silu_id) != 1 {
                continue;
            }

            matches.push(Match {
                mul_id: node.id,
                up_narrow_id: up_narrow.id,
                silu_id,
                gate_narrow_id,
                cat_id: up_narrow.inputs[0],
                out_n: n,
                gate_first,
            });
            consumed.insert(up_narrow.id, ());
            consumed.insert(gate_narrow_id, ());
            consumed.insert(silu_id, ());
        }

        if matches.is_empty() {
            return graph;
        }

        // Rebuild graph, replacing matched mul nodes with FusedSwiGLU.
        let mut rw = Rewriter::new(&graph.name);
        let match_by_mul: HashMap<NodeId, &Match> = matches.iter().map(|m| (m.mul_id, m)).collect();

        for node in graph.nodes() {
            if consumed.contains_key(&node.id) {
                continue;
            }

            if let Some(m) = match_by_mul.get(&node.id) {
                // Output shape = mul's output shape (= [..., N]).
                let out_shape = node.shape.clone();
                debug_assert_eq!(
                    out_shape.dim(out_shape.rank() - 1).unwrap_static(),
                    m.out_n,
                    "FuseSwiGLU: output last dim should be N"
                );
                let fused_id = rw.add_fused(
                    Op::FusedSwiGLU {
                        cast_to: None,
                        gate_first: m.gate_first,
                    },
                    &[m.cat_id],
                    out_shape,
                );
                rw.replace(node.id, fused_id);
                continue;
            }

            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

// ── Pass 5: Fuse Attention Block (QKV → SDPA → OutProj) ────────────────
