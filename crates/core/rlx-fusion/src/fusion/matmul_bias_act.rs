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

//! Fusion passes — pattern-match and replace subgraphs with fused ops.
//!
//! Each pass scans the graph in reverse topological order, looking for
//! specific multi-node patterns and replacing them with single fused nodes.
//! These are the same fusions we hand-coded in burnembed's ndarray_fused.rs.

#![allow(unused_imports)]

use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

// ── Helper: graph rewriter ──────────────────────────────────────────────

use crate::graph_rewrite::Rewriter;

// ── Pass 1: MatMul + Bias + Activation → FusedMatMulBiasAct ─────────────

use super::*;

/// Fuses `matmul → add(bias) → activation` into a single FusedMatMulBiasAct.
///
/// This is the single most impactful fusion — it eliminates two intermediate
/// tensors and three memory passes (matmul write, bias read+write, act read+write)
/// down to one (matmul write with inline bias+activation).
///
/// Also fuses `matmul → add(bias)` without activation.
///
/// Epilogue activations are fused only when every backend can apply them
/// inline with the matmul (today: Gelu and Silu). Other activations — e.g.
/// Exp in qwen35 softplus — stay as separate ops so Metal does not silently
/// drop the epilogue.
pub struct FuseMatMulBiasAct;


impl Pass for FuseMatMulBiasAct {
    fn name(&self) -> &str {
        "fuse_matmul_bias_act"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut rw = Rewriter::new(&graph.name);
        // Track which nodes are consumed by fusion (skip them in copy)
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();

        // Forward pass: copy nodes, detect patterns
        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            // Pattern: MatMul → Add(bias) → Activation
            // or:      MatMul → Add(bias)
            if matches!(node.op, Op::MatMul) {
                let mm_id = node.id;
                let mm_users: Vec<_> = graph.users(mm_id);

                // Check for single-use Add(bias) consumer
                if mm_users.len() == 1 {
                    let add_node = graph.node(mm_users[0]);
                    if let Op::Binary(BinaryOp::Add) = &add_node.op {
                        // Determine which input is the bias (the non-matmul one)
                        let (bias_id, _mm_input) = if add_node.inputs[0] == mm_id {
                            (add_node.inputs[1], add_node.inputs[0])
                        } else {
                            (add_node.inputs[0], add_node.inputs[1])
                        };

                        // Check if bias is a param/const with broadcastable shape
                        let bias_shape = graph.shape(bias_id);
                        if bias_shape.rank() <= 1 {
                            let add_id = add_node.id;
                            let add_users = graph.users(add_id);

                            // Check for activation consumer
                            let mut activation = None;
                            let mut act_id = None;
                            if add_users.len() == 1 {
                                let act_node = graph.node(add_users[0]);
                                if let Op::Activation(a) = &act_node.op
                                    && fusible_mm_bias_epilogue_activation(*a)
                                {
                                    activation = Some(*a);
                                    act_id = Some(act_node.id);
                                }
                            }

                            // Emit fused node. Bias may be declared after
                            // the matmul in the source graph — copy it early
                            // instead of requiring builders to order params first.
                            let out_shape = if let Some(aid) = act_id {
                                graph.shape(aid).clone()
                            } else {
                                add_node.shape.clone()
                            };

                            rw.ensure_mapped(&graph, &[node.inputs[0], node.inputs[1], bias_id]);
                            let fused_id = rw.add_fused(
                                Op::FusedMatMulBiasAct { activation },
                                &[node.inputs[0], node.inputs[1], bias_id],
                                out_shape,
                            );

                            // Map old nodes to the fused result
                            rw.replace(mm_id, fused_id);
                            rw.replace(add_id, fused_id);
                            fused_away.insert(add_id, ());
                            if let Some(aid) = act_id {
                                rw.replace(aid, fused_id);
                                fused_away.insert(aid, ());
                            }
                            continue;
                        }
                    }
                }
            }

            // No fusion — copy as-is
            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

// ── Pass 2: Add(residual) + LayerNorm → FusedResidualLN ─────────────────

