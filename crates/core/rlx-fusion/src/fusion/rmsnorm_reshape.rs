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

//! `rmsnorm_reshape` — extracted from the `fusion` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

// ── Helper: graph rewriter ──────────────────────────────────────────────

use crate::graph_rewrite::Rewriter;

// ── Pass 1: MatMul + Bias + Activation → FusedMatMulBiasAct ─────────────

use super::*;

/// Fuses `rms_norm([…, H]) → reshape([∏leading, H])` into a single
/// `RmsNorm` with the flattened output shape, eliminating a memcpy.
///
/// Matches the Qwen3.5 pre-norm pattern where normalized activations
/// are immediately reshaped to 2-D for matmul.
pub struct FuseRmsNormReshape;

impl Pass for FuseRmsNormReshape {
    fn name(&self) -> &str {
        "fuse_rms_norm_reshape"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut is_output: HashMap<NodeId, ()> = HashMap::new();
        for &oid in &graph.outputs {
            is_output.insert(oid, ());
        }

        let mut flat_shape: HashMap<NodeId, Shape> = HashMap::new();
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for node in graph.nodes() {
            if let Op::RmsNorm { .. } = &node.op {
                if graph.use_count(node.id) != 1 || is_output.contains_key(&node.id) {
                    continue;
                }
                let Some(reshape_id) = sole_consumer(&graph, node.id) else {
                    continue;
                };
                if is_output.contains_key(&reshape_id) {
                    continue;
                }
                let reshape = graph.node(reshape_id);
                if let Op::Reshape { new_shape } = &reshape.op {
                    if let Some(flat) = leading_flatten_shape(&node.shape, new_shape) {
                        flat_shape.insert(node.id, flat);
                        fused_away.insert(reshape_id, ());
                    }
                }
            }
        }

        let mut rw = Rewriter::new(&graph.name);

        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            if let Op::RmsNorm { axis, eps, .. } = &node.op {
                if let Some(flat) = flat_shape.get(&node.id) {
                    let Some(reshape_id) = sole_consumer(&graph, node.id) else {
                        rw.copy_node(node);
                        continue;
                    };
                    let fused_id = rw.add_fused(
                        Op::RmsNorm {
                            axis: *axis,
                            eps: *eps,
                        },
                        &node.inputs,
                        flat.clone(),
                    );
                    rw.replace(node.id, fused_id);
                    rw.replace(reshape_id, fused_id);
                    continue;
                }
            }

            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

// ── Pass 3b: Dual MatMul SwiGLU (gate+up before shared-input concat) ─────
