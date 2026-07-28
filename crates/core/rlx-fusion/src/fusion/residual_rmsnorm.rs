// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `residual_rmsnorm` — extracted from the `fusion` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

// ── Helper: graph rewriter ──────────────────────────────────────────────

use crate::graph_rewrite::Rewriter;

// ── Pass 1: MatMul + Bias + Activation → FusedMatMulBiasAct ─────────────

use super::*;

/// Fuses `add(x, residual) → rms_norm` into [`Op::FusedResidualRmsNorm`].
pub struct FuseResidualRmsNorm;

impl Pass for FuseResidualRmsNorm {
    fn name(&self) -> &str {
        "fuse_residual_rms_norm"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut is_output: HashMap<NodeId, ()> = HashMap::new();
        for &oid in &graph.outputs {
            is_output.insert(oid, ());
        }
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for node in graph.nodes() {
            if let Op::RmsNorm { .. } = &node.op {
                let rn_input_id = node.inputs[0];
                let rn_input = graph.node(rn_input_id);
                if matches!(rn_input.op, Op::Binary(BinaryOp::Add))
                    && graph.use_count(rn_input_id) == 1
                    && !is_output.contains_key(&rn_input_id)
                {
                    fused_away.insert(rn_input_id, ());
                }
            }
        }

        let mut rw = Rewriter::new(&graph.name);

        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            if let Op::RmsNorm { eps, .. } = &node.op {
                let rn_input_id = node.inputs[0];
                let rn_input = graph.node(rn_input_id);

                if matches!(rn_input.op, Op::Binary(BinaryOp::Add))
                    && fused_away.contains_key(&rn_input_id)
                {
                    let (x_id, residual_id) = (rn_input.inputs[0], rn_input.inputs[1]);
                    let gamma_id = node.inputs[1];
                    let beta_id = node.inputs[2];

                    let fused_id = rw.add_fused(
                        Op::FusedResidualRmsNorm {
                            has_bias: false,
                            eps: *eps,
                        },
                        &[x_id, residual_id, gamma_id, beta_id],
                        node.shape.clone(),
                    );

                    rw.replace(rn_input_id, fused_id);
                    rw.replace(node.id, fused_id);
                    continue;
                }
            }

            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

// ── Pass 2c: RmsNorm → Reshape(leading flatten) ─────────────────────────
