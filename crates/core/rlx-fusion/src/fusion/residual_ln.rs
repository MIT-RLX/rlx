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

/// Fuses `add(x, residual) → layer_norm` into FusedResidualLN.
///
/// Also detects `add(x, residual) → add(bias) → layer_norm` for the
/// bias variant (used in BERT's output projection).
pub struct FuseResidualLN;


impl Pass for FuseResidualLN {
    fn name(&self) -> &str {
        "fuse_residual_ln"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Graph outputs hold implicit references to their producing
        // nodes that don't show up in any node's `inputs` (use_count
        // walks node inputs only). Treat being-a-graph-output as a
        // use so we don't fuse-away an intermediate the caller still
        // wants to read — this used to silently corrupt multi-block
        // encoders (e.g. SAM 2 stage outputs) by collapsing the
        // residual add of block N into block N+1's LN.
        let mut is_output: HashMap<NodeId, ()> = HashMap::new();
        for &oid in &graph.outputs {
            is_output.insert(oid, ());
        }
        // Pre-scan: find all Add nodes consumed by LayerNorm
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for node in graph.nodes() {
            if let Op::LayerNorm { .. } = &node.op {
                let ln_input_id = node.inputs[0];
                let ln_input = graph.node(ln_input_id);
                if matches!(ln_input.op, Op::Binary(BinaryOp::Add))
                    && graph.use_count(ln_input_id) == 1
                    && !is_output.contains_key(&ln_input_id)
                {
                    fused_away.insert(ln_input_id, ());
                }
            }
        }

        let mut rw = Rewriter::new(&graph.name);

        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            if let Op::LayerNorm { eps, .. } = &node.op {
                let ln_input_id = node.inputs[0];
                let ln_input = graph.node(ln_input_id);

                if matches!(ln_input.op, Op::Binary(BinaryOp::Add))
                    && fused_away.contains_key(&ln_input_id)
                {
                    let (x_id, residual_id) = (ln_input.inputs[0], ln_input.inputs[1]);
                    let gamma_id = node.inputs[1];
                    let beta_id = node.inputs[2];

                    let fused_id = rw.add_fused(
                        Op::FusedResidualLN {
                            has_bias: false,
                            eps: *eps,
                        },
                        &[x_id, residual_id, gamma_id, beta_id],
                        node.shape.clone(),
                    );

                    rw.replace(ln_input_id, fused_id);
                    rw.replace(node.id, fused_id);
                    continue;
                }
            }

            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

// ── Pass 2b: Add(residual) + RmsNorm → FusedResidualRmsNorm ─────────────

