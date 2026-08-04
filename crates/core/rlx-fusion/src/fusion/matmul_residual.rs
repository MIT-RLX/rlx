// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `matmul_residual` — fuse `Add(MatMul(a, b), residual)` into
//! [`Op::FusedMatMulResidual`] so a backend can fold the transformer residual
//! add into the matmul's store instead of a separate elementwise-add dispatch.
//! Registered only for backends that claim `OpKind::FusedMatMulResidual`
//! (today: Metal, where the saving matters on a launch-latency-bound decode);
//! everyone else keeps the plain `MatMul` + `Add`.

#![allow(unused_imports)]

use crate::graph_rewrite::Rewriter;
use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

use super::*;

/// Fuses `matmul → add(residual)` into a single [`Op::FusedMatMulResidual`].
///
/// The residual is a **full** `[m, n]` tensor (same shape as the matmul
/// result) — the transformer's `add(skip, o_proj)` / `add(h, down_proj)`. This
/// is deliberately distinct from [`FuseMatMulBiasAct`], which only matches a
/// rank-≤1 broadcast bias; the two never compete for the same `Add`.
pub struct FuseMatMulResidual;

impl Pass for FuseMatMulResidual {
    fn name(&self) -> &str {
        "fuse_matmul_residual"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut rw = Rewriter::new(&graph.name);
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            if matches!(node.op, Op::MatMul) {
                let mm_id = node.id;
                // The matmul must feed ONLY the add (its result is consumed
                // solely by the residual). The add's own output may have any
                // number of users (skip stream + norm) — that is fine.
                let mm_users = graph.users(mm_id);
                if mm_users.len() == 1 {
                    let add_node = graph.node(mm_users[0]);
                    if let Op::Binary(BinaryOp::Add) = &add_node.op {
                        let residual_id = if add_node.inputs[0] == mm_id {
                            add_node.inputs[1]
                        } else {
                            add_node.inputs[0]
                        };

                        // Only fuse a genuine elementwise residual: the added
                        // tensor must match the matmul output shape exactly
                        // (no broadcast) and be rank > 1 (rank-≤1 is the bias
                        // fusion's domain). The kernel adds `R[o]` per element.
                        let mm_shape = graph.shape(mm_id);
                        let res_shape = graph.shape(residual_id);
                        let rhs_shape = graph.shape(node.inputs[1]);
                        // The residual-epilogue kernel is f32-only (output,
                        // residual, AND weight). Leave the matmul+add unfused
                        // when any is non-f32 — notably an f16-resident weight,
                        // which routes to the half-precision gemv instead.
                        let same_shape = mm_shape.dtype() == DType::F32
                            && res_shape.dtype() == DType::F32
                            && rhs_shape.dtype() == DType::F32
                            && mm_shape.rank() == res_shape.rank()
                            && mm_shape.rank() > 1
                            && (0..mm_shape.rank()).all(|d| mm_shape.dim(d) == res_shape.dim(d));

                        if same_shape {
                            let add_id = add_node.id;
                            rw.ensure_mapped(
                                &graph,
                                &[node.inputs[0], node.inputs[1], residual_id],
                            );
                            let fused_id = rw.add_fused(
                                Op::FusedMatMulResidual,
                                &[node.inputs[0], node.inputs[1], residual_id],
                                add_node.shape.clone(),
                            );
                            rw.replace(mm_id, fused_id);
                            rw.replace(add_id, fused_id);
                            fused_away.insert(add_id, ());
                            continue;
                        }
                    }
                }
            }

            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}
