// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::ScaledGroupedMatMul` (MXFP4 / low-precision *grouped* MoE GEMM)
//! to primitive ops, so every backend that lacks a native FP4-grouped kernel
//! still runs it correctly.
//!
//! The op is the expert-indexed analogue of `Op::ScaledMatMul`: both operands
//! are packed [`ScaledFormat`] codes with per-block (`scale_layout`) rescaling,
//! `weight [E,N,K]` carries one TN slab per expert, and `expert_idx [M]` routes
//! each token. The decomposition reconstructs f32 operands and defers to the
//! already-portable `Op::GroupedMatMul` segmented GEMM:
//!
//! ```text
//!   input_f32  = ScaledDequantize(input_codes, input_s)          # [M,K]
//!   weight_f32 = ScaledDequantize(weight_codes, weight_s)        # [E,N,K]
//!   weight_kn  = Transpose(weight_f32, [0,2,1])                  # [E,K,N]
//!   out        = GroupedMatMul(input_f32, weight_kn, expert_idx) # [M,N]
//!   out       += Gather(bias, expert_idx)  (iff has_bias)        # [M,N]
//! ```
//!
//! Backends that DO claim `ScaledGroupedMatMul` natively (CPU oracle, CUDA /
//! ROCm FP4 tensor-core, Metal host-stage) keep the fused node; this pass only
//! fires when the op is unsupported for the target.

use crate::pass::Pass;
use rlx_ir::op::BinaryOp;
use rlx_ir::*;
use std::collections::HashMap;

pub struct LowerScaledGroupedMatMul;

impl Pass for LowerScaledGroupedMatMul {
    // Lifted from the scan `run` already performs: without these kinds
    // the pass rebuilds the graph node-for-node and returns it unchanged.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::ScaledGroupedMatMul]
    }

    fn name(&self) -> &str {
        "lower_scaled_grouped_matmul"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::ScaledGroupedMatMul { .. }))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = match &node.op {
                Op::ScaledGroupedMatMul {
                    lhs_format,
                    rhs_format,
                    scale_layout,
                    has_bias,
                } => {
                    // Inputs: input_codes[M,K], weight_codes[E,N,K], input_s,
                    // weight_s, expert_idx[M], (bias[E,N]).
                    let input = id_map[&node.inputs[0]];
                    let weight = id_map[&node.inputs[1]];
                    let input_s = id_map[&node.inputs[2]];
                    let weight_s = id_map[&node.inputs[3]];
                    let expert_idx = id_map[&node.inputs[4]];

                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let w_shape = &graph.node(node.inputs[1]).shape;
                    let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
                    let k = in_shape.dim(in_shape.rank() - 1).unwrap_static();
                    let e = w_shape.dim(0).unwrap_static();
                    let n = w_shape.dim(w_shape.rank() - 2).unwrap_static();

                    // 1) Dequantize activations [M,K] → f32.
                    let input_f32 = new_graph.add_node(
                        Op::ScaledDequantize {
                            format: *lhs_format,
                            scale_layout: *scale_layout,
                        },
                        vec![input, input_s],
                        Shape::new(&[m, k], DType::F32),
                    );
                    // 2) Dequantize the packed expert stack as [E·N, K] (2D keeps
                    //    the ScaledDequantize host-fallbacks unambiguous — scales
                    //    are already row-major [E·N, nblk]) and reshape to [E,N,K].
                    let w_codes_2d = new_graph.add_node(
                        Op::Reshape {
                            new_shape: vec![(e * n) as i64, k as i64],
                        },
                        vec![weight],
                        Shape::new(&[e * n, k], DType::U8),
                    );
                    let w_f32_2d = new_graph.add_node(
                        Op::ScaledDequantize {
                            format: *rhs_format,
                            scale_layout: *scale_layout,
                        },
                        vec![w_codes_2d, weight_s],
                        Shape::new(&[e * n, k], DType::F32),
                    );
                    let w_f32_3d = new_graph.add_node(
                        Op::Reshape {
                            new_shape: vec![e as i64, n as i64, k as i64],
                        },
                        vec![w_f32_2d],
                        Shape::new(&[e, n, k], DType::F32),
                    );
                    // 3) Transpose to [E,K,N] — GroupedMatMul's weight layout.
                    let weight_kn = new_graph.add_node(
                        Op::Transpose {
                            perm: vec![0, 2, 1],
                        },
                        vec![w_f32_3d],
                        Shape::new(&[e, k, n], DType::F32),
                    );
                    // 4) Indexed batched (segmented) matmul.
                    let out_base = new_graph.add_node(
                        Op::GroupedMatMul,
                        vec![input_f32, weight_kn, expert_idx],
                        Shape::new(&[m, n], DType::F32),
                    );
                    // 5) Optional per-expert bias: out += bias[expert_idx].
                    if *has_bias {
                        let bias = id_map[&node.inputs[5]];
                        let eidx_i64 = new_graph.add_node(
                            Op::Cast { to: DType::I64 },
                            vec![expert_idx],
                            Shape::new(&[m], DType::I64),
                        );
                        let bias_tok = new_graph.add_node(
                            Op::Gather { axis: 0 },
                            vec![bias, eidx_i64],
                            Shape::new(&[m, n], DType::F32),
                        );
                        new_graph.add_node(
                            Op::Binary(BinaryOp::Add),
                            vec![out_base, bias_tok],
                            node.shape.clone(),
                        )
                    } else {
                        out_base
                    }
                }
                _ => {
                    let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                    new_graph.add_node(node.op.clone(), inputs, node.shape.clone())
                }
            };
            id_map.insert(node.id, new_id);
        }

        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}
