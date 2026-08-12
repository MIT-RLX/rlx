// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::SynthMatMulBackward` to primitive ops — the all-backend correctness
//! oracle, bit-identical to the expansion the generic synth VJP used to emit
//! inline. Backends with a native fused backward kernel (Metal) keep the node;
//! this pass only fires when the op is unsupported for the target.
//!
//! Inputs: `[x [m,k], indices [n,k/d] (u8), codebook [ne,d], upstream [m,n]]`.
//!
//! ```text
//!   wrt = Dx:                                  wrt = Codebook:
//!     idx_i64 = Cast(indices, I64)               up_t   = Transpose(upstream,[1,0])  # [n,m]
//!     rows    = Gather(codebook, idx_flat)       grad_w = MatMul(up_t, x)            # [n,k]
//!     w_bt    = Reshape(rows, [n,k])             blocks = Reshape(grad_w, [n·kb, d])
//!     dx      = MatMul(upstream, w_bt)  # [m,k]  dcb    = ScatterAdd(blocks, idx_f32) # [ne,d]
//! ```

use crate::pass::Pass;
use rlx_ir::*;
use std::collections::HashMap;

pub struct LowerSynthMatMulBackward;

impl Pass for LowerSynthMatMulBackward {
    // Lifted from the scan `run` already performs: without these kinds
    // the pass rebuilds the graph node-for-node and returns it unchanged.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::SynthMatMulBackward]
    }

    fn name(&self) -> &str {
        "lower_synth_matmul_backward"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::SynthMatMulBackward { .. }))
        {
            return graph;
        }

        let mut g = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = match &node.op {
                Op::SynthMatMulBackward {
                    kind: SynthKind::Codebook { entry_dim, .. },
                    wrt,
                } => {
                    let d = (*entry_dim as usize).max(1);
                    let x = id_map[&node.inputs[0]];
                    let indices = id_map[&node.inputs[1]];
                    let codebook = id_map[&node.inputs[2]];
                    let upstream = id_map[&node.inputs[3]];

                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let idx_shape = &graph.node(node.inputs[1]).shape;
                    let m = x_shape.dim(0).unwrap_static();
                    let k = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                    let n = idx_shape.dim(0).unwrap_static();
                    let kb = k / d;
                    let p = n * kb;

                    match wrt {
                        SynthBwdWrt::Dx => {
                            // Reconstruct Ŵᵀ [n,k] via the same gather the forward uses.
                            let idx_i64 = g.add_node(
                                Op::Cast { to: DType::I64 },
                                vec![indices],
                                Shape::new(&[n, kb], DType::I64),
                            );
                            let idx_flat = g.add_node(
                                Op::Reshape {
                                    new_shape: vec![p as i64],
                                },
                                vec![idx_i64],
                                Shape::new(&[p], DType::I64),
                            );
                            let rows = g.add_node(
                                Op::Gather { axis: 0 },
                                vec![codebook, idx_flat],
                                Shape::new(&[p, d], DType::F32),
                            );
                            let w_bt = g.add_node(
                                Op::Reshape {
                                    new_shape: vec![n as i64, k as i64],
                                },
                                vec![rows],
                                Shape::new(&[n, k], DType::F32),
                            );
                            // dx = upstream · Ŵᵀ : [m,n]·[n,k] → [m,k].
                            g.add_node(Op::MatMul, vec![upstream, w_bt], node.shape.clone())
                        }
                        SynthBwdWrt::Codebook => {
                            // grad_W = upstreamᵀ · x : [n,m]·[m,k] → [n,k].
                            let up_t = g.add_node(
                                Op::Transpose { perm: vec![1, 0] },
                                vec![upstream],
                                Shape::new(&[n, m], DType::F32),
                            );
                            let grad_w = g.add_node(
                                Op::MatMul,
                                vec![up_t, x],
                                Shape::new(&[n, k], DType::F32),
                            );
                            let blocks = g.add_node(
                                Op::Reshape {
                                    new_shape: vec![p as i64, d as i64],
                                },
                                vec![grad_w],
                                Shape::new(&[p, d], DType::F32),
                            );
                            // ScatterAdd reads f32-encoded indices, scatters along axis 0.
                            let idx_f32 = g.add_node(
                                Op::Cast { to: DType::F32 },
                                vec![indices],
                                Shape::new(&[n, kb], DType::F32),
                            );
                            let idx_f32_flat = g.add_node(
                                Op::Reshape {
                                    new_shape: vec![p as i64],
                                },
                                vec![idx_f32],
                                Shape::new(&[p], DType::F32),
                            );
                            g.add_node(
                                Op::ScatterAdd,
                                vec![blocks, idx_f32_flat],
                                node.shape.clone(),
                            )
                        }
                    }
                }
                _ => {
                    let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                    g.add_node(node.op.clone(), inputs, node.shape.clone())
                }
            };
            id_map.insert(node.id, new_id);
        }

        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
        g.set_outputs(new_outputs);
        g
    }
}
