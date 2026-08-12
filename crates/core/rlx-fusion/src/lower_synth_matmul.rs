// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::SynthMatMul` (codebook weight-synthesis matmul) to primitive
//! ops, so every backend without a native synthesis kernel still runs it
//! correctly. This is the all-backend correctness oracle.
//!
//! `SynthKind::Codebook` stores the weight transposed (`[n, k]`) as codebook
//! indices; the decomposition reconstructs the dense weight via a gather and
//! defers to the portable `Op::MatMul`:
//!
//! ```text
//!   idx_i64  = Cast(indices, I64)                     # [n, k/d]
//!   idx_flat = Reshape(idx_i64, [n·(k/d)])            # [n·(k/d)]
//!   rows     = Gather(codebook, idx_flat, axis=0)     # [n·(k/d), d]
//!   w_bt     = Reshape(rows, [n, k])                  # [n, k]
//!   w_kn     = Transpose(w_bt, [1,0])                 # [k, n]
//!   out      = MatMul(x, w_kn)                        # [m, n]
//! ```
//!
//! Backends that claim `SynthMatMul` natively (the CPU oracle, and Metal once
//! the fused kernel lands) keep the fused node; this pass only fires when the
//! op is unsupported for the target.

use crate::pass::Pass;
use rlx_ir::*;
use std::collections::HashMap;

pub struct LowerSynthMatMul;

impl Pass for LowerSynthMatMul {
    // Lifted from the scan `run` already performs: without these kinds
    // the pass rebuilds the graph node-for-node and returns it unchanged.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::SynthMatMul]
    }

    fn name(&self) -> &str {
        "lower_synth_matmul"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::SynthMatMul { .. }))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = match &node.op {
                Op::SynthMatMul {
                    kind: SynthKind::Codebook { entry_dim, .. },
                } => {
                    // Inputs: x[M,K], indices[N,K/d] (u8), codebook[num_entries,d].
                    let d = (*entry_dim as usize).max(1);
                    let x = id_map[&node.inputs[0]];
                    let indices = id_map[&node.inputs[1]];
                    let codebook = id_map[&node.inputs[2]];

                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let idx_shape = &graph.node(node.inputs[1]).shape;
                    let m = x_shape.dim(0).unwrap_static();
                    let k = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                    let n = idx_shape.dim(0).unwrap_static();
                    let kb = k / d; // codebook blocks per output column
                    let p = n * kb; // total gathered rows

                    // 1) Codebook indices → i64 (Gather requires i64 indices).
                    let idx_i64 = new_graph.add_node(
                        Op::Cast { to: DType::I64 },
                        vec![indices],
                        Shape::new(&[n, kb], DType::I64),
                    );
                    // 2) Flatten to a single gather axis.
                    let idx_flat = new_graph.add_node(
                        Op::Reshape {
                            new_shape: vec![p as i64],
                        },
                        vec![idx_i64],
                        Shape::new(&[p], DType::I64),
                    );
                    // 3) Gather centroids: [n·kb, d].
                    let rows = new_graph.add_node(
                        Op::Gather { axis: 0 },
                        vec![codebook, idx_flat],
                        Shape::new(&[p, d], DType::F32),
                    );
                    // 4) Reassemble the transposed weight [n, k].
                    let w_bt = new_graph.add_node(
                        Op::Reshape {
                            new_shape: vec![n as i64, k as i64],
                        },
                        vec![rows],
                        Shape::new(&[n, k], DType::F32),
                    );
                    // 5) Transpose to [k, n] — the standard matmul weight layout.
                    let w_kn = new_graph.add_node(
                        Op::Transpose { perm: vec![1, 0] },
                        vec![w_bt],
                        Shape::new(&[k, n], DType::F32),
                    );
                    // 6) out = x · W = [m, n].
                    let _ = m;
                    new_graph.add_node(Op::MatMul, vec![x, w_kn], node.shape.clone())
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
