// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::SynthReconstruct` to primitives — the all-backend correctness
//! oracle (bit-identical to the fused kernel). `w_bt[n,k]` from indices `[n, k/d]`
//! + codebook: `Cast → Reshape → Gather → Reshape(→w_bt[n,k])`. The caller emits
//! the `Transpose` to `W[k,n]` separately. Backends with the native fused kernel
//! (Metal) keep the node.

use crate::pass::Pass;
use rlx_ir::*;
use std::collections::HashMap;

pub struct LowerSynthReconstruct;

impl Pass for LowerSynthReconstruct {
    fn name(&self) -> &str {
        "lower_synth_reconstruct"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::SynthReconstruct { .. }))
        {
            return graph;
        }
        let mut g = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        for node in graph.nodes() {
            let new_id = match &node.op {
                Op::SynthReconstruct {
                    kind: SynthKind::Codebook { entry_dim, .. },
                } => {
                    let d = (*entry_dim as usize).max(1);
                    let indices = id_map[&node.inputs[0]];
                    let codebook = id_map[&node.inputs[1]];
                    let idx_shape = &graph.node(node.inputs[0]).shape;
                    let n = idx_shape.dim(0).unwrap_static();
                    let kb = idx_shape.dim(1).unwrap_static();
                    let (k, p) = (kb * d, n * kb);
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
                    // → w_bt[n,k] (node.shape); the forward `Transpose` is emitted by the caller.
                    let _ = k;
                    g.add_node(
                        Op::Reshape {
                            new_shape: vec![n as i64, (kb * d) as i64],
                        },
                        vec![rows],
                        node.shape.clone(),
                    )
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
