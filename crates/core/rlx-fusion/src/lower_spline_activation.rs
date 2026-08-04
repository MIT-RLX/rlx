// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::SplineActivation` (KAN Gaussian-RBF spline) to primitive ops, so
//! every backend without a native kernel still runs it correctly. Unlike
//! `SynthMatMul` (U8 indices — see `lower_synth_matmul`), the operands here are
//! all f32, so this decomposition runs on GPU backends too.
//!
//! Forward: `y[.., c] = Σ_g coeff[c,g]·exp(-inv_h²·(x[.., c] − center_g)²)`. The
//! decomposition builds the `[.., C, G]` RBF basis and contracts it with the
//! coefficients over the basis axis:
//!
//! ```text
//!   x_b     = Expand(Reshape(x, [.., C, 1]), [.., C, G])
//!   cen_b   = Expand(Const(centers,[1..1,G]), [.., C, G])
//!   basis   = Exp( (x_b − cen_b)² · (−inv_h²) )
//!   coeff_b = Expand(Reshape(coeff, [1..1,C,G]), [.., C, G])
//!   y       = ReduceSum(basis · coeff_b, axis = last)          # [.., C]
//! ```

use crate::pass::Pass;
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::*;
use std::collections::HashMap;

pub struct LowerSplineActivation;

impl Pass for LowerSplineActivation {
    fn name(&self) -> &str {
        "lower_spline_activation"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::SplineActivation { .. }))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = match &node.op {
                Op::SplineActivation {
                    num_basis,
                    grid_min,
                    grid_max,
                } => {
                    let g = *num_basis as usize;
                    let step = if g > 1 {
                        (grid_max - grid_min) / (g as f32 - 1.0)
                    } else {
                        1.0
                    };
                    let neg_inv_h2 = -1.0f32 / (step * step);

                    let x = id_map[&node.inputs[0]];
                    let coeff = id_map[&node.inputs[1]];
                    let x_shape = graph.node(node.inputs[0]).shape.clone();
                    let xr = x_shape.rank();
                    let x_dims: Vec<usize> =
                        (0..xr).map(|i| x_shape.dim(i).unwrap_static()).collect();
                    let c = x_dims[xr - 1];

                    let mut w_dims = x_dims.clone();
                    w_dims.push(g);
                    let w_shape = Shape::new(&w_dims, DType::F32);
                    let w_i64: Vec<i64> = w_dims.iter().map(|&d| d as i64).collect();
                    let mut xe_dims = x_dims.clone();
                    xe_dims.push(1);
                    let xe_i64: Vec<i64> = xe_dims.iter().map(|&d| d as i64).collect();

                    // x → [.., C, 1] → [.., C, G]
                    let x_e = new_graph.add_node(
                        Op::Reshape { new_shape: xe_i64 },
                        vec![x],
                        Shape::new(&xe_dims, DType::F32),
                    );
                    let x_b = new_graph.add_node(
                        Op::Expand {
                            target_shape: w_i64.clone(),
                        },
                        vec![x_e],
                        w_shape.clone(),
                    );

                    // centers [1..1, G] → [.., C, G]
                    let centers: Vec<f32> = (0..g).map(|gi| grid_min + gi as f32 * step).collect();
                    let cen_bytes: Vec<u8> = centers.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let mut cen_dims = vec![1usize; xr];
                    cen_dims.push(g);
                    let cen_c = new_graph.add_node(
                        Op::Constant { data: cen_bytes },
                        vec![],
                        Shape::new(&cen_dims, DType::F32),
                    );
                    let cen_b = new_graph.add_node(
                        Op::Expand {
                            target_shape: w_i64.clone(),
                        },
                        vec![cen_c],
                        w_shape.clone(),
                    );

                    // basis = exp((x − center)² · −inv_h²)
                    let diff = new_graph.add_node(
                        Op::Binary(BinaryOp::Sub),
                        vec![x_b, cen_b],
                        w_shape.clone(),
                    );
                    let dsq = new_graph.add_node(
                        Op::Binary(BinaryOp::Mul),
                        vec![diff, diff],
                        w_shape.clone(),
                    );
                    let k_bytes = neg_inv_h2.to_le_bytes().to_vec();
                    let k = new_graph.add_node(
                        Op::Constant { data: k_bytes },
                        vec![],
                        Shape::from_dims(&[Dim::Static(1)], DType::F32),
                    );
                    let scaled = new_graph.add_node(
                        Op::Binary(BinaryOp::Mul),
                        vec![dsq, k],
                        w_shape.clone(),
                    );
                    let basis = new_graph.add_node(
                        Op::Activation(Activation::Exp),
                        vec![scaled],
                        w_shape.clone(),
                    );

                    // coeff [1..1, C, G] → [.., C, G]
                    let mut co_dims = vec![1usize; xr - 1];
                    co_dims.push(c);
                    co_dims.push(g);
                    let co_i64: Vec<i64> = co_dims.iter().map(|&d| d as i64).collect();
                    let coeff_r = new_graph.add_node(
                        Op::Reshape { new_shape: co_i64 },
                        vec![coeff],
                        Shape::new(&co_dims, DType::F32),
                    );
                    let coeff_b = new_graph.add_node(
                        Op::Expand {
                            target_shape: w_i64,
                        },
                        vec![coeff_r],
                        w_shape.clone(),
                    );

                    // y = Σ_g basis · coeff  → [.., C]
                    let prod = new_graph.add_node(
                        Op::Binary(BinaryOp::Mul),
                        vec![basis, coeff_b],
                        w_shape,
                    );
                    new_graph.add_node(
                        Op::Reduce {
                            op: ReduceOp::Sum,
                            axes: vec![xr], // trailing num_basis axis
                            keep_dim: false,
                        },
                        vec![prod],
                        node.shape.clone(),
                    )
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
