// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::SplineActivationBackwardX` / `SplineActivationBackwardCoeff` to
//! primitives — the all-backend correctness oracle (bit-identical to the fused
//! Metal kernels). Rebuilds the KAN Gaussian-RBF basis over `[.., C, G]` and
//! contracts it, exactly like the original `vjp_spline_activation`. Backends with
//! the native fused kernels (Metal) claim the ops and keep them.

use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

pub struct LowerSplineActivationBackward;

impl Pass for LowerSplineActivationBackward {
    // Lifted from the scan `run` already performs: without these kinds
    // the pass rebuilds the graph node-for-node and returns it unchanged.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[
            OpKind::SplineActivationBackwardX,
            OpKind::SplineActivationBackwardCoeff,
        ]
    }

    fn name(&self) -> &str {
        "lower_spline_activation_backward"
    }

    fn run(&self, graph: Graph) -> Graph {
        let has = graph.nodes().iter().any(|n| {
            matches!(
                n.op,
                Op::SplineActivationBackwardX { .. } | Op::SplineActivationBackwardCoeff { .. }
            )
        });
        if !has {
            return graph;
        }
        let mut g = Graph::new(&graph.name);
        let mut m: HashMap<NodeId, NodeId> = HashMap::new();
        for node in graph.nodes() {
            let new_id = match &node.op {
                Op::SplineActivationBackwardX {
                    num_basis,
                    grid_min,
                    grid_max,
                } => {
                    let x = m[&node.inputs[0]];
                    let coeff = m[&node.inputs[1]];
                    let up = m[&node.inputs[2]];
                    lower_dx(&mut g, x, coeff, up, *num_basis, *grid_min, *grid_max)
                }
                Op::SplineActivationBackwardCoeff {
                    num_basis,
                    grid_min,
                    grid_max,
                } => {
                    let x = m[&node.inputs[0]];
                    let up = m[&node.inputs[1]];
                    lower_dcoeff(
                        &mut g,
                        x,
                        up,
                        node.shape.clone(),
                        *num_basis,
                        *grid_min,
                        *grid_max,
                    )
                }
                _ => {
                    let inputs: Vec<NodeId> = node.inputs.iter().map(|i| m[i]).collect();
                    g.add_node(node.op.clone(), inputs, node.shape.clone())
                }
            };
            m.insert(node.id, new_id);
        }
        let outs: Vec<NodeId> = graph.outputs.iter().map(|i| m[i]).collect();
        g.set_outputs(outs);
        g
    }
}

/// Shared: build `basis[.., C, G]` and `diff[.., C, G]` from `x`.
fn build_basis(
    g: &mut Graph,
    x: NodeId,
    x_shape: &Shape,
    nb: u32,
    grid_min: f32,
    grid_max: f32,
) -> (NodeId, NodeId, Shape, usize) {
    let gb = nb as usize;
    let step = if gb > 1 {
        (grid_max - grid_min) / (gb as f32 - 1.0)
    } else {
        1.0
    };
    let inv_h2 = 1.0f32 / (step * step);
    let xr = x_shape.rank();
    let x_dims: Vec<usize> = (0..xr).map(|i| x_shape.dim(i).unwrap_static()).collect();
    let mut w_dims = x_dims.clone();
    w_dims.push(gb);
    let w_shape = Shape::new(&w_dims, DType::F32);
    let w_i64: Vec<i64> = w_dims.iter().map(|&d| d as i64).collect();
    let mut xe = x_dims.clone();
    xe.push(1);
    let xe_i64: Vec<i64> = xe.iter().map(|&d| d as i64).collect();
    let xe_shape = Shape::new(&xe, DType::F32);
    let expand = |g: &mut Graph, src: NodeId| {
        g.add_node(
            Op::Expand {
                target_shape: w_i64.clone(),
            },
            vec![src],
            w_shape.clone(),
        )
    };
    let x_e = g.add_node(Op::Reshape { new_shape: xe_i64 }, vec![x], xe_shape.clone());
    let x_b = expand(g, x_e);
    let centers: Vec<f32> = (0..gb).map(|gi| grid_min + gi as f32 * step).collect();
    let cen_bytes: Vec<u8> = centers.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut cen_dims = vec![1usize; xr];
    cen_dims.push(gb);
    let centers_c = g.add_node(
        Op::Constant { data: cen_bytes },
        vec![],
        Shape::new(&cen_dims, DType::F32),
    );
    let centers_b = expand(g, centers_c);
    let diff = g.add_node(
        Op::Binary(BinaryOp::Sub),
        vec![x_b, centers_b],
        w_shape.clone(),
    );
    let dsq = g.add_node(Op::Binary(BinaryOp::Mul), vec![diff, diff], w_shape.clone());
    let neg = scalar(g, -inv_h2);
    let scaled = g.add_node(Op::Binary(BinaryOp::Mul), vec![dsq, neg], w_shape.clone());
    let basis = g.add_node(
        Op::Activation(Activation::Exp),
        vec![scaled],
        w_shape.clone(),
    );
    (basis, diff, w_shape, xr)
}

fn lower_dx(
    g: &mut Graph,
    x: NodeId,
    coeff: NodeId,
    up: NodeId,
    nb: u32,
    grid_min: f32,
    grid_max: f32,
) -> NodeId {
    let x_shape = g.node(x).shape.clone();
    let coeff_shape = g.node(coeff).shape.clone();
    let step = if nb > 1 {
        (grid_max - grid_min) / (nb as f32 - 1.0)
    } else {
        1.0
    };
    let inv_h2 = 1.0f32 / (step * step);
    let (basis, diff, w_shape, xr) = build_basis(g, x, &x_shape, nb, grid_min, grid_max);
    let c = coeff_shape.dim(0).unwrap_static();
    let w_i64: Vec<i64> = (0..w_shape.rank())
        .map(|i| w_shape.dim(i).unwrap_static() as i64)
        .collect();
    // coeff [1,..,1,C,G] → [.., C, G]
    let mut co = vec![1usize; xr - 1];
    co.push(c);
    co.push(nb as usize);
    let co_i64: Vec<i64> = co.iter().map(|&d| d as i64).collect();
    let coeff_r = g.add_node(
        Op::Reshape { new_shape: co_i64 },
        vec![coeff],
        Shape::new(&co, DType::F32),
    );
    let coeff_b = g.add_node(
        Op::Expand {
            target_shape: w_i64,
        },
        vec![coeff_r],
        w_shape.clone(),
    );
    let m2 = scalar(g, -2.0 * inv_h2);
    let dfac = g.add_node(Op::Binary(BinaryOp::Mul), vec![diff, m2], w_shape.clone());
    let bder = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![basis, dfac],
        w_shape.clone(),
    );
    let weighted = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![bder, coeff_b],
        w_shape.clone(),
    );
    let dydx = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![xr],
            keep_dim: false,
        },
        vec![weighted],
        x_shape.clone(),
    );
    g.add_node(Op::Binary(BinaryOp::Mul), vec![up, dydx], x_shape)
}

fn lower_dcoeff(
    g: &mut Graph,
    x: NodeId,
    up: NodeId,
    dcoeff_shape: Shape,
    nb: u32,
    grid_min: f32,
    grid_max: f32,
) -> NodeId {
    let x_shape = g.node(x).shape.clone();
    let (basis, _diff, w_shape, xr) = build_basis(g, x, &x_shape, nb, grid_min, grid_max);
    let w_i64: Vec<i64> = (0..w_shape.rank())
        .map(|i| w_shape.dim(i).unwrap_static() as i64)
        .collect();
    let mut xe: Vec<i64> = (0..xr)
        .map(|i| x_shape.dim(i).unwrap_static() as i64)
        .collect();
    xe.push(1);
    let up_e = g.add_node(
        Op::Reshape {
            new_shape: xe.clone(),
        },
        vec![up],
        Shape::new(
            &xe.iter().map(|&d| d as usize).collect::<Vec<_>>(),
            DType::F32,
        ),
    );
    let up_b = g.add_node(
        Op::Expand {
            target_shape: w_i64,
        },
        vec![up_e],
        w_shape.clone(),
    );
    let p = g.add_node(Op::Binary(BinaryOp::Mul), vec![basis, up_b], w_shape);
    // Σ over leading batch axes 0..xr-1 → [C, G].
    let axes: Vec<usize> = (0..xr - 1).collect();
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes,
            keep_dim: false,
        },
        vec![p],
        dcoeff_shape,
    )
}

fn scalar(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    )
}
