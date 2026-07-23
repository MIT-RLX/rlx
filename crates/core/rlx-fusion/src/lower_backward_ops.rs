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

//! Lower dedicated backward ops (`ReluBackward`, `ActivationBackward`) to
//! primitives (`Compare`, `Where`, `Binary`, `Activation`) for backends
//! that do not implement closed-form gradient kernels (e.g. Metal).

use std::collections::HashMap;

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, CmpOp};
use rlx_ir::*;

fn scalar_const(g: &mut Graph, v: f32, dtype: DType) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], dtype),
    )
}

fn broadcast_like(g: &mut Graph, scalar: NodeId, like: NodeId) -> NodeId {
    let like_shape = g.shape(like);
    let dims: Vec<usize> = like_shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let target: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
    let dtype = like_shape.dtype();
    g.add_node(
        Op::Expand {
            target_shape: target,
        },
        vec![scalar],
        Shape::new(&dims, dtype),
    )
}

fn broadcast_scalar(g: &mut Graph, v: f32, like: NodeId) -> NodeId {
    let dtype = g.shape(like).dtype();
    let s = scalar_const(g, v, dtype);
    broadcast_like(g, s, like)
}

fn broadcast_channel_param(g: &mut Graph, param: NodeId, like: NodeId) -> NodeId {
    let like_dims: Vec<usize> = g
        .shape(like)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let rank = like_dims.len();
    let dtype = g.shape(like).dtype();
    let channels = like_dims[rank - 1];
    let mut pad_dims = vec![1i64; rank.saturating_sub(1)];
    pad_dims.push(channels as i64);
    let padded = g.reshape_(param, pad_dims);
    let target: Vec<i64> = like_dims.iter().map(|&d| d as i64).collect();
    let shape = Shape::new(
        &target.iter().map(|&d| d as usize).collect::<Vec<_>>(),
        dtype,
    );
    g.add_node(
        Op::Expand {
            target_shape: target,
        },
        vec![padded],
        shape,
    )
}

fn reduce_all_but_last(g: &mut Graph, x: NodeId) -> NodeId {
    let rank = g.shape(x).rank();
    if rank <= 1 {
        return x;
    }
    g.sum(x, (0..rank - 1).collect(), false)
}

fn lower_batch_norm_inference_backward_input(
    g: &mut Graph,
    x: NodeId,
    gamma: NodeId,
    _mean: NodeId,
    var: NodeId,
    dy: NodeId,
    out_shape: Shape,
    eps: f32,
) -> NodeId {
    let _ = out_shape;
    let var_b = broadcast_channel_param(g, var, x);
    let eps_arr = broadcast_scalar(g, eps, var_b);
    let var_eps = g.add(var_b, eps_arr);
    let sqrt_var = g.sqrt(var_eps);
    let one = broadcast_scalar(g, 1.0, var_b);
    let inv_std = g.div(one, sqrt_var);
    let gamma_b = broadcast_channel_param(g, gamma, x);
    let scale = g.mul(gamma_b, inv_std);
    g.mul(dy, scale)
}

fn lower_batch_norm_inference_backward_gamma(
    g: &mut Graph,
    x: NodeId,
    mean: NodeId,
    var: NodeId,
    dy: NodeId,
    out_shape: Shape,
    eps: f32,
) -> NodeId {
    let _ = out_shape;
    let mean_b = broadcast_channel_param(g, mean, x);
    let var_b = broadcast_channel_param(g, var, x);
    let x_centered = g.sub(x, mean_b);
    let eps_arr = broadcast_scalar(g, eps, var_b);
    let var_eps = g.add(var_b, eps_arr);
    let sqrt_var = g.sqrt(var_eps);
    let one = broadcast_scalar(g, 1.0, var_b);
    let inv_std = g.div(one, sqrt_var);
    let xhat = g.mul(x_centered, inv_std);
    let prod = g.mul(dy, xhat);
    reduce_all_but_last(g, prod)
}

fn lower_batch_norm_inference_backward_beta(g: &mut Graph, dy: NodeId, out_shape: Shape) -> NodeId {
    let _ = out_shape;
    reduce_all_but_last(g, dy)
}

fn compare_gt(g: &mut Graph, lhs: NodeId, rhs: NodeId) -> NodeId {
    let s = shape::compare_shape(g.shape(lhs), g.shape(rhs)).expect("gt compare");
    g.add_node(Op::Compare(CmpOp::Gt), vec![lhs, rhs], s)
}

fn where_(g: &mut Graph, cond: NodeId, on_true: NodeId, on_false: NodeId, out: Shape) -> NodeId {
    g.add_node(Op::Where, vec![cond, on_true, on_false], out)
}

/// `dx = dy where x > 0 else 0`.
fn lower_relu_backward(g: &mut Graph, x: NodeId, dy: NodeId, out_shape: Shape) -> NodeId {
    let zero = broadcast_scalar(g, 0.0, x);
    let mask = compare_gt(g, x, zero);
    where_(g, mask, dy, zero, out_shape)
}

/// Closed-form activation backward using primitives. Mirrors
/// `rlx-cpu/src/thunk.rs::activation_backward_kernel` and MLX compose.
fn lower_activation_backward(
    g: &mut Graph,
    kind: Activation,
    x: NodeId,
    dy: NodeId,
    out_shape: Shape,
) -> NodeId {
    match kind {
        Activation::Relu => lower_relu_backward(g, x, dy, out_shape),
        Activation::Log => g.div(dy, x),
        Activation::Exp => {
            let ex = g.exp(x);
            g.mul(dy, ex)
        }
        Activation::Neg => g.neg(dy),
        Activation::Round => dy,
        Activation::Sigmoid => {
            let s = g.activation(Activation::Sigmoid, x, shape::unary_shape(g.shape(x)));
            let one = broadcast_scalar(g, 1.0, x);
            let one_minus_s = g.sub(one, s);
            let factor = g.mul(s, one_minus_s);
            g.mul(dy, factor)
        }
        Activation::Tanh => {
            let t = g.tanh(x);
            let t_sq = g.mul(t, t);
            let one = broadcast_scalar(g, 1.0, x);
            let factor = g.sub(one, t_sq);
            g.mul(dy, factor)
        }
        Activation::Silu => {
            let s = g.activation(Activation::Sigmoid, x, shape::unary_shape(g.shape(x)));
            let one = broadcast_scalar(g, 1.0, x);
            let one_minus_s = g.sub(one, s);
            let x_times = g.mul(x, one_minus_s);
            let inner = g.add(one, x_times);
            let factor = g.mul(s, inner);
            g.mul(dy, factor)
        }
        Activation::Sqrt => {
            let s = g.sqrt(x);
            let half = broadcast_scalar(g, 0.5, x);
            let num = g.mul(dy, half);
            let grad = g.div(num, s);
            let zero = broadcast_scalar(g, 0.0, x);
            let pos = compare_gt(g, x, zero);
            where_(g, pos, grad, zero, out_shape)
        }
        Activation::Rsqrt => {
            let s = g.sqrt(x);
            let neg_half = broadcast_scalar(g, -0.5, x);
            let xs = g.mul(x, s);
            let num = g.mul(dy, neg_half);
            let grad = g.div(num, xs);
            let zero = broadcast_scalar(g, 0.0, x);
            let pos = compare_gt(g, x, zero);
            where_(g, pos, grad, zero, out_shape)
        }
        Activation::Abs => {
            let zero = broadcast_scalar(g, 0.0, x);
            let pos = compare_gt(g, x, zero);
            let neg = g.neg(dy);
            where_(g, pos, dy, neg, out_shape)
        }
        Activation::Sin => {
            let c = g.activation(Activation::Cos, x, shape::unary_shape(g.shape(x)));
            g.mul(dy, c)
        }
        Activation::Cos => {
            let s = g.activation(Activation::Sin, x, shape::unary_shape(g.shape(x)));
            let prod = g.mul(dy, s);
            g.neg(prod)
        }
        Activation::Tan => {
            let t = g.tanh(x);
            let t_sq = g.mul(t, t);
            let one = broadcast_scalar(g, 1.0, x);
            let factor = g.add(one, t_sq);
            g.mul(dy, factor)
        }
        Activation::Atan => {
            let x_sq = g.mul(x, x);
            let one = broadcast_scalar(g, 1.0, x);
            let denom = g.add(one, x_sq);
            g.div(dy, denom)
        }
        Activation::Recip => {
            let x_sq = g.mul(x, x);
            let neg_dy = g.neg(dy);
            g.div(neg_dy, x_sq)
        }
        Activation::Gelu | Activation::GeluApprox => {
            lower_gelu_approx_backward(g, x, dy, out_shape)
        }
    }
}

/// Tanh-approximation GELU backward (works for both `Gelu` and `GeluApprox`).
fn lower_gelu_approx_backward(g: &mut Graph, x: NodeId, dy: NodeId, _out_shape: Shape) -> NodeId {
    const C: f32 = 0.797_884_6;
    const A: f32 = 0.044_715;

    let half = broadcast_scalar(g, 0.5, x);
    let one = broadcast_scalar(g, 1.0, x);
    let c_arr = broadcast_scalar(g, C, x);
    let a_arr = broadcast_scalar(g, A, x);
    let three_a = broadcast_scalar(g, 3.0 * A, x);

    let x_sq = g.mul(x, x);
    let x_cu = g.mul(x_sq, x);
    let a_x_cu = g.mul(a_arr, x_cu);
    let inner_sum = g.add(x, a_x_cu);
    let inner = g.mul(c_arr, inner_sum);
    let t = g.tanh(inner);
    let one_plus_t = g.add(one, t);
    let term1 = g.mul(half, one_plus_t);
    let t_sq = g.mul(t, t);
    let one_minus_t_sq = g.sub(one, t_sq);
    let three_a_x_sq = g.mul(three_a, x_sq);
    let one_plus_3ax2 = g.add(one, three_a_x_sq);
    let dinner = g.mul(c_arr, one_plus_3ax2);
    let half_x = g.mul(half, x);
    let part2_a = g.mul(half_x, one_minus_t_sq);
    let term2 = g.mul(part2_a, dinner);
    let deriv = g.add(term1, term2);
    g.mul(dy, deriv)
}

/// Rewrite `ReluBackward` / `ActivationBackward` / BatchNorm inference backward nodes to primitive ops.
pub struct LowerBackwardOps;

impl Pass for LowerBackwardOps {
    fn name(&self) -> &str {
        "lower_backward_ops"
    }

    fn run(&self, graph: Graph) -> Graph {
        let needs = graph.nodes().iter().any(|n| {
            matches!(
                n.op,
                Op::ReluBackward
                    | Op::ActivationBackward { .. }
                    | Op::BatchNormInferenceBackwardInput { .. }
                    | Op::BatchNormInferenceBackwardGamma { .. }
                    | Op::BatchNormInferenceBackwardBeta
            )
        });
        if !needs {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = match &node.op {
                Op::ReluBackward => {
                    let x = id_map[&node.inputs[0]];
                    let dy = id_map[&node.inputs[1]];
                    lower_relu_backward(&mut new_graph, x, dy, node.shape.clone())
                }
                Op::ActivationBackward { kind } => {
                    let x = id_map[&node.inputs[0]];
                    let dy = id_map[&node.inputs[1]];
                    lower_activation_backward(&mut new_graph, *kind, x, dy, node.shape.clone())
                }
                Op::BatchNormInferenceBackwardInput { eps } => {
                    let x = id_map[&node.inputs[0]];
                    let gamma = id_map[&node.inputs[1]];
                    let mean = id_map[&node.inputs[2]];
                    let var = id_map[&node.inputs[3]];
                    let dy = id_map[&node.inputs[4]];
                    lower_batch_norm_inference_backward_input(
                        &mut new_graph,
                        x,
                        gamma,
                        mean,
                        var,
                        dy,
                        node.shape.clone(),
                        *eps,
                    )
                }
                Op::BatchNormInferenceBackwardGamma { eps } => {
                    let x = id_map[&node.inputs[0]];
                    let mean = id_map[&node.inputs[1]];
                    let var = id_map[&node.inputs[2]];
                    let dy = id_map[&node.inputs[3]];
                    lower_batch_norm_inference_backward_gamma(
                        &mut new_graph,
                        x,
                        mean,
                        var,
                        dy,
                        node.shape.clone(),
                        *eps,
                    )
                }
                Op::BatchNormInferenceBackwardBeta => {
                    let dy = id_map[&node.inputs[0]];
                    lower_batch_norm_inference_backward_beta(&mut new_graph, dy, node.shape.clone())
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
