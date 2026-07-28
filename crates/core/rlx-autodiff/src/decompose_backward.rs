// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decompose first-order `*Backward` kernels into primitive MIR so a second
//! reverse-mode pass can differentiate through them.

use rlx_ir::op::{Activation, BinaryOp, OpKind};
use rlx_ir::{Graph, NodeId, Op, Shape};

use crate::activation_deriv::activation_deriv_wrt_x;
use crate::compose::broadcast_scalar;
use crate::decompose_backward_kernels::{
    SCAN_DECOMPOSE_MAX_LENGTH, compose_ada_layer_norm_backward, compose_conv2d_backward_input,
    compose_conv2d_backward_weight, compose_conv2d_backward_weight_im2col,
    compose_conv3d_backward_input, compose_conv3d_backward_weight, compose_cumsum_backward,
    compose_fake_quantize_backward, compose_gated_residual_backward, compose_gather_backward,
    compose_group_norm_backward_beta, compose_group_norm_backward_gamma,
    compose_group_norm_backward_input, compose_layer_norm_backward_gamma,
    compose_layer_norm_backward_input, compose_max_pool2d_backward, compose_max_pool3d_backward,
    compose_rms_norm_backward_beta, compose_rms_norm_backward_gamma,
    compose_rms_norm_backward_input, compose_rope_backward, compose_scan_backward,
    compose_scan_backward_xs, compose_softmax_cross_entropy_backward, conv_di_decompose_eligible,
    conv_dw_im2col_eligible, emit_attention_backward,
};

/// Rewrite `*Backward` ops into primitive chains; copy everything else.
pub fn decompose_backward_ops(g: Graph) -> Graph {
    decompose_backward_ops_except(g, &[])
}

/// Like [`decompose_backward_ops`] but leaves any op whose kind appears in
/// `preserved` untouched. Useful when a backend has fused kernels for
/// some `*Backward` ops but not others (e.g. `LayerNormBackwardInput`
/// available natively, but `LayerNormBackwardGamma` needs the primitive
/// chain to run efficiently).
pub fn decompose_backward_ops_except(g: Graph, preserved: &[OpKind]) -> Graph {
    let mut g = g;
    for _ in 0..6 {
        if !contains_training_backward_except(&g, preserved) {
            break;
        }
        g = decompose_backward_ops_once_except(g, preserved);
    }
    g
}

fn contains_training_backward_except(g: &Graph, preserved: &[OpKind]) -> bool {
    g.nodes().iter().any(|n| {
        if preserved.contains(&n.op.kind()) {
            return false;
        }
        matches!(
            n.op,
            Op::ReluBackward
                | Op::ActivationBackward { .. }
                | Op::LayerNormBackwardInput { .. }
                | Op::LayerNormBackwardGamma { .. }
                | Op::RmsNormBackwardInput { .. }
                | Op::RmsNormBackwardGamma { .. }
                | Op::RmsNormBackwardBeta { .. }
                | Op::GroupNormBackwardInput { .. }
                | Op::GroupNormBackwardGamma { .. }
                | Op::GroupNormBackwardBeta { .. }
                | Op::RopeBackward { .. }
                | Op::Conv2dBackwardInput { .. }
                | Op::Conv2dBackwardWeight { .. }
                | Op::MaxPool2dBackward { .. }
                | Op::Conv3dBackwardInput { .. }
                | Op::Conv3dBackwardWeight { .. }
                | Op::MaxPool3dBackward { .. }
                | Op::AttentionBackward { .. }
                | Op::CumsumBackward { .. }
                | Op::GatherBackward { .. }
                | Op::SoftmaxCrossEntropyBackward
                | Op::FakeQuantizeBackward { .. }
                | Op::ScanBackward { .. }
                | Op::ScanBackwardXs { .. }
                | Op::AdaLayerNormBackward { .. }
                | Op::GatedResidualBackward
        )
    })
}

fn decompose_backward_ops_once_except(g: Graph, preserved: &[OpKind]) -> Graph {
    let mut out = Graph::new(format!("{}_decompose", g.name));
    let mut id_map = std::collections::HashMap::new();

    for node in g.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        // If this op kind is preserved, copy verbatim — skip the
        // decomposition arms below.
        if preserved.contains(&node.op.kind()) {
            let new_id = out.add_node(node.op.clone(), new_inputs, node.shape.clone());
            id_map.insert(node.id, new_id);
            continue;
        }
        let new_id = match &node.op {
            Op::ReluBackward => emit_relu_backward(&mut out, new_inputs, &node.shape),
            Op::ActivationBackward { kind } => {
                emit_activation_backward(&mut out, *kind, new_inputs, &node.shape)
            }
            Op::LayerNormBackwardInput { axis, eps } => {
                let [x, gamma, dy] = new_inputs[..] else {
                    panic!("LayerNormBackwardInput expects [x, gamma, dy]");
                };
                compose_layer_norm_backward_input(&mut out, x, gamma, dy, *axis, *eps, &node.shape)
            }
            Op::LayerNormBackwardGamma { axis, eps } => {
                let [x, dy] = new_inputs[..] else {
                    panic!("LayerNormBackwardGamma expects [x, dy]");
                };
                compose_layer_norm_backward_gamma(&mut out, x, dy, *axis, *eps, &node.shape)
            }
            Op::RmsNormBackwardInput { axis, eps } => {
                let [x, gamma, beta, dy] = new_inputs[..] else {
                    panic!("RmsNormBackwardInput expects [x, gamma, beta, dy]");
                };
                compose_rms_norm_backward_input(
                    &mut out,
                    x,
                    gamma,
                    beta,
                    dy,
                    *axis,
                    *eps,
                    &node.shape,
                )
            }
            Op::RmsNormBackwardGamma { axis, eps } => {
                let [x, _gamma, _beta, dy] = new_inputs[..] else {
                    panic!("RmsNormBackwardGamma expects [x, gamma, beta, dy]");
                };
                compose_rms_norm_backward_gamma(&mut out, x, dy, *axis, *eps, &node.shape)
            }
            Op::RmsNormBackwardBeta { axis: _, eps: _ } => {
                let [_x, _gamma, _beta, dy] = new_inputs[..] else {
                    panic!("RmsNormBackwardBeta expects [x, gamma, beta, dy]");
                };
                compose_rms_norm_backward_beta(&mut out, dy, &node.shape)
            }
            Op::Conv2dBackwardInput {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                let [dy, w] = new_inputs[..] else {
                    panic!("Conv2dBackwardInput expects [dy, w]");
                };
                let static_shapes = g.node(node.inputs[0]).shape.is_static()
                    && g.node(node.inputs[1]).shape.is_static()
                    && node.shape.is_static();
                let di_ok = conv_di_decompose_eligible(
                    &g.node(node.inputs[0]).shape,
                    &g.node(node.inputs[1]).shape,
                    &node.shape,
                );
                if static_shapes || di_ok {
                    compose_conv2d_backward_input(
                        &mut out,
                        dy,
                        w,
                        &node.shape,
                        [kernel_size[0], kernel_size[1]],
                        [stride[0], stride[1]],
                        [padding[0], padding[1]],
                        [dilation[0], dilation[1]],
                        *groups,
                    )
                } else {
                    out.add_node(node.op.clone(), new_inputs, node.shape.clone())
                }
            }
            Op::Conv2dBackwardWeight {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                let [x, dy] = new_inputs[..] else {
                    panic!("Conv2dBackwardWeight expects [x, dy]");
                };
                let static_shapes = g.node(node.inputs[0]).shape.is_static()
                    && g.node(node.inputs[1]).shape.is_static()
                    && node.shape.is_static();
                let im2col_ok = conv_dw_im2col_eligible(
                    &g.node(node.inputs[0]).shape,
                    &g.node(node.inputs[1]).shape,
                    &node.shape,
                );
                if static_shapes {
                    compose_conv2d_backward_weight(
                        &mut out,
                        x,
                        dy,
                        &node.shape,
                        [kernel_size[0], kernel_size[1]],
                        [stride[0], stride[1]],
                        [padding[0], padding[1]],
                        [dilation[0], dilation[1]],
                        *groups,
                    )
                } else if im2col_ok {
                    compose_conv2d_backward_weight_im2col(
                        &mut out,
                        x,
                        dy,
                        &node.shape,
                        [kernel_size[0], kernel_size[1]],
                        [stride[0], stride[1]],
                        [padding[0], padding[1]],
                        [dilation[0], dilation[1]],
                        *groups,
                    )
                } else {
                    out.add_node(node.op.clone(), new_inputs, node.shape.clone())
                }
            }
            Op::GroupNormBackwardInput { num_groups, eps } => {
                let [x, gamma, beta, dy] = new_inputs[..] else {
                    panic!("GroupNormBackwardInput expects [x, gamma, beta, dy]");
                };
                compose_group_norm_backward_input(
                    &mut out,
                    x,
                    gamma,
                    beta,
                    dy,
                    *num_groups,
                    *eps,
                    &node.shape,
                )
            }
            Op::GroupNormBackwardGamma { num_groups, eps } => {
                let [x, dy] = new_inputs[..] else {
                    panic!("GroupNormBackwardGamma expects [x, dy]");
                };
                compose_group_norm_backward_gamma(&mut out, x, dy, *num_groups, *eps, &node.shape)
            }
            Op::GroupNormBackwardBeta {
                num_groups: _,
                eps: _,
            } => {
                let [_x, dy] = new_inputs[..] else {
                    panic!("GroupNormBackwardBeta expects [x, dy]");
                };
                compose_group_norm_backward_beta(&mut out, dy, &node.shape)
            }
            Op::RopeBackward { head_dim, n_rot } => {
                let [dy, cos, sin] = new_inputs[..] else {
                    panic!("RopeBackward expects [dy, cos, sin]");
                };
                compose_rope_backward(&mut out, dy, cos, sin, *head_dim, *n_rot)
            }
            Op::AttentionBackward {
                num_heads,
                head_dim,
                mask_kind,
                wrt,
            } => emit_attention_backward(
                &mut out,
                *wrt,
                &new_inputs,
                &node.shape,
                *num_heads,
                *head_dim,
                *mask_kind,
            ),
            Op::MaxPool2dBackward {
                kernel_size,
                stride,
                padding,
            } => {
                let [x, dy] = new_inputs[..] else {
                    panic!("MaxPool2dBackward expects [x, dy]");
                };
                let ks = [kernel_size[0], kernel_size[1]];
                let st = [stride[0], stride[1]];
                let pad = [padding[0], padding[1]];
                compose_max_pool2d_backward(&mut out, x, dy, &node.shape, ks, st, pad)
            }
            Op::MaxPool3dBackward {
                kernel_size,
                stride,
                padding,
            } => {
                let [x, dy] = new_inputs[..] else {
                    panic!("MaxPool3dBackward expects [x, dy]");
                };
                let ks = [kernel_size[0], kernel_size[1], kernel_size[2]];
                let st = [stride[0], stride[1], stride[2]];
                let pad = [padding[0], padding[1], padding[2]];
                match compose_max_pool3d_backward(&mut out, x, dy, &node.shape, ks, st, pad) {
                    Some(id) => id,
                    None => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
                }
            }
            Op::Conv3dBackwardInput {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                let [dy, w] = new_inputs[..] else {
                    panic!("Conv3dBackwardInput expects [dy, w]");
                };
                compose_conv3d_backward_input(
                    &mut out,
                    dy,
                    w,
                    &node.shape,
                    [kernel_size[0], kernel_size[1], kernel_size[2]],
                    [stride[0], stride[1], stride[2]],
                    [padding[0], padding[1], padding[2]],
                    [dilation[0], dilation[1], dilation[2]],
                    *groups,
                )
            }
            Op::Conv3dBackwardWeight {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                let [x, dy] = new_inputs[..] else {
                    panic!("Conv3dBackwardWeight expects [x, dy]");
                };
                compose_conv3d_backward_weight(
                    &mut out,
                    x,
                    dy,
                    &node.shape,
                    [kernel_size[0], kernel_size[1], kernel_size[2]],
                    [stride[0], stride[1], stride[2]],
                    [padding[0], padding[1], padding[2]],
                    [dilation[0], dilation[1], dilation[2]],
                    *groups,
                )
            }
            Op::CumsumBackward { axis, exclusive } => {
                let [dy] = new_inputs[..] else {
                    panic!("CumsumBackward expects [dy]");
                };
                compose_cumsum_backward(&mut out, dy, &node.shape, *axis, *exclusive)
            }
            Op::GatherBackward { axis } => {
                let [dy, indices] = new_inputs[..] else {
                    panic!("GatherBackward expects [dy, indices]");
                };
                compose_gather_backward(&mut out, dy, indices, &node.shape, *axis)
            }
            Op::SoftmaxCrossEntropyBackward => {
                let [logits, labels, d_loss] = new_inputs[..] else {
                    panic!("SoftmaxCrossEntropyBackward expects [logits, labels, d_loss]");
                };
                compose_softmax_cross_entropy_backward(
                    &mut out,
                    logits,
                    labels,
                    d_loss,
                    &node.shape,
                )
            }
            Op::AdaLayerNormBackward { norm, eps } => {
                let [x, scale, shift, dy] = new_inputs[..] else {
                    panic!("AdaLayerNormBackward expects [x, scale, shift, dy]");
                };
                compose_ada_layer_norm_backward(
                    &mut out,
                    x,
                    scale,
                    shift,
                    dy,
                    *norm,
                    *eps,
                    &node.shape,
                )
            }
            Op::GatedResidualBackward => {
                let [x, y, gate, dy] = new_inputs[..] else {
                    panic!("GatedResidualBackward expects [x, y, gate, dy]");
                };
                compose_gated_residual_backward(&mut out, x, y, gate, dy, &node.shape)
            }
            Op::FakeQuantizeBackward { bits, axis, ste } => {
                let [x, dy] = new_inputs[..] else {
                    panic!("FakeQuantizeBackward expects [x, dy]");
                };
                compose_fake_quantize_backward(&mut out, x, dy, &node.shape, *bits, *axis, *ste)
            }
            Op::ScanBackward {
                body_vjp,
                length,
                save_trajectory,
                num_checkpoints,
                num_xs,
                forward_body,
                ..
            } => {
                let init = new_inputs[0];
                let trajectory = new_inputs[1];
                let upstream = new_inputs[2];
                let xs = &new_inputs[3..3 + *num_xs as usize];
                let vjp = decompose_backward_ops((**body_vjp).clone());
                let decompose_scan = *save_trajectory
                    && *length <= SCAN_DECOMPOSE_MAX_LENGTH
                    && (*num_checkpoints == 0
                        || *num_checkpoints == *length
                        || forward_body.is_some());
                if decompose_scan {
                    compose_scan_backward(
                        &mut out,
                        init,
                        trajectory,
                        upstream,
                        xs,
                        &vjp,
                        forward_body.as_deref(),
                        *length,
                        *save_trajectory,
                        *num_checkpoints,
                        &node.shape,
                    )
                } else {
                    out.add_node(
                        Op::ScanBackward {
                            body_vjp: Box::new(vjp),
                            length: *length,
                            save_trajectory: *save_trajectory,
                            num_checkpoints: *num_checkpoints,
                            num_xs: *num_xs,
                            forward_body: forward_body.clone(),
                        },
                        new_inputs,
                        node.shape.clone(),
                    )
                }
            }
            Op::ScanBackwardXs {
                body_vjp,
                length,
                save_trajectory,
                num_checkpoints,
                num_xs,
                xs_idx,
                forward_body,
                ..
            } => {
                let init = new_inputs[0];
                let trajectory = new_inputs[1];
                let upstream = new_inputs[2];
                let xs = &new_inputs[3..3 + *num_xs as usize];
                let vjp = decompose_backward_ops((**body_vjp).clone());
                let decompose_scan = *save_trajectory
                    && *length <= SCAN_DECOMPOSE_MAX_LENGTH
                    && (*num_checkpoints == 0
                        || *num_checkpoints == *length
                        || forward_body.is_some());
                if decompose_scan {
                    compose_scan_backward_xs(
                        &mut out,
                        init,
                        trajectory,
                        upstream,
                        xs,
                        &vjp,
                        forward_body.as_deref(),
                        *length,
                        *save_trajectory,
                        *num_checkpoints,
                        *xs_idx,
                        &node.shape,
                    )
                } else {
                    out.add_node(
                        Op::ScanBackwardXs {
                            body_vjp: Box::new(vjp),
                            length: *length,
                            save_trajectory: *save_trajectory,
                            num_checkpoints: *num_checkpoints,
                            num_xs: *num_xs,
                            xs_idx: *xs_idx,
                            forward_body: forward_body.clone(),
                        },
                        new_inputs,
                        node.shape.clone(),
                    )
                }
            }
            other => out.add_node(other.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|o| id_map[o]).collect());
    out
}

fn emit_relu_backward(g: &mut Graph, inputs: Vec<NodeId>, out_shape: &Shape) -> NodeId {
    let [x, dy] = inputs[..] else {
        panic!("ReluBackward expects [x, dy]");
    };
    let deriv = activation_deriv_wrt_x(g, Activation::Relu, x, None, out_shape);
    g.binary(BinaryOp::Mul, dy, deriv, out_shape.clone())
}

fn emit_activation_backward(
    g: &mut Graph,
    kind: Activation,
    inputs: Vec<NodeId>,
    out_shape: &Shape,
) -> NodeId {
    let [x, dy] = inputs[..] else {
        panic!("ActivationBackward expects [x, dy]");
    };
    let deriv = activation_deriv_wrt_x(g, kind, x, None, out_shape);
    g.binary(BinaryOp::Mul, dy, deriv, out_shape.clone())
}

/// Prepare a [`crate::autodiff::grad_with_loss`] graph for an outer forward-mode
/// (`jvp`) pass: decompose `*Backward` ops and bake in `d_output = 1`.
pub fn prepare_grad_graph_for_jvp(g: Graph) -> Graph {
    let mut g = decompose_backward_ops(g);
    crate::compose::internalize_d_output(&mut g);
    g
}

/// Prepare a [`crate::autodiff::grad_with_loss`] graph for another reverse pass.
pub fn decompose_backward_for_ad(mut g: Graph, wrt_idx: usize) -> Graph {
    g = decompose_backward_ops(g);
    crate::compose::internalize_d_output(&mut g);
    let grad_out = g.outputs[1 + wrt_idx];
    g.set_outputs(vec![grad_out]);
    g
}

/// Directional contraction: `sum(grad ⊙ direction)`.
pub fn contract_grad_with_direction(g: &mut Graph, grad: NodeId, direction: NodeId) -> NodeId {
    let grad_shape = g.node(grad).shape.clone();
    let dir = broadcast_scalar(g, direction, &grad_shape);
    let prod = g.binary(rlx_ir::op::BinaryOp::Mul, grad, dir, grad_shape.clone());
    if grad_shape.rank() == 0 {
        return prod;
    }
    let axes: Vec<usize> = (0..grad_shape.rank()).collect();
    let out_shape = if axes.len() == grad_shape.rank() {
        Shape::scalar(grad_shape.dtype())
    } else {
        let dims: Vec<rlx_ir::shape::Dim> = grad_shape
            .dims()
            .iter()
            .enumerate()
            .filter_map(|(i, d)| if axes.contains(&i) { None } else { Some(*d) })
            .collect();
        if dims.is_empty() {
            Shape::scalar(grad_shape.dtype())
        } else {
            Shape::from_dims(&dims, grad_shape.dtype())
        }
    };
    g.reduce(prod, rlx_ir::op::ReduceOp::Sum, axes, false, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::grad_with_loss;
    use crate::prepare_ad::prepare_graph_for_ad;
    use rlx_ir::infer::GraphExt;
    use rlx_ir::op::ReduceOp;
    use rlx_ir::{DType, Graph, Op, Shape};

    fn assert_input_backward_decomposed(g: &Graph) {
        for node in g.nodes() {
            assert!(
                !matches!(
                    node.op,
                    Op::ReluBackward
                        | Op::ActivationBackward { .. }
                        | Op::LayerNormBackwardInput { .. }
                        | Op::LayerNormBackwardGamma { .. }
                        | Op::RmsNormBackwardInput { .. }
                        | Op::RmsNormBackwardGamma { .. }
                        | Op::RmsNormBackwardBeta { .. }
                        | Op::GroupNormBackwardInput { .. }
                        | Op::GroupNormBackwardGamma { .. }
                        | Op::GroupNormBackwardBeta { .. }
                        | Op::RopeBackward { .. }
                        | Op::Conv2dBackwardInput { .. }
                        | Op::Conv2dBackwardWeight { .. }
                        | Op::MaxPool2dBackward { .. }
                        | Op::Conv3dBackwardInput { .. }
                        | Op::Conv3dBackwardWeight { .. }
                        | Op::MaxPool3dBackward { .. }
                        | Op::AttentionBackward { .. }
                        | Op::CumsumBackward { .. }
                        | Op::GatherBackward { .. }
                        | Op::SoftmaxCrossEntropyBackward
                        | Op::FakeQuantizeBackward { .. }
                        | Op::ScanBackward { .. }
                        | Op::ScanBackwardXs { .. }
                ),
                "leftover input backward op {:?}",
                node.op
            );
        }
    }

    #[test]
    fn decompose_gelu_activation_backward() {
        use rlx_ir::op::Activation;

        let f = DType::F32;
        let mut g = Graph::new("gelu_decomp");
        let x = g.input("x", Shape::scalar(f));
        let y = g.activation(Activation::Gelu, x, Shape::scalar(f));
        let loss = g.reduce(y, ReduceOp::Sum, vec![], false, Shape::scalar(f));
        g.set_outputs(vec![loss]);
        let prep = prepare_graph_for_ad(g);
        let bwd = grad_with_loss(&prep, &[x]);
        let decomposed = decompose_backward_ops(bwd);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_silu_activation_backward() {
        use rlx_ir::op::Activation;

        let f = DType::F32;
        let mut g = Graph::new("silu_decomp");
        let x = g.input("x", Shape::scalar(f));
        let y = g.activation(Activation::Silu, x, Shape::scalar(f));
        let loss = g.reduce(y, ReduceOp::Sum, vec![], false, Shape::scalar(f));
        g.set_outputs(vec![loss]);
        let prep = prepare_graph_for_ad(g);
        let bwd = grad_with_loss(&prep, &[x]);
        let decomposed = decompose_backward_ops(bwd);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_layer_norm_backward_gamma() {
        let f = DType::F32;
        let shape = Shape::new(&[2, 4], f);
        let mut g = Graph::new("ln_gamma_decomp");
        let x = g.input("x", shape.clone());
        let dy = g.input("dy", shape);
        let dgamma = g.add_node(
            Op::LayerNormBackwardGamma {
                axis: -1,
                eps: 1e-5,
            },
            vec![x, dy],
            Shape::new(&[4], f),
        );
        g.set_outputs(vec![dgamma]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_rms_norm_backward_gamma_and_beta() {
        let f = DType::F32;
        let shape = Shape::new(&[2, 4], f);
        let mut g = Graph::new("rms_gamma_beta_decomp");
        let x = g.input("x", shape.clone());
        let gamma = g.input("gamma", Shape::new(&[4], f));
        let beta = g.input("beta", Shape::new(&[4], f));
        let dy = g.input("dy", shape);
        let dgamma = g.add_node(
            Op::RmsNormBackwardGamma {
                axis: -1,
                eps: 1e-5,
            },
            vec![x, gamma, beta, dy],
            Shape::new(&[4], f),
        );
        let dbeta = g.add_node(
            Op::RmsNormBackwardBeta {
                axis: -1,
                eps: 1e-5,
            },
            vec![x, gamma, beta, dy],
            Shape::new(&[4], f),
        );
        let sum = g.binary(
            rlx_ir::op::BinaryOp::Add,
            dgamma,
            dbeta,
            Shape::new(&[4], f),
        );
        g.set_outputs(vec![sum]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_conv2d_backward_weight() {
        let f = DType::F32;
        let mut g = Graph::new("conv_dw_decomp");
        let x = g.input("x", Shape::new(&[1, 1, 4, 4], f));
        let dy = g.input("dy", Shape::new(&[1, 1, 4, 4], f));
        let dw = g.conv2d_backward_weight(
            x,
            dy,
            Shape::new(&[1, 1, 3, 3], f),
            vec![3, 3],
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
            1,
        );
        g.set_outputs(vec![dw]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_conv2d_backward_weight_groups2() {
        let f = DType::F32;
        let mut g = Graph::new("conv_dw_grp_decomp");
        let x = g.input("x", Shape::new(&[1, 2, 4, 4], f));
        let dy = g.input("dy", Shape::new(&[1, 2, 4, 4], f));
        let dw = g.conv2d_backward_weight(
            x,
            dy,
            Shape::new(&[2, 1, 3, 3], f),
            vec![3, 3],
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
            2,
        );
        g.set_outputs(vec![dw]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_conv2d_backward_weight_tiled_im2col() {
        let f = DType::F32;
        let mut g = Graph::new("conv_dw_tiled_decomp");
        let x = g.input("x", Shape::new(&[1, 4, 32, 32], f));
        let dy = g.input("dy", Shape::new(&[1, 4, 32, 32], f));
        let dw = g.conv2d_backward_weight(
            x,
            dy,
            Shape::new(&[4, 4, 3, 3], f),
            vec![3, 3],
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
            1,
        );
        g.set_outputs(vec![dw]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_group_norm_backward_ops() {
        let f = DType::F32;
        let shape = Shape::new(&[1, 4, 2, 2], f);
        let mut g = Graph::new("gn_decomp");
        let x = g.input("x", shape.clone());
        let gamma = g.input("gamma", Shape::new(&[4], f));
        let beta = g.input("beta", Shape::new(&[4], f));
        let dy = g.input("dy", shape.clone());
        let dx = g.group_norm_backward_input(x, gamma, beta, dy, 2, 1e-5);
        g.set_outputs(vec![dx]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_rope_backward() {
        let f = DType::F32;
        let mut g = Graph::new("rope_decomp");
        let dy = g.input("dy", Shape::new(&[1, 2, 4], f));
        let cos = g.input("cos", Shape::new(&[2], f));
        let sin = g.input("sin", Shape::new(&[2], f));
        let dx = g.rope_backward(dy, cos, sin, 4, 4);
        g.set_outputs(vec![dx]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_max_pool2d_backward() {
        let f = DType::F32;
        let mut g = Graph::new("maxpool_decomp");
        let x = g.input("x", Shape::new(&[1, 1, 4, 4], f));
        let dy = g.input("dy", Shape::new(&[1, 1, 2, 2], f));
        let dx = g.maxpool2d_backward(x, dy, vec![2, 2], vec![2, 2], vec![0, 0]);
        g.set_outputs(vec![dx]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_max_pool3d_backward() {
        let f = DType::F32;
        let mut g = Graph::new("maxpool3d_decomp");
        let x = g.input("x", Shape::new(&[1, 1, 4, 4, 4], f));
        let dy = g.input("dy", Shape::new(&[1, 1, 2, 2, 2], f));
        let dx = g.maxpool3d_backward(x, dy, vec![2, 2, 2], vec![2, 2, 2], vec![0, 0, 0]);
        g.set_outputs(vec![dx]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_conv3d_backward_input() {
        let f = DType::F32;
        let mut g = Graph::new("conv3d_di_decomp");
        let dy = g.input("dy", Shape::new(&[1, 1, 4, 4, 4], f));
        let w = g.input("w", Shape::new(&[1, 1, 3, 3, 3], f));
        let dx = g.conv3d_backward_input(
            dy,
            w,
            Shape::new(&[1, 1, 4, 4, 4], f),
            vec![3, 3, 3],
            vec![1, 1, 1],
            vec![1, 1, 1],
            vec![1, 1, 1],
            1,
        );
        g.set_outputs(vec![dx]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_conv3d_backward_weight() {
        let f = DType::F32;
        let mut g = Graph::new("conv3d_dw_decomp");
        let x = g.input("x", Shape::new(&[1, 1, 4, 4, 4], f));
        let dy = g.input("dy", Shape::new(&[1, 1, 2, 2, 2], f));
        let dw = g.conv3d_backward_weight(
            x,
            dy,
            Shape::new(&[1, 1, 3, 3, 3], f),
            vec![3, 3, 3],
            vec![1, 1, 1],
            vec![0, 0, 0],
            vec![1, 1, 1],
            1,
        );
        g.set_outputs(vec![dw]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn max_pool3d_overlapping_preserved() {
        let f = DType::F32;
        let mut g = Graph::new("maxpool3d_overlap");
        let x = g.input("x", Shape::new(&[1, 1, 4, 4, 4], f));
        let dy = g.input("dy", Shape::new(&[1, 1, 3, 3, 3], f));
        let dx = g.maxpool3d_backward(x, dy, vec![2, 2, 2], vec![1, 1, 1], vec![0, 0, 0]);
        g.set_outputs(vec![dx]);
        let decomposed = decompose_backward_ops(g);
        assert!(
            decomposed
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::MaxPool3dBackward { .. })),
            "overlapping MaxPool3dBackward must be preserved, not panicked"
        );
    }

    #[test]
    fn decompose_rms_norm_backward_input() {
        let f = DType::F32;
        let mut g = Graph::new("rms_decomp");
        let x = g.input("x", Shape::new(&[2, 4], f));
        let gamma = g.input("gamma", Shape::new(&[4], f));
        let beta = g.input("beta", Shape::new(&[4], f));
        let y = g.rms_norm(x, gamma, beta, 1e-5);
        let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1], false, Shape::scalar(f));
        g.set_outputs(vec![loss]);
        let prep = prepare_graph_for_ad(g);
        let bwd = grad_with_loss(&prep, &[x]);
        let decomposed = decompose_backward_ops(bwd);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_layer_norm_backward_input() {
        let f = DType::F32;
        let shape = Shape::new(&[2, 4], f);
        let mut g = Graph::new("ln_decomp");
        let x = g.input("x", shape.clone());
        let gamma = g.input("gamma", Shape::new(&[4], f));
        let beta = g.input("beta", Shape::new(&[4], f));
        let y = g.layer_norm(x, gamma, beta, -1, 1e-5, shape);
        let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1], false, Shape::scalar(f));
        g.set_outputs(vec![loss]);
        let prep = prepare_graph_for_ad(g);
        let bwd = grad_with_loss(&prep, &[x]);
        let decomposed = decompose_backward_ops(bwd);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_attention_from_forward_first_grad() {
        use rlx_ir::op::MaskKind;

        let f = DType::F32;
        let mut g = Graph::new("attn_fwd");
        let q = g.input("q", Shape::new(&[1, 2, 3, 2], f));
        let k = g.input("k", Shape::new(&[1, 2, 3, 2], f));
        let v = g.input("v", Shape::new(&[1, 2, 3, 2], f));
        let out = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 2,
                mask_kind: MaskKind::None,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, 2, 3, 2], f),
        );
        let loss = g.reduce(
            out,
            ReduceOp::Sum,
            vec![0, 1, 2, 3],
            false,
            Shape::scalar(f),
        );
        g.set_outputs(vec![loss]);
        let prep = prepare_graph_for_ad(g);
        let bwd = grad_with_loss(&prep, &[q]);
        let decomposed = decompose_backward_for_ad(bwd, 0);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_attention_causal_from_forward_first_grad() {
        use rlx_ir::op::MaskKind;

        let f = DType::F32;
        let mut g = Graph::new("attn_causal_fwd");
        let q = g.input("q", Shape::new(&[1, 2, 3, 2], f));
        let k = g.input("k", Shape::new(&[1, 2, 3, 2], f));
        let v = g.input("v", Shape::new(&[1, 2, 3, 2], f));
        let out = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 2,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, 2, 3, 2], f),
        );
        let loss = g.reduce(
            out,
            ReduceOp::Sum,
            vec![0, 1, 2, 3],
            false,
            Shape::scalar(f),
        );
        g.set_outputs(vec![loss]);
        let prep = prepare_graph_for_ad(g);
        let bwd = grad_with_loss(&prep, &[q]);
        let decomposed = decompose_backward_for_ad(bwd, 0);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_attention_sliding_window_from_forward_first_grad() {
        use rlx_ir::op::MaskKind;

        let f = DType::F32;
        let mut g = Graph::new("attn_sw_fwd");
        let q = g.input("q", Shape::new(&[1, 2, 3, 2], f));
        let k = g.input("k", Shape::new(&[1, 2, 3, 2], f));
        let v = g.input("v", Shape::new(&[1, 2, 3, 2], f));
        let out = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 2,
                mask_kind: MaskKind::SlidingWindow(1),
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, 2, 3, 2], f),
        );
        let loss = g.reduce(
            out,
            ReduceOp::Sum,
            vec![0, 1, 2, 3],
            false,
            Shape::scalar(f),
        );
        g.set_outputs(vec![loss]);
        let prep = prepare_graph_for_ad(g);
        let bwd = grad_with_loss(&prep, &[q]);
        let decomposed = decompose_backward_for_ad(bwd, 0);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_attention_backward_merges_subgraph() {
        use rlx_ir::op::{AttentionBwdWrt, MaskKind};

        let f = DType::F32;
        let shape = Shape::new(&[1, 2, 3, 2], f);
        let mut g = Graph::new("attn_bwd_node");
        let q = g.input("q", shape.clone());
        let k = g.input("k", shape.clone());
        let v = g.input("v", shape.clone());
        let dy = g.input("dy", shape.clone());
        let dq = g.attention_backward(
            AttentionBwdWrt::Query,
            q,
            k,
            v,
            dy,
            2,
            2,
            MaskKind::None,
            None,
        );
        g.set_outputs(vec![dq]);
        let once = decompose_backward_ops(g);
        assert!(once.nodes().len() > 4, "expected merged subgraph expansion");
        assert_input_backward_decomposed(&once);
    }

    #[test]
    fn decompose_cumsum_backward() {
        let f = DType::F32;
        let shape = Shape::new(&[2, 4], f);
        let mut g = Graph::new("cumsum_decomp");
        let dy = g.input("dy", shape.clone());
        let dx = g.cumsum_backward(dy, shape, -1, false);
        g.set_outputs(vec![dx]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_gather_backward_axis0() {
        let f = DType::F32;
        let mut g = Graph::new("gather_decomp");
        let dy = g.input("dy", Shape::new(&[3], f));
        let indices = g.input("indices", Shape::new(&[3], f));
        let dtable = g.gather_backward(dy, indices, Shape::new(&[4], f), 0);
        g.set_outputs(vec![dtable]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_softmax_cross_entropy_backward() {
        let f = DType::F32;
        let mut g = Graph::new("sce_decomp");
        let logits = g.input("logits", Shape::new(&[2, 4], f));
        let labels = g.input("labels", Shape::new(&[2], f));
        let d_loss = g.input("d_loss", Shape::new(&[2], f));
        let dlogits = g.softmax_cross_entropy_backward(logits, labels, d_loss);
        g.set_outputs(vec![dlogits]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_fake_quantize_backward() {
        use rlx_ir::op::SteKind;

        let f = DType::F32;
        let shape = Shape::new(&[8], f);
        let mut g = Graph::new("fq_decomp");
        let x = g.input("x", shape.clone());
        let dy = g.input("dy", shape.clone());
        let dx = g.add_node(
            Op::FakeQuantizeBackward {
                bits: 8,
                axis: None,
                ste: SteKind::ClippedIdentity,
            },
            vec![x, dy],
            shape,
        );
        g.set_outputs(vec![dx]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_conv2d_backward_input_dynamic() {
        use rlx_ir::shape::Dim;

        let f = DType::F32;
        let mut g = Graph::new("dyn_conv_di");
        let dy = g.input(
            "dy",
            Shape::from_dims(
                &[
                    Dim::Dynamic(0),
                    Dim::Static(1),
                    Dim::Static(4),
                    Dim::Static(4),
                ],
                f,
            ),
        );
        let w = g.input("w", Shape::new(&[1, 1, 3, 3], f));
        let dx = g.conv2d_backward_input(
            dy,
            w,
            Shape::from_dims(
                &[
                    Dim::Dynamic(0),
                    Dim::Static(1),
                    Dim::Static(4),
                    Dim::Static(4),
                ],
                f,
            ),
            vec![3, 3],
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
            1,
        );
        g.set_outputs(vec![dx]);
        let decomposed = decompose_backward_ops(g);
        // The input gradient is the conv adjoint = a transposed convolution
        // (was incorrectly a plain `Conv` — the wrong gradient).
        assert!(
            decomposed
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ConvTranspose2d { .. })),
            "expected ConvTranspose2d in dynamic conv di decompose"
        );
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_conv2d_backward_weight_dynamic_im2col() {
        use rlx_ir::shape::Dim;

        let f = DType::F32;
        let mut g = Graph::new("dyn_conv_dw_im2col");
        let x = g.input(
            "x",
            Shape::from_dims(
                &[
                    Dim::Dynamic(0),
                    Dim::Static(1),
                    Dim::Static(4),
                    Dim::Static(4),
                ],
                f,
            ),
        );
        let dy = g.input(
            "dy",
            Shape::from_dims(
                &[
                    Dim::Dynamic(0),
                    Dim::Static(1),
                    Dim::Static(4),
                    Dim::Static(4),
                ],
                f,
            ),
        );
        let dw = g.conv2d_backward_weight(
            x,
            dy,
            Shape::new(&[1, 1, 3, 3], f),
            vec![3, 3],
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
            1,
        );
        g.set_outputs(vec![dw]);
        let decomposed = decompose_backward_ops(g);
        assert!(
            decomposed
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Im2Col { .. })),
            "expected Im2Col in dynamic conv dw decompose"
        );
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_conv2d_backward_weight_after_bind() {
        use rlx_ir::DimBinding;
        use rlx_ir::dynamic::bind_graph;
        use rlx_ir::dynamic::sym;
        use rlx_ir::shape::Dim;

        let f = DType::F32;
        let mut g = Graph::new("dyn_conv_dw");
        let x = g.input(
            "x",
            Shape::from_dims(
                &[
                    Dim::Dynamic(0),
                    Dim::Static(1),
                    Dim::Static(4),
                    Dim::Static(4),
                ],
                f,
            ),
        );
        let dy = g.input(
            "dy",
            Shape::from_dims(
                &[
                    Dim::Dynamic(0),
                    Dim::Static(1),
                    Dim::Static(4),
                    Dim::Static(4),
                ],
                f,
            ),
        );
        let dw = g.conv2d_backward_weight(
            x,
            dy,
            Shape::new(&[1, 1, 3, 3], f),
            vec![3, 3],
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
            1,
        );
        g.set_outputs(vec![dw]);
        let decomposed = decompose_backward_ops(g);
        assert!(
            decomposed
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Im2Col { .. })),
            "decompose before bind should emit Im2Col"
        );
        let bound = bind_graph(
            &decomposed,
            &DimBinding::from_pairs(&[(sym::BATCH, 1), (sym::ROWS, 4 * 4)]),
        );
        assert_input_backward_decomposed(&bound);
    }

    #[test]
    fn decompose_scan_backward_length_129() {
        use rlx_ir::op::BinaryOp;

        let n = 2usize;
        let length = 129u32;
        let carry = Shape::new(&[n], DType::F32);
        let xs_shape = Shape::new(&[length as usize, n], DType::F32);

        let mut body = Graph::new("scan_body_129");
        let bc = body.input("carry", carry.clone());
        let bx = body.input("x_t", carry.clone());
        let by = body.binary(BinaryOp::Add, bc, bx, carry.clone());
        body.set_outputs(vec![by]);

        let mut g = Graph::new("scan_ho_129");
        let init = g.input("init", carry.clone());
        let xs = g.input("xs", xs_shape);
        let scan_out = g.add_node(
            Op::Scan {
                body: Box::new(body),
                length,
                save_trajectory: true,
                num_bcast: 0,
                num_xs: 1,
                num_checkpoints: 0,
            },
            vec![init, xs],
            Shape::new(&[length as usize, n], DType::F32),
        );
        let loss = g.reduce(
            scan_out,
            ReduceOp::Sum,
            vec![0, 1],
            false,
            Shape::scalar(DType::F32),
        );
        g.set_outputs(vec![loss]);
        let prep = prepare_graph_for_ad(g);
        let bwd = grad_with_loss(&prep, &[xs]);
        let decomposed = decompose_backward_for_ad(bwd, 0);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_scan_backward_griewank() {
        use rlx_ir::op::BinaryOp;

        let n = 2usize;
        let length = 4u32;
        let k = 2u32;
        let carry = Shape::new(&[n], DType::F64);

        let mut body = Graph::new("griewank_body");
        let bc = body.input("carry", carry.clone());
        let ones_bytes: Vec<u8> = (0..n).flat_map(|_| 1.0_f64.to_le_bytes()).collect();
        let ones = body.add_node(Op::Constant { data: ones_bytes }, vec![], carry.clone());
        let by = body.binary(BinaryOp::Add, bc, ones, carry.clone());
        body.set_outputs(vec![by]);

        let body_vjp = {
            let carry_id = body
                .nodes()
                .iter()
                .find(|node| matches!(node.op, Op::Input { .. }))
                .map(|node| node.id)
                .unwrap();
            crate::autodiff::grad(&body, &[carry_id])
        };

        let mut g = Graph::new("griewank_bwd");
        let init = g.input("init", carry.clone());
        let trajectory = g.input("trajectory", Shape::new(&[k as usize, n], DType::F64));
        let upstream = g.input("upstream", Shape::new(&[length as usize, n], DType::F64));
        let dinit = g.add_node(
            Op::ScanBackward {
                body_vjp: Box::new(body_vjp),
                length,
                save_trajectory: true,
                num_checkpoints: k,
                num_xs: 0,
                forward_body: Some(Box::new(body)),
            },
            vec![init, trajectory, upstream],
            carry.clone(),
        );
        g.set_outputs(vec![dinit]);
        let decomposed = decompose_backward_ops(g);
        assert_input_backward_decomposed(&decomposed);
    }

    #[test]
    fn decompose_scan_backward_from_forward_second_grad() {
        use rlx_ir::op::BinaryOp;

        let n = 2usize;
        let length = 3u32;
        let carry = Shape::new(&[n], DType::F32);
        let xs_shape = Shape::new(&[length as usize, n], DType::F32);

        let mut body = Graph::new("scan_body_decomp");
        let bc = body.input("carry", carry.clone());
        let bx = body.input("x_t", carry.clone());
        let by = body.binary(BinaryOp::Add, bc, bx, carry.clone());
        body.set_outputs(vec![by]);

        let mut g = Graph::new("scan_ho_decomp");
        let init = g.input("init", carry.clone());
        let xs = g.input("xs", xs_shape);
        let scan_out = g.add_node(
            Op::Scan {
                body: Box::new(body),
                length,
                save_trajectory: true,
                num_bcast: 0,
                num_xs: 1,
                num_checkpoints: 0,
            },
            vec![init, xs],
            Shape::new(&[length as usize, n], DType::F32),
        );
        let loss = g.reduce(
            scan_out,
            ReduceOp::Sum,
            vec![0, 1],
            false,
            Shape::scalar(DType::F32),
        );
        g.set_outputs(vec![loss]);
        let prep = prepare_graph_for_ad(g);
        let bwd = grad_with_loss(&prep, &[xs]);
        let decomposed = decompose_backward_for_ad(bwd, 0);
        assert_input_backward_decomposed(&decomposed);
    }
}
