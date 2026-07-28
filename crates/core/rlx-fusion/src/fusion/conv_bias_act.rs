// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `conv_bias_act` — fuse `Conv → bias-add → activation` into one op.

#![allow(unused_imports)]

use crate::graph_rewrite::Rewriter;
use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

use super::*;

/// Activations `FusedConvBiasAct` folds — ONLY `Relu` (plus identity, i.e. no
/// activation node). These are exactly what cuDNN's fused
/// `cudnnConvolutionBiasActivationForward` applies natively, and rig benchmarks
/// ([conv+bias+relu on an NVIDIA GPU]) show that fully-fused cuDNN path is 1.5–2.1×
/// the unfused `conv → bias → relu` for batch-1.
///
/// Sigmoid/Tanh/SiLU/GELU are deliberately NOT folded: cuDNN's fused call
/// rejects them, so they'd fall to the direct-conv + `conv_bias_act_epilogue`
/// path — which the same benchmarks show is SLOWER than the unfused
/// conv + elementwise-region the pipeline already produces (the naive
/// per-element channel-index epilogue loses to the region kernel). Fusing them
/// would be a perf regression, so we leave them for the region path.
pub fn fusible_conv_activation(a: Activation) -> bool {
    matches!(a, Activation::Relu)
}

/// True for the conv shapes cuDNN's fused conv-bias-activation serves well and
/// wins on: 2-D, un-grouped, kernel > 1×1. 1×1 / depthwise / grouped convs fall
/// to the epilogue path, which benchmarks slower than the unfused region — so
/// they are not fused. Mirrors the `cudnn_ok_shape` guard in the CUDA runtime.
fn cudnn_friendly_conv(kernel_size: &[usize], groups: usize) -> bool {
    groups == 1 && kernel_size.len() == 2 && kernel_size[0] > 1 && kernel_size[1] > 1
}

/// Peel `Expand` / `Reshape` wrappers off a conv-bias operand and, if the
/// underlying tensor is the rank-1 `[C_out]` bias vector, return its id.
///
/// The canonical builder emits `bias[C] → Reshape([1,C,1,1]) → Expand([N,C,H,W])
/// → Add`; some builders skip the `Expand` and rely on the `Add`'s implicit
/// middle-broadcast. Both reduce to the same contiguous `[C_out]` vector, which
/// is exactly what cuDNN's `biasDesc` (`[1,C,1,1]`) consumes. Anything else
/// (higher-rank bias, wrong length) returns `None` so no fusion happens.
fn trace_rank1_bias(graph: &Graph, mut id: NodeId, c_out: usize) -> Option<NodeId> {
    // Bounded peel — a bias never nests more than Reshape∘Expand deep.
    let mut peeled = 0;
    for _ in 0..4 {
        let node = graph.node(id);
        match &node.op {
            Op::Expand { .. } | Op::Reshape { .. } => {
                id = node.inputs[0];
                peeled += 1;
            }
            _ => break,
        }
    }
    // Require a Reshape/Expand wrapper: channel bias is always broadcast up
    // from `[C]` via `[1,C,1,1]`. A *bare* rank-1 operand added to an NCHW
    // conv output would be a trailing-dim (spatial) broadcast — a different
    // computation our channel-broadcast decompose must not silently rewrite.
    if peeled == 0 {
        return None;
    }
    let shape = graph.shape(id);
    if shape.rank() == 1 && shape.dim(0).unwrap_static() == c_out {
        Some(id)
    } else {
        None
    }
}

/// Fuses `conv → add(bias) → [Relu]` into a single [`Op::FusedConvBiasAct`],
/// targeting cuDNN's fully-fused `cudnnConvolutionBiasActivationForward`.
///
/// Deliberately narrow — it fires ONLY for the case rig benchmarks show wins:
/// a cuDNN-friendly conv (2-D, un-grouped, kernel > 1×1; see
/// [`cudnn_friendly_conv`]) with a Relu or no-op epilogue (see
/// [`fusible_conv_activation`]). There it collapses conv + bias-add + relu into
/// one cuDNN call (1.5–2.1× the unfused path at batch 1). Other shapes (1×1,
/// depthwise, grouped) and activations (sigmoid/tanh/silu/gelu) are left for the
/// existing conv + elementwise-region path, which benchmarks faster than the
/// epilogue-kernel fallback would be. Every backend that does not claim
/// `FusedConvBiasAct` decomposes it back to primitives in `unfuse`, so emitting
/// the fused op never changes semantics.
pub struct FuseConvBiasAct;

impl Pass for FuseConvBiasAct {
    fn name(&self) -> &str {
        "fuse_conv_bias_act"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut rw = Rewriter::new(&graph.name);
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            // Pattern: Conv → Add(bias) → [Relu]. Only cuDNN-friendly shapes
            // with a Relu / no-op epilogue are fused — that's the fully-fused
            // cuDNN path that benchmarks faster; other shapes/activations stay
            // on the (faster) conv + elementwise-region path.
            if let Op::Conv {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } = &node.op
                && cudnn_friendly_conv(kernel_size, *groups)
            {
                // conv weight is [C_out, C_in/groups, kH, kW]; C_out = dim 0.
                let c_out = graph.shape(node.inputs[1]).dim(0).unwrap_static();
                let conv_id = node.id;
                let conv_users = graph.users(conv_id);

                if conv_users.len() == 1 {
                    let add_node = graph.node(conv_users[0]);
                    if let Op::Binary(BinaryOp::Add) = &add_node.op {
                        // The non-conv operand carries the bias.
                        let bias_operand = if add_node.inputs[0] == conv_id {
                            add_node.inputs[1]
                        } else {
                            add_node.inputs[0]
                        };

                        if let Some(bias_id) = trace_rank1_bias(&graph, bias_operand, c_out) {
                            let add_id = add_node.id;
                            let add_users = graph.users(add_id);

                            // Epilogue activation. A single Relu consumer folds
                            // in; a single NON-fusible activation (sigmoid/tanh/
                            // silu/gelu) makes us BAIL entirely — fusing bias
                            // only and leaving that activation separate is not a
                            // win, so the whole chain stays on the region path.
                            let mut activation = None;
                            let mut act_id = None;
                            if add_users.len() == 1 {
                                let act_node = graph.node(add_users[0]);
                                if let Op::Activation(a) = &act_node.op {
                                    if fusible_conv_activation(*a) {
                                        activation = Some(*a);
                                        act_id = Some(act_node.id);
                                    } else {
                                        rw.copy_node(node);
                                        continue;
                                    }
                                }
                            }

                            let out_shape = if let Some(aid) = act_id {
                                graph.shape(aid).clone()
                            } else {
                                add_node.shape.clone()
                            };

                            rw.ensure_mapped(&graph, &[node.inputs[0], node.inputs[1], bias_id]);
                            let fused_id = rw.add_fused(
                                Op::FusedConvBiasAct {
                                    kernel_size: kernel_size.clone(),
                                    stride: stride.clone(),
                                    padding: padding.clone(),
                                    dilation: dilation.clone(),
                                    groups: *groups,
                                    activation,
                                    has_residual: false,
                                },
                                &[node.inputs[0], node.inputs[1], bias_id],
                                out_shape,
                            );

                            rw.replace(conv_id, fused_id);
                            rw.replace(add_id, fused_id);
                            fused_away.insert(add_id, ());
                            if let Some(aid) = act_id {
                                rw.replace(aid, fused_id);
                                fused_away.insert(aid, ());
                            }
                            continue;
                        }
                    }
                }
            }

            // No fusion — copy as-is.
            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

/// The operand of a binary `node` that is NOT `this_id`.
fn other_operand(node: &rlx_ir::Node, this_id: NodeId) -> NodeId {
    if node.inputs[0] == this_id {
        node.inputs[1]
    } else {
        node.inputs[0]
    }
}

/// Fuses a host-pre-folded BatchNorm affine into the conv:
/// `Conv(no bias) → Mul(per-channel scale) → Add(per-channel shift) → Relu`
/// → `FusedConvBiasAct(x, w·scale, shift, Relu)`.
///
/// Frozen-BN CNNs fold BatchNorm into a per-channel affine on the host at build
/// time — the graph is `conv(no bias) → Mul(scale[1,C,1,1]) → Add(shift[1,C,1,1])`
/// (e.g. funasr CAM++ `batchnorm2d`), so the narrower [`FuseConvBiasAct`] never
/// sees a plain conv-bias. Since `scale` is per-OUTPUT-channel,
/// `conv(x,w)·scale == conv(x, w·scale)`, so we fold it into the weights (a
/// small weight-sized `Mul`, cheaper than the activation-sized `Mul` it replaces
/// — a win on every backend) and reuse the validated `FusedConvBiasAct` path
/// with `shift` as the bias. Same cuDNN-friendly-shape + Relu guard as
/// [`FuseConvBiasAct`]; the weight fold reorders float ops, so it is a
/// near-exact (not bit-exact) rewrite.
pub struct FuseConvAffineAct;

/// A matched `Conv → Mul(scale) → Add(shift) → [Add(residual)] → Relu` chain.
struct AffineMatch {
    scale_id: NodeId,
    shift_id: NodeId,
    /// Full `[N,C,H,W]` residual (cuDNN `z`), when the chain has a residual add.
    residual_id: Option<NodeId>,
    out_shape: Shape,
    /// Intermediate nodes to fuse away: Mul, shift-Add, [residual-Add], Relu.
    fuse_ids: Vec<NodeId>,
}

impl Pass for FuseConvAffineAct {
    fn name(&self) -> &str {
        "fuse_conv_affine_act"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut rw = Rewriter::new(&graph.name);
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            if let Op::Conv {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } = &node.op
                && cudnn_friendly_conv(kernel_size, *groups)
            {
                let w_id = node.inputs[1];
                let c_out = graph.shape(w_id).dim(0).unwrap_static();
                let conv_id = node.id;

                if let Some(m) = match_conv_affine(&graph, conv_id, c_out) {
                    let x_id = node.inputs[0];
                    let w_shape = graph.shape(w_id).clone();
                    let dtype = w_shape.dtype();

                    let mut to_map = vec![x_id, w_id, m.scale_id, m.shift_id];
                    to_map.extend(m.residual_id);
                    rw.ensure_mapped(&graph, &to_map);
                    let x_new = rw.map(x_id);
                    let w_new = rw.map(w_id);
                    let scale_new = rw.map(m.scale_id);
                    let shift_new = rw.map(m.shift_id);

                    // Fold per-output-channel scale into the weights:
                    // scale [C] → [C,1,1,1], then w' = w · scale (broadcast).
                    let scale_r = rw.new_graph.add_node(
                        Op::Reshape {
                            new_shape: vec![c_out as i64, 1, 1, 1],
                        },
                        vec![scale_new],
                        Shape::new(&[c_out, 1, 1, 1], dtype),
                    );
                    let w_scaled = rw.new_graph.add_node(
                        Op::Binary(BinaryOp::Mul),
                        vec![w_new, scale_r],
                        w_shape,
                    );

                    let mut inputs = vec![x_new, w_scaled, shift_new];
                    inputs.extend(m.residual_id.map(|r| rw.map(r)));
                    let fused = rw.new_graph.add_node(
                        Op::FusedConvBiasAct {
                            kernel_size: kernel_size.clone(),
                            stride: stride.clone(),
                            padding: padding.clone(),
                            dilation: dilation.clone(),
                            groups: *groups,
                            activation: Some(Activation::Relu),
                            has_residual: m.residual_id.is_some(),
                        },
                        inputs,
                        m.out_shape,
                    );

                    rw.replace(conv_id, fused);
                    for id in &m.fuse_ids {
                        rw.replace(*id, fused);
                        fused_away.insert(*id, ());
                    }
                    continue;
                }
            }

            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

/// Match `Conv → Mul(scale) → Add(shift) → [Add(residual)] → Relu` — every link
/// single-use, `scale`/`shift` rank-1 `[C_out]` (traced through Reshape). The
/// optional residual is a full-shape `[N,C,H,W]` tensor (distinguished from a
/// per-channel bias by requiring an exact shape match, not a broadcast).
fn match_conv_affine(graph: &Graph, conv_id: NodeId, c_out: usize) -> Option<AffineMatch> {
    let conv_users = graph.users(conv_id);
    if conv_users.len() != 1 {
        return None;
    }
    let mul = graph.node(conv_users[0]);
    let Op::Binary(BinaryOp::Mul) = &mul.op else {
        return None;
    };
    let scale_id = trace_rank1_bias(graph, other_operand(mul, conv_id), c_out)?;

    let mul_users = graph.users(mul.id);
    if mul_users.len() != 1 {
        return None;
    }
    let shift_add = graph.node(mul_users[0]);
    let Op::Binary(BinaryOp::Add) = &shift_add.op else {
        return None;
    };
    let shift_id = trace_rank1_bias(graph, other_operand(shift_add, mul.id), c_out)?;

    let shift_users = graph.users(shift_add.id);
    if shift_users.len() != 1 {
        return None;
    }
    let after = graph.node(shift_users[0]);

    let mut fuse_ids = vec![mul.id, shift_add.id];
    let mut residual_id = None;

    // Either a terminal Relu (no residual) or an `Add(residual) → Relu`.
    let relu = if matches!(after.op, Op::Activation(Activation::Relu)) {
        after
    } else if let Op::Binary(BinaryOp::Add) = &after.op {
        let resid = other_operand(after, shift_add.id);
        // Residual is the full block tensor — exact shape match, not a broadcast.
        if graph.shape(resid).dims() != shift_add.shape.dims() {
            return None;
        }
        let res_users = graph.users(after.id);
        if res_users.len() != 1 {
            return None;
        }
        let relu = graph.node(res_users[0]);
        if !matches!(relu.op, Op::Activation(Activation::Relu)) {
            return None;
        }
        residual_id = Some(resid);
        fuse_ids.push(after.id);
        relu
    } else {
        return None;
    };
    fuse_ids.push(relu.id);

    Some(AffineMatch {
        scale_id,
        shift_id,
        residual_id,
        out_shape: relu.shape.clone(),
        fuse_ids,
    })
}
