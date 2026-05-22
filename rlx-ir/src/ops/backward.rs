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

//! Backward / training op builders.
//!
//! These nodes are emitted by `rlx-opt::autodiff` when it walks a
//! forward graph in reverse and needs a closed-form gradient kernel
//! (rather than composing one from primitives). Output shapes follow
//! directly from the forward shapes: `relu_backward` and
//! `maxpool2d_backward` match the original input; conv backward shapes
//! match the original input / weight; cross-entropy returns one loss
//! per row of logits.
//!
//! Shape checks here are debug-only; the verifier in `verify.rs` does
//! the rigorous version.

use crate::op::{AttentionBwdWrt, MaskKind};
use crate::{DType, Graph, NodeId, Op, Shape};

impl Graph {
    /// ReLU backward: `dx = dy where x > 0 else 0`. Output shape matches `x`.
    pub fn relu_backward(&mut self, x: NodeId, dy: NodeId) -> NodeId {
        let x_shape = self.shape(x).clone();
        debug_assert_eq!(
            self.shape(x),
            self.shape(dy),
            "relu_backward: x and dy must have identical shapes"
        );
        self.push(Op::ReluBackward, vec![x, dy], x_shape, None)
    }

    /// Element-wise activation backward — closed-form derivative of
    /// any single-input activation other than ReLU. See
    /// `Op::ActivationBackward` for the per-kind formulae.
    pub fn activation_backward(
        &mut self,
        kind: crate::op::Activation,
        x: NodeId,
        dy: NodeId,
    ) -> NodeId {
        let x_shape = self.shape(x).clone();
        debug_assert_eq!(
            self.shape(x),
            self.shape(dy),
            "activation_backward: x and dy must have identical shapes"
        );
        self.push(Op::ActivationBackward { kind }, vec![x, dy], x_shape, None)
    }

    /// LayerNorm backward w.r.t. the input. Inputs `[x, gamma, dy]`.
    /// Output shape matches `x`. Currently axis = -1 only.
    pub fn layer_norm_backward_input(
        &mut self,
        x: NodeId,
        gamma: NodeId,
        dy: NodeId,
        axis: i32,
        eps: f32,
    ) -> NodeId {
        let x_shape = self.shape(x).clone();
        debug_assert_eq!(
            self.shape(x),
            self.shape(dy),
            "layer_norm_backward_input: x and dy must match"
        );
        self.push(
            Op::LayerNormBackwardInput { axis, eps },
            vec![x, gamma, dy],
            x_shape,
            None,
        )
    }

    /// RMSNorm backward w.r.t. input. Inputs `[x, gamma, beta, dy]`.
    pub fn rms_norm_backward_input(
        &mut self,
        x: NodeId,
        gamma: NodeId,
        beta: NodeId,
        dy: NodeId,
        axis: i32,
        eps: f32,
    ) -> NodeId {
        let x_shape = self.shape(x).clone();
        self.push(
            Op::RmsNormBackwardInput { axis, eps },
            vec![x, gamma, beta, dy],
            x_shape,
            None,
        )
    }

    pub fn rms_norm_backward_gamma(
        &mut self,
        x: NodeId,
        gamma: NodeId,
        beta: NodeId,
        dy: NodeId,
        axis: i32,
        eps: f32,
    ) -> NodeId {
        self.push(
            Op::RmsNormBackwardGamma { axis, eps },
            vec![x, gamma, beta, dy],
            self.shape(gamma).clone(),
            None,
        )
    }

    pub fn rms_norm_backward_beta(
        &mut self,
        x: NodeId,
        gamma: NodeId,
        beta: NodeId,
        dy: NodeId,
        axis: i32,
        eps: f32,
    ) -> NodeId {
        self.push(
            Op::RmsNormBackwardBeta { axis, eps },
            vec![x, gamma, beta, dy],
            self.shape(beta).clone(),
            None,
        )
    }

    pub fn rope_backward(
        &mut self,
        dy: NodeId,
        cos: NodeId,
        sin: NodeId,
        head_dim: usize,
        n_rot: usize,
    ) -> NodeId {
        let out_shape = self.shape(dy).clone();
        self.push(
            Op::RopeBackward { head_dim, n_rot },
            vec![dy, cos, sin],
            out_shape,
            None,
        )
    }

    pub fn cumsum_backward(
        &mut self,
        dy: NodeId,
        out_shape: Shape,
        axis: i32,
        exclusive: bool,
    ) -> NodeId {
        self.push(
            Op::CumsumBackward { axis, exclusive },
            vec![dy],
            out_shape,
            None,
        )
    }

    pub fn gather_backward(
        &mut self,
        dy: NodeId,
        indices: NodeId,
        table_shape: Shape,
        axis: i32,
    ) -> NodeId {
        self.push(
            Op::GatherBackward { axis },
            vec![dy, indices],
            table_shape,
            None,
        )
    }

    /// GroupNorm (NCHW) backward w.r.t. input. Inputs `[x, gamma, beta, dy]`.
    pub fn group_norm_backward_input(
        &mut self,
        x: NodeId,
        gamma: NodeId,
        beta: NodeId,
        dy: NodeId,
        num_groups: usize,
        eps: f32,
    ) -> NodeId {
        let x_shape = self.shape(x).clone();
        self.push(
            Op::GroupNormBackwardInput { num_groups, eps },
            vec![x, gamma, beta, dy],
            x_shape,
            None,
        )
    }

    /// GroupNorm backward w.r.t. gamma. Inputs `[x, dy]`.
    pub fn group_norm_backward_gamma(
        &mut self,
        x: NodeId,
        dy: NodeId,
        gamma_shape: Shape,
        num_groups: usize,
        eps: f32,
    ) -> NodeId {
        self.push(
            Op::GroupNormBackwardGamma { num_groups, eps },
            vec![x, dy],
            gamma_shape,
            None,
        )
    }

    /// GroupNorm backward w.r.t. beta. Inputs `[x, dy]`.
    pub fn group_norm_backward_beta(
        &mut self,
        x: NodeId,
        dy: NodeId,
        beta_shape: Shape,
        num_groups: usize,
        eps: f32,
    ) -> NodeId {
        self.push(
            Op::GroupNormBackwardBeta { num_groups, eps },
            vec![x, dy],
            beta_shape,
            None,
        )
    }

    /// LayerNorm backward w.r.t. gamma. Inputs `[x, dy]`. Output shape
    /// is provided by the caller — typically the gamma's shape, e.g.
    /// `[D]` for a per-feature 1-D gamma.
    pub fn layer_norm_backward_gamma(
        &mut self,
        x: NodeId,
        dy: NodeId,
        gamma_shape: Shape,
        axis: i32,
        eps: f32,
    ) -> NodeId {
        debug_assert_eq!(
            self.shape(x),
            self.shape(dy),
            "layer_norm_backward_gamma: x and dy must match"
        );
        self.push(
            Op::LayerNormBackwardGamma { axis, eps },
            vec![x, dy],
            gamma_shape,
            None,
        )
    }

    /// 2D max-pool backward. `x` is the original NCHW input; `dy` is
    /// the upstream gradient with shape matching the pool's output.
    /// Output shape matches `x`.
    pub fn maxpool2d_backward(
        &mut self,
        x: NodeId,
        dy: NodeId,
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
    ) -> NodeId {
        let x_shape = self.shape(x).clone();
        debug_assert_eq!(kernel_size.len(), 2, "maxpool2d_backward: 2-D only");
        debug_assert_eq!(stride.len(), 2);
        debug_assert_eq!(padding.len(), 2);
        self.push(
            Op::MaxPool2dBackward {
                kernel_size,
                stride,
                padding,
            },
            vec![x, dy],
            x_shape,
            None,
        )
    }

    /// Conv2D backward w.r.t. input. `dy` has the conv output shape;
    /// `w` is the forward weight `[C_out, C_in/groups, kH, kW]`. The
    /// output shape (the original input shape) is supplied by the
    /// caller because it can't be unambiguously derived from `dy.shape`
    /// alone in the presence of strides + padding.
    pub fn conv2d_backward_input(
        &mut self,
        dy: NodeId,
        w: NodeId,
        x_shape: Shape,
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
    ) -> NodeId {
        debug_assert_eq!(kernel_size.len(), 2);
        debug_assert_eq!(stride.len(), 2);
        debug_assert_eq!(padding.len(), 2);
        debug_assert_eq!(dilation.len(), 2);
        self.push(
            Op::Conv2dBackwardInput {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            },
            vec![dy, w],
            x_shape,
            None,
        )
    }

    /// Conv2D backward w.r.t. weight. Output shape matches the forward
    /// weight `[C_out, C_in/groups, kH, kW]`.
    pub fn conv2d_backward_weight(
        &mut self,
        x: NodeId,
        dy: NodeId,
        w_shape: Shape,
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
    ) -> NodeId {
        debug_assert_eq!(kernel_size.len(), 2);
        debug_assert_eq!(stride.len(), 2);
        debug_assert_eq!(padding.len(), 2);
        debug_assert_eq!(dilation.len(), 2);
        self.push(
            Op::Conv2dBackwardWeight {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            },
            vec![x, dy],
            w_shape,
            None,
        )
    }

    /// Fused softmax + cross-entropy with f32-encoded integer labels.
    /// `logits [N, C]`, `labels [N]` → `[N]` per-row loss.
    pub fn softmax_cross_entropy_with_logits(&mut self, logits: NodeId, labels: NodeId) -> NodeId {
        let logits_shape = self.shape(logits);
        debug_assert_eq!(
            logits_shape.rank(),
            2,
            "sce_with_logits: logits must be 2-D [N, C]"
        );
        let n = logits_shape.dim(0);
        let dtype = logits_shape.dtype();
        let out_shape = Shape::from_dims(&[n], dtype);
        self.push(
            Op::SoftmaxCrossEntropyWithLogits,
            vec![logits, labels],
            out_shape,
            None,
        )
    }

    /// Backward of `softmax_cross_entropy_with_logits`.
    /// `[logits, labels, d_loss]` → `dlogits` shaped like `logits`.
    pub fn softmax_cross_entropy_backward(
        &mut self,
        logits: NodeId,
        labels: NodeId,
        d_loss: NodeId,
    ) -> NodeId {
        let logits_shape = self.shape(logits).clone();
        debug_assert_eq!(
            logits_shape.rank(),
            2,
            "sce_backward: logits must be 2-D [N, C]"
        );
        self.push(
            Op::SoftmaxCrossEntropyBackward,
            vec![logits, labels, d_loss],
            logits_shape,
            None,
        )
    }

    /// Element-wise complex squared-magnitude: `|z|² = re² + im²`.
    /// Input must be `DType::C64`; output is same logical shape but
    /// `DType::F32`. The canonical real-valued loss surface for
    /// Wirtinger reverse-mode AD on complex graphs.
    pub fn complex_norm_sq(&mut self, z: NodeId) -> NodeId {
        let z_shape = self.shape(z).clone();
        debug_assert_eq!(
            z_shape.dtype(),
            DType::C64,
            "complex_norm_sq: input must be C64, got {:?}",
            z_shape.dtype()
        );
        let out_shape = Shape::from_dims(z_shape.dims(), DType::F32);
        self.push(Op::ComplexNormSq, vec![z], out_shape, None)
    }

    /// Scaled dot-product attention backward w.r.t. `q`, `k`, or `v`.
    /// See [`Op::AttentionBackward`]. When `mask_kind` is [`MaskKind::Custom`]
    /// or [`MaskKind::Bias`], pass the same mask tensor used in forward.
    pub fn attention_backward(
        &mut self,
        wrt: AttentionBwdWrt,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        dy: NodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        mask: Option<NodeId>,
    ) -> NodeId {
        let out_shape = match wrt {
            AttentionBwdWrt::Query => self.shape(q).clone(),
            AttentionBwdWrt::Key => self.shape(k).clone(),
            AttentionBwdWrt::Value => self.shape(v).clone(),
        };
        let mut inputs = vec![q, k, v, dy];
        if matches!(mask_kind, MaskKind::Custom | MaskKind::Bias) {
            inputs.push(mask.expect("attention_backward: mask required for Custom/Bias"));
        }
        self.push(
            Op::AttentionBackward {
                num_heads,
                head_dim,
                mask_kind,
                wrt,
            },
            inputs,
            out_shape,
            None,
        )
    }

    /// Emit `dQ`, `dK`, and `dV` for one [`Op::Attention`] forward node.
    pub fn attention_backward_all(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        dy: NodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        mask: Option<NodeId>,
    ) -> (NodeId, NodeId, NodeId) {
        let dq = self.attention_backward(
            AttentionBwdWrt::Query,
            q,
            k,
            v,
            dy,
            num_heads,
            head_dim,
            mask_kind,
            mask,
        );
        let dk = self.attention_backward(
            AttentionBwdWrt::Key,
            q,
            k,
            v,
            dy,
            num_heads,
            head_dim,
            mask_kind,
            mask,
        );
        let dv = self.attention_backward(
            AttentionBwdWrt::Value,
            q,
            k,
            v,
            dy,
            num_heads,
            head_dim,
            mask_kind,
            mask,
        );
        (dq, dk, dv)
    }

    /// Wirtinger backward for [`complex_norm_sq`]: given upstream `g`
    /// (real, same shape as the forward output) and the original
    /// complex input `z`, returns `dz = g · z` as C64.
    pub fn complex_norm_sq_backward(&mut self, z: NodeId, g: NodeId) -> NodeId {
        let z_shape = self.shape(z).clone();
        debug_assert_eq!(z_shape.dtype(), DType::C64);
        debug_assert_eq!(self.shape(g).dtype(), DType::F32);
        debug_assert_eq!(
            z_shape.dims(),
            self.shape(g).dims(),
            "complex_norm_sq_backward: z and g must share logical shape"
        );
        self.push(Op::ComplexNormSqBackward, vec![z, g], z_shape, None)
    }

    /// Element-wise complex conjugate: `z̄ = re - i·im`. Input must be
    /// `DType::C64`; output is the same shape and dtype. Used by
    /// Wirtinger VJP rules on C64 binary ops.
    pub fn conjugate(&mut self, z: NodeId) -> NodeId {
        let z_shape = self.shape(z).clone();
        debug_assert_eq!(
            z_shape.dtype(),
            DType::C64,
            "conjugate: input must be C64, got {:?}",
            z_shape.dtype()
        );
        self.push(Op::Conjugate, vec![z], z_shape, None)
    }
}
