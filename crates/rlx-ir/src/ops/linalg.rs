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

//! Linear-algebra builders: matmul, LoRA, dequant, fused
//! matmul+bias+activation (plan #53).

use crate::op::Activation;
use crate::quant::QuantScheme;
use crate::{Graph, NodeId, Op, Shape};

impl Graph {
    /// Matrix multiply.
    pub fn matmul(&mut self, lhs: NodeId, rhs: NodeId, out_shape: Shape) -> NodeId {
        self.push(Op::MatMul, vec![lhs, rhs], out_shape, None)
    }

    /// Dense linear solve `x = A⁻¹·b`. `A` must be `[N, N]`; `b` is
    /// `[N]` for a single right-hand side or `[N, K]` for multiple.
    /// `out_shape` matches `b`'s shape.
    pub fn dense_solve(&mut self, a: NodeId, b: NodeId, out_shape: Shape) -> NodeId {
        self.push(Op::DenseSolve, vec![a, b], out_shape, None)
    }

    /// Batched dense linear solve. `A` is `[B, N, N]`; `b` is
    /// `[B, N]` (single-RHS) or `[B, N, K]` (multi-RHS). Per-batch
    /// independent — each slice solved as a separate `dense_solve`.
    /// Typically constructed by `vmap` of `dense_solve`.
    pub fn batched_dense_solve(&mut self, a: NodeId, b: NodeId, out_shape: Shape) -> NodeId {
        self.push(Op::BatchedDenseSolve, vec![a, b], out_shape, None)
    }

    /// Fused LoRA matmul: out = x·W + scale * (x·A)·B.
    /// Inputs: x [m, k], w [k, n], a [k, r], b [r, n]. r is the
    /// LoRA rank; scale is the alpha/rank coefficient.
    pub fn lora_matmul(
        &mut self,
        x: NodeId,
        w: NodeId,
        a: NodeId,
        b: NodeId,
        scale: f32,
        shape: Shape,
    ) -> NodeId {
        self.push(Op::LoraMatMul { scale }, vec![x, w, a, b], shape, None)
    }

    /// Fused dequant + matmul. See [`Op::DequantMatMul`] for per-scheme
    /// input layout (4 inputs for legacy/NVFP4, 2 for GGUF).
    pub fn dequant_matmul(
        &mut self,
        x: NodeId,
        w_q: NodeId,
        scale: NodeId,
        zp: NodeId,
        scheme: QuantScheme,
        shape: Shape,
    ) -> NodeId {
        self.push(
            Op::DequantMatMul { scheme },
            vec![x, w_q, scale, zp],
            shape,
            None,
        )
    }

    /// GGUF / K-quant packed weights — `[x, packed_w_bytes]` only.
    pub fn dequant_matmul_packed(
        &mut self,
        x: NodeId,
        packed_w: NodeId,
        scheme: QuantScheme,
        shape: Shape,
    ) -> NodeId {
        debug_assert!(
            scheme.is_gguf(),
            "dequant_matmul_packed requires a GGUF QuantScheme"
        );
        self.push(Op::DequantMatMul { scheme }, vec![x, packed_w], shape, None)
    }

    /// NVFP4 (E2M1) block matmul — group size 16, FP8 block scales,
    /// optional f32 global scale (defaults to 1.0 when unset at runtime).
    pub fn dequant_matmul_nvfp4(
        &mut self,
        x: NodeId,
        w_q: NodeId,
        block_scales: NodeId,
        global_scale: NodeId,
        shape: Shape,
    ) -> NodeId {
        self.dequant_matmul(
            x,
            w_q,
            block_scales,
            global_scale,
            QuantScheme::Nvfp4Block,
            shape,
        )
    }

    /// Fused matmul + bias + activation (created by optimization passes).
    pub fn fused_matmul_bias_act(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: NodeId,
        activation: Option<Activation>,
        shape: Shape,
    ) -> NodeId {
        self.push(
            Op::FusedMatMulBiasAct { activation },
            vec![input, weight, bias],
            shape,
            None,
        )
    }

    /// Real INT8-arithmetic matmul: i8 inputs, i32 bias, i8 output.
    /// `mult = x_scale · w_scale / out_scale`. Caller's responsible
    /// for asserting the input dtypes — the builder just plumbs the
    /// shape with `dtype = I8` since that's what the kernel writes.
    pub fn q_matmul(
        &mut self,
        x: NodeId,
        w: NodeId,
        bias: NodeId,
        x_zp: i32,
        w_zp: i32,
        out_zp: i32,
        mult: f32,
        out_shape: Shape,
    ) -> NodeId {
        debug_assert_eq!(
            out_shape.dtype(),
            crate::DType::I8,
            "q_matmul output dtype must be I8"
        );
        self.push(
            Op::QMatMul {
                x_zp,
                w_zp,
                out_zp,
                mult,
            },
            vec![x, w, bias],
            out_shape,
            None,
        )
    }

    /// Real INT8-arithmetic 2-D convolution. NCHW layout matching
    /// `Op::Conv`. `mult = x_scale · w_scale / out_scale`.
    #[allow(clippy::too_many_arguments)]
    pub fn q_conv2d(
        &mut self,
        x: NodeId,
        w: NodeId,
        bias: NodeId,
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
        x_zp: i32,
        w_zp: i32,
        out_zp: i32,
        mult: f32,
        out_shape: Shape,
    ) -> NodeId {
        debug_assert_eq!(
            out_shape.dtype(),
            crate::DType::I8,
            "q_conv2d output dtype must be I8"
        );
        self.push(
            Op::QConv2d {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                x_zp,
                w_zp,
                out_zp,
                mult,
            },
            vec![x, w, bias],
            out_shape,
            None,
        )
    }
}
