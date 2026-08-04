// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Linear-algebra builders: matmul, LoRA, dequant, fused
//! matmul+bias+activation (plan #53).

use crate::op::{Activation, SynthKind};
use crate::quant::{QuantScheme, ScaleLayout, ScaledFormat};
use crate::{DType, Graph, NodeId, Op, Shape};

impl Graph {
    /// Matrix multiply.
    pub fn matmul(&mut self, lhs: NodeId, rhs: NodeId, out_shape: Shape) -> NodeId {
        self.push(Op::MatMul, vec![lhs, rhs], out_shape, None)
    }

    /// Dynamically quantize `x` (logical `[rows, cols]`, blocks along the last
    /// axis) to low-precision `fmt` codes plus a scale tensor, per `layout`.
    /// Returns `(codes, scale)`: `codes` is `DType::U8` with `x`'s shape;
    /// `scale`'s shape/dtype follow the layout (`[1]` f32 for per-tensor,
    /// `[rows, cols/block]` u8 for block layouts). The building block of
    /// [`scaled_matmul`](Self::scaled_matmul); `fmt` may be any
    /// [`ScaledFormat`], including a parameterized [`ScaledFormat::Custom`].
    pub fn scaled_quantize(
        &mut self,
        x: NodeId,
        fmt: ScaledFormat,
        layout: ScaleLayout,
    ) -> (NodeId, NodeId) {
        let xs = self.node(x).shape.clone();
        let cols = xs.dim(xs.rank() - 1).unwrap_static();
        let rows = xs.num_elements().unwrap() / cols.max(1);
        let scale_shape = match layout {
            ScaleLayout::PerTensor => Shape::new(&[1], layout.scale_dtype()),
            _ => Shape::new(
                &[rows, cols.div_ceil(layout.block() as usize)],
                layout.scale_dtype(),
            ),
        };
        let scale = self.push(
            Op::ScaledQuantScale {
                format: fmt,
                scale_layout: layout,
            },
            vec![x],
            scale_shape,
            None,
        );
        let codes = self.push(
            Op::ScaledQuantize {
                format: fmt,
                scale_layout: layout,
            },
            vec![x, scale],
            xs.with_dtype(DType::U8),
            None,
        );
        (codes, scale)
    }

    /// Reconstruct f32 from packed `codes` + `scale` — the inverse of
    /// [`scaled_quantize`](Self::scaled_quantize).
    pub fn scaled_dequantize(
        &mut self,
        codes: NodeId,
        scale: NodeId,
        fmt: ScaledFormat,
        layout: ScaleLayout,
    ) -> NodeId {
        let shape = self.node(codes).shape.clone().with_dtype(DType::F32);
        self.push(
            Op::ScaledDequantize {
                format: fmt,
                scale_layout: layout,
            },
            vec![codes, scale],
            shape,
            None,
        )
    }

    /// Native low-precision GEMM (TN: `lhs [m,k] · rhs [n,k]ᵀ → [m,n]` f32).
    /// Both operands are dynamically quantized to `fmt`/`layout` and fed
    /// straight into the scaled matmul with f32 accumulation — no hand-wiring of
    /// [`Op::ScaledQuantScale`]/[`Op::ScaledQuantize`]. `rhs` must already be
    /// K-last (`[n, k]`); transpose a `[k, n]` weight first. `fmt` may be any
    /// [`ScaledFormat`], including a parameterized [`ScaledFormat::Custom`]
    /// (e.g. `ScaledFormat::custom(3, 0)` for `f4e3m0`).
    pub fn scaled_matmul(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        fmt: ScaledFormat,
        layout: ScaleLayout,
    ) -> NodeId {
        self.scaled_matmul_bias(lhs, rhs, None, fmt, layout)
    }

    /// [`scaled_matmul`](Self::scaled_matmul) with an optional f32 bias `[n]`
    /// added to each output row.
    pub fn scaled_matmul_bias(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        bias: Option<NodeId>,
        fmt: ScaledFormat,
        layout: ScaleLayout,
    ) -> NodeId {
        let m = self.node(lhs).shape.dim(0).unwrap_static();
        let n = self.node(rhs).shape.dim(0).unwrap_static();
        let (lq, ls) = self.scaled_quantize(lhs, fmt, layout);
        let (rq, rs) = self.scaled_quantize(rhs, fmt, layout);
        let mut inputs = vec![lq, rq, ls, rs];
        if let Some(b) = bias {
            inputs.push(b);
        }
        self.push(
            Op::ScaledMatMul {
                lhs_format: fmt,
                rhs_format: fmt,
                scale_layout: layout,
                has_bias: bias.is_some(),
            },
            inputs,
            Shape::new(&[m, n], DType::F32),
            None,
        )
    }

    /// Native low-precision *grouped* (MoE) GEMM — the expert-indexed
    /// [`scaled_matmul`](Self::scaled_matmul). Dynamically quantizes f32
    /// `input [M,K]` and the per-expert f32 weight stack `weight [E,N,K]`
    /// (K-last) to `fmt`/`layout` codes, then wires
    /// [`Op::ScaledGroupedMatMul`]. `expert_idx [M]` is the f32-encoded
    /// expert id per token. Output `[M,N]` f32, with f32 accumulation.
    pub fn scaled_grouped_matmul(
        &mut self,
        input: NodeId,
        weight: NodeId,
        expert_idx: NodeId,
        fmt: ScaledFormat,
        layout: ScaleLayout,
    ) -> NodeId {
        self.scaled_grouped_matmul_bias(input, weight, expert_idx, None, fmt, layout)
    }

    /// [`scaled_grouped_matmul`](Self::scaled_grouped_matmul) with an optional
    /// per-expert f32 bias `[E, N]` added to each routed output row.
    pub fn scaled_grouped_matmul_bias(
        &mut self,
        input: NodeId,
        weight: NodeId,
        expert_idx: NodeId,
        bias: Option<NodeId>,
        fmt: ScaledFormat,
        layout: ScaleLayout,
    ) -> NodeId {
        let m = self.node(input).shape.dim(0).unwrap_static();
        let wshape = self.node(weight).shape.clone();
        // weight is [E, N, K]; N is the output dim, K the contraction axis.
        let e = wshape.dim(0).unwrap_static();
        let n = wshape.dim(wshape.rank() - 2).unwrap_static();
        let k = wshape.dim(wshape.rank() - 1).unwrap_static();
        // Activations [M,K] quantize with the generic (2-D) helper.
        let (iq, is) = self.scaled_quantize(input, fmt, layout);
        // The per-expert weight stack keeps its natural rank: the scale is
        // `[E, N, ⌈K/block⌉]` (or `[1]` per-tensor), matching `ScaledQuantScale`
        // shape inference so the verifier is happy. Byte layout is identical to
        // the flattened `[E·N, ⌈K/block⌉]` the oracle indexes.
        let w_scale_shape = match layout {
            ScaleLayout::PerTensor => Shape::new(&[1], layout.scale_dtype()),
            _ => Shape::new(
                &[e, n, k.div_ceil(layout.block() as usize)],
                layout.scale_dtype(),
            ),
        };
        let ws = self.push(
            Op::ScaledQuantScale {
                format: fmt,
                scale_layout: layout,
            },
            vec![weight],
            w_scale_shape,
            None,
        );
        let wq = self.push(
            Op::ScaledQuantize {
                format: fmt,
                scale_layout: layout,
            },
            vec![weight, ws],
            wshape.with_dtype(DType::U8),
            None,
        );
        let mut inputs = vec![iq, wq, is, ws, expert_idx];
        if let Some(b) = bias {
            inputs.push(b);
        }
        self.push(
            Op::ScaledGroupedMatMul {
                lhs_format: fmt,
                rhs_format: fmt,
                scale_layout: layout,
                has_bias: bias.is_some(),
            },
            inputs,
            Shape::new(&[m, n], DType::F32),
            None,
        )
    }

    /// Dense linear solve `x = A⁻¹·b`. `A` must be `[N, N]`; `b` is
    /// `[N]` for a single right-hand side or `[N, K]` for multiple.
    /// `out_shape` matches `b`'s shape.
    pub fn dense_solve(&mut self, a: NodeId, b: NodeId, out_shape: Shape) -> NodeId {
        self.push(Op::DenseSolve, vec![a, b], out_shape, None)
    }

    /// Cholesky factorization `A = L·Lᵀ`. `a` is `[n, n]` symmetric
    /// positive-definite; the output is the lower-triangular `L` (same shape,
    /// strict upper triangle zeroed).
    pub fn cholesky(&mut self, a: NodeId, out_shape: Shape) -> NodeId {
        self.push(Op::Cholesky, vec![a], out_shape, None)
    }

    /// Triangular solve `op(A)·X = B`. `a` is `[n, n]` triangular (lower/upper
    /// per `lower`), `op(A)` is `Aᵀ` when `transpose`; `b`/output are `[n]` or
    /// `[n, nrhs]`.
    pub fn triangular_solve(
        &mut self,
        a: NodeId,
        b: NodeId,
        lower: bool,
        transpose: bool,
        out_shape: Shape,
    ) -> NodeId {
        self.push(
            Op::TriangularSolve { lower, transpose },
            vec![a, b],
            out_shape,
            None,
        )
    }

    /// Determinant of the square matrix `a` `[n, n]` → scalar (`out_shape` `[]`).
    pub fn det(&mut self, a: NodeId, out_shape: Shape) -> NodeId {
        self.push(Op::Det, vec![a], out_shape, None)
    }

    /// `log|det(a)|` of the square matrix `a` `[n, n]` → scalar (`out_shape` `[]`).
    pub fn logdet(&mut self, a: NodeId, out_shape: Shape) -> NodeId {
        self.push(Op::LogDet, vec![a], out_shape, None)
    }

    /// Sort `x` along `axis` (`descending` = largest-first). Output same shape.
    pub fn sort(&mut self, x: NodeId, axis: usize, descending: bool, out_shape: Shape) -> NodeId {
        self.push(Op::Sort { axis, descending }, vec![x], out_shape, None)
    }

    /// Indices that would sort `x` along `axis` (as f32). Output same shape.
    pub fn argsort(
        &mut self,
        x: NodeId,
        axis: usize,
        descending: bool,
        out_shape: Shape,
    ) -> NodeId {
        self.push(Op::ArgSort { axis, descending }, vec![x], out_shape, None)
    }

    /// One factor of the thin SVD `a = U·diag(S)·Vᵀ` (`a` `[m,n]`,
    /// `k=min(m,n)`). `out_shape`: `U` `[m,k]`, `S` `[k]`, `Vt` `[k,n]`.
    pub fn svd(&mut self, a: NodeId, part: crate::op::SvdPart, out_shape: Shape) -> NodeId {
        self.push(Op::Svd { part }, vec![a], out_shape, None)
    }

    /// One factor of the thin QR `a = Q·R` (`a` `[m,n]`, `k=min(m,n)`).
    /// `out_shape`: `Q` `[m,k]`, `R` `[k,n]`.
    pub fn qr(&mut self, a: NodeId, part: crate::op::QrPart, out_shape: Shape) -> NodeId {
        self.push(Op::Qr { part }, vec![a], out_shape, None)
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

    /// On-chip codebook weight-synthesis matmul. See [`Op::SynthMatMul`].
    /// `indices` is `[n, k/entry_dim]` u8; `codebook` is
    /// `[num_entries, entry_dim]` f32; output is `[m, n]`. The weight is
    /// reconstructed inside the matmul inner loop, never materialized.
    pub fn synth_matmul(
        &mut self,
        x: NodeId,
        indices: NodeId,
        codebook: NodeId,
        kind: SynthKind,
        shape: Shape,
    ) -> NodeId {
        self.push(
            Op::SynthMatMul { kind },
            vec![x, indices, codebook],
            shape,
            None,
        )
    }

    /// [`synth_matmul`](Self::synth_matmul) with a LOW-PRECISION codebook (fp8 /
    /// fp4 / nvf4 / custom `fpXmYeZ`). The centroids are stored as `fmt`/`layout`
    /// codes (`codebook_codes` U8 + `codebook_scale` from
    /// [`scaled_quantize`](Self::scaled_quantize)) and decoded to f32 via
    /// [`Op::ScaledDequantize`] before the synth matmul — so ANY [`ScaledFormat`],
    /// including a parameterized [`ScaledFormat::Custom`] (e.g. `custom(3,0)` =
    /// `f4e3m0`) or an [`ScaleLayout::Nvfp4`] layout, works on every backend with
    /// no new kernel. The codebook is tiny, so the decode is negligible.
    #[allow(clippy::too_many_arguments)]
    pub fn synth_matmul_qcodebook(
        &mut self,
        x: NodeId,
        indices: NodeId,
        codebook_codes: NodeId,
        codebook_scale: NodeId,
        kind: SynthKind,
        fmt: ScaledFormat,
        layout: ScaleLayout,
        shape: Shape,
    ) -> NodeId {
        let cb_f32 = self.scaled_dequantize(codebook_codes, codebook_scale, fmt, layout);
        self.synth_matmul(x, indices, cb_f32, kind, shape)
    }

    /// KAN learnable spline activation. See [`Op::SplineActivation`]. `x` is
    /// `[.., C]`, `coeff` is `[C, num_basis]`; the output matches `x`. Each
    /// channel's univariate function is a Gaussian-RBF expansion with learned
    /// per-channel coefficients.
    pub fn spline_activation(
        &mut self,
        x: NodeId,
        coeff: NodeId,
        num_basis: u32,
        grid_min: f32,
        grid_max: f32,
    ) -> NodeId {
        let shape = self.node(x).shape.clone();
        self.push(
            Op::SplineActivation {
                num_basis,
                grid_min,
                grid_max,
            },
            vec![x, coeff],
            shape,
            None,
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
