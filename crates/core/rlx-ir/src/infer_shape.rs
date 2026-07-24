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

//! Re-derive output shapes from inputs — used by the verifier to catch
//! builder / pass bugs that assign the wrong `Node::shape`.

use crate::op::*;
use crate::shape;
use crate::shape::Dim;
use crate::{DType, Graph, Node, Shape};

/// Infer the output shape of `node` from its op and input shapes.
///
/// Returns `None` when inference is not implemented for the op (the
/// verifier skips those nodes rather than failing open).
pub fn infer_output_shape(graph: &Graph, node: &Node) -> Option<Shape> {
    let in_shape = |i: usize| graph.shape(node.inputs[i]);
    match &node.op {
        Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => None,

        Op::MatMul => shape::matmul_shape(in_shape(0), in_shape(1)).ok(),
        Op::LogMel => crate::audio::log_mel_output_shape(in_shape(0), in_shape(1)).ok(),
        Op::LogMelBackward => Some(shape::unary_shape(in_shape(0))),
        Op::WelchPeaks { k, n_segments } => {
            crate::audio::welch_peaks_output_shape(in_shape(0), *k, *n_segments).ok()
        }

        // ── Riemannian / SPD-manifold layers ────────────────────
        Op::BiMap => {
            // W [m,n] · X [n,n] · Wᵀ → [m,m]
            let m = in_shape(0).dim(0).unwrap_static();
            Some(Shape::new(&[m, m], in_shape(0).dtype()))
        }
        // ReEig/LogEig emit a packed [Y (n²), λ (n), U (n²)] buffer so the
        // backward reuses the eigendecomposition; the builder narrows Y.
        Op::ReEig { .. } | Op::LogEig { .. } => {
            let n = in_shape(0).dim(0).unwrap_static();
            Some(Shape::new(&[2 * n * n + n], in_shape(0).dtype()))
        }
        Op::SpdBatchNorm { .. } => Some(shape::unary_shape(in_shape(0))),
        Op::SpdKarcherMean { .. } | Op::SpdKarcherMeanWeighted { .. } => {
            // [batch, n, n] (+ optional weights [batch]) → [n, n]
            let n = in_shape(0).dim(1).unwrap_static();
            Some(Shape::new(&[n, n], in_shape(0).dtype()))
        }
        // Arbitrary-base AIRM maps / transport are all [n,n] → [n,n]; the
        // result matches the tangent/point operand (last input).
        Op::SpdLogMap | Op::SpdExpMap => Some(shape::unary_shape(in_shape(1))),
        Op::SpdParallelTransport => Some(shape::unary_shape(in_shape(2))),
        // Batched matrix function preserves the [batch, n, n] shape.
        Op::SpdMatrixFnBatch { .. } => Some(shape::unary_shape(in_shape(0))),
        // Backward reads (λ [n], U [n²], dY [n²]) → dX [n, n]; input 0 is λ.
        Op::ReEigBackward { .. } | Op::LogEigBackward { .. } => {
            let n = in_shape(0).dim(0).unwrap_static();
            Some(Shape::new(&[n, n], in_shape(0).dtype()))
        }
        // dX = dY shape (input 2 of [mean, G, dY]).
        Op::SpdBatchNormBackwardX { .. } => Some(shape::unary_shape(in_shape(2))),
        // dG = G shape (input 2 of [X, mean, G, dY]).
        Op::SpdBatchNormBackwardG { .. } => Some(shape::unary_shape(in_shape(2))),
        // Packed per-input gradients: [k, n, n] where k = #differentiable
        // inputs (base+arg = 2, from+to+v = 3). n from the base/from operand.
        Op::SpdLogMapBackward | Op::SpdExpMapBackward => {
            let n = in_shape(0).dim(0).unwrap_static();
            Some(Shape::new(&[2, n, n], in_shape(0).dtype()))
        }
        Op::SpdParallelTransportBackward => {
            let n = in_shape(0).dim(0).unwrap_static();
            Some(Shape::new(&[3, n, n], in_shape(0).dtype()))
        }
        // dX = X shape ([B, n, n]).
        Op::SpdMatrixFnBatchBackward { .. } => Some(shape::unary_shape(in_shape(0))),
        // Eigh packs [λ (n) ∥ U (n²)] = [n²+n].
        Op::Eigh => {
            let n = in_shape(0).dim(0).unwrap_static();
            Some(Shape::new(&[n * n + n], in_shape(0).dtype()))
        }
        Op::EighBatch => {
            let b = in_shape(0).dim(0).unwrap_static();
            let n = in_shape(0).dim(1).unwrap_static();
            Some(Shape::new(&[b, n * n + n], in_shape(0).dtype()))
        }
        // Backward → Ā; recover n from the packed length L=n²+n ⇒ n=(√(1+4L)−1)/2.
        Op::EighBackward => {
            let len = in_shape(0).dim(0).unwrap_static();
            let n = (((1 + 4 * len) as f64).sqrt().round() as usize - 1) / 2;
            Some(Shape::new(&[n, n], in_shape(0).dtype()))
        }
        Op::EighBatchBackward => {
            let b = in_shape(0).dim(0).unwrap_static();
            let len = in_shape(0).dim(1).unwrap_static();
            let n = (((1 + 4 * len) as f64).sqrt().round() as usize - 1) / 2;
            Some(Shape::new(&[b, n, n], in_shape(0).dtype()))
        }
        Op::Binary(_) => shape::binary_shape(in_shape(0), in_shape(1)).ok(),
        Op::Compare(_) => shape::compare_shape(in_shape(0), in_shape(1)).ok(),
        Op::Where => {
            let branches = shape::binary_shape(in_shape(1), in_shape(2)).ok()?;
            shape::binary_shape(in_shape(0), &branches)
                .ok()
                .map(|s| s.with_dtype(branches.dtype()))
        }
        Op::Fma => {
            let ab = shape::binary_shape(in_shape(0), in_shape(1)).ok()?;
            shape::binary_shape(&ab, in_shape(2)).ok()
        }

        Op::Activation(_) | Op::ReluBackward | Op::Conjugate => {
            Some(shape::unary_shape(in_shape(0)))
        }
        Op::ComplexNormSq => Some(Shape::from_dims(in_shape(0).dims(), DType::F32)),
        Op::ComplexNormSqBackward => Some(shape::unary_shape(in_shape(0))),
        Op::Cast { to } => Some(shape::cast_shape(in_shape(0), *to)),
        Op::StopGradient => Some(shape::unary_shape(in_shape(0))),

        Op::RngNormal { .. } | Op::RngUniform { .. } => {
            if node.inputs.is_empty() {
                Some(node.shape.clone())
            } else {
                Some(shape::unary_shape(in_shape(0)))
            }
        }

        Op::Reduce { axes, keep_dim, .. } => shape::reduce_shape(in_shape(0), axes, *keep_dim).ok(),
        Op::ArgMax { axis, keep_dim } | Op::ArgMin { axis, keep_dim } => {
            shape::reduce_shape(in_shape(0), &[*axis], *keep_dim).ok()
        }
        Op::Softmax { .. } => Some(shape::softmax_shape(in_shape(0))),
        Op::Cumsum { .. } => Some(shape::unary_shape(in_shape(0))),

        Op::Reshape { new_shape } => shape::reshape_shape(in_shape(0), new_shape).ok(),
        Op::Transpose { perm } => shape::transpose_shape(in_shape(0), perm).ok(),
        Op::Narrow { axis, len, .. } => shape::narrow_shape(in_shape(0), *axis, *len).ok(),
        Op::Concat { axis } => {
            let inputs: Vec<&Shape> = node.inputs.iter().map(|&id| graph.shape(id)).collect();
            shape::concat_shape(&inputs, *axis).ok()
        }
        Op::Gather { axis } => shape::gather_shape(in_shape(0), in_shape(1), *axis).ok(),
        // ScatterND / ScatterElements output matches `data` (input 0).
        Op::ScatterNd { .. } | Op::ScatterElements { .. } => Some(shape::unary_shape(in_shape(0))),
        // GatherElements output matches `indices` (input 1).
        Op::GatherElements { .. } => Some(in_shape(1).clone()),
        // GatherND: indices[:-1] + data[batch+k:] — fall back to declared shape.
        Op::GatherNd { .. } => Some(node.shape.clone()),
        // Reverse flips element order along axes; shape is unchanged.
        Op::Reverse { .. } => Some(shape::unary_shape(in_shape(0))),
        Op::Expand { target_shape } => shape::expand_shape(in_shape(0), target_shape).ok(),

        Op::LayerNorm { .. } | Op::LayerNorm2d { .. } | Op::GroupNorm { .. } => {
            Some(shape::unary_shape(in_shape(0)))
        }
        Op::RmsNorm { .. } => {
            let in_s = in_shape(0);
            let out = &node.shape;
            // `FuseRmsNormReshape` keeps the 3-D (or higher) input but
            // assigns a leading-flattened `[∏leading, H]` output shape.
            if out.rank() == 2 && in_s.rank() > 2 {
                if let Some(flat) = shape::leading_flatten_fused_shape(in_s) {
                    if flat == *out {
                        return Some(out.clone());
                    }
                }
            }
            Some(shape::unary_shape(in_s))
        }
        Op::ResizeNearest2x => {
            let in_s = in_shape(0);
            if in_s.rank() == 4 {
                Some(Shape::new(
                    &[
                        in_s.dim(0).unwrap_static(),
                        in_s.dim(1).unwrap_static(),
                        in_s.dim(2).unwrap_static() * 2,
                        in_s.dim(3).unwrap_static() * 2,
                    ],
                    in_s.dtype(),
                ))
            } else {
                None
            }
        }
        Op::Interpolate3d { size } => {
            let in_s = in_shape(0);
            if in_s.rank() == 5 && size.len() == 3 {
                Some(Shape::new(
                    &[
                        in_s.dim(0).unwrap_static(),
                        in_s.dim(1).unwrap_static(),
                        size[0],
                        size[1],
                        size[2],
                    ],
                    in_s.dtype(),
                ))
            } else {
                None
            }
        }
        Op::Attention { .. } => Some(shape::attention_shape(in_shape(0))),
        Op::Rope { .. } => Some(shape::unary_shape(in_shape(0))),
        Op::AxialRope2d { .. } => Some(shape::unary_shape(in_shape(0))),

        Op::Im2Col {
            kernel_size,
            stride,
            padding,
            dilation,
        } => {
            let ks = [kernel_size[0], kernel_size.get(1).copied().unwrap_or(1)];
            let st = [stride[0], stride.get(1).copied().unwrap_or(1)];
            let pad = [padding[0], padding.get(1).copied().unwrap_or(0)];
            let dil = [dilation[0], dilation.get(1).copied().unwrap_or(1)];
            shape::im2col_output_shape(in_shape(0), ks, st, pad, dil).ok()
        }

        Op::FusedMatMulBiasAct { .. } => shape::matmul_shape(in_shape(0), in_shape(1)).ok(),
        // Like `Op::Conv`, the output shape is set explicitly by the fusion pass
        // (from the pre-fusion conv/activation output); nothing to infer here.
        Op::FusedConvBiasAct { .. } => None,
        Op::FusedSwiGLU { .. } => None,
        Op::FusedResidualLN { .. } | Op::FusedResidualRmsNorm { .. } => {
            Some(shape::unary_shape(in_shape(0)))
        }

        // DiT adaLN-Zero: out shape == x; scale/shift must broadcast to x
        // (typically `[B,1,D]` over `[B,S,D]`) with matching last feature dim.
        Op::AdaLayerNorm { .. } => {
            let x = in_shape(0);
            let scale = in_shape(1);
            let shift = in_shape(2);
            let b_scale = shape::broadcast(scale, x).ok()?;
            let b_shift = shape::broadcast(shift, x).ok()?;
            if b_scale.dims() != x.dims() || b_shift.dims() != x.dims() {
                return None;
            }
            Some(shape::unary_shape(x))
        }
        // DiT gated residual: out == x; y same shape as x; gate broadcasts to x.
        Op::GatedResidual => {
            let x = in_shape(0);
            let y = in_shape(1);
            let gate = in_shape(2);
            if y.dims() != x.dims() {
                return None;
            }
            let b_gate = shape::broadcast(gate, x).ok()?;
            if b_gate.dims() != x.dims() {
                return None;
            }
            Some(shape::unary_shape(x))
        }
        // Packed DiT modulation backward: 1-D [nx + 2·ns] / [nx + ny + ng].
        Op::AdaLayerNormBackward { .. } => {
            let x = in_shape(0);
            let scale = in_shape(1);
            let shift = in_shape(2);
            let dy = in_shape(3);
            if dy.dims() != x.dims() || scale.dims() != shift.dims() {
                return None;
            }
            let b_scale = shape::broadcast(scale, x).ok()?;
            let b_shift = shape::broadcast(shift, x).ok()?;
            if b_scale.dims() != x.dims() || b_shift.dims() != x.dims() {
                return None;
            }
            let nx = x.num_elements()?;
            let ns = scale.num_elements()?;
            Some(Shape::new(&[nx + 2 * ns], x.dtype()))
        }
        Op::GatedResidualBackward => {
            let x = in_shape(0);
            let y = in_shape(1);
            let gate = in_shape(2);
            let dy = in_shape(3);
            if y.dims() != x.dims() || dy.dims() != x.dims() {
                return None;
            }
            let b_gate = shape::broadcast(gate, x).ok()?;
            if b_gate.dims() != x.dims() {
                return None;
            }
            let nx = x.num_elements()?;
            let ng = gate.num_elements()?;
            Some(Shape::new(&[nx + nx + ng], x.dtype()))
        }

        Op::DequantMatMul { .. } | Op::LoraMatMul { .. } | Op::QMatMul { .. } => {
            shape::matmul_shape(in_shape(0), in_shape(1)).ok()
        }

        // Full linear convolution: last axis of x (`L`) grows to `L + M − 1`,
        // where `M` is the rank-1 impulse-response length (input 1).
        Op::PartitionedConv { .. } => {
            let x = in_shape(0);
            let ir = in_shape(1);
            let l = x.dim(x.rank() - 1).unwrap_static();
            let m = ir.dim(0).unwrap_static();
            let mut dims: Vec<Dim> = x.dims().to_vec();
            *dims.last_mut().unwrap() = Dim::Static(l + m - 1);
            Some(Shape::from_dims(&dims, x.dtype()))
        }

        // Native low-precision GEMM, TN layout: lhs [m,k], rhs [n,k] (K-last),
        // out = [m,n] f32 (f32 is the accumulation type — operands are U8 codes).
        Op::ScaledMatMul { .. } => {
            let lhs = in_shape(0);
            let rhs = in_shape(1);
            if lhs.rank() < 2 || rhs.rank() < 2 {
                None
            } else {
                let m = lhs.dims()[lhs.rank() - 2];
                let n = rhs.dims()[rhs.rank() - 2];
                Some(Shape::from_dims(&[m, n], DType::F32))
            }
        }

        // Quantize keeps the logical shape, switches dtype to packed U8 codes.
        Op::ScaledQuantize { .. } => Some(shape::unary_shape(in_shape(0)).with_dtype(DType::U8)),

        // Dequantize: codes (U8) → f32, same logical shape.
        Op::ScaledDequantize { .. } => Some(shape::unary_shape(in_shape(0)).with_dtype(DType::F32)),

        // Scale tensor: one value (per-tensor), or one per block along the
        // last (K) axis (block / NVFP4). Dtype follows the layout.
        Op::ScaledQuantScale {
            scale_layout,
            format,
        } => {
            let _ = format;
            let sd = scale_layout.scale_dtype();
            match scale_layout {
                crate::ScaleLayout::PerTensor => Some(Shape::new(&[1], sd)),
                crate::ScaleLayout::BlockMxE8M0 { block }
                | crate::ScaleLayout::Nvfp4 { group: block } => {
                    let x = in_shape(0);
                    let dims = x.dims();
                    match dims.last() {
                        Some(Dim::Static(k)) => {
                            let mut out: Vec<usize> = dims[..dims.len() - 1]
                                .iter()
                                .map(|d| d.unwrap_static())
                                .collect();
                            out.push((*k).div_ceil(*block as usize));
                            Some(Shape::new(&out, sd))
                        }
                        // Dynamic K: builder must supply the shape explicitly.
                        _ => None,
                    }
                }
            }
        }

        Op::GaussianSplatRender { width, height, .. } => Some(Shape::new(
            &[(*width as usize) * (*height as usize) * 4],
            in_shape(0).dtype(),
        )),

        Op::GaussianSplatRenderBackward { .. } => {
            let count = in_shape(0).num_elements().unwrap_or(0) / 3;
            let sh_len = in_shape(5).num_elements().unwrap_or(0);
            let sh_coeff_count = if count == 0 {
                1
            } else {
                (sh_len / (count * 3)).max(1)
            };
            let packed = crate::ops::splat::gaussian_splat_packed_grad_len(count, sh_coeff_count);
            Some(Shape::new(&[packed], in_shape(0).dtype()))
        }

        Op::GaussianSplatPrepare {
            width,
            height,
            tile_size,
            max_list_entries,
            ..
        } => {
            let count = in_shape(0).num_elements().unwrap_or(0) / 3;
            let len = crate::ops::splat::gaussian_splat_prep_packed_len(
                count,
                *max_list_entries,
                *width,
                *height,
                *tile_size,
            );
            Some(Shape::new(&[len], in_shape(0).dtype()))
        }

        Op::GaussianSplatRasterize { width, height, .. } => Some(Shape::new(
            &[(*width as usize) * (*height as usize) * 4],
            in_shape(0).dtype(),
        )),

        Op::DotGeneral { .. }
        | Op::If { .. }
        | Op::While { .. }
        | Op::SelectiveScan { .. }
        | Op::GatedDeltaNet { .. }
        | Op::Mamba2 { .. }
        | Op::FusedAttentionBlock { .. }
        | Op::FusedTransformerLayer { .. } => Some(shape::unary_shape(in_shape(0))),
        // x `[batch, seq, input]` → y `[batch, seq, hidden]` (preserve
        // batch/seq, static or dynamic; only the feature axis changes).
        Op::Lstm {
            hidden_size,
            bidirectional,
            ..
        }
        | Op::Gru {
            hidden_size,
            bidirectional,
            ..
        }
        | Op::Rnn {
            hidden_size,
            bidirectional,
            ..
        } => {
            let d = if *bidirectional { 2 } else { 1 };
            Some(
                in_shape(0)
                    .clone()
                    .with_dim(2, Dim::Static(d * *hidden_size)),
            )
        }
        Op::Scan {
            length,
            save_trajectory,
            ..
        } => {
            let carry = in_shape(0);
            if *save_trajectory {
                let mut dims = vec![Dim::Static(*length as usize)];
                for i in 0..carry.rank() {
                    dims.push(carry.dim(i));
                }
                Some(Shape::from_dims(&dims, carry.dtype()))
            } else {
                Some(shape::unary_shape(carry))
            }
        }
        Op::ElementwiseRegion {
            prologue, chain, ..
        } => {
            // A fused elementwise chain broadcasts across ALL of its inputs, not
            // just input 0 — e.g. `(g[1,C,1] + g2[1,C,1]) + x[1,C,T]) * mask[1,1,T]`
            // is `[1,C,T]`. Folding only input 0 mis-infers `[1,C,1]` and trips the
            // verifier on strict backends (MLX/wgpu compile path).
            let mut in_s = in_shape(0).clone();
            for i in 1..node.inputs.len() {
                if let Ok(b) = shape::binary_shape(&in_s, in_shape(i)) {
                    in_s = b;
                }
            }
            if *prologue == RegionPrologue::ResizeNearest2x && in_s.rank() == 4 {
                in_s = Shape::new(
                    &[
                        in_s.dim(0).unwrap_static(),
                        in_s.dim(1).unwrap_static(),
                        in_s.dim(2).unwrap_static() * 2,
                        in_s.dim(3).unwrap_static() * 2,
                    ],
                    in_s.dtype(),
                );
            }
            // Output dtype = dtype of the chain's final step, NOT input 0's:
            // input 0 may be a bool `Where` condition (`where(cond, a, b) + …`),
            // so inheriting its dtype mis-types the region as bool.
            if let Some(dt) = chain_output_dtype(chain, &|i| in_shape(i).dtype()) {
                in_s = in_s.with_dtype(dt);
            }
            Some(in_s)
        }
        Op::BatchElementwiseRegion {
            prologue,
            num_batch_inputs,
            ..
        } => {
            let n = *num_batch_inputs as usize;
            let mut out_s = in_shape(0).clone();
            if *prologue == RegionPrologue::ResizeNearest2x && out_s.rank() == 4 {
                out_s = Shape::new(
                    &[
                        out_s.dim(0).unwrap_static(),
                        out_s.dim(1).unwrap_static(),
                        out_s.dim(2).unwrap_static() * 2,
                        out_s.dim(3).unwrap_static() * 2,
                    ],
                    out_s.dtype(),
                );
            }
            if out_s.rank() >= 1 && n > 1 {
                let mut batch_dim = 0usize;
                for i in 0..n.min(node.inputs.len()) {
                    batch_dim += in_shape(i).dim(0).unwrap_static();
                }
                if batch_dim > 0 {
                    out_s = out_s.with_dim(0, shape::Dim::Static(batch_dim));
                }
            }
            Some(out_s)
        }
        Op::TransformRegion { steps, .. } => {
            let mut in_s = in_shape(0).clone();
            for step in steps {
                if !matches!(step, TransformStep::ResizeNearest2x(_)) {
                    return None;
                }
                if in_s.rank() != 4 {
                    return None;
                }
                in_s = Shape::new(
                    &[
                        in_s.dim(0).unwrap_static(),
                        in_s.dim(1).unwrap_static(),
                        in_s.dim(2).unwrap_static() * 2,
                        in_s.dim(3).unwrap_static() * 2,
                    ],
                    in_s.dtype(),
                );
            }
            Some(in_s)
        }
        // NCDHW 3-D convs: kernel size comes from the 5-D weight
        // (`[C_out, C_in/g, kD, kH, kW]` for conv, `[C_in, C_out/g, ...]` for
        // transpose), so we can re-derive the output shape here (unlike the
        // 2-D convs, whose kernel size lives on the op).
        Op::Conv3d {
            stride,
            padding,
            dilation,
            groups,
        } => {
            let w = in_shape(1);
            if w.rank() != 5 {
                return None;
            }
            let ks = [
                w.dim(2).unwrap_static(),
                w.dim(3).unwrap_static(),
                w.dim(4).unwrap_static(),
            ];
            shape::conv3d_output_shape(in_shape(0), w, ks, *stride, *padding, *dilation, *groups)
                .ok()
        }
        Op::ConvTranspose3d {
            stride,
            padding,
            dilation,
            output_padding,
            groups,
        } => {
            let w = in_shape(1);
            if w.rank() != 5 {
                return None;
            }
            let ks = [
                w.dim(2).unwrap_static(),
                w.dim(3).unwrap_static(),
                w.dim(4).unwrap_static(),
            ];
            shape::conv_transpose3d_output_shape(
                in_shape(0),
                w,
                ks,
                *stride,
                *padding,
                *dilation,
                *output_padding,
                *groups,
            )
            .ok()
        }

        Op::Custom { .. }
        | Op::CustomFn { .. }
        | Op::Conv { .. }
        | Op::ConvTranspose2d { .. }
        | Op::Pool { .. }
        | Op::Fft { .. }
        | Op::FftButterflyStage { .. } => None,
        _ => None,
    }
}

/// Output dtype of a fused elementwise `chain`, by walking each step:
/// `Compare → Bool`, `Cast → its dtype`, everything else → the dtype of its
/// (value) operand. `input_dtype(i)` resolves a chain input's dtype.
fn chain_output_dtype(
    chain: &[ChainStep],
    input_dtype: &dyn Fn(usize) -> crate::DType,
) -> Option<crate::DType> {
    let operand = |o: &ChainOperand, step_dt: &[crate::DType]| -> crate::DType {
        match o {
            ChainOperand::Input(i) => input_dtype(*i as usize),
            ChainOperand::Step(j) => step_dt[*j as usize],
        }
    };
    let mut step_dt: Vec<crate::DType> = Vec::with_capacity(chain.len());
    for step in chain {
        let dt = match step {
            ChainStep::Compare(..) => crate::DType::Bool,
            ChainStep::Cast(d, _) => *d,
            ChainStep::Activation(_, o) => operand(o, &step_dt),
            ChainStep::Binary(_, l, _) => operand(l, &step_dt),
            ChainStep::Where(_, t, _) => operand(t, &step_dt),
        };
        step_dt.push(dt);
    }
    step_dt.last().copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Op};

    #[test]
    fn scan_save_trajectory_infers_length_by_carry() {
        let mut g = Graph::new("scan_traj");
        let init = g.input("init", Shape::new(&[80], DType::F32));
        let body = Graph::new("body");
        let scan = g.add_node(
            Op::Scan {
                body: Box::new(body),
                length: 70,
                save_trajectory: true,
                num_bcast: 0,
                num_xs: 0,
                num_checkpoints: 0,
            },
            vec![init],
            Shape::new(&[70, 80], DType::F32),
        );
        let node = g.node(scan).clone();
        let inferred = infer_output_shape(&g, &node).expect("scan infer");
        assert_eq!(inferred.dims(), node.shape.dims());
    }

    #[test]
    fn scan_without_trajectory_infers_carry_only() {
        let mut g = Graph::new("scan_carry");
        let init = g.input("init", Shape::new(&[80], DType::F32));
        let body = Graph::new("body");
        let scan = g.add_node(
            Op::Scan {
                body: Box::new(body),
                length: 70,
                save_trajectory: false,
                num_bcast: 0,
                num_xs: 0,
                num_checkpoints: 0,
            },
            vec![init],
            Shape::new(&[80], DType::F32),
        );
        let node = g.node(scan).clone();
        let inferred = infer_output_shape(&g, &node).expect("scan infer");
        assert_eq!(inferred.dims(), node.shape.dims());
    }
}
