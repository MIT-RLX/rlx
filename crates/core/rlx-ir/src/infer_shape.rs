// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Re-derive output shapes from inputs — used by the verifier to catch
//! builder / pass bugs that assign the wrong `Node::shape`.

use crate::op::*;
use crate::shape;
use crate::shape::Dim;
use crate::{DType, Graph, Node, Shape};

/// Records a shape rule that **ran and said no**, so it can be told apart from
/// a rule that does not exist.
///
/// `Option<Shape>` cannot express that difference, and collapsing the two is
/// what made `expand-from-non-unit-dim` invisible: `expand_shape` returned
/// `Err("cannot broadcast 128 with 256")`, the `.ok()` below turned it into
/// `None`, and [`crate::verify::verify_shapes`] reads `None` as "no rule for
/// this op — skip the node". A precise diagnosis was computed and discarded at
/// every one of these call sites.
///
/// Only the first rejection per node is kept: later ones are derived from the
/// same bad operands and would just be noise.
trait SinkErr {
    fn sink(self, out: &mut Option<String>) -> Option<Shape>;
}

impl SinkErr for Result<Shape, String> {
    fn sink(self, out: &mut Option<String>) -> Option<Shape> {
        match self {
            Ok(shape) => Some(shape),
            Err(message) => {
                out.get_or_insert(message);
                None
            }
        }
    }
}

/// Infer the output shape of `node` from its op and input shapes.
///
/// Returns `None` when inference is not implemented for the op (the
/// verifier skips those nodes rather than failing open), **and also** when a
/// rule rejected the operands. Use [`infer_output_shape_reporting`] to tell
/// those apart — the verifier does.
pub fn infer_output_shape(graph: &Graph, node: &Node) -> Option<Shape> {
    infer_output_shape_reporting(graph, node).0
}

/// [`infer_output_shape`] plus the reason it gave up, when it had one.
///
/// `(Some(shape), _)` — inferred. `(None, Some(why))` — a rule ran and
/// rejected these operands; that is a verifier finding, not a coverage gap.
/// `(None, None)` — no rule for this op.
pub fn infer_output_shape_reporting(graph: &Graph, node: &Node) -> (Option<Shape>, Option<String>) {
    let mut rejected = None;
    let shape = infer_output_shape_inner(graph, node, &mut rejected);
    (shape, rejected)
}

fn infer_output_shape_inner(
    graph: &Graph,
    node: &Node,
    rejected: &mut Option<String>,
) -> Option<Shape> {
    let in_shape = |i: usize| graph.shape(node.inputs[i]);
    match &node.op {
        Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => None,

        Op::MatMul => shape::matmul_shape(in_shape(0), in_shape(1)).sink(rejected),
        // MoE grouped GEMM: input [M,K], expert bank [E,K,N] → [M,N]. When the
        // operands disagree the rejection now reaches the verifier, which
        // reports it device-free — the backends still reject it themselves,
        // with the same message, but no longer have to be the first to notice.
        Op::GroupedMatMul => shape::grouped_matmul_shape(in_shape(0), in_shape(1)).sink(rejected),
        Op::LogMel => crate::audio::log_mel_output_shape(in_shape(0), in_shape(1)).sink(rejected),
        Op::LogMelBackward => Some(shape::unary_shape(in_shape(0))),
        Op::WelchPeaks { k, n_segments } => {
            crate::audio::welch_peaks_output_shape(in_shape(0), *k, *n_segments).sink(rejected)
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
        Op::Binary(_) => shape::binary_shape(in_shape(0), in_shape(1)).sink(rejected),
        Op::Compare(_) => shape::compare_shape(in_shape(0), in_shape(1)).sink(rejected),
        Op::Where => {
            let branches = shape::binary_shape(in_shape(1), in_shape(2)).sink(rejected)?;
            shape::binary_shape(in_shape(0), &branches)
                .sink(rejected)
                .map(|s| s.with_dtype(branches.dtype()))
        }
        Op::Fma => {
            let ab = shape::binary_shape(in_shape(0), in_shape(1)).sink(rejected)?;
            shape::binary_shape(&ab, in_shape(2)).sink(rejected)
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

        Op::Reduce { axes, keep_dim, .. } => {
            shape::reduce_shape(in_shape(0), axes, *keep_dim).sink(rejected)
        }
        Op::Histogram { bins, .. } => Some(Shape::new(&[*bins], DType::F32)),
        Op::ArgMax { axis, keep_dim } | Op::ArgMin { axis, keep_dim } => {
            shape::reduce_shape(in_shape(0), &[*axis], *keep_dim).sink(rejected)
        }
        Op::Softmax { .. } => Some(shape::softmax_shape(in_shape(0))),
        Op::Cumsum { .. } | Op::CumProd { .. } | Op::CumMax { .. } => {
            Some(shape::unary_shape(in_shape(0)))
        }

        Op::Reshape { new_shape } => shape::reshape_shape(in_shape(0), new_shape).sink(rejected),
        Op::Transpose { perm } => shape::transpose_shape(in_shape(0), perm).sink(rejected),
        Op::Narrow { axis, len, .. } => {
            shape::narrow_shape(in_shape(0), *axis, *len).sink(rejected)
        }
        Op::Concat { axis } => {
            let inputs: Vec<&Shape> = node.inputs.iter().map(|&id| graph.shape(id)).collect();
            shape::concat_shape(&inputs, *axis).sink(rejected)
        }
        // In-place append: output is the [0..pos+1] prefix of `cache` (input 0)
        // along `axis`; it aliases cache's buffer. `None` on a malformed operand
        // pair so the declared-vs-inferred check stays quiet and `verify_op`
        // reports the actual rule that was broken instead.
        Op::KvAppend { axis, pos } => {
            shape::kv_append_shape(in_shape(0), in_shape(1), *axis, *pos).sink(rejected)
        }
        Op::Gather { axis } => shape::gather_shape(in_shape(0), in_shape(1), *axis).sink(rejected),
        // ScatterND / ScatterElements output matches `data` (input 0).
        Op::ScatterNd { .. } | Op::ScatterElements { .. } => Some(shape::unary_shape(in_shape(0))),
        // GatherElements output matches `indices` (input 1).
        Op::GatherElements { .. } => Some(in_shape(1).clone()),
        // GatherND: indices[:-1] + data[batch+k:] — fall back to declared shape.
        Op::GatherNd { .. } => Some(node.shape.clone()),
        // Reverse flips element order along axes; shape is unchanged.
        Op::Reverse { .. } => Some(shape::unary_shape(in_shape(0))),
        Op::Pad { pads, .. } => shape::pad_shape(in_shape(0), pads).sink(rejected),
        Op::Slice { axis, len, .. } => shape::slice_shape(in_shape(0), *axis, *len).sink(rejected),
        // Roll is a permutation of elements — shape is preserved exactly.
        Op::Roll { .. } => Some(shape::unary_shape(in_shape(0))),
        Op::Clamp { .. } | Op::Trilu { .. } => Some(shape::unary_shape(in_shape(0))),
        Op::Tile { reps } => shape::tile_shape(in_shape(0), reps).sink(rejected),
        Op::Expand { target_shape } => {
            shape::expand_shape(in_shape(0), target_shape).sink(rejected)
        }

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
        Op::Attention {
            num_heads,
            head_dim,
            v_head_dim,
            ..
        } => Some(shape::attention_shape_vdim(
            in_shape(0),
            *num_heads,
            *head_dim,
            v_head_dim.unwrap_or(*head_dim),
        )),
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
            shape::im2col_output_shape(in_shape(0), ks, st, pad, dil).sink(rejected)
        }

        Op::FusedMatMulBiasAct { .. } => {
            shape::matmul_shape(in_shape(0), in_shape(1)).sink(rejected)
        }
        Op::FusedMatMulResidual => shape::matmul_shape(in_shape(0), in_shape(1)).sink(rejected),
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
            let b_scale = shape::broadcast(scale, x).sink(rejected)?;
            let b_shift = shape::broadcast(shift, x).sink(rejected)?;
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
            let b_gate = shape::broadcast(gate, x).sink(rejected)?;
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
            let b_scale = shape::broadcast(scale, x).sink(rejected)?;
            let b_shift = shape::broadcast(shift, x).sink(rejected)?;
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
            let b_gate = shape::broadcast(gate, x).sink(rejected)?;
            if b_gate.dims() != x.dims() {
                return None;
            }
            let nx = x.num_elements()?;
            let ng = gate.num_elements()?;
            Some(Shape::new(&[nx + nx + ng], x.dtype()))
        }

        // A *packed* weight is a 1-D byte blob whose shape says nothing about
        // `[k, n]` — the block layout does, and that lives in the scheme. So
        // there is genuinely no rule here, and this returns `None` **without**
        // recording a rejection: `matmul_shape` would answer "requires rank >=
        // 2, got 2 and 1", which is a true statement about the wrong operand.
        //
        // Before rejections were reported this distinction did not matter,
        // because both outcomes were discarded. It matters now: reporting it
        // would fail every quantized corpus case for having done the correct
        // thing.
        //
        // A weight that *is* declared rank-2 goes through a real rule — but
        // which one depends on the scheme. GGUF's is `[n, k]` (its native
        // `[out_dim, in_dim]` order), so the plain `[k, n]` matmul rule reads
        // those axes backwards and rejects a correct graph.
        Op::DequantMatMul { scheme } if scheme.is_gguf() => {
            if in_shape(1).rank() < 2 {
                return None;
            }
            shape::dequant_matmul_shape(in_shape(0), in_shape(1)).sink(rejected)
        }

        // `w [k, n]` for the Int8 / NVFP4 `DequantMatMul` schemes, `LoraMatMul`
        // and `QMatMul` alike — see each op's input contract.
        Op::DequantMatMul { .. } | Op::LoraMatMul { .. } | Op::QMatMul { .. } => {
            if in_shape(1).rank() < 2 {
                return None;
            }
            shape::matmul_shape(in_shape(0), in_shape(1)).sink(rejected)
        }

        // x [m, k] · Wᵀ → [m, n]; `n` is the row count of the `indices`
        // tensor ([n, k/entry_dim]). The weight is synthesized in-loop,
        // so its shape is derived from the operands, not a stored [k,n].
        Op::SynthMatMul { .. } => {
            let m = in_shape(0).dim(0).unwrap_static();
            let n = in_shape(1).dim(0).unwrap_static();
            Some(Shape::new(&[m, n], in_shape(0).dtype()))
        }

        // Fused synth backward: `dx` matches `x` [m,k] (input 0); `d_codebook`
        // matches the codebook [num_entries, entry_dim] (input 2).
        Op::SynthMatMulBackward { wrt, .. } => Some(match wrt {
            SynthBwdWrt::Dx => in_shape(0).clone(),
            SynthBwdWrt::Codebook => in_shape(2).clone(),
        }),

        // Reconstruct dense weight W[k,n]: indices [n, k/entry_dim] → k = dim1·ed,
        // n = dim0.
        Op::SynthReconstruct {
            kind: SynthKind::Codebook { entry_dim, .. },
        } => {
            // `w_bt[n,k]` — the backward-friendly layout; caller transposes to `W[k,n]`.
            let idx = in_shape(0);
            let n = idx.dim(0).unwrap_static();
            let k = idx.dim(1).unwrap_static() * *entry_dim as usize;
            Some(Shape::new(&[n, k], DType::F32))
        }

        // KAN spline activation is shape-preserving (per-channel univariate map).
        Op::SplineActivation { .. } => Some(shape::unary_shape(in_shape(0))),
        // dx has x's shape (input 0).
        Op::SplineActivationBackwardX { .. } => Some(shape::unary_shape(in_shape(0))),
        // dcoeff is [C, num_basis]; C = last dim of x (input 0).
        Op::SplineActivationBackwardCoeff { num_basis, .. } => {
            let x = in_shape(0);
            let c = x.dim(x.rank() - 1).unwrap_static();
            Some(Shape::new(&[c, *num_basis as usize], DType::F32))
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

        // Native low-precision *grouped* (MoE) GEMM: input [M,K], per-expert
        // weight [E,N,K] (K-last), expert_idx [M] → out [M,N] f32.
        Op::ScaledGroupedMatMul { .. } => {
            let input = in_shape(0);
            let weight = in_shape(1);
            if input.rank() < 2 || weight.rank() < 3 {
                None
            } else {
                let m = input.dims()[input.rank() - 2];
                let n = weight.dims()[weight.rank() - 2];
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

        // Packed 1-D gradient bundle; see `Op::GatedDeltaNetBackward`.
        Op::GatedDeltaNetBackward {
            state_size,
            carry_state,
            gate_per_channel,
        } => {
            let q = in_shape(0);
            let dims: Vec<usize> = (0..q.rank())
                .map(|i| match q.dim(i) {
                    crate::Dim::Static(v) => v,
                    _ => 0,
                })
                .collect();
            // [B, S, H, N]; a dynamic axis leaves the packed length unknowable.
            if dims.len() != 4 || dims.contains(&0) {
                None
            } else {
                let layout = crate::gdn::GdnBackwardLayout::new(
                    dims[0],
                    dims[1],
                    dims[2],
                    *state_size,
                    *gate_per_channel,
                    *carry_state,
                );
                Some(shape::Shape::new(&[layout.total_elems()], q.dtype()))
            }
        }

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
                .sink(rejected)
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
            .sink(rejected)
        }

        Op::Custom { .. }
        | Op::CustomFn { .. }
        | Op::Conv { .. }
        | Op::ConvTranspose2d { .. }
        | Op::Pool { .. }
        | Op::Fft { .. }
        | Op::FftQ { .. }
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
