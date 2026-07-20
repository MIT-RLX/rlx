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

//! `lower` — extracted from the `lower` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::hlo::{
    Computation, ConvDimNumbers, DotDimNumbers, GatherDimNumbers, HloBuilder, Literal, LiteralData,
    ProgramShape, ScatterDimNumbers, Shape, Window, WindowDim, prim, prim_of,
};
use rlx_ir::op::{
    Activation, AdaNormKind, BinaryOp, ChainOperand, ChainStep, CmpOp, MaskKind, ReduceOp,
    RegionPrologue, TransformStep,
};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, NodeId, Op};
use std::collections::HashMap;

use super::*;

impl<'a> LowerCtx<'a> {
    pub(crate) fn lower_node(&mut self, nid: NodeId) -> i64 {
        let n = self.graph.node(nid);
        let out_shape = self.ir_shape(nid);
        let out_dt = self.dtype(nid);

        match &n.op {
            // Inputs / Params already handled by the caller — they
            // never reach lower_node.
            Op::Input { .. } | Op::Param { .. } => unreachable!(),

            Op::Constant { data } => self.lower_constant(data, out_shape, out_dt),

            Op::Activation(act) => {
                let x = self.hlo(n.inputs[0]);
                self.lower_activation(*act, x, out_shape)
            }

            Op::Cast { to } => {
                let x = self.hlo(n.inputs[0]);
                let from = self.dtype(n.inputs[0]);
                self.lower_cast(x, from, *to, &out_shape.dimensions)
            }

            // INT8 quantization, per-tensor (axis=None) or per-channel
            // (axis=Some(d), scales/zero_points indexed by axis d).
            //   q = saturate_i8(round(x / scale[c]) + zero_point[c])
            // For per-channel we materialize a 1-D constant of length
            // `input.dim(axis)` and broadcast along the channel axis.
            Op::Quantize {
                axis,
                scales,
                zero_points,
            } => {
                let x = self.hlo(n.inputs[0]);
                let in_prim = prim_of(self.dtype(n.inputs[0]));
                let inv_b = self.broadcast_q_factor(
                    *axis,
                    &scales.iter().map(|s| 1.0 / *s).collect::<Vec<_>>(),
                    &out_shape.dimensions,
                    in_prim,
                );
                let scaled = self.entry.binary(
                    "multiply",
                    x,
                    inv_b,
                    Shape::array(in_prim, &out_shape.dimensions),
                );
                let rounded = self.entry.unary(
                    "round-nearest-even",
                    scaled,
                    Shape::array(in_prim, &out_shape.dimensions),
                );
                let zp_b = self.broadcast_q_factor(
                    *axis,
                    &zero_points.iter().map(|z| *z as f32).collect::<Vec<_>>(),
                    &out_shape.dimensions,
                    in_prim,
                );
                let added = self.entry.binary(
                    "add",
                    rounded,
                    zp_b,
                    Shape::array(in_prim, &out_shape.dimensions),
                );
                // Convert handles saturation per HLO semantics.
                self.entry.convert(added, out_shape)
            }
            Op::Dequantize {
                axis,
                scales,
                zero_points,
            } => {
                let q = self.hlo(n.inputs[0]);
                let promoted = self
                    .entry
                    .convert(q, Shape::array(prim::F32, &out_shape.dimensions));
                let zp_b = self.broadcast_q_factor(
                    *axis,
                    &zero_points.iter().map(|z| *z as f32).collect::<Vec<_>>(),
                    &out_shape.dimensions,
                    prim::F32,
                );
                let centered = self.entry.binary(
                    "subtract",
                    promoted,
                    zp_b,
                    Shape::array(prim::F32, &out_shape.dimensions),
                );
                let s_b = self.broadcast_q_factor(*axis, scales, &out_shape.dimensions, prim::F32);
                self.entry.binary("multiply", centered, s_b, out_shape)
            }

            Op::Binary(op) => {
                let a = self.hlo(n.inputs[0]);
                let b = self.hlo(n.inputs[1]);
                self.lower_binary(*op, a, b, n.inputs[0], n.inputs[1], out_shape)
            }

            Op::Compare(op) => {
                let a = self.hlo(n.inputs[0]);
                let b = self.hlo(n.inputs[1]);
                let dir = match op {
                    CmpOp::Eq => "EQ",
                    CmpOp::Ne => "NE",
                    CmpOp::Lt => "LT",
                    CmpOp::Le => "LE",
                    CmpOp::Gt => "GT",
                    CmpOp::Ge => "GE",
                };
                let (a, b) =
                    self.broadcast_pair_to(a, b, n.inputs[0], n.inputs[1], &out_shape.dimensions);
                self.entry
                    .compare(a, b, dir, Shape::pred(&out_shape.dimensions))
            }

            Op::Where => {
                let c = self.hlo(n.inputs[0]);
                let t = self.hlo(n.inputs[1]);
                let f = self.hlo(n.inputs[2]);
                self.entry.select(c, t, f, out_shape)
            }

            Op::ElementwiseRegion {
                chain,
                num_inputs,
                scalar_input_mask,
                input_modulus,
                prologue,
                prologue_input,
            } => self.lower_elementwise_region(
                &n.inputs,
                chain,
                *num_inputs,
                *scalar_input_mask,
                input_modulus,
                out_shape,
                *prologue,
                *prologue_input,
            ),

            Op::TransformRegion { steps, num_inputs } => {
                self.lower_transform_region(&n.inputs, steps, *num_inputs, out_shape)
            }

            Op::BatchElementwiseRegion {
                chain,
                num_batch_inputs,
                scalar_input_mask,
                input_modulus,
                prologue,
                prologue_input,
            } => self.lower_batch_elementwise_region(
                &n.inputs,
                chain,
                *num_batch_inputs,
                *scalar_input_mask,
                input_modulus,
                *prologue,
                *prologue_input,
                &n.shape,
                out_shape,
            ),

            Op::MatMul => self.lower_matmul(n.inputs[0], n.inputs[1], out_shape),

            Op::DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
            } => {
                let a = self.hlo(n.inputs[0]);
                let b = self.hlo(n.inputs[1]);
                let dn = DotDimNumbers {
                    lhs_contracting: lhs_contracting.iter().map(|&x| x as i64).collect(),
                    rhs_contracting: rhs_contracting.iter().map(|&x| x as i64).collect(),
                    lhs_batch: lhs_batch.iter().map(|&x| x as i64).collect(),
                    rhs_batch: rhs_batch.iter().map(|&x| x as i64).collect(),
                };
                self.entry.dot_general(a, b, dn, out_shape)
            }

            Op::LayerNorm { axis, eps } => self.lower_layernorm(
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                *axis,
                *eps,
                out_shape,
            ),

            Op::RmsNorm { axis, eps } => self.lower_rmsnorm(
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                *axis,
                *eps,
                out_shape,
            ),

            Op::FusedResidualLN { has_bias, eps } => {
                self.lower_fused_residual_ln(&n.inputs, *has_bias, *eps, out_shape)
            }

            Op::AdaLayerNorm { norm, eps } => {
                self.lower_ada_layer_norm(&n.inputs, *norm, *eps, out_shape)
            }

            Op::GatedResidual => self.lower_gated_residual(&n.inputs, out_shape),

            Op::AdaLayerNormBackward { norm, eps } => {
                self.lower_ada_layer_norm_backward(&n.inputs, *norm, *eps, out_shape)
            }

            Op::GatedResidualBackward => self.lower_gated_residual_backward(&n.inputs, out_shape),

            Op::FusedMatMulBiasAct { activation } => {
                self.lower_fused_matmul_bias_act(&n.inputs, *activation, out_shape)
            }

            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
                score_scale: _,
                attn_logit_softcap: _,
            } => self.lower_attention(&n.inputs, *num_heads, *head_dim, *mask_kind, out_shape),

            Op::Rope {
                head_dim, n_rot: _, ..
            } => self.lower_rope(n.inputs[0], n.inputs[1], n.inputs[2], *head_dim, out_shape),

            Op::Reshape { new_shape: _ } => {
                let x = self.hlo(n.inputs[0]);
                self.entry.reshape(x, out_shape)
            }
            Op::StopGradient => {
                // Pure forward identity — XLA `stop_gradient` only affects AD,
                // which has already run by lowering time. Alias the input HLO.
                self.hlo(n.inputs[0])
            }
            Op::Transpose { perm } => {
                let x = self.hlo(n.inputs[0]);
                let perm_i64: Vec<i64> = perm.iter().map(|&p| p as i64).collect();
                self.entry.transpose(x, &perm_i64, out_shape)
            }
            Op::Narrow { axis, start, len } => {
                let x = self.hlo(n.inputs[0]);
                let in_dims = self.ir_shape_dims(n.inputs[0]);
                let mut starts = vec![0i64; in_dims.len()];
                let mut limits = in_dims.clone();
                let strides = vec![1i64; in_dims.len()];
                starts[*axis] = *start as i64;
                limits[*axis] = (*start + *len) as i64;
                self.entry.slice(x, &starts, &limits, &strides, out_shape)
            }
            Op::Concat { axis } => {
                let xs: Vec<i64> = n.inputs.iter().map(|&id| self.hlo(id)).collect();
                self.entry.concat(&xs, *axis as i64, out_shape)
            }
            Op::Expand { target_shape: _ } => {
                let x = self.hlo(n.inputs[0]);
                let in_dims = self.ir_shape_dims(n.inputs[0]);
                self.broadcast_to_target(x, &in_dims, out_shape)
            }
            Op::Gather { axis } => self.lower_gather(n.inputs[0], n.inputs[1], *axis, out_shape),

            Op::Reduce { op, axes, keep_dim } => {
                self.lower_reduce(n.inputs[0], *op, axes, *keep_dim, out_shape)
            }

            Op::Softmax { axis } => self.lower_softmax(n.inputs[0], *axis, out_shape),

            Op::Cumsum { axis, exclusive } => {
                self.lower_cumsum(n.inputs[0], *axis, *exclusive, out_shape)
            }

            Op::Fft { inverse, norm } => self.lower_fft(n.inputs[0], *inverse, *norm, out_shape),

            Op::Conv {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => self.lower_conv(
                n.inputs[0],
                n.inputs[1],
                kernel_size,
                stride,
                padding,
                dilation,
                *groups,
                out_shape,
            ),

            Op::Pool {
                kind,
                kernel_size,
                stride,
                padding,
            } => self.lower_pool(n.inputs[0], *kind, kernel_size, stride, padding, out_shape),

            Op::ScatterAdd => self.lower_scatter_add(n.inputs[0], n.inputs[1], out_shape),

            Op::ScatterNd { reduction } => {
                self.lower_scatter_nd(n.inputs[0], n.inputs[1], n.inputs[2], *reduction, out_shape)
            }

            Op::GatherNd { batch_dims } => {
                self.lower_gather_nd(n.inputs[0], n.inputs[1], *batch_dims, out_shape)
            }

            Op::ScatterElements { axis, reduction } => self.lower_scatter_elements(
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                *axis,
                *reduction,
                out_shape,
            ),

            Op::GatherElements { axis } => {
                self.lower_gather_elements(n.inputs[0], n.inputs[1], *axis, out_shape)
            }

            Op::TopK { k } => self.lower_topk(n.inputs[0], *k, out_shape),

            Op::GroupedMatMul => {
                self.lower_grouped_matmul(n.inputs[0], n.inputs[1], n.inputs[2], out_shape)
            }

            Op::DequantMatMul { scheme } if scheme.is_gguf() => {
                self.lower_dequant_matmul_gguf(n.inputs[0], n.inputs[1], *scheme, out_shape)
            }

            Op::DequantMatMul { scheme } => self.lower_dequant_matmul(
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                n.inputs[3],
                *scheme,
                out_shape,
            ),

            Op::QMatMul {
                x_zp,
                w_zp,
                out_zp,
                mult,
            } => self.lower_qmatmul(
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                *x_zp,
                *w_zp,
                *out_zp,
                *mult,
                out_shape,
            ),

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
            } => self.lower_qconv2d(
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                kernel_size,
                stride,
                padding,
                dilation,
                *groups,
                *x_zp,
                *w_zp,
                *out_zp,
                *mult,
                out_shape,
            ),

            Op::Sample {
                top_k,
                top_p,
                temperature,
                seed,
            } => self.lower_sample(n.inputs[0], *top_k, *top_p, *temperature, *seed, out_shape),

            Op::SelectiveScan { state_size } => self.lower_selective_scan(
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                n.inputs[3],
                n.inputs[4],
                *state_size,
                out_shape,
            ),

            Op::RngNormal {
                mean,
                scale,
                key: _,
                op_seed: _,
            } => self.lower_rng_normal(*mean, *scale, out_shape),

            Op::RngUniform {
                low,
                high,
                key: _,
                op_seed: _,
            } => self.lower_rng_uniform(*low, *high, out_shape),

            // Backward / training ops — no rlx-tpu support yet.
            Op::ReluBackward
            | Op::ActivationBackward { .. }
            | Op::MaxPool2dBackward { .. }
            | Op::Conv2dBackwardInput { .. }
            | Op::Conv2dBackwardWeight { .. }
            | Op::SoftmaxCrossEntropy
            | Op::SoftmaxCrossEntropyWithLogits
            | Op::SoftmaxCrossEntropyBackward
            | Op::LayerNormBackwardInput { .. }
            | Op::LayerNormBackwardGamma { .. }
            | Op::FakeQuantize { .. }
            | Op::FakeQuantizeBackward { .. }
            | Op::FakeQuantizeLSQ { .. }
            | Op::FakeQuantizeLSQBackwardX { .. }
            | Op::FakeQuantizeLSQBackwardScale { .. } => panic!(
                "rlx-tpu: training/backward op {:?} not supported — \
                 inference only.",
                n.op
            ),

            // Should have been removed by unfuse — reaching here is
            // a bug in the compile pipeline.
            Op::FusedSwiGLU { .. }
            | Op::LoraMatMul { .. }
            | Op::FusedAttentionBlock { .. }
            | Op::FusedTransformerLayer { .. }
            | Op::If { .. }
            | Op::While { .. } => panic!(
                "rlx-tpu: composed op {:?} should have been unfused \
                 before lowering — bug in pipeline.",
                n.op
            ),

            // Custom ops have no XLA/PJRT-side lowering today: PJRT
            // doesn't expose a `custom_call` we own, and the kernel
            // would need to be a separately-loaded XLA plugin. Reject
            // explicitly so the failure names the op rather than
            // bottoming out as an obscure HLO error.
            //
            // `collective.*` custom ops are the exception: they run as
            // host segments (see `crate::segment` / `crate::collective_host`),
            // so a graph containing them compiles via segmented
            // orchestration and never reaches whole-graph HLO lowering.
            // Reaching here with a collective op therefore means the
            // orchestration predicate (`segment::needs_orchestration`)
            // missed it — a bug — so name the op either way.
            Op::Custom { name, .. } if crate::segment::COLLECTIVE_OPS.contains(&name.as_str()) => {
                panic!(
                    "rlx-tpu: collective op '{name}' reached whole-graph HLO \
                     lowering — it must run as a host segment via segmented \
                     orchestration. This is a bug in \
                     `segment::needs_orchestration`.",
                )
            }
            Op::Custom { name, .. } => panic!(
                "rlx-tpu: Op::Custom('{name}') has no TPU lowering. \
                 Custom ops are CPU-only today; either move this op \
                 onto Device::Cpu or contribute an XLA-side lowering.",
            ),

            // DenseSolve is CPU-only today (uses LAPACK dgesv). No
            // XLA equivalent in our lowering yet.
            Op::DenseSolve => panic!(
                "rlx-tpu: Op::DenseSolve has no TPU lowering — \
                 use Device::Cpu for graphs containing dense solves.",
            ),

            // Op::Scan / ScanBackward* should be rewritten by legalize
            // (`LowerScan` + backward decompose) before HLO. If any escape,
            // pin the graph to Device::Cpu rather than silently mislowering.
            Op::Scan { .. } | Op::ScanBackward { .. } | Op::ScanBackwardXs { .. } => panic!(
                "rlx-tpu: Op::Scan / ScanBackward escaped legalize — \
                 expected LowerScan / backward decompose. Pin this graph \
                 to Device::Cpu, or ensure legalize_or_rewrite_for_backend runs."
            ),

            Op::BatchedDenseSolve | Op::CustomFn { .. } => panic!(
                "rlx-tpu: BatchedDenseSolve / CustomFn have no TPU lowering yet — \
                 use Device::Cpu.",
            ),

            Op::GaussianSplatRender { .. } | Op::GaussianSplatRenderBackward { .. } => panic!(
                "rlx-tpu: Gaussian splat ops are host-only; graphs containing \
                 them must compile via segmented orchestration (not whole-graph HLO)."
            ),

            _ => panic!("rlx-tpu: unsupported op {:?}", n.op),
        }
    }

    // ── Constants ──────────────────────────────────────────────

    pub(crate) fn lower_constant(&self, data: &[u8], shape: Shape, dt: DType) -> i64 {
        // Decode the bytes per dtype into the matching LiteralData
        // variant. Constants in rlx-ir are stored as native-endian
        // bytes, but on the platforms we run (Mac / Linux x86_64 /
        // aarch64) that's little-endian — same as proto wire.
        let n = (shape.num_elements() as usize).max(1);
        let lit = match dt {
            DType::F32 => {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&data[i * 4..i * 4 + 4]);
                    v.push(f32::from_le_bytes(b));
                }
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::F32(v),
                }
            }
            DType::F64 => {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&data[i * 8..i * 8 + 8]);
                    v.push(f64::from_le_bytes(b));
                }
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::F64(v),
                }
            }
            DType::F16 => Literal {
                shape: shape.clone(),
                data: LiteralData::F16Bytes(data.to_vec()),
            },
            DType::BF16 => Literal {
                shape: shape.clone(),
                data: LiteralData::BF16Bytes(data.to_vec()),
            },
            DType::I8 => Literal {
                shape: shape.clone(),
                data: LiteralData::S8Bytes(data.to_vec()),
            },
            DType::U8 => Literal {
                shape: shape.clone(),
                data: LiteralData::U8(data.to_vec()),
            },
            DType::Bool => Literal {
                shape: shape.clone(),
                data: LiteralData::Pred(data.to_vec()),
            },
            DType::I16 => {
                // s16s field uses raw bytes per upstream proto; emit as bytes.
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::S8Bytes(data.to_vec()),
                }
            }
            DType::I32 => {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&data[i * 4..i * 4 + 4]);
                    v.push(i32::from_le_bytes(b));
                }
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::S32(v),
                }
            }
            DType::I64 => {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&data[i * 8..i * 8 + 8]);
                    v.push(i64::from_le_bytes(b));
                }
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::S64(v),
                }
            }
            DType::U32 => {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&data[i * 4..i * 4 + 4]);
                    v.push(u32::from_le_bytes(b));
                }
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::U32(v),
                }
            }
            DType::C64 => {
                // Complex64 constants are stored as interleaved
                // [re, im, re, im, ...] f32 pairs (8 bytes / element),
                // which is exactly XLA's `c64s` wire layout.
                let mut v = Vec::with_capacity(n * 2);
                for i in 0..(n * 2) {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&data[i * 4..i * 4 + 4]);
                    v.push(f32::from_le_bytes(b));
                }
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::C64(v),
                }
            }
            DType::C128 => {
                // Complex128 constants are stored as interleaved
                // [re, im, re, im, ...] f64 pairs (16 bytes / element),
                // which is exactly XLA's `c128s` wire layout.
                let mut v = Vec::with_capacity(n * 2);
                for i in 0..(n * 2) {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&data[i * 8..i * 8 + 8]);
                    v.push(f64::from_le_bytes(b));
                }
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::C128(v),
                }
            }
        };
        self.entry.constant(lit)
    }

    // ── Cast ───────────────────────────────────────────────────

    /// Lower `Op::Cast`. Real↔real casts map straight onto XLA
    /// `convert`. Complex involvement uses the dedicated HLO ops
    /// (`complex` / `real`) so the semantics are unambiguous. The
    /// complex *component* type follows the width of the complex
    /// dtype: C64 → F32 components, C128 → F64 components.
    ///   * real → C64:  `complex(convert_f32(x), 0)` — imag = 0.
    ///   * real → C128: `complex(convert_f64(x), 0)` — imag = 0.
    ///   * C64 → real:  `real(x)` (F32), then convert to the requested
    ///     real dtype if it isn't already F32.
    ///   * C128 → real: `real(x)` (F64), then convert to the requested
    ///     real dtype if it isn't already F64.
    ///   * complex → complex: `convert` (covers C64↔C128 width change
    ///     and same-dtype identity).
    /// These match rlx's real↔complex cast semantics and XLA's
    /// `convert` complex handling (which upstream disallows going the
    /// complex direction, so we spell it out explicitly).
    pub(crate) fn lower_cast(&self, x: i64, from: DType, to: DType, dims: &[i64]) -> i64 {
        // Real component (prim + rlx DType) of a complex dtype.
        fn complex_component(dt: DType) -> (i32, DType) {
            match dt {
                DType::C64 => (prim::F32, DType::F32),
                DType::C128 => (prim::F64, DType::F64),
                _ => unreachable!("complex_component on non-complex {dt:?}"),
            }
        }
        match (from.is_complex(), to.is_complex()) {
            // real → real: plain elementwise convert (handles every
            // scalar pair — XLA `convert` covers int/float/bool).
            (false, false) => self.entry.convert(x, Shape::array(prim_of(to), dims)),
            // real → complex: XLA `complex(real, imag)` needs real
            // operands of the target's component width (F32 for C64,
            // F64 for C128); convert the source, then pair with a zero
            // imaginary part.
            (false, true) => {
                let (comp_prim, comp_dt) = complex_component(to);
                let re = if from == comp_dt {
                    x
                } else {
                    self.entry.convert(x, Shape::array(comp_prim, dims))
                };
                let zero = if comp_dt == DType::F64 {
                    self.entry.constant_f64_scalar(0.0)
                } else {
                    self.entry.constant_f32_scalar(0.0)
                };
                let imag = self.entry.broadcast(zero, &[], Shape::array(comp_prim, dims));
                self.entry
                    .binary("complex", re, imag, Shape::array(prim_of(to), dims))
            }
            // complex → real: take the real part (source's component
            // width), then convert to the requested real dtype when it
            // differs from that component type.
            (true, false) => {
                let (comp_prim, comp_dt) = complex_component(from);
                let re = self.entry.unary("real", x, Shape::array(comp_prim, dims));
                if to == comp_dt {
                    re
                } else {
                    self.entry.convert(re, Shape::array(prim_of(to), dims))
                }
            }
            // complex → complex: `convert` covers same-dtype identity
            // and C64↔C128 component-width changes.
            (true, true) => self.entry.convert(x, Shape::array(prim_of(to), dims)),
        }
    }

    // ── Activation ─────────────────────────────────────────────

    pub(crate) fn lower_activation(&self, act: Activation, x: i64, shape: Shape) -> i64 {
        let elt = shape.element_type;
        match act {
            // Direct HLO unary opcodes.
            Activation::Exp => self.entry.unary("exponential", x, shape),
            Activation::Log => self.entry.unary("log", x, shape),
            Activation::Sqrt => self.entry.unary("sqrt", x, shape),
            Activation::Rsqrt => self.entry.unary("rsqrt", x, shape),
            Activation::Neg => self.entry.unary("negate", x, shape),
            Activation::Abs => self.entry.unary("abs", x, shape),
            Activation::Round => self.entry.unary("round-nearest-even", x, shape),
            Activation::Sin => self.entry.unary("sine", x, shape),
            Activation::Cos => self.entry.unary("cosine", x, shape),
            Activation::Tanh => self.entry.unary("tanh", x, shape),
            // sigmoid(x) → HLO `logistic`.
            Activation::Sigmoid => self.entry.unary("logistic", x, shape),
            // silu(x) = x * sigmoid(x).
            Activation::Silu => {
                let s = self.entry.unary("logistic", x, shape.clone());
                self.entry.binary("multiply", x, s, shape)
            }
            // relu(x) = max(x, 0).
            Activation::Relu => {
                let zero = self.entry.constant(Literal {
                    shape: Shape::scalar(elt),
                    data: LiteralData::F32(vec![0.0]), // ignored for non-f32; see below
                });
                // For non-f32 element types, use a typed scalar zero
                // by converting f32 0.0 through `convert`.
                let zero = if elt == prim::F32 {
                    zero
                } else {
                    self.entry.convert(zero, Shape::scalar(elt))
                };
                let zero_b = self.entry.broadcast(zero, &[], shape.clone());
                self.entry.binary("maximum", x, zero_b, shape)
            }
            // GELU exact: 0.5 * x * (1 + erf(x / sqrt(2))).
            Activation::Gelu => {
                let half = self.const_in_dtype(elt, 0.5);
                let one = self.const_in_dtype(elt, 1.0);
                let inv_sqrt_2 = self.const_in_dtype(elt, std::f32::consts::FRAC_1_SQRT_2);
                let half_b = self.entry.broadcast(half, &[], shape.clone());
                let one_b = self.entry.broadcast(one, &[], shape.clone());
                let inv_b = self.entry.broadcast(inv_sqrt_2, &[], shape.clone());
                let scaled = self.entry.binary("multiply", x, inv_b, shape.clone());
                let erfed = self.entry.unary("erf", scaled, shape.clone());
                let one_plus = self.entry.binary("add", one_b, erfed, shape.clone());
                let half_x = self.entry.binary("multiply", x, half_b, shape.clone());
                self.entry.binary("multiply", half_x, one_plus, shape)
            }
            // GELU approx (tanh form):
            // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
            Activation::GeluApprox => {
                let half = self.const_in_dtype(elt, 0.5);
                let one = self.const_in_dtype(elt, 1.0);
                let c = self.const_in_dtype(elt, (2.0_f32 / std::f32::consts::PI).sqrt());
                let k = self.const_in_dtype(elt, 0.044715);
                let half_b = self.entry.broadcast(half, &[], shape.clone());
                let one_b = self.entry.broadcast(one, &[], shape.clone());
                let c_b = self.entry.broadcast(c, &[], shape.clone());
                let k_b = self.entry.broadcast(k, &[], shape.clone());
                let x2 = self.entry.binary("multiply", x, x, shape.clone());
                let x3 = self.entry.binary("multiply", x2, x, shape.clone());
                let kx3 = self.entry.binary("multiply", k_b, x3, shape.clone());
                let inner = self.entry.binary("add", x, kx3, shape.clone());
                let scaled = self.entry.binary("multiply", c_b, inner, shape.clone());
                let tanhed = self.entry.unary("tanh", scaled, shape.clone());
                let one_plus = self.entry.binary("add", one_b, tanhed, shape.clone());
                let half_x = self.entry.binary("multiply", x, half_b, shape.clone());
                self.entry.binary("multiply", half_x, one_plus, shape)
            }
            Activation::Tan => self.entry.unary("tan", x, shape),
            Activation::Atan => self.entry.unary("atan", x, shape),
        }
    }

    pub(crate) fn lower_binary(
        &self,
        op: BinaryOp,
        a: i64,
        b: i64,
        a_id: NodeId,
        b_id: NodeId,
        out: Shape,
    ) -> i64 {
        let opcode = match op {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "subtract",
            BinaryOp::Mul => "multiply",
            BinaryOp::Div => "divide",
            BinaryOp::Max => "maximum",
            BinaryOp::Min => "minimum",
            BinaryOp::Pow => "power",
        };
        let (a, b) = self.broadcast_pair_to(a, b, a_id, b_id, &out.dimensions);
        self.entry.binary(opcode, a, b, out)
    }

    pub(crate) fn lower_elementwise_region(
        &mut self,
        inputs: &[NodeId],
        chain: &[ChainStep],
        num_inputs: u32,
        scalar_input_mask: u32,
        input_modulus: &[u32; 16],
        out_shape: Shape,
        prologue: RegionPrologue,
        prologue_input: u32,
    ) -> i64 {
        // Walk the chain, materializing each step as a regular HLO
        // op. ChainOperand::Input(i) refers to inputs[i] (broadcast
        // to output shape if scalar/tiled). ChainOperand::Step(i)
        // refers to the i-th already-emitted step result.
        let n = num_inputs as usize;
        let mut input_hlo: Vec<i64> = Vec::with_capacity(n);
        for i in 0..n {
            let id = self.hlo(inputs[i]);
            let in_dims = self.ir_shape_dims(inputs[i]);
            let in_dt = self.dtype(inputs[i]);
            let target = Shape::array(prim_of(in_dt), &out_shape.dimensions);
            let scalar = scalar_input_mask & (1u32 << i) != 0;
            let _ = input_modulus[i]; // tiling fully captured by broadcast_to_target
            let placed = if scalar {
                self.entry.broadcast(id, &[], target)
            } else {
                self.broadcast_to_target(id, &in_dims, target)
            };
            input_hlo.push(placed);
        }
        if prologue == RegionPrologue::ResizeNearest2x {
            let pi = prologue_input as usize;
            if pi >= n {
                panic!(
                    "rlx-tpu ElementwiseRegion: prologue_input={pi} out of range (num_inputs={n})"
                );
            }
            let up_shape = self.resize_nearest_2x_shape(inputs[pi]);
            input_hlo[pi] = self.lower_resize_nearest_2x_nchw(input_hlo[pi], inputs[pi], up_shape);
        }
        let mut step_results: Vec<i64> = Vec::with_capacity(chain.len());

        let resolve = |op: &ChainOperand, ins: &[i64], steps: &[i64]| -> i64 {
            match op {
                ChainOperand::Input(i) => ins[*i as usize],
                ChainOperand::Step(i) => steps[*i as usize],
            }
        };
        for step in chain {
            let result = match step {
                ChainStep::Activation(act, src) => {
                    let x = resolve(src, &input_hlo, &step_results);
                    self.lower_activation(*act, x, out_shape.clone())
                }
                ChainStep::Cast(dt, src) => {
                    let x = resolve(src, &input_hlo, &step_results);
                    self.entry
                        .convert(x, Shape::array(prim_of(*dt), &out_shape.dimensions))
                }
                ChainStep::Binary(op, lhs, rhs) => {
                    let a = resolve(lhs, &input_hlo, &step_results);
                    let b = resolve(rhs, &input_hlo, &step_results);
                    let opcode = match op {
                        BinaryOp::Add => "add",
                        BinaryOp::Sub => "subtract",
                        BinaryOp::Mul => "multiply",
                        BinaryOp::Div => "divide",
                        BinaryOp::Max => "maximum",
                        BinaryOp::Min => "minimum",
                        BinaryOp::Pow => "power",
                    };
                    self.entry.binary(opcode, a, b, out_shape.clone())
                }
                ChainStep::Compare(op, lhs, rhs) => {
                    let a = resolve(lhs, &input_hlo, &step_results);
                    let b = resolve(rhs, &input_hlo, &step_results);
                    let dir = match op {
                        CmpOp::Eq => "EQ",
                        CmpOp::Ne => "NE",
                        CmpOp::Lt => "LT",
                        CmpOp::Le => "LE",
                        CmpOp::Gt => "GT",
                        CmpOp::Ge => "GE",
                    };
                    self.entry
                        .compare(a, b, dir, Shape::pred(&out_shape.dimensions))
                }
                ChainStep::Where(c, t, f) => {
                    let cv = resolve(c, &input_hlo, &step_results);
                    let tv = resolve(t, &input_hlo, &step_results);
                    let fv = resolve(f, &input_hlo, &step_results);
                    self.entry.select(cv, tv, fv, out_shape.clone())
                }
            };
            step_results.push(result);
        }
        // Output is the last step's result.
        *step_results.last().unwrap_or(&0)
    }

    pub(crate) fn lower_resize_nearest_2x_nchw(
        &mut self,
        x: i64,
        input_id: NodeId,
        out: Shape,
    ) -> i64 {
        let dims = self.ir_shape_dims(input_id);
        assert_eq!(dims.len(), 4);
        let dt = prim_of(self.dtype(input_id));
        let rank6 = vec![dims[0], dims[1], dims[2], dims[3], 1, 1];
        let r1 = self.entry.reshape(x, Shape::array(dt, &rank6));
        let rank6_out = vec![dims[0], dims[1], dims[2], dims[3], 2, 2];
        let bcast_dims: Vec<i64> = (0..4).collect();
        let r2 = self
            .entry
            .broadcast(r1, &bcast_dims, Shape::array(dt, &rank6_out));
        self.entry.reshape(r2, out)
    }

    pub(crate) fn lower_transform_region(
        &mut self,
        inputs: &[NodeId],
        steps: &[TransformStep],
        num_inputs: u32,
        out_shape: Shape,
    ) -> i64 {
        let n = num_inputs as usize;
        if n == 0 || inputs.is_empty() {
            panic!("rlx-tpu TransformRegion: no inputs");
        }
        let mut cur = self.hlo(inputs[0]);
        for step in steps {
            match step {
                TransformStep::ResizeNearest2x(src) => {
                    let (x, id) = match src {
                        ChainOperand::Input(i) => {
                            let idx = *i as usize;
                            if idx >= inputs.len() {
                                panic!("rlx-tpu TransformRegion: input index {idx} out of range");
                            }
                            (self.hlo(inputs[idx]), inputs[idx])
                        }
                        ChainOperand::Step(_) => {
                            panic!("rlx-tpu TransformRegion: step operands are not supported");
                        }
                    };
                    let up = self.resize_nearest_2x_shape(id);
                    cur = self.lower_resize_nearest_2x_nchw(x, id, up);
                }
            }
        }
        let _ = out_shape;
        cur
    }

    pub(crate) fn lower_batch_elementwise_region(
        &mut self,
        inputs: &[NodeId],
        chain: &[ChainStep],
        num_batch: u32,
        scalar_input_mask: u32,
        input_modulus: &[u32; 16],
        prologue: RegionPrologue,
        prologue_input: u32,
        batch_out: &rlx_ir::Shape,
        out_shape: Shape,
    ) -> i64 {
        let n = num_batch as usize;
        if inputs.len() != n {
            panic!(
                "rlx-tpu BatchElementwiseRegion: declared {n} batch inputs but node has {}",
                inputs.len()
            );
        }
        let slice_shape = rlx_ir::batch_region_slice_shape(batch_out);
        let slice_dims: Vec<i64> = (0..slice_shape.rank())
            .map(|d| slice_shape.dim(d).unwrap_static() as i64)
            .collect();
        let slice_out = Shape::array(prim_of(batch_out.dtype()), &slice_dims);
        let mut slices = Vec::with_capacity(n);
        for &in_id in inputs {
            slices.push(self.lower_elementwise_region(
                std::slice::from_ref(&in_id),
                chain,
                1,
                scalar_input_mask,
                input_modulus,
                slice_out.clone(),
                prologue,
                prologue_input,
            ));
        }
        self.entry.concat(&slices, 0, out_shape)
    }

    // ── MatMul ─────────────────────────────────────────────────

    pub(crate) fn lower_matmul(&mut self, a_id: NodeId, b_id: NodeId, out: Shape) -> i64 {
        let a = self.hlo(a_id);
        let b = self.hlo(b_id);
        let a_dims = self.ir_shape_dims(a_id);
        let b_dims = self.ir_shape_dims(b_id);
        // [..., M, K] × [..., K, N] → [..., M, N] with batch dims
        // broadcast. For HLO, the cleanest expression is:
        //   contracting: lhs=last, rhs=second_to_last
        //   batch: leading dims (must match exactly; we materialize
        //          a broadcast first if they don't).
        let a_rank = a_dims.len();
        let b_rank = b_dims.len();
        let (max_rank, a_b, b_b) = match (a_rank, b_rank) {
            (r, s) if r == s => (r, a, b),
            (r, s) if r < s => {
                // Add leading 1s to a, then broadcast.
                let pad = s - r;
                let mut tgt = vec![1i64; pad];
                tgt.extend_from_slice(&a_dims);
                let r1 = self.entry.reshape(a,
                    Shape::array(prim_of(self.dtype(a_id)), &tgt));
                let mut full = b_dims[..pad].to_vec();
                full.extend_from_slice(&a_dims);
                let r2 = self.broadcast_to_target(r1, &tgt,
                    Shape::array(prim_of(self.dtype(a_id)), &full));
                (s, r2, b)
            }
            (r, s) /* r > s */ => {
                let pad = r - s;
                let mut tgt = vec![1i64; pad];
                tgt.extend_from_slice(&b_dims);
                let r1 = self.entry.reshape(b,
                    Shape::array(prim_of(self.dtype(b_id)), &tgt));
                let mut full = a_dims[..pad].to_vec();
                full.extend_from_slice(&b_dims);
                let r2 = self.broadcast_to_target(r1, &tgt,
                    Shape::array(prim_of(self.dtype(b_id)), &full));
                (r, a, r2)
            }
        };
        let contracting_a = (max_rank - 1) as i64;
        let contracting_b = (max_rank - 2) as i64;
        let batch: Vec<i64> = (0..max_rank as i64 - 2).collect();
        let dn = DotDimNumbers {
            lhs_contracting: vec![contracting_a],
            rhs_contracting: vec![contracting_b],
            lhs_batch: batch.clone(),
            rhs_batch: batch,
        };
        self.entry.dot_general(a_b, b_b, dn, out)
    }

    // ── LayerNorm ──────────────────────────────────────────────

    pub(crate) fn lower_layernorm(
        &mut self,
        x_id: NodeId,
        gamma_id: NodeId,
        beta_id: NodeId,
        axis: i32,
        eps: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let gamma = self.hlo(gamma_id);
        let beta = self.hlo(beta_id);
        let x_dims = self.ir_shape_dims(x_id);
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);
        let rank = x_dims.len();
        let ax = if axis < 0 {
            (rank as i32 + axis) as i64
        } else {
            axis as i64
        };

        let mut reduced = x_dims.clone();
        reduced[ax as usize] = 1;
        let kept_shape = Shape::array(prim_ty, &reduced);

        // mean = sum(x) / N
        let summed = self.reduce_one(
            x,
            ax,
            "add",
            0.0,
            x_dt,
            x_dims
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
                .collect(),
        );
        let n = x_dims[ax as usize] as f32;
        let n_c = self.const_in_dtype(prim_ty, n);
        let summed_shape_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let summed_shape = Shape::array(prim_ty, &summed_shape_dims);
        let n_b = self.entry.broadcast(n_c, &[], summed_shape.clone());
        let mean = self
            .entry
            .binary("divide", summed, n_b, summed_shape.clone());
        // Reshape to keep the reduced axis as size 1, then broadcast back.
        let mean_kept = self.entry.reshape(mean, kept_shape.clone());
        let mean_b = self.broadcast_align(mean_kept, &reduced, out.clone());

        // centered = x - mean
        let centered = self.entry.binary("subtract", x, mean_b, out.clone());
        let sq = self
            .entry
            .binary("multiply", centered, centered, out.clone());

        // var = sum(sq) / N
        let var_summed = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_shape_dims.clone());
        let var = self
            .entry
            .binary("divide", var_summed, n_b, summed_shape.clone());
        let var_kept = self.entry.reshape(var, kept_shape);

        let eps_c = self.const_in_dtype(prim_ty, eps);
        let var_eps_kept_shape = Shape::array(prim_ty, &reduced);
        let eps_b = self.entry.broadcast(eps_c, &[], var_eps_kept_shape.clone());
        let var_eps = self
            .entry
            .binary("add", var_kept, eps_b, var_eps_kept_shape.clone());
        let inv_std = self.entry.unary("rsqrt", var_eps, var_eps_kept_shape);
        let inv_std_b = self.broadcast_align(inv_std, &reduced, out.clone());

        let normed = self
            .entry
            .binary("multiply", centered, inv_std_b, out.clone());

        // scaled = normed * gamma + beta (gamma/beta have axis-only shape).
        let g_dims = self.ir_shape_dims(gamma_id);
        let b_dims = self.ir_shape_dims(beta_id);
        let g_b = self.broadcast_param_to_axis(gamma, &g_dims, ax, &x_dims, prim_ty);
        let b_b = self.broadcast_param_to_axis(beta, &b_dims, ax, &x_dims, prim_ty);
        let scaled = self.entry.binary("multiply", normed, g_b, out.clone());
        self.entry.binary("add", scaled, b_b, out)
    }

    pub(crate) fn lower_rmsnorm(
        &mut self,
        x_id: NodeId,
        gamma_id: NodeId,
        _beta_id: NodeId,
        axis: i32,
        eps: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let gamma = self.hlo(gamma_id);
        let x_dims = self.ir_shape_dims(x_id);
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);
        let rank = x_dims.len();
        let ax = if axis < 0 {
            (rank as i32 + axis) as i64
        } else {
            axis as i64
        };

        let mut reduced = x_dims.clone();
        reduced[ax as usize] = 1;
        let summed_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let kept_shape = Shape::array(prim_ty, &reduced);
        let summed_shape = Shape::array(prim_ty, &summed_dims);

        let sq = self.entry.binary("multiply", x, x, out.clone());
        let sq_sum = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_dims.clone());
        let n = x_dims[ax as usize] as f32;
        let n_c = self.const_in_dtype(prim_ty, n);
        let n_b = self.entry.broadcast(n_c, &[], summed_shape.clone());
        let sq_mean = self
            .entry
            .binary("divide", sq_sum, n_b, summed_shape.clone());

        let eps_c = self.const_in_dtype(prim_ty, eps);
        let eps_b = self.entry.broadcast(eps_c, &[], summed_shape.clone());
        let var_eps = self
            .entry
            .binary("add", sq_mean, eps_b, summed_shape.clone());
        let inv = self.entry.unary("rsqrt", var_eps, summed_shape);
        let inv_kept = self.entry.reshape(inv, kept_shape);
        let inv_b = self.broadcast_align(inv_kept, &reduced, out.clone());
        let normed = self.entry.binary("multiply", x, inv_b, out.clone());
        let g_dims = self.ir_shape_dims(gamma_id);
        let g_b = self.broadcast_param_to_axis(gamma, &g_dims, ax, &x_dims, prim_ty);
        self.entry.binary("multiply", normed, g_b, out)
    }

    // ── FusedResidualLN ────────────────────────────────────────

    pub(crate) fn lower_fused_residual_ln(
        &mut self,
        inputs: &[NodeId],
        has_bias: bool,
        eps: f32,
        out: Shape,
    ) -> i64 {
        // inputs: [x, residual, [bias], gamma, beta]
        let x = self.hlo(inputs[0]);
        let r = self.hlo(inputs[1]);
        let summed = self.entry.binary("add", x, r, out.clone());
        let pre_ln = if has_bias {
            let b = self.hlo(inputs[2]);
            let b_dims = self.ir_shape_dims(inputs[2]);
            let target = out.clone();
            let b_b = self.broadcast_to_target(b, &b_dims, target);
            self.entry.binary("add", summed, b_b, out.clone())
        } else {
            summed
        };
        let (gi, bi) = if has_bias { (3, 4) } else { (2, 3) };
        let gamma_id = inputs[gi];
        let beta_id = inputs[bi];
        // Synthesize a temporary IR-less LayerNorm by going through
        // lower_layernorm's mechanics directly. We compute mean / var
        // over axis -1 (matches all CPU/Metal/CUDA emitters).
        let x_dims = out.dimensions.clone();
        let x_dt = self.graph.node(inputs[0]).shape.dtype();
        let prim_ty = prim_of(x_dt);
        let rank = x_dims.len();
        let ax = (rank - 1) as i64;
        let mut reduced = x_dims.clone();
        reduced[ax as usize] = 1;
        let summed_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let kept_shape = Shape::array(prim_ty, &reduced);
        let summed_shape = Shape::array(prim_ty, &summed_dims);

        let pre_sum = self.reduce_one(pre_ln, ax, "add", 0.0, x_dt, summed_dims.clone());
        let n = x_dims[ax as usize] as f32;
        let n_c = self.const_in_dtype(prim_ty, n);
        let n_b = self.entry.broadcast(n_c, &[], summed_shape.clone());
        let mean = self
            .entry
            .binary("divide", pre_sum, n_b, summed_shape.clone());
        let mean_kept = self.entry.reshape(mean, kept_shape.clone());
        let mean_b = self.broadcast_align(mean_kept, &reduced, out.clone());
        let centered = self.entry.binary("subtract", pre_ln, mean_b, out.clone());
        let sq = self
            .entry
            .binary("multiply", centered, centered, out.clone());
        let sq_sum = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_dims);
        let var = self
            .entry
            .binary("divide", sq_sum, n_b, summed_shape.clone());
        let var_kept = self.entry.reshape(var, kept_shape);
        let eps_c = self.const_in_dtype(prim_ty, eps);
        let eps_b = self
            .entry
            .broadcast(eps_c, &[], Shape::array(prim_ty, &reduced));
        let var_eps = self
            .entry
            .binary("add", var_kept, eps_b, Shape::array(prim_ty, &reduced));
        let inv_std = self
            .entry
            .unary("rsqrt", var_eps, Shape::array(prim_ty, &reduced));
        let inv_std_b = self.broadcast_align(inv_std, &reduced, out.clone());
        let normed = self
            .entry
            .binary("multiply", centered, inv_std_b, out.clone());

        let gamma = self.hlo(gamma_id);
        let beta = self.hlo(beta_id);
        let g_dims = self.ir_shape_dims(gamma_id);
        let b_dims = self.ir_shape_dims(beta_id);
        let g_b = self.broadcast_param_to_axis(gamma, &g_dims, ax, &x_dims, prim_ty);
        let b_b = self.broadcast_param_to_axis(beta, &b_dims, ax, &x_dims, prim_ty);
        let scaled = self.entry.binary("multiply", normed, g_b, out.clone());
        self.entry.binary("add", scaled, b_b, out)
    }

    // ── AdaLayerNorm / GatedResidual ───────────────────────────

    pub(crate) fn lower_ada_layer_norm(
        &mut self,
        inputs: &[NodeId],
        norm: AdaNormKind,
        eps: f32,
        out: Shape,
    ) -> i64 {
        let x_id = inputs[0];
        let scale_id = inputs[1];
        let shift_id = inputs[2];
        let x = self.hlo(x_id);
        let x_dims = out.dimensions.clone();
        let x_dt = self.graph.node(x_id).shape.dtype();
        let prim_ty = prim_of(x_dt);
        let rank = x_dims.len();
        let ax = (rank - 1) as i64;
        let mut reduced = x_dims.clone();
        reduced[ax as usize] = 1;
        let summed_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let kept_shape = Shape::array(prim_ty, &reduced);
        let summed_shape = Shape::array(prim_ty, &summed_dims);
        let n_elems = x_dims[ax as usize] as f32;
        let n_c = self.const_in_dtype(prim_ty, n_elems);
        let n_b = self.entry.broadcast(n_c, &[], summed_shape.clone());

        let n = match norm {
            AdaNormKind::LayerNorm => {
                let pre_sum = self.reduce_one(x, ax, "add", 0.0, x_dt, summed_dims.clone());
                let mean = self
                    .entry
                    .binary("divide", pre_sum, n_b, summed_shape.clone());
                let mean_kept = self.entry.reshape(mean, kept_shape.clone());
                let mean_b = self.broadcast_align(mean_kept, &reduced, out.clone());
                let centered = self.entry.binary("subtract", x, mean_b, out.clone());
                let sq = self
                    .entry
                    .binary("multiply", centered, centered, out.clone());
                let sq_sum = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_dims);
                let var = self
                    .entry
                    .binary("divide", sq_sum, n_b, summed_shape.clone());
                let var_kept = self.entry.reshape(var, kept_shape);
                let eps_c = self.const_in_dtype(prim_ty, eps);
                let eps_b = self
                    .entry
                    .broadcast(eps_c, &[], Shape::array(prim_ty, &reduced));
                let var_eps =
                    self.entry
                        .binary("add", var_kept, eps_b, Shape::array(prim_ty, &reduced));
                let inv_std = self
                    .entry
                    .unary("rsqrt", var_eps, Shape::array(prim_ty, &reduced));
                let inv_std_b = self.broadcast_align(inv_std, &reduced, out.clone());
                self.entry
                    .binary("multiply", centered, inv_std_b, out.clone())
            }
            AdaNormKind::RmsNorm => {
                let sq = self.entry.binary("multiply", x, x, out.clone());
                let sq_sum = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_dims);
                let sq_mean = self
                    .entry
                    .binary("divide", sq_sum, n_b, summed_shape.clone());
                let eps_c = self.const_in_dtype(prim_ty, eps);
                let eps_b = self.entry.broadcast(eps_c, &[], summed_shape.clone());
                let var_eps = self
                    .entry
                    .binary("add", sq_mean, eps_b, summed_shape.clone());
                let inv = self.entry.unary("rsqrt", var_eps, summed_shape);
                let inv_kept = self.entry.reshape(inv, kept_shape);
                let inv_b = self.broadcast_align(inv_kept, &reduced, out.clone());
                self.entry.binary("multiply", x, inv_b, out.clone())
            }
        };

        let scale = self.hlo(scale_id);
        let shift = self.hlo(shift_id);
        let scale_dims = self.ir_shape_dims(scale_id);
        let shift_dims = self.ir_shape_dims(shift_id);
        let scale_b = self.broadcast_to_target(scale, &scale_dims, out.clone());
        let shift_b = self.broadcast_to_target(shift, &shift_dims, out.clone());
        let n_scale = self.entry.binary("multiply", n, scale_b, out.clone());
        let m = self.entry.binary("add", n, n_scale, out.clone());
        self.entry.binary("add", m, shift_b, out)
    }

    pub(crate) fn lower_gated_residual(&mut self, inputs: &[NodeId], out: Shape) -> i64 {
        let x = self.hlo(inputs[0]);
        let y = self.hlo(inputs[1]);
        let gate = self.hlo(inputs[2]);
        let gate_dims = self.ir_shape_dims(inputs[2]);
        let gate_b = self.broadcast_to_target(gate, &gate_dims, out.clone());
        let gy = self.entry.binary("multiply", gate_b, y, out.clone());
        self.entry.binary("add", x, gy, out)
    }

    /// Sum broadcast axes of `grad` (shape `full_dims`) down to `target_dims`.
    pub(crate) fn sum_unbroadcast_hlo(
        &mut self,
        grad: i64,
        full_dims: &[i64],
        target_dims: &[i64],
        dt: DType,
    ) -> i64 {
        if full_dims == target_dims {
            return grad;
        }
        let prim_ty = prim_of(dt);
        let g_rank = full_dims.len();
        let t_rank = target_dims.len();
        let extra = g_rank.saturating_sub(t_rank);
        let mut axes: Vec<usize> = (0..extra).collect();
        for i in 0..t_rank {
            if target_dims[i] == 1 && full_dims[extra + i] > 1 {
                axes.push(extra + i);
            }
        }
        let mut current = grad;
        let mut running_dims = full_dims.to_vec();
        for &ax in &axes {
            running_dims[ax] = 1;
            let kept_shape = Shape::array(prim_ty, &running_dims);
            let collapsed_dims: Vec<i64> = running_dims
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| if i == ax { None } else { Some(d) })
                .collect();
            let collapsed_shape = Shape::array(prim_ty, &collapsed_dims);
            let red = self.reducer("add", prim_ty);
            let init = self.const_in_dtype(prim_ty, 0.0);
            let reduced = self
                .entry
                .reduce(current, init, &red, &[ax as i64], collapsed_shape);
            current = self.entry.reshape(reduced, kept_shape);
        }
        self.entry
            .reshape(current, Shape::array(prim_ty, target_dims))
    }

    /// Flatten each grad to 1-D and concat on axis 0 (packed DiT reverse layout).
    pub(crate) fn pack_flat_grads_hlo(&mut self, grads: &[(i64, &[i64])], out: Shape) -> i64 {
        let prim_ty = out.element_type;
        let mut flats = Vec::with_capacity(grads.len());
        for (hlo, dims) in grads {
            let n: i64 = dims.iter().product();
            let flat_shape = Shape::array(prim_ty, &[n]);
            flats.push(self.entry.reshape(*hlo, flat_shape));
        }
        self.entry.concat(&flats, 0, out)
    }

    /// LayerNorm input reverse with γ=1 (`sy = dn` HLO).
    pub(crate) fn lower_layernorm_dx_gamma1_hlo(
        &mut self,
        x: i64,
        dn: i64,
        eps: f32,
        out: Shape,
        x_dt: DType,
    ) -> i64 {
        let x_dims = out.dimensions.clone();
        let prim_ty = prim_of(x_dt);
        let rank = x_dims.len();
        let ax = (rank - 1) as i64;
        let mut reduced = x_dims.clone();
        reduced[ax as usize] = 1;
        let kept_shape = Shape::array(prim_ty, &reduced);
        let summed_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let summed_shape = Shape::array(prim_ty, &summed_dims);
        let n_elems = x_dims[ax as usize] as f32;
        let n_c = self.const_in_dtype(prim_ty, n_elems);
        let n_b = self.entry.broadcast(n_c, &[], summed_shape.clone());

        let pre_sum = self.reduce_one(x, ax, "add", 0.0, x_dt, summed_dims.clone());
        let mean = self
            .entry
            .binary("divide", pre_sum, n_b, summed_shape.clone());
        let mean_kept = self.entry.reshape(mean, kept_shape.clone());
        let mean_b = self.broadcast_align(mean_kept, &reduced, out.clone());
        let centered = self.entry.binary("subtract", x, mean_b, out.clone());
        let sq = self
            .entry
            .binary("multiply", centered, centered, out.clone());
        let sq_sum = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_dims.clone());
        let var = self
            .entry
            .binary("divide", sq_sum, n_b, summed_shape.clone());
        let var_kept = self.entry.reshape(var, kept_shape.clone());
        let eps_c = self.const_in_dtype(prim_ty, eps);
        let eps_b = self
            .entry
            .broadcast(eps_c, &[], Shape::array(prim_ty, &reduced));
        let var_eps = self
            .entry
            .binary("add", var_kept, eps_b, Shape::array(prim_ty, &reduced));
        let inv_std = self
            .entry
            .unary("rsqrt", var_eps, Shape::array(prim_ty, &reduced));
        let inv_std_b = self.broadcast_align(inv_std, &reduced, out.clone());
        let xhat = self
            .entry
            .binary("multiply", centered, inv_std_b, out.clone());

        let m_sy = self.reduce_one(dn, ax, "add", 0.0, x_dt, summed_dims.clone());
        let m_sy_div = self.entry.binary("divide", m_sy, n_b, summed_shape.clone());
        let m_sy_kept = self.entry.reshape(m_sy_div, kept_shape.clone());
        let m_sy_b = self.broadcast_align(m_sy_kept, &reduced, out.clone());
        let sy_xh = self.entry.binary("multiply", dn, xhat, out.clone());
        let m_sxh = self.reduce_one(sy_xh, ax, "add", 0.0, x_dt, summed_dims);
        let m_sxh_div = self.entry.binary("divide", m_sxh, n_b, summed_shape);
        let m_sxh_kept = self.entry.reshape(m_sxh_div, kept_shape);
        let m_sxh_b = self.broadcast_align(m_sxh_kept, &reduced, out.clone());
        let term1 = self.entry.binary("subtract", dn, m_sy_b, out.clone());
        let term2 = self.entry.binary("multiply", xhat, m_sxh_b, out.clone());
        let inner = self.entry.binary("subtract", term1, term2, out.clone());
        self.entry.binary("multiply", inv_std_b, inner, out)
    }

    /// RMSNorm input reverse with γ=1 (`sy = dn` HLO).
    pub(crate) fn lower_rms_norm_dx_gamma1_hlo(
        &mut self,
        x: i64,
        dn: i64,
        eps: f32,
        out: Shape,
        x_dt: DType,
    ) -> i64 {
        let x_dims = out.dimensions.clone();
        let prim_ty = prim_of(x_dt);
        let rank = x_dims.len();
        let ax = (rank - 1) as i64;
        let mut reduced = x_dims.clone();
        reduced[ax as usize] = 1;
        let kept_shape = Shape::array(prim_ty, &reduced);
        let summed_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let summed_shape = Shape::array(prim_ty, &summed_dims);
        let n_elems = x_dims[ax as usize] as f32;
        let n_c = self.const_in_dtype(prim_ty, n_elems);
        let n_b = self.entry.broadcast(n_c, &[], summed_shape.clone());

        let sq = self.entry.binary("multiply", x, x, out.clone());
        let sq_sum = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_dims.clone());
        let sq_mean = self
            .entry
            .binary("divide", sq_sum, n_b, summed_shape.clone());
        let eps_c = self.const_in_dtype(prim_ty, eps);
        let eps_b = self.entry.broadcast(eps_c, &[], summed_shape.clone());
        let var_eps = self
            .entry
            .binary("add", sq_mean, eps_b, summed_shape.clone());
        let inv_r = self.entry.unary("rsqrt", var_eps, summed_shape.clone());
        let inv_r_kept = self.entry.reshape(inv_r, kept_shape.clone());
        let inv_r_b = self.broadcast_align(inv_r_kept, &reduced, out.clone());
        let inv_r2 = self.entry.binary("multiply", inv_r_b, inv_r_b, out.clone());
        let dy_gx = self.entry.binary("multiply", dn, x, out.clone());
        let dot = self.reduce_one(dy_gx, ax, "add", 0.0, x_dt, summed_dims);
        let dot_div = self.entry.binary("divide", dot, n_b, summed_shape);
        let dot_kept = self.entry.reshape(dot_div, kept_shape);
        let dot_b = self.broadcast_align(dot_kept, &reduced, out.clone());
        let x_dot = self.entry.binary("multiply", x, dot_b, out.clone());
        let x_dot_scaled = self.entry.binary("multiply", x_dot, inv_r2, out.clone());
        let term = self.entry.binary("subtract", dn, x_dot_scaled, out.clone());
        self.entry.binary("multiply", inv_r_b, term, out)
    }

    /// Affine-free ada norm value `n` (γ=1, β=0) for backward.
    fn lower_ada_norm_value(
        &mut self,
        x_id: NodeId,
        norm: AdaNormKind,
        eps: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let x_dims = out.dimensions.clone();
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);
        let rank = x_dims.len();
        let ax = (rank - 1) as i64;
        let mut reduced = x_dims.clone();
        reduced[ax as usize] = 1;
        let kept_shape = Shape::array(prim_ty, &reduced);
        let summed_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let summed_shape = Shape::array(prim_ty, &summed_dims);
        let n_elems = x_dims[ax as usize] as f32;
        let n_c = self.const_in_dtype(prim_ty, n_elems);
        let n_b = self.entry.broadcast(n_c, &[], summed_shape.clone());

        match norm {
            AdaNormKind::LayerNorm => {
                let pre_sum = self.reduce_one(x, ax, "add", 0.0, x_dt, summed_dims.clone());
                let mean = self
                    .entry
                    .binary("divide", pre_sum, n_b, summed_shape.clone());
                let mean_kept = self.entry.reshape(mean, kept_shape.clone());
                let mean_b = self.broadcast_align(mean_kept, &reduced, out.clone());
                let centered = self.entry.binary("subtract", x, mean_b, out.clone());
                let sq = self
                    .entry
                    .binary("multiply", centered, centered, out.clone());
                let sq_sum = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_dims);
                let var = self
                    .entry
                    .binary("divide", sq_sum, n_b, summed_shape.clone());
                let var_kept = self.entry.reshape(var, kept_shape);
                let eps_c = self.const_in_dtype(prim_ty, eps);
                let eps_b = self
                    .entry
                    .broadcast(eps_c, &[], Shape::array(prim_ty, &reduced));
                let var_eps =
                    self.entry
                        .binary("add", var_kept, eps_b, Shape::array(prim_ty, &reduced));
                let inv_std = self
                    .entry
                    .unary("rsqrt", var_eps, Shape::array(prim_ty, &reduced));
                let inv_std_b = self.broadcast_align(inv_std, &reduced, out.clone());
                self.entry
                    .binary("multiply", centered, inv_std_b, out.clone())
            }
            AdaNormKind::RmsNorm => {
                let sq = self.entry.binary("multiply", x, x, out.clone());
                let sq_sum = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_dims);
                let sq_mean = self
                    .entry
                    .binary("divide", sq_sum, n_b, summed_shape.clone());
                let eps_c = self.const_in_dtype(prim_ty, eps);
                let eps_b = self.entry.broadcast(eps_c, &[], summed_shape.clone());
                let var_eps = self
                    .entry
                    .binary("add", sq_mean, eps_b, summed_shape.clone());
                let inv = self.entry.unary("rsqrt", var_eps, summed_shape);
                let inv_kept = self.entry.reshape(inv, kept_shape);
                let inv_b = self.broadcast_align(inv_kept, &reduced, out.clone());
                self.entry.binary("multiply", x, inv_b, out.clone())
            }
        }
    }

    /// Packed DiT adaLN reverse — mirrors `compose_ada_layer_norm_backward`.
    pub(crate) fn lower_ada_layer_norm_backward(
        &mut self,
        inputs: &[NodeId],
        norm: AdaNormKind,
        eps: f32,
        out: Shape,
    ) -> i64 {
        let x_id = inputs[0];
        let scale_id = inputs[1];
        let dy_id = inputs[3];
        let x = self.hlo(x_id);
        let dy = self.hlo(dy_id);
        let scale = self.hlo(scale_id);
        let x_dims = self.ir_shape_dims(x_id);
        let scale_dims = self.ir_shape_dims(scale_id);
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);
        let x_shape = Shape::array(prim_ty, &x_dims);

        let n = self.lower_ada_norm_value(x_id, norm, eps, x_shape.clone());
        let one_c = self.const_in_dtype(prim_ty, 1.0);
        let ones_b = self.entry.broadcast(one_c, &[], x_shape.clone());
        let scale_b = self.broadcast_to_target(scale, &scale_dims, x_shape.clone());
        let one_plus = self.entry.binary("add", ones_b, scale_b, x_shape.clone());
        let dn = self.entry.binary("multiply", dy, one_plus, x_shape.clone());

        let dx = match norm {
            AdaNormKind::LayerNorm => {
                self.lower_layernorm_dx_gamma1_hlo(x, dn, eps, x_shape.clone(), x_dt)
            }
            AdaNormKind::RmsNorm => {
                self.lower_rms_norm_dx_gamma1_hlo(x, dn, eps, x_shape.clone(), x_dt)
            }
        };

        let dsf = self.entry.binary("multiply", dy, n, x_shape.clone());
        let dscale = self.sum_unbroadcast_hlo(dsf, &x_dims, &scale_dims, x_dt);
        let dshift = self.sum_unbroadcast_hlo(dy, &x_dims, &scale_dims, x_dt);
        self.pack_flat_grads_hlo(
            &[(dx, &x_dims), (dscale, &scale_dims), (dshift, &scale_dims)],
            out,
        )
    }

    /// Packed DiT gated residual reverse — mirrors `compose_gated_residual_backward`.
    pub(crate) fn lower_gated_residual_backward(&mut self, inputs: &[NodeId], out: Shape) -> i64 {
        let y_id = inputs[1];
        let gate_id = inputs[2];
        let dy_id = inputs[3];
        let dy = self.hlo(dy_id);
        let y = self.hlo(y_id);
        let gate = self.hlo(gate_id);
        let x_dims = self.ir_shape_dims(inputs[0]);
        let gate_dims = self.ir_shape_dims(gate_id);
        let x_dt = self.dtype(dy_id);
        let prim_ty = prim_of(x_dt);
        let x_shape = Shape::array(prim_ty, &x_dims);

        let dx = dy;
        let gate_b = self.broadcast_to_target(gate, &gate_dims, x_shape.clone());
        let dy_out = self.entry.binary("multiply", dy, gate_b, x_shape.clone());
        let dgate_full = self.entry.binary("multiply", dy, y, x_shape);
        let dgate = self.sum_unbroadcast_hlo(dgate_full, &x_dims, &gate_dims, x_dt);
        self.pack_flat_grads_hlo(
            &[(dx, &x_dims), (dy_out, &x_dims), (dgate, &gate_dims)],
            out,
        )
    }

    // ── FusedMatMulBiasAct ─────────────────────────────────────

    pub(crate) fn lower_fused_matmul_bias_act(
        &mut self,
        inputs: &[NodeId],
        activation: Option<Activation>,
        out: Shape,
    ) -> i64 {
        let mm = self.lower_matmul(inputs[0], inputs[1], out.clone());
        let bias = self.hlo(inputs[2]);
        let b_dims = self.ir_shape_dims(inputs[2]);
        let bias_b = self.broadcast_to_target(bias, &b_dims, out.clone());
        let added = self.entry.binary("add", mm, bias_b, out.clone());
        match activation {
            None => added,
            Some(act) => self.lower_activation(act, added, out),
        }
    }

    // ── Attention ──────────────────────────────────────────────

    pub(crate) fn lower_attention(
        &mut self,
        inputs: &[NodeId],
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        out: Shape,
    ) -> i64 {
        // Inputs: Q, K, V [, mask].
        // After unfuse, all are rank-4 [B, H, S, D] (rank-3 was promoted).
        let q = self.hlo(inputs[0]);
        let k = self.hlo(inputs[1]);
        let v = self.hlo(inputs[2]);
        let q_dims = self.ir_shape_dims(inputs[0]);
        let k_dims = self.ir_shape_dims(inputs[1]);
        let dt = self.dtype(inputs[0]);
        let prim_ty = prim_of(dt);
        let _ = num_heads;
        let _ = head_dim;
        let b_dim = q_dims[0];
        let h_dim = q_dims[1];
        let s_q = q_dims[2];
        let s_k = k_dims[2];
        let d_dim = q_dims[3];

        // QK^T: [B, H, S_q, D] x [B, H, S_k, D] → [B, H, S_q, S_k]
        // Contracting axis = 3 on both sides; batch = [0, 1].
        let qk_shape = Shape::array(prim_ty, &[b_dim, h_dim, s_q, s_k]);
        let qk_dn = DotDimNumbers {
            lhs_contracting: vec![3],
            rhs_contracting: vec![3],
            lhs_batch: vec![0, 1],
            rhs_batch: vec![0, 1],
        };
        let qk = self.entry.dot_general(q, k, qk_dn, qk_shape.clone());
        // Scale by 1 / sqrt(d).
        let scale = self.const_in_dtype(prim_ty, 1.0 / (d_dim as f32).sqrt());
        let scale_b = self.entry.broadcast(scale, &[], qk_shape.clone());
        let scaled = self.entry.binary("multiply", qk, scale_b, qk_shape.clone());

        // Apply mask.
        let masked = match mask_kind {
            MaskKind::None => scaled,
            MaskKind::Causal => self.apply_causal_mask(scaled, qk_shape.clone(), s_q, s_k, prim_ty),
            MaskKind::SlidingWindow(w) => self.apply_sliding_window_mask(
                scaled,
                qk_shape.clone(),
                s_q,
                s_k,
                w as i64,
                prim_ty,
            ),
            MaskKind::Custom | MaskKind::Bias => {
                // 4th input is the mask, additive [B, ?, S_q, S_k].
                let mask = self.hlo(inputs[3]);
                let mask_dims = self.ir_shape_dims(inputs[3]);
                let mask_b = self.broadcast_to_target(mask, &mask_dims, qk_shape.clone());
                self.entry.binary("add", scaled, mask_b, qk_shape.clone())
            }
        };

        // Softmax along last axis.
        let probs = self.lower_softmax_id(masked, qk_shape.clone(), 3);

        // probs @ V: [B, H, S_q, S_k] x [B, H, S_k, D] → [B, H, S_q, D]
        let av_dn = DotDimNumbers {
            lhs_contracting: vec![3],
            rhs_contracting: vec![2],
            lhs_batch: vec![0, 1],
            rhs_batch: vec![0, 1],
        };
        self.entry.dot_general(probs, v, av_dn, out)
    }

    pub(crate) fn lower_rope(
        &mut self,
        x_id: NodeId,
        cos_id: NodeId,
        sin_id: NodeId,
        head_dim: usize,
        out: Shape,
    ) -> i64 {
        // Standard non-interleaved RoPE: split last dim in halves
        // (x1, x2). Output: [x1*cos - x2*sin, x1*sin + x2*cos].
        let x = self.hlo(x_id);
        let cos = self.hlo(cos_id);
        let sin = self.hlo(sin_id);
        let x_dims = self.ir_shape_dims(x_id);
        let dt = self.dtype(x_id);
        let prim_ty = prim_of(dt);
        let half = head_dim / 2;
        let last = x_dims.len() - 1;

        let mut starts1 = vec![0i64; x_dims.len()];
        let mut limits1 = x_dims.clone();
        let mut starts2 = vec![0i64; x_dims.len()];
        let mut limits2 = x_dims.clone();
        let strides = vec![1i64; x_dims.len()];
        starts1[last] = 0;
        limits1[last] = half as i64;
        starts2[last] = half as i64;
        limits2[last] = head_dim as i64;
        let mut half_dims = x_dims.clone();
        half_dims[last] = half as i64;
        let half_shape = Shape::array(prim_ty, &half_dims);
        let x1 = self
            .entry
            .slice(x, &starts1, &limits1, &strides, half_shape.clone());
        let x2 = self
            .entry
            .slice(x, &starts2, &limits2, &strides, half_shape.clone());

        let cos_dims = self.ir_shape_dims(cos_id);
        let sin_dims = self.ir_shape_dims(sin_id);
        let cos_b = self.broadcast_to_target(cos, &cos_dims, half_shape.clone());
        let sin_b = self.broadcast_to_target(sin, &sin_dims, half_shape.clone());

        let x1c = self.entry.binary("multiply", x1, cos_b, half_shape.clone());
        let x2s = self.entry.binary("multiply", x2, sin_b, half_shape.clone());
        let r1 = self.entry.binary("subtract", x1c, x2s, half_shape.clone());
        let x1s = self.entry.binary("multiply", x1, sin_b, half_shape.clone());
        let x2c = self.entry.binary("multiply", x2, cos_b, half_shape.clone());
        let r2 = self.entry.binary("add", x1s, x2c, half_shape);
        self.entry.concat(&[r1, r2], last as i64, out)
    }

    // ── FFT ────────────────────────────────────────────────────

    pub(crate) fn lower_fft(
        &mut self,
        x_id: NodeId,
        inverse: bool,
        norm: rlx_ir::fft::FftNorm,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let dtype = self.dtype(x_id);
        let prim_ty = prim_of(dtype);
        match dtype {
            DType::F32 | DType::F64 => {}
            DType::C64 => {
                let dims = self.ir_shape_dims(x_id);
                let rank = dims.len();
                assert!(rank >= 1);
                let n = dims[rank - 1];
                let ax = (rank - 1) as i64;
                let fft_type = if inverse { 1 } else { 0 };
                let out_shape = out.clone();
                let y = self.entry.fft(x, fft_type, &[n], &[ax], out_shape.clone());
                if norm.output_scale(n as usize, inverse) != 1.0 {
                    let scale = norm.output_scale(n as usize, inverse) as f32;
                    let s = self.const_scalar_f32(scale);
                    return self.entry.binary("multiply", y, s, out_shape);
                }
                return y;
            }
            other => panic!("rlx-tpu: Op::Fft unsupported dtype {other:?}"),
        }
        let dims = self.ir_shape_dims(x_id);
        let rank = dims.len();
        assert!(rank >= 1, "rlx-tpu: Op::Fft input must have rank >= 1");
        let last = dims[rank - 1];
        assert!(
            last % 2 == 0,
            "rlx-tpu: Op::Fft last axis {last} must be even (2N real-block layout)"
        );
        let n = last / 2;
        let ax = (rank - 1) as i64;

        let mut prefix = dims.clone();
        prefix.pop();
        prefix.push(n);
        let plane = Shape::array(prim_ty, &prefix);
        let cx_ty = Shape::tuple(vec![plane.clone(), plane.clone()]);

        let starts0 = vec![0i64; rank];
        let mut limit_re = dims.clone();
        limit_re[rank - 1] = n;
        let re = self
            .entry
            .slice(x, &starts0, &limit_re, &vec![1i64; rank], plane.clone());

        let mut start_im = vec![0i64; rank];
        start_im[rank - 1] = n;
        let im = self
            .entry
            .slice(x, &start_im, &dims, &vec![1i64; rank], plane.clone());

        let cx = self.entry.tuple(&[re, im], cx_ty.clone());
        let fft_type = if inverse { 1 } else { 0 };
        let y = self.entry.fft(cx, fft_type, &[n], &[ax], cx_ty);
        let mut y_re = self.entry.get_tuple_element(y, 0, plane.clone());
        let mut y_im = self.entry.get_tuple_element(y, 1, plane.clone());
        let scale = norm.output_scale(n as usize, inverse);
        if scale != 1.0 {
            let mut s = self.const_scalar_f32(scale as f32);
            if dtype == DType::F64 {
                s = self.entry.convert(s, Shape::scalar(prim_ty));
            }
            y_re = self.entry.binary("multiply", y_re, s, plane.clone());
            y_im = self.entry.binary("multiply", y_im, s, plane.clone());
        }
        self.entry.concat(&[y_re, y_im], ax, out)
    }

    // ── Gather ─────────────────────────────────────────────────

    pub(crate) fn lower_gather(
        &mut self,
        table_id: NodeId,
        indices_id: NodeId,
        axis: usize,
        out: Shape,
    ) -> i64 {
        // Embedding-lookup style gather. Indices are integer
        // (we treat the index dtype as S32 for HLO; the IR may carry
        // them as f32-encoded so we convert if needed).
        let table = self.hlo(table_id);
        let idx = self.hlo(indices_id);
        let idx_dt = self.dtype(indices_id);
        let idx_s32 = if matches!(idx_dt, DType::I32 | DType::I64 | DType::U32) {
            idx
        } else {
            // Convert f32-encoded indices to s32.
            let idx_dims = self.ir_shape_dims(indices_id);
            self.entry.convert(idx, Shape::array(prim::S32, &idx_dims))
        };
        let table_dims = self.ir_shape_dims(table_id);
        let idx_dims = self.ir_shape_dims(indices_id);
        let mut slice_sizes = table_dims.clone();
        slice_sizes[axis] = 1;
        // HLO gather output shape: indices' batch dims interleaved
        // with operand's offset dims. `offset_dims` lists the
        // OUTPUT positions that come from the operand (after
        // `collapsed_slice_dims` are dropped). For an embedding
        // lookup against a 2-D table [V, H] with indices [B, S],
        // the output is [B, S, H]: the offset dim H lands at output
        // position `idx_rank` (= 2). Generalizing: with
        // `n_offset = table_rank - collapsed_slice_dims.len()`
        // operand-derived dims, they occupy the trailing positions
        // [idx_rank .. idx_rank + n_offset).
        let n_offset = (table_dims.len() - 1) as i64;
        let idx_rank = idx_dims.len() as i64;
        let offset_dims: Vec<i64> = (idx_rank..idx_rank + n_offset).collect();
        let dn = GatherDimNumbers {
            offset_dims,
            collapsed_slice_dims: vec![axis as i64],
            start_index_map: vec![axis as i64],
            index_vector_dim: idx_rank,
        };
        self.entry.gather(table, idx_s32, dn, slice_sizes, out)
    }

    // ── Reduce ─────────────────────────────────────────────────

    pub(crate) fn lower_reduce(
        &mut self,
        x_id: NodeId,
        op: ReduceOp,
        axes: &[usize],
        keep_dim: bool,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let x_dims = self.ir_shape_dims(x_id);
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);
        let axes_i64: Vec<i64> = axes.iter().map(|&a| a as i64).collect();

        // Reducer + identity element + post-divide.
        let (opcode, init_v, divide_by_n) = match op {
            ReduceOp::Sum => ("add", 0.0_f32, false),
            ReduceOp::Mean => ("add", 0.0_f32, true),
            ReduceOp::Max => ("maximum", f32::NEG_INFINITY, false),
            ReduceOp::Min => ("minimum", f32::INFINITY, false),
            ReduceOp::Prod => ("multiply", 1.0_f32, false),
        };
        let red = self.reducer(opcode, prim_ty);
        let init = self.const_in_dtype(prim_ty, init_v);

        // Determine intermediate (no keep_dim) shape.
        let collapsed_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if axes.contains(&i) { None } else { Some(d) })
            .collect();
        let collapsed_shape = Shape::array(prim_ty, &collapsed_dims);
        let mut reduced = self
            .entry
            .reduce(x, init, &red, &axes_i64, collapsed_shape.clone());
        if matches!(op, ReduceOp::Mean) {
            let n: i64 = axes.iter().map(|&a| x_dims[a]).product();
            let n_c = self.const_in_dtype(prim_ty, n as f32);
            let n_b = self.entry.broadcast(n_c, &[], collapsed_shape.clone());
            reduced = self.entry.binary("divide", reduced, n_b, collapsed_shape);
        }
        let _ = divide_by_n;
        if keep_dim {
            self.entry.reshape(reduced, out)
        } else {
            reduced
        }
    }

    // ── Softmax ────────────────────────────────────────────────

    pub(crate) fn lower_softmax(&mut self, x_id: NodeId, axis: i32, out: Shape) -> i64 {
        let x = self.hlo(x_id);
        self.lower_softmax_id(x, out, axis as i64)
    }

    pub(crate) fn lower_softmax_id(&mut self, x: i64, out: Shape, axis: i64) -> i64 {
        let dims = out.dimensions.clone();
        let prim_ty = out.element_type;
        let rank = dims.len() as i64;
        let ax = if axis < 0 { rank + axis } else { axis };

        // Numerically-stable softmax: x' = x - max(x); y = exp(x') / sum(exp(x'))
        let collapsed: Vec<i64> = dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let mut kept = dims.clone();
        kept[ax as usize] = 1;
        let collapsed_shape = Shape::array(prim_ty, &collapsed);
        let kept_shape = Shape::array(prim_ty, &kept);

        let red_max = self.reducer("maximum", prim_ty);
        let init_max = self.const_in_dtype(prim_ty, f32::NEG_INFINITY);
        let max_v = self
            .entry
            .reduce(x, init_max, &red_max, &[ax], collapsed_shape.clone());
        let max_kept = self.entry.reshape(max_v, kept_shape.clone());
        let max_b = self.broadcast_align(max_kept, &kept, out.clone());
        let centered = self.entry.binary("subtract", x, max_b, out.clone());
        let exped = self.entry.unary("exponential", centered, out.clone());

        let red_sum = self.reducer("add", prim_ty);
        let init_sum = self.const_in_dtype(prim_ty, 0.0);
        let sum_v = self
            .entry
            .reduce(exped, init_sum, &red_sum, &[ax], collapsed_shape);
        let sum_kept = self.entry.reshape(sum_v, kept_shape);
        let sum_b = self.broadcast_align(sum_kept, &kept, out.clone());
        self.entry.binary("divide", exped, sum_b, out)
    }

    // ── Cumsum ─────────────────────────────────────────────────

    pub(crate) fn lower_cumsum(
        &mut self,
        x_id: NodeId,
        axis: i32,
        exclusive: bool,
        out: Shape,
    ) -> i64 {
        // HLO has no `cumsum` primitive — use `reduce-window` with a
        // window that spans the whole prefix along the chosen axis.
        let x = self.hlo(x_id);
        let dims = self.ir_shape_dims(x_id);
        let prim_ty = prim_of(self.dtype(x_id));
        let rank = dims.len() as i32;
        let ax = if axis < 0 {
            (rank + axis) as i64
        } else {
            axis as i64
        };

        let init = self.const_in_dtype(prim_ty, 0.0);
        let red = self.reducer("add", prim_ty);

        let mut window_dims = vec![
            WindowDim {
                size: 1,
                stride: 1,
                padding_low: 0,
                padding_high: 0,
                window_dilation: 1,
                base_dilation: 1,
            };
            dims.len()
        ];
        // Inclusive scan: window of size = full axis length, with
        // padding_low = N-1 so each prefix sees [0..i].
        window_dims[ax as usize] = WindowDim {
            size: dims[ax as usize],
            stride: 1,
            padding_low: dims[ax as usize] - 1,
            padding_high: 0,
            window_dilation: 1,
            base_dilation: 1,
        };
        let window = Window {
            dimensions: window_dims,
        };
        let scanned = self.entry.reduce_window(x, init, &red, window, out.clone());
        if exclusive {
            // Shift-right-by-one along axis: pad a leading 0 and slice.
            let zero = self.const_in_dtype(prim_ty, 0.0);
            let mut pad_cfg = vec![(0i64, 0i64, 0i64); dims.len()];
            pad_cfg[ax as usize] = (1, 0, 0);
            let mut padded_dims = dims.clone();
            padded_dims[ax as usize] += 1;
            let padded =
                self.entry
                    .pad(scanned, zero, pad_cfg, Shape::array(prim_ty, &padded_dims));
            let mut starts = vec![0i64; dims.len()];
            let mut limits = padded_dims.clone();
            let strides = vec![1i64; dims.len()];
            starts[ax as usize] = 0;
            limits[ax as usize] = dims[ax as usize];
            self.entry.slice(padded, &starts, &limits, &strides, out)
        } else {
            scanned
        }
    }

    // ── Conv ───────────────────────────────────────────────────

    pub(crate) fn lower_conv(
        &mut self,
        x_id: NodeId,
        w_id: NodeId,
        kernel_size: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        groups: usize,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let w = self.hlo(w_id);
        let x_rank = self.ir_shape_dims(x_id).len();
        // Convention: input is [N, C, *spatial], weight is
        // [C_out, C_in/groups, *spatial]. HLO Convolution expects an
        // explicit dimension-numbers proto.
        let n_spatial = x_rank - 2;
        let cdn = ConvDimNumbers {
            input_batch_dim: 0,
            input_feature_dim: 1,
            input_spatial_dims: (2..2 + n_spatial as i64).collect(),
            kernel_output_feature_dim: 0,
            kernel_input_feature_dim: 1,
            kernel_spatial_dims: (2..2 + n_spatial as i64).collect(),
            output_batch_dim: 0,
            output_feature_dim: 1,
            output_spatial_dims: (2..2 + n_spatial as i64).collect(),
        };
        let mut window_dims = Vec::with_capacity(n_spatial);
        for i in 0..n_spatial {
            window_dims.push(WindowDim {
                size: kernel_size[i] as i64,
                stride: stride[i] as i64,
                padding_low: padding[i] as i64,
                padding_high: padding[i] as i64,
                window_dilation: dilation[i] as i64,
                base_dilation: 1,
            });
        }
        let window = Window {
            dimensions: window_dims,
        };
        self.entry
            .convolution(x, w, window, cdn, groups as i64, out)
    }

    // ── Pool ───────────────────────────────────────────────────

    pub(crate) fn lower_pool(
        &mut self,
        x_id: NodeId,
        kind: ReduceOp,
        kernel_size: &[usize],
        stride: &[usize],
        padding: &[usize],
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let x_dims = self.ir_shape_dims(x_id);
        let prim_ty = prim_of(self.dtype(x_id));
        let n_spatial = x_dims.len() - 2;
        let (opcode, init_v) = match kind {
            ReduceOp::Sum | ReduceOp::Mean => ("add", 0.0_f32),
            ReduceOp::Max => ("maximum", f32::NEG_INFINITY),
            ReduceOp::Min => ("minimum", f32::INFINITY),
            ReduceOp::Prod => ("multiply", 1.0_f32),
        };
        let red = self.reducer(opcode, prim_ty);
        let init = self.const_in_dtype(prim_ty, init_v);

        let mut window_dims = vec![
            WindowDim {
                size: 1,
                stride: 1,
                padding_low: 0,
                padding_high: 0,
                window_dilation: 1,
                base_dilation: 1,
            };
            x_dims.len()
        ];
        for i in 0..n_spatial {
            window_dims[2 + i] = WindowDim {
                size: kernel_size[i] as i64,
                stride: stride[i] as i64,
                padding_low: padding[i] as i64,
                padding_high: padding[i] as i64,
                window_dilation: 1,
                base_dilation: 1,
            };
        }
        let window = Window {
            dimensions: window_dims,
        };
        let pooled = self.entry.reduce_window(x, init, &red, window, out.clone());

        if matches!(kind, ReduceOp::Mean) {
            // Divide by window size.
            let denom = kernel_size.iter().product::<usize>() as f32;
            let denom_c = self.const_in_dtype(prim_ty, denom);
            let denom_b = self.entry.broadcast(denom_c, &[], out.clone());
            self.entry.binary("divide", pooled, denom_b, out)
        } else {
            pooled
        }
    }

    // ── ScatterAdd ─────────────────────────────────────────────

    pub(crate) fn lower_scatter_add(
        &mut self,
        updates_id: NodeId,
        indices_id: NodeId,
        out: Shape,
    ) -> i64 {
        // Build a zero-initialized destination of shape `out`, then
        // scatter-add updates rows at indices.
        let updates = self.hlo(updates_id);
        let idx = self.hlo(indices_id);
        let idx_dt = self.dtype(indices_id);
        let idx_s32 = if matches!(idx_dt, DType::I32 | DType::I64 | DType::U32) {
            idx
        } else {
            let id_dims = self.ir_shape_dims(indices_id);
            self.entry.convert(idx, Shape::array(prim::S32, &id_dims))
        };

        let prim_ty = out.element_type;
        let zero = self.const_in_dtype(prim_ty, 0.0);
        let dest = self.entry.broadcast(zero, &[], out.clone());
        let combiner = self.reducer("add", prim_ty);

        // Indices semantics: each element of `idx_s32` selects a row
        // along axis 0 of `dest`. ScatterDimNumbers reflects that:
        //   update_window_dims = [1, 2, ..., rank-1]   (trailing dims of update)
        //   inserted_window_dims = [0]
        //   scatter_dims_to_operand_dims = [0]
        //   index_vector_dim = idx_rank
        let upd_rank = self.ir_shape_dims(updates_id).len() as i64;
        let dn = ScatterDimNumbers {
            update_window_dims: (1..upd_rank).collect(),
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: self.ir_shape_dims(indices_id).len() as i64,
        };
        self.entry
            .scatter(dest, idx_s32, updates, &combiner, dn, out)
    }

    // ── ScatterNd (ONNX ScatterND) ─────────────────────────────

    pub(crate) fn lower_scatter_nd(
        &mut self,
        data_id: NodeId,
        indices_id: NodeId,
        updates_id: NodeId,
        reduction: rlx_ir::ScatterNdReduction,
        out: Shape,
    ) -> i64 {
        let data = self.hlo(data_id);
        let updates = self.hlo(updates_id);
        let idx = self.hlo(indices_id);
        let idx_dt = self.dtype(indices_id);
        let idx_s32 = if matches!(idx_dt, DType::I32 | DType::I64 | DType::U32) {
            idx
        } else {
            let id_dims = self.ir_shape_dims(indices_id);
            self.entry.convert(idx, Shape::array(prim::S32, &id_dims))
        };

        let data_dims = self.ir_shape_dims(data_id);
        let idx_dims = self.ir_shape_dims(indices_id);
        let upd_dims = self.ir_shape_dims(updates_id);
        let k = *idx_dims.last().unwrap_or(&1);
        let indices_rank = idx_dims.len() as i64;
        let updates_rank = upd_dims.len() as i64;
        // ONNX: updates trailing window = data[k:]; those axes are update_window_dims.
        let update_window_dims: Vec<i64> = ((indices_rank - 1)..updates_rank).collect();
        let inserted_window_dims: Vec<i64> = (0..k).collect();
        let scatter_dims_to_operand_dims: Vec<i64> = (0..k).collect();
        let dn = ScatterDimNumbers {
            update_window_dims,
            inserted_window_dims,
            scatter_dims_to_operand_dims,
            index_vector_dim: indices_rank - 1,
        };
        let prim_ty = out.element_type;
        let combiner = match reduction {
            rlx_ir::ScatterNdReduction::None => self
                .builder
                .make_scatter_replace(&format!("scatter_nd_replace_{prim_ty}"), prim_ty),
            rlx_ir::ScatterNdReduction::Add => self.reducer("add", prim_ty),
            rlx_ir::ScatterNdReduction::Mul => self.reducer("multiply", prim_ty),
            rlx_ir::ScatterNdReduction::Max => self.reducer("maximum", prim_ty),
            rlx_ir::ScatterNdReduction::Min => self.reducer("minimum", prim_ty),
        };
        let _ = data_dims;
        self.entry
            .scatter(data, idx_s32, updates, &combiner, dn, out)
    }

    // ── GatherNd (ONNX GatherND) ───────────────────────────────

    pub(crate) fn lower_gather_nd(
        &mut self,
        data_id: NodeId,
        indices_id: NodeId,
        batch_dims: i32,
        out: Shape,
    ) -> i64 {
        assert_eq!(
            batch_dims, 0,
            "rlx-tpu: GatherNd batch_dims={batch_dims} not supported (need 0)"
        );
        let data = self.hlo(data_id);
        let idx = self.hlo(indices_id);
        let idx_dt = self.dtype(indices_id);
        let idx_s32 = if matches!(idx_dt, DType::I32 | DType::I64 | DType::U32) {
            idx
        } else {
            let id_dims = self.ir_shape_dims(indices_id);
            self.entry.convert(idx, Shape::array(prim::S32, &id_dims))
        };
        let data_dims = self.ir_shape_dims(data_id);
        let idx_dims = self.ir_shape_dims(indices_id);
        let k = *idx_dims.last().unwrap_or(&1);
        let indices_rank = idx_dims.len() as i64;
        let mut slice_sizes = data_dims.clone();
        for d in slice_sizes.iter_mut().take(k as usize) {
            *d = 1;
        }
        let n_offset = (data_dims.len() as i64) - k;
        let offset_dims: Vec<i64> = ((indices_rank - 1)..(indices_rank - 1 + n_offset)).collect();
        let dn = GatherDimNumbers {
            offset_dims,
            collapsed_slice_dims: (0..k).collect(),
            start_index_map: (0..k).collect(),
            index_vector_dim: indices_rank - 1,
        };
        self.entry.gather(data, idx_s32, dn, slice_sizes, out)
    }

    // Expand take_along / scatter_elements indices into full ND indices
    // of shape `[*indices_shape, rank]` for GatherNd / ScatterNd.
    fn expand_elements_indices_nd(
        &mut self,
        data_id: NodeId,
        indices_id: NodeId,
        axis: i32,
    ) -> (i64, Vec<i64>) {
        let data_dims = self.ir_shape_dims(data_id);
        let idx_dims = self.ir_shape_dims(indices_id);
        let rank = data_dims.len() as i64;
        let axis = if axis < 0 { axis + rank as i32 } else { axis } as i64;
        let idx = self.hlo(indices_id);
        let idx_dt = self.dtype(indices_id);
        let idx_s32 = if matches!(idx_dt, DType::I32 | DType::I64 | DType::U32) {
            idx
        } else {
            self.entry.convert(idx, Shape::array(prim::S32, &idx_dims))
        };
        // Stack per-axis coordinates: iota for non-axis dims, indices for axis.
        let mut comps: Vec<i64> = Vec::with_capacity(rank as usize);
        for d in 0..rank {
            let comp = if d == axis {
                idx_s32
            } else {
                let iota_shape = Shape::array(prim::S32, &idx_dims);
                self.entry.iota(d, iota_shape)
            };
            // [..., 1]
            let mut unsqueeze_dims = idx_dims.clone();
            unsqueeze_dims.push(1);
            comps.push(
                self.entry
                    .reshape(comp, Shape::array(prim::S32, &unsqueeze_dims)),
            );
        }
        let mut nd_dims = idx_dims.clone();
        nd_dims.push(rank);
        let stacked = self
            .entry
            .concat(&comps, rank, Shape::array(prim::S32, &nd_dims));
        (stacked, nd_dims)
    }

    pub(crate) fn lower_gather_elements(
        &mut self,
        data_id: NodeId,
        indices_id: NodeId,
        axis: i32,
        out: Shape,
    ) -> i64 {
        let (nd_idx, _nd_dims) = self.expand_elements_indices_nd(data_id, indices_id, axis);
        let data = self.hlo(data_id);
        let data_dims = self.ir_shape_dims(data_id);
        let idx_dims = self.ir_shape_dims(indices_id);
        let k = data_dims.len() as i64;
        let indices_rank = (idx_dims.len() as i64) + 1; // after expand
        let slice_sizes = vec![1i64; k as usize];
        let offset_dims: Vec<i64> = vec![];
        let dn = GatherDimNumbers {
            offset_dims,
            collapsed_slice_dims: (0..k).collect(),
            start_index_map: (0..k).collect(),
            index_vector_dim: indices_rank - 1,
        };
        self.entry.gather(data, nd_idx, dn, slice_sizes, out)
    }

    pub(crate) fn lower_scatter_elements(
        &mut self,
        data_id: NodeId,
        indices_id: NodeId,
        updates_id: NodeId,
        axis: i32,
        reduction: rlx_ir::ScatterNdReduction,
        out: Shape,
    ) -> i64 {
        let (nd_idx, _nd_dims) = self.expand_elements_indices_nd(data_id, indices_id, axis);
        let data = self.hlo(data_id);
        let updates = self.hlo(updates_id);
        let data_dims = self.ir_shape_dims(data_id);
        let idx_dims = self.ir_shape_dims(indices_id);
        let k = data_dims.len() as i64;
        let indices_rank = (idx_dims.len() as i64) + 1;
        let updates_rank = self.ir_shape_dims(updates_id).len() as i64;
        // Full multi-index → no update window dims.
        let update_window_dims: Vec<i64> = ((indices_rank - 1)..updates_rank).collect();
        let inserted_window_dims: Vec<i64> = (0..k).collect();
        let scatter_dims_to_operand_dims: Vec<i64> = (0..k).collect();
        let dn = ScatterDimNumbers {
            update_window_dims,
            inserted_window_dims,
            scatter_dims_to_operand_dims,
            index_vector_dim: indices_rank - 1,
        };
        let prim_ty = out.element_type;
        let combiner = match reduction {
            rlx_ir::ScatterNdReduction::None => self
                .builder
                .make_scatter_replace(&format!("scatter_el_replace_{prim_ty}"), prim_ty),
            rlx_ir::ScatterNdReduction::Add => self.reducer("add", prim_ty),
            rlx_ir::ScatterNdReduction::Mul => self.reducer("multiply", prim_ty),
            rlx_ir::ScatterNdReduction::Max => self.reducer("maximum", prim_ty),
            rlx_ir::ScatterNdReduction::Min => self.reducer("minimum", prim_ty),
        };
        self.entry
            .scatter(data, nd_idx, updates, &combiner, dn, out)
    }

    // ── TopK ──────────────────────────────────────────────────────
    //
    // Sort (descending) along the last axis, paired with an iota of
    // indices, then slice the leading k. Indices come back as f32
    // because rlx-ir is f32 at the I/O boundary.

    pub(crate) fn lower_topk(&mut self, x_id: NodeId, k: usize, out: Shape) -> i64 {
        let x = self.hlo(x_id);
        let dims = self.ir_shape_dims(x_id);
        let prim_ty = prim_of(self.dtype(x_id));
        let last_axis = (dims.len() - 1) as i64;

        let iota_shape = Shape::array(prim::S32, &dims);
        let indices = self.entry.iota(last_axis, iota_shape.clone());

        // Comparator: (kx, ky, vx, vy) -> kx > ky.
        let cmp = self.builder.computation("topk_descending");
        let key_s = Shape::scalar(prim_ty);
        let val_s = Shape::scalar(prim::S32);
        let p0 = cmp.parameter(0, "kx", key_s.clone());
        let p1 = cmp.parameter(1, "ky", key_s.clone());
        let _p2 = cmp.parameter(2, "vx", val_s.clone());
        let _p3 = cmp.parameter(3, "vy", val_s.clone());
        let r = cmp.compare(p0, p1, "GT", Shape::scalar(prim::PRED));
        cmp.set_root(r);
        cmp.set_program_shape(ProgramShape {
            parameters: vec![key_s.clone(), key_s.clone(), val_s.clone(), val_s.clone()],
            parameter_names: vec!["kx".into(), "ky".into(), "vx".into(), "vy".into()],
            result: Shape::scalar(prim::PRED),
        });

        let val_full = Shape::array(prim_ty, &dims);
        let idx_full = Shape::array(prim::S32, &dims);
        let tup = Shape::tuple(vec![val_full, idx_full.clone()]);
        let sorted = self.entry.sort(&[x, indices], &cmp, last_axis, true, tup);
        let sorted_idx = self.entry.get_tuple_element(sorted, 1, idx_full);

        let mut starts = vec![0i64; dims.len()];
        let mut limits = dims.clone();
        let strides = vec![1i64; dims.len()];
        starts[last_axis as usize] = 0;
        limits[last_axis as usize] = k as i64;
        let mut slice_dims = dims.clone();
        slice_dims[last_axis as usize] = k as i64;
        let sliced = self.entry.slice(
            sorted_idx,
            &starts,
            &limits,
            &strides,
            Shape::array(prim::S32, &slice_dims),
        );
        // Indices → f32 (rlx-ir convention).
        self.entry.convert(sliced, out)
    }

    // ── GroupedMatMul ────────────────────────────────────────────
    //
    // For each token `i`, output[i] = input[i] @ weight[expert_idx[i]].
    // Lowered as gather(weight, idx) to materialize per-token weights
    // [M,K,N], then a batched dot_general with batch axis = M.

    pub(crate) fn lower_grouped_matmul(
        &mut self,
        input_id: NodeId,
        weight_id: NodeId,
        expert_id: NodeId,
        out: Shape,
    ) -> i64 {
        let input = self.hlo(input_id);
        let weight = self.hlo(weight_id);
        let exp_idx = self.hlo(expert_id);
        let exp_dt = self.dtype(expert_id);
        let m_dims = self.ir_shape_dims(input_id); // [M, K]
        let w_dims = self.ir_shape_dims(weight_id); // [E, K, N]
        let m = m_dims[0];
        let k = m_dims[1];
        let n = w_dims[2];

        let exp_s32 = if matches!(exp_dt, DType::I32 | DType::I64 | DType::U32) {
            exp_idx
        } else {
            self.entry.convert(exp_idx, Shape::array(prim::S32, &[m]))
        };
        // Gather wants index_vector_dim, so reshape [M] → [M, 1].
        let exp_2d = self
            .entry
            .reshape(exp_s32, Shape::array(prim::S32, &[m, 1]));

        let dn = GatherDimNumbers {
            offset_dims: vec![1, 2],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
        };
        let weight_prim = prim_of(self.dtype(weight_id));
        let gathered = self.entry.gather(
            weight,
            exp_2d,
            dn,
            vec![1, k, n],
            Shape::array(weight_prim, &[m, k, n]),
        );

        let dn = DotDimNumbers {
            lhs_contracting: vec![1],
            rhs_contracting: vec![1],
            lhs_batch: vec![0],
            rhs_batch: vec![0],
        };
        self.entry.dot_general(input, gathered, dn, out)
    }

    // ── DequantMatMul ────────────────────────────────────────────
    //
    // Non-GGUF: dequantize w_q in HLO (convert + per-block scale/zp tile) then dot.
    //
    // GGUF (`scheme.is_gguf()`): host-dequant at lowering time via
    // `dequant_gguf_bytes`, embed as f32 Constant, `dot_general`. No on-device
    // GGUF kernels on TPU — weights must be available when the HLO module is built
    // (`Op::Constant`, or `Op::Param` with bytes in `LowerParamBytes`).

    pub(crate) fn lower_dequant_matmul_gguf(
        &mut self,
        x_id: NodeId,
        w_id: NodeId,
        scheme: QuantScheme,
        out: Shape,
    ) -> i64 {
        let x_dims = self.ir_shape_dims(x_id);
        let k = *x_dims.last().expect("DequantMatMul x rank >= 1");
        let n = *out.dimensions.last().expect("DequantMatMul out rank >= 1");
        let w_hlo = if self.gguf_weight_is_deferred(w_id) {
            self.hlo(w_id)
        } else {
            let n_elems = (k * n) as usize;
            let bytes = self.gguf_weight_bytes(w_id);
            let w_f32 = dequant_gguf_bytes(scheme, &bytes, n_elems)
                .unwrap_or_else(|e| panic!("rlx-tpu: GGUF host dequant failed: {e}"));
            let mut w_bytes = Vec::with_capacity(w_f32.len() * 4);
            for v in &w_f32 {
                w_bytes.extend_from_slice(&v.to_le_bytes());
            }
            let kn_f32 = Shape::array(prim::F32, &[k, n]);
            self.lower_constant(&w_bytes, kn_f32, DType::F32)
        };
        let x = self.hlo(x_id);
        let dn = DotDimNumbers {
            lhs_contracting: vec![1],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
        };
        self.entry.dot_general(x, w_hlo, dn, out)
    }

    pub(crate) fn lower_dequant_matmul(
        &mut self,
        x_id: NodeId,
        w_id: NodeId,
        s_id: NodeId,
        z_id: NodeId,
        scheme: QuantScheme,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let w_q = self.hlo(w_id);
        let scale = self.hlo(s_id);
        let zp = self.hlo(z_id);
        let w_dims = self.ir_shape_dims(w_id); // [K, N]
        let k = w_dims[0];
        let n = w_dims[1];
        let block = match scheme {
            QuantScheme::Int8Block { block_size }
            | QuantScheme::Int8BlockAsym { block_size }
            | QuantScheme::Int4Block { block_size } => block_size as i64,
            // Fp8 schemes are per-tensor; treat as one-block-of-K.
            QuantScheme::Fp8E4m3 | QuantScheme::Fp8E5m2 => k,
            QuantScheme::GgufQ4_0
            | QuantScheme::GgufQ8_0
            | QuantScheme::GgufQ4_1
            | QuantScheme::GgufQ5_0
            | QuantScheme::GgufQ5_1
            | QuantScheme::GgufQ1_0
            | QuantScheme::GgufQ2_0 => panic!(
                "rlx-tpu: GGUF / NVFP4 quant schemes have no HLO lowering — dequantize on CPU first."
            ),
            QuantScheme::GgufQ4K
            | QuantScheme::GgufQ5K
            | QuantScheme::GgufQ6K
            | QuantScheme::GgufQ8K
            | QuantScheme::GgufQ2K
            | QuantScheme::GgufQ3K
            | QuantScheme::Nvfp4Block
            | QuantScheme::GgufIQ4NL
            | QuantScheme::GgufIQ4XS
            | QuantScheme::GgufIQ2XXS
            | QuantScheme::GgufIQ2XS
            | QuantScheme::GgufIQ2S
            | QuantScheme::GgufIQ3XXS
            | QuantScheme::GgufIQ3S
            | QuantScheme::GgufIQ1S
            | QuantScheme::GgufIQ1M
            | QuantScheme::GgufTQ1_0
            | QuantScheme::GgufTQ2_0
            | QuantScheme::GgufMXFP4
            | QuantScheme::GgufNVFP4 => panic!(
                "rlx-tpu: GGUF / NVFP4 quant schemes have no HLO lowering — dequantize on CPU first."
            ),
        };
        let kb = (k + block - 1) / block;

        let kn_f32 = Shape::array(prim::F32, &[k, n]);
        let w_f = self.entry.convert(w_q, kn_f32.clone());

        // Helper: broadcast a [K/block, N] tile to [K, N] by tiling
        // each row `block` times. HLO's `broadcast` requires the
        // operand's dim sizes to match the target's at the dims named
        // by broadcast_dims — it does NOT auto-expand size-1 dims.
        // So go [kb, n] → [kb, block, n] (broadcast_dims = [0, 2],
        // adds a fresh size-`block` axis at dim 1) → [k, n] (reshape).
        let tile_block = |this: &Self, t: i64, t_dt: i32| -> i64 {
            let t_b = this
                .entry
                .broadcast(t, &[0, 2], Shape::array(t_dt, &[kb, block, n]));
            this.entry.reshape(t_b, Shape::array(t_dt, &[k, n]))
        };
        let scale_kn = tile_block(self, scale, prim::F32);
        let zp_kn = tile_block(self, zp, prim::F32);

        let centered = self.entry.binary("subtract", w_f, zp_kn, kn_f32.clone());
        let w_dq = self.entry.binary("multiply", centered, scale_kn, kn_f32);

        let dn = DotDimNumbers {
            lhs_contracting: vec![1],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
        };
        self.entry.dot_general(x, w_dq, dn, out)
    }

    // ── QMatMul ───────────────────────────────────────────────────
    //
    // Real INT8 matmul: promote x, w to S32, subtract zero points,
    // dot, add bias, scale by `mult` in F32, round, +out_zp, clamp
    // to [-128, 127], convert back to S8.

    pub(crate) fn lower_qmatmul(
        &mut self,
        x_id: NodeId,
        w_id: NodeId,
        b_id: NodeId,
        x_zp: i32,
        w_zp: i32,
        out_zp: i32,
        mult: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let w = self.hlo(w_id);
        let bias = self.hlo(b_id);
        let x_dims = self.ir_shape_dims(x_id);
        let w_dims = self.ir_shape_dims(w_id);
        let m = x_dims[0];
        let k = x_dims[1];
        let n = w_dims[1];
        let mn_s32 = Shape::array(prim::S32, &[m, n]);
        let mn_f32 = Shape::array(prim::F32, &[m, n]);

        let x_s32 = self.entry.convert(x, Shape::array(prim::S32, &[m, k]));
        let w_s32 = self.entry.convert(w, Shape::array(prim::S32, &[k, n]));

        let xzp_c = self.entry.constant_s32_scalar(x_zp);
        let xzp_b = self
            .entry
            .broadcast(xzp_c, &[], Shape::array(prim::S32, &[m, k]));
        let x_centered =
            self.entry
                .binary("subtract", x_s32, xzp_b, Shape::array(prim::S32, &[m, k]));

        let wzp_c = self.entry.constant_s32_scalar(w_zp);
        let wzp_b = self
            .entry
            .broadcast(wzp_c, &[], Shape::array(prim::S32, &[k, n]));
        let w_centered =
            self.entry
                .binary("subtract", w_s32, wzp_b, Shape::array(prim::S32, &[k, n]));

        let dn = DotDimNumbers {
            lhs_contracting: vec![1],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
        };
        let acc = self
            .entry
            .dot_general(x_centered, w_centered, dn, mn_s32.clone());
        let bias_b = self.entry.broadcast(bias, &[1], mn_s32.clone());
        let with_bias = self.entry.binary("add", acc, bias_b, mn_s32.clone());

        let acc_f32 = self.entry.convert(with_bias, mn_f32.clone());
        let m_c = self.entry.constant_f32_scalar(mult);
        let m_b = self.entry.broadcast(m_c, &[], mn_f32.clone());
        let scaled = self.entry.binary("multiply", acc_f32, m_b, mn_f32.clone());
        let rounded = self
            .entry
            .unary("round-nearest-even", scaled, mn_f32.clone());
        let oz_c = self.entry.constant_f32_scalar(out_zp as f32);
        let oz_b = self.entry.broadcast(oz_c, &[], mn_f32.clone());
        let with_oz = self.entry.binary("add", rounded, oz_b, mn_f32.clone());

        let lo_c = self.entry.constant_f32_scalar(-128.0);
        let hi_c = self.entry.constant_f32_scalar(127.0);
        let lo_b = self.entry.broadcast(lo_c, &[], mn_f32.clone());
        let hi_b = self.entry.broadcast(hi_c, &[], mn_f32.clone());
        let cl_lo = self.entry.binary("maximum", with_oz, lo_b, mn_f32.clone());
        let cl = self.entry.binary("minimum", cl_lo, hi_b, mn_f32);

        self.entry.convert(cl, out)
    }

    // ── QConv2d ──────────────────────────────────────────────────
    //
    // Same arithmetic shape as QMatMul, but wrapped around a 2-D
    // convolution. Inputs are NCHW int8; bias is per-output-channel
    // s32 in accumulator scale.

    pub(crate) fn lower_qconv2d(
        &mut self,
        x_id: NodeId,
        w_id: NodeId,
        b_id: NodeId,
        kernel_size: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        groups: usize,
        x_zp: i32,
        w_zp: i32,
        out_zp: i32,
        mult: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let w = self.hlo(w_id);
        let bias = self.hlo(b_id);
        let x_dims = self.ir_shape_dims(x_id);
        let w_dims = self.ir_shape_dims(w_id);
        let out_dims = out.dimensions.clone();

        let x_s32_shape = Shape::array(prim::S32, &x_dims);
        let w_s32_shape = Shape::array(prim::S32, &w_dims);
        let out_s32 = Shape::array(prim::S32, &out_dims);
        let out_f32 = Shape::array(prim::F32, &out_dims);

        let x_s32 = self.entry.convert(x, x_s32_shape.clone());
        let w_s32 = self.entry.convert(w, w_s32_shape.clone());

        let xzp_c = self.entry.constant_s32_scalar(x_zp);
        let xzp_b = self.entry.broadcast(xzp_c, &[], x_s32_shape.clone());
        let x_centered = self.entry.binary("subtract", x_s32, xzp_b, x_s32_shape);
        let wzp_c = self.entry.constant_s32_scalar(w_zp);
        let wzp_b = self.entry.broadcast(wzp_c, &[], w_s32_shape.clone());
        let w_centered = self.entry.binary("subtract", w_s32, wzp_b, w_s32_shape);

        let n_spatial = x_dims.len() - 2;
        let cdn = ConvDimNumbers {
            input_batch_dim: 0,
            input_feature_dim: 1,
            input_spatial_dims: (2..2 + n_spatial as i64).collect(),
            kernel_output_feature_dim: 0,
            kernel_input_feature_dim: 1,
            kernel_spatial_dims: (2..2 + n_spatial as i64).collect(),
            output_batch_dim: 0,
            output_feature_dim: 1,
            output_spatial_dims: (2..2 + n_spatial as i64).collect(),
        };
        let mut window_dims = Vec::with_capacity(n_spatial);
        for i in 0..n_spatial {
            window_dims.push(WindowDim {
                size: kernel_size[i] as i64,
                stride: stride[i] as i64,
                padding_low: padding[i] as i64,
                padding_high: padding[i] as i64,
                window_dilation: dilation[i] as i64,
                base_dilation: 1,
            });
        }
        let window = Window {
            dimensions: window_dims,
        };
        let acc = self.entry.convolution(
            x_centered,
            w_centered,
            window,
            cdn,
            groups as i64,
            out_s32.clone(),
        );

        // Broadcast bias [C_out] across batch + spatial (axis 1 in NCHW).
        let bias_b = self.entry.broadcast(bias, &[1], out_s32.clone());
        let with_bias = self.entry.binary("add", acc, bias_b, out_s32);

        let acc_f32 = self.entry.convert(with_bias, out_f32.clone());
        let m_c = self.entry.constant_f32_scalar(mult);
        let m_b = self.entry.broadcast(m_c, &[], out_f32.clone());
        let scaled = self.entry.binary("multiply", acc_f32, m_b, out_f32.clone());
        let rounded = self
            .entry
            .unary("round-nearest-even", scaled, out_f32.clone());
        let oz_c = self.entry.constant_f32_scalar(out_zp as f32);
        let oz_b = self.entry.broadcast(oz_c, &[], out_f32.clone());
        let with_oz = self.entry.binary("add", rounded, oz_b, out_f32.clone());

        let lo_c = self.entry.constant_f32_scalar(-128.0);
        let hi_c = self.entry.constant_f32_scalar(127.0);
        let lo_b = self.entry.broadcast(lo_c, &[], out_f32.clone());
        let hi_b = self.entry.broadcast(hi_c, &[], out_f32.clone());
        let cl_lo = self.entry.binary("maximum", with_oz, lo_b, out_f32.clone());
        let cl = self.entry.binary("minimum", cl_lo, hi_b, out_f32);
        self.entry.convert(cl, out)
    }

    // ── In-graph RNG (ONNX Random*) ───────────────────────────────
    //
    // Lowers to XLA `rng` with UNIFORM (1) or NORMAL (2). Uses native
    // PJRT/XLA semantics — not bit-identical to RLX Philox/Ort on CPU.
    // `RngBackend::Zero` fills with a broadcast scalar zero instead.

    pub(crate) fn lower_rng_uniform(&self, low: f32, high: f32, out: Shape) -> i64 {
        if self.rng.backend == rlx_ir::RngBackend::Zero {
            let zero = self.entry.constant_f32_scalar(0.0);
            return self.entry.broadcast(zero, &[], out);
        }
        let a = self.entry.constant_f32_scalar(low);
        let b = self.entry.constant_f32_scalar(high);
        self.entry.rng(a, b, /*RNG_UNIFORM=*/ 1, out)
    }

    pub(crate) fn lower_rng_normal(&self, mean: f32, scale: f32, out: Shape) -> i64 {
        if self.rng.backend == rlx_ir::RngBackend::Zero {
            let zero = self.entry.constant_f32_scalar(0.0);
            return self.entry.broadcast(zero, &[], out);
        }
        let a = self.entry.constant_f32_scalar(mean);
        let b = self.entry.constant_f32_scalar(scale);
        self.entry.rng(a, b, /*RNG_NORMAL=*/ 2, out)
    }

    // ── Sample ────────────────────────────────────────────────────
    //
    // Decomposition:
    //   * temperature == 0 → argmax via topk(k=1)
    //   * top_k > 0 → filter logits below the k-th largest to -inf
    //   * top_p < 1.0 → filter via threshold = sorted_logits at the
    //     boundary index (first k where cumsum(softmax) ≥ top_p);
    //     no scatter-back needed because the kept set is exactly
    //     the largest-N logits, expressible as a value threshold
    //   * temperature  > 0 → multinomial via inverse-CDF on a
    //     uniform random [B] sample.
    //
    // RNG: XLA's `rng` op with UNIFORM distribution. Bit-exact match
    // to CUDA's Philox state would require lowering the same
    // counter-encoded seed, which the framework doesn't expose
    // through `rng-bit-generator` in a portable way, so we
    // deliberately don't aim for bit parity here — only that the
    // distribution is correct.

    pub(crate) fn lower_sample(
        &mut self,
        logits_id: NodeId,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        seed: u64,
        out: Shape,
    ) -> i64 {
        let _ = seed;
        let logits = self.hlo(logits_id);
        let dims = self.ir_shape_dims(logits_id);
        assert_eq!(dims.len(), 2, "Op::Sample expects [B, V] logits");
        let b = dims[0];
        let v = dims[1];
        let bv_f32 = Shape::array(prim::F32, &[b, v]);
        let b_s32 = Shape::array(prim::S32, &[b]);

        if temperature == 0.0 {
            // Greedy: argmax via topk(k=1) on the value axis, then
            // squeeze.
            let topk_shape = Shape::array(prim::F32, &[b, 1]);
            let topk_idx_f32 =
                self.lower_topk_inner(logits, &dims, prim::F32, 1, topk_shape.clone());
            let squeezed = self
                .entry
                .reshape(topk_idx_f32, Shape::array(prim::F32, &[b]));
            return if out.element_type == prim::F32 {
                squeezed
            } else {
                self.entry.convert(squeezed, out)
            };
        }

        // Scale by 1/temperature.
        let inv_t = self.entry.constant_f32_scalar(1.0 / temperature);
        let inv_t_b = self.entry.broadcast(inv_t, &[], bv_f32.clone());
        let mut logits = self
            .entry
            .binary("multiply", logits, inv_t_b, bv_f32.clone());

        // Optional top-k filter: zero out values below the k-th
        // largest by replacing them with -inf.
        if top_k > 0 && (top_k as i64) < v {
            let k_i = top_k as i64;
            // Sort descending paired with iota indices.
            let cmp = self.builder.computation("topk_cmp_for_sample");
            let key_s = Shape::scalar(prim::F32);
            let val_s = Shape::scalar(prim::S32);
            let p0 = cmp.parameter(0, "kx", key_s.clone());
            let p1 = cmp.parameter(1, "ky", key_s.clone());
            let _ = cmp.parameter(2, "vx", val_s.clone());
            let _ = cmp.parameter(3, "vy", val_s.clone());
            let r = cmp.compare(p0, p1, "GT", Shape::scalar(prim::PRED));
            cmp.set_root(r);
            cmp.set_program_shape(ProgramShape {
                parameters: vec![key_s.clone(), key_s.clone(), val_s.clone(), val_s.clone()],
                parameter_names: vec!["kx".into(), "ky".into(), "vx".into(), "vy".into()],
                result: Shape::scalar(prim::PRED),
            });
            let idx = self.entry.iota(1, Shape::array(prim::S32, &[b, v]));
            let tup = Shape::tuple(vec![bv_f32.clone(), Shape::array(prim::S32, &[b, v])]);
            let sorted = self.entry.sort(&[logits, idx], &cmp, 1, true, tup);
            let sorted_vals = self.entry.get_tuple_element(sorted, 0, bv_f32.clone());
            // Threshold = sorted_vals[..., k-1]
            let starts = vec![0, k_i - 1];
            let limits = vec![b, k_i];
            let strides = vec![1, 1];
            let kth = self.entry.slice(
                sorted_vals,
                &starts,
                &limits,
                &strides,
                Shape::array(prim::F32, &[b, 1]),
            );
            let kth_b = self.entry.broadcast(
                self.entry.reshape(kth, Shape::array(prim::F32, &[b])),
                &[0],
                bv_f32.clone(),
            );
            let mask = self
                .entry
                .compare(logits, kth_b, "LT", Shape::array(prim::PRED, &[b, v]));
            let neg_inf = self.entry.constant_f32_scalar(f32::NEG_INFINITY);
            let neg_inf_b = self.entry.broadcast(neg_inf, &[], bv_f32.clone());
            logits = self.entry.select(mask, neg_inf_b, logits, bv_f32.clone());
        }

        // Optional top-p (nucleus) filter. Idea: the kept set is the
        // smallest contiguous prefix of the sorted-descending logits
        // whose softmaxed cumulative probability mass first reaches
        // `top_p`. Because the kept set is exactly the largest-N
        // logits, we can express the filter as a value threshold —
        // no "scatter back to original order" needed. The threshold
        // is the value of the boundary token (smallest kept).
        if top_p < 1.0 - 1e-7 {
            // Sort logits descending (we don't need indices here,
            // but the comparator API takes paired k/v). Use iota for
            // the unused value half — XLA's sort needs a comparator
            // and we already have the topk_cmp_for_sample shape.
            let cmp = self.builder.computation("topp_cmp");
            let key_s = Shape::scalar(prim::F32);
            let val_s = Shape::scalar(prim::S32);
            let p0 = cmp.parameter(0, "kx", key_s.clone());
            let p1 = cmp.parameter(1, "ky", key_s.clone());
            let _ = cmp.parameter(2, "vx", val_s.clone());
            let _ = cmp.parameter(3, "vy", val_s.clone());
            let r = cmp.compare(p0, p1, "GT", Shape::scalar(prim::PRED));
            cmp.set_root(r);
            cmp.set_program_shape(ProgramShape {
                parameters: vec![key_s.clone(), key_s.clone(), val_s.clone(), val_s.clone()],
                parameter_names: vec!["kx".into(), "ky".into(), "vx".into(), "vy".into()],
                result: Shape::scalar(prim::PRED),
            });
            let idx = self.entry.iota(1, Shape::array(prim::S32, &[b, v]));
            let tup = Shape::tuple(vec![bv_f32.clone(), Shape::array(prim::S32, &[b, v])]);
            let sorted = self.entry.sort(&[logits, idx], &cmp, 1, true, tup);
            let sorted_vals = self.entry.get_tuple_element(sorted, 0, bv_f32.clone());

            // softmax of sorted vals → cumsum along last axis.
            let s_probs = self.lower_softmax_id(sorted_vals, bv_f32.clone(), 1);
            let s_cum = self.scan_along_last_axis(s_probs, &[b, v], prim::F32, "add", 0.0);

            // Find the boundary — smallest k such that cum[b, k] >= p.
            // First-true via cumsum-of-bool == 1.
            let p_const = self.entry.constant_f32_scalar(top_p);
            let p_b = self.entry.broadcast(p_const, &[], bv_f32.clone());
            let above = self
                .entry
                .compare(s_cum, p_b, "GE", Shape::array(prim::PRED, &[b, v]));
            let above_s32 = self.entry.convert(above, Shape::array(prim::S32, &[b, v]));
            let above_cumcount =
                self.scan_along_last_axis(above_s32, &[b, v], prim::S32, "add", 0.0);
            let one_s32 = self.entry.constant_s32_scalar(1);
            let one_b32 = self
                .entry
                .broadcast(one_s32, &[], Shape::array(prim::S32, &[b, v]));
            let first_geq = self.entry.compare(
                above_cumcount,
                one_b32,
                "EQ",
                Shape::array(prim::PRED, &[b, v]),
            );

            // Threshold[b] = sorted_vals[b, first_geq_idx]. We pull
            // it out by select(first_geq, sorted_vals, -inf) followed
            // by reduce-max along axis 1.
            let neg_inf = self.entry.constant_f32_scalar(f32::NEG_INFINITY);
            let neg_inf_b = self.entry.broadcast(neg_inf, &[], bv_f32.clone());
            let masked_for_thresh =
                self.entry
                    .select(first_geq, sorted_vals, neg_inf_b, bv_f32.clone());
            let red = self.reducer("maximum", prim::F32);
            let init_neg = self.entry.constant_f32_scalar(f32::NEG_INFINITY);
            let init_neg_s = self.entry.convert(init_neg, Shape::scalar(prim::F32));
            let threshold = self.entry.reduce(
                masked_for_thresh,
                init_neg_s,
                &red,
                &[1],
                Shape::array(prim::F32, &[b]),
            );

            // Apply the threshold to the original logits: anything
            // strictly below the boundary value is replaced with
            // -inf. Ties at the boundary are kept (matches HF /
            // Llama-style "include the first overshooting token").
            let thresh_b = self.entry.broadcast(threshold, &[0], bv_f32.clone());
            let keep =
                self.entry
                    .compare(logits, thresh_b, "GE", Shape::array(prim::PRED, &[b, v]));
            let neg_inf2 = self.entry.constant_f32_scalar(f32::NEG_INFINITY);
            let neg_inf_b2 = self.entry.broadcast(neg_inf2, &[], bv_f32.clone());
            logits = self.entry.select(keep, logits, neg_inf_b2, bv_f32.clone());
        }

        // softmax → probs → cumsum (cdf).
        let probs = self.lower_softmax_id(logits, bv_f32.clone(), 1);
        let cdf = self.scan_along_last_axis(probs, &[b, v], prim::F32, "add", 0.0);

        // Uniform random [B] in [0, 1).
        let zero = self.entry.constant_f32_scalar(0.0);
        let one = self.entry.constant_f32_scalar(1.0);
        let u = self.entry.rng(
            zero,
            one,
            /*UNIFORM=*/ 1,
            Shape::array(prim::F32, &[b]),
        );
        let u_b = self.entry.broadcast(u, &[0], bv_f32.clone());

        // Find the first column where cdf >= u.
        let ge = self
            .entry
            .compare(cdf, u_b, "GE", Shape::array(prim::PRED, &[b, v]));
        let ge_s32 = self.entry.convert(ge, Shape::array(prim::S32, &[b, v]));
        let cumcount = self.scan_along_last_axis(ge_s32, &[b, v], prim::S32, "add", 0.0);
        let one_s32 = self.entry.constant_s32_scalar(1);
        let one_b = self
            .entry
            .broadcast(one_s32, &[], Shape::array(prim::S32, &[b, v]));
        let first_eq = self
            .entry
            .compare(cumcount, one_b, "EQ", Shape::array(prim::PRED, &[b, v]));
        let idx_iota = self.entry.iota(1, Shape::array(prim::S32, &[b, v]));
        let zero_s32 = self.entry.constant_s32_scalar(0);
        let zero_s32_b = self
            .entry
            .broadcast(zero_s32, &[], Shape::array(prim::S32, &[b, v]));
        let masked = self.entry.select(
            first_eq,
            idx_iota,
            zero_s32_b,
            Shape::array(prim::S32, &[b, v]),
        );
        // reduce-max along axis 1 → [B] s32.
        let red = self.reducer("maximum", prim::S32);
        let init = self.entry.constant_s32_scalar(0);
        let token_s32 = self.entry.reduce(masked, init, &red, &[1], b_s32);
        // → f32.
        if out.element_type == prim::F32 {
            self.entry.convert(token_s32, out)
        } else {
            let f = self.entry.convert(token_s32, Shape::array(prim::F32, &[b]));
            if out.element_type == prim::F32 {
                f
            } else {
                self.entry.convert(f, out)
            }
        }
    }

    /// argmax via topk-1 on `x` of shape `dims` (last axis is the
    /// reduction). Returns f32 indices reshaped to `out_shape`.
    pub(crate) fn lower_topk_inner(
        &mut self,
        x: i64,
        dims: &[i64],
        prim_ty: i32,
        k: usize,
        out_shape: Shape,
    ) -> i64 {
        let last_axis = (dims.len() - 1) as i64;
        let cmp = self.builder.computation("topk_inner_descending");
        let key_s = Shape::scalar(prim_ty);
        let val_s = Shape::scalar(prim::S32);
        let p0 = cmp.parameter(0, "kx", key_s.clone());
        let p1 = cmp.parameter(1, "ky", key_s.clone());
        let _ = cmp.parameter(2, "vx", val_s.clone());
        let _ = cmp.parameter(3, "vy", val_s.clone());
        let r = cmp.compare(p0, p1, "GT", Shape::scalar(prim::PRED));
        cmp.set_root(r);
        cmp.set_program_shape(ProgramShape {
            parameters: vec![key_s.clone(), key_s.clone(), val_s.clone(), val_s.clone()],
            parameter_names: vec!["kx".into(), "ky".into(), "vx".into(), "vy".into()],
            result: Shape::scalar(prim::PRED),
        });
        let idx = self.entry.iota(last_axis, Shape::array(prim::S32, dims));
        let tup = Shape::tuple(vec![
            Shape::array(prim_ty, dims),
            Shape::array(prim::S32, dims),
        ]);
        let sorted = self.entry.sort(&[x, idx], &cmp, last_axis, true, tup);
        let sorted_idx = self
            .entry
            .get_tuple_element(sorted, 1, Shape::array(prim::S32, dims));
        let mut starts = vec![0i64; dims.len()];
        let mut limits = dims.to_vec();
        let strides = vec![1i64; dims.len()];
        starts[last_axis as usize] = 0;
        limits[last_axis as usize] = k as i64;
        let mut slice_dims = dims.to_vec();
        slice_dims[last_axis as usize] = k as i64;
        let sliced = self.entry.slice(
            sorted_idx,
            &starts,
            &limits,
            &strides,
            Shape::array(prim::S32, &slice_dims),
        );
        self.entry.convert(sliced, out_shape)
    }

    pub(crate) fn lower_selective_scan(
        &mut self,
        x_id: NodeId,
        delta_id: NodeId,
        a_id: NodeId,
        b_id: NodeId,
        c_id: NodeId,
        state_size: usize,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let delta = self.hlo(delta_id);
        let a = self.hlo(a_id);
        let bb = self.hlo(b_id);
        let cc = self.hlo(c_id);
        let x_dims = self.ir_shape_dims(x_id); // [B, L, D]
        let b = x_dims[0];
        let l = x_dims[1];
        let d = x_dims[2];
        let n = state_size as i64;

        let bd = Shape::array(prim::F32, &[b, d]);
        let bn = Shape::array(prim::F32, &[b, n]);
        let bdn = Shape::array(prim::F32, &[b, d, n]);
        let bld = Shape::array(prim::F32, &[b, l, d]);
        let s32_scalar = Shape::scalar(prim::S32);

        // Carry tuple (extended): (i, state, outs, x, delta, a, b, c).
        // HLO `while` only takes the carry as parameter, so the
        // per-step inputs are threaded through it.
        let bld_t = bld.clone();
        let dn_a = Shape::array(prim::F32, &[d, n]);
        let bln = Shape::array(prim::F32, &[b, l, n]);
        let big_tup = Shape::tuple(vec![
            s32_scalar.clone(),
            bdn.clone(),
            bld_t.clone(),
            bld_t.clone(),
            bld_t.clone(),
            dn_a.clone(),
            bln.clone(),
            bln.clone(),
        ]);

        // Initial values, packed into the carry tuple.
        let i0 = self.entry.constant_s32_scalar(0);
        let zero_f = self.entry.constant_f32_scalar(0.0);
        let state0 = self.entry.broadcast(zero_f, &[], bdn.clone());
        let outs0 = self.entry.broadcast(zero_f, &[], bld.clone());
        let big_init = self
            .entry
            .tuple(&[i0, state0, outs0, x, delta, a, bb, cc], big_tup.clone());

        // Reducer for the body's per-step axis-2 sum. Create it BEFORE
        // the body so it lands earlier in the computation list — XLA's
        // proto deserializer rejects forward references.
        let red = self
            .builder
            .make_reducer(&format!("scan_red_{}", state_size), "add", prim::F32);

        // Cond: i < L.
        let cond2 = self.builder.computation("scan_cond_big");
        let p = cond2.parameter(0, "carry", big_tup.clone());
        let ci2 = cond2.get_tuple_element(p, 0, s32_scalar.clone());
        let l_c = cond2.constant_s32_scalar(l as i32);
        let pr = cond2.compare(ci2, l_c, "LT", Shape::scalar(prim::PRED));
        cond2.set_root(pr);
        cond2.set_program_shape(ProgramShape {
            parameters: vec![big_tup.clone()],
            parameter_names: vec!["carry".into()],
            result: Shape::scalar(prim::PRED),
        });

        // Body.
        let body = self.builder.computation("scan_body");
        let bp = body.parameter(0, "carry", big_tup.clone());
        let bi = body.get_tuple_element(bp, 0, s32_scalar.clone());
        let bstate = body.get_tuple_element(bp, 1, bdn.clone());
        let bouts = body.get_tuple_element(bp, 2, bld.clone());
        let bx = body.get_tuple_element(bp, 3, bld.clone());
        let bdelta = body.get_tuple_element(bp, 4, bld.clone());
        let ba = body.get_tuple_element(bp, 5, dn_a.clone());
        let bb_t = body.get_tuple_element(bp, 6, bln.clone());
        let bc_t = body.get_tuple_element(bp, 7, bln.clone());

        let zero_idx = body.constant_s32_scalar(0);

        // Slice x/delta at step i: dynamic-slice [B, 1, D], reshape [B, D].
        let x_slc = body.dynamic_slice(
            bx,
            &[zero_idx, bi, zero_idx],
            vec![b, 1, d],
            Shape::array(prim::F32, &[b, 1, d]),
        );
        let x_t = body.reshape(x_slc, bd.clone());
        let d_slc = body.dynamic_slice(
            bdelta,
            &[zero_idx, bi, zero_idx],
            vec![b, 1, d],
            Shape::array(prim::F32, &[b, 1, d]),
        );
        let delta_t = body.reshape(d_slc, bd.clone());
        // Slice b/c at step i: dynamic-slice [B, 1, N], reshape [B, N].
        let b_slc = body.dynamic_slice(
            bb_t,
            &[zero_idx, bi, zero_idx],
            vec![b, 1, n],
            Shape::array(prim::F32, &[b, 1, n]),
        );
        let b_step = body.reshape(b_slc, bn.clone());
        let c_slc = body.dynamic_slice(
            bc_t,
            &[zero_idx, bi, zero_idx],
            vec![b, 1, n],
            Shape::array(prim::F32, &[b, 1, n]),
        );
        let c_step = body.reshape(c_slc, bn.clone());

        // decay = exp(delta_t[..., None] * a[None, ..., :])  [B, D, N]
        let delta_3 = body.broadcast(delta_t, &[0, 1], bdn.clone());
        let a_3 = body.broadcast(ba, &[1, 2], bdn.clone());
        let prod_da = body.binary("multiply", delta_3, a_3, bdn.clone());
        let decay = body.unary("exponential", prod_da, bdn.clone());

        // update = delta_t[...,None] * b_step[:,None,:] * x_t[...,None]
        let b_3 = body.broadcast(b_step, &[0, 2], bdn.clone());
        let x_3 = body.broadcast(x_t, &[0, 1], bdn.clone());
        let db = body.binary("multiply", delta_3, b_3, bdn.clone());
        let update = body.binary("multiply", db, x_3, bdn.clone());

        let state_decayed = body.binary("multiply", bstate, decay, bdn.clone());
        let new_state = body.binary("add", state_decayed, update, bdn.clone());

        // y[t] = sum_n new_state[b,d,n] * c_step[b,n]  → [B, D]
        let c_3 = body.broadcast(c_step, &[0, 2], bdn.clone());
        let prod_sc = body.binary("multiply", new_state, c_3, bdn.clone());
        // reduce sum over axis 2 (reducer was hoisted above body).
        let init_v = body.constant_f32_scalar(0.0);
        let y_t = body.reduce(prod_sc, init_v, &red, &[2], bd.clone());
        // Reshape y_t [B, D] → [B, 1, D] to fit dynamic-update-slice.
        let y_t_3 = body.reshape(y_t, Shape::array(prim::F32, &[b, 1, d]));
        let new_outs =
            body.dynamic_update_slice(bouts, y_t_3, &[zero_idx, bi, zero_idx], bld.clone());

        // i' = i + 1.
        let one_c = body.constant_s32_scalar(1);
        let bi1 = body.binary("add", bi, one_c, s32_scalar.clone());

        // Re-pack the body's output tuple in the same shape as the
        // input carry.
        let new_tup = body.tuple(
            &[bi1, new_state, new_outs, bx, bdelta, ba, bb_t, bc_t],
            big_tup.clone(),
        );
        body.set_root(new_tup);
        body.set_program_shape(ProgramShape {
            parameters: vec![big_tup.clone()],
            parameter_names: vec!["carry".into()],
            result: big_tup.clone(),
        });

        // While.
        let final_tup = self.entry.while_loop(big_init, &cond2, &body, big_tup);
        // Extract outputs (slot 2) — that's the result.
        let outs = self.entry.get_tuple_element(final_tup, 2, bld);
        if out.element_type == prim::F32 {
            outs
        } else {
            self.entry.convert(outs, out)
        }
    }
}
