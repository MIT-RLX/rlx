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

//! Composed HLO lowers for newly claimed TPU ops (QAT, argmax, complex, FMA,
//! AxialRope2d, Im2Col, ConvTranspose, PerTensor FP8 Scaled*).

use crate::hlo::{
    ConvDimNumbers, DotDimNumbers, GatherDimNumbers, Literal, LiteralData, ProgramShape, Shape,
    Window, WindowDim, prim, prim_of,
};
use rlx_ir::op::ScaleMode;
use rlx_ir::{DType, NodeId, ScaleLayout, ScaledFormat};

use super::*;

/// Per-tensor 8-bit scaled formats that compose to F32 HLO (LUT decode + scale).
pub(crate) fn scaled_fp8_hlo_ok(format: ScaledFormat, layout: ScaleLayout) -> bool {
    matches!(layout, ScaleLayout::PerTensor) && format.bit_width() == 8
}

impl<'a> LowerCtx<'a> {
    /// `out = a * b + c` (mul then add — two roundings; matches `LowerFma`).
    pub(crate) fn lower_fma(&self, a_id: NodeId, b_id: NodeId, c_id: NodeId, out: Shape) -> i64 {
        let a = self.hlo(a_id);
        let b = self.hlo(b_id);
        let c = self.hlo(c_id);
        let (a, b) = self.broadcast_pair_to(a, b, a_id, b_id, &out.dimensions);
        let prod = self.entry.binary("multiply", a, b, out.clone());
        let c_dims = self.ir_shape_dims(c_id);
        let c_b = self.broadcast_to_target(c, &c_dims, out.clone());
        self.entry.binary("add", prod, c_b, out)
    }

    /// LayerNorm2d NCHW: normalize across channel axis (1) at each spatial site.
    pub(crate) fn lower_layer_norm2d(
        &mut self,
        x_id: NodeId,
        gamma_id: NodeId,
        beta_id: NodeId,
        eps: f32,
        out: Shape,
    ) -> i64 {
        self.lower_layernorm(x_id, gamma_id, beta_id, 1, eps, out)
    }

    /// FakeQuantize Fixed / PerBatch (symmetric). EMA is rejected.
    pub(crate) fn lower_fake_quantize(
        &mut self,
        inputs: &[NodeId],
        bits: u8,
        axis: Option<usize>,
        scale_mode: ScaleMode,
        out: Shape,
    ) -> i64 {
        let q_max = match bits {
            8 => 127.0f32,
            4 => 7.0,
            2 => 1.0,
            n => panic!("rlx-tpu FakeQuantize: unsupported bits {n}"),
        };
        let x = self.hlo(inputs[0]);
        let dims = out.dimensions.clone();
        let prim_ty = out.element_type;
        let scale = match scale_mode {
            ScaleMode::Fixed => {
                assert!(
                    inputs.len() >= 2,
                    "rlx-tpu FakeQuantize Fixed requires a scale state input"
                );
                let s = self.hlo(inputs[1]);
                let s_dims = self.ir_shape_dims(inputs[1]);
                self.broadcast_q_param(s, &s_dims, axis, &dims, prim_ty)
            }
            ScaleMode::PerBatch => {
                let abs_x = self.entry.unary("abs", x, out.clone());
                let (max_abs, kept) =
                    self.reduce_abs_max_for_fake_quant(abs_x, axis, &dims, prim_ty);
                let q = self.const_in_dtype(prim_ty, q_max);
                let q_shape = if kept.is_empty() {
                    Shape::scalar(prim_ty)
                } else {
                    Shape::array(prim_ty, &kept)
                };
                let q_b = self.entry.broadcast(q, &[], q_shape.clone());
                let s = self.entry.binary("divide", max_abs, q_b, q_shape.clone());
                let eps = self.const_in_dtype(prim_ty, 1e-12);
                let eps_b = self.entry.broadcast(eps, &[], q_shape.clone());
                let s = self.entry.binary("maximum", s, eps_b, q_shape);
                if kept.is_empty() {
                    self.entry.broadcast(s, &[], out.clone())
                } else {
                    self.broadcast_align(s, &kept, out.clone())
                }
            }
            ScaleMode::EMA { .. } => panic!(
                "rlx-tpu FakeQuantize EMA is not supported — use Fixed or PerBatch \
                 (or Device::Cpu)."
            ),
        };

        let one = self.const_in_dtype(prim_ty, 1.0);
        let one_b = self.entry.broadcast(one, &[], out.clone());
        let inv = self.entry.binary("divide", one_b, scale, out.clone());
        let scaled = self.entry.binary("multiply", x, inv, out.clone());
        let rounded = self.entry.unary("round-nearest-even", scaled, out.clone());
        let neg_q = self.const_in_dtype(prim_ty, -q_max);
        let pos_q = self.const_in_dtype(prim_ty, q_max);
        let neg_b = self.entry.broadcast(neg_q, &[], out.clone());
        let pos_b = self.entry.broadcast(pos_q, &[], out.clone());
        let clipped = self.entry.binary("maximum", rounded, neg_b, out.clone());
        let clipped = self.entry.binary("minimum", clipped, pos_b, out.clone());
        self.entry.binary("multiply", clipped, scale, out)
    }

    /// Broadcast a quant scale / zero-point to `target` (scalar or along `axis`).
    pub(crate) fn broadcast_q_param(
        &self,
        s: i64,
        s_dims: &[i64],
        axis: Option<usize>,
        target: &[i64],
        prim_ty: i32,
    ) -> i64 {
        match axis {
            None => {
                let flat = if s_dims.iter().product::<i64>() == 1 {
                    self.entry.reshape(s, Shape::scalar(prim_ty))
                } else {
                    s
                };
                self.entry
                    .broadcast(flat, &[], Shape::array(prim_ty, target))
            }
            Some(ax) => self.broadcast_param_to_axis(s, s_dims, ax as i64, target, prim_ty),
        }
    }

    /// Reduce max over all axes except optional channel axis; returns (value, kept_dims).
    pub(crate) fn reduce_abs_max_for_fake_quant(
        &mut self,
        x: i64,
        axis: Option<usize>,
        dims: &[i64],
        prim_ty: i32,
    ) -> (i64, Vec<i64>) {
        let x_dt = if prim_ty == prim::F64 {
            DType::F64
        } else {
            DType::F32
        };
        match axis {
            None => {
                let mut v = x;
                let mut cur = dims.to_vec();
                for ax in (0..dims.len()).rev() {
                    let out_dims: Vec<i64> = cur
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &d)| if i == ax { None } else { Some(d) })
                        .collect();
                    v = self.reduce_one(v, ax as i64, "maximum", f32::NEG_INFINITY, x_dt, out_dims);
                    cur.remove(ax);
                }
                (v, vec![])
            }
            Some(ax) => {
                let mut v = x;
                let mut cur = dims.to_vec();
                let axes: Vec<usize> = (0..dims.len()).filter(|&i| i != ax).collect();
                for &a in axes.iter().rev() {
                    let adj = axes.iter().filter(|&&x| x < a).count();
                    let ax_i = (a - adj) as i64;
                    let out_dims: Vec<i64> = cur
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &d)| if i == ax_i as usize { None } else { Some(d) })
                        .collect();
                    v = self.reduce_one(v, ax_i, "maximum", f32::NEG_INFINITY, x_dt, out_dims);
                    cur.remove(ax_i as usize);
                }
                let mut kept = vec![1i64; dims.len()];
                kept[ax] = dims[ax];
                let reshaped = self.entry.reshape(v, Shape::array(prim_ty, &kept));
                (reshaped, kept)
            }
        }
    }

    /// ArgMax / ArgMin along `axis` via iota + sort; optional `keep_dim`.
    pub(crate) fn lower_argmax_argmin(
        &mut self,
        x_id: NodeId,
        axis: usize,
        keep_dim: bool,
        is_max: bool,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let dims = self.ir_shape_dims(x_id);
        let prim_ty = prim_of(self.dtype(x_id));
        let ax = axis as i64;

        let iota_shape = Shape::array(prim::S32, &dims);
        let indices = self.entry.iota(ax, iota_shape);

        let cmp = self.builder.computation(if is_max {
            "argmax_descending"
        } else {
            "argmin_ascending"
        });
        let key_s = Shape::scalar(prim_ty);
        let val_s = Shape::scalar(prim::S32);
        let p0 = cmp.parameter(0, "kx", key_s.clone());
        let p1 = cmp.parameter(1, "ky", key_s.clone());
        let _p2 = cmp.parameter(2, "vx", val_s.clone());
        let _p3 = cmp.parameter(3, "vy", val_s.clone());
        let dir = if is_max { "GT" } else { "LT" };
        let r = cmp.compare(p0, p1, dir, Shape::scalar(prim::PRED));
        cmp.set_root(r);
        cmp.set_program_shape(ProgramShape {
            parameters: vec![key_s.clone(), key_s, val_s.clone(), val_s],
            parameter_names: vec!["kx".into(), "ky".into(), "vx".into(), "vy".into()],
            result: Shape::scalar(prim::PRED),
        });

        let val_full = Shape::array(prim_ty, &dims);
        let idx_full = Shape::array(prim::S32, &dims);
        let tup = Shape::tuple(vec![val_full, idx_full.clone()]);
        let sorted = self.entry.sort(&[x, indices], &cmp, ax, true, tup);
        let sorted_idx = self.entry.get_tuple_element(sorted, 1, idx_full);

        let mut starts = vec![0i64; dims.len()];
        let mut limits = dims.clone();
        let strides = vec![1i64; dims.len()];
        starts[ax as usize] = 0;
        limits[ax as usize] = 1;
        let mut slice_dims = dims;
        slice_dims[ax as usize] = 1;
        let sliced = self.entry.slice(
            sorted_idx,
            &starts,
            &limits,
            &strides,
            Shape::array(prim::S32, &slice_dims),
        );
        if keep_dim {
            self.entry.convert(sliced, out)
        } else {
            let squeezed: Vec<i64> = slice_dims
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
                .collect();
            let reshaped = self
                .entry
                .reshape(sliced, Shape::array(prim::S32, &squeezed));
            self.entry.convert(reshaped, out)
        }
    }

    /// `|z|² = re² + im²` for native XLA C64.
    pub(crate) fn lower_complex_norm_sq(&self, z_id: NodeId, out: Shape) -> i64 {
        let z = self.hlo(z_id);
        let dims = self.ir_shape_dims(z_id);
        let re = self.entry.unary("real", z, Shape::array(prim::F32, &dims));
        let im = self.entry.unary("imag", z, Shape::array(prim::F32, &dims));
        let re2 = self.entry.binary("multiply", re, re, out.clone());
        let im2 = self.entry.binary("multiply", im, im, out.clone());
        self.entry.binary("add", re2, im2, out)
    }

    /// Complex conjugate via native XLA `conjugate` (C64 / C128).
    pub(crate) fn lower_conjugate(&self, z_id: NodeId, out: Shape) -> i64 {
        let z = self.hlo(z_id);
        self.entry.unary("conjugate", z, out)
    }

    /// Wirtinger: `∂|z|²/∂z̄ = z`, scaled by real upstream `g` → `g·z`.
    pub(crate) fn lower_complex_norm_sq_backward(
        &self,
        z_id: NodeId,
        g_id: NodeId,
        out: Shape,
    ) -> i64 {
        let z = self.hlo(z_id);
        let g = self.hlo(g_id);
        let dims = self.ir_shape_dims(z_id);
        let re = self.entry.unary("real", z, Shape::array(prim::F32, &dims));
        let im = self.entry.unary("imag", z, Shape::array(prim::F32, &dims));
        let g_dims = self.ir_shape_dims(g_id);
        let g_b = self.broadcast_to_target(g, &g_dims, Shape::array(prim::F32, &dims));
        let gre = self
            .entry
            .binary("multiply", g_b, re, Shape::array(prim::F32, &dims));
        let gim = self
            .entry
            .binary("multiply", g_b, im, Shape::array(prim::F32, &dims));
        self.entry.binary("complex", gre, gim, out)
    }

    /// ReluBackward: `dx = dy where x > 0 else 0`.
    pub(crate) fn lower_relu_backward(&self, x_id: NodeId, dy_id: NodeId, out: Shape) -> i64 {
        let x = self.hlo(x_id);
        let dy = self.hlo(dy_id);
        let zero = self.const_in_dtype(out.element_type, 0.0);
        let zero_b = self.entry.broadcast(zero, &[], out.clone());
        let pred = self
            .entry
            .compare(x, zero_b, "GT", Shape::pred(&out.dimensions));
        self.entry.select(pred, dy, zero_b, out)
    }

    /// SAM2-style axial 2-D RoPE: constant cos/sin tables + interleaved rotate.
    pub(crate) fn lower_axial_rope2d(
        &mut self,
        x_id: NodeId,
        end_x: usize,
        end_y: usize,
        head_dim: usize,
        num_heads: usize,
        theta: f32,
        repeat_factor: usize,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let dims = self.ir_shape_dims(x_id);
        assert_eq!(dims.len(), 3, "rlx-tpu AxialRope2d: [B,S,H*D]");
        let (b, seq, hidden) = (dims[0], dims[1] as usize, dims[2] as usize);
        let hd = head_dim;
        let nh = num_heads;
        assert!(
            hd > 0 && hd.is_multiple_of(4),
            "rlx-tpu AxialRope2d: head_dim"
        );
        assert_eq!(hidden, nh * hd, "rlx-tpu AxialRope2d: hidden != nh*hd");
        let half = hd / 2;
        let q4 = hd / 4;
        let spatial = end_x * end_y;
        let repeat = repeat_factor.max(1);
        assert_eq!(
            seq,
            spatial * repeat,
            "rlx-tpu AxialRope2d: seq != end_x*end_y*repeat"
        );
        let prim_ty = prim::F32;

        let mut freqs = vec![0f32; q4];
        for i in 0..q4 {
            freqs[i] = 1.0 / theta.powf((4 * i) as f32 / hd as f32);
        }
        let mut cos_x = vec![0f32; seq * q4];
        let mut sin_x = vec![0f32; seq * q4];
        let mut cos_y = vec![0f32; seq * q4];
        let mut sin_y = vec![0f32; seq * q4];
        for tok in 0..seq {
            let pos = tok / repeat;
            let tx = (pos % end_x) as f32;
            let ty = (pos / end_x) as f32;
            for c in 0..q4 {
                let ax = tx * freqs[c];
                let ay = ty * freqs[c];
                cos_x[tok * q4 + c] = ax.cos();
                sin_x[tok * q4 + c] = ax.sin();
                cos_y[tok * q4 + c] = ay.cos();
                sin_y[tok * q4 + c] = ay.sin();
            }
        }
        let tab_shape = Shape::array(prim_ty, &[seq as i64, q4 as i64]);
        let cos_x = self.entry.constant(Literal {
            shape: tab_shape.clone(),
            data: LiteralData::F32(cos_x),
        });
        let sin_x = self.entry.constant(Literal {
            shape: tab_shape.clone(),
            data: LiteralData::F32(sin_x),
        });
        let cos_y = self.entry.constant(Literal {
            shape: tab_shape.clone(),
            data: LiteralData::F32(cos_y),
        });
        let sin_y = self.entry.constant(Literal {
            shape: tab_shape.clone(),
            data: LiteralData::F32(sin_y),
        });

        let x4 = self.entry.reshape(
            x,
            Shape::array(prim_ty, &[b, seq as i64, nh as i64, hd as i64]),
        );
        let half_shape = Shape::array(prim_ty, &[b, seq as i64, nh as i64, half as i64]);
        let strides4 = [1i64; 4];
        let x_lo = self.entry.slice(
            x4,
            &[0, 0, 0, 0],
            &[b, seq as i64, nh as i64, half as i64],
            &strides4,
            half_shape.clone(),
        );
        let x_hi = self.entry.slice(
            x4,
            &[0, 0, 0, half as i64],
            &[b, seq as i64, nh as i64, hd as i64],
            &strides4,
            half_shape.clone(),
        );

        let rotate = |this: &mut LowerCtx<'_>, half_x: i64, cos: i64, sin: i64| -> i64 {
            let pair_shape = Shape::array(prim_ty, &[b, seq as i64, nh as i64, q4 as i64, 2]);
            let pairs = this.entry.reshape(half_x, pair_shape.clone());
            let even_shape = Shape::array(prim_ty, &[b, seq as i64, nh as i64, q4 as i64, 1]);
            let strides5 = [1i64; 5];
            let x_even = this.entry.slice(
                pairs,
                &[0, 0, 0, 0, 0],
                &[b, seq as i64, nh as i64, q4 as i64, 1],
                &strides5,
                even_shape.clone(),
            );
            let x_odd = this.entry.slice(
                pairs,
                &[0, 0, 0, 0, 1],
                &[b, seq as i64, nh as i64, q4 as i64, 2],
                &strides5,
                even_shape.clone(),
            );
            let cos_b = this.entry.reshape(
                cos,
                Shape::array(prim_ty, &[1, seq as i64, 1, q4 as i64, 1]),
            );
            let sin_b = this.entry.reshape(
                sin,
                Shape::array(prim_ty, &[1, seq as i64, 1, q4 as i64, 1]),
            );
            let cos_b = this.broadcast_to_target(
                cos_b,
                &[1, seq as i64, 1, q4 as i64, 1],
                even_shape.clone(),
            );
            let sin_b = this.broadcast_to_target(
                sin_b,
                &[1, seq as i64, 1, q4 as i64, 1],
                even_shape.clone(),
            );
            let xe_c = this
                .entry
                .binary("multiply", x_even, cos_b, even_shape.clone());
            let xo_s = this
                .entry
                .binary("multiply", x_odd, sin_b, even_shape.clone());
            let y_even = this
                .entry
                .binary("subtract", xe_c, xo_s, even_shape.clone());
            let xo_c = this
                .entry
                .binary("multiply", x_odd, cos_b, even_shape.clone());
            let xe_s = this
                .entry
                .binary("multiply", x_even, sin_b, even_shape.clone());
            let y_odd = this.entry.binary("add", xo_c, xe_s, even_shape.clone());
            let y_pairs = this.entry.concat(&[y_even, y_odd], 4, pair_shape);
            this.entry.reshape(y_pairs, half_shape.clone())
        };

        let y_lo = rotate(self, x_lo, cos_x, sin_x);
        let y_hi = rotate(self, x_hi, cos_y, sin_y);
        let y4 = self.entry.concat(
            &[y_lo, y_hi],
            3,
            Shape::array(prim_ty, &[b, seq as i64, nh as i64, hd as i64]),
        );
        self.entry.reshape(y4, out)
    }

    /// NCHW im2col → `[N·H_out·W_out, C·kH·kW]` via pad + strided slices.
    pub(crate) fn lower_im2col(
        &mut self,
        x_id: NodeId,
        kernel_size: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        out: Shape,
    ) -> i64 {
        assert_eq!(kernel_size.len(), 2, "rlx-tpu Im2Col: 2D NCHW only");
        let x = self.hlo(x_id);
        let dims = self.ir_shape_dims(x_id);
        assert_eq!(dims.len(), 4, "rlx-tpu Im2Col: NCHW only");
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let (kh, kw) = (kernel_size[0] as i64, kernel_size[1] as i64);
        let (sh, sw) = (stride[0] as i64, stride[1] as i64);
        let (ph, pw) = (padding[0] as i64, padding[1] as i64);
        let (dh, dw) = (dilation[0] as i64, dilation[1] as i64);
        let h_out = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        let w_out = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
        let prim_ty = out.element_type;

        let padded = if ph > 0 || pw > 0 {
            let zero = self.const_in_dtype(prim_ty, 0.0);
            let pad_cfg = vec![(0, 0, 0), (0, 0, 0), (ph, ph, 0), (pw, pw, 0)];
            let pad_dims = [n, c, h + 2 * ph, w + 2 * pw];
            self.entry
                .pad(x, zero, pad_cfg, Shape::array(prim_ty, &pad_dims))
        } else {
            x
        };

        let mut patches = Vec::with_capacity((kh * kw) as usize);
        let win_shape = Shape::array(prim_ty, &[n, c, h_out, w_out, 1]);
        for ki in 0..kh {
            for kj in 0..kw {
                let start = [0, 0, ki * dh, kj * dw];
                let limits = [
                    n,
                    c,
                    ki * dh + (h_out - 1) * sh + 1,
                    kj * dw + (w_out - 1) * sw + 1,
                ];
                let strides = [1, 1, sh, sw];
                let win = self.entry.slice(
                    padded,
                    &start,
                    &limits,
                    &strides,
                    Shape::array(prim_ty, &[n, c, h_out, w_out]),
                );
                patches.push(self.entry.reshape(win, win_shape.clone()));
            }
        }
        let stacked = self.entry.concat(
            &patches,
            4,
            Shape::array(prim_ty, &[n, c, h_out, w_out, kh * kw]),
        );
        let t = self.entry.transpose(
            stacked,
            &[0, 2, 3, 1, 4],
            Shape::array(prim_ty, &[n, h_out, w_out, c, kh * kw]),
        );
        self.entry.reshape(t, out)
    }

    /// ConvTranspose via ConvGeneralDilated (lhs dilation = stride, flipped kernel).
    /// PyTorch weight `[C_in, C_out/g, *k]` → OIHW `[C_out, C_in/g, *k]` + window_reversal.
    pub(crate) fn lower_conv_transpose(
        &mut self,
        x_id: NodeId,
        w_id: NodeId,
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        groups: usize,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let w = self.hlo(w_id);
        let x_dims = self.ir_shape_dims(x_id);
        let w_dims = self.ir_shape_dims(w_id);
        let out_dims = out.dimensions.clone();
        let n_spatial = x_dims.len() - 2;
        assert_eq!(
            w_dims.len(),
            x_dims.len(),
            "rlx-tpu ConvTranspose: rank match"
        );
        assert_eq!(stride.len(), n_spatial);
        assert_eq!(padding.len(), n_spatial);
        assert_eq!(dilation.len(), n_spatial);
        let g = groups.max(1) as i64;
        let prim_ty = out.element_type;
        let c_in = x_dims[1];
        let c_out = out_dims[1];
        assert!(
            c_in % g == 0 && c_out % g == 0,
            "rlx-tpu ConvTranspose: groups must divide channels"
        );
        let c_in_per_g = c_in / g;
        let c_out_per_g = c_out / g;

        // [C_in, C_out/g, *k] → [C_out, C_in/g, *k]
        let w_oi = if g == 1 {
            let mut perm = vec![1, 0];
            perm.extend(2..2 + n_spatial as i64);
            let mut oi_dims = vec![c_out, c_in];
            oi_dims.extend_from_slice(&w_dims[2..]);
            self.entry
                .transpose(w, &perm, Shape::array(prim_ty, &oi_dims))
        } else {
            let mut split_dims = vec![g, c_in_per_g, c_out_per_g];
            split_dims.extend_from_slice(&w_dims[2..]);
            let split = self.entry.reshape(w, Shape::array(prim_ty, &split_dims));
            // [G, Cin/g, Cout/g, *k] → [G, Cout/g, Cin/g, *k]
            let mut perm = vec![0i64, 2, 1];
            perm.extend(3..3 + n_spatial as i64);
            let mut perm_dims = vec![g, c_out_per_g, c_in_per_g];
            perm_dims.extend_from_slice(&w_dims[2..]);
            let perm_w = self
                .entry
                .transpose(split, &perm, Shape::array(prim_ty, &perm_dims));
            let mut oi_dims = vec![c_out, c_in_per_g];
            oi_dims.extend_from_slice(&w_dims[2..]);
            self.entry.reshape(perm_w, Shape::array(prim_ty, &oi_dims))
        };

        let mut window_dims = Vec::with_capacity(n_spatial);
        for i in 0..n_spatial {
            let k = w_dims[2 + i];
            let s = stride[i] as i64;
            let p = padding[i] as i64;
            let d = dilation[i] as i64;
            let in_sz = x_dims[2 + i];
            let out_sz = out_dims[2 + i];
            let pad_lo = d * (k - 1) - p;
            let pad_hi = out_sz - 1 - s * (in_sz - 1) + p;
            window_dims.push(WindowDim {
                size: k,
                stride: 1,
                padding_low: pad_lo,
                padding_high: pad_hi,
                window_dilation: d,
                base_dilation: s,
                window_reversal: true,
            });
        }
        let window = Window {
            dimensions: window_dims,
        };
        let spatial: Vec<i64> = (2..2 + n_spatial as i64).collect();
        let cdn = ConvDimNumbers {
            input_batch_dim: 0,
            input_feature_dim: 1,
            input_spatial_dims: spatial.clone(),
            kernel_output_feature_dim: 0,
            kernel_input_feature_dim: 1,
            kernel_spatial_dims: spatial.clone(),
            output_batch_dim: 0,
            output_feature_dim: 1,
            output_spatial_dims: spatial,
        };
        self.entry.convolution(x, w_oi, window, cdn, g, out)
    }

    fn scaled_decode_lut(&self, format: ScaledFormat) -> i64 {
        let lut: Vec<f32> = (0..256)
            .map(|c| rlx_ir::lowp_codec::decode(format, c as u8))
            .collect();
        self.entry.constant(Literal {
            shape: Shape::array(prim::F32, &[256]),
            data: LiteralData::F32(lut),
        })
    }

    /// U8 codes → f32 via 256-entry decode LUT × per-tensor scale.
    pub(crate) fn lower_scaled_dequantize(
        &mut self,
        codes_id: NodeId,
        scale_id: NodeId,
        format: ScaledFormat,
        scale_layout: ScaleLayout,
        out: Shape,
    ) -> i64 {
        assert!(
            scaled_fp8_hlo_ok(format, scale_layout),
            "rlx-tpu ScaledDequantize: only PerTensor 8-bit formats compose to HLO"
        );
        let codes = self.hlo(codes_id);
        let scale = self.hlo(scale_id);
        let dims = out.dimensions.clone();
        let idx = self.entry.convert(codes, Shape::array(prim::S32, &dims));
        let lut = self.scaled_decode_lut(format);
        let dn = GatherDimNumbers {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: dims.len() as i64,
        };
        let decoded = self
            .entry
            .gather(lut, idx, dn, vec![1], Shape::array(prim::F32, &dims));
        let scale_dims = self.ir_shape_dims(scale_id);
        let scale_b = self.broadcast_to_target(scale, &scale_dims, out.clone());
        self.entry.binary("multiply", decoded, scale_b, out)
    }

    /// Per-tensor scale = max(|x|) / max_finite(format).
    pub(crate) fn lower_scaled_quant_scale(
        &mut self,
        x_id: NodeId,
        format: ScaledFormat,
        scale_layout: ScaleLayout,
        out: Shape,
    ) -> i64 {
        assert!(
            scaled_fp8_hlo_ok(format, scale_layout),
            "rlx-tpu ScaledQuantScale: only PerTensor 8-bit formats compose to HLO"
        );
        let x = self.hlo(x_id);
        let dims = self.ir_shape_dims(x_id);
        let prim_ty = prim::F32;
        let abs_x = self.entry.unary("abs", x, Shape::array(prim_ty, &dims));
        let mut v = abs_x;
        let mut cur = dims;
        for ax in (0..cur.len()).rev() {
            let out_dims: Vec<i64> = cur
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| if i == ax { None } else { Some(d) })
                .collect();
            v = self.reduce_one(
                v,
                ax as i64,
                "maximum",
                f32::NEG_INFINITY,
                DType::F32,
                out_dims,
            );
            cur.remove(ax);
        }
        let maxf = self.const_scalar_f32(format.max_finite());
        let scale = self.entry.binary("divide", v, maxf, Shape::scalar(prim_ty));
        let one = self.const_scalar_f32(1.0);
        let zero = self.const_scalar_f32(0.0);
        let pred = self
            .entry
            .compare(scale, zero, "GT", Shape::scalar(prim::PRED));
        let scale = self.entry.select(pred, scale, one, Shape::scalar(prim_ty));
        self.entry.reshape(scale, out)
    }

    /// FP8 ScaledMatMul as dequant(lhs)·dequant(rhs)ᵀ (+ optional bias) in F32.
    pub(crate) fn lower_scaled_matmul(
        &mut self,
        inputs: &[NodeId],
        lhs_format: ScaledFormat,
        rhs_format: ScaledFormat,
        scale_layout: ScaleLayout,
        has_bias: bool,
        out: Shape,
    ) -> i64 {
        assert!(
            scaled_fp8_hlo_ok(lhs_format, scale_layout)
                && scaled_fp8_hlo_ok(rhs_format, scale_layout),
            "rlx-tpu ScaledMatMul: only PerTensor 8-bit formats compose to HLO"
        );
        let lhs_id = inputs[0];
        let rhs_id = inputs[1];
        let lhs_scale_id = inputs[2];
        let rhs_scale_id = inputs[3];
        let lhs_dims = self.ir_shape_dims(lhs_id);
        let rhs_dims = self.ir_shape_dims(rhs_id);
        assert_eq!(lhs_dims.len(), 2, "rlx-tpu ScaledMatMul: rank-2 TN");
        assert_eq!(rhs_dims.len(), 2, "rlx-tpu ScaledMatMul: rank-2 TN");
        let (m, k) = (lhs_dims[0], lhs_dims[1]);
        let (n, k2) = (rhs_dims[0], rhs_dims[1]);
        assert_eq!(k, k2, "rlx-tpu ScaledMatMul: K mismatch");

        let lhs = self.lower_scaled_dequantize(
            lhs_id,
            lhs_scale_id,
            lhs_format,
            scale_layout,
            Shape::array(prim::F32, &[m, k]),
        );
        let rhs = self.lower_scaled_dequantize(
            rhs_id,
            rhs_scale_id,
            rhs_format,
            scale_layout,
            Shape::array(prim::F32, &[n, k]),
        );
        // TN: out[i,j] = Σ_p lhs[i,p]·rhs[j,p]
        let dn = DotDimNumbers {
            lhs_contracting: vec![1],
            rhs_contracting: vec![1],
            lhs_batch: vec![],
            rhs_batch: vec![],
        };
        let mut y = self.entry.dot_general(lhs, rhs, dn, out.clone());
        if has_bias {
            assert!(inputs.len() >= 5, "rlx-tpu ScaledMatMul: missing bias");
            let bias = self.hlo(inputs[4]);
            let bias_dims = self.ir_shape_dims(inputs[4]);
            let bias_b = self.broadcast_to_target(bias, &bias_dims, out.clone());
            y = self.entry.binary("add", y, bias_b, out);
        }
        y
    }
}
