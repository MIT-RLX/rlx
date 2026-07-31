// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shape-inferred HIR builder — mirrors [`crate::infer::GraphExt`] for
//! [`super::HirModule`], emitting [`super::HirOp::Mir`] primitives.

use crate::hir::{HirModule, HirNodeId};
use crate::op::*;
use crate::quant::{ScaleLayout, ScaledFormat};
use crate::shape;
use crate::{DType, Op, Shape};

/// Interpolation mode for [`HirMut::grid_sample2d`] (PyTorch `grid_sample`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GridMode {
    Nearest,
    Bilinear,
    Bicubic,
}

/// Out-of-bounds handling for [`HirMut::grid_sample2d`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GridPad {
    Zeros,
    Border,
    Reflection,
}

/// Mutable HIR builder view — implements [`HirGraphExt`] without conflicting
/// with [`HirModule`]'s block-level `rms_norm` / `rope` methods.
pub struct HirMut<'a>(pub &'a mut HirModule);

impl<'a> HirMut<'a> {
    pub fn new(hir: &'a mut HirModule) -> Self {
        Self(hir)
    }

    pub fn inner(&mut self) -> &mut HirModule {
        self.0
    }

    pub fn input(&mut self, name: impl Into<String>, shape: Shape) -> HirNodeId {
        self.0.input(name, shape)
    }

    pub fn param(&mut self, name: impl Into<String>, shape: Shape) -> HirNodeId {
        self.0.param(name, shape)
    }

    pub fn set_outputs(&mut self, outputs: Vec<HirNodeId>) {
        self.0.set_outputs(outputs);
    }

    /// Scaled dot-product attention with a caller-supplied mask tensor
    /// (matches legacy [`crate::Graph::attention`]).
    pub fn attention(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        mask: HirNodeId,
        num_heads: usize,
        head_dim: usize,
        shape: Shape,
    ) -> HirNodeId {
        self.0.mir(
            crate::ops::attention::attention_kind_op(
                num_heads,
                head_dim,
                MaskKind::Custom,
                None,
                None,
            ),
            vec![q, k, v, mask],
            shape,
        )
    }

    /// Additive attention bias `[batch, num_heads, query_len, key_len]`.
    pub fn attention_bias(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        bias: HirNodeId,
        num_heads: usize,
        head_dim: usize,
        shape: Shape,
    ) -> HirNodeId {
        self.0.mir(
            crate::ops::attention::attention_kind_op(
                num_heads,
                head_dim,
                MaskKind::Bias,
                None,
                None,
            ),
            vec![q, k, v, bias],
            shape,
        )
    }

    /// Bilinear 2-D resize of NCHW `x` to `[N, C, out_h, out_w]` — see
    /// [`Self::resize_separable`].
    pub fn resize_bilinear2d(
        &mut self,
        x: HirNodeId,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
    ) -> HirNodeId {
        self.resize_separable(x, out_h, out_w, align_corners, false, false)
    }

    /// Bicubic 2-D resize of NCHW `x` to `[N, C, out_h, out_w]` — see
    /// [`Self::resize_separable`] (PyTorch cubic-convolution kernel, `a = -0.75`).
    pub fn resize_bicubic2d(
        &mut self,
        x: HirNodeId,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
    ) -> HirNodeId {
        self.resize_separable(x, out_h, out_w, align_corners, true, false)
    }

    /// Antialiased bilinear resize (`_upsample_bilinear2d_aa`): downsampling
    /// widens the filter by the scale factor and renormalizes.
    pub fn resize_bilinear2d_aa(
        &mut self,
        x: HirNodeId,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
    ) -> HirNodeId {
        self.resize_separable(x, out_h, out_w, align_corners, false, true)
    }

    /// Antialiased bicubic resize (`_upsample_bicubic2d_aa`).
    pub fn resize_bicubic2d_aa(
        &mut self,
        x: HirNodeId,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
    ) -> HirNodeId {
        self.resize_separable(x, out_h, out_w, align_corners, true, true)
    }

    /// Separable 2-D resize (bilinear/bicubic, optionally antialiased) of NCHW `x`.
    ///
    /// Every one of these filters is separable, so a resize is exactly two 1-D
    /// resamples (W then H), each a matmul by a constant `[in, out]`
    /// interpolation matrix built per PyTorch's `align_corners` (and antialias)
    /// convention. This lands `upsample_{bilinear,bicubic}2d(_aa)` on the
    /// universal `MatMul`/`reshape`/`transpose` ops — bit-exact on every backend,
    /// with no dedicated kernel.
    fn resize_separable(
        &mut self,
        x: HirNodeId,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
        cubic: bool,
        antialias: bool,
    ) -> HirNodeId {
        let in_s = self.shape(x).clone();
        let n = in_s.dim(0).unwrap_static();
        let c = in_s.dim(1).unwrap_static();
        let h_in = in_s.dim(2).unwrap_static();
        let w_in = in_s.dim(3).unwrap_static();

        // ── W (last) axis: [N·C·H_in, W_in] @ [W_in, out_w] ──
        let wmat = {
            let data = interp_matrix_bytes(w_in, out_w, align_corners, cubic, antialias);
            self.0.mir(
                Op::Constant { data },
                vec![],
                Shape::new(&[w_in, out_w], DType::F32),
            )
        };
        let x2 = self.reshape_(x, vec![(n * c * h_in) as i64, w_in as i64]);
        let xw = self.mm(x2, wmat);
        let xw4 = self.reshape_(xw, vec![n as i64, c as i64, h_in as i64, out_w as i64]);

        // ── H axis: move H last, matmul by [H_in, out_h], move back ──
        let xt = self.transpose_(xw4, vec![0, 1, 3, 2]); // [N, C, out_w, H_in]
        let xt2 = self.reshape_(xt, vec![(n * c * out_w) as i64, h_in as i64]);
        let hmat = {
            let data = interp_matrix_bytes(h_in, out_h, align_corners, cubic, antialias);
            self.0.mir(
                Op::Constant { data },
                vec![],
                Shape::new(&[h_in, out_h], DType::F32),
            )
        };
        let xh = self.mm(xt2, hmat);
        let xh4 = self.reshape_(xh, vec![n as i64, c as i64, out_w as i64, out_h as i64]);
        self.transpose_(xh4, vec![0, 1, 3, 2]) // [N, C, out_h, out_w]
    }

    // ── grid_sample helpers (operate on [S,1] coordinate tensors) ────────────
    fn gs_const(&mut self, v: f64) -> HirNodeId {
        let data = (v as f32).to_le_bytes().to_vec();
        self.0
            .mir(Op::Constant { data }, vec![], Shape::new(&[1], DType::F32))
    }
    /// `a <op> b` for same-shape (or scalar-`b`) operands; shape = shape(a).
    fn gs_bin(&mut self, op: BinaryOp, a: HirNodeId, b: HirNodeId) -> HirNodeId {
        let s = self.shape(a).clone();
        self.0.mir(Op::Binary(op), vec![a, b], s)
    }
    /// `a <op> scalar` — `a` stays the full-size lhs so CPU broadcast is valid.
    fn gs_scalar(&mut self, op: BinaryOp, a: HirNodeId, v: f64) -> HirNodeId {
        let k = self.gs_const(v);
        self.gs_bin(op, a, k)
    }
    fn gs_round(&mut self, a: HirNodeId) -> HirNodeId {
        let s = self.shape(a).clone();
        self.0.mir(Op::Activation(Activation::Round), vec![a], s)
    }
    fn gs_abs(&mut self, a: HirNodeId) -> HirNodeId {
        let s = self.shape(a).clone();
        self.0.mir(Op::Activation(Activation::Abs), vec![a], s)
    }
    /// `Compare` yields a bool tensor; cast it to f32 {0,1} so it can be used in
    /// arithmetic on the f32 arena (a raw bool operand under-sizes / mis-reads).
    fn gs_cmp_f32(&mut self, op: CmpOp, a: HirNodeId, b: HirNodeId) -> HirNodeId {
        let s = self.shape(a).clone();
        let cmp = self.0.mir(Op::Compare(op), vec![a, b], s.clone());
        self.0.mir(Op::Cast { to: DType::F32 }, vec![cmp], s)
    }
    /// `floor(a) = round(a) − (round(a) > a)` — exact for all finite `a`.
    fn gs_floor(&mut self, a: HirNodeId) -> HirNodeId {
        let r = self.gs_round(a);
        let gt = self.gs_cmp_f32(CmpOp::Gt, r, a);
        self.gs_bin(BinaryOp::Sub, r, gt)
    }
    fn gs_clamp(&mut self, a: HirNodeId, lo: f64, hi: f64) -> HirNodeId {
        let a = self.gs_scalar(BinaryOp::Max, a, lo);
        self.gs_scalar(BinaryOp::Min, a, hi)
    }
    fn gs_one_minus(&mut self, a: HirNodeId) -> HirNodeId {
        let neg = self.gs_scalar(BinaryOp::Mul, a, -1.0);
        self.gs_scalar(BinaryOp::Add, neg, 1.0)
    }
    fn gs_expand(&mut self, a: HirNodeId, s: usize, c: usize) -> HirNodeId {
        self.0.mir(
            Op::Expand {
                target_shape: vec![s as i64, c as i64],
            },
            vec![a],
            Shape::new(&[s, c], DType::F32),
        )
    }

    /// Unnormalize a `[S,1]` normalized grid coord to pixel space, then apply
    /// the padding transform to the *continuous* coordinate (PyTorch order).
    fn gs_coord(&mut self, g: HirNodeId, size: usize, ac: bool, pad: GridPad) -> HirNodeId {
        let coord = if ac {
            let a = 0.5 * (size as f64 - 1.0);
            let ga = self.gs_scalar(BinaryOp::Mul, g, a);
            self.gs_scalar(BinaryOp::Add, ga, a)
        } else {
            let a = size as f64 * 0.5;
            let ga = self.gs_scalar(BinaryOp::Mul, g, a);
            self.gs_scalar(BinaryOp::Add, ga, a - 0.5)
        };
        match pad {
            GridPad::Zeros => coord,
            GridPad::Border => self.gs_clamp(coord, 0.0, size as f64 - 1.0),
            GridPad::Reflection => {
                let r = self.gs_reflect(coord, size, ac);
                self.gs_clamp(r, 0.0, size as f64 - 1.0)
            }
        }
    }

    /// Reflect a continuous coordinate into `[0, size-1]` per PyTorch
    /// `reflect_coordinates` (span/min depend on `align_corners`).
    fn gs_reflect(&mut self, coord: HirNodeId, size: usize, ac: bool) -> HirNodeId {
        let (min, span) = if ac {
            (0.0, size as f64 - 1.0)
        } else {
            (-0.5, size as f64)
        };
        if span <= 0.0 {
            return self.gs_scalar(BinaryOp::Mul, coord, 0.0); // size 1 → all → 0
        }
        let cm = self.gs_scalar(BinaryOp::Sub, coord, min);
        let x = self.gs_abs(cm);
        let q = self.gs_scalar(BinaryOp::Mul, x, 1.0 / span);
        let fq = self.gs_floor(q);
        let span_fq = self.gs_scalar(BinaryOp::Mul, fq, span);
        let extra = self.gs_bin(BinaryOp::Sub, x, span_fq); // x mod span
        // parity = fq − 2·floor(fq/2)  ∈ {0,1}
        let half = self.gs_scalar(BinaryOp::Mul, fq, 0.5);
        let fhalf = self.gs_floor(half);
        let two_fhalf = self.gs_scalar(BinaryOp::Mul, fhalf, 2.0);
        let parity = self.gs_bin(BinaryOp::Sub, fq, two_fhalf);
        let even = self.gs_scalar(BinaryOp::Add, extra, min);
        let neg_extra = self.gs_scalar(BinaryOp::Mul, extra, -1.0);
        let odd = self.gs_scalar(BinaryOp::Add, neg_extra, min + span);
        let inv_p = self.gs_one_minus(parity);
        let e = self.gs_bin(BinaryOp::Mul, even, inv_p);
        let o = self.gs_bin(BinaryOp::Mul, odd, parity);
        self.gs_bin(BinaryOp::Add, e, o)
    }

    /// `1` if the integer index `idx ∈ [0, size-1]`, else `0` — computed with
    /// `max`/`min` arithmetic (no `Compare`, so no bool-arena hazard):
    /// `1 − min(1, relu(−idx) + relu(idx − (size−1)))`.
    fn gs_inbounds(&mut self, idx: HirNodeId, size: usize) -> HirNodeId {
        let neg = self.gs_scalar(BinaryOp::Mul, idx, -1.0);
        let lo = self.gs_scalar(BinaryOp::Max, neg, 0.0);
        let hi_shift = self.gs_scalar(BinaryOp::Add, idx, -(size as f64 - 1.0));
        let hi = self.gs_scalar(BinaryOp::Max, hi_shift, 0.0);
        let oob = self.gs_bin(BinaryOp::Add, lo, hi);
        let oobc = self.gs_scalar(BinaryOp::Min, oob, 1.0);
        let neg_oob = self.gs_scalar(BinaryOp::Mul, oobc, -1.0);
        self.gs_scalar(BinaryOp::Add, neg_oob, 1.0)
    }

    /// Zeros-padding validity mask (`1` where integer sample `(xi,yi)` is in
    /// bounds).
    fn gs_validity(&mut self, xi: HirNodeId, yi: HirNodeId, w: usize, h: usize) -> HirNodeId {
        let vx = self.gs_inbounds(xi, w);
        let vy = self.gs_inbounds(yi, h);
        self.gs_bin(BinaryOp::Mul, vx, vy)
    }

    /// Gather one integer sample `(xi,yi)` from `table [S_in,C]`, weighted by
    /// `weight [S,1]` (× validity for zeros-padding). Returns `[S,C]`.
    #[allow(clippy::too_many_arguments)]
    fn gs_tap(
        &mut self,
        table: HirNodeId,
        xi: HirNodeId,
        yi: HirNodeId,
        weight: HirNodeId,
        w: usize,
        h: usize,
        s: usize,
        c: usize,
        pad: GridPad,
    ) -> HirNodeId {
        let cxi = self.gs_clamp(xi, 0.0, w as f64 - 1.0);
        let cyi = self.gs_clamp(yi, 0.0, h as f64 - 1.0);
        let row = self.gs_scalar(BinaryOp::Mul, cyi, w as f64);
        let flat = self.gs_bin(BinaryOp::Add, row, cxi);
        let flat1d = self.reshape_(flat, vec![s as i64]);
        let gathered = self.gather_(table, flat1d, 0); // [S, C]
        let w_full = if pad == GridPad::Zeros {
            let v = self.gs_validity(xi, yi, w, h);
            self.gs_bin(BinaryOp::Mul, weight, v)
        } else {
            weight
        };
        let w_exp = self.gs_expand(w_full, s, c);
        self.gs_bin(BinaryOp::Mul, gathered, w_exp)
    }

    /// Keys cubic weight `cubic_filter(dist)` as a tensor, where `dist` is a
    /// scalar-affine function `slope·t + bias` of `t [S,1]`, evaluated on the
    /// branch (`inner` = |dist|<1) selected at build time by the tap position.
    fn gs_cubic_w(&mut self, t: HirNodeId, slope: f64, bias: f64, inner: bool) -> HirNodeId {
        const A: f64 = -0.75;
        // dist = slope·t + bias (|dist| within one branch across t∈[0,1)).
        let d = {
            let st = self.gs_scalar(BinaryOp::Mul, t, slope);
            self.gs_scalar(BinaryOp::Add, st, bias)
        };
        let ad = self.gs_abs(d);
        let ad2 = self.gs_bin(BinaryOp::Mul, ad, ad);
        if inner {
            // (A+2)|x|³ − (A+3)|x|² + 1
            let p = self.gs_scalar(BinaryOp::Mul, ad, A + 2.0);
            let p = self.gs_scalar(BinaryOp::Add, p, -(A + 3.0));
            let p = self.gs_bin(BinaryOp::Mul, p, ad2);
            self.gs_scalar(BinaryOp::Add, p, 1.0)
        } else {
            // A|x|³ − 5A|x|² + 8A|x| − 4A
            let p = self.gs_scalar(BinaryOp::Mul, ad, A);
            let p = self.gs_scalar(BinaryOp::Add, p, -5.0 * A);
            let p = self.gs_bin(BinaryOp::Mul, p, ad);
            let p = self.gs_scalar(BinaryOp::Add, p, 8.0 * A);
            let p = self.gs_bin(BinaryOp::Mul, p, ad);
            self.gs_scalar(BinaryOp::Add, p, -4.0 * A)
        }
    }

    /// `grid_sample(input[N,C,H,W], grid[N,H_out,W_out,2])` → `[N,C,H_out,W_out]`.
    ///
    /// Fully general (all `mode`/`padding`/`align_corners`), expressed only with
    /// universal ops (gather + `Round`-based floor + arithmetic) — no kernel.
    /// Batch is unrolled (shapes are static).
    pub fn grid_sample2d(
        &mut self,
        input: HirNodeId,
        grid: HirNodeId,
        mode: GridMode,
        pad: GridPad,
        ac: bool,
    ) -> HirNodeId {
        let in_s = self.shape(input).clone();
        let (n, c, h, w) = (
            in_s.dim(0).unwrap_static(),
            in_s.dim(1).unwrap_static(),
            in_s.dim(2).unwrap_static(),
            in_s.dim(3).unwrap_static(),
        );
        let g_s = self.shape(grid).clone();
        let (ho, wo) = (g_s.dim(1).unwrap_static(), g_s.dim(2).unwrap_static());
        let s = ho * wo;

        let mut batch_outs = Vec::with_capacity(n);
        for bn in 0..n {
            let in_n = self.narrow_(input, 0, bn, 1);
            let in_flat = self.reshape_(in_n, vec![c as i64, (h * w) as i64]);
            let table = self.transpose_(in_flat, vec![1, 0]); // [S_in, C]

            let g_n = self.narrow_(grid, 0, bn, 1);
            let g_flat = self.reshape_(g_n, vec![s as i64, 2]);
            let gx0 = self.narrow_(g_flat, 1, 0, 1);
            let gx = self.reshape_(gx0, vec![s as i64, 1]);
            let gy0 = self.narrow_(g_flat, 1, 1, 1);
            let gy = self.reshape_(gy0, vec![s as i64, 1]);
            let cx = self.gs_coord(gx, w, ac, pad);
            let cy = self.gs_coord(gy, h, ac, pad);

            let out_n = match mode {
                GridMode::Nearest => {
                    let ix = self.gs_round(cx);
                    let iy = self.gs_round(cy);
                    let ones = self.gs_scalar(BinaryOp::Mul, ix, 0.0); // [S,1] zeros
                    let ones = self.gs_scalar(BinaryOp::Add, ones, 1.0); // → ones
                    self.gs_tap(table, ix, iy, ones, w, h, s, c, pad)
                }
                GridMode::Bilinear => {
                    let ix0 = self.gs_floor(cx);
                    let iy0 = self.gs_floor(cy);
                    let ix1 = self.gs_scalar(BinaryOp::Add, ix0, 1.0);
                    let iy1 = self.gs_scalar(BinaryOp::Add, iy0, 1.0);
                    let wx1 = self.gs_bin(BinaryOp::Sub, cx, ix0);
                    let wy1 = self.gs_bin(BinaryOp::Sub, cy, iy0);
                    let wx0 = self.gs_one_minus(wx1);
                    let wy0 = self.gs_one_minus(wy1);
                    let taps = [
                        (ix0, iy0, wx0, wy0),
                        (ix1, iy0, wx1, wy0),
                        (ix0, iy1, wx0, wy1),
                        (ix1, iy1, wx1, wy1),
                    ];
                    let mut acc: Option<HirNodeId> = None;
                    for (xi, yi, wxk, wyk) in taps {
                        let weight = self.gs_bin(BinaryOp::Mul, wxk, wyk);
                        let contrib = self.gs_tap(table, xi, yi, weight, w, h, s, c, pad);
                        acc = Some(match acc {
                            None => contrib,
                            Some(a) => self.gs_bin(BinaryOp::Add, a, contrib),
                        });
                    }
                    acc.unwrap()
                }
                GridMode::Bicubic => {
                    let bx = self.gs_floor(cx);
                    let by = self.gs_floor(cy);
                    let tx = self.gs_bin(BinaryOp::Sub, cx, bx);
                    let ty = self.gs_bin(BinaryOp::Sub, cy, by);
                    // 4 taps at base+{-1,0,1,2}; weights are cubic_filter(dist).
                    let offs = [-1.0f64, 0.0, 1.0, 2.0];
                    // dist(t) for tap k: |k_off − t|; branch inner = (k in {0,1} rel).
                    let wx: Vec<HirNodeId> = (0..4)
                        .map(|k| {
                            let (slope, bias, inner) = cubic_dist_params(k);
                            self.gs_cubic_w(tx, slope, bias, inner)
                        })
                        .collect();
                    let wy: Vec<HirNodeId> = (0..4)
                        .map(|k| {
                            let (slope, bias, inner) = cubic_dist_params(k);
                            self.gs_cubic_w(ty, slope, bias, inner)
                        })
                        .collect();
                    let mut acc: Option<HirNodeId> = None;
                    for (ky, oy) in offs.iter().enumerate() {
                        let yi = self.gs_scalar(BinaryOp::Add, by, *oy);
                        for (kx, ox) in offs.iter().enumerate() {
                            let xi = self.gs_scalar(BinaryOp::Add, bx, *ox);
                            let weight = self.gs_bin(BinaryOp::Mul, wx[kx], wy[ky]);
                            let contrib = self.gs_tap(table, xi, yi, weight, w, h, s, c, pad);
                            acc = Some(match acc {
                                None => contrib,
                                Some(a) => self.gs_bin(BinaryOp::Add, a, contrib),
                            });
                        }
                    }
                    acc.unwrap()
                }
            };

            let out_t = self.transpose_(out_n, vec![1, 0]); // [C, S]
            let out_r = self.reshape_(out_t, vec![1, c as i64, ho as i64, wo as i64]);
            batch_outs.push(out_r);
        }
        if batch_outs.len() == 1 {
            batch_outs.pop().unwrap()
        } else {
            self.concat_(batch_outs, 0)
        }
    }

    // ── SPDNet / Riemannian manifold layers ─────────────────────────────────
    // Mirror `crate::Graph::{bimap,reeig,logeig,spd_batch_norm_transport}`
    // (see `ops/manifold.rs`) at the HIR level, emitting via `mir(...)`. These
    // operate on symmetric-positive-definite matrices and are F64-first (the
    // CPU kernels `expect_f64`); the caller must supply F64-typed inputs.

    /// BiMap (bilinear mapping) SPDNet layer: `Y = W · X · Wᵀ`.
    /// `w` is `[m, n]`, `x` is `[n, n]` symmetric SPD; output `Y` is `[m, m]`.
    pub fn bimap(&mut self, w: HirNodeId, x: HirNodeId) -> HirNodeId {
        let ws = self.shape(w).clone();
        let m = ws.dim(0).unwrap_static();
        let dtype = ws.dtype();
        self.0
            .mir(Op::BiMap, vec![w, x], Shape::new(&[m, m], dtype))
    }

    /// ReEig (eigenvalue rectification) SPDNet nonlinearity:
    /// `Y = U · max(ε, Σ) · Uᵀ`. `x` is `[n, n]` symmetric SPD; output `Y` is
    /// the same shape (the SPD analogue of ReLU). See [`Self::spectral_layer`].
    pub fn reeig(&mut self, x: HirNodeId, eps: f32) -> HirNodeId {
        self.spectral_layer(Op::ReEig { eps }, x)
    }

    /// LogEig SPDNet layer: `Y = logm(X) = U · log(Σ) · Uᵀ`. Maps the SPD
    /// manifold to the tangent space at the identity. `x` is `[n, n]`; output
    /// `Y` is the same shape. See [`Self::spectral_layer`].
    pub fn logeig(&mut self, x: HirNodeId, eps: f32) -> HirNodeId {
        self.spectral_layer(Op::LogEig { eps }, x)
    }

    /// Shared builder for the packed spectral layers (ReEig / LogEig): push the
    /// op (output `[2n²+n]` = `Y ∥ λ ∥ U`, so the backward reuses the
    /// eigendecomposition), then narrow + reshape the leading `Y` block to
    /// `[n, n]`. Mirrors `Graph::spectral_layer` in `ops/manifold.rs`.
    fn spectral_layer(&mut self, op: Op, x: HirNodeId) -> HirNodeId {
        let xs = self.shape(x).clone();
        let n = xs.dim(0).unwrap_static();
        let dtype = xs.dtype();
        let packed = self.0.mir(op, vec![x], Shape::new(&[2 * n * n + n], dtype));
        let y_flat = self.0.mir(
            Op::Narrow {
                axis: 0,
                start: 0,
                len: n * n,
            },
            vec![packed],
            Shape::new(&[n * n], dtype),
        );
        self.0.mir(
            Op::Reshape {
                new_shape: vec![n as i64, n as i64],
            },
            vec![y_flat],
            Shape::new(&[n, n], dtype),
        )
    }

    /// SPD batch-norm affine transport (eval / inference form):
    /// `Y_i = G^{1/2} (M^{-1/2} X_i M^{-1/2}) G^{1/2}`.
    /// `x` is `[batch, n, n]`, `mean` (frozen running Fréchet mean) and `g`
    /// (learnable SPD bias) are `[n, n]`. Output matches `x`. Mirrors
    /// `Graph::spd_batch_norm_transport`.
    pub fn spd_batch_norm_transport(
        &mut self,
        x: HirNodeId,
        mean: HirNodeId,
        g: HirNodeId,
        eps: f32,
    ) -> HirNodeId {
        let shape = self.shape(x).clone();
        self.0
            .mir(Op::SpdBatchNorm { eps }, vec![x, mean, g], shape)
    }
}

/// For bicubic tap `k` (positions `base + {-1,0,1,2}`), the distance to the
/// sample point `base+t` is `|off_k − t|`; return `(slope, bias, inner)` so the
/// weight is `cubic_filter(slope·t + bias)` on the correct kernel branch.
/// k=0 → dist 1+t (outer); k=1 → t (inner); k=2 → 1−t (inner); k=3 → 2−t (outer).
fn cubic_dist_params(k: usize) -> (f64, f64, bool) {
    match k {
        0 => (1.0, 1.0, false),  // 1 + t,  |·| ∈ [1,2)
        1 => (1.0, 0.0, true),   // t,       |·| ∈ [0,1)
        2 => (-1.0, 1.0, true),  // 1 − t,   |·| ∈ (0,1]
        _ => (-1.0, 2.0, false), // 2 − t,   |·| ∈ (1,2]
    }
}

/// Source coordinate for output index `j`, matching PyTorch's
/// `area_pixel_compute_source_index`. `cubic` keeps a negative source (bicubic),
/// while linear clamps it to 0.
fn resize_source_index(
    j: usize,
    l_in: usize,
    l_out: usize,
    align_corners: bool,
    cubic: bool,
) -> f64 {
    if align_corners {
        if l_out <= 1 {
            0.0
        } else {
            j as f64 * (l_in as f64 - 1.0) / (l_out as f64 - 1.0)
        }
    } else {
        let s = (j as f64 + 0.5) * (l_in as f64 / l_out as f64) - 0.5;
        if cubic { s } else { s.max(0.0) }
    }
}

/// PyTorch cubic-convolution kernel weights (Keys, `a = -0.75`) for fractional
/// offset `t ∈ [0, 1)`, over the 4 taps at `floor(src) + {-1, 0, 1, 2}`.
fn cubic_coeffs(t: f64) -> [f64; 4] {
    [
        cubic_filter(t + 1.0),
        cubic_filter(t),
        cubic_filter(1.0 - t),
        cubic_filter(2.0 - t),
    ]
}

/// Triangle (linear) reconstruction filter, support radius 1.
fn linear_filter(x: f64) -> f64 {
    let a = x.abs();
    if a < 1.0 { 1.0 - a } else { 0.0 }
}

/// Keys cubic filter (`a = -0.75`), support radius 2.
fn cubic_filter(x: f64) -> f64 {
    const A: f64 = -0.75;
    let a = x.abs();
    if a < 1.0 {
        ((A + 2.0) * a - (A + 3.0)) * a * a + 1.0
    } else if a < 2.0 {
        ((A * a - 5.0 * A) * a + 8.0 * A) * a - 4.0 * A
    } else {
        0.0
    }
}

/// Row-major `[l_in, l_out]` interpolation matrix (`out[.., j] = Σ_i in·M[i,j]`).
/// Non-antialiased path: linear (2 taps) or bicubic (4 taps), edge-clamped,
/// matching PyTorch's `upsample_{bilinear,bicubic}2d`.
fn interp_matrix_plain(l_in: usize, l_out: usize, align_corners: bool, cubic: bool) -> Vec<f32> {
    let mut m = vec![0f32; l_in * l_out];
    let clamp_idx = |i: i64| -> usize { i.clamp(0, l_in as i64 - 1) as usize };
    for j in 0..l_out {
        let src = resize_source_index(j, l_in, l_out, align_corners, cubic);
        let base = src.floor();
        let t = src - base;
        if cubic {
            let ix = base as i64;
            for (k, &w) in cubic_coeffs(t).iter().enumerate() {
                let idx = clamp_idx(ix - 1 + k as i64);
                m[idx * l_out + j] += w as f32;
            }
        } else {
            let i0 = clamp_idx(base as i64);
            let i1 = clamp_idx(base as i64 + 1);
            m[i0 * l_out + j] += (1.0 - t) as f32;
            m[i1 * l_out + j] += t as f32;
        }
    }
    m
}

/// Antialiased interpolation matrix, matching PyTorch's aa resize: for
/// downsampling (`scale ≥ 1`) the filter is widened by `scale` and the per-output
/// weights are renormalized to sum to 1; for upsampling it reduces to the plain
/// filter. `interp_size·0.5` = 1 (linear) / 2 (cubic).
fn interp_matrix_aa(l_in: usize, l_out: usize, align_corners: bool, cubic: bool) -> Vec<f32> {
    let scale = if align_corners {
        if l_out <= 1 {
            0.0
        } else {
            (l_in as f64 - 1.0) / (l_out as f64 - 1.0)
        }
    } else {
        l_in as f64 / l_out as f64
    };
    // Antialias only affects downsampling; PyTorch ignores it when upsampling.
    if scale < 1.0 {
        return interp_matrix_plain(l_in, l_out, align_corners, cubic);
    }
    let radius = if cubic { 2.0 } else { 1.0 };
    let (support, invscale) = (radius * scale, 1.0 / scale);
    let filt = |x: f64| {
        if cubic {
            cubic_filter(x)
        } else {
            linear_filter(x)
        }
    };
    let mut m = vec![0f32; l_in * l_out];
    for j in 0..l_out {
        let center = scale * (j as f64 + 0.5);
        // `as i64` truncates toward zero — matches C++ static_cast<int64_t>.
        let xmin = (((center - support + 0.5) as i64).max(0)) as usize;
        let xmax = (((center + support + 0.5) as i64).min(l_in as i64)) as usize;
        let mut total = 0.0f64;
        let mut ws = Vec::with_capacity(xmax.saturating_sub(xmin));
        for i in xmin..xmax {
            let w = filt((i as f64 + 0.5 - center) * invscale);
            ws.push(w);
            total += w;
        }
        if total != 0.0 {
            for (k, w) in ws.iter().enumerate() {
                m[(xmin + k) * l_out + j] = (w / total) as f32;
            }
        }
    }
    m
}

/// Row-major `[l_in, l_out]` f32 interpolation matrix as LE bytes.
fn interp_matrix_bytes(
    l_in: usize,
    l_out: usize,
    align_corners: bool,
    cubic: bool,
    antialias: bool,
) -> Vec<u8> {
    let m = if antialias {
        interp_matrix_aa(l_in, l_out, align_corners, cubic)
    } else {
        interp_matrix_plain(l_in, l_out, align_corners, cubic)
    };
    let mut bytes = Vec::with_capacity(m.len() * 4);
    for v in m {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Ergonomic shape-inferred building on [`HirMut`].
pub trait HirGraphExt {
    fn shape(&self, id: HirNodeId) -> &Shape;

    fn add_node(&mut self, op: Op, inputs: Vec<HirNodeId>, shape: Shape) -> HirNodeId;

    fn mm(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;

    /// Dynamically quantize `x` (logical `[rows, cols]`, blocks along the last
    /// axis) to low-precision `fmt` codes + a scale tensor, per `layout`.
    /// Returns `(codes, scale)`. Building block of [`scaled_matmul`](Self::scaled_matmul).
    fn scaled_quantize(
        &mut self,
        x: HirNodeId,
        fmt: ScaledFormat,
        layout: ScaleLayout,
    ) -> (HirNodeId, HirNodeId) {
        let xs = self.shape(x).clone();
        let cols = xs.dim(xs.rank() - 1).unwrap_static();
        let rows = xs.num_elements().unwrap() / cols.max(1);
        let scale_shape = match layout {
            ScaleLayout::PerTensor => Shape::new(&[1], layout.scale_dtype()),
            _ => Shape::new(
                &[rows, cols.div_ceil(layout.block() as usize)],
                layout.scale_dtype(),
            ),
        };
        let scale = self.add_node(
            Op::ScaledQuantScale {
                format: fmt,
                scale_layout: layout,
            },
            vec![x],
            scale_shape,
        );
        let codes = self.add_node(
            Op::ScaledQuantize {
                format: fmt,
                scale_layout: layout,
            },
            vec![x, scale],
            xs.with_dtype(DType::U8),
        );
        (codes, scale)
    }

    /// Native low-precision GEMM (TN: `lhs [m,k] · rhs [n,k]ᵀ → [m,n]` f32).
    /// Both operands are dynamically quantized to `fmt`/`layout`. `rhs` must be
    /// K-last (`[n, k]`). `fmt` may be any [`ScaledFormat`], including a
    /// parameterized `Custom` (e.g. `ScaledFormat::custom(3, 0)` for `f4e3m0`).
    fn scaled_matmul(
        &mut self,
        lhs: HirNodeId,
        rhs: HirNodeId,
        fmt: ScaledFormat,
        layout: ScaleLayout,
    ) -> HirNodeId {
        let m = self.shape(lhs).dim(0).unwrap_static();
        let n = self.shape(rhs).dim(0).unwrap_static();
        let (lq, ls) = self.scaled_quantize(lhs, fmt, layout);
        let (rq, rs) = self.scaled_quantize(rhs, fmt, layout);
        self.add_node(
            Op::ScaledMatMul {
                lhs_format: fmt,
                rhs_format: fmt,
                scale_layout: layout,
                has_bias: false,
            },
            vec![lq, rq, ls, rs],
            Shape::new(&[m, n], DType::F32),
        )
    }

    fn add(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;
    fn sub(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;
    fn mul(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;
    fn div(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;

    fn gelu(&mut self, x: HirNodeId) -> HirNodeId;
    fn gelu_approx(&mut self, x: HirNodeId) -> HirNodeId;
    fn silu(&mut self, x: HirNodeId) -> HirNodeId;
    fn relu(&mut self, x: HirNodeId) -> HirNodeId;
    fn exp(&mut self, x: HirNodeId) -> HirNodeId;
    fn recip(&mut self, x: HirNodeId) -> HirNodeId;
    fn sqrt(&mut self, x: HirNodeId) -> HirNodeId;
    fn neg(&mut self, x: HirNodeId) -> HirNodeId;
    fn tanh(&mut self, x: HirNodeId) -> HirNodeId;

    fn ln(&mut self, x: HirNodeId, gamma: HirNodeId, beta: HirNodeId, eps: f32) -> HirNodeId;
    fn group_norm(
        &mut self,
        x: HirNodeId,
        gamma: HirNodeId,
        beta: HirNodeId,
        num_groups: usize,
        eps: f32,
    ) -> HirNodeId;
    fn layer_norm2d(
        &mut self,
        x: HirNodeId,
        gamma: HirNodeId,
        beta: HirNodeId,
        eps: f32,
    ) -> HirNodeId;
    fn resize_nearest_2x(&mut self, x: HirNodeId) -> HirNodeId;
    fn conv2d(
        &mut self,
        x: HirNodeId,
        weight: HirNodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        groups: usize,
        out_shape: Shape,
    ) -> HirNodeId;
    fn conv_transpose2d(
        &mut self,
        x: HirNodeId,
        weight: HirNodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_padding: [usize; 2],
        groups: usize,
        out_shape: Shape,
    ) -> HirNodeId;
    fn rms_norm(&mut self, x: HirNodeId, gamma: HirNodeId, beta: HirNodeId, eps: f32) -> HirNodeId;

    /// DiT adaLN-Zero: `norm(x)·(1+scale)+shift`.
    fn ada_layer_norm(
        &mut self,
        x: HirNodeId,
        scale: HirNodeId,
        shift: HirNodeId,
        norm: AdaNormKind,
        eps: f32,
    ) -> HirNodeId;

    /// DiT gated residual: `x + gate·y`.
    fn gated_residual(&mut self, x: HirNodeId, y: HirNodeId, gate: HirNodeId) -> HirNodeId;

    fn sum(&mut self, x: HirNodeId, axes: Vec<usize>, keep_dim: bool) -> HirNodeId;
    fn mean(&mut self, x: HirNodeId, axes: Vec<usize>, keep_dim: bool) -> HirNodeId;
    fn sm(&mut self, x: HirNodeId, axis: i32) -> HirNodeId;

    fn reshape_(&mut self, x: HirNodeId, new_shape: Vec<i64>) -> HirNodeId;
    fn transpose_(&mut self, x: HirNodeId, perm: Vec<usize>) -> HirNodeId;
    fn narrow_(&mut self, x: HirNodeId, axis: usize, start: usize, len: usize) -> HirNodeId;
    /// Pad each axis by `[low, high]` using `mode` (twin of `Graph::pad_`).
    /// MLA uses it to zero-pad V's `v_head_dim` up to the QK `head_dim` so the
    /// asymmetric-head attention can still ride the symmetric [`Op::Attention`].
    fn pad_(&mut self, x: HirNodeId, pads: Vec<[usize; 2]>, mode: PadMode) -> HirNodeId;
    fn concat_(&mut self, inputs: Vec<HirNodeId>, axis: usize) -> HirNodeId;
    fn gather_(&mut self, table: HirNodeId, indices: HirNodeId, axis: usize) -> HirNodeId;

    fn eq(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;
    fn lt(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;

    fn attention_(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        mask: HirNodeId,
        num_heads: usize,
        head_dim: usize,
    ) -> HirNodeId;

    fn attention_kind(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        shape: Shape,
    ) -> HirNodeId {
        self.attention_kind_opts(q, k, v, num_heads, head_dim, mask_kind, shape, None, None)
    }

    fn attention_kind_opts(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        shape: Shape,
        score_scale: Option<f32>,
        attn_logit_softcap: Option<f32>,
    ) -> HirNodeId;

    fn rope(&mut self, x: HirNodeId, cos: HirNodeId, sin: HirNodeId, head_dim: usize) -> HirNodeId;
    /// RoPE with an explicit pairing flavor ([`crate::op::RopeStyle`]).
    fn rope_styled(
        &mut self,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        head_dim: usize,
        style: crate::op::RopeStyle,
    ) -> HirNodeId;
    fn rope_n(
        &mut self,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        head_dim: usize,
        n_rot: usize,
    ) -> HirNodeId;
    /// Partial RoPE (`n_rot`) with an explicit pairing flavor.
    fn rope_n_styled(
        &mut self,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        head_dim: usize,
        n_rot: usize,
        style: crate::op::RopeStyle,
    ) -> HirNodeId;

    fn cast(&mut self, x: HirNodeId, to: DType) -> HirNodeId;
    fn activation(&mut self, act: Activation, input: HirNodeId, shape: Shape) -> HirNodeId;

    fn gated_delta_net(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state_size: usize,
        shape: Shape,
    ) -> HirNodeId;

    fn gated_delta_net_carry(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state: HirNodeId,
        state_size: usize,
        shape: Shape,
    ) -> HirNodeId;

    /// Per-channel (`g` is `[b, s, h, n]`) gated delta-net — Kimi-K3 KDA.
    fn gated_delta_net_pc(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state_size: usize,
        shape: Shape,
    ) -> HirNodeId;
}

impl HirGraphExt for HirMut<'_> {
    fn shape(&self, id: HirNodeId) -> &Shape {
        &self.0.node(id).shape
    }

    fn add_node(&mut self, op: Op, inputs: Vec<HirNodeId>, shape: Shape) -> HirNodeId {
        self.0.mir(op, inputs, shape)
    }

    fn mm(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s =
            shape::matmul_shape(self.shape(lhs), self.shape(rhs)).expect("matmul shape inference");
        self.0.mir(Op::MatMul, vec![lhs, rhs], s)
    }

    fn add(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("add shape inference");
        self.0.mir(Op::Binary(BinaryOp::Add), vec![lhs, rhs], s)
    }

    fn sub(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("sub shape inference");
        self.0.mir(Op::Binary(BinaryOp::Sub), vec![lhs, rhs], s)
    }

    fn mul(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("mul shape inference");
        self.0.mir(Op::Binary(BinaryOp::Mul), vec![lhs, rhs], s)
    }

    fn div(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("div shape inference");
        self.0.mir(Op::Binary(BinaryOp::Div), vec![lhs, rhs], s)
    }

    fn gelu(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Gelu), vec![x], s)
    }

    fn gelu_approx(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0
            .mir(Op::Activation(Activation::GeluApprox), vec![x], s)
    }

    fn silu(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Silu), vec![x], s)
    }

    fn relu(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Relu), vec![x], s)
    }

    fn exp(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Exp), vec![x], s)
    }

    fn recip(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Recip), vec![x], s)
    }

    fn sqrt(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Sqrt), vec![x], s)
    }

    fn neg(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Neg), vec![x], s)
    }

    fn tanh(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Tanh), vec![x], s)
    }

    fn ln(&mut self, x: HirNodeId, gamma: HirNodeId, beta: HirNodeId, eps: f32) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0
            .mir(Op::LayerNorm { axis: -1, eps }, vec![x, gamma, beta], s)
    }

    fn group_norm(
        &mut self,
        x: HirNodeId,
        gamma: HirNodeId,
        beta: HirNodeId,
        num_groups: usize,
        eps: f32,
    ) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0
            .mir(Op::GroupNorm { num_groups, eps }, vec![x, gamma, beta], s)
    }

    fn layer_norm2d(
        &mut self,
        x: HirNodeId,
        gamma: HirNodeId,
        beta: HirNodeId,
        eps: f32,
    ) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::LayerNorm2d { eps }, vec![x, gamma, beta], s)
    }

    fn resize_nearest_2x(&mut self, x: HirNodeId) -> HirNodeId {
        let in_s = self.shape(x);
        let out = Shape::new(
            &[
                in_s.dim(0).unwrap_static(),
                in_s.dim(1).unwrap_static(),
                in_s.dim(2).unwrap_static() * 2,
                in_s.dim(3).unwrap_static() * 2,
            ],
            in_s.dtype(),
        );
        self.0.mir(Op::ResizeNearest2x, vec![x], out)
    }

    fn conv2d(
        &mut self,
        x: HirNodeId,
        weight: HirNodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        groups: usize,
        out_shape: Shape,
    ) -> HirNodeId {
        self.0.mir(
            Op::Conv {
                kernel_size: kernel_size.to_vec(),
                stride: stride.to_vec(),
                padding: padding.to_vec(),
                dilation: vec![1, 1],
                groups,
            },
            vec![x, weight],
            out_shape,
        )
    }

    fn conv_transpose2d(
        &mut self,
        x: HirNodeId,
        weight: HirNodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_padding: [usize; 2],
        groups: usize,
        out_shape: Shape,
    ) -> HirNodeId {
        self.0.mir(
            Op::ConvTranspose2d {
                kernel_size: kernel_size.to_vec(),
                stride: stride.to_vec(),
                padding: padding.to_vec(),
                dilation: dilation.to_vec(),
                output_padding: output_padding.to_vec(),
                groups,
            },
            vec![x, weight],
            out_shape,
        )
    }

    fn rms_norm(&mut self, x: HirNodeId, gamma: HirNodeId, beta: HirNodeId, eps: f32) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0
            .mir(Op::RmsNorm { axis: -1, eps }, vec![x, gamma, beta], s)
    }

    fn ada_layer_norm(
        &mut self,
        x: HirNodeId,
        scale: HirNodeId,
        shift: HirNodeId,
        norm: AdaNormKind,
        eps: f32,
    ) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0
            .mir(Op::AdaLayerNorm { norm, eps }, vec![x, scale, shift], s)
    }

    fn gated_residual(&mut self, x: HirNodeId, y: HirNodeId, gate: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::GatedResidual, vec![x, y, gate], s)
    }

    fn sum(&mut self, x: HirNodeId, axes: Vec<usize>, keep_dim: bool) -> HirNodeId {
        let s =
            shape::reduce_shape(self.shape(x), &axes, keep_dim).expect("reduce shape inference");
        self.0.mir(
            Op::Reduce {
                op: ReduceOp::Sum,
                axes,
                keep_dim,
            },
            vec![x],
            s,
        )
    }

    fn mean(&mut self, x: HirNodeId, axes: Vec<usize>, keep_dim: bool) -> HirNodeId {
        let s =
            shape::reduce_shape(self.shape(x), &axes, keep_dim).expect("reduce shape inference");
        self.0.mir(
            Op::Reduce {
                op: ReduceOp::Mean,
                axes,
                keep_dim,
            },
            vec![x],
            s,
        )
    }

    fn sm(&mut self, x: HirNodeId, axis: i32) -> HirNodeId {
        let s = shape::softmax_shape(self.shape(x));
        self.0.mir(Op::Softmax { axis }, vec![x], s)
    }

    fn reshape_(&mut self, x: HirNodeId, new_shape: Vec<i64>) -> HirNodeId {
        let s = shape::reshape_shape(self.shape(x), &new_shape).expect("reshape shape inference");
        self.0.mir(Op::Reshape { new_shape }, vec![x], s)
    }

    fn transpose_(&mut self, x: HirNodeId, perm: Vec<usize>) -> HirNodeId {
        let s = shape::transpose_shape(self.shape(x), &perm).expect("transpose shape inference");
        self.0.mir(Op::Transpose { perm }, vec![x], s)
    }

    fn narrow_(&mut self, x: HirNodeId, axis: usize, start: usize, len: usize) -> HirNodeId {
        let s = shape::narrow_shape(self.shape(x), axis, len).expect("narrow shape inference");
        self.0.mir(Op::Narrow { axis, start, len }, vec![x], s)
    }

    fn pad_(&mut self, x: HirNodeId, pads: Vec<[usize; 2]>, mode: PadMode) -> HirNodeId {
        let s = shape::pad_shape(self.shape(x), &pads).expect("pad shape inference");
        self.0.mir(Op::Pad { pads, mode }, vec![x], s)
    }

    fn concat_(&mut self, inputs: Vec<HirNodeId>, axis: usize) -> HirNodeId {
        let shapes: Vec<&Shape> = inputs.iter().map(|&id| self.shape(id)).collect();
        let s = shape::concat_shape(&shapes, axis).expect("concat shape inference");
        self.0.mir(Op::Concat { axis }, inputs, s)
    }

    fn gather_(&mut self, table: HirNodeId, indices: HirNodeId, axis: usize) -> HirNodeId {
        let s = shape::gather_shape(self.shape(table), self.shape(indices), axis)
            .expect("gather shape inference");
        self.0.mir(Op::Gather { axis }, vec![table, indices], s)
    }

    fn eq(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("eq shape inference");
        self.0.mir(Op::Compare(CmpOp::Eq), vec![lhs, rhs], s)
    }

    fn lt(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("lt shape inference");
        self.0.mir(Op::Compare(CmpOp::Lt), vec![lhs, rhs], s)
    }

    fn attention_(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        mask: HirNodeId,
        num_heads: usize,
        head_dim: usize,
    ) -> HirNodeId {
        let s = shape::attention_shape(self.shape(q));
        HirMut::attention(self, q, k, v, mask, num_heads, head_dim, s)
    }

    fn attention_kind(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        shape: Shape,
    ) -> HirNodeId {
        self.attention_kind_opts(q, k, v, num_heads, head_dim, mask_kind, shape, None, None)
    }

    fn attention_kind_opts(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        shape: Shape,
        score_scale: Option<f32>,
        attn_logit_softcap: Option<f32>,
    ) -> HirNodeId {
        self.0.mir(
            crate::ops::attention::attention_kind_op(
                num_heads,
                head_dim,
                mask_kind,
                score_scale,
                attn_logit_softcap,
            ),
            vec![q, k, v],
            shape,
        )
    }

    fn rope(&mut self, x: HirNodeId, cos: HirNodeId, sin: HirNodeId, head_dim: usize) -> HirNodeId {
        self.rope_styled(x, cos, sin, head_dim, crate::op::RopeStyle::NeoX)
    }

    /// RoPE with an explicit pairing flavor ([`crate::op::RopeStyle`]).
    fn rope_styled(
        &mut self,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        head_dim: usize,
        style: crate::op::RopeStyle,
    ) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(
            Op::Rope {
                head_dim,
                n_rot: head_dim,
                style,
            },
            vec![x, cos, sin],
            s,
        )
    }

    fn rope_n(
        &mut self,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        head_dim: usize,
        n_rot: usize,
    ) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        HirModule::rope(self.0, x, cos, sin, head_dim, n_rot, s)
    }

    fn rope_n_styled(
        &mut self,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        head_dim: usize,
        n_rot: usize,
        style: crate::op::RopeStyle,
    ) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(
            Op::Rope {
                head_dim,
                n_rot,
                style,
            },
            vec![x, cos, sin],
            s,
        )
    }

    fn cast(&mut self, x: HirNodeId, to: DType) -> HirNodeId {
        let s = self.shape(x).clone().with_dtype(to);
        self.0.mir(Op::Cast { to }, vec![x], s)
    }

    fn activation(&mut self, act: Activation, input: HirNodeId, shape: Shape) -> HirNodeId {
        self.0.mir(Op::Activation(act), vec![input], shape)
    }

    fn gated_delta_net(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state_size: usize,
        shape: Shape,
    ) -> HirNodeId {
        HirModule::gated_delta_net(self.0, q, k, v, g, beta, state_size, shape)
    }

    fn gated_delta_net_carry(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state: HirNodeId,
        state_size: usize,
        shape: Shape,
    ) -> HirNodeId {
        HirModule::gated_delta_net_carry(self.0, q, k, v, g, beta, state, state_size, shape)
    }

    fn gated_delta_net_pc(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state_size: usize,
        shape: Shape,
    ) -> HirNodeId {
        HirModule::gated_delta_net_pc(self.0, q, k, v, g, beta, state_size, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HirModule;

    #[test]
    fn resize_bilinear2d_shape() {
        let mut hir = HirModule::new("bl".to_string());
        let mut b = HirMut::new(&mut hir);
        let x = b.input("x", Shape::new(&[1, 3, 8, 8], DType::F32));
        let y = b.resize_bilinear2d(x, 16, 20, false);
        let s = b.shape(y);
        assert_eq!(s.dim(0).unwrap_static(), 1);
        assert_eq!(s.dim(1).unwrap_static(), 3);
        assert_eq!(s.dim(2).unwrap_static(), 16);
        assert_eq!(s.dim(3).unwrap_static(), 20);
    }

    fn matrix(l_in: usize, l_out: usize, align_corners: bool, cubic: bool) -> Vec<f32> {
        interp_matrix_plain(l_in, l_out, align_corners, cubic)
    }

    #[test]
    fn interp_matrix_2x_align_false() {
        // linear [in=2, out=4]: out = [in0, .75·in0+.25·in1, .25·in0+.75·in1, in1]
        let m = matrix(2, 4, false, false);
        let want = [1.0, 0.75, 0.25, 0.0, 0.0, 0.25, 0.75, 1.0];
        for (g, w) in m.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{m:?}");
        }
    }

    #[test]
    fn cubic_matrix_partition_of_unity() {
        // Every output column's weights sum to 1 (constant-preserving), incl. edges.
        for &(li, lo, ac) in &[(4usize, 8usize, false), (5, 3, true), (6, 13, false)] {
            let m = matrix(li, lo, ac, true);
            for j in 0..lo {
                let col: f32 = (0..li).map(|i| m[i * lo + j]).sum();
                assert!(
                    (col - 1.0).abs() < 1e-5,
                    "col {j} sum {col} ({li}->{lo} ac={ac})"
                );
            }
        }
    }

    #[test]
    fn cubic_matrix_interpolates_samples() {
        // align_corners=True maps j=0→sample 0 and j=out-1→sample in-1 exactly,
        // so those columns are one-hot (cubic passes through the samples).
        let (li, lo) = (4usize, 7usize);
        let m = matrix(li, lo, true, true);
        assert!((m[0] - 1.0).abs() < 1e-6); // col 0 → sample 0
        assert!((m[(li - 1) * lo + (lo - 1)] - 1.0).abs() < 1e-6); // last col → last sample
    }

    #[test]
    fn aa_matrix_bilinear_downsample_4_to_2() {
        // PyTorch antialias bilinear 4→2 (align_corners=False): widened triangle,
        // renormalized. Columns: [3/7,3/7,1/7,0] and [0,1/7,3/7,3/7].
        let m = interp_matrix_aa(4, 2, false, false);
        let (a, b) = (3.0f32 / 7.0, 1.0f32 / 7.0);
        let want = [a, 0.0, a, b, b, a, 0.0, a]; // row-major [4,2]
        for (g, w) in m.iter().zip(want) {
            assert!((g - w).abs() < 1e-4, "aa {m:?}");
        }
    }

    #[test]
    fn aa_matches_plain_on_upsample() {
        // Antialias is a no-op when upsampling (scale ≤ 1): aa == plain.
        for &cubic in &[false, true] {
            let aa = interp_matrix_aa(3, 8, false, cubic);
            let plain = interp_matrix_plain(3, 8, false, cubic);
            for (g, w) in aa.iter().zip(&plain) {
                assert!(
                    (g - w).abs() < 1e-5,
                    "cubic={cubic}: aa {aa:?} vs plain {plain:?}"
                );
            }
        }
    }

    #[test]
    fn aa_matrix_partition_of_unity() {
        for &(li, lo, cubic) in &[(8usize, 3usize, false), (10, 4, true), (7, 2, false)] {
            let m = interp_matrix_aa(li, lo, false, cubic);
            for j in 0..lo {
                let col: f32 = (0..li).map(|i| m[i * lo + j]).sum();
                assert!((col - 1.0).abs() < 1e-5, "col {j} sum {col}");
            }
        }
    }
}
