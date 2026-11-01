// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integer matrix kernels and the f32 element-wise pieces around them.
//!
//! Weights are int8 (one scale per output channel) or int4 (one scale per
//! group); activations are quantized to int8 per row on the fly. Integer sums
//! are exact in any order, so tiling changes no bits. Everything else is f32
//! through `libm`, so host and MCU builds agree exactly.

use crate::blob::{MAX_GROUPS, Mat, Q4, Q8, Q8G, quantize_row};

/// `Σ a·b` over int8.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(all(target_arch = "arm", feature = "dsp"))]
    {
        dsp::dot_i8(a, b)
    }
    #[cfg(not(all(target_arch = "arm", feature = "dsp")))]
    {
        let mut acc = 0i32;
        for (&x, &y) in a.iter().zip(b) {
            acc += x as i32 * y as i32;
        }
        acc
    }
}

/// Four int8 dot products sharing one activation row.
#[inline(always)]
pub fn dot_1x4(x: &[i8], w: [&[i8]; 4]) -> [i32; 4] {
    #[cfg(all(target_arch = "arm", feature = "dsp"))]
    {
        dsp::dot_1x4(x, w)
    }
    #[cfg(not(all(target_arch = "arm", feature = "dsp")))]
    {
        let n = x.len();
        let (w0, w1, w2, w3) = (&w[0][..n], &w[1][..n], &w[2][..n], &w[3][..n]);
        let mut a = [0i32; 4];
        for i in 0..n {
            let xi = x[i] as i32;
            a[0] += xi * w0[i] as i32;
            a[1] += xi * w1[i] as i32;
            a[2] += xi * w2[i] as i32;
            a[3] += xi * w3[i] as i32;
        }
        a
    }
}

/// A 2-row × 4-output int8 tile: each loaded weight serves two rows, each
/// loaded activation four outputs — about 4 instructions per MAC on rv32
/// against 7 for a plain dot product. (Unrolling further measured slower.)
#[inline(always)]
pub fn dot_2x4(x0: &[i8], x1: &[i8], w: [&[i8]; 4]) -> [[i32; 4]; 2] {
    #[cfg(all(target_arch = "arm", feature = "dsp"))]
    {
        dsp::dot_2x4(x0, x1, w)
    }
    #[cfg(not(all(target_arch = "arm", feature = "dsp")))]
    {
        let n = x0.len();
        let x1 = &x1[..n];
        let (w0, w1, w2, w3) = (&w[0][..n], &w[1][..n], &w[2][..n], &w[3][..n]);
        let mut a = [[0i32; 4]; 2];
        for i in 0..n {
            let (p, q) = (x0[i] as i32, x1[i] as i32);
            let (v0, v1, v2, v3) = (w0[i] as i32, w1[i] as i32, w2[i] as i32, w3[i] as i32);
            a[0][0] += p * v0;
            a[0][1] += p * v1;
            a[0][2] += p * v2;
            a[0][3] += p * v3;
            a[1][0] += q * v0;
            a[1][1] += q * v1;
            a[1][2] += q * v2;
            a[1][3] += q * v3;
        }
        a
    }
}

/// Arm DSP-extension paths (feature `dsp`; Cortex-M4/M7/M33 built with a
/// matching `-C target-cpu`): four int8 values
/// per word, `SXTB16` widens the even and the odd bytes to two i16 pairs, and
/// each `SMLAD` adds two products. Integer sums are exact in any order, so
/// these return the same bits as the portable loops.
#[cfg(all(target_arch = "arm", feature = "dsp"))]
mod dsp {
    /// Even bytes (0, 2) and odd bytes (1, 3) of `w`, each sign-extended to
    /// two i16 halves.
    #[inline(always)]
    fn split(w: u32) -> (u32, u32) {
        let (e, o): (u32, u32);
        // SAFETY: register-only arithmetic.
        unsafe {
            core::arch::asm!(
                "sxtb16 {e}, {w}",
                "sxtb16 {o}, {w}, ror #8",
                w = in(reg) w, e = out(reg) e, o = lateout(reg) o,
                options(pure, nomem, nostack),
            );
        }
        (e, o)
    }

    /// `acc + a.lo·b.lo + a.hi·b.hi` over signed i16 halves.
    #[inline(always)]
    fn smlad(a: u32, b: u32, acc: i32) -> i32 {
        let r: i32;
        // SAFETY: register-only arithmetic.
        unsafe {
            core::arch::asm!(
                "smlad {r}, {a}, {b}, {c}",
                a = in(reg) a, b = in(reg) b, c = in(reg) acc, r = lateout(reg) r,
                options(pure, nomem, nostack),
            );
        }
        r
    }

    /// Four int8 values as one little-endian word.
    #[inline(always)]
    fn word(b: &[i8; 4]) -> u32 {
        u32::from_le_bytes([b[0] as u8, b[1] as u8, b[2] as u8, b[3] as u8])
    }

    pub(super) fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let mut acc = 0i32;
        for (x, y) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
            let (ae, ao) = split(word(x));
            let (be, bo) = split(word(y));
            acc = smlad(ae, be, smlad(ao, bo, acc));
        }
        for i in n / 4 * 4..n {
            acc += a[i] as i32 * b[i] as i32;
        }
        acc
    }

    /// One row against four: 4 accumulators, the row's two halves and one
    /// weight word's two halves fit the Cortex-M register file, where a 2×4
    /// tile spills to the stack.
    #[inline(always)]
    pub(super) fn dot_1x4(x: &[i8], w: [&[i8]; 4]) -> [i32; 4] {
        let n = x.len();
        let (w0, w1, w2, w3) = (&w[0][..n], &w[1][..n], &w[2][..n], &w[3][..n]);
        let (mut a0, mut a1, mut a2, mut a3) = (0i32, 0i32, 0i32, 0i32);
        let it = x
            .as_chunks::<4>()
            .0
            .iter()
            .zip(w0.as_chunks::<4>().0)
            .zip(w1.as_chunks::<4>().0)
            .zip(w2.as_chunks::<4>().0)
            .zip(w3.as_chunks::<4>().0);
        for ((((xc, c0), c1), c2), c3) in it {
            let (xe, xo) = split(word(xc));
            let (e, o) = split(word(c0));
            a0 = smlad(xe, e, smlad(xo, o, a0));
            let (e, o) = split(word(c1));
            a1 = smlad(xe, e, smlad(xo, o, a1));
            let (e, o) = split(word(c2));
            a2 = smlad(xe, e, smlad(xo, o, a2));
            let (e, o) = split(word(c3));
            a3 = smlad(xe, e, smlad(xo, o, a3));
        }
        let mut a = [a0, a1, a2, a3];
        for i in n / 4 * 4..n {
            let xi = x[i] as i32;
            a[0] += xi * w0[i] as i32;
            a[1] += xi * w1[i] as i32;
            a[2] += xi * w2[i] as i32;
            a[3] += xi * w3[i] as i32;
        }
        a
    }

    /// Longest row [`linear_q8`] widens on the stack (two rows of i16:
    /// 4 KB).
    pub(super) const X16_MAX: usize = 1024;

    /// Widen two quantized rows for [`dot_2x2`]: per 4 values, row 0's even
    /// pair `(x0, x2)` and odd pair `(x1, x3)` as i16 halves, then row 1's.
    /// That is the lane order `SXTB16` gives a weight word.
    fn widen(q0: &[i8], q1: &[i8], buf: &mut [u32]) {
        let pack = |lo: i8, hi: i8| (lo as i16 as u16 as u32) | ((hi as i16 as u16 as u32) << 16);
        let groups = buf.as_chunks_mut::<4>().0.iter_mut();
        for ((g, a), b) in groups.zip(q0.as_chunks::<4>().0).zip(q1.as_chunks::<4>().0) {
            g[0] = pack(a[0], a[2]);
            g[1] = pack(a[1], a[3]);
            g[2] = pack(b[0], b[2]);
            g[3] = pack(b[1], b[3]);
        }
    }

    /// Two widened rows against weight rows `w` and `w + k`:
    /// `[[x0·w0, x0·w1], [x1·w0, x1·w1]]`. `k` a multiple of 4; `x` from
    /// [`widen`]. 19 instructions per 16 MACs and exactly the 12 registers
    /// inline assembly gets on Thumb (r0–r5, r8–r12, lr).
    #[inline(always)]
    fn dot_2x2(x: &[u32], w: &[i8], k: usize) -> [[i32; 2]; 2] {
        assert!(k.is_multiple_of(4) && k > 0 && x.len() >= k && w.len() >= 2 * k);
        let (mut a00, mut a01, mut a10, mut a11) = (0i32, 0i32, 0i32, 0i32);
        // SAFETY: reads `k` bytes from each of the two weight rows (bounds
        // asserted; unaligned LDR is legal on v7-M/v8-M mainline) and `k`
        // words of `x` (word-aligned `u32` storage, so LDRD is legal).
        unsafe {
            core::arch::asm!(
                "2:",
                "ldr    {t}, [{pw}, {k}]",
                "sxtb16 {u}, {t}, ror #8",
                "sxtb16 {t}, {t}",
                "ldrd   {xa}, {xb}, [{px}]",
                "smlad  {a01}, {xa}, {t}, {a01}",
                "smlad  {a01}, {xb}, {u}, {a01}",
                "ldrd   {xa}, {xb}, [{px}, #8]",
                "smlad  {a11}, {xa}, {t}, {a11}",
                "smlad  {a11}, {xb}, {u}, {a11}",
                "ldr    {t}, [{pw}], #4",
                "sxtb16 {u}, {t}, ror #8",
                "sxtb16 {t}, {t}",
                "smlad  {a10}, {xa}, {t}, {a10}",
                "smlad  {a10}, {xb}, {u}, {a10}",
                "ldrd   {xa}, {xb}, [{px}], #16",
                "smlad  {a00}, {xa}, {t}, {a00}",
                "smlad  {a00}, {xb}, {u}, {a00}",
                "subs   {n}, {n}, #1",
                "bne    2b",
                pw = inout(reg) w.as_ptr() => _,
                px = inout(reg) x.as_ptr() => _,
                k = in(reg) k,
                n = inout(reg) k / 4 => _,
                t = out(reg) _,
                u = out(reg) _,
                xa = out(reg) _,
                xb = out(reg) _,
                a00 = inout(reg) a00,
                a01 = inout(reg) a01,
                a10 = inout(reg) a10,
                a11 = inout(reg) a11,
                options(nostack, readonly),
            );
        }
        [[a00, a01], [a10, a11]]
    }

    /// [`super::linear_q8`] for `k % 4 == 0`, `k <= X16_MAX`: row pairs in
    /// 2×2 tiles of [`dot_2x2`]. The same per-row quantization, the same
    /// exact integer sums and the same `out`, so the same bits.
    pub(super) fn linear_q8(
        x: &[f32],
        rows: usize,
        w: &super::Q8<'_>,
        y: &mut [f32],
        xq: &mut [i8],
        out: impl Fn(i32, f32, usize) -> f32,
    ) {
        let (k, n) = (w.cols, w.rows);
        let mut buf = [0u32; X16_MAX];
        let mut r = 0;
        while r < rows {
            let pair = r + 1 < rows;
            let (q0, q1) = xq[..2 * k].split_at_mut(k);
            let s0 = super::quantize_row(&x[r * k..(r + 1) * k], q0);
            let s1 = if pair {
                super::quantize_row(&x[(r + 1) * k..(r + 2) * k], q1)
            } else {
                q1.fill(0);
                0.0
            };
            widen(q0, q1, &mut buf[..k]);
            let mut o = 0;
            while o + 2 <= n {
                let a = dot_2x2(&buf[..k], &w.data[o * k..(o + 2) * k], k);
                y[r * n + o] = out(a[0][0], s0, o);
                y[r * n + o + 1] = out(a[0][1], s0, o + 1);
                if pair {
                    y[(r + 1) * n + o] = out(a[1][0], s1, o);
                    y[(r + 1) * n + o + 1] = out(a[1][1], s1, o + 1);
                }
                o += 2;
            }
            if o < n {
                y[r * n + o] = out(dot_i8(q0, w.row(o)), s0, o);
                if pair {
                    y[(r + 1) * n + o] = out(dot_i8(q1, w.row(o)), s1, o);
                }
            }
            r += if pair { 2 } else { 1 };
        }
    }

    /// Two [`dot_1x4`]s: the weights are read twice, but nothing spills.
    #[inline(always)]
    pub(super) fn dot_2x4(x0: &[i8], x1: &[i8], w: [&[i8]; 4]) -> [[i32; 4]; 2] {
        [dot_1x4(x0, w), dot_1x4(x1, w)]
    }
}

/// One int8 row against one int4 row: `Σ_g scale_g · Σ_{i∈g} x_i q_i`.
#[inline]
pub fn dot_q4(x: &[i8], codes: &[u8], scales: &[f32], group: usize) -> f32 {
    let mut total = 0f32;
    for (g, &s) in scales.iter().enumerate() {
        let base = g * group;
        let mut acc = 0i32;
        let bytes = &codes[base / 2..(base + group) / 2];
        let xs = &x[base..base + group];
        for (j, &b) in bytes.iter().enumerate() {
            let lo = ((b << 4) as i8) >> 4;
            let hi = (b as i8) >> 4;
            acc += xs[2 * j] as i32 * lo as i32 + xs[2 * j + 1] as i32 * hi as i32;
        }
        total += acc as f32 * s;
    }
    total
}

/// `y[r] = x[r] · Wᵀ + b` for `rows` rows of an int8 matrix. `xq` holds two
/// quantized rows (`2 * w.cols`).
pub fn linear_q8(
    x: &[f32],
    rows: usize,
    w: &Q8<'_>,
    b: Option<&[f32]>,
    y: &mut [f32],
    xq: &mut [i8],
) {
    let (k, n) = (w.cols, w.rows);
    let out =
        |acc: i32, sx: f32, o: usize| acc as f32 * (sx * w.scales[o]) + b.map_or(0.0, |b| b[o]);
    #[cfg(all(target_arch = "arm", feature = "dsp"))]
    if k.is_multiple_of(4) && k <= dsp::X16_MAX {
        return dsp::linear_q8(x, rows, w, y, xq, out);
    }
    let mut r = 0;
    while r < rows {
        let pair = r + 1 < rows;
        let (q0, q1) = xq[..2 * k].split_at_mut(k);
        let s0 = quantize_row(&x[r * k..(r + 1) * k], q0);
        let s1 = if pair {
            quantize_row(&x[(r + 1) * k..(r + 2) * k], q1)
        } else {
            0.0
        };
        let (q0, q1) = (&*q0, &*q1);
        let mut o = 0;
        while o + 4 <= n {
            let ws = [w.row(o), w.row(o + 1), w.row(o + 2), w.row(o + 3)];
            if pair {
                let a = dot_2x4(q0, q1, ws);
                for j in 0..4 {
                    y[r * n + o + j] = out(a[0][j], s0, o + j);
                    y[(r + 1) * n + o + j] = out(a[1][j], s1, o + j);
                }
            } else {
                let a = dot_1x4(q0, ws);
                for j in 0..4 {
                    y[r * n + o + j] = out(a[j], s0, o + j);
                }
            }
            o += 4;
        }
        while o < n {
            y[r * n + o] = out(dot_i8(q0, w.row(o)), s0, o);
            if pair {
                y[(r + 1) * n + o] = out(dot_i8(q1, w.row(o)), s1, o);
            }
            o += 1;
        }
        r += if pair { 2 } else { 1 };
    }
}

/// [`linear_q8`] for an int4 matrix.
pub fn linear_q4(
    x: &[f32],
    rows: usize,
    w: &Q4<'_>,
    b: Option<&[f32]>,
    y: &mut [f32],
    xq: &mut [i8],
) {
    let (k, n) = (w.cols, w.rows);
    for r in 0..rows {
        let sx = quantize_row(&x[r * k..(r + 1) * k], &mut xq[..k]);
        let xq = &xq[..k];
        for o in 0..n {
            let (codes, scales) = w.row(o);
            y[r * n + o] = dot_q4(xq, codes, scales, w.group) * sx + b.map_or(0.0, |b| b[o]);
        }
    }
}

/// [`linear_q8`] for a group-scaled int8 matrix. The activations are
/// quantized per group too, so both sides of every group product carry their
/// own scale: `Σ_g sx_g · sw_g · Σ_{i∈g} x_i w_i`.
pub fn linear_q8g(
    x: &[f32],
    rows: usize,
    w: &Q8G<'_>,
    b: Option<&[f32]>,
    y: &mut [f32],
    xq: &mut [i8],
) {
    let (k, n, g) = (w.cols, w.rows, w.group);
    let ng = k / g;
    let mut sx = [0f32; MAX_GROUPS];
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        for gi in 0..ng {
            sx[gi] = quantize_row(&xr[gi * g..(gi + 1) * g], &mut xq[gi * g..(gi + 1) * g]);
        }
        let xq = &xq[..k];
        for o in 0..n {
            let (wr, ws) = w.row(o);
            let mut acc = 0f32;
            for gi in 0..ng {
                let d = dot_i8(&xq[gi * g..(gi + 1) * g], &wr[gi * g..(gi + 1) * g]);
                acc += d as f32 * (sx[gi] * ws[gi]);
            }
            y[r * n + o] = acc + b.map_or(0.0, |b| b[o]);
        }
    }
}

/// `y[r] = x[r] · Wᵀ + b` for any matrix width.
pub fn linear(
    x: &[f32],
    rows: usize,
    w: &Mat<'_>,
    b: Option<&[f32]>,
    y: &mut [f32],
    xq: &mut [i8],
) {
    match w {
        Mat::Q8(m) => linear_q8(x, rows, m, b, y, xq),
        Mat::Q4(m) => linear_q4(x, rows, m, b, y, xq),
        Mat::Q8G(m) => linear_q8g(x, rows, m, b, y, xq),
    }
}

/// `argmax_o (x · W[o])` without storing the logits — for an LM head whose
/// output row is much wider than the scratch worth spending on it. Ties go
/// to the lower index. `xq` holds one quantized row (`w.cols()`).
pub fn argmax_linear(x: &[f32], w: &Mat<'_>, xq: &mut [i8]) -> usize {
    let k = w.cols();
    let sx = quantize_row(&x[..k], &mut xq[..k]);
    let xq = &xq[..k];
    let (mut best, mut best_v) = (0usize, f32::NEG_INFINITY);
    for o in 0..w.rows() {
        let v = match w {
            Mat::Q8(m) => dot_i8(xq, m.row(o)) as f32 * (sx * m.scales[o]),
            Mat::Q4(m) => {
                let (codes, scales) = m.row(o);
                dot_q4(xq, codes, scales, m.group) * sx
            }
            Mat::Q8G(m) => {
                let (wr, ws) = m.row(o);
                let mut f = 0f32;
                for (gi, &s) in ws.iter().enumerate() {
                    let g = gi * m.group..(gi + 1) * m.group;
                    f += dot_i8(&xq[g.clone()], &wr[g]) as f32 * s;
                }
                f * sx
            }
        };
        if v > best_v {
            best = o;
            best_v = v;
        }
    }
    best
}

/// Row `r` of a matrix, dequantized into `out` (embedding lookups).
pub fn dequant_row(w: &Mat<'_>, r: usize, out: &mut [f32]) {
    match w {
        Mat::Q8(m) => {
            let s = m.scales[r];
            for (o, &q) in out.iter_mut().zip(m.row(r)) {
                *o = q as f32 * s;
            }
        }
        Mat::Q4(m) => {
            let (codes, scales) = m.row(r);
            for (c, o) in out.iter_mut().enumerate().take(m.cols) {
                *o = Q4::code(codes, c) as f32 * scales[c / m.group];
            }
        }
        Mat::Q8G(m) => {
            let (vals, scales) = m.row(r);
            for (c, o) in out.iter_mut().enumerate().take(m.cols) {
                *o = vals[c] as f32 * scales[c / m.group];
            }
        }
    }
}

/// LayerNorm over rows of width `d`. `b = None` means no bias.
pub fn layer_norm(
    x: &[f32],
    rows: usize,
    d: usize,
    g: &[f32],
    b: Option<&[f32]>,
    eps: f32,
    y: &mut [f32],
) {
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let mut mean = 0.0f32;
        for &v in xr {
            mean += v;
        }
        mean /= d as f32;
        let mut var = 0.0f32;
        for &v in xr {
            let c = v - mean;
            var += c * c;
        }
        var /= d as f32;
        let inv = 1.0 / libm::sqrtf(var + eps);
        let yr = &mut y[r * d..(r + 1) * d];
        for i in 0..d {
            yr[i] = (xr[i] - mean) * inv * g[i] + b.map_or(0.0, |b| b[i]);
        }
    }
}

/// `x · σ(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + libm::expf(-x))
}

/// `1 / (1 + e^-x)`.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

/// Exact (erf) GELU, PyTorch's default.
#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * core::f32::consts::FRAC_1_SQRT_2))
}

/// In-place softmax. Entries equal to `f32::NEG_INFINITY` get probability 0.
pub fn softmax(x: &mut [f32]) {
    let mut mx = f32::NEG_INFINITY;
    for &v in x.iter() {
        mx = mx.max(v);
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = libm::expf(*v - mx);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Split the first `n` elements off a scratch slice.
pub fn carve<'s, T>(buf: &mut &'s mut [T], n: usize) -> &'s mut [T] {
    let (a, b) = core::mem::take(buf).split_at_mut(n);
    *buf = b;
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_dot_matches_dequantized() {
        let x: [i8; 8] = [1, -2, 3, 4, -5, 6, 7, -8];
        // codes 1..=8 wrapped into -8..=7 nibbles
        let q: [i8; 8] = [1, 2, -3, 4, 5, -6, 7, -8];
        let mut codes = [0u8; 4];
        for (i, &v) in q.iter().enumerate() {
            codes[i / 2] |= ((v as u8) & 0xf) << (4 * (i % 2));
        }
        let scales = [0.5f32, 2.0];
        let want: f32 = (0..8)
            .map(|i| x[i] as f32 * q[i] as f32 * scales[i / 4])
            .sum();
        assert_eq!(dot_q4(&x, &codes, &scales, 4), want);
    }

    #[test]
    fn softmax_sums_to_one_and_masks() {
        let mut v = [1.0, 2.0, f32::NEG_INFINITY, 0.5];
        softmax(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(v[2], 0.0);
    }
}
