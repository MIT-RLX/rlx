// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NNPACK's 16×16 Fourier output convolution (`nnp_convolution_output`, `ft16x16`,
//! `Maratyszcza/NNPACK@c07e3a0`) in Rust, operation for operation as torch 2.10's
//! macOS arm64 build computes it:
//!
//! * the psimd transforms (`psimd/fft/{aos,real,soa,dualreal}.h`,
//!   `2d-fourier-16x16.c`) lane by lane, with an FMA exactly where clang's default
//!   `-ffp-contract=on` fuses — a `x ± a*b` written in one expression, the LEFT product
//!   first (`xr*cr - xi*ci` → `fma(xr, cr, -(xi*ci))`);
//! * the NEON tuple GEMMs (`c4gemm-conjb`, `s4c2gemm-conjb`), whose `vfmaq`/`vfmsq`
//!   are explicit FMAs, accumulated over input channels in NNPACK's cache blocks
//!   (256 channels; blocks joined by a plain add);
//! * flush-to-zero: every NNPACK stage runs under `PTHREADPOOL_FLAG_DISABLE_DENORMALS`
//!   (FPCR.FZ), so each arithmetic result and operand is flushed here in software;
//! * signed zeros: a block column NNPACK transforms as zeros can produce `-0.0`
//!   where an untransformed one stays `+0.0`, so which columns are transformed is
//!   reproduced too.
//!
//! The memory blocking and the fast/partial GEMM kernel split do not change any
//! output element's arithmetic, so they are not reproduced.

type V = [f32; 4];

const SQRT2_OVER_2: f32 = f32::from_bits(0x3F35_04F3); // 0x1.6A09E6p-1
const SQRT2_OVER_4: f32 = f32::from_bits(0x3EB5_04F3); // 0x1.6A09E6p-2
const COS_1PI_OVER_8: f32 = f32::from_bits(0x3F6C_835E); // 0x1.D906BCp-1
const COS_3PI_OVER_8: f32 = f32::from_bits(0x3EC3_EF15); // 0x1.87DE2Ap-2
const COS_2PI_OVER_8: f32 = SQRT2_OVER_2;
const COS_1PI_OVER_4: f32 = SQRT2_OVER_2;
const SIN_1PI_OVER_8: f32 = COS_3PI_OVER_8;
const SIN_3PI_OVER_8: f32 = COS_1PI_OVER_8;
const SIN_2PI_OVER_8: f32 = SQRT2_OVER_2;
const SIN_1PI_OVER_4: f32 = SQRT2_OVER_2;

/// Input channels per tuple-GEMM block (NNPACK's L1 blocking on Apple arm64:
/// `round_down(32 KiB / 32 B / (mr + nr), 2)`).
const INPUT_CHANNELS_BLOCK: usize = 256;

// ── scalar arithmetic under FPCR.FZ ────────────────────────────────────────────

#[inline(always)]
fn z(x: f32) -> f32 {
    if x.is_subnormal() {
        0.0f32.copysign(x)
    } else {
        x
    }
}
#[inline(always)]
fn add(a: f32, b: f32) -> f32 {
    z(z(a) + z(b))
}
#[inline(always)]
fn sub(a: f32, b: f32) -> f32 {
    z(z(a) - z(b))
}
#[inline(always)]
fn mul(a: f32, b: f32) -> f32 {
    z(z(a) * z(b))
}
/// `a·b + c`, one rounding.
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    z(z(a).mul_add(z(b), z(c)))
}

// ── lane-wise vectors ──────────────────────────────────────────────────────────

#[inline(always)]
fn vmap2(a: V, b: V, f: impl Fn(f32, f32) -> f32) -> V {
    [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
}
#[inline(always)]
fn vadd(a: V, b: V) -> V {
    vmap2(a, b, add)
}
#[inline(always)]
fn vsub(a: V, b: V) -> V {
    vmap2(a, b, sub)
}
#[inline(always)]
fn vmul(a: V, b: V) -> V {
    vmap2(a, b, mul)
}
#[inline(always)]
fn vfma(a: V, b: V, c: V) -> V {
    [
        fma(a[0], b[0], c[0]),
        fma(a[1], b[1], c[1]),
        fma(a[2], b[2], c[2]),
        fma(a[3], b[3], c[3]),
    ]
}
#[inline(always)]
fn vneg(a: V) -> V {
    [-a[0], -a[1], -a[2], -a[3]]
}
#[inline(always)]
fn splat(x: f32) -> V {
    [x; 4]
}
/// `psimd_butterfly_f32`: `(a + b, a - b)`.
#[inline(always)]
fn bf(a: &mut V, b: &mut V) {
    let (na, nb) = (vadd(*a, *b), vsub(*a, *b));
    *a = na;
    *b = nb;
}
#[inline(always)]
fn concat_lo(a: V, b: V) -> V {
    [a[0], a[1], b[0], b[1]]
}
#[inline(always)]
fn concat_hi(a: V, b: V) -> V {
    [a[2], a[3], b[2], b[3]]
}
#[inline(always)]
fn concat_even(a: V, b: V) -> V {
    [a[0], a[2], b[0], b[2]]
}
#[inline(always)]
fn concat_odd(a: V, b: V) -> V {
    [a[1], a[3], b[1], b[3]]
}
#[inline(always)]
fn interleave_lo(a: V, b: V) -> V {
    [a[0], b[0], a[1], b[1]]
}
#[inline(always)]
fn interleave_hi(a: V, b: V) -> V {
    [a[2], b[2], a[3], b[3]]
}
/// `__builtin_shufflevector(a, b, 4, 3, 2, 1)`.
#[inline(always)]
fn shuf_4321(a: V, b: V) -> V {
    [b[0], a[3], a[2], a[1]]
}
/// `__builtin_shufflevector(a, b, 4, 1, 2, 3)`.
#[inline(always)]
fn shuf_4123(a: V, b: V) -> V {
    [b[0], a[1], a[2], a[3]]
}

/// `psimd_cmulc_soa_f32`: `x · conj(c)` — `xr*cr + xi*ci`, `xi*cr - xr*ci`.
#[inline(always)]
fn cmulc(xr: &mut V, xi: &mut V, cr: V, ci: V) {
    let nr = vfma(*xr, cr, vmul(*xi, ci));
    let ni = vfma(*xi, cr, vneg(vmul(*xr, ci)));
    *xr = nr;
    *xi = ni;
}
/// `psimd_cmul_soa_f32`: `x · c` — `xr*cr - xi*ci`, `xi*cr + xr*ci`.
#[inline(always)]
fn cmul(xr: &mut V, xi: &mut V, cr: V, ci: V) {
    let nr = vfma(*xr, cr, vneg(vmul(*xi, ci)));
    let ni = vfma(*xi, cr, vmul(*xr, ci));
    *xr = nr;
    *xi = ni;
}

const FFT8_COS: V = [1.0, COS_1PI_OVER_4, 0.0, -SQRT2_OVER_2];
const FFT8_SIN: V = [0.0, SIN_1PI_OVER_4, 1.0, SQRT2_OVER_2];
const FFT4_COS: V = [1.0, 0.0, 1.0, 0.0];
const FFT4_SIN: V = [0.0, 1.0, 0.0, 1.0];
const FFT16_COS_LO: V = [1.0, COS_1PI_OVER_8, COS_2PI_OVER_8, COS_3PI_OVER_8];
const FFT16_SIN_LO: V = [0.0, SIN_1PI_OVER_8, SIN_2PI_OVER_8, SIN_3PI_OVER_8];
const FFT16_COS_HI: V = [0.0, -COS_3PI_OVER_8, -SQRT2_OVER_2, -COS_1PI_OVER_8];
const FFT16_SIN_HI: V = [1.0, COS_1PI_OVER_8, SQRT2_OVER_2, COS_3PI_OVER_8];

fn scaled(v: V, s: f32) -> V {
    // constant-folded in C (`COS_xPI_OVER_8 * 0.0625f`): exact, powers of two
    [v[0] * s, v[1] * s, v[2] * s, v[3] * s]
}

/// `psimd_fft16_soa_f32`: a 16-point complex FFT over the four lanes of four
/// vectors (`r` = real, `i` = imaginary), outputs in NNPACK's bit-reversed order.
fn fft16_soa(r: &mut [V; 4], i: &mut [V; 4]) {
    let (mut w0123r, mut w4567r, mut w89abr, mut wcdefr) = (r[0], r[1], r[2], r[3]);
    let (mut w0123i, mut w4567i, mut w89abi, mut wcdefi) = (i[0], i[1], i[2], i[3]);
    bf(&mut w0123r, &mut w89abr);
    bf(&mut w4567r, &mut wcdefr);
    bf(&mut w0123i, &mut w89abi);
    bf(&mut w4567i, &mut wcdefi);
    cmulc(&mut w89abr, &mut w89abi, FFT16_COS_LO, FFT16_SIN_LO);
    cmulc(&mut wcdefr, &mut wcdefi, FFT16_COS_HI, FFT16_SIN_HI);
    bf(&mut w0123r, &mut w4567r);
    bf(&mut w89abr, &mut wcdefr);
    bf(&mut w0123i, &mut w4567i);
    bf(&mut w89abi, &mut wcdefi);
    cmulc(&mut w4567r, &mut w4567i, FFT8_COS, FFT8_SIN);
    cmulc(&mut wcdefr, &mut wcdefi, FFT8_COS, FFT8_SIN);
    let mut w0189r = concat_lo(w0123r, w89abr);
    let mut w23abr = concat_hi(w0123r, w89abr);
    bf(&mut w0189r, &mut w23abr);
    let mut w0189i = concat_lo(w0123i, w89abi);
    let mut w23abi = concat_hi(w0123i, w89abi);
    bf(&mut w0189i, &mut w23abi);
    let mut w45cdr = concat_lo(w4567r, wcdefr);
    let mut w67efr = concat_hi(w4567r, wcdefr);
    bf(&mut w45cdr, &mut w67efr);
    let mut w45cdi = concat_lo(w4567i, wcdefi);
    let mut w67efi = concat_hi(w4567i, wcdefi);
    bf(&mut w45cdi, &mut w67efi);
    cmulc(&mut w23abr, &mut w23abi, FFT4_COS, FFT4_SIN);
    cmulc(&mut w67efr, &mut w67efi, FFT4_COS, FFT4_SIN);
    let mut w084cr = concat_even(w0189r, w45cdr);
    let mut w195dr = concat_odd(w0189r, w45cdr);
    bf(&mut w084cr, &mut w195dr);
    let mut w084ci = concat_even(w0189i, w45cdi);
    let mut w195di = concat_odd(w0189i, w45cdi);
    bf(&mut w084ci, &mut w195di);
    let mut w2a6er = concat_even(w23abr, w67efr);
    let mut w3b7fr = concat_odd(w23abr, w67efr);
    bf(&mut w2a6er, &mut w3b7fr);
    let mut w2a6ei = concat_even(w23abi, w67efi);
    let mut w3b7fi = concat_odd(w23abi, w67efi);
    bf(&mut w2a6ei, &mut w3b7fi);
    *r = [w084cr, w2a6er, w195dr, w3b7fr];
    *i = [w084ci, w2a6ei, w195di, w3b7fi];
}

/// `psimd_ifft16_soa_f32` (inverse of [`fft16_soa`], scaled by 1/16).
fn ifft16_soa(r: &mut [V; 4], i: &mut [V; 4]) {
    let (mut w084cr, mut w2a6er, mut w195dr, mut w3b7fr) = (r[0], r[1], r[2], r[3]);
    let (mut w084ci, mut w2a6ei, mut w195di, mut w3b7fi) = (i[0], i[1], i[2], i[3]);
    bf(&mut w084cr, &mut w195dr);
    let mut w0189r = interleave_lo(w084cr, w195dr);
    let mut w45cdr = interleave_hi(w084cr, w195dr);
    bf(&mut w084ci, &mut w195di);
    let mut w0189i = interleave_lo(w084ci, w195di);
    let mut w45cdi = interleave_hi(w084ci, w195di);
    bf(&mut w2a6er, &mut w3b7fr);
    let mut w23abr = interleave_lo(w2a6er, w3b7fr);
    let mut w67efr = interleave_hi(w2a6er, w3b7fr);
    bf(&mut w2a6ei, &mut w3b7fi);
    let mut w23abi = interleave_lo(w2a6ei, w3b7fi);
    let mut w67efi = interleave_hi(w2a6ei, w3b7fi);
    cmul(&mut w23abr, &mut w23abi, FFT4_COS, FFT4_SIN);
    cmul(&mut w67efr, &mut w67efi, FFT4_COS, FFT4_SIN);
    bf(&mut w0189r, &mut w23abr);
    let mut w0123r = concat_lo(w0189r, w23abr);
    let mut w89abr = concat_hi(w0189r, w23abr);
    bf(&mut w0189i, &mut w23abi);
    let mut w0123i = concat_lo(w0189i, w23abi);
    let mut w89abi = concat_hi(w0189i, w23abi);
    bf(&mut w45cdr, &mut w67efr);
    let mut w4567r = concat_lo(w45cdr, w67efr);
    let mut wcdefr = concat_hi(w45cdr, w67efr);
    bf(&mut w45cdi, &mut w67efi);
    let mut w4567i = concat_lo(w45cdi, w67efi);
    let mut wcdefi = concat_hi(w45cdi, w67efi);
    cmul(&mut w4567r, &mut w4567i, FFT8_COS, FFT8_SIN);
    cmul(&mut wcdefr, &mut wcdefi, FFT8_COS, FFT8_SIN);
    bf(&mut w0123r, &mut w4567r);
    bf(&mut w89abr, &mut wcdefr);
    bf(&mut w0123i, &mut w4567i);
    bf(&mut w89abi, &mut wcdefi);
    cmul(
        &mut w89abr,
        &mut w89abi,
        scaled(FFT16_COS_LO, 0.0625),
        scaled(FFT16_SIN_LO, 0.0625),
    );
    cmul(
        &mut wcdefr,
        &mut wcdefi,
        scaled(FFT16_COS_HI, 0.0625),
        scaled(FFT16_SIN_HI, 0.0625),
    );
    let s = splat(0.0625);
    w0123r = vmul(w0123r, s);
    w0123i = vmul(w0123i, s);
    w4567r = vmul(w4567r, s);
    w4567i = vmul(w4567i, s);
    bf(&mut w0123r, &mut w89abr);
    bf(&mut w4567r, &mut wcdefr);
    bf(&mut w0123i, &mut w89abi);
    bf(&mut w4567i, &mut wcdefi);
    *r = [w0123r, w4567r, w89abr, wcdefr];
    *i = [w0123i, w4567i, w89abi, wcdefi];
}

/// `psimd_fft16_dualreal_f32`: the FFTs of two real 16-sequences `s` and `h`, packed.
fn fft16_dualreal(s: &mut [V; 4], h: &mut [V; 4]) {
    fft16_soa(s, h);
    let (x0123, x4567, x89ab, xcdef) = (s[0], s[1], s[2], s[3]);
    let (y0123, y4567, y89ab, ycdef) = (h[0], h[1], h[2], h[3]);
    let x0fed = shuf_4321(xcdef, x0123);
    let y0fed = shuf_4321(ycdef, y0123);
    let xcba9 = shuf_4321(x89ab, xcdef);
    let ycba9 = shuf_4321(y89ab, ycdef);
    let half = splat(0.5);
    let r0246 = vmul(half, vadd(x0123, x0fed));
    let r1357 = vmul(half, vadd(y0123, y0fed));
    let r8ace = vmul(half, vadd(x4567, xcba9));
    let r9bdf = vmul(half, vadd(y4567, ycba9));
    let i_246 = vmul(half, vsub(y0123, y0fed));
    let i_357 = vmul(half, vsub(x0fed, x0123));
    let i8ace = vmul(half, vsub(y4567, ycba9));
    let i9bdf = vmul(half, vsub(xcba9, x4567));
    let i0246 = shuf_4123(i_246, x89ab);
    let i1357 = shuf_4123(i_357, y89ab);
    *s = [
        interleave_lo(r0246, r1357),
        interleave_hi(r0246, r1357),
        interleave_lo(r8ace, r9bdf),
        interleave_hi(r8ace, r9bdf),
    ];
    *h = [
        interleave_lo(i0246, i1357),
        interleave_hi(i0246, i1357),
        interleave_lo(i8ace, i9bdf),
        interleave_hi(i8ace, i9bdf),
    ];
}

/// `psimd_ifft16_dualreal_f32`.
fn ifft16_dualreal(s: &mut [V; 4], h: &mut [V; 4]) {
    let (r0123, r4567, r89ab, rcdef) = (s[0], s[1], s[2], s[3]);
    let (i0123, i4567, i89ab, icdef) = (h[0], h[1], h[2], h[3]);
    let r0246 = concat_even(r0123, r4567);
    let r1357 = concat_odd(r0123, r4567);
    let r8ace = concat_even(r89ab, rcdef);
    let r9bdf = concat_odd(r89ab, rcdef);
    let i0246 = concat_even(i0123, i4567);
    let i1357 = concat_odd(i0123, i4567);
    let i8ace = concat_even(i89ab, icdef);
    let i9bdf = concat_odd(i89ab, icdef);
    let scba9 = vadd(r8ace, i9bdf);
    let s_fed = vadd(r0246, i1357);
    let hcba9 = vsub(r9bdf, i8ace);
    let h_fed = vsub(r1357, i0246);
    s[0] = shuf_4123(vsub(r0246, i1357), r0246);
    s[1] = vsub(r8ace, i9bdf);
    s[2] = shuf_4321(scba9, i0246);
    s[3] = shuf_4321(s_fed, scba9);
    h[0] = shuf_4123(vadd(r1357, i0246), r1357);
    h[1] = vadd(r9bdf, i8ace);
    h[2] = shuf_4321(hcba9, i1357);
    h[3] = shuf_4321(h_fed, hcba9);
    ifft16_soa(s, h);
}

/// `psimd_fft16_real_f32` for one column (one lane): `x[r]` is row `r` when the
/// row is loaded (`row_offset <= r < row_offset + row_count`), else absent. The
/// output rows are `f0r f8r f1r f1i … f7r f7i`.
fn fft16_real(x: &[Option<f32>; 16]) -> [f32; 16] {
    // psimd_fft8_aos_f32: rows k and k+8 into (lo, hi) = (w_k, w_{k+4}) pairs
    let mut w = [0.0f32; 16]; // w0r w0i w1r w1i w2r w2i w3r w3i w4r … w7i
    for k in 0..8 {
        let (lo, hi) = (k, k + 8);
        if let Some(v) = x[lo] {
            w[k] = v;
            w[k + 8] = v;
        }
        if let Some(v) = x[hi] {
            w[k + 8] = v;
            let (a, b) = (w[k], w[k + 8]);
            w[k] = add(a, b);
            w[k + 8] = sub(a, b);
        }
    }
    let [
        mut w0r,
        mut w0i,
        mut w1r,
        mut w1i,
        mut w2r,
        mut w2i,
        mut w3r,
        mut w3i,
        mut w4r,
        mut w4i,
        mut w5r,
        mut w5i,
        mut w6r,
        mut w6i,
        mut w7r,
        mut w7i,
    ] = w;
    let s2 = SQRT2_OVER_2;
    let new_w5r = mul(s2, add(w5i, w5r));
    let new_w5i = mul(s2, sub(w5i, w5r));
    let new_w7r = mul(s2, sub(w7i, w7r));
    let minus_new_w7i = mul(s2, add(w7i, w7r));
    w5r = new_w5r;
    w5i = new_w5i;
    std::mem::swap(&mut w6r, &mut w6i);
    w7r = new_w7r;
    w7i = minus_new_w7i;
    let b = |a: &mut f32, b: &mut f32| {
        let (na, nb) = (add(*a, *b), sub(*a, *b));
        *a = na;
        *b = nb;
    };
    let bn = |a: &mut f32, b: &mut f32| {
        let (na, nb) = (sub(*a, *b), add(*a, *b));
        *a = na;
        *b = nb;
    };
    b(&mut w0r, &mut w2r);
    b(&mut w0i, &mut w2i);
    b(&mut w1r, &mut w3r);
    b(&mut w1i, &mut w3i);
    b(&mut w4r, &mut w6r);
    bn(&mut w4i, &mut w6i);
    b(&mut w5r, &mut w7r);
    bn(&mut w5i, &mut w7i);
    std::mem::swap(&mut w3r, &mut w3i);
    std::mem::swap(&mut w7r, &mut w7i);
    b(&mut w0r, &mut w1r);
    b(&mut w0i, &mut w1i);
    b(&mut w2r, &mut w3r);
    bn(&mut w2i, &mut w3i);
    b(&mut w4r, &mut w5r);
    b(&mut w4i, &mut w5i);
    b(&mut w6r, &mut w7r);
    bn(&mut w6i, &mut w7i);
    std::mem::swap(&mut w1r, &mut w4r);
    std::mem::swap(&mut w1i, &mut w4i);
    std::mem::swap(&mut w3r, &mut w6r);
    std::mem::swap(&mut w3i, &mut w6i);
    // psimd_fft16_real_f32
    let half = 0.5f32;
    let g1r = mul(half, add(w1r, w7r));
    let g1i = mul(half, sub(w1i, w7i));
    let g2r = mul(half, add(w2r, w6r));
    let g2i = mul(half, sub(w2i, w6i));
    let g3r = mul(half, add(w3r, w5r));
    let g3i = mul(half, sub(w3i, w5i));
    let two_h1r = add(w1i, w7i);
    let two_h1i = sub(w7r, w1r);
    let two_h2r = add(w2i, w6i);
    let two_h2i = sub(w6r, w2r);
    let two_h3r = add(w3i, w5i);
    let two_h3i = sub(w5r, w3r);
    let h2_plus = mul(SQRT2_OVER_4, add(two_h2i, two_h2r));
    let h2_minus = mul(SQRT2_OVER_4, sub(two_h2i, two_h2r));
    let c1 = COS_1PI_OVER_8 * 0.5;
    let c3 = COS_3PI_OVER_8 * 0.5;
    [
        add(w0r, w0i),
        sub(w0r, w0i),
        fma(two_h1i, c3, fma(two_h1r, c1, g1r)),
        fma(-two_h1r, c3, fma(two_h1i, c1, g1i)),
        add(g2r, h2_plus),
        add(h2_minus, g2i),
        fma(two_h3i, c1, fma(two_h3r, c3, g3r)),
        fma(-two_h3r, c1, fma(two_h3i, c3, g3i)),
        w4r,
        -w4i,
        fma(-two_h3i, c1, fma(-two_h3r, c3, g3r)),
        fma(-two_h3r, c1, fma(two_h3i, c3, -g3i)),
        sub(g2r, h2_plus),
        sub(h2_minus, g2i),
        fma(-two_h1i, c3, fma(-two_h1r, c1, g1r)),
        fma(-two_h1r, c3, fma(two_h1i, c1, -g1i)),
    ]
}

/// `psimd_ifft16_real_f32` for one column: block rows `f0r f8r f1r f1i … f7i` →
/// the 16 signal rows.
fn ifft16_real(f: &[f32; 16]) -> [f32; 16] {
    let s = 0.5f32;
    let [
        f0r,
        f8r,
        f1r,
        f1i,
        f2r,
        f2i,
        f3r,
        f3i,
        f4r,
        f4i,
        f5r,
        f5i,
        f6r,
        f6i,
        f7r,
        f7i,
    ] = *f;
    let (f0r, f8r, f1r, f1i, f2r, f2i, f3r, f3i) = (
        mul(f0r, s),
        mul(f8r, s),
        mul(f1r, s),
        mul(f1i, s),
        mul(f2r, s),
        mul(f2i, s),
        mul(f3r, s),
        mul(f3i, s),
    );
    let (f5r, f5i, f6r, f6i, f7r, f7i) = (
        mul(f5r, s),
        mul(f5i, s),
        mul(f6r, s),
        mul(f6i, s),
        mul(f7r, s),
        mul(f7i, s),
    );
    let w0r = add(f0r, f8r);
    let w0i = sub(f0r, f8r);
    let w4r = f4r;
    let w4i = -f4i;
    let g2r = add(f2r, f6r);
    let g2i = sub(f2i, f6i);
    let h2r = sub(f2r, f6r);
    let h2i = add(f2i, f6i);
    let h2_plus = add(h2r, h2i);
    let h2_minus = sub(h2r, h2i);
    let s2 = SQRT2_OVER_2;
    let w2r = fma(-s2, h2_plus, g2r);
    let w2i = fma(s2, h2_minus, g2i);
    let w6r = fma(s2, h2_plus, g2r);
    let w6i = fma(s2, h2_minus, -g2i);
    let g1r = add(f1r, f7r);
    let g1i = sub(f1i, f7i);
    let g3r = add(f3r, f5r);
    let g3i = sub(f3i, f5i);
    let h1r = sub(f1r, f7r);
    let h1i = add(f1i, f7i);
    let h3r = sub(f3r, f5r);
    let h3i = add(f3i, f5i);
    let (c1, c3) = (COS_1PI_OVER_8, COS_3PI_OVER_8);
    let w1r = fma(-h1r, c3, fma(-h1i, c1, g1r));
    let w1i = fma(-h1i, c3, fma(h1r, c1, g1i));
    let w7r = fma(h1r, c3, fma(h1i, c1, g1r));
    let w7i = fma(-h1i, c3, fma(h1r, c1, -g1i));
    let w3r = fma(-h3r, c1, fma(-h3i, c3, g3r));
    let w3i = fma(-h3i, c1, fma(h3r, c3, g3i));
    let w5r = fma(h3r, c1, fma(h3i, c3, g3r));
    let w5i = fma(-h3i, c1, fma(h3r, c3, -g3i));
    ifft8_aos([
        w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i,
    ])
}

/// `psimd_ifft8_aos_f32` → rows 0..8 (`t_lo`) and 8..16 (`t_hi`).
fn ifft8_aos(w: [f32; 16]) -> [f32; 16] {
    let [
        mut w0r,
        mut w0i,
        mut w1r,
        mut w1i,
        mut w2r,
        mut w2i,
        mut w3r,
        mut w3i,
        mut w4r,
        mut w4i,
        mut w5r,
        mut w5i,
        mut w6r,
        mut w6i,
        mut w7r,
        mut w7i,
    ] = w;
    let b = |a: &mut f32, b: &mut f32| {
        let (na, nb) = (add(*a, *b), sub(*a, *b));
        *a = na;
        *b = nb;
    };
    let bn = |a: &mut f32, b: &mut f32| {
        let (na, nb) = (sub(*a, *b), add(*a, *b));
        *a = na;
        *b = nb;
    };
    std::mem::swap(&mut w1r, &mut w4r);
    std::mem::swap(&mut w1i, &mut w4i);
    std::mem::swap(&mut w3r, &mut w6r);
    std::mem::swap(&mut w3i, &mut w6i);
    b(&mut w0r, &mut w1r);
    b(&mut w0i, &mut w1i);
    b(&mut w2r, &mut w3r);
    b(&mut w2i, &mut w3i);
    b(&mut w4r, &mut w5r);
    b(&mut w4i, &mut w5i);
    b(&mut w6r, &mut w7r);
    b(&mut w6i, &mut w7i);
    std::mem::swap(&mut w3r, &mut w3i);
    std::mem::swap(&mut w7r, &mut w7i);
    b(&mut w0r, &mut w2r);
    b(&mut w0i, &mut w2i);
    bn(&mut w1r, &mut w3r);
    b(&mut w1i, &mut w3i);
    b(&mut w4r, &mut w6r);
    b(&mut w4i, &mut w6i);
    bn(&mut w5r, &mut w7r);
    b(&mut w5i, &mut w7i);
    let s = SQRT2_OVER_2 * 0.125;
    let new_w5r = mul(s, sub(w5r, w5i));
    let new_w5i = mul(s, add(w5r, w5i));
    let minus_new_w7r = mul(s, add(w7r, w7i));
    let new_w7i = mul(s, sub(w7r, w7i));
    w5r = new_w5r;
    w5i = new_w5i;
    std::mem::swap(&mut w6r, &mut w6i);
    w7r = minus_new_w7r;
    w7i = new_w7i;
    let sc = 0.125f32;
    for v in [
        &mut w0r, &mut w0i, &mut w1r, &mut w1i, &mut w2r, &mut w2i, &mut w3r, &mut w3i, &mut w4r,
        &mut w4i, &mut w6r, &mut w6i,
    ] {
        *v = mul(*v, sc);
    }
    let mut out = [0.0f32; 16];
    b(&mut w0r, &mut w4r);
    (out[0], out[8]) = (w0r, w4r);
    b(&mut w0i, &mut w4i);
    (out[1], out[9]) = (w0i, w4i);
    b(&mut w1r, &mut w5r);
    (out[2], out[10]) = (w1r, w5r);
    b(&mut w1i, &mut w5i);
    (out[3], out[11]) = (w1i, w5i);
    bn(&mut w2r, &mut w6r);
    (out[4], out[12]) = (w2r, w6r);
    b(&mut w2i, &mut w6i);
    (out[5], out[13]) = (w2i, w6i);
    bn(&mut w3r, &mut w7r);
    (out[6], out[14]) = (w3r, w7r);
    b(&mut w3i, &mut w7i);
    (out[7], out[15]) = (w3i, w7i);
    out
}

/// A 16×16 block, `[row][column]`.
type Block = [[f32; 16]; 16];

fn quads(row: &[f32; 16]) -> [V; 4] {
    [
        [row[0], row[1], row[2], row[3]],
        [row[4], row[5], row[6], row[7]],
        [row[8], row[9], row[10], row[11]],
        [row[12], row[13], row[14], row[15]],
    ]
}
fn set_quads(row: &mut [f32; 16], q: &[V; 4]) {
    for (k, v) in q.iter().enumerate() {
        row[4 * k..4 * k + 4].copy_from_slice(v);
    }
}

/// The row stage shared by the forward transforms: dual-real FFT of rows 0, 1 and a
/// complex FFT of each row pair (2k, 2k+1).
fn rows_forward(block: &mut Block) {
    for r in (0..16).step_by(2) {
        let (mut s, mut h) = (quads(&block[r]), quads(&block[r + 1]));
        if r == 0 {
            fft16_dualreal(&mut s, &mut h);
        } else {
            fft16_soa(&mut s, &mut h);
        }
        set_quads(&mut block[r], &s);
        set_quads(&mut block[r + 1], &h);
    }
}

/// `nnp_fft16x16_with_offset__psimd`: the `row_count × column_count` patch of `data`
/// (row stride `stride`) placed at `(row_offset, column_offset)` in a zero 16×16 block,
/// transformed. Returns the 32 tuples of 8 floats in NNPACK's order.
fn fft16x16(
    data: &[f32],
    stride: usize,
    row_count: usize,
    column_count: usize,
    row_offset: usize,
    column_offset: usize,
) -> [[f32; 8]; 32] {
    let mut block: Block = [[0.0; 16]; 16];
    let rows = |get: &dyn Fn(usize) -> f32| -> [Option<f32>; 16] {
        let mut x = [None; 16];
        for (r, slot) in x.iter_mut().enumerate() {
            if r >= row_offset && r < row_offset + row_count {
                *slot = Some(get(r));
            }
        }
        x
    };
    if column_count >= 4 {
        // every data column transformed in place; the rest of the block stays +0
        for c in 0..column_count {
            let col = rows(&|r| data[(r - row_offset) * stride + c]);
            let f = fft16_real(&col);
            for (r, v) in f.iter().enumerate() {
                block[r][column_offset + c] = *v;
            }
        }
    } else {
        // the patch is copied into the block and four block columns (zeros
        // included) are transformed
        for r in 0..row_count {
            for c in 0..column_count {
                block[row_offset + r][column_offset + c] = data[r * stride + c];
            }
        }
        let column = column_offset.min(12);
        let src = block;
        for c in column..column + 4 {
            let col = rows(&|r| src[r][c]);
            let f = fft16_real(&col);
            for (r, v) in f.iter().enumerate() {
                block[r][c] = *v;
            }
        }
    }
    rows_forward(&mut block);
    let mut t = [[0.0f32; 8]; 32];
    for r in (0..16).step_by(2) {
        for q in 0..4 {
            let k = (r / 2) * 4 + q;
            t[k][..4].copy_from_slice(&block[r][4 * q..4 * q + 4]);
            t[k][4..].copy_from_slice(&block[r + 1][4 * q..4 * q + 4]);
        }
    }
    t
}

/// `nnp_ifft16x16_with_bias__psimd`: the tuples back to a block, `bias` added to the
/// DC term, the top-left `row_count × column_count` written to `out` (row stride
/// `stride`).
fn ifft16x16_with_bias(
    t: &[[f32; 8]; 32],
    bias: f32,
    out: &mut [f32],
    stride: usize,
    row_count: usize,
    column_count: usize,
) {
    let mut block: Block = [[0.0; 16]; 16];
    for r in (0..16).step_by(2) {
        for q in 0..4 {
            let k = (r / 2) * 4 + q;
            block[r][4 * q..4 * q + 4].copy_from_slice(&t[k][..4]);
            block[r + 1][4 * q..4 * q + 4].copy_from_slice(&t[k][4..]);
        }
    }
    // block.as_float[0][0] += (*bias) * 256.0f — one expression, fused
    block[0][0] = fma(bias, 256.0, block[0][0]);
    for r in (0..16).step_by(2) {
        let (mut s, mut h) = (quads(&block[r]), quads(&block[r + 1]));
        if r == 0 {
            ifft16_dualreal(&mut s, &mut h);
        } else {
            ifft16_soa(&mut s, &mut h);
        }
        set_quads(&mut block[r], &s);
        set_quads(&mut block[r + 1], &h);
    }
    // column groups beyond `column_count` are not transformed (and not read)
    for q in 0..4 {
        if q > 0 && column_count <= 4 * q {
            break;
        }
        for c in 4 * q..4 * q + 4 {
            let f: [f32; 16] = std::array::from_fn(|r| block[r][c]);
            let x = ifft16_real(&f);
            for (r, v) in x.iter().enumerate() {
                block[r][c] = *v;
            }
        }
    }
    for r in 0..row_count {
        for c in 0..column_count {
            out[r * stride + c] = block[r][c];
        }
    }
}

/// `acc += a · conj(b)` for one tuple, as the NEON tuple GEMMs accumulate: tuple 0
/// (`s4c2gemm`) holds two real pairs in lanes 0, 1 and complex numbers in lanes 2, 3;
/// the others (`c4gemm`) four complex numbers.
#[inline]
fn tuple_mac(acc: &mut [f32; 8], a: &[f32; 8], b: &[f32; 8], real_pairs: bool) {
    for l in 0..4 {
        let (ar, ai, br, bi) = (a[l], a[4 + l], b[l], b[4 + l]);
        if real_pairs && l < 2 {
            acc[l] = fma(ar, br, acc[l]);
            acc[4 + l] = fma(ai, bi, acc[4 + l]);
        } else {
            acc[l] = fma(ai, bi, fma(ar, br, acc[l]));
            acc[4 + l] = fma(-ar, bi, fma(ai, br, acc[4 + l]));
        }
    }
}

/// NNPACK's `enum nnp_status` codes this convolution can return.
pub mod status {
    pub const INVALID_BATCH_SIZE: i32 = 2;
    pub const INVALID_INPUT_CHANNELS: i32 = 4;
    pub const INVALID_OUTPUT_CHANNELS: i32 = 5;
    pub const INVALID_INPUT_SIZE: i32 = 10;
    pub const INVALID_INPUT_PADDING: i32 = 12;
    pub const INVALID_KERNEL_SIZE: i32 = 13;
    pub const UNSUPPORTED_ALGORITHM: i32 = 26;
    /// Not NNPACK's: a buffer shorter than its shape (C would read out of bounds).
    pub const BUFFER_TOO_SMALL: i32 = -1;
}

/// `nnp_convolution_output(nnp_convolution_algorithm_auto, …)` as torch's
/// `_nnpack_spatial_convolution` calls it for a batch of 2 or more (a batch of 1 goes
/// to `nnp_convolution_inference`, not ported; torch's conv dispatch only picks NNPACK
/// at batch ≥ 16) — stride 1, identity activation: NCHW
/// `input` (`batch × in_ch × in_h × in_w`), `kernel` (`out_ch × in_ch × k_h × k_w`),
/// `bias` (`out_ch`), padding `[top, right, bottom, left]`. Returns the NCHW output, or
/// NNPACK's status ([`status`]) for the arguments it refuses — validated in its order.
/// The configurations `auto` sends to `ft8x8` / `wt8x8` are not ported and return
/// `UNSUPPORTED_ALGORITHM`.
#[allow(clippy::too_many_arguments)]
pub fn conv_output(
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    in_h: usize,
    in_w: usize,
    pad: [usize; 4],
    k_h: usize,
    k_w: usize,
    input: &[f32],
    kernel: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, i32> {
    use status::*;
    let icb = INPUT_CHANNELS_BLOCK;
    let [top, right, bottom, left] = pad;
    // validate_convolution_arguments, in its order
    if batch == 0 {
        return Err(INVALID_BATCH_SIZE);
    }
    if in_ch == 0 {
        return Err(INVALID_INPUT_CHANNELS);
    }
    if out_ch == 0 {
        return Err(INVALID_OUTPUT_CHANNELS);
    }
    if in_h.min(in_w) == 0 {
        return Err(INVALID_INPUT_SIZE);
    }
    if top.max(bottom) >= k_h || left.max(right) >= k_w {
        return Err(INVALID_INPUT_PADDING);
    }
    if k_h.min(k_w) == 0 {
        return Err(INVALID_KERNEL_SIZE);
    }
    // nnp_convolution_algorithm_auto: ft16x16 for kernels wider than 8, and for
    // smaller kernels when 8x8 tiles would need more than 4x as many tiles
    // (otherwise ft8x8 / wt8x8, which are not ported)
    let (oh, ow) = (top + in_h + bottom)
        .checked_sub(k_h)
        .zip((left + in_w + right).checked_sub(k_w))
        .map(|(h, w)| (h + 1, w + 1))
        .ok_or(INVALID_INPUT_SIZE)?;
    if k_h.max(k_w) <= 8 {
        let tiles = |t: usize| oh.div_ceil(t - k_h + 1) * ow.div_ceil(t - k_w + 1);
        if tiles(8) <= 4 * tiles(16) {
            return Err(UNSUPPORTED_ALGORITHM);
        }
    }
    if k_h > 16 || k_w > 16 {
        return Err(UNSUPPORTED_ALGORITHM);
    }
    if input.len() < batch * in_ch * in_h * in_w
        || kernel.len() < out_ch * in_ch * k_h * k_w
        || bias.len() < out_ch
    {
        return Err(BUFFER_TOO_SMALL);
    }
    // padding < kernel keeps every tile's first input row/column inside the input
    let (out_h, out_w) = (top + in_h + bottom - k_h + 1, left + in_w + right - k_w + 1);
    let (tile_oh, tile_ow) = (16 - k_h + 1, 16 - k_w + 1);
    let doz = |a: usize, b: usize| a.saturating_sub(b);
    // kernel transforms, [oc][ic]
    let ktr: Vec<[[f32; 8]; 32]> = (0..out_ch * in_ch)
        .map(|i| fft16x16(&kernel[i * k_h * k_w..], k_w, k_h, k_w, 0, 0))
        .collect();
    let mut out = vec![0.0f32; batch * out_ch * out_h * out_w];
    for y in (0..out_h).step_by(tile_oh) {
        let input_y = doz(y, top).min(in_h);
        let row_offset = doz(top, y);
        let row_count = (in_h - input_y).min(16 - row_offset);
        for x in (0..out_w).step_by(tile_ow) {
            let input_x = doz(x, left).min(in_w);
            let column_offset = doz(left, x);
            let column_count = (in_w - input_x).min(16 - column_offset);
            debug_assert!(row_count > 0 && column_count > 0);
            let itr: Vec<[[f32; 8]; 32]> = (0..batch * in_ch)
                .map(|i| {
                    fft16x16(
                        &input[i * in_h * in_w + input_y * in_w + input_x..],
                        in_w,
                        row_count,
                        column_count,
                        row_offset,
                        column_offset,
                    )
                })
                .collect();
            let (rc, cc) = (tile_oh.min(out_h - y), tile_ow.min(out_w - x));
            for b in 0..batch {
                for oc in 0..out_ch {
                    let mut t = [[0.0f32; 8]; 32];
                    for (k, tk) in t.iter_mut().enumerate() {
                        for blk in (0..in_ch).step_by(icb) {
                            let mut acc = [0.0f32; 8];
                            for ic in blk..(blk + icb).min(in_ch) {
                                tuple_mac(
                                    &mut acc,
                                    &itr[b * in_ch + ic][k],
                                    &ktr[oc * in_ch + ic][k],
                                    k == 0,
                                );
                            }
                            if blk == 0 {
                                *tk = acc;
                            } else {
                                for l in 0..8 {
                                    tk[l] = add(tk[l], acc[l]);
                                }
                            }
                        }
                    }
                    let base = (b * out_ch + oc) * out_h * out_w + y * out_w + x;
                    ifft16x16_with_bias(&t, bias[oc], &mut out[base..], out_w, rc, cc);
                }
            }
        }
    }
    Ok(out)
}
