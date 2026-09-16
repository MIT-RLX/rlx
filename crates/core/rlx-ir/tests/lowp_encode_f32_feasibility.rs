// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Can the low-precision encoder be ported to a shading language?**
//!
//! `lowp_codec::encode` picks the nearest representable code by exhaustive
//! search, comparing `|decode(code) as f64 - x as f64|` and breaking ties toward
//! an even code LSB. The `f64` is deliberate — see the comment on `encode`.
//!
//! That matters well beyond the CPU: WGSL has no `f64` at all, and naga's GLSL
//! front end does not implement doubles either. So a native `Op::ScaledQuantize`
//! on wgpu or Vulkan must do the same search in `f32` — and if `f32` ever picks
//! a different code, those backends would quantize differently from the oracle
//! every other backend is checked against. Silent, small, and everywhere.
//!
//! This settles it by measurement instead of by argument about mantissa widths:
//! run both encoders over a wide sweep of `f32` inputs for every format and
//! report the first disagreement. It is a *feasibility gate* — the answer is
//! recorded here so the next person porting these kernels does not have to
//! re-derive it, and so the answer cannot silently change under a codec edit.

use rlx_ir::ScaledFormat;
use rlx_ir::lowp_codec::{decode, encode, max_finite};

const FORMATS: &[ScaledFormat] = &[
    ScaledFormat::F8E4M3,
    ScaledFormat::F8E5M2,
    ScaledFormat::F8E4M3Fnuz,
    ScaledFormat::F8E5M2Fnuz,
    ScaledFormat::F6E2M3,
    ScaledFormat::F6E3M2,
    ScaledFormat::F4E2M1,
];

/// `lowp_codec::encode` with every `f64` replaced by `f32` — i.e. exactly what a
/// WGSL / GLSL port of the kernel is able to compute.
fn encode_f32_only(fmt: ScaledFormat, x: f32) -> u8 {
    if x.is_nan() {
        return 0;
    }
    if x.is_infinite() {
        return encode_f32_only(fmt, x.signum() * max_finite(fmt));
    }
    let n_codes: u16 = 1 << fmt.bit_width();
    let mut best: u8 = 0;
    let mut best_err = f32::INFINITY;
    let mut best_mant_lsb: u8 = 1;
    for c in 0..n_codes {
        let code = c as u8;
        let v = decode(fmt, code);
        if !v.is_finite() {
            continue;
        }
        let err = (v - x).abs();
        let mant_lsb = code & 1;
        if err < best_err || (err == best_err && mant_lsb < best_mant_lsb) {
            best_err = err;
            best = code;
            best_mant_lsb = mant_lsb;
        }
    }
    best
}

/// A deterministic spread of f32 values: the grid points themselves, the exact
/// midpoints between adjacent grid points (where ties live), and a xorshift
/// sweep across the format's dynamic range and well past it.
fn probes(fmt: ScaledFormat) -> Vec<f32> {
    let mut xs = Vec::new();
    let n_codes: u16 = 1 << fmt.bit_width();

    // Every representable value, and every midpoint between two of them — the
    // midpoints are the whole reason the tie-break exists.
    let mut grid: Vec<f32> = (0..n_codes)
        .map(|c| decode(fmt, c as u8))
        .filter(|v| v.is_finite())
        .collect();
    grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    grid.dedup();
    for w in grid.windows(2) {
        xs.push(w[0]);
        xs.push(0.5 * (w[0] + w[1]));
        // Just off the midpoint in both directions.
        xs.push(f32::from_bits(
            (0.5 * (w[0] + w[1])).to_bits().wrapping_add(1),
        ));
        xs.push(f32::from_bits(
            (0.5 * (w[0] + w[1])).to_bits().wrapping_sub(1),
        ));
    }
    xs.push(*grid.last().unwrap());

    // Pseudo-random f32s spanning the format's range and far outside it, so the
    // "x is enormous relative to the grid" case is covered too.
    let mx = max_finite(fmt);
    let mut state: u32 = 0x1234_5678;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..20_000 {
        let u = (next() >> 8) as f32 / 16_777_216.0; // [0,1)
        let sign = if next() & 1 == 0 { 1.0 } else { -1.0 };
        // Scales from far below the grid to far above it.
        let scale = match next() % 5 {
            0 => mx * 1e-6,
            1 => mx * 0.01,
            2 => mx,
            3 => mx * 4.0,
            _ => mx * 1e6,
        };
        xs.push(sign * u * scale);
    }
    xs.push(0.0);
    xs.push(-0.0);
    xs.push(f32::MIN_POSITIVE);
    xs.push(-f32::MIN_POSITIVE);
    xs
}

#[test]
fn f32_only_encoder_agrees_with_the_f64_oracle() {
    let mut disagreements = 0usize;
    let mut first: Option<String> = None;

    for &fmt in FORMATS {
        for x in probes(fmt) {
            let a = encode(fmt, x);
            let b = encode_f32_only(fmt, x);
            if a != b {
                disagreements += 1;
                if first.is_none() {
                    first = Some(format!(
                        "{fmt:?}: x={x:e} (bits {:#010x}) — oracle code {a} → {}, \
                         f32-only code {b} → {}",
                        x.to_bits(),
                        decode(fmt, a),
                        decode(fmt, b)
                    ));
                }
            }
        }
    }

    assert_eq!(
        disagreements,
        0,
        "the f32-only encoder is NOT a faithful port — {disagreements} disagreements.\n\
         First: {}\n\
         Consequence: a native ScaledQuantize on wgpu / Vulkan (neither has f64) \
         would quantize differently from the CPU oracle. Keep the encoder on the \
         host for those backends, or widen the search to a compensated form.",
        first.unwrap_or_default()
    );
}
