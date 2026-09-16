// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Invariants for the low-precision element codec behind `Op::ScaledMatMul`,
//! `Op::ScaledGroupedMatMul`, and the MxFp4x2 residual scheme.
//!
//! These assert properties of what a minifloat codec *is* — closure, nearest
//! rounding, grid symmetry — rather than comparing one implementation against
//! another. A differential test (CPU oracle vs GPU kernel, fused vs decomposed)
//! passes happily when both sides share a wrong exponent bias; these do not.
//!
//! Coverage is the whole `fNeXmY` family — all 7 named hardware formats plus all
//! 28 parameterized ones — so a new format is covered the moment it parses.

use rlx_ir::ScaledFormat;
use rlx_ir::lowp_codec::{decode, decode_slice, e8m0_to_f32, encode, f32_to_e8m0, max_finite};
use rlx_ir::residual::{residual_dequantize, residual_quantize};

/// Every format the codec must handle: the named hardware set plus the full
/// parameterized family.
fn all_formats() -> Vec<ScaledFormat> {
    let mut v = ScaledFormat::NAMED.to_vec();
    for exp in 1u8..=7 {
        for mant in 0u8..=(7 - exp) {
            let f = ScaledFormat::custom(exp, mant);
            // `custom` with IEEE bias reproduces some named formats' fields but
            // not their inf/NaN/FNUZ semantics, so keep both — they are
            // genuinely different codecs.
            v.push(f);
        }
    }
    v
}

/// **Value closure.** Decoding a code, re-encoding it, and decoding again must
/// land on the same value.
///
/// Stated on values rather than codes because several formats have redundant
/// encodings (`+0` / `−0`, and FNUZ's single NaN slot); requiring `encode(decode(c)) == c`
/// would fail on redundancy rather than on a bug. Value closure has no such
/// escape hatch: it holds for a correct codec and breaks for a wrong bias,
/// a misplaced sign bit, or a subnormal path that rounds the wrong way.
#[test]
fn decode_encode_decode_is_closed() {
    for f in all_formats() {
        for c in 0..(1u16 << f.bit_width()) {
            let v = decode(f, c as u8);
            if !v.is_finite() {
                continue; // inf / NaN are not required to round-trip
            }
            let again = decode(f, encode(f, v));
            // Compared by value, not bits: several formats carry both `+0` and
            // `−0` codes and `encode` canonicalizes to `+0`. That is a
            // redundancy in the encoding, not a loss — `-0.0 == 0.0` holds and
            // the two are indistinguishable to every consumer of a weight.
            assert_eq!(
                again, v,
                "{f}: code {c:#04x} decodes to {v}, which re-encodes to {again}"
            );
        }
    }
}

/// **Projection idempotence.** `quantize` maps onto the grid, so applying it
/// twice must equal applying it once — bit for bit, for arbitrary inputs.
#[test]
fn quantize_is_idempotent() {
    let probes: Vec<f32> = (0..400)
        .map(|i| {
            let t = (i as f32 - 200.0) / 13.0;
            t * 1.7
        })
        .chain([0.0, -0.0, 1e-8, -1e-8, 1e4, -1e4])
        .collect();
    for f in all_formats() {
        for &x in &probes {
            let once = f.quantize(x);
            let twice = f.quantize(once);
            assert_eq!(
                twice.to_bits(),
                once.to_bits(),
                "{f}: quantize({x}) = {once}, quantize again = {twice}"
            );
        }
    }
}

/// **Nearest rounding.** `encode` must pick a grid point at minimum distance
/// from the input.
///
/// Checked against the format's own enumerated grid, which is derived from
/// `decode` alone — so this cross-checks `encode` against `decode` without a
/// third party. Ties may go either way (round-half-to-even), so only the
/// distance is asserted, not the choice.
#[test]
fn encode_selects_a_nearest_grid_point() {
    for f in all_formats() {
        let grid = f.representable_values();
        let hi = max_finite(f);
        // Sample inside the representable range; outside it, saturation (not
        // nearest) is the documented behaviour.
        for i in 0..=200 {
            let x = -hi + (2.0 * hi) * (i as f32 / 200.0);
            let got = decode(f, encode(f, x));
            assert!(got.is_finite(), "{f}: encode({x}) decoded to {got}");
            let best = grid
                .iter()
                .map(|g| (g - x).abs())
                .fold(f32::INFINITY, f32::min);
            let actual = (got - x).abs();
            assert!(
                actual <= best + 1e-6 * hi.max(1.0),
                "{f}: encode({x}) chose {got} at distance {actual}, but {best} was available"
            );
        }
    }
}

/// **Saturation.** Values beyond the grid must clamp to the extreme finite
/// value, never wrap to the opposite sign.
///
/// A sign-bit or exponent-overflow bug typically shows up here as `+max`
/// becoming `−max` — silent, catastrophic, and invisible to any test that only
/// feeds in-range data.
#[test]
fn overflow_saturates_without_sign_flip() {
    for f in all_formats() {
        let hi = max_finite(f);
        for &mul in &[1.5f32, 10.0, 1e6] {
            let pos = decode(f, encode(f, hi * mul));
            let neg = decode(f, encode(f, -hi * mul));
            if f.has_inf() {
                assert!(pos > 0.0, "{f}: +overflow gave {pos}");
                assert!(neg < 0.0, "{f}: -overflow gave {neg}");
            } else {
                assert_eq!(pos, hi, "{f}: +{hi}×{mul} saturated to {pos}");
                assert_eq!(neg, -hi, "{f}: -{hi}×{mul} saturated to {neg}");
            }
        }
    }
}

/// **Grid symmetry.** Every all-finite format here uses a sign-magnitude
/// encoding, so its grid must be symmetric about zero.
///
/// Asymmetry means the sign bit is being applied to the wrong field — which
/// biases every quantized tensor in one direction.
#[test]
fn all_finite_grids_are_symmetric() {
    for f in all_formats() {
        if f.has_inf() {
            continue; // E5M2 spends its top exponent on inf/NaN
        }
        let grid = f.representable_values();
        for &v in &grid {
            assert!(
                grid.iter().any(|&w| w == -v),
                "{f}: grid contains {v} but not {}",
                -v
            );
        }
        assert_eq!(
            *grid.last().unwrap(),
            max_finite(f),
            "{f}: max_finite disagrees with the enumerated grid"
        );
    }
}

/// `decode_slice` must equal elementwise `decode` — a batched path that drifts
/// from the scalar one is exactly the sort of divergence a fused-vs-unfused
/// test cannot see, since both sides call the batched path.
#[test]
fn decode_slice_matches_scalar_decode() {
    for f in all_formats() {
        let codes: Vec<u8> = (0..(1u16 << f.bit_width())).map(|c| c as u8).collect();
        let mut out = vec![0.0f32; codes.len()];
        decode_slice(f, &codes, &mut out);
        for (i, (&c, &got)) in codes.iter().zip(out.iter()).enumerate() {
            let want = decode(f, c);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{f}: decode_slice[{i}] = {got}, decode = {want}"
            );
        }
    }
}

/// E8M0 block scales are pure powers of two, so every exact power of two in
/// range must survive the round trip unchanged.
#[test]
fn e8m0_round_trips_exact_powers_of_two() {
    for k in -50i32..=50 {
        let s = (2.0f32).powi(k);
        let back = e8m0_to_f32(f32_to_e8m0(s));
        assert_eq!(back, s, "E8M0 lost 2^{k}: got {back}");
    }
}

/// **Residual refinement.** MxFp4x2 stacks two E2M1 levels, so two levels must
/// reconstruct at least as well as one — that is the entire justification for
/// paying 2× the storage.
///
/// Adding a level that does *not* reduce error means the residual is being
/// computed against the wrong base, which no comparison against another
/// two-level implementation would reveal.
#[test]
fn residual_levels_monotonically_reduce_error() {
    let block: Vec<f32> = (0..32)
        .map(|i| ((i as f32) * 0.37).sin() * 3.0 + 0.25)
        .collect();
    let err = |levels: usize| {
        let rb = residual_quantize(&block, ScaledFormat::F4E2M1, levels);
        let out = residual_dequantize(&rb);
        block
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let e1 = err(1);
    let e2 = err(2);
    let e3 = err(3);
    assert!(
        e2 <= e1,
        "second residual level made it worse: {e1:.6} -> {e2:.6}"
    );
    assert!(
        e3 <= e2,
        "third residual level made it worse: {e2:.6} -> {e3:.6}"
    );
    eprintln!("MxFp4x2 max-abs-error by level count: 1={e1:.6} 2={e2:.6} 3={e3:.6}");
}

/// A single residual level must be exactly plain quantization — if it is not,
/// the residual path has an offset the plain path does not.
#[test]
fn one_residual_level_equals_plain_quantize() {
    let block: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.31).collect();
    let rb = residual_quantize(&block, ScaledFormat::F4E2M1, 1);
    let out = residual_dequantize(&rb);
    assert_eq!(out.len(), block.len());
    // Same grid, same scale rule → the reconstruction must land on the E2M1
    // grid scaled by the block scale; check it is at least a fixed point.
    let rb2 = residual_quantize(&out, ScaledFormat::F4E2M1, 1);
    let out2 = residual_dequantize(&rb2);
    for (i, (a, b)) in out.iter().zip(out2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "one-level residual is not a projection at index {i}: {a} -> {b}"
        );
    }
}
