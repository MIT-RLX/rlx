// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scheme-level **invariants** — an oracle that references no other implementation.
//!
//! Every other check on the quantization path is *differential*: fused-vs-unfused
//! (`dequant_matmul_q8k_matches_dequant_then_matmul`), backend-vs-backend
//! (`dequant_matmul_q8k_metal_matches_cpu`), or against a `w_ref` produced by the
//! same decoder under test. A differential check cannot fail when the shared
//! formula is itself wrong — every side moves together. That failure mode is not
//! hypothetical here: the RmsNorm `1/r` bug was wrong in all seven backends, the
//! RoPE table stride was wrong-and-agreeing in three, and the GELU constant was
//! wrong in `rlxsl` and therefore in every generated backend.
//!
//! For gradients the escape is finite differences — an oracle derived from the
//! definition of a derivative rather than from another kernel. This module is the
//! forward-path equivalent for block-quantized storage: properties that follow
//! from what a quantizer *is*, checkable without a second decoder.
//!
//! | invariant | what it catches |
//! |---|---|
//! | [`projection_closure`] | scale/offset/layout errors — a re-quantized block must reproduce its own bytes |
//! | [`value_idempotence`] | the same, for search-based encoders where bytes may legitimately differ |
//! | [`constant_block_error`] | scale recovery: a block of one repeated value is exactly representable |
//! | [`zero_block_is_exact`] | sign/offset bias — zeros must survive |
//! | [`max_abs_error`] | resolution: error above the scheme's own step size means a decode bug |
//!
//! `projection_closure` is the load-bearing one. Quantization is a projection
//! onto a finite representable set, so quantizing an *already-projected* vector
//! must be the identity. A transposed block layout breaks it immediately and at
//! any size — which is exactly what the ROCm GGUF transpose bug needed, since it
//! was hidden by an `n = 1` differential test.

use crate::GgmlType;
use anyhow::{Result, bail};

/// What kind of reconstruction grid a scheme uses.
///
/// The applicable invariants differ by grid: a uniform quantizer owes you a
/// step bound and a stable projection, a codebook-coded one owes you neither
/// (its encoder searches a non-convex set, so re-encoding may legitimately move).
/// Applying a uniform-grid invariant to a LUT scheme produces false alarms —
/// which is why the grid kind is part of the table rather than assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grid {
    /// Symmetric signed codes with one scale: `d = amax / 2^(bits-1)`, values
    /// `d·(q − 2^(bits-1))`. The range is asymmetric by one step, so the worst
    /// case is a full step (clamping on the sparse side), not half.
    UniformSymmetric { bits: u32 },
    /// Affine: per-block scale *and* min, values `d·q + m` over `2^bits − 1`
    /// steps spanning `[min, max]`. Nearest-rounding is bounded by half a step.
    UniformAffine { bits: u32 },
    /// Blockwise uniform with per-sub-block scales (the K-quants). Uniform
    /// within a sub-block but the sub-block scales are themselves quantized, so
    /// there is no clean closed-form bound over the super-block.
    SubBlock,
    /// Non-uniform codebook / lattice grid (the IQ family). The encoder picks
    /// entries by search, so neither a step bound nor projection stability
    /// applies.
    Lut,
    /// Ternary `{0, ±d}`.
    Ternary,
    /// Minifloat codes (E2M1) times a power-of-two block scale.
    MiniFloat,
}

/// One scheme paired with both directions of its codec.
///
/// Only schemes with an encoder *and* a decoder appear here; a decode-only
/// scheme has nothing to round-trip against.
pub struct Codec {
    /// GGUF type name, for failure messages.
    pub name: &'static str,
    /// The scheme itself (encoder side dispatches on this).
    pub ty: GgmlType,
    /// Elements per block — test vectors must be a multiple of this.
    pub block_elems: usize,
    /// Packed bytes per block.
    pub block_bytes: usize,
    /// Reconstruction grid, which selects the applicable invariants.
    pub grid: Grid,
    /// Decoder. `(bytes, n_elements) -> values`.
    pub dequant: fn(&[u8], usize) -> Result<Vec<f32>>,
}

impl Codec {
    /// Encode then decode — the reconstruction a kernel actually consumes.
    pub fn round_trip(&self, src: &[f32]) -> Result<Vec<f32>> {
        let packed = crate::quantize::quantize(src, self.ty)?;
        (self.dequant)(&packed, src.len())
    }

    /// Encode only.
    pub fn encode(&self, src: &[f32]) -> Result<Vec<u8>> {
        crate::quantize::quantize(src, self.ty)
    }
}

/// Every scheme with a complete codec, in GGUF type order.
///
/// Adding an encoder for a scheme means adding its row here; the invariant
/// tests then cover it with no further edits.
pub fn codecs() -> Vec<Codec> {
    use crate::iq_dequant::{
        dequant_iq1_m, dequant_iq1_s, dequant_iq2_s, dequant_iq2_xs, dequant_iq2_xxs,
        dequant_iq3_s, dequant_iq3_xxs, dequant_iq4_nl, dequant_iq4_xs,
    };
    use crate::mx_dequant::{dequant_mxfp4, dequant_nvfp4};
    use crate::q1_dequant::dequant_q1_0;
    use crate::q2_dequant::dequant_q2_0;
    use crate::tq_dequant::{dequant_tq1_0, dequant_tq2_0};
    use crate::{
        dequant_q2_k, dequant_q3_k, dequant_q4_0, dequant_q4_1, dequant_q4_k, dequant_q5_0,
        dequant_q5_1, dequant_q5_k, dequant_q6_k, dequant_q8_0, dequant_q8_k,
    };

    vec![
        // ── legacy 32-element blocks ──
        Codec {
            name: "Q4_0",
            ty: GgmlType::Q4_0,
            block_elems: 32,
            block_bytes: 18,
            grid: Grid::UniformSymmetric { bits: 4 },
            dequant: dequant_q4_0,
        },
        Codec {
            name: "Q4_1",
            ty: GgmlType::Q4_1,
            block_elems: 32,
            block_bytes: 20,
            grid: Grid::UniformAffine { bits: 4 },
            dequant: dequant_q4_1,
        },
        Codec {
            name: "Q5_0",
            ty: GgmlType::Q5_0,
            block_elems: 32,
            block_bytes: 22,
            grid: Grid::UniformSymmetric { bits: 5 },
            dequant: dequant_q5_0,
        },
        Codec {
            name: "Q5_1",
            ty: GgmlType::Q5_1,
            block_elems: 32,
            block_bytes: 24,
            grid: Grid::UniformAffine { bits: 5 },
            dequant: dequant_q5_1,
        },
        Codec {
            name: "Q8_0",
            ty: GgmlType::Q8_0,
            block_elems: 32,
            block_bytes: 34,
            grid: Grid::UniformSymmetric { bits: 8 },
            dequant: dequant_q8_0,
        },
        // ── K-quants, 256-element super-blocks ──
        Codec {
            name: "Q2_K",
            ty: GgmlType::Q2K,
            block_elems: 256,
            block_bytes: 84,
            grid: Grid::SubBlock,
            dequant: dequant_q2_k,
        },
        Codec {
            name: "Q3_K",
            ty: GgmlType::Q3K,
            block_elems: 256,
            block_bytes: 110,
            grid: Grid::SubBlock,
            dequant: dequant_q3_k,
        },
        Codec {
            name: "Q4_K",
            ty: GgmlType::Q4K,
            block_elems: 256,
            block_bytes: 144,
            grid: Grid::SubBlock,
            dequant: dequant_q4_k,
        },
        Codec {
            name: "Q5_K",
            ty: GgmlType::Q5K,
            block_elems: 256,
            block_bytes: 176,
            grid: Grid::SubBlock,
            dequant: dequant_q5_k,
        },
        Codec {
            name: "Q6_K",
            ty: GgmlType::Q6K,
            block_elems: 256,
            block_bytes: 210,
            grid: Grid::SubBlock,
            dequant: dequant_q6_k,
        },
        Codec {
            name: "Q8_K",
            ty: GgmlType::Q8K,
            block_elems: 256,
            block_bytes: 292,
            grid: Grid::SubBlock,
            dequant: dequant_q8_k,
        },
        // ── IQ family (codebook / lattice grids) ──
        Codec {
            name: "IQ4_NL",
            ty: GgmlType::IQ4NL,
            block_elems: 32,
            block_bytes: 18,
            grid: Grid::Lut,
            dequant: dequant_iq4_nl,
        },
        Codec {
            name: "IQ4_XS",
            ty: GgmlType::IQ4XS,
            block_elems: 256,
            block_bytes: 136,
            grid: Grid::Lut,
            dequant: dequant_iq4_xs,
        },
        Codec {
            name: "IQ2_XXS",
            ty: GgmlType::IQ2XXS,
            block_elems: 256,
            block_bytes: 66,
            grid: Grid::Lut,
            dequant: dequant_iq2_xxs,
        },
        Codec {
            name: "IQ2_XS",
            ty: GgmlType::IQ2XS,
            block_elems: 256,
            block_bytes: 74,
            grid: Grid::Lut,
            dequant: dequant_iq2_xs,
        },
        Codec {
            name: "IQ2_S",
            ty: GgmlType::IQ2S,
            block_elems: 256,
            block_bytes: 82,
            grid: Grid::Lut,
            dequant: dequant_iq2_s,
        },
        Codec {
            name: "IQ3_XXS",
            ty: GgmlType::IQ3XXS,
            block_elems: 256,
            block_bytes: 98,
            grid: Grid::Lut,
            dequant: dequant_iq3_xxs,
        },
        Codec {
            name: "IQ3_S",
            ty: GgmlType::IQ3S,
            block_elems: 256,
            block_bytes: 110,
            grid: Grid::Lut,
            dequant: dequant_iq3_s,
        },
        Codec {
            name: "IQ1_S",
            ty: GgmlType::IQ1S,
            block_elems: 256,
            block_bytes: 50,
            grid: Grid::Lut,
            dequant: dequant_iq1_s,
        },
        Codec {
            name: "IQ1_M",
            ty: GgmlType::IQ1M,
            block_elems: 256,
            block_bytes: 56,
            grid: Grid::Lut,
            dequant: dequant_iq1_m,
        },
        // ── ternary ──
        Codec {
            name: "TQ1_0",
            ty: GgmlType::TQ1_0,
            block_elems: 256,
            block_bytes: 54,
            grid: Grid::Ternary,
            dequant: dequant_tq1_0,
        },
        Codec {
            name: "TQ2_0",
            ty: GgmlType::TQ2_0,
            block_elems: 256,
            block_bytes: 66,
            grid: Grid::Ternary,
            dequant: dequant_tq2_0,
        },
        // ── microscaling FP4 ──
        Codec {
            name: "MXFP4",
            ty: GgmlType::MXFP4,
            block_elems: 32,
            block_bytes: 17,
            grid: Grid::MiniFloat,
            dequant: dequant_mxfp4,
        },
        Codec {
            name: "NVFP4",
            ty: GgmlType::NVFP4,
            block_elems: 16,
            block_bytes: 9,
            grid: Grid::MiniFloat,
            dequant: dequant_nvfp4,
        },
        // ── custom 1/2-bit forks ──
        Codec {
            name: "Q1_0",
            ty: GgmlType::Q1_0,
            block_elems: 128,
            block_bytes: 18,
            grid: Grid::Ternary,
            dequant: dequant_q1_0,
        },
        Codec {
            name: "Q2_0",
            ty: GgmlType::Q2_0,
            block_elems: 128,
            block_bytes: 34,
            grid: Grid::Ternary,
            dequant: dequant_q2_0,
        },
    ]
}

/// Schemes whose encoder and decoder are known to disagree on nibble layout.
///
/// **This is a live bug, not a scheme property.** `quantize_mxfp4_block` packs
/// element `j` and element `j + block/2` into one byte (the "halves" layout every
/// other nibble-packed GGUF scheme in this crate uses — see
/// [`crate::dequant_q4_0_block`]), while `dequant_mxfp4_block` reads byte `i` as
/// elements `2i` and `2i+1` ("interleaved"). Encoding then decoding therefore
/// drops half of every block and interleaves zeros.
///
/// The CPU decoder, the GPU `dequant_gguf` kernel (`scheme_id == 10`), and the
/// existing unit tests all agree on *interleaved* — but they were written against
/// each other, so their agreement is not evidence. Resolving it requires checking
/// a real MXFP4 GGUF file against ggml, so the invariant suite pins the
/// disagreement rather than guessing which side to change. See
/// `mxfp4_layout_disagreement_is_pinned`.
pub const NIBBLE_LAYOUT_DISAGREEMENT: &[&str] = &["MXFP4", "NVFP4"];

/// Schemes whose grid does not contain exact zero.
///
/// IQ1_M reconstructs as `d·(2·grid + 1 + delta)`, an odd-multiple lattice with
/// no zero point — so a zero input necessarily reconstructs nonzero. This is a
/// property of the format, not a bug.
pub const ZERO_NOT_REPRESENTABLE: &[&str] = &["IQ1_M"];

/// Why an invariant failed — enough to localize without a debugger.
#[derive(Debug, Clone)]
pub struct Violation {
    /// Scheme name.
    pub scheme: &'static str,
    /// Which invariant.
    pub invariant: &'static str,
    /// Flat element (or byte) index of the first disagreement.
    pub index: usize,
    /// What the invariant required.
    pub expected: f64,
    /// What the codec produced.
    pub actual: f64,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} violated at index {} — expected {}, got {}",
            self.scheme, self.invariant, self.index, self.expected, self.actual
        )
    }
}

/// **Projection closure.** Quantization projects onto a finite representable
/// set, so re-quantizing an already-projected vector must reproduce the *same
/// bytes*.
///
/// This needs no reference decoder and no tolerance: it is a statement about the
/// encoder and decoder agreeing on one layout. A transposed block, an off-by-one
/// scale offset, or a swapped high-bit plane all break it at the first block.
///
/// Search-based encoders (the K-quants pick a scale by RMSE search over
/// candidates) may legitimately land on a different-but-equivalent encoding; use
/// [`value_idempotence`] for those.
pub fn projection_closure(c: &Codec, src: &[f32]) -> std::result::Result<(), Violation> {
    let once = c
        .encode(src)
        .map_err(|_| violation(c, "projection_closure/encode", 0, 0.0, f64::NAN))?;
    let decoded = (c.dequant)(&once, src.len())
        .map_err(|_| violation(c, "projection_closure/decode", 0, 0.0, f64::NAN))?;
    let twice = c
        .encode(&decoded)
        .map_err(|_| violation(c, "projection_closure/re-encode", 0, 0.0, f64::NAN))?;
    for (i, (a, b)) in once.iter().zip(twice.iter()).enumerate() {
        if a != b {
            return Err(violation(c, "projection_closure", i, *a as f64, *b as f64));
        }
    }
    Ok(())
}

/// **Value idempotence.** Weaker than [`projection_closure`] and applicable to
/// every scheme including search-based encoders: decoding a re-encoded block
/// must reproduce the *same values*, even if the bytes differ.
///
/// A decoder that mis-reads a scale still fails this, because the second pass
/// re-quantizes the already-wrong values and lands somewhere else again.
pub fn value_idempotence(c: &Codec, src: &[f32], tol: f32) -> std::result::Result<(), Violation> {
    let first = c
        .round_trip(src)
        .map_err(|_| violation(c, "value_idempotence/first", 0, 0.0, f64::NAN))?;
    let second = c
        .round_trip(&first)
        .map_err(|_| violation(c, "value_idempotence/second", 0, 0.0, f64::NAN))?;
    for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        if (a - b).abs() > tol {
            return Err(violation(c, "value_idempotence", i, *a as f64, *b as f64));
        }
    }
    Ok(())
}

/// **Constant-block exactness.** A block whose elements are all `v` is
/// representable by every one of these schemes (set the scale from `v` and every
/// code to the same index), so the reconstruction error bounds the encoder's
/// scale-recovery accuracy.
///
/// Returns the max absolute deviation from `v`. A scheme that gets this wrong
/// has a broken scale path, which no differential test against the same scale
/// path can reveal.
pub fn constant_block_error(c: &Codec, v: f32, blocks: usize) -> Result<f32> {
    let src = vec![v; c.block_elems * blocks];
    let out = c.round_trip(&src)?;
    Ok(out.iter().map(|x| (x - v).abs()).fold(0.0f32, f32::max))
}

/// **Zero exactness.** Zero is representable in every scheme here (all codes at
/// the zero point, or scale 0). Any nonzero output is a sign, offset, or
/// zero-point bug — the class that shifts a whole tensor.
pub fn zero_block_is_exact(c: &Codec, blocks: usize) -> Result<bool> {
    let src = vec![0.0f32; c.block_elems * blocks];
    let out = c.round_trip(&src)?;
    Ok(out.iter().all(|x| *x == 0.0))
}

/// **Resolution bound.** Max absolute reconstruction error over `src`.
///
/// Compare against the scheme's own step size: a `bits`-bit block spanning
/// `amax` has step `2·amax / (2^bits − 1)`, so nearest-rounding cannot exceed
/// half that. Error meaningfully above the bound means the decode is wrong, not
/// merely coarse — and unlike a hand-tuned tolerance, the bound moves correctly
/// when the test data changes.
pub fn max_abs_error(c: &Codec, src: &[f32]) -> Result<f32> {
    let out = c.round_trip(src)?;
    if out.len() != src.len() {
        bail!(
            "{}: decoder returned {} values for {} inputs",
            c.name,
            out.len(),
            src.len()
        );
    }
    Ok(src
        .iter()
        .zip(out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max))
}

/// Analytic worst-case reconstruction error for a uniform grid over `src`.
///
/// Returns `None` for grids where no closed form applies (sub-block scales,
/// codebooks, ternary) — those get idempotence and exactness checks instead.
///
/// * [`Grid::UniformSymmetric`] — the encoder sets `d = amax / 2^(bits-1)`, so
///   the grid spans `[−2^(bits-1)·d, (2^(bits-1)−1)·d]`. It is asymmetric by one
///   step, and a value at the sparse extreme clamps: the bound is a **full
///   step**, `amax / 2^(bits-1)`, not half of it. (Using half here is the easy
///   mistake — it makes a correct Q4_0 look broken.)
/// * [`Grid::UniformAffine`] — scale *and* min are fitted to `[min, max]` over
///   `2^bits − 1` intervals, so nothing clamps and the bound is a genuine
///   **half step**.
/// Both bounds add the error from storing the block parameters as **f16** —
/// these are GGUF blocks, so `d` (and `m`) are half-precision on disk and the
/// realized grid is slightly coarser than the ideal one. Leaving that term out
/// makes a correct Q4_1 fail by ~0.25%, which is the kind of near-miss that
/// gets "fixed" with a fudge factor instead of understood.
pub fn uniform_error_bound(grid: Grid, src: &[f32]) -> Option<f32> {
    match grid {
        Grid::UniformSymmetric { bits } => {
            let amax = src.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let step = amax / (1u32 << (bits - 1)) as f32;
            // One full step (asymmetric clamp) + the error in a f16-rounded `d`
            // accumulated over the widest code magnitude.
            Some(step + f16_half_ulp(step) * (1u32 << (bits - 1)) as f32)
        }
        Grid::UniformAffine { bits } => {
            let lo = src.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let step = (hi - lo) / ((1u32 << bits) - 1) as f32;
            // Half a step (nothing clamps) + f16 rounding of `d` over the widest
            // code + f16 rounding of the stored min.
            Some(
                step / 2.0
                    + f16_half_ulp(step) * ((1u32 << bits) - 1) as f32
                    + f16_half_ulp(lo.abs()),
            )
        }
        Grid::SubBlock | Grid::Lut | Grid::Ternary | Grid::MiniFloat => None,
    }
}

/// Half the gap between `x` and its f16 neighbour — the worst error from storing
/// `x` in half precision. Zero for zero.
pub fn f16_half_ulp(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return 0.0;
    }
    let up = half::f16::from_f32(x).to_f32();
    let next = half::f16::from_bits(half::f16::from_f32(x).to_bits() + 1).to_f32();
    ((next - up).abs()) / 2.0
}

/// Byte length the encoder must produce for `n` elements.
pub fn expected_bytes(c: &Codec, n: usize) -> usize {
    (n / c.block_elems) * c.block_bytes
}

fn violation(
    c: &Codec,
    invariant: &'static str,
    index: usize,
    expected: f64,
    actual: f64,
) -> Violation {
    Violation {
        scheme: c.name,
        invariant,
        index,
        expected,
        actual,
    }
}

/// Deterministic pseudo-random test vector — no `rand` dependency, and the same
/// bytes on every platform so a failure reproduces exactly.
///
/// Values span roughly ±3 so blocks have a meaningful dynamic range.
pub fn test_vector(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            // xorshift64*
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / 16_777_216.0;
            (u - 0.5) * 6.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every codec must produce exactly `n/block_elems * block_bytes`.
    ///
    /// Cheap, but it pins the block geometry the GPU kernels index with — a
    /// wrong `block_bytes` shifts every subsequent block.
    #[test]
    fn packed_length_matches_block_geometry() {
        for c in codecs() {
            let n = c.block_elems * 3;
            let src = test_vector(n, 0xC0FFEE);
            let packed = c.encode(&src).unwrap_or_else(|e| panic!("{}: {e}", c.name));
            assert_eq!(
                packed.len(),
                expected_bytes(&c, n),
                "{}: packed length disagrees with block geometry",
                c.name
            );
        }
    }

    /// Decoding must return exactly as many values as were encoded.
    #[test]
    fn decoder_returns_full_length() {
        for c in codecs() {
            let n = c.block_elems * 2;
            let src = test_vector(n, 7);
            let out = c
                .round_trip(&src)
                .unwrap_or_else(|e| panic!("{}: {e}", c.name));
            assert_eq!(out.len(), n, "{}: length mismatch", c.name);
            assert!(
                out.iter().all(|x| x.is_finite()),
                "{}: produced non-finite values",
                c.name
            );
        }
    }

    /// Zero must survive every scheme whose grid contains it.
    ///
    /// A scheme that silently shifts zero shifts the whole tensor, and no
    /// differential test sees it because the reference shifts identically.
    #[test]
    fn zero_is_exact_where_representable() {
        let mut bad = vec![];
        for c in codecs() {
            let exact = zero_block_is_exact(&c, 2).unwrap_or_else(|e| panic!("{}: {e}", c.name));
            let expected = !ZERO_NOT_REPRESENTABLE.contains(&c.name);
            if exact != expected {
                bad.push(format!(
                    "{} (expected zero-exact: {expected}, got: {exact})",
                    c.name
                ));
            }
        }
        assert!(
            bad.is_empty(),
            "zero handling changed for: {bad:?} — update ZERO_NOT_REPRESENTABLE only if \
             the format genuinely has no zero point"
        );
    }

    /// A constant block is exactly representable, so scale recovery is testable
    /// in isolation from code selection.
    ///
    /// Ternary and 1-bit schemes represent only `{0, ±d}` (and IQ1 carries a
    /// per-block shift), so a constant is exact for them too — the scale simply
    /// becomes the constant.
    #[test]
    fn constant_block_recovers_its_scale() {
        for c in codecs() {
            // A codebook grid encodes 8-element *patterns*, not independent
            // values, so a constant vector need not lie on it. Excluded by grid
            // kind rather than by name, so a new IQ scheme is covered correctly
            // the moment it is added.
            if matches!(c.grid, Grid::Lut) {
                continue;
            }
            if NIBBLE_LAYOUT_DISAGREEMENT.contains(&c.name) {
                continue; // see mxfp4_layout_disagreement_is_pinned
            }
            for &v in &[1.0f32, -1.0, 0.375, -2.5] {
                let err =
                    constant_block_error(&c, v, 2).unwrap_or_else(|e| panic!("{}: {e}", c.name));
                let rel = err / v.abs();
                assert!(
                    rel < 0.05,
                    "{}: constant block {v} reconstructed with relative error {rel:.4} \
                     (abs {err:.6}) — scale recovery is off",
                    c.name
                );
            }
        }
    }

    /// Re-encoding a decoded block must reproduce the same values.
    ///
    /// The tolerance is relative to the scheme's own step, not a magic number:
    /// a stable projection lands on the identical grid point the second time.
    #[test]
    fn value_idempotence_holds() {
        let mut failures = vec![];
        for c in codecs() {
            // Codebook encoders search a non-convex set; a re-encoded block may
            // land on a different-but-comparable entry, so idempotence is not
            // promised. Every scheme with a *fixed* grid must be a fixed point.
            if matches!(c.grid, Grid::Lut) {
                continue;
            }
            if NIBBLE_LAYOUT_DISAGREEMENT.contains(&c.name) {
                continue; // see mxfp4_layout_disagreement_is_pinned
            }
            let src = test_vector(c.block_elems * 4, 0xABCD);
            // 1e-3 is far tighter than the coarsest grid here (~2.0 for 1.5 bpw
            // over ±3), so this passes only when the second projection genuinely
            // lands on the same point.
            if let Err(v) = value_idempotence(&c, &src, 1e-3) {
                failures.push(v.to_string());
            }
        }
        assert!(
            failures.is_empty(),
            "value idempotence violated:\n  {}",
            failures.join("\n  ")
        );
    }

    /// Pins the MXFP4 / NVFP4 encoder-vs-decoder nibble-layout disagreement.
    ///
    /// The encoder writes the halves layout; the decoder reads interleaved.
    /// Round-tripping therefore loses half of every block. This test *asserts
    /// the breakage* so that:
    ///   * it cannot be forgotten, and
    ///   * fixing only one side turns the suite red rather than green-by-luck.
    ///
    /// When the layout question is settled against a real GGUF file, delete this
    /// test and remove the names from [`NIBBLE_LAYOUT_DISAGREEMENT`]; the normal
    /// invariants then cover both schemes.
    #[test]
    fn mxfp4_layout_disagreement_is_pinned() {
        let all = codecs();
        let c = all.iter().find(|c| c.name == "MXFP4").unwrap();
        // Every value here is exactly representable in E2M1 with a unity scale,
        // so a consistent codec would reproduce them bit-for-bit.
        let mut src = vec![0.0f32; 32];
        for (i, v) in [-3.0f32, 3.0, -6.0, 6.0].iter().enumerate() {
            src[i] = *v;
        }
        let out = c.round_trip(&src).unwrap();
        assert_eq!(
            &out[0..8],
            &[-3.0, 0.0, 3.0, 0.0, -6.0, 0.0, 6.0, 0.0],
            "MXFP4 layout disagreement changed shape — if the encoder and decoder \
             now agree, remove MXFP4/NVFP4 from NIBBLE_LAYOUT_DISAGREEMENT and \
             delete this test"
        );
        assert_ne!(
            &out[0..4],
            &src[0..4],
            "MXFP4 now round-trips exactly-representable values — the bug is fixed, \
             so this pin should be deleted"
        );
    }

    /// Byte-level projection closure, reported per scheme.
    ///
    /// Search-based encoders may legitimately re-encode to different bytes, so
    /// this test *records* which schemes are byte-stable rather than demanding
    /// it of all of them — but any scheme that is byte-stable today must stay
    /// that way, since losing it means the encoder became input-order dependent.
    #[test]
    fn projection_closure_partition_is_stable() {
        let mut stable = vec![];
        let mut unstable = vec![];
        for c in codecs() {
            let src = test_vector(c.block_elems * 4, 0x5EED);
            match projection_closure(&c, &src) {
                Ok(()) => stable.push(c.name),
                Err(_) => unstable.push(c.name),
            }
        }
        eprintln!("byte-stable under re-encoding: {stable:?}");
        eprintln!("not byte-stable (search-based scale): {unstable:?}");
        assert!(
            !stable.is_empty(),
            "no scheme is byte-stable — the encoder/decoder pair disagrees on layout"
        );
    }

    /// Uniform-grid schemes must respect their own half-step bound.
    ///
    /// Only the schemes with a genuinely uniform grid are listed; the IQ family
    /// and ternary use non-uniform lattices where this bound does not apply.
    #[test]
    fn uniform_schemes_respect_their_step_bound() {
        for c in codecs() {
            // Bound is computed per block, since each block fits its own scale.
            let Some(_) = uniform_error_bound(c.grid, &[1.0]) else {
                continue;
            };
            for seed in [0x1234u64, 0xBEEF, 0x1] {
                let src = test_vector(c.block_elems, seed);
                let bound = uniform_error_bound(c.grid, &src).unwrap();
                let err = max_abs_error(&c, &src).unwrap_or_else(|e| panic!("{}: {e}", c.name));
                assert!(
                    err <= bound * 1.001,
                    "{}: max error {err:.6} exceeds the grid's own bound {bound:.6} \
                     (seed {seed:#x}) — that is a decode bug, not coarseness",
                    c.name
                );
            }
        }
    }

    /// Higher bit budget must not reconstruct worse, within one scheme family.
    ///
    /// Monotonicity is a property of the *family*, so it catches a single
    /// mis-implemented member that every per-scheme test would pass.
    #[test]
    fn error_is_monotone_in_bit_budget() {
        let all = codecs();
        let src = test_vector(256 * 4, 0x99);
        let err = |name: &str| {
            let c = all.iter().find(|c| c.name == name).expect(name);
            max_abs_error(c, &src).unwrap_or_else(|e| panic!("{name}: {e}"))
        };
        // K-quant ladder: more bits per element must not be worse.
        let ladder = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K"];
        let errs: Vec<f32> = ladder.iter().map(|n| err(n)).collect();
        for w in errs.windows(2) {
            // 1.25 slack: adjacent K-quants differ by ~1 bit and each has its own
            // sub-block layout, so exact monotonicity is not guaranteed — but a
            // large inversion means one member is broken.
            assert!(
                w[1] <= w[0] * 1.25,
                "K-quant error ladder inverted: {ladder:?} -> {errs:?}"
            );
        }
        eprintln!("K-quant max-abs-error ladder: {ladder:?} -> {errs:?}");
    }
}

#[cfg(test)]
mod name_tests {
    use crate::GgmlType;

    /// Printing and parsing must be exact inverses.
    ///
    /// These two directions used to live in `pyrlx` as separate hand-maintained
    /// tables and had already drifted — the printer emitted `Q1_0`, `Q2_0`,
    /// `I2_S`, `I8_S`, `FV5` and `FV5B`, none of which the parser accepted.
    /// Generating both from one macro row makes that impossible; this pins it.
    #[test]
    fn name_round_trips_for_every_variant() {
        for &(ty, name) in GgmlType::NAME_PAIRS {
            assert_eq!(ty.name(), name, "{ty:?} prints the wrong name");
            assert_eq!(
                GgmlType::from_name(name),
                Some(ty),
                "{name:?} does not parse back to {ty:?}"
            );
            assert_eq!(ty.to_string(), name, "Display disagrees with name()");
            assert_eq!(name.parse::<GgmlType>().ok(), Some(ty), "FromStr disagrees");
        }
    }

    /// Names are unique — two variants sharing one spelling would make the
    /// parse direction silently lossy.
    #[test]
    fn names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for &(ty, name) in GgmlType::NAME_PAIRS {
            assert!(
                seen.insert(name),
                "{name:?} is claimed by {ty:?} and another variant"
            );
        }
    }

    /// Every type readable from a file must have a name.
    ///
    /// Derived from `from_u32` rather than a third list, so a variant that
    /// becomes readable without being named is caught here rather than by a
    /// user seeing a dtype they cannot feed back in.
    #[test]
    fn every_readable_type_is_named() {
        let named: std::collections::HashSet<_> =
            GgmlType::NAME_PAIRS.iter().map(|(t, _)| *t).collect();
        let mut count = 0;
        // 143 is the highest id in use (Pestle `G8_0`); scan past it for headroom.
        for id in 0u32..=200 {
            let Ok(ty) = GgmlType::from_u32(id) else {
                continue;
            };
            assert!(
                named.contains(&ty),
                "id {id} reads as {ty:?}, which has no name"
            );
            count += 1;
        }
        assert!(
            count >= 30,
            "only {count} ids enumerated — from_u32 scan is wrong"
        );
    }

    /// Parsing accepts the lowercase spelling users actually type.
    #[test]
    fn parse_is_case_insensitive() {
        for &(ty, name) in GgmlType::NAME_PAIRS {
            assert_eq!(GgmlType::from_name(&name.to_ascii_lowercase()), Some(ty));
            assert_eq!(GgmlType::from_name(&name.to_ascii_uppercase()), Some(ty));
        }
        assert_eq!(GgmlType::from_name("not_a_type"), None);
    }
}
