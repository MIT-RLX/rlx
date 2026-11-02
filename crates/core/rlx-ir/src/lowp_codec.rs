// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical scalar f32 ↔ low-precision codec for [`ScaledFormat`].
//!
//! One place owns the bit-exact decode/encode for every native-compute
//! low-precision element format (FP8 e4m3/e5m2 OCP + AMD FNUZ, FP6 e2m3/e3m2,
//! FP4 e2m1). Hardware tensor cores do this in silicon; this module is the
//! CPU **parity oracle** every GPU backend is checked against, plus the
//! reference quantizer that produces operands for [`crate::op::Op::ScaledMatMul`].
//!
//! Decode is exact (closed-form from the field bits). Encode is
//! nearest-representable by exhaustive search of the (≤256-entry) code space —
//! simple and obviously correct for an oracle, with round-half-to-even ties and
//! saturation of overflow/inf to the largest finite magnitude (the saturating
//! quantization mode hardware uses). NaN inputs map to `0`.

// Float methods `core` lacks (`FloatExt`) without `std`.
#[allow(unused_imports)]
use crate::prelude::*;
use crate::quant::ScaledFormat;

/// Decode one packed low-precision code (in the low [`ScaledFormat::bit_width`]
/// bits of `code`) to f32. Returns `f32::NAN` / `±inf` for the formats that
/// encode them.
#[inline]
pub fn decode(fmt: ScaledFormat, code: u8) -> f32 {
    // FP4 has a tiny fixed table shared with the existing NVFP4 path.
    if matches!(fmt, ScaledFormat::F4E2M1) {
        return crate::nvfp4::fp4_e2m1_to_f32(code);
    }

    let (e_bits, m_bits, bias) = fmt.fields();
    let width = e_bits + m_bits; // bits excluding sign
    let sign_bit = (code >> width) & 1;
    let exp = (u32::from(code) >> m_bits) & ((1u32 << e_bits) - 1);
    let mant = u32::from(code) & ((1u32 << m_bits) - 1);
    let sign = if sign_bit == 1 { -1.0f32 } else { 1.0f32 };
    let max_exp = (1u32 << e_bits) - 1;

    // Special encodings, per family.
    if fmt.is_fnuz() {
        // AMD FNUZ: the would-be −0 slot (sign=1, exp=0, mant=0) is the
        // sole NaN; there is no infinity and no negative zero.
        if sign_bit == 1 && exp == 0 && mant == 0 {
            return f32::NAN;
        }
    } else if fmt.has_inf() {
        // OCP E5M2 (IEEE-like): max exponent → inf (mant 0) or NaN.
        if exp == max_exp {
            return if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }
    } else if matches!(fmt, ScaledFormat::F8E4M3) {
        // OCP E4M3 (finite): only S.1111.111 is NaN; everything else finite.
        if exp == max_exp && mant == (1u32 << m_bits) - 1 {
            return f32::NAN;
        }
    }
    // FP6 e2m3 / e3m2 are all-finite — no special codes.

    let m_div = (1u32 << m_bits) as f32;
    let val = if exp == 0 {
        // Subnormal (and zero): no implicit leading 1.
        (mant as f32 / m_div) * 2f32.powi(1 - bias)
    } else {
        (1.0 + mant as f32 / m_div) * 2f32.powi(exp as i32 - bias)
    };
    sign * val
}

/// Encode an f32 to the nearest representable code (round-half-to-even,
/// saturating overflow/inf to the max finite magnitude, NaN → `0`).
#[inline]
pub fn encode(fmt: ScaledFormat, x: f32) -> u8 {
    if x.is_nan() {
        return 0;
    }
    // ±inf saturates to the largest finite magnitude of the matching sign. We
    // snap to ±max_finite directly rather than a generic huge value: a value far
    // outside the grid is equidistant from every finite code in f64, so the
    // tie-break would wrongly pick code 0.
    if x.is_infinite() {
        return encode(fmt, x.signum() * max_finite(fmt));
    }
    let n_codes: u16 = 1 << fmt.bit_width();
    let mut best: u8 = 0;
    let mut best_err = f64::INFINITY;
    let mut best_mant_lsb: u8 = 1; // prefer even mantissa LSB on ties
    let xd = x as f64;
    for c in 0..n_codes {
        let code = c as u8;
        let v = decode(fmt, code);
        if !v.is_finite() {
            continue; // never round *to* inf/NaN; overflow saturates instead
        }
        let err = (v as f64 - xd).abs();
        let mant_lsb = code & 1;
        if err < best_err || (err == best_err && mant_lsb < best_mant_lsb) {
            best_err = err;
            best = code;
            best_mant_lsb = mant_lsb;
        }
    }
    best
}

/// Largest finite magnitude representable in `fmt`.
pub fn max_finite(fmt: ScaledFormat) -> f32 {
    let n_codes: u16 = 1 << fmt.bit_width();
    (0..n_codes)
        .map(|c| decode(fmt, c as u8).abs())
        .filter(|v| v.is_finite())
        .fold(0.0f32, f32::max)
}

/// Decode a whole buffer of codes (1 code per byte) into f32. Goes through a
/// 256-entry table built from [`decode`], so it is bit-identical to it and
/// cheap enough for whole checkpoints.
pub fn decode_slice(fmt: ScaledFormat, codes: &[u8], out: &mut [f32]) {
    debug_assert_eq!(codes.len(), out.len());
    let table: [f32; 256] = core::array::from_fn(|c| decode(fmt, c as u8));
    for (o, &c) in out.iter_mut().zip(codes) {
        *o = table[c as usize];
    }
}

/// Round a positive f32 scale **up** to the nearest power of two and return its
/// OCP **E8M0** byte (biased exponent, bias 127; `0xFF` is NaN, `0x00` is
/// 2^−127). Used for `ScaleLayout::BlockMxE8M0`.
#[inline]
pub fn f32_to_e8m0(scale: f32) -> u8 {
    if !scale.is_finite() || scale <= 0.0 {
        return 0; // degenerate block → smallest scale
    }
    // ceil(log2(scale)) as a biased E8M0 exponent.
    let mut exp = scale.log2().ceil() as i32 + 127;
    exp = exp.clamp(0, 254);
    exp as u8
}

/// Decode an OCP E8M0 scale byte to f32 (`2^(byte-127)`).
#[inline]
pub fn e8m0_to_f32(byte: u8) -> f32 {
    if byte == 0xFF {
        return f32::NAN;
    }
    2f32.powi(byte as i32 - 127)
}

/// Which scale each element of a weight uses, for the scale layouts FP8
/// checkpoints store beside their weights (HF `fp8` `weight_scale_inv`,
/// compressed-tensors `weight_scale`).
///
/// The weight is `shape = [lead.., rows, cols]`. Its scale is one value
/// (per-tensor), `[rows]` / `[lead.., rows, 1]` (per-channel), or a
/// `[lead.., ceil(rows/b0), ceil(cols/b1)]` grid of `b0×b1` blocks
/// (DeepSeek-V3 / Qwen3-FP8 use 128×128). Every case is one block grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleGrid {
    /// Rows and columns of each matrix (a per-tensor scale folds every
    /// leading dim into `rows`).
    pub rows: usize,
    pub cols: usize,
    /// `[b0, b1]`: the block one scale covers.
    pub block: [usize; 2],
    /// `[ceil(rows/b0), ceil(cols/b1)]`: scales per matrix.
    pub grid: [usize; 2],
}

impl ScaleGrid {
    /// The grid for `shape` and `scale_shape`. `block` gives `[b0, b1]`
    /// explicitly; `None` infers it per axis — 128 when that fits the grid
    /// (the HF default), else `ceil(n / blocks)`.
    pub fn new(
        shape: &[usize],
        scale_shape: &[usize],
        block: Option<[usize; 2]>,
    ) -> Result<Self, String> {
        let bad = |why: &str| {
            Err(format!(
                "weight {shape:?} with scale {scale_shape:?}: {why}"
            ))
        };
        let r = shape.len();
        if scale_shape.iter().product::<usize>() == 1 {
            let cols = shape.last().copied().unwrap_or(1);
            let rows = shape[..r.saturating_sub(1)].iter().product();
            return Ok(Self {
                rows,
                cols,
                block: [rows.max(1), cols.max(1)],
                grid: [1, 1],
            });
        }
        if r < 2 {
            return bad("only a per-tensor scale fits a rank < 2 weight");
        }
        // `[rows]` for a 2-D weight is per-channel: `[rows, 1]`.
        let grid: Vec<usize> = if scale_shape.len() == 1 && r == 2 {
            vec![scale_shape[0], 1]
        } else {
            scale_shape.to_vec()
        };
        if grid.len() != r || grid[..r - 2] != shape[..r - 2] {
            return bad("scale rank / leading dims differ from the weight");
        }
        let (rows, cols) = (shape[r - 2], shape[r - 1]);
        let (sr, sc) = (grid[r - 2], grid[r - 1]);
        let infer = |n: usize, s: usize| -> Option<usize> {
            if s == 0 || s > n {
                return None;
            }
            if n.div_ceil(128) == s {
                return Some(128);
            }
            let b = n.div_ceil(s);
            (n.div_ceil(b) == s).then_some(b)
        };
        let (b0, b1) = match block {
            Some([b0, b1]) if b0 > 0 && b1 > 0 => (b0, b1),
            Some(_) => return bad("block size must be non-zero"),
            None => match (infer(rows, sr), infer(cols, sc)) {
                (Some(b0), Some(b1)) => (b0, b1),
                _ => return bad("no block size tiles the weight into the scale grid"),
            },
        };
        if rows.div_ceil(b0) != sr || cols.div_ceil(b1) != sc {
            return bad(&format!("{b0}x{b1} blocks do not give a {sr}x{sc} grid"));
        }
        Ok(Self {
            rows,
            cols,
            block: [b0, b1],
            grid: [sr, sc],
        })
    }

    /// Index into the scale tensor of element `(i, j)` of matrix `m`.
    #[inline]
    pub fn index(&self, m: usize, i: usize, j: usize) -> usize {
        let [b0, b1] = self.block;
        let [sr, sc] = self.grid;
        m * sr * sc + (i / b0) * sc + j / b1
    }
}

/// Multiply decoded weights by their stored scales, laid out as
/// [`ScaleGrid`] describes.
pub fn apply_weight_scales(
    data: &mut [f32],
    shape: &[usize],
    scale: &[f32],
    scale_shape: &[usize],
    block: Option<[usize; 2]>,
) -> Result<(), String> {
    if data.len() != shape.iter().product::<usize>()
        || scale.len() != scale_shape.iter().product::<usize>()
    {
        return Err(format!(
            "weight {shape:?} with scale {scale_shape:?}: data does not match its shape"
        ));
    }
    let g = ScaleGrid::new(shape, scale_shape, block)?;
    if data.is_empty() {
        return Ok(());
    }
    let [sr, sc] = g.grid;
    let [b0, b1] = g.block;
    for (mat, sm) in data
        .chunks_exact_mut(g.rows * g.cols)
        .zip(scale.chunks_exact(sr * sc))
    {
        for (i, row) in mat.chunks_exact_mut(g.cols).enumerate() {
            let srow = &sm[(i / b0) * sc..][..sc];
            for (j, v) in row.iter_mut().enumerate() {
                *v *= srow[j / b1];
            }
        }
    }
    Ok(())
}

/// [`QuantScheme::Fp8E4m3B128`](crate::quant::QuantScheme::Fp8E4m3B128) /
/// [`QuantScheme::Fp8E5m2B128`](crate::quant::QuantScheme::Fp8E5m2B128):
/// FP8 codes with one f32 scale per 128 elements along K, packed as 132-byte
/// `f32 scale | 128 codes` blocks, row-major `[out, in]`.
pub mod fp8_b128 {
    use super::decode;
    use crate::prelude::*;
    use crate::quant::{QuantScheme, ScaledFormat};

    /// Elements per block.
    pub const QK: usize = 128;
    /// Bytes per block: an f32 scale + [`QK`] codes.
    pub const BLOCK_BYTES: usize = 4 + QK;

    /// The element format of a block-FP8 scheme; `None` for any other.
    pub const fn format(scheme: QuantScheme) -> Option<ScaledFormat> {
        match scheme {
            QuantScheme::Fp8E4m3B128 => Some(ScaledFormat::F8E4M3),
            QuantScheme::Fp8E5m2B128 => Some(ScaledFormat::F8E5M2),
            _ => None,
        }
    }

    /// The block-FP8 scheme for OCP `fmt` (E4M3 / E5M2).
    pub const fn scheme(fmt: ScaledFormat) -> Option<QuantScheme> {
        match fmt {
            ScaledFormat::F8E4M3 => Some(QuantScheme::Fp8E4m3B128),
            ScaledFormat::F8E5M2 => Some(QuantScheme::Fp8E5m2B128),
            _ => None,
        }
    }

    /// Packed size of an `[n, k]` weight; `None` unless `k` is a multiple
    /// of [`QK`].
    pub fn packed_bytes(n: usize, k: usize) -> Option<usize> {
        k.is_multiple_of(QK).then(|| n * (k / QK) * BLOCK_BYTES)
    }

    /// Pack FP8 `codes` (`[n, k]`, either format) with `scales` (`[n, k / QK]`,
    /// one per row and block — a 128×128 grid repeats its value down 128 rows).
    pub fn pack(codes: &[u8], scales: &[f32], n: usize, k: usize) -> Result<Vec<u8>, String> {
        let bytes =
            packed_bytes(n, k).ok_or_else(|| format!("fp8_b128: k={k} not a multiple of {QK}"))?;
        let nb = k / QK;
        if codes.len() != n * k || scales.len() != n * nb {
            return Err(format!(
                "fp8_b128: {} codes / {} scales for [{n}, {k}] (want {} / {})",
                codes.len(),
                scales.len(),
                n * k,
                n * nb
            ));
        }
        let mut out = Vec::with_capacity(bytes);
        for (q, s) in codes.as_chunks::<QK>().0.iter().zip(scales) {
            out.extend_from_slice(&s.to_le_bytes());
            out.extend_from_slice(q);
        }
        Ok(out)
    }

    /// Pack a checkpoint's FP8 weight — `codes` of `shape` (`[out, in]` or an
    /// `[E, out, in]` expert stack) with its stored `scale` laid out as
    /// [`super::ScaleGrid`] describes (`None`: no scale, i.e. 1) — as blocks.
    /// `Ok(None)` when the layout can't be one scale per 128 along `in`: `in`
    /// not a multiple of [`QK`], or a scale that changes inside a 128 run.
    pub fn pack_scaled(
        codes: &[u8],
        shape: &[usize],
        scale: Option<(&[f32], &[usize])>,
        block: Option<[usize; 2]>,
    ) -> Result<Option<Vec<u8>>, String> {
        let (experts, n, k) = match *shape {
            [n, k] => (1, n, k),
            [e, n, k] => (e, n, k),
            _ => return Ok(None),
        };
        if !k.is_multiple_of(QK) {
            return Ok(None);
        }
        let (nb, rows) = (k / QK, experts * n);
        let scales = match scale {
            None => vec![1.0f32; rows * nb],
            Some((scale, sshape)) => {
                let g = super::ScaleGrid::new(shape, sshape, block)?;
                if scale.len() != sshape.iter().product::<usize>() {
                    return Err(format!(
                        "fp8_b128: scale {sshape:?} holds {} values",
                        scale.len()
                    ));
                }
                if !(g.block[1].is_multiple_of(QK) || g.grid[1] == 1) {
                    return Ok(None);
                }
                (0..rows * nb)
                    .map(|t| {
                        let (row, j) = (t / nb, (t % nb) * QK);
                        // A per-tensor grid folds the experts into its rows.
                        let (m, i) = if g.rows == n {
                            (row / n, row % n)
                        } else {
                            (0, row)
                        };
                        scale[g.index(m, i, j)]
                    })
                    .collect()
            }
        };
        pack(codes, &scales, rows, k).map(Some)
    }

    /// Dequantize one block of `fmt` codes into `out` (`decode(code) · scale`).
    pub fn dequant_block(fmt: ScaledFormat, block: &[u8], out: &mut [f32]) {
        let s = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        for (o, &c) in out[..QK].iter_mut().zip(&block[4..BLOCK_BYTES]) {
            *o = decode(fmt, c) * s;
        }
    }

    /// Dequantize `n_elems` packed `fmt` elements to f32.
    pub fn dequant(fmt: ScaledFormat, bytes: &[u8], n_elems: usize) -> Result<Vec<f32>, String> {
        if !n_elems.is_multiple_of(QK) || bytes.len() != n_elems / QK * BLOCK_BYTES {
            return Err(format!(
                "fp8_b128: {} bytes for {n_elems} elements (want {})",
                bytes.len(),
                n_elems / QK * BLOCK_BYTES
            ));
        }
        let table: [f32; 256] = core::array::from_fn(|c| decode(fmt, c as u8));
        let mut out = vec![0f32; n_elems];
        for (o, b) in out
            .as_chunks_mut::<QK>()
            .0
            .iter_mut()
            .zip(bytes.as_chunks::<BLOCK_BYTES>().0)
        {
            let s = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            for (v, &c) in o.iter_mut().zip(&b[4..]) {
                *v = table[c as usize] * s;
            }
        }
        Ok(out)
    }

    /// [`dequant`] for a block-FP8 `scheme`; errors on any other scheme.
    pub fn dequant_scheme(
        scheme: QuantScheme,
        bytes: &[u8],
        n_elems: usize,
    ) -> Result<Vec<f32>, String> {
        let fmt = format(scheme).ok_or_else(|| format!("fp8_b128: {scheme} is not block FP8"))?;
        dequant(fmt, bytes, n_elems)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::ScaledFormat::*;

    const ALL: [ScaledFormat; 7] = [
        F8E4M3, F8E5M2, F8E4M3Fnuz, F8E5M2Fnuz, F6E2M3, F6E3M2, F4E2M1,
    ];

    #[test]
    fn round_trip_every_representable_code() {
        // decode → encode → decode must be a fixed point for every finite code.
        for fmt in ALL {
            let n = 1u16 << fmt.bit_width();
            for c in 0..n {
                let v = decode(fmt, c as u8);
                if !v.is_finite() {
                    continue;
                }
                let re = decode(fmt, encode(fmt, v));
                assert_eq!(
                    re, v,
                    "{fmt}: code {c:#04x} decoded {v}, re-encoded to {re}"
                );
            }
        }
    }

    #[test]
    fn known_max_finite() {
        assert_eq!(max_finite(F8E4M3), 448.0);
        assert_eq!(max_finite(F8E5M2), 57344.0);
        assert_eq!(max_finite(F8E4M3Fnuz), 240.0);
        assert_eq!(max_finite(F8E5M2Fnuz), 57344.0);
        assert_eq!(max_finite(F6E2M3), 7.5);
        assert_eq!(max_finite(F6E3M2), 28.0);
        assert_eq!(max_finite(F4E2M1), 6.0);
    }

    #[test]
    fn unity_decodes_to_one() {
        // 1.0 = (1 + 0) * 2^0 → exp == bias, mant == 0.
        for fmt in ALL {
            let (e, m, bias) = fmt.fields();
            let code = ((bias as u32) << m) as u8;
            let _ = e;
            assert_eq!(decode(fmt, code), 1.0, "{fmt}: 1.0 code {code:#04x}");
            // and encode(1.0) round-trips
            assert_eq!(decode(fmt, encode(fmt, 1.0)), 1.0, "{fmt}");
        }
    }

    #[test]
    fn e4m3_has_one_nan_no_inf() {
        assert!(decode(F8E4M3, 0x7F).is_nan());
        assert!(decode(F8E4M3, 0xFF).is_nan());
        // every other code is finite
        let finite = (0..256u16)
            .filter(|&c| decode(F8E4M3, c as u8).is_finite())
            .count();
        assert_eq!(finite, 254);
    }

    #[test]
    fn e5m2_has_inf_and_nan() {
        assert_eq!(decode(F8E5M2, 0x7C), f32::INFINITY); // 0.11111.00
        assert_eq!(decode(F8E5M2, 0xFC), f32::NEG_INFINITY);
        assert!(decode(F8E5M2, 0x7D).is_nan());
    }

    #[test]
    fn fnuz_single_nan_no_neg_zero() {
        assert!(decode(F8E4M3Fnuz, 0x80).is_nan());
        assert!(decode(F8E5M2Fnuz, 0x80).is_nan());
        // +0 only
        assert_eq!(decode(F8E4M3Fnuz, 0x00), 0.0);
        // no infinities anywhere
        assert!(!(0..256u16).any(|c| decode(F8E4M3Fnuz, c as u8).is_infinite()));
    }

    #[test]
    fn saturates_on_overflow() {
        // Values past the max finite saturate (never become inf for finite fmts).
        for fmt in ALL {
            let big = 1.0e9;
            let v = decode(fmt, encode(fmt, big));
            assert!(v.is_finite(), "{fmt} overflow not finite");
            assert_eq!(v, max_finite(fmt), "{fmt} did not saturate to max");
            // ±inf saturates too (never encodes to an inf/NaN slot, even for E5M2
            // which has a representable inf code).
            assert_eq!(
                decode(fmt, encode(fmt, f32::INFINITY)),
                max_finite(fmt),
                "{fmt} +inf did not saturate"
            );
            assert_eq!(
                decode(fmt, encode(fmt, f32::NEG_INFINITY)),
                -max_finite(fmt),
                "{fmt} -inf did not saturate"
            );
        }
    }

    #[test]
    fn e8m0_round_trips_powers_of_two() {
        for p in -10..=10i32 {
            let s = 2f32.powi(p);
            assert_eq!(e8m0_to_f32(f32_to_e8m0(s)), s, "2^{p}");
        }
    }

    // --- Parameterized `Custom { exp_bits, mant_bits, bias }` minifloats ---

    #[test]
    fn custom_f4e3m0_value_table() {
        // f4e3m0 = 1 sign + 3 exp + 0 mant, IEEE bias 3, all-finite. The 8
        // non-negative codes are {±0, 0.25, 0.5, 1, 2, 4, 8, 16}; setting the
        // sign bit (bit 3) negates.
        let f = ScaledFormat::custom(3, 0);
        assert_eq!(f.bit_width(), 4);
        assert_eq!(f.fields(), (3, 0, 3));
        let pos = [0.0f32, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
        for (code, &want) in pos.iter().enumerate() {
            assert_eq!(decode(f, code as u8), want, "code {code}");
            assert_eq!(decode(f, code as u8 | 0x8), -want, "code {}", code + 8);
        }
        assert_eq!(max_finite(f), 16.0);
        // No NaN/inf slots — every code is finite.
        assert!((0..16u8).all(|c| decode(f, c).is_finite()));
    }

    #[test]
    fn custom_e2m1_matches_named_f4e2m1() {
        // A Custom built with E2M1's fields must decode bit-identically to the
        // named F4E2M1 (LUT fast-path) for all 16 codes — proving the generic
        // formula reproduces the hand-written table.
        let custom = ScaledFormat::custom(2, 1); // bias 1
        assert_eq!(custom.fields(), ScaledFormat::F4E2M1.fields());
        for c in 0..16u8 {
            let named = decode(ScaledFormat::F4E2M1, c);
            let generic = decode(custom, c);
            assert!(
                named.to_bits() == generic.to_bits(),
                "e2m1 code {c:#04x}: named {named} vs custom {generic}"
            );
        }
        assert_eq!(max_finite(custom), max_finite(ScaledFormat::F4E2M1));
    }

    #[test]
    fn custom_encode_is_nearest_and_saturates() {
        let f = ScaledFormat::custom(3, 0); // f4e3m0, max finite 16
        for v in [0.25f32, 0.5, 1.0, 2.0, 8.0, 16.0, -4.0, -0.5] {
            assert_eq!(decode(f, encode(f, v)), v, "exact grid point {v}");
        }
        assert_eq!(
            decode(f, encode(f, 100.0)),
            16.0,
            "finite overflow saturates"
        );
        assert_eq!(
            decode(f, encode(f, f32::INFINITY)),
            16.0,
            "+inf saturates to max"
        );
        assert_eq!(
            decode(f, encode(f, f32::NEG_INFINITY)),
            -16.0,
            "-inf saturates"
        );
        assert_eq!(encode(f, f32::NAN), 0, "NaN encodes to 0");
    }

    #[test]
    fn custom_round_trip_every_representable_code() {
        // decode → encode → decode is a fixed point for a spread of exp/mant
        // splits, including a 6-bit and an 8-bit custom.
        for f in [
            ScaledFormat::custom(3, 0), // f4e3m0
            ScaledFormat::custom(2, 2), // f5e2m2
            ScaledFormat::custom(4, 3), // f8e4m3-shaped (finite, no special codes)
            ScaledFormat::custom(3, 4), // f8e3m4
        ] {
            let n = 1u16 << f.bit_width();
            for c in 0..n {
                let v = decode(f, c as u8);
                assert!(v.is_finite(), "{f}: code {c:#04x} unexpectedly non-finite");
                assert_eq!(decode(f, encode(f, v)), v, "{f}: code {c:#04x} ({v})");
            }
        }
    }
}
