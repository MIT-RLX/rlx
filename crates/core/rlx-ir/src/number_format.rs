// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! How numbers are represented when a graph runs: `f32`, `f64`, `f64` as two
//! `f32` (double-single), or any binary float format from `f16` down to 4-bit
//! minifloats, emulated on top of `f32`.
//!
//! [`FloatFormat::round`] is the canonical scalar definition of "round an `f32`
//! to this format": round to nearest, ties to even, gradual underflow, ±0 and
//! NaN pass through, everything at or beyond the largest finite value
//! (infinities included) saturates. Every backend's `Op::RoundToFormat` kernel
//! must be bit-identical to it, and `rlx_tensor::lowp` delegates to it.
//!
//! Inputs and results below `f32`'s smallest normal (`~1.2e-38`) are flushed
//! to a signed zero. Apple GPUs flush `f32` subnormals in hardware even with
//! precise math, so without this the same graph would round differently per
//! backend. It only affects `bf16`'s own subnormals, which most `bf16`
//! hardware also flushes; `f16` and every narrower format keep theirs.
//!
//! Formats with 8 or fewer bits are normally used with a shared scale
//! ([`FormatScaling`]). Scales are powers of two chosen by the OCP MX rule
//! ([`FloatFormat::pow2_scale`]), so applying them is exact on every backend,
//! where dividing by an arbitrary `f32` scale is not (GPU division is not
//! always correctly rounded).

use crate::prelude::*;
use core::fmt;
use core::str::FromStr;

use crate::quant::ScaledFormat;

/// A binary floating-point format emulated on `f32`: a sign bit, `exp_bits`
/// exponent bits with `bias`, and `man_bits` explicit mantissa bits.
///
/// `max_normal` is the largest finite magnitude. IEEE-style formats (`f16`,
/// `bf16`, `f8e5m2`, the `fXmYeZ` family) reserve the top exponent for inf/NaN;
/// the OCP MX element formats (`f4e2m1`, `f6e2m3`, `f6e3m2`, the `fNeXmY` family)
/// are all-finite. Both are expressed by `max_normal` alone.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatFormat {
    pub exp_bits: u8,
    pub man_bits: u8,
    pub bias: i32,
    pub max_normal: f32,
}

// `max_normal` is never NaN, so equality is reflexive.
impl Eq for FloatFormat {}

impl core::hash::Hash for FloatFormat {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        (
            self.exp_bits,
            self.man_bits,
            self.bias,
            self.max_normal.to_bits(),
        )
            .hash(state);
    }
}

impl FloatFormat {
    /// IEEE binary16.
    pub const F16: Self = Self::ieee(5, 10, 65504.0);
    /// bfloat16.
    pub const BF16: Self = Self::ieee(8, 7, 3.389_531_4e38);
    /// TensorFloat-32 (e8m10).
    pub const TF32: Self = Self::ieee(8, 10, 3.401_162_1e38);

    /// An IEEE-style format (bias `2^(exp_bits-1) - 1`) with an explicit maximum.
    pub const fn ieee(exp_bits: u8, man_bits: u8, max_normal: f32) -> Self {
        Self {
            exp_bits,
            man_bits,
            bias: (1i32 << (exp_bits - 1)) - 1,
            max_normal,
        }
    }

    /// The element grid of a [`ScaledFormat`] (FP8 / FP6 / FP4 / custom minifloat).
    pub fn from_scaled(fmt: ScaledFormat) -> Self {
        let (e, m, bias) = fmt.fields();
        Self {
            exp_bits: e as u8,
            man_bits: m as u8,
            bias,
            max_normal: fmt.max_finite(),
        }
    }

    /// Total bits: sign + exponent + mantissa.
    pub const fn bits(self) -> u32 {
        1 + self.exp_bits as u32 + self.man_bits as u32
    }

    /// Smallest normal exponent.
    pub const fn emin(self) -> i32 {
        1 - self.bias
    }

    /// Exponent of `max_normal` (`floor(log2(max_normal))`).
    pub fn emax(self) -> i32 {
        f32_exponent(self.max_normal)
    }

    /// Round `x` to this format. See the module docs for the exact semantics.
    pub fn round(self, x: f32) -> f32 {
        let bits = x.to_bits();
        let abs_bits = bits & 0x7fff_ffff;
        let sign = bits & 0x8000_0000;
        // ±0 and NaN pass through; f32 subnormals flush.
        if abs_bits == 0 || abs_bits > 0x7f80_0000 {
            return x;
        }
        if abs_bits < 0x0080_0000 {
            return f32::from_bits(sign);
        }
        let ax = f32::from_bits(abs_bits);
        if ax >= self.max_normal {
            return f32::from_bits(sign | self.max_normal.to_bits());
        }
        // Below the smallest normal the step stays at the subnormal step.
        let ee = f32_exponent(ax).max(self.emin());
        let shift = self.man_bits as i32 - ee;
        // `ax * 2^shift` has at most `man_bits + 1` integer bits, so every step
        // here is exact in f64 and the result is representable in f32.
        let v = (ax as f64) * pow2_f64(shift);
        let q = (v.round_ties_even() * pow2_f64(-shift)) as f32;
        f32::from_bits(sign | flush_subnormal(q.min(self.max_normal)).to_bits())
    }

    /// Round every element of `xs` in place.
    pub fn round_slice(self, xs: &mut [f32]) {
        for x in xs {
            *x = self.round(*x);
        }
    }

    /// Shared power-of-two scale for a block whose largest magnitude is `amax`
    /// (the OCP MX rule): `2^(floor(log2(amax)) - emax)`, so the block's largest
    /// element lands in the format's top binade. A zero or non-finite `amax`
    /// gives `1.0`. The exponent is clamped to the normal `f32` range.
    pub fn pow2_scale(self, amax: f32) -> f32 {
        if amax == 0.0 || !amax.is_finite() {
            return 1.0;
        }
        let e = (f32_exponent(amax) - self.emax()).clamp(-126, 127);
        f32::from_bits(((e + 127) as u32) << 23)
    }

    /// Power-of-two scale of one block: the OCP MX scale
    /// ([`Self::pow2_scale`]) or, when that clips the block's largest
    /// elements to `max_normal`, twice it if that rounds the block with a
    /// smaller squared error. MX hardware decodes either (an E8M0 scale is any
    /// power of two). On an ETTh1 forecast benchmark this took Timestral's
    /// f8 error over f32 from 8.6% to 3.3% and f4's from 81% to 60%.
    ///
    /// The comparison sums `d * d` in f32, in index order, with no fused
    /// multiply-add, so every backend makes the same choice; ties keep the MX
    /// scale. Without clipping the twice-coarser grid is a subset of the finer
    /// one, so it cannot win and is not tried.
    pub fn block_scale(self, block: &[f32]) -> f32 {
        let amax = block.iter().fold(0f32, |a, &x| a.max(x.abs()));
        let s = self.pow2_scale(amax);
        // `amax / s` is exact (power-of-two `s`); the doubled scale must stay
        // a normal f32.
        if !amax.is_finite() || amax / s <= self.max_normal || s >= f32::from_bits(254 << 23) {
            return s;
        }
        let sq_err = |s: f32| {
            let mut e = 0f32;
            for &x in block {
                let d = self.round_scaled(x, s) - x;
                let d2 = d * d;
                e += d2;
            }
            e
        };
        if sq_err(2.0 * s) < sq_err(s) {
            2.0 * s
        } else {
            s
        }
    }

    /// Round with a power-of-two scale: `round(x / s) * s`.
    #[inline]
    pub fn round_scaled(self, x: f32, scale: f32) -> f32 {
        let y = self.round(x / scale) * scale;
        if y.is_nan() {
            y
        } else {
            y.signum() * flush_subnormal(y.abs())
        }
    }

    /// Round `xs` with `scaling`, treating it as rows of length `last_dim`
    /// (blocks never cross a row). This is the reference for every backend.
    pub fn round_slice_scaled(self, xs: &mut [f32], scaling: FormatScaling, last_dim: usize) {
        match scaling {
            FormatScaling::None => self.round_slice(xs),
            FormatScaling::PerTensor => {
                let amax = xs.iter().fold(0f32, |a, &x| a.max(x.abs()));
                let s = self.pow2_scale(amax);
                for x in xs {
                    *x = self.round_scaled(*x, s);
                }
            }
            FormatScaling::Block(n) => {
                let n = n as usize;
                let last_dim = last_dim.max(1);
                for row in xs.chunks_mut(last_dim) {
                    for block in row.chunks_mut(n) {
                        let s = self.block_scale(block);
                        for x in block {
                            *x = self.round_scaled(*x, s);
                        }
                    }
                }
            }
            FormatScaling::FineBlock(n) => {
                let n = n as usize;
                let last_dim = last_dim.max(1);
                for row in xs.chunks_mut(last_dim) {
                    for block in row.chunks_mut(n) {
                        let amax = block.iter().fold(0f32, |a, &x| a.max(x.abs()));
                        let s = self.fine_scale(amax);
                        let r = fine_reciprocal(s);
                        for x in block {
                            *x = self.round_fine(*x, s, r);
                        }
                    }
                }
            }
        }
    }

    /// [`FormatScaling::FineBlock`] scale for a block whose largest magnitude
    /// is `amax`: the smallest `(1 + m/8) · 2^k` with `s · max_normal ≥ amax`,
    /// so no element clips. `1.0` for a zero or non-finite `amax`. Every
    /// product here is exact (a 4-bit significand times the format's) and every
    /// test a comparison, so all backends find the same scale.
    pub fn fine_scale(self, amax: f32) -> f32 {
        if amax == 0.0 || !amax.is_finite() {
            return 1.0;
        }
        let k0 = (f32_exponent(amax) - self.emax() - 1).clamp(-126, 125);
        for k in k0..=k0 + 2 {
            let p = f32::from_bits(((k + 127) as u32) << 23);
            for m in 0..8 {
                let s = (1.0 + m as f32 / 8.0) * p;
                if s * self.max_normal >= amax {
                    return s;
                }
            }
        }
        f32::from_bits(0x7f70_0000)
    }

    /// `round(x · r) · s`, `r` = [`fine_reciprocal`]`(s)`: [`FormatScaling::FineBlock`]
    /// rounding. `round(x·r)` has at most `man_bits + 1` significant bits and
    /// `s` four, so the product is exact for formats up to 19 mantissa bits.
    #[inline]
    pub fn round_fine(self, x: f32, s: f32, r: f32) -> f32 {
        let y = self.round(x * r) * s;
        if y.is_nan() {
            y
        } else {
            y.signum() * flush_subnormal(y.abs())
        }
    }

    /// Parse a format name. Accepts:
    ///
    /// - `f16` / `half`, `bf16`
    /// - `f8` / `fp8` / `e4m3` (OCP E4M3), `bf8` / `e5m2`, `f6` (OCP E2M3),
    ///   `f4` / `fp4` / `e2m1` / `nvf4`
    /// - `fNeXmY` (e.g. `f4e3m0`): an all-finite minifloat, as [`ScaledFormat`]
    ///   parses it (≤ 8 bits)
    /// - `fXmYeZ` (e.g. `f12m7e4`): an IEEE-style format with the top exponent
    ///   reserved, up to `e8m23`
    pub fn parse(spec: &str) -> Option<Self> {
        let s = spec.trim().to_ascii_lowercase();
        let named = match s.as_str() {
            "f16" | "half" | "fp16" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
            // TensorFloat-32: f32's 8-bit exponent, 10 mantissa bits — what
            // NVIDIA tensor cores (and ROCm XF32) round matmul operands to.
            "tf32" => Some(Self::TF32),
            "f8" | "fp8" | "e4m3" | "f8e4m3" => Some(Self::from_scaled(ScaledFormat::F8E4M3)),
            "bf8" | "e5m2" | "f8e5m2" => Some(Self::ieee(5, 2, 57344.0)),
            "f6" | "fp6" | "e2m3" => Some(Self::from_scaled(ScaledFormat::F6E2M3)),
            "e3m2" => Some(Self::from_scaled(ScaledFormat::F6E3M2)),
            "f4" | "fp4" | "e2m1" | "nvf4" => Some(Self::from_scaled(ScaledFormat::F4E2M1)),
            _ => None,
        };
        if named.is_some() {
            return named;
        }
        if let Ok(fmt) = s.parse::<ScaledFormat>() {
            return Some(Self::from_scaled(fmt));
        }
        // Generic IEEE-style `f{total}m{man}e{exp}`.
        let rest = s.strip_prefix('f')?;
        let (total, rest) = rest.split_once('m')?;
        let (man, exp) = rest.split_once('e')?;
        let (total, man, exp): (u32, u32, u32) =
            (total.parse().ok()?, man.parse().ok()?, exp.parse().ok()?);
        if exp == 0 || exp > 8 || man > 23 || 1 + man + exp != total {
            return None;
        }
        let bias = (1i32 << (exp - 1)) - 1;
        let max = (2.0 - pow2_f64(-(man as i32))) * pow2_f64(bias);
        Some(Self::ieee(
            exp as u8,
            man as u8,
            max.min(f32::MAX as f64) as f32,
        ))
    }
}

impl fmt::Display for FloatFormat {
    /// The first spelling that [`FloatFormat::parse`]s back to `self`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (n, e, m) = (self.bits(), self.exp_bits, self.man_bits);
        let candidates = [
            "f16".to_string(),
            "bf16".to_string(),
            "tf32".to_string(),
            format!("f{n}e{e}m{m}"),
            format!("f{n}e{e}m{m}fnuz"),
            format!("f{n}m{m}e{e}"),
        ];
        match candidates.iter().find(|c| Self::parse(c) == Some(*self)) {
            Some(name) => f.write_str(name),
            // Not reachable through `parse`; spell out every field.
            None => write!(
                f,
                "f{n}e{e}m{m}(bias {}, max {})",
                self.bias, self.max_normal
            ),
        }
    }
}

/// How values share a scale before rounding to a [`FloatFormat`].
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatScaling {
    /// Round the raw values (the norm for `f16` / `bf16`).
    None,
    /// One power-of-two scale for the whole tensor.
    PerTensor,
    /// One power-of-two scale per run of `n` consecutive elements along the
    /// last axis (OCP MX uses 32). A shorter final block is allowed.
    Block(u32),
    /// One scale per run of `n` elements with a 3-bit mantissa, as NVFP4's
    /// E4M3 block scales (NVFP4 uses 16): the smallest such value at least
    /// `amax / max_normal`, so the block's largest element lands at the top of
    /// the format instead of anywhere in a 2x range. See
    /// [`FloatFormat::fine_scale`]. The exponent has f32's range (NVFP4 gets
    /// that from a second, per-tensor scale).
    FineBlock(u32),
}

impl FormatScaling {
    /// The default for a format: none above 8 bits, NVFP4-style fine-scaled
    /// 16-element blocks at or below. Fine scales beat OCP MX's power-of-two
    /// ones on every format measured (Timestral f4 on ETTh1: +61% → +20% WQL
    /// over f32; Qwen3-0.6B f6: +11% → +7.5% perplexity); `@b32` selects MX.
    pub fn default_for(fmt: FloatFormat) -> Self {
        if fmt.bits() > 8 {
            Self::None
        } else {
            Self::FineBlock(16)
        }
    }
}

/// How numbers are represented when a graph runs.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NumberFormat {
    /// IEEE binary32 — the default; nothing is rewritten.
    #[default]
    F32,
    /// Binary64: native where the backend has it, double-single elsewhere.
    F64,
    /// Double-single: each value is an unevaluated sum of two `f32`
    /// (~48 significant bits), on every backend including CPU.
    F64x2,
    /// Values rounded to `format` (weights once, every intermediate after the op
    /// that produces it); arithmetic inside an op accumulates in `f32`.
    Float {
        format: FloatFormat,
        scaling: FormatScaling,
    },
    /// Weights in one format and intermediates in another, written
    /// `weights/activations` (`f4/f8`); `None` leaves that side in f32
    /// (`f4/f32`: 4-bit weights, f32 compute). Low-bit deployments usually
    /// quantize weights harder than activations: with fine scales, f4 weights
    /// alone cost Qwen3-0.6B 13% perplexity where f4 everywhere costs 430%.
    Mixed {
        weights: Option<(FloatFormat, FormatScaling)>,
        activations: Option<(FloatFormat, FormatScaling)>,
    },
}

impl NumberFormat {
    /// `format` with its default scaling.
    pub fn float(format: FloatFormat) -> Self {
        Self::Float {
            format,
            scaling: FormatScaling::default_for(format),
        }
    }

    /// The format weights are rounded to, if any.
    pub fn weights(&self) -> Option<(FloatFormat, FormatScaling)> {
        match *self {
            Self::Float { format, scaling } => Some((format, scaling)),
            Self::Mixed { weights, .. } => weights,
            _ => None,
        }
    }

    /// The format intermediate values are rounded to, if any.
    pub fn activations(&self) -> Option<(FloatFormat, FormatScaling)> {
        match *self {
            Self::Float { format, scaling } => Some((format, scaling)),
            Self::Mixed { activations, .. } => activations,
            _ => None,
        }
    }
}

/// One side of a [`NumberFormat::Mixed`], or a plain float format.
fn parse_float_spec(s: &str) -> Result<Option<(FloatFormat, FormatScaling)>, String> {
    match s.parse::<NumberFormat>()? {
        NumberFormat::F32 => Ok(None),
        NumberFormat::Float { format, scaling } => Ok(Some((format, scaling))),
        other => Err(format!(
            "{other} cannot be one side of weights/activations; use f32 or a float format"
        )),
    }
}

fn write_float_spec(
    f: &mut fmt::Formatter<'_>,
    spec: Option<(FloatFormat, FormatScaling)>,
) -> fmt::Result {
    match spec {
        None => write!(f, "f32"),
        Some((format, scaling)) => write!(f, "{}", NumberFormat::Float { format, scaling }),
    }
}

impl FromStr for NumberFormat {
    type Err = String;

    /// Parse `f32`, `f64`, `f64x2` (alias `df64`), or a [`FloatFormat`] name
    /// with an optional scaling suffix: `@none`, `@tensor`, `@b<N>` (e.g.
    /// `f4@b16`). Without a suffix the format's default scaling applies.
    /// `weights/activations` (e.g. `f4/f8`, `f4/f32`) gives each its own
    /// format ([`NumberFormat::Mixed`]).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_ascii_lowercase();
        if let Some((w, a)) = s.split_once('/') {
            let (weights, activations) = (parse_float_spec(w)?, parse_float_spec(a)?);
            return Ok(match (weights, activations) {
                (None, None) => Self::F32,
                (Some((format, scaling)), Some(a)) if a == (format, scaling) => {
                    Self::Float { format, scaling }
                }
                _ => Self::Mixed {
                    weights,
                    activations,
                },
            });
        }
        match s.as_str() {
            "f32" | "fp32" | "float" => return Ok(Self::F32),
            "f64" | "fp64" | "double" => return Ok(Self::F64),
            "f64x2" | "df64" | "double-single" => return Ok(Self::F64x2),
            _ => {}
        }
        let (name, suffix) = match s.split_once('@') {
            Some((n, sfx)) => (n, Some(sfx)),
            None => (s.as_str(), None),
        };
        let format = FloatFormat::parse(name).ok_or_else(|| {
            format!(
                "unknown number format {s:?}: expected f32, f64, f64x2, f16, bf16, f8, f6, f4, \
                 fNeXmY (e.g. f4e3m0) or fXmYeZ (e.g. f12m7e4), optionally with @none, @tensor, @b<N> or @fb<N>"
            )
        })?;
        let scaling = match suffix {
            None => FormatScaling::default_for(format),
            Some("none") => FormatScaling::None,
            Some("tensor") => FormatScaling::PerTensor,
            Some(b) if b.starts_with("fb") => {
                let n: u32 = b[2..]
                    .parse()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| format!("bad scaling suffix @{b}: expected fb<N>"))?;
                FormatScaling::FineBlock(n)
            }
            Some(b) => {
                let n: u32 = b
                    .strip_prefix('b')
                    .and_then(|n| n.parse().ok())
                    .filter(|&n| n > 0)
                    .ok_or_else(|| {
                        format!("bad scaling suffix @{b}: expected none, tensor or b<N>")
                    })?;
                FormatScaling::Block(n)
            }
        };
        Ok(Self::Float { format, scaling })
    }
}

impl fmt::Display for NumberFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"),
            Self::F64x2 => write!(f, "f64x2"),
            Self::Mixed {
                weights,
                activations,
            } => {
                write_float_spec(f, *weights)?;
                write!(f, "/")?;
                write_float_spec(f, *activations)
            }
            Self::Float { format, scaling } => {
                write!(f, "{format}")?;
                if *scaling != FormatScaling::default_for(*format) {
                    match scaling {
                        FormatScaling::None => write!(f, "@none")?,
                        FormatScaling::PerTensor => write!(f, "@tensor")?,
                        FormatScaling::Block(n) => write!(f, "@b{n}")?,
                        FormatScaling::FineBlock(n) => write!(f, "@fb{n}")?,
                    }
                }
                Ok(())
            }
        }
    }
}

/// Reciprocals of the eight significands `1 + m/8` a fine scale can have,
/// each rounded once to f32. Kernels carry the same table.
pub const FINE_RECIPROCALS: [f32; 8] = [
    1.0, 0.8888889, 0.8, 0.72727275, 0.6666667, 0.61538464, 0.5714286, 0.53333336,
];

/// `1 / s` for a [`FloatFormat::fine_scale`] `s`, from [`FINE_RECIPROCALS`]
/// and the exponent: exact up to the table's single rounding, and the same on
/// every backend. Flushes to `0` when `1/s` is below the normal range.
pub fn fine_reciprocal(s: f32) -> f32 {
    let bits = s.to_bits();
    let m = ((bits >> 20) & 7) as usize;
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    let ne = -e;
    if ne < -126 {
        return 0.0;
    }
    let p = f32::from_bits(((ne.min(127) + 127) as u32) << 23);
    FINE_RECIPROCALS[m] * p
}

/// `0` for a non-negative `x` below `f32`'s smallest normal, else `x`.
#[inline]
fn flush_subnormal(x: f32) -> f32 {
    if x.to_bits() < 0x0080_0000 { 0.0 } else { x }
}

/// Unbiased exponent of a non-negative finite `f32` (`-127` for subnormals).
#[inline]
fn f32_exponent(x: f32) -> i32 {
    ((x.to_bits() >> 23) & 0xff) as i32 - 127
}

#[inline]
fn pow2_f64(n: i32) -> f64 {
    f64::from_bits(((n + 1023) as u64) << 52)
}

#[cfg(test)]
mod tests {
    #[test]
    fn tf32_is_e8m10_and_round_trips() {
        let t = FloatFormat::parse("tf32").unwrap();
        assert_eq!((t.exp_bits, t.man_bits, t.bias), (8, 10, 127));
        assert_eq!(
            FloatFormat::parse("f19m10e8"),
            Some(t),
            "generic spelling agrees"
        );
        assert_eq!(t.to_string(), "tf32");
        // 1 + 2⁻¹² is below tf32's 10-bit mantissa step; 1 + 2⁻¹⁰ is not.
        assert_eq!(t.round(1.0 + 2f32.powi(-12)), 1.0);
        assert_eq!(t.round(1.0 + 2f32.powi(-10)), 1.0 + 2f32.powi(-10));
    }

    use super::*;

    #[test]
    fn mixed_formats_parse_and_print() {
        let m: NumberFormat = "f4/f8".parse().unwrap();
        assert_eq!(m.weights().unwrap().0, FloatFormat::parse("f4").unwrap());
        assert_eq!(
            m.activations().unwrap().0,
            FloatFormat::parse("f8").unwrap()
        );
        assert_eq!(m.to_string().parse::<NumberFormat>(), Ok(m));
        let w: NumberFormat = "f4@b32/f32".parse().unwrap();
        assert_eq!(w.activations(), None);
        assert_eq!(w.weights().unwrap().1, FormatScaling::Block(32));
        assert_eq!(w.to_string().parse::<NumberFormat>(), Ok(w));
        assert_eq!("f8/f8".parse::<NumberFormat>(), "f8".parse());
        assert_eq!("f32/f32".parse::<NumberFormat>(), Ok(NumberFormat::F32));
        assert!("f64/f8".parse::<NumberFormat>().is_err());
    }

    #[test]
    fn fine_scales_are_exact_and_never_clip() {
        for (m, &r) in FINE_RECIPROCALS.iter().enumerate() {
            assert_eq!(r, (1.0 / (1.0 + m as f64 / 8.0)) as f32, "table entry {m}");
        }
        let f4 = FloatFormat::parse("f4").unwrap();
        // amax 7.9: 7.9/6 = 1.3167 rounds up to 1.375, so 7.9 maps to 5.745 → 6.
        let s = f4.fine_scale(7.9);
        assert_eq!(s, 1.375);
        assert_eq!(f4.round_fine(7.9, s, fine_reciprocal(s)), 6.0 * 1.375);
        for &amax in &[1e-30f32, 0.003, 1.0, 5.99, 6.0, 7.9, 1234.5, 3e37] {
            let s = f4.fine_scale(amax);
            assert!(s * 6.0 >= amax * 0.999_999, "{amax}: scale {s} clips");
            assert_eq!(
                s.to_bits() & 0x000f_ffff,
                0,
                "{amax}: more than 3 mantissa bits"
            );
            assert!((fine_reciprocal(s) as f64 * s as f64 - 1.0).abs() < 1e-7);
        }
        let n: NumberFormat = "f4@fb16".parse().unwrap();
        assert!(matches!(
            n,
            NumberFormat::Float {
                scaling: FormatScaling::FineBlock(16),
                ..
            }
        ));
        assert_eq!(n.to_string().parse::<NumberFormat>(), Ok(n));
    }

    #[test]
    fn block_scale_avoids_clipping_only_when_it_pays() {
        let f4 = FloatFormat::parse("f4").unwrap(); // e2m1: max 6, MX scale puts amax in [4, 8)
        // No clipping: the MX scale.
        assert_eq!(f4.block_scale(&[5.0, 1.0, 0.5]), f4.pow2_scale(5.0));
        // 7.9 clips to 6 under the MX scale (error 1.9); at twice the scale it
        // rounds to 8 and 1.0 / 0.5 stay exact on the coarser grid.
        let s = f4.pow2_scale(7.9);
        assert_eq!(f4.block_scale(&[7.9, 1.0, 0.5]), 2.0 * s);
        // Clipping 6.5 to 6 costs as much as the coarser grid costs the 0.75s;
        // a tie keeps the MX scale.
        let fine = [6.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75];
        assert_eq!(f4.block_scale(&fine), f4.pow2_scale(6.5));
        // Non-finite maxima keep the MX convention.
        assert_eq!(f4.block_scale(&[f32::INFINITY, 1.0]), 1.0);
    }

    #[test]
    fn named_formats_parse() {
        assert_eq!(FloatFormat::parse("f16"), Some(FloatFormat::F16));
        assert_eq!(FloatFormat::parse("f4").unwrap().max_normal, 6.0);
        assert_eq!(FloatFormat::parse("f6").unwrap().max_normal, 7.5);
        assert_eq!(FloatFormat::parse("f8").unwrap().max_normal, 448.0);
        // All-finite fNeXmY vs IEEE-style fXmYeZ keep their existing meanings.
        assert_eq!(FloatFormat::parse("f4e3m0").unwrap().max_normal, 16.0);
        assert_eq!(FloatFormat::parse("f4m0e3").unwrap().max_normal, 8.0);
        assert_eq!(FloatFormat::parse("f12m7e4").unwrap().man_bits, 7);
        assert!(FloatFormat::parse("f5m1e2").is_none(), "bits must add up");
    }

    #[test]
    fn number_format_round_trips_through_text() {
        for s in [
            "f32",
            "f64",
            "f64x2",
            "f16",
            "bf16",
            "f4e2m1",
            "f4e2m1@b16",
            "f16@tensor",
            "f8e4m3@none",
        ] {
            let f: NumberFormat = s.parse().unwrap();
            assert_eq!(f.to_string(), s, "{s}");
        }
        assert_eq!("f4".parse::<NumberFormat>().unwrap().to_string(), "f4e2m1");
        assert!("f4@b0".parse::<NumberFormat>().is_err());
    }

    #[test]
    fn f16_rounding_matches_hardware_half() {
        // Exhaustive over a stride of f32 bit patterns, vs the `half` semantics
        // reproduced by hand: f32 -> nearest-even f16 -> f32.
        let f16 = FloatFormat::F16;
        let mut checked = 0;
        for bits in (0u32..=0x7f7f_ffff).step_by(997) {
            let x = f32::from_bits(bits);
            let want = f16_round_reference(x);
            assert_eq!(f16.round(x).to_bits(), want.to_bits(), "x = {x:e}");
            checked += 1;
        }
        assert!(checked > 2_000_000);
    }

    /// f32 → f16 → f32 via integer arithmetic, saturating instead of producing inf.
    fn f16_round_reference(x: f32) -> f32 {
        let ax = x.abs();
        if ax >= 65504.0 {
            return 65504f32.copysign(x);
        }
        // Scale into the f16 integer grid of the right binade, round half-even.
        let e = (ax.log2().floor() as i32).max(-14);
        let ulp = (2f64).powi(e - 10);
        let q = ((ax as f64) / ulp).round_ties_even() * ulp;
        // log2 can land on the next binade just below a power of two; the
        // f64 step above is then twice too coarse, but such inputs round to
        // the power of two either way at 10 mantissa bits.
        (q as f32).min(65504.0).copysign(x)
    }

    #[test]
    fn ties_go_to_even_and_subnormals_survive() {
        let f4 = FloatFormat::parse("f4").unwrap(); // e2m1: 0, .5, 1, 1.5, 2, 3, 4, 6
        assert_eq!(f4.round(2.5), 2.0);
        assert_eq!(f4.round(3.5), 4.0);
        assert_eq!(f4.round(0.25), 0.0);
        assert_eq!(f4.round(0.26), 0.5);
        assert_eq!(f4.round(-5.0), -4.0);
        assert_eq!(f4.round(f32::NEG_INFINITY), -6.0);
        assert!(f4.round(f32::NAN).is_nan());
        assert_eq!(f4.round(-0.0).to_bits(), (-0.0f32).to_bits());
        let bf16 = FloatFormat::BF16;
        let tiny = f32::from_bits(1); // smallest f32 subnormal
        assert_eq!(bf16.round(tiny), 0.0, "f32 subnormals flush");
        assert_eq!(bf16.round(-tiny).to_bits(), (-0.0f32).to_bits());
        // f16 subnormals are f32 normals, so they survive.
        assert_eq!(FloatFormat::F16.round(6.0e-8), 5.9604645e-8);
    }

    #[test]
    fn matches_scaled_format_quantize() {
        for fmt in ScaledFormat::NAMED
            .into_iter()
            .chain([ScaledFormat::custom(3, 0)])
        {
            let ff = FloatFormat::from_scaled(fmt);
            for i in -4000..4000 {
                let x = i as f32 * 0.0137;
                assert_eq!(ff.round(x), fmt.quantize(x), "{fmt} x={x}");
            }
        }
    }

    #[test]
    fn mx_block_scaling() {
        let f4 = FloatFormat::parse("f4").unwrap();
        // amax 0.3: floor(log2 0.3) = -2, emax(e2m1) = 2 → scale 2^-4.
        assert_eq!(f4.pow2_scale(0.3), 0.0625);
        let mut xs = [0.3f32, 0.01, -0.2, 0.0];
        f4.round_slice_scaled(&mut xs, FormatScaling::Block(32), 4);
        assert_eq!(xs, [0.25, 0.0, -0.1875, 0.0]);
    }
}
