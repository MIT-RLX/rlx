// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Element data types for tensors.

/// Scalar element type. Matches hardware-supported types.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U32,
    Bool,
    /// Complex with f32 real and f32 imaginary components, stored
    /// interleaved as `[re, im, re, im, ...]`. 8 bytes per complex
    /// element. Element-wise ops (Add/Sub/Mul/Conj) follow the
    /// standard complex algebra. Reverse-mode AD on this dtype is
    /// **not yet wired** — Wirtinger conventions (∂/∂z vs ∂/∂z̄)
    /// belong to a separate pass that knows to emit conjugate-aware
    /// VJPs. The forward path is sufficient for AC analysis and
    /// FFT-based workflows that don't need to differentiate through
    /// complex math (and in fact, FFT today already encodes complex
    /// as 2N-real-block; this dtype is the natural successor).
    C64,
    /// Complex with f64 real and f64 imaginary components, stored
    /// interleaved as `[re, im, re, im, ...]`. 16 bytes per complex
    /// element — the double-precision sibling of [`Self::C64`]. This
    /// dtype covers the dtype plumbing (size/flags/promotion/Display)
    /// and the CAST matrix only: complex *arithmetic* (FFT, cgemm,
    /// complex binary/activation, Wirtinger AD) remains C64-only, and
    /// those sites reject C128 rather than compute on it.
    C128,
}

impl DType {
    /// Size in bytes of one element.
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::C64 => 8,
            Self::C128 => 16,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::F64)
    }

    /// True for complex-valued dtypes. Complex elementwise ops follow
    /// standard complex algebra, distinct from the float real/imag
    /// components (e.g. complex multiply ≠ paired-real multiply).
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::C64 | Self::C128)
    }

    pub const fn is_int(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::U8 | Self::U32
        )
    }

    /// Promotion rank — higher means "wider, more expressive". The
    /// promoted dtype of a binary op is `max(rank(lhs), rank(rhs))`.
    /// Borrowed from MAX's `dtype_promotion.py` pattern: one module
    /// owns the table; ops query it instead of re-implementing ad-hoc
    /// rules.
    ///
    /// Ranks (low → high):
    ///   0 = Bool, 1 = U8/I8, 2 = I16/BF16, 3 = F16, 4 = U32/I32,
    ///   5 = I64, 6 = F32, 7 = F64, 8 = C64, 9 = C128.
    /// Floats outrank ints of the same width (matches PyTorch /
    /// NumPy). BF16 promotes to F32 against F16 since BF16 has
    /// wider range but F16 has more mantissa.
    pub const fn promotion_rank(self) -> u8 {
        match self {
            Self::Bool => 0,
            Self::U8 | Self::I8 => 1,
            Self::I16 | Self::BF16 => 2,
            Self::F16 => 3,
            Self::U32 | Self::I32 => 4,
            Self::I64 => 5,
            Self::F32 => 6,
            Self::F64 => 7,
            Self::C64 => 8,
            // C128 (complex f64) outranks C64 (complex f32): a binary
            // op mixing the two promotes to C128 (wider components).
            Self::C128 => 9,
        }
    }

    /// Result dtype for a binary op between `self` and `other`.
    /// Mixed int+float → float at least as wide as either input.
    /// `f16 + bf16 → f32` (no clean lossless target).
    pub fn promote(self, other: Self) -> Self {
        if self == other {
            return self;
        }
        // Special case: f16 + bf16 → f32 (their domains are too
        // different to lose precision in either direction).
        if matches!(
            (self, other),
            (Self::F16, Self::BF16) | (Self::BF16, Self::F16)
        ) {
            return Self::F32;
        }
        // Mixed int+float: bump to the smallest float that covers both.
        let promote_int_to_float = |int: Self, float: Self| -> Self {
            match (int, float) {
                (_, Self::F64) => Self::F64,
                (Self::I64, _) => Self::F64, // 64-bit int needs F64
                (_, Self::F32) => Self::F32,
                (_, Self::F16) | (_, Self::BF16) => Self::F32, // safe upcast
                _ => float,
            }
        };
        match (
            self.is_int(),
            other.is_int(),
            self.is_float(),
            other.is_float(),
        ) {
            (true, false, false, true) => promote_int_to_float(self, other),
            (false, true, true, false) => promote_int_to_float(other, self),
            _ => {
                if self.promotion_rank() >= other.promotion_rank() {
                    self
                } else {
                    other
                }
            }
        }
    }
}

fn integral_scalar(value: f64, name: &str) -> Result<i64, String> {
    if !value.is_finite() {
        return Err(format!(
            "constant value {value} is not finite for dtype {name}"
        ));
    }
    if value.fract() != 0.0 {
        return Err(format!(
            "constant value {value} must be integral for dtype {name}"
        ));
    }
    Ok(value as i64)
}

/// Encode a scalar as little-endian bytes for [`crate::op::Op::Constant`].
pub fn scalar_constant_bytes(value: f64, dtype: DType) -> Result<Vec<u8>, String> {
    let out_of_range =
        |name: &str| format!("constant value {value} is out of range for dtype {name}");
    match dtype {
        DType::F32 => Ok((value as f32).to_le_bytes().to_vec()),
        DType::F64 => Ok(value.to_le_bytes().to_vec()),
        DType::I8 => {
            let v = integral_scalar(value, "i8")?;
            if !(i8::MIN as i64..=i8::MAX as i64).contains(&v) {
                return Err(out_of_range("i8"));
            }
            Ok((v as i8).to_le_bytes().to_vec())
        }
        DType::I16 => {
            let v = integral_scalar(value, "i16")?;
            if !(i16::MIN as i64..=i16::MAX as i64).contains(&v) {
                return Err(out_of_range("i16"));
            }
            Ok((v as i16).to_le_bytes().to_vec())
        }
        DType::I32 => {
            let v = integral_scalar(value, "i32")?;
            if !(i32::MIN as i64..=i32::MAX as i64).contains(&v) {
                return Err(out_of_range("i32"));
            }
            Ok((v as i32).to_le_bytes().to_vec())
        }
        DType::I64 => {
            if !value.is_finite() {
                return Err(format!(
                    "constant value {value} is not finite for dtype i64"
                ));
            }
            if value.fract() != 0.0 {
                return Err(format!(
                    "constant value {value} must be integral for dtype i64"
                ));
            }
            // `i64::MAX as f64` rounds up to 2^63; use open bounds at ±2^63.
            if value >= 9.223372036854776e18 || value < -9.223372036854776e18 {
                return Err(out_of_range("i64"));
            }
            Ok((value as i64).to_le_bytes().to_vec())
        }
        DType::U8 => {
            let v = integral_scalar(value, "u8")?;
            if !(0..=u8::MAX as i64).contains(&v) {
                return Err(out_of_range("u8"));
            }
            Ok((v as u8).to_le_bytes().to_vec())
        }
        DType::U32 => {
            let v = integral_scalar(value, "u32")?;
            if v < 0 || v > u32::MAX as i64 {
                return Err(out_of_range("u32"));
            }
            Ok((v as u32).to_le_bytes().to_vec())
        }
        DType::Bool => Ok(vec![u8::from(value != 0.0)]),
        DType::F16 | DType::BF16 | DType::C64 | DType::C128 => Err(format!(
            "scalar literal dtype '{dtype:?}' is built via f32 constant + cast"
        )),
    }
}

/// Per-element semantics that don't fit into a flat `DType` enum
/// (plan #40). Mirrors MAX's `layout/element.mojo` `Element` type:
/// `DType` says "f8", but two FP8 variants exist (e4m3 and e5m2)
/// with different range/precision tradeoffs. Saturation policy
/// (clamp on overflow vs. wrap) is similarly orthogonal.
///
/// Today most ops only care about `dtype`; downstream quantization
/// kernels read `subtype` and `saturating` to pick the right
/// dequant. Building this in early prevents the "every op grew its
/// own ad-hoc fp8 flag" mess MAX hit in v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Element {
    pub dtype: DType,
    /// Subtype within `dtype` for FP8 variants etc. `Standard`
    /// for everything else.
    pub subtype: ElementSubtype,
    /// Whether arithmetic saturates on overflow (true for the
    /// quantized accumulator paths) or wraps (default).
    pub saturating: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementSubtype {
    Standard,
    /// FP8 e4m3 (4 exp bits, 3 mantissa) — lower range, more
    /// precision; matches NVIDIA's "FNUZ" Hopper format.
    Fp8E4m3,
    /// FP8 e5m2 (5 exp bits, 2 mantissa) — wider range, less
    /// precision; closer to bf16 in dynamic range.
    Fp8E5m2,
}

impl Element {
    pub const fn new(dtype: DType) -> Self {
        Self {
            dtype,
            subtype: ElementSubtype::Standard,
            saturating: false,
        }
    }
    pub const fn fp8_e4m3() -> Self {
        Self {
            dtype: DType::U8,
            subtype: ElementSubtype::Fp8E4m3,
            saturating: true,
        }
    }
    pub const fn fp8_e5m2() -> Self {
        Self {
            dtype: DType::U8,
            subtype: ElementSubtype::Fp8E5m2,
            saturating: true,
        }
    }
    pub const fn saturating(self) -> Self {
        Self {
            saturating: true,
            ..self
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
            Self::F64 => write!(f, "f64"),
            Self::I8 => write!(f, "i8"),
            Self::I16 => write!(f, "i16"),
            Self::I32 => write!(f, "i32"),
            Self::I64 => write!(f, "i64"),
            Self::U8 => write!(f, "u8"),
            Self::U32 => write!(f, "u32"),
            Self::Bool => write!(f, "bool"),
            Self::C64 => write!(f, "c64"),
            Self::C128 => write!(f, "c128"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_constructors() {
        let f = Element::new(DType::F32);
        assert_eq!(f.dtype, DType::F32);
        assert_eq!(f.subtype, ElementSubtype::Standard);
        assert!(!f.saturating);

        let e4 = Element::fp8_e4m3();
        assert_eq!(e4.subtype, ElementSubtype::Fp8E4m3);
        assert!(e4.saturating);
        assert_eq!(e4.dtype, DType::U8);

        let s = Element::new(DType::I32).saturating();
        assert!(s.saturating);
        assert_eq!(s.dtype, DType::I32);
    }

    #[test]
    fn promote_same() {
        assert_eq!(DType::F32.promote(DType::F32), DType::F32);
        assert_eq!(DType::I8.promote(DType::I8), DType::I8);
    }

    #[test]
    fn promote_int_widening() {
        assert_eq!(DType::I8.promote(DType::I16), DType::I16);
        assert_eq!(DType::I32.promote(DType::I64), DType::I64);
    }

    #[test]
    fn promote_int_to_float() {
        assert_eq!(DType::I32.promote(DType::F32), DType::F32);
        assert_eq!(DType::I64.promote(DType::F32), DType::F64);
        assert_eq!(DType::I8.promote(DType::F16), DType::F32);
    }

    #[test]
    fn promote_f16_bf16_goes_to_f32() {
        assert_eq!(DType::F16.promote(DType::BF16), DType::F32);
        assert_eq!(DType::BF16.promote(DType::F16), DType::F32);
    }

    #[test]
    fn promote_is_commutative_for_well_defined_pairs() {
        let pairs = [
            (DType::F32, DType::F16),
            (DType::I32, DType::F64),
            (DType::Bool, DType::I8),
        ];
        for (a, b) in pairs {
            assert_eq!(
                a.promote(b),
                b.promote(a),
                "promote({a},{b}) should equal promote({b},{a})"
            );
        }
    }

    #[test]
    fn scalar_constant_bytes_round_trips() {
        assert_eq!(
            scalar_constant_bytes(2.5, DType::F32).unwrap(),
            2.5f32.to_le_bytes().to_vec()
        );
        assert_eq!(
            scalar_constant_bytes(-1.0, DType::F64).unwrap(),
            (-1.0f64).to_le_bytes().to_vec()
        );
        assert_eq!(
            scalar_constant_bytes(7.0, DType::I32).unwrap(),
            7i32.to_le_bytes()
        );
        assert_eq!(scalar_constant_bytes(0.0, DType::Bool).unwrap(), vec![0]);
        assert_eq!(scalar_constant_bytes(1.0, DType::Bool).unwrap(), vec![1]);
    }

    #[test]
    fn scalar_constant_bytes_rejects_out_of_range() {
        assert!(scalar_constant_bytes(128.0, DType::I8).is_err());
        assert!(scalar_constant_bytes(-1.0, DType::U32).is_err());
        assert!(scalar_constant_bytes(9.223372036854776e18, DType::I64).is_err());
        assert!(scalar_constant_bytes(2.5, DType::I32).is_err());
    }

    #[test]
    fn scalar_constant_bytes_rejects_low_precision_direct() {
        assert!(scalar_constant_bytes(1.0, DType::F16).is_err());
        assert!(scalar_constant_bytes(1.0, DType::BF16).is_err());
        assert!(scalar_constant_bytes(1.0, DType::C64).is_err());
        assert!(scalar_constant_bytes(1.0, DType::C128).is_err());
    }

    #[test]
    fn c128_size_and_flags() {
        // 16 bytes: interleaved (re: f64, im: f64), double C64's 8.
        assert_eq!(DType::C128.size_bytes(), 16);
        assert_eq!(DType::C64.size_bytes(), 8);
        // Complex flag: both C64 and C128 are complex, nothing else is.
        assert!(DType::C128.is_complex());
        assert!(DType::C64.is_complex());
        assert!(!DType::F64.is_complex());
        // C128 is neither float nor int.
        assert!(!DType::C128.is_float());
        assert!(!DType::C128.is_int());
        assert_eq!(DType::C128.to_string(), "c128");
    }

    #[test]
    fn c128_promotion() {
        // C128 outranks C64; mixing the two promotes to C128.
        assert_eq!(DType::C64.promote(DType::C128), DType::C128);
        assert_eq!(DType::C128.promote(DType::C64), DType::C128);
        assert_eq!(DType::C128.promote(DType::C128), DType::C128);
        assert!(DType::C128.promotion_rank() > DType::C64.promotion_rank());
    }
}
