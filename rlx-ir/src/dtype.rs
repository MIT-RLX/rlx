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
}

impl DType {
    /// Size in bytes of one element.
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::C64 => 8,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::F64)
    }

    /// True for complex-valued dtypes. Complex elementwise ops follow
    /// standard complex algebra, distinct from the float real/imag
    /// components (e.g. complex multiply ≠ paired-real multiply).
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::C64)
    }

    pub const fn is_int(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::U8 | Self::U32
        )
    }

    /// Promotion rank — higher means "wider, more expressive". The
    /// promoted dtype of a binary op is `max(rank(lhs), rank(rhs))`.
    /// Borrowed from MAX's `dtype_promotion.py` pattern (#55 in
    /// PLAN.md): one module owns the table; ops query it instead of
    /// re-implementing ad-hoc rules.
    ///
    /// Ranks (low → high):
    ///   0 = Bool, 1 = U8/I8, 2 = I16/BF16, 3 = F16, 4 = U32/I32,
    ///   5 = I64, 6 = F32, 7 = F64.
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
}
