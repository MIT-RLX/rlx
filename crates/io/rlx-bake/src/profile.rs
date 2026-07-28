// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Named bake optimization profiles (`--opt` / [`BakeProfile`]).

use crate::BakeOptions;
use std::fmt;
use std::str::FromStr;

/// High-level bake configuration. Fine flags can still override individual passes.
///
/// | Profile | Intent | Weight rewrites | Cleanup |
/// |---------|--------|-----------------|---------|
/// | [`Merge`](Self::Merge) | One file, same dense compute | none | off |
/// | [`Fold`](Self::Fold) | Fold weight-only math; keep dense GEMM | none | on |
/// | [`Exact`](Self::Exact) | Lossless value-based rewrites (default) | skip + ternary | on |
/// | [`Size`](Self::Size) | Smaller payload (may change numerics) | skip + ternary + Q8 | on |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BakeProfile {
    /// Specialize params + unfold weight table only. No packing, skip, or fold.
    Merge,
    /// Specialize + algebraic simplify / DCE / constant fold. No skip / ternary / quant.
    Fold,
    /// Exact rewrites: skip zero matmuls + ternary TQ2 pack + cleanup. No Q8.
    #[default]
    Exact,
    /// Prefer smaller weight bytes: [`Exact`](Self::Exact) plus Q8_0 pack of remaining matmuls.
    Size,
}

impl BakeProfile {
    /// All profile names accepted by [`FromStr`] / the CLI (canonical first).
    pub fn all_names() -> &'static [&'static str] {
        &[
            "merge", "fold", "exact", "size", // canonical
            "none", "raw",     // → merge
            "cleanup", // → fold
            "lossless", "default", "compute", // → exact
            "compact", "quant", // → size
        ]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Merge => "merge",
            Self::Fold => "fold",
            Self::Exact => "exact",
            Self::Size => "size",
        }
    }

    /// Human-readable intent (for CLI / logs).
    pub fn description(self) -> &'static str {
        match self {
            Self::Merge => "package only (same dense MatMul compute)",
            Self::Fold => "fold weight-only math; keep dense GEMM",
            Self::Exact => "lossless skip + ternary pack + cleanup",
            Self::Size => "exact + Q8_0 pack remaining matmul weights",
        }
    }

    pub fn options(self) -> BakeOptions {
        BakeOptions::from_profile(self)
    }
}

impl fmt::Display for BakeProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for BakeProfile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "merge" | "none" | "raw" => Ok(Self::Merge),
            "fold" | "cleanup" => Ok(Self::Fold),
            "exact" | "lossless" | "default" | "compute" => Ok(Self::Exact),
            "size" | "compact" | "quant" => Ok(Self::Size),
            other => Err(format!(
                "unknown bake profile {other:?}; expected one of: merge, fold, exact, size \
                 (aliases: none/raw, cleanup, lossless/default/compute, compact/quant)"
            )),
        }
    }
}

impl From<BakeProfile> for BakeOptions {
    fn from(profile: BakeProfile) -> Self {
        BakeOptions::from_profile(profile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_aliases() {
        assert_eq!("merge".parse::<BakeProfile>().unwrap(), BakeProfile::Merge);
        assert_eq!("none".parse::<BakeProfile>().unwrap(), BakeProfile::Merge);
        assert_eq!("fold".parse::<BakeProfile>().unwrap(), BakeProfile::Fold);
        assert_eq!("exact".parse::<BakeProfile>().unwrap(), BakeProfile::Exact);
        assert_eq!(
            "compute".parse::<BakeProfile>().unwrap(),
            BakeProfile::Exact
        );
        assert_eq!("size".parse::<BakeProfile>().unwrap(), BakeProfile::Size);
        assert_eq!("quant".parse::<BakeProfile>().unwrap(), BakeProfile::Size);
        assert!("bogus".parse::<BakeProfile>().is_err());
    }

    #[test]
    fn size_enables_quant() {
        let o = BakeProfile::Size.options();
        assert!(o.quant && o.ternary && o.skip_zero);
        assert_eq!(o.memory, crate::MemoryMode::Compact);
        let e = BakeProfile::Exact.options();
        assert!(!e.quant && e.ternary && e.skip_zero);
        assert!(e.dedupe_constants && !e.keep_folded_bindings);
        let m = BakeProfile::Merge.options();
        assert!(!m.skip_zero && !m.ternary && !m.quant && !m.constant_folding);
        assert_eq!(m.memory, crate::MemoryMode::Duplex);
        let f = BakeProfile::Fold.options();
        assert!(!f.skip_zero && !f.ternary && f.constant_folding);
    }
}
