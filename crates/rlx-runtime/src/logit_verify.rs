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

//! Logit / output verification (plan #61).
//!
//! Borrowed from MAX's
//! `tests/integration/accuracy/verify` pattern: every model gets a
//! parity test that diffs RLX's output vs a reference (HuggingFace
//! transformers, ONNX Runtime, hand-fused, ...) using cosine
//! similarity, KL divergence, and absolute tolerance.
//!
//! Pure data layer — no HF / ORT integration here. Test code calls
//! `compare(out, reference, tolerance)` and gets back a structured
//! report it can `assert!` against. Hooking this up to specific
//! reference implementations is per-bench wiring (see `burnembed`).

#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    pub max_abs: f32,
    pub max_rel: f32,
    pub min_cosine: f32,
}

impl Tolerance {
    /// Strict — for f32-vs-f32 comparisons.
    pub const STRICT: Self = Self {
        max_abs: 1e-4,
        max_rel: 1e-3,
        min_cosine: 0.9999,
    };
    /// Loose — for f16 / bf16 against f32 reference.
    pub const LOOSE_F16: Self = Self {
        max_abs: 5e-2,
        max_rel: 5e-2,
        min_cosine: 0.999,
    };
}

#[derive(Debug, Clone)]
pub struct Diff {
    pub n: usize,
    pub max_abs: f32,
    pub max_rel: f32,
    pub mean_abs: f32,
    pub cosine: f32,
    pub argmax_diff_index: usize,
}

#[derive(Debug)]
pub enum VerifyError {
    LengthMismatch { got: usize, expected: usize },
    ToleranceExceeded { diff: Diff, tolerance: Tolerance },
    Nonfinite,
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { got, expected } => {
                write!(f, "length mismatch: got {got}, expected {expected}")
            }
            Self::ToleranceExceeded { diff, tolerance } => write!(
                f,
                "tolerance exceeded: max_abs={:.3e} max_rel={:.3e} cos={:.5} \
                    (limits abs={:.3e}, rel={:.3e}, cos>={:.5})",
                diff.max_abs,
                diff.max_rel,
                diff.cosine,
                tolerance.max_abs,
                tolerance.max_rel,
                tolerance.min_cosine
            ),
            Self::Nonfinite => write!(f, "non-finite value (NaN or inf) in output"),
        }
    }
}

impl std::error::Error for VerifyError {}

/// Compute the per-pair diff between `out` and `ref`. NaN/inf in
/// either side is a hard fail.
pub fn diff(out: &[f32], reference: &[f32]) -> Result<Diff, VerifyError> {
    if out.len() != reference.len() {
        return Err(VerifyError::LengthMismatch {
            got: out.len(),
            expected: reference.len(),
        });
    }
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut sum_abs = 0f32;
    let mut argmax = 0usize;
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for (i, (&a, &b)) in out.iter().zip(reference).enumerate() {
        if !a.is_finite() || !b.is_finite() {
            return Err(VerifyError::Nonfinite);
        }
        let d = (a - b).abs();
        sum_abs += d;
        if d > max_abs {
            max_abs = d;
            argmax = i;
        }
        let denom = b.abs().max(1e-12);
        let rel = d / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    let n = out.len();
    let cosine = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        1.0
    };
    Ok(Diff {
        n,
        max_abs,
        max_rel,
        mean_abs: if n > 0 { sum_abs / n as f32 } else { 0.0 },
        cosine,
        argmax_diff_index: argmax,
    })
}

/// `diff` + tolerance check. Use this in tests / parity harnesses.
pub fn compare(out: &[f32], reference: &[f32], tol: Tolerance) -> Result<Diff, VerifyError> {
    let d = diff(out, reference)?;
    if d.max_abs > tol.max_abs || d.max_rel > tol.max_rel || d.cosine < tol.min_cosine {
        return Err(VerifyError::ToleranceExceeded {
            diff: d,
            tolerance: tol,
        });
    }
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_passes() {
        let a = vec![1.0, 2.0, 3.0];
        let r = compare(&a, &a, Tolerance::STRICT).unwrap();
        assert_eq!(r.max_abs, 0.0);
        assert!((r.cosine - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tiny_diff_passes_strict() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0 + 1e-6, 2.0 - 1e-6, 3.0];
        compare(&a, &b, Tolerance::STRICT).unwrap();
    }

    #[test]
    fn big_diff_fails() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0, 2.0, 99.0];
        let err = compare(&a, &b, Tolerance::STRICT).unwrap_err();
        assert!(matches!(err, VerifyError::ToleranceExceeded { .. }));
    }

    #[test]
    fn nan_is_hard_fail() {
        let a = [1.0f32, f32::NAN];
        let b = [1.0, 0.0];
        let err = compare(&a, &b, Tolerance::STRICT).unwrap_err();
        assert!(matches!(err, VerifyError::Nonfinite));
    }
}
