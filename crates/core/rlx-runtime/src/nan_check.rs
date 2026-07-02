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

//! NaN/inf check epilogue (plan #18).
//!
//! Borrowed from MAX's `Mogg/MOGGKernelAPI/nan_check.mojo` pattern.
//! When the `nan-check` Cargo feature is on, [`scan`] reports the
//! first NaN or inf in a slice — useful as a debug epilogue on
//! every output buffer to localize precision blow-ups to the op
//! that introduced them.
//!
//! Always present in the API surface so callers can compile against
//! it; the feature flag controls whether it's a real scan or a
//! no-op (returns `Ok(())` immediately).

/// What was found in a buffer that fails the check.
#[derive(Debug, Clone, Copy)]
pub enum BadValue {
    Nan,
    PosInf,
    NegInf,
}

#[derive(Debug)]
pub struct NanCheckError {
    pub kind: BadValue,
    pub index: usize,
    pub label: String,
}

impl std::fmt::Display for NanCheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = match self.kind {
            BadValue::Nan => "NaN",
            BadValue::PosInf => "+inf",
            BadValue::NegInf => "-inf",
        };
        write!(f, "{} at index {} of `{}`", what, self.index, self.label)
    }
}

impl std::error::Error for NanCheckError {}

/// Scan `data` for the first NaN or infinity. With the `nan-check`
/// feature OFF, returns `Ok(())` immediately (the optimizer
/// eliminates the call). With it ON, walks the slice — the cost is
/// O(n) but only paid when a caller opts in.
#[cfg(feature = "nan-check")]
#[inline(always)]
pub fn scan(label: &str, data: &[f32]) -> Result<(), NanCheckError> {
    for (i, &v) in data.iter().enumerate() {
        if v.is_nan() {
            return Err(NanCheckError {
                kind: BadValue::Nan,
                index: i,
                label: label.to_string(),
            });
        }
        if v.is_infinite() {
            let kind = if v > 0.0 {
                BadValue::PosInf
            } else {
                BadValue::NegInf
            };
            return Err(NanCheckError {
                kind,
                index: i,
                label: label.to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(not(feature = "nan-check"))]
#[inline(always)]
pub fn scan(_label: &str, _data: &[f32]) -> Result<(), NanCheckError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_data_passes() {
        let data = [1.0, 2.0, -3.5, 0.0];
        assert!(scan("clean", &data).is_ok());
    }

    #[cfg(feature = "nan-check")]
    #[test]
    fn detects_nan() {
        let data = [1.0, f32::NAN, 3.0];
        let err = scan("nan", &data).unwrap_err();
        assert!(matches!(err.kind, BadValue::Nan));
        assert_eq!(err.index, 1);
    }

    #[cfg(feature = "nan-check")]
    #[test]
    fn detects_pos_inf() {
        let data = [f32::INFINITY, 0.0];
        let err = scan("inf", &data).unwrap_err();
        assert!(matches!(err.kind, BadValue::PosInf));
        assert_eq!(err.index, 0);
    }
}
