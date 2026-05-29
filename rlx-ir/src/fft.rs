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

//! Shared metadata for `Op::Fft` lowering and host-fallback dispatch.
//!
//! The [`FftNorm`] enum, [`FftGpuPlan`], and helpers such as [`next_pow2`],
//! [`fftfreq`], and [`gpu_fft_native_eligible`] are used by every backend
//! that implements `Op::Fft`. Graph-level signal-processing helpers
//! (`rfft`, `irfft`, `stft`, …) live in [`crate::ops::fft_ops`].

use crate::{DType, Shape};

/// Normalization mode for `Op::Fft`.
///
/// * **`Backward`** — both directions unscaled (`ifft(fft(x)) = N·x`). RLX
///   default; AD-friendly.
/// * **`Forward`** — `ifft` scaled by `1/N` after butterflies (gpu-fft /
///   NumPy `norm='backward'` IFFT semantics).
/// * **`Ortho`** — both directions scaled by `1/√N`.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FftNorm {
    #[default]
    Backward,
    Forward,
    Ortho,
}

impl FftNorm {
    /// Post-transform scalar applied to every complex element.
    pub fn output_scale(self, n: usize, inverse: bool) -> f64 {
        let n = n as f64;
        match self {
            FftNorm::Backward => 1.0,
            FftNorm::Forward => {
                if inverse {
                    1.0 / n
                } else {
                    1.0
                }
            }
            FftNorm::Ortho => 1.0 / n.sqrt(),
        }
    }

    /// Stable wire tag for GPU uniform buffers and FFI.
    pub fn tag(self) -> u32 {
        match self {
            FftNorm::Backward => 0,
            FftNorm::Forward => 1,
            FftNorm::Ortho => 2,
        }
    }

    /// Decode a wire tag from [`Self::tag`].
    pub fn from_tag(tag: u32) -> Self {
        match tag {
            0 => FftNorm::Backward,
            1 => FftNorm::Forward,
            2 => FftNorm::Ortho,
            other => panic!("fft: unknown FftNorm tag {other}"),
        }
    }
}

/// Next power of two ≥ `n` (`n == 0` → `1`).
pub fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    1_usize << ((n - 1).ilog2() + 1)
}

/// Bit-reverse `x` in `bits` bits (gpu-fft compatible).
pub fn bit_reverse(mut x: usize, bits: u32) -> usize {
    x = x.reverse_bits() >> (usize::BITS - bits);
    x
}

/// Shared-memory tile size for GPU FFT (matches gpu-fft).
pub const FFT_TILE_SIZE: usize = 1024;
pub const FFT_TILE_BITS: usize = 10;
pub const FFT_WG_SIZE: usize = 256;

/// Launch plan for multi-kernel pow-2 GPU FFT (inner tile + outer stages).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftGpuPlan {
    pub n: usize,
    pub log2n: usize,
    pub inner_stages: usize,
    /// Quarter-strides `q` for each fused radix-4 outer stage.
    pub outer_rad4_q: Vec<usize>,
    /// Trailing radix-2 half-stride when `(log2n - inner_stages)` is odd.
    pub outer_r2_hs: Option<usize>,
}

impl FftGpuPlan {
    /// Pow-2 `n ≥ 2`. Returns `None` when `n` is not a power of two.
    pub fn new(n: usize) -> Option<Self> {
        if n < 2 || !n.is_power_of_two() {
            return None;
        }
        let log2n = n.trailing_zeros() as usize;
        let inner_stages = log2n.min(FFT_TILE_BITS);
        let mut outer_rad4_q = Vec::new();
        let mut rem = log2n.saturating_sub(inner_stages);
        let mut s = inner_stages;
        while rem >= 2 {
            outer_rad4_q.push(1_usize << s);
            s += 2;
            rem -= 2;
        }
        let outer_r2_hs = if rem >= 1 { Some(1_usize << s) } else { None };
        Some(FftGpuPlan {
            n,
            log2n,
            inner_stages,
            outer_rad4_q,
            outer_r2_hs,
        })
    }

    /// Single fused inner kernel covers the full transform (no outer stages).
    pub fn single_inner_only(&self) -> bool {
        self.outer_rad4_q.is_empty() && self.outer_r2_hs.is_none()
    }
}

/// Per-row geometry for a 1D FFT along the **last** axis (after any
/// transpose lowering has moved the target axis to last).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FftMeta {
    /// Number of independent FFT rows (product of all non-FFT axes).
    pub outer: usize,
    /// Complex length along the FFT axis.
    pub n_complex: usize,
    /// Storage extent of the last axis (`2·N` for F32/F64 2N-block,
    /// `N` for native `DType::C64` interleaved layout).
    pub axis_extent: usize,
}

impl FftMeta {
    /// Bytes per FFT row along the last axis.
    pub fn row_bytes(&self, dtype: DType) -> usize {
        match dtype {
            DType::F32 => self.axis_extent * 4,
            DType::F64 => self.axis_extent * 8,
            DType::C64 => self.axis_extent * 8,
            other => panic!("fft: unsupported dtype {other:?}"),
        }
    }
}

/// Infer FFT batch geometry from an `Op::Fft` node shape.
pub fn fft_meta(shape: &Shape) -> FftMeta {
    let rank = shape.rank();
    assert!(rank >= 1, "fft: tensor must have at least 1 axis");
    let axis_extent = shape.dim(rank - 1).unwrap_static();
    let n_complex = match shape.dtype() {
        DType::C64 => axis_extent,
        DType::F32 | DType::F64 => {
            assert!(
                axis_extent.is_multiple_of(2),
                "fft: last axis size {axis_extent} must be even (2N real-block layout)"
            );
            axis_extent / 2
        }
        other => panic!("fft: requires F32, F64, or C64, got {other:?}"),
    };
    let total = shape.num_elements().unwrap_or(0);
    assert!(
        axis_extent > 0 && total.is_multiple_of(axis_extent),
        "fft: shape {shape:?} is not divisible by last-axis extent {axis_extent}"
    );
    FftMeta {
        outer: total / axis_extent,
        n_complex,
        axis_extent,
    }
}

/// Default `fftn` axes: every dimension of a rank-`r` tensor.
pub fn fftn_axes_all(rank: usize) -> Vec<usize> {
    (0..rank).collect()
}

/// True when f32 pow-2 FFT can use native GPU kernels (not host Bluestein).
pub fn gpu_fft_native_eligible(dtype: DType, n_complex: usize) -> bool {
    matches!(dtype, DType::F32) && n_complex.is_power_of_two() && n_complex >= 2
}

/// Prime factors of `n` (`n >= 2`), ascending.
pub fn prime_factors(mut n: usize) -> Vec<usize> {
    if n < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut d = 2usize;
    while d * d <= n {
        while n.is_multiple_of(d) {
            out.push(d);
            n /= d;
        }
        d += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

/// Byte span in the arena covering FFT src/dst row regions (for partial host sync).
pub fn fft_arena_byte_span(
    src_byte_off: usize,
    dst_byte_off: usize,
    row_bytes: usize,
    outer: usize,
) -> (usize, usize) {
    let len = outer * row_bytes;
    let start = src_byte_off.min(dst_byte_off);
    let end = src_byte_off.max(dst_byte_off) + len;
    (start, end - start)
}

/// Sample frequencies for length-`n` FFT (cycles/sample, NumPy `fftfreq` convention).
pub fn fftfreq(n: usize) -> Vec<f64> {
    assert!(n > 0, "fftfreq: n must be positive");
    (0..n)
        .map(|k| {
            let f = if k <= n / 2 {
                k as f64
            } else {
                k as f64 - n as f64
            };
            f / n as f64
        })
        .collect()
}

/// Sample frequencies for length-`n` real FFT (`rfft` has `n/2 + 1` bins).
pub fn rfftfreq(n: usize) -> Vec<f64> {
    assert!(n > 0, "rfftfreq: n must be positive");
    let half = n / 2 + 1;
    (0..half).map(|k| k as f64 / n as f64).collect()
}

/// Normalize FFT axis list: unique, sorted ascending, in-range.
pub fn normalize_fftn_axes(rank: usize, axes: &[usize]) -> Vec<usize> {
    let mut out: Vec<usize> = axes.to_vec();
    out.sort_unstable();
    out.dedup();
    for &ax in &out {
        assert!(
            ax < rank,
            "fftn: axis {ax} out of range for rank-{rank} tensor"
        );
    }
    out
}
