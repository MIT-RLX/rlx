// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
        // I32 is the fixed-point transform's type (`Op::FftQ`); it shares the
        // 2N real-block layout, which is what this function measures.
        DType::F32 | DType::F64 | DType::I32 => {
            assert!(
                axis_extent.is_multiple_of(2),
                "fft: last axis size {axis_extent} must be even (2N real-block layout)"
            );
            axis_extent / 2
        }
        other => panic!("fft: requires F32, F64, C64, or I32, got {other:?}"),
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

/// How a fixed-point transform keeps its datapath in range.
///
/// The choice is a precision decision, not a detail. Halving every stage is the
/// textbook answer and costs one bit per stage — ten bits over a 1024-point
/// transform — which is often more than the caller can spare.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FftQScale {
    /// No scaling, wrapping arithmetic. The caller guarantees headroom: a
    /// length-`n` transform grows magnitude by at most `n`, so `|x| < 2^31 / n`
    /// is always safe, and real audio is far below that (i16 samples through a
    /// 1024-point transform reach 25 bits, six short of overflow). Keeps every
    /// bit the input had, and is the fastest path.
    #[default]
    None,
    /// No scaling, saturating arithmetic. Same precision and headroom as
    /// [`Self::None`], but an input that overflows anyway clips instead of
    /// wrapping — a loud frame becomes a wrong-but-bounded spectrum rather
    /// than a sign-flipped one. Prefer this unless the bound is proven.
    Saturating,
    /// Halve on alternate stages. Absorbs `2^(log2(n)/2)` of growth for
    /// `log2(n)/2` bits — the middle of the trade, for inputs whose headroom is
    /// known to be tight rather than absent.
    EveryOther,
    /// Halve after every butterfly stage, with rounding. Cannot overflow for
    /// any input, and costs `log2(n)` bits of precision.
    PerStage,
}

impl FftQScale {
    /// Bits of precision this policy gives up on a length-`n` transform.
    pub fn bits_lost(self, n: usize) -> u32 {
        let stages = n.trailing_zeros();
        match self {
            Self::None | Self::Saturating => 0,
            Self::EveryOther => stages.div_ceil(2),
            Self::PerStage => stages,
        }
    }

    /// Magnitude growth this policy leaves for the caller to have headroom for.
    /// `None`/`Saturating` absorb none of the `n`-fold worst case.
    pub fn headroom_needed(self, n: usize) -> u32 {
        n.trailing_zeros() - self.bits_lost(n)
    }

    pub fn tag(self) -> u32 {
        match self {
            Self::None => 0,
            Self::PerStage => 1,
            Self::Saturating => 2,
            Self::EveryOther => 3,
        }
    }

    pub fn from_tag(t: u32) -> Self {
        match t {
            1 => Self::PerStage,
            2 => Self::Saturating,
            3 => Self::EveryOther,
            _ => Self::None,
        }
    }
}

/// Fractional bits in a fixed-point twiddle factor.
///
/// Q30 in an `i32`, multiplied into `i64`. The per-stage rounding error is then
/// about `2^-30`, accumulating to roughly `2^-26` over ten stages — far below
/// what a Q15 table would leave.
pub const FFT_Q_TWIDDLE_FRAC: u32 = 30;

/// Twiddles `e^{∓2πik/n}` for `k in 0..n/2`, as Q30 `(re, im)`.
fn twiddles_q30(n: usize, inverse: bool) -> (Vec<i32>, Vec<i32>) {
    let scale = (1i64 << FFT_Q_TWIDDLE_FRAC) as f64;
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut re = Vec::with_capacity(n / 2);
    let mut im = Vec::with_capacity(n / 2);
    for k in 0..n / 2 {
        let a = sign * 2.0 * core::f64::consts::PI * k as f64 / n as f64;
        re.push((a.cos() * scale).round() as i32);
        im.push((a.sin() * scale).round() as i32);
    }
    (re, im)
}

#[inline]
fn rshift_round(v: i64, s: u32) -> i32 {
    if s == 0 {
        return v as i32;
    }
    ((v + (1 << (s - 1))) >> s) as i32
}

/// In-place radix-2 fixed-point FFT over `outer` rows of the 2N-block layout
/// (`n` real parts then `n` imaginary parts), matching [`crate::Op::Fft`]'s
/// `F32` convention.
///
/// `n` must be a power of two: Bluestein's algorithm for other lengths needs a
/// chirp whose fixed-point error analysis is a different problem, and the
/// caller is better served by an explicit failure than by a quietly poor
/// transform.
///
/// # Accuracy
///
/// Error is **absolute**, not relative. Every butterfly rounds to an integer,
/// so it contributes a fraction of an LSB whatever the signal level, and the
/// relative accuracy a caller sees is whatever their input's magnitude makes
/// of that. Scale the input to fill the headroom — with [`FftQScale::None`]
/// that is `2^31 / n`, so full-scale i16 audio through a 1024-point transform
/// uses 25 of the 31 bits and lands within `3e-5` of a direct DFT.
pub fn fft1d_q32_block(
    data: &mut [i32],
    outer: usize,
    n: usize,
    inverse: bool,
    norm: FftNorm,
    scale: FftQScale,
) -> Result<(), String> {
    if !n.is_power_of_two() {
        return Err(format!(
            "fixed-point FFT length {n} is not a power of two; only radix-2 is supported"
        ));
    }
    if data.len() != outer * 2 * n {
        return Err(format!(
            "fixed-point FFT expects {} elements ({outer} x 2 x {n}), got {}",
            outer * 2 * n,
            data.len()
        ));
    }
    if n < 2 {
        return Ok(());
    }
    let plan = FftQ32Plan::new(n, inverse, norm, scale)?;
    for row in data.chunks_exact_mut(2 * n) {
        plan.run_row(row);
    }
    Ok(())
}

/// Precomputed state shared by every row of a fixed-point block transform.
///
/// Rows are independent and share only these read-only twiddles, so a caller
/// that has a thread pool can fan out across rows without recomputing them.
/// `rlx-ir` carries no rayon dependency by design; this is the seam that lets
/// `rlx_cpu::thunk::fft1d_q32_block_parallel` — and through it every GPU
/// backend's host fallback — parallelize the batch.
///
/// Splitting a block into per-row calls to [`fft1d_q32_block`] would work too,
/// but would rebuild the twiddle tables once per row.
pub struct FftQ32Plan {
    tw_re: Vec<i32>,
    tw_im: Vec<i32>,
    n: usize,
    bits: u32,
    inverse: bool,
    norm: FftNorm,
    scale: FftQScale,
}

impl FftQ32Plan {
    /// Build the shared state for length-`n` rows. `n` must be a power of two.
    pub fn new(n: usize, inverse: bool, norm: FftNorm, scale: FftQScale) -> Result<Self, String> {
        if !n.is_power_of_two() {
            return Err(format!(
                "fixed-point FFT length {n} is not a power of two; only radix-2 is supported"
            ));
        }
        let (tw_re, tw_im) = twiddles_q30(n, inverse);
        Ok(Self {
            tw_re,
            tw_im,
            n,
            bits: n.trailing_zeros(),
            inverse,
            norm,
            scale,
        })
    }

    /// Elements per row: `2 * n`, laid out `[re | im]`.
    pub fn row_elems(&self) -> usize {
        2 * self.n
    }

    /// Transform one row in place. Touches nothing outside `row`, which is what
    /// makes a parallel fan-out bit-identical to the serial loop.
    ///
    /// # Panics
    /// If `row.len() != self.row_elems()`.
    pub fn run_row(&self, row: &mut [i32]) {
        assert_eq!(
            row.len(),
            2 * self.n,
            "FftQ32Plan::run_row expects a {}-element row",
            2 * self.n
        );
        let n = self.n;
        // Bit-reversal permutation.
        for i in 0..n {
            let j = ((i as u32).reverse_bits() >> (32 - self.bits)) as usize;
            if j > i {
                row.swap(i, j);
                row.swap(n + i, n + j);
            }
        }
        let mut len = 2;
        let mut stage = 0u32;
        while len <= n {
            let step = n / len;
            for base in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let (wr, wi) = (
                        i64::from(self.tw_re[k * step]),
                        i64::from(self.tw_im[k * step]),
                    );
                    let (a, b) = (base + k, base + k + len / 2);
                    let (br, bi) = (i64::from(row[b]), i64::from(row[n + b]));
                    let tr = rshift_round(wr * br - wi * bi, FFT_Q_TWIDDLE_FRAC);
                    let ti = rshift_round(wr * bi + wi * br, FFT_Q_TWIDDLE_FRAC);
                    let (ar, ai) = (row[a], row[n + a]);
                    if self.scale == FftQScale::Saturating {
                        row[b] = ar.saturating_sub(tr);
                        row[n + b] = ai.saturating_sub(ti);
                        row[a] = ar.saturating_add(tr);
                        row[n + a] = ai.saturating_add(ti);
                    } else {
                        row[b] = ar.wrapping_sub(tr);
                        row[n + b] = ai.wrapping_sub(ti);
                        row[a] = ar.wrapping_add(tr);
                        row[n + a] = ai.wrapping_add(ti);
                    }
                }
            }
            let halve = match self.scale {
                FftQScale::PerStage => true,
                FftQScale::EveryOther => stage.is_multiple_of(2),
                FftQScale::None | FftQScale::Saturating => false,
            };
            if halve {
                for v in &mut row[..2 * n] {
                    *v = rshift_round(i64::from(*v), 1);
                }
            }
            stage += 1;
            len <<= 1;
        }

        // Output normalisation. `Backward` is the identity; `Forward` is an
        // exact shift because n is a power of two; `Ortho` needs 1/sqrt(n),
        // which is not, so it goes through a Q30 multiply.
        match self.norm {
            FftNorm::Backward => {}
            FftNorm::Forward => {
                if !self.inverse {
                    // Forward normalisation divides the forward transform.
                    for v in &mut row[..2 * n] {
                        *v = rshift_round(i64::from(*v), self.bits);
                    }
                } else {
                    // and leaves the inverse alone.
                }
            }
            FftNorm::Ortho => {
                let s = ((1i64 << FFT_Q_TWIDDLE_FRAC) as f64 / (n as f64).sqrt()).round() as i64;
                for v in &mut row[..2 * n] {
                    *v = rshift_round(i64::from(*v) * s, FFT_Q_TWIDDLE_FRAC);
                }
            }
        }
    }
}

#[cfg(test)]
mod q_tests {
    use super::*;

    /// Direct DFT in f64, the reference these are held to.
    fn dft(re: &[f64], im: &[f64], inverse: bool) -> (Vec<f64>, Vec<f64>) {
        let n = re.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        let mut or = vec![0.0; n];
        let mut oi = vec![0.0; n];
        for k in 0..n {
            for t in 0..n {
                let a = sign * 2.0 * core::f64::consts::PI * (k * t) as f64 / n as f64;
                or[k] += re[t] * a.cos() - im[t] * a.sin();
                oi[k] += re[t] * a.sin() + im[t] * a.cos();
            }
        }
        (or, oi)
    }

    fn block(re: &[i32], im: &[i32]) -> Vec<i32> {
        let mut v = re.to_vec();
        v.extend_from_slice(im);
        v
    }

    /// Accuracy is *absolute*: every butterfly rounds to an integer, so the
    /// error is a fraction of an LSB per stage regardless of signal level, and
    /// the relative accuracy a caller sees is whatever their input's magnitude
    /// makes of that. Full-scale audio is the intended case.
    #[test]
    fn matches_a_direct_dft() {
        for n in [8usize, 64, 256, 1024] {
            let re: Vec<i32> = (0..n)
                .map(|t| ((t as f64 * 0.37).sin() * 20000.0) as i32)
                .collect();
            let im = vec![0i32; n];
            let mut data = block(&re, &im);
            fft1d_q32_block(&mut data, 1, n, false, FftNorm::Backward, FftQScale::None).unwrap();

            let (wr, wi) = dft(
                &re.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                &vec![0.0; n],
                false,
            );
            let peak = wr
                .iter()
                .zip(&wi)
                .fold(0.0f64, |m, (a, b)| m.max((a * a + b * b).sqrt()));
            let mut worst = 0.0f64;
            for k in 0..n {
                let dr = f64::from(data[k]) - wr[k];
                let di = f64::from(data[n + k]) - wi[k];
                worst = worst.max((dr * dr + di * di).sqrt() / peak);
            }
            assert!(worst < 3e-5, "n={n}: worst bin {worst:.2e} of peak");
        }
    }

    #[test]
    fn round_trips_to_n_times_the_input() {
        // Backward normalisation is unnormalised both ways, so ifft(fft(x)) = n·x.
        let n = 256usize;
        let re: Vec<i32> = (0..n)
            .map(|t| (((t * 37 % 211) as i32) - 105) * 150)
            .collect();
        let mut data = block(&re, &vec![0; n]);
        fft1d_q32_block(&mut data, 1, n, false, FftNorm::Backward, FftQScale::None).unwrap();
        fft1d_q32_block(&mut data, 1, n, true, FftNorm::Backward, FftQScale::None).unwrap();
        // Rounding accumulates over 2·log2(n) stages, so the round trip is
        // exact to a relative tolerance, not to the last integer.
        let peak = re.iter().map(|v| v.abs()).max().unwrap() as f64 * n as f64;
        for t in 0..n {
            let want = f64::from(re[t]) * n as f64;
            let rel = (f64::from(data[t]) - want).abs() / peak;
            assert!(rel < 1e-3, "sample {t}: {} vs {want} ({rel:.2e})", data[t]);
        }
    }

    /// The scaling policy is a precision decision, and the difference is large
    /// enough to be worth stating in the type.
    #[test]
    fn per_stage_scaling_costs_a_bit_per_stage() {
        let n = 1024usize;
        let re: Vec<i32> = (0..n)
            .map(|t| ((t as f64 * 0.11).cos() * 20000.0) as i32)
            .collect();
        let (wr, wi) = dft(
            &re.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            &vec![0.0; n],
            false,
        );
        let peak = wr
            .iter()
            .zip(&wi)
            .fold(0.0f64, |m, (a, b)| m.max((a * a + b * b).sqrt()));

        let err = |scale: FftQScale, gain: f64| {
            let mut d = block(&re, &vec![0; n]);
            fft1d_q32_block(&mut d, 1, n, false, FftNorm::Backward, scale).unwrap();
            let mut worst = 0.0f64;
            for k in 0..n {
                let dr = f64::from(d[k]) * gain - wr[k];
                let di = f64::from(d[n + k]) * gain - wi[k];
                worst = worst.max((dr * dr + di * di).sqrt() / peak);
            }
            worst
        };
        let none = err(FftQScale::None, 1.0);
        let staged = err(FftQScale::PerStage, n as f64);
        assert!(none < 1e-4, "unscaled error {none:.2e}");
        // Both are reported so the trade is visible when this test is read.
        assert!(
            staged > none * 50.0,
            "per-stage {staged:.2e} should be far worse than unscaled {none:.2e}"
        );
    }

    #[test]
    fn rejects_lengths_it_cannot_do_well() {
        let mut d = vec![0i32; 2 * 12];
        let e = fft1d_q32_block(&mut d, 1, 12, false, FftNorm::Backward, FftQScale::None);
        assert!(e.unwrap_err().contains("power of two"));
    }
}

#[cfg(test)]
mod q_scale_tests {
    use super::*;

    fn tone(n: usize, amp: f64) -> Vec<i32> {
        (0..n)
            .map(|t| ((t as f64 * 0.19).sin() * amp) as i32)
            .collect()
    }

    fn run(re: &[i32], n: usize, scale: FftQScale) -> Vec<i32> {
        let mut d = re.to_vec();
        d.extend(core::iter::repeat_n(0, n));
        fft1d_q32_block(&mut d, 1, n, false, FftNorm::Backward, scale).unwrap();
        d
    }

    /// Precision falls in the documented order, and `bits_lost` predicts it.
    #[test]
    fn scaling_policies_trade_precision_for_headroom() {
        let n = 1024usize;
        let re = tone(n, 20000.0);
        let mut base = re.clone();
        base.extend(core::iter::repeat_n(0, n));
        // Reference: the same transform with no scaling, which keeps every bit.
        let exact = run(&re, n, FftQScale::None);
        let peak = (0..n)
            .map(|k| {
                let (a, b) = (f64::from(exact[k]), f64::from(exact[n + k]));
                (a * a + b * b).sqrt()
            })
            .fold(0.0f64, f64::max);

        let err_of = |scale: FftQScale| {
            let got = run(&re, n, scale);
            let gain = f64::from(1u32 << scale.bits_lost(n));
            (0..n)
                .map(|k| {
                    let dr = f64::from(got[k]) * gain - f64::from(exact[k]);
                    let di = f64::from(got[n + k]) * gain - f64::from(exact[n + k]);
                    (dr * dr + di * di).sqrt() / peak
                })
                .fold(0.0f64, f64::max)
        };

        let sat = err_of(FftQScale::Saturating);
        let every = err_of(FftQScale::EveryOther);
        let per = err_of(FftQScale::PerStage);
        assert!(
            sat < 1e-9,
            "saturating should match unscaled exactly: {sat:.2e}"
        );
        assert!(
            every < per,
            "every-other {every:.2e} should beat per-stage {per:.2e}"
        );
        assert_eq!(FftQScale::None.bits_lost(n), 0);
        assert_eq!(FftQScale::EveryOther.bits_lost(n), 5);
        assert_eq!(FftQScale::PerStage.bits_lost(n), 10);
        assert_eq!(FftQScale::PerStage.headroom_needed(n), 0);
        assert_eq!(FftQScale::None.headroom_needed(n), 10);
    }

    /// The point of `Saturating`: an input with no headroom clips instead of
    /// wrapping, so a loud frame stays loud rather than changing sign.
    #[test]
    fn saturating_clips_where_none_wraps() {
        let n = 64usize;
        // Deliberately over the `2^31 / n` bound.
        let re: Vec<i32> = (0..n).map(|_| i32::MAX / 4).collect();
        let wrapped = run(&re, n, FftQScale::None);
        let clipped = run(&re, n, FftQScale::Saturating);
        // DC of a constant run is n·x, which overflows.
        assert!(
            wrapped[0] < 0,
            "expected the unscaled path to wrap: {}",
            wrapped[0]
        );
        assert_eq!(clipped[0], i32::MAX, "saturating should clip at the top");
    }

    /// And `PerStage` survives that same input with no headroom at all.
    #[test]
    fn per_stage_needs_no_headroom() {
        let n = 64usize;
        let re: Vec<i32> = (0..n).map(|_| i32::MAX / 4).collect();
        let out = run(&re, n, FftQScale::PerStage);
        // DC = n·x scaled down by n, so back to x.
        let want = i32::MAX / 4;
        let rel = (f64::from(out[0]) - f64::from(want)).abs() / f64::from(want);
        assert!(rel < 1e-6, "DC {} vs {want} ({rel:.2e})", out[0]);
    }
}
