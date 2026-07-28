// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Audio frontend helpers for `Op::LogMel` / `Op::LogMelBackward`.

use crate::{DType, Shape, shape::Dim};

/// Geometry for `Op::LogMel` from input spectrum + filter shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogMelMeta {
    pub outer: usize,
    pub n_fft: usize,
    pub n_bins: usize,
    pub n_mels: usize,
}

/// Infer log-mel geometry. Spectrum last axis is RLX FFT block layout
/// `[re_0..re_{N-1}, im_0..im_{N-1}]` with length `2 * n_fft`.
pub fn log_mel_meta(spectrum: &Shape, filters: &Shape) -> Result<LogMelMeta, String> {
    if spectrum.dtype() != DType::F32 {
        return Err(format!(
            "Op::LogMel spectrum must be F32, got {:?}",
            spectrum.dtype()
        ));
    }
    if filters.dtype() != DType::F32 {
        return Err(format!(
            "Op::LogMel filters must be F32, got {:?}",
            filters.dtype()
        ));
    }
    if spectrum.rank() < 1 {
        return Err("Op::LogMel spectrum must have rank >= 1".into());
    }
    if filters.rank() != 2 {
        return Err("Op::LogMel filters must be rank-2 [n_mels, n_bins]".into());
    }
    let n_fft2 = spectrum.dim(spectrum.rank() - 1).unwrap_static();
    if !n_fft2.is_multiple_of(2) {
        return Err(format!(
            "Op::LogMel spectrum last dim must be even (2*n_fft), got {n_fft2}"
        ));
    }
    let n_fft = n_fft2 / 2;
    let n_bins = n_fft / 2 + 1;
    let n_mels = filters.dim(0).unwrap_static();
    let filt_bins = filters.dim(1).unwrap_static();
    if filt_bins != n_bins {
        return Err(format!(
            "Op::LogMel filters second dim {filt_bins} != n_bins {n_bins} (n_fft={n_fft})"
        ));
    }
    let outer = spectrum.num_elements().unwrap_or(0) / n_fft2.max(1);
    Ok(LogMelMeta {
        outer,
        n_fft,
        n_bins,
        n_mels,
    })
}

/// Output shape: same leading dims as spectrum, last dim replaced by `n_mels`.
pub fn log_mel_output_shape(spectrum: &Shape, filters: &Shape) -> Result<Shape, String> {
    let meta = log_mel_meta(spectrum, filters)?;
    if spectrum.rank() < 1 {
        return Err("Op::LogMel spectrum rank >= 1 required".into());
    }
    Ok(spectrum
        .clone()
        .with_dim(spectrum.rank() - 1, Dim::Static(meta.n_mels)))
}

fn power_to_log_mel_frame(
    power: &[f32],
    filters: &[f32],
    n_mels: usize,
    n_bins: usize,
) -> Vec<f32> {
    let mut mel = vec![0f32; n_mels];
    for m in 0..n_mels {
        let mut acc = 0f32;
        for k in 0..n_bins {
            acc += filters[m * n_bins + k] * power[k];
        }
        mel[m] = acc.max(1e-10).log10();
    }
    let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = max - 8.0;
    for v in mel.iter_mut() {
        *v = (*v).max(floor);
        *v = (*v + 4.0) / 4.0;
    }
    mel
}

fn log_mel_vjp_frame(
    spectrum_block: &[f32],
    filters: &[f32],
    dy: &[f32],
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
    d_spec: &mut [f32],
) {
    debug_assert_eq!(spectrum_block.len(), n_fft * 2);
    debug_assert_eq!(dy.len(), n_mels);
    debug_assert_eq!(d_spec.len(), n_fft * 2);

    let mut power = vec![0f32; n_bins];
    for k in 0..n_bins {
        let re = spectrum_block[k];
        let im = spectrum_block[n_fft + k];
        power[k] = re * re + im * im;
    }

    let mut mel_raw = vec![0f32; n_mels];
    let mut mel_energy = vec![0f32; n_mels];
    for m in 0..n_mels {
        let mut acc = 0f32;
        for k in 0..n_bins {
            acc += filters[m * n_bins + k] * power[k];
        }
        mel_energy[m] = acc;
        mel_raw[m] = acc.max(1e-10).log10();
    }
    let max = mel_raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = max - 8.0;

    let mut d_mel = vec![0f32; n_mels];
    for m in 0..n_mels {
        let clamped = mel_raw[m].max(floor);
        if (clamped - floor).abs() <= 1e-6 {
            d_mel[m] = 0.0;
        } else {
            d_mel[m] = dy[m] * 0.25;
        }
    }

    let ln10 = std::f32::consts::LN_10;
    for m in 0..n_mels {
        if mel_energy[m] <= 1e-10 {
            continue;
        }
        let d_log = d_mel[m] / (mel_energy[m] * ln10);
        for k in 0..n_bins {
            let d_power = d_log * filters[m * n_bins + k];
            let re = spectrum_block[k];
            let im = spectrum_block[n_fft + k];
            d_spec[k] += d_power * 2.0 * re;
            d_spec[n_fft + k] += d_power * 2.0 * im;
        }
    }
}

/// Whisper-style log-mel from one-sided power of block-layout spectrum.
pub fn log_mel_block_f32(
    spectrum: &[f32],
    filters: &[f32],
    outer: usize,
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(spectrum.len(), outer * n_fft * 2);
    debug_assert_eq!(filters.len(), n_mels * n_bins);
    debug_assert_eq!(out.len(), outer * n_mels);
    for b in 0..outer {
        let spec_base = b * n_fft * 2;
        let mel_base = b * n_mels;
        let mut power = vec![0f32; n_bins];
        for k in 0..n_bins {
            let re = spectrum[spec_base + k];
            let im = spectrum[spec_base + n_fft + k];
            power[k] = re * re + im * im;
        }
        let mel = power_to_log_mel_frame(&power, filters, n_mels, n_bins);
        out[mel_base..mel_base + n_mels].copy_from_slice(&mel);
    }
}

/// Interleaved complex spectrum `[re0, im0, re1, im1, …]`.
pub fn log_mel_interleaved_f32(
    spectrum: &[f32],
    filters: &[f32],
    outer: usize,
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(spectrum.len(), outer * n_fft * 2);
    for b in 0..outer {
        let spec_base = b * n_fft * 2;
        let mel_base = b * n_mels;
        let mut power = vec![0f32; n_bins];
        for k in 0..n_bins {
            let re = spectrum[spec_base + k * 2];
            let im = spectrum[spec_base + k * 2 + 1];
            power[k] = re * re + im * im;
        }
        let mel = power_to_log_mel_frame(&power, filters, n_mels, n_bins);
        out[mel_base..mel_base + n_mels].copy_from_slice(&mel);
    }
}

/// VJP w.r.t. block-layout spectrum for `Op::LogMelBackward`.
pub fn log_mel_block_vjp(
    spectrum: &[f32],
    filters: &[f32],
    dy: &[f32],
    outer: usize,
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
    d_spec: &mut [f32],
) {
    debug_assert_eq!(spectrum.len(), outer * n_fft * 2);
    debug_assert_eq!(dy.len(), outer * n_mels);
    debug_assert_eq!(d_spec.len(), outer * n_fft * 2);
    for b in 0..outer {
        let spec_base = b * n_fft * 2;
        let dy_base = b * n_mels;
        log_mel_vjp_frame(
            &spectrum[spec_base..spec_base + n_fft * 2],
            filters,
            &dy[dy_base..dy_base + n_mels],
            n_fft,
            n_bins,
            n_mels,
            &mut d_spec[spec_base..spec_base + n_fft * 2],
        );
    }
}

/// Geometry for `Op::WelchPeaks` from block-layout segment spectra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WelchPeaksMeta {
    pub welch_batch: usize,
    pub n_fft: usize,
    pub n_bins: usize,
    pub n_segments: usize,
    pub k: usize,
}

pub fn welch_peaks_meta(
    spectrum: &Shape,
    k: usize,
    n_segments: usize,
) -> Result<WelchPeaksMeta, String> {
    if spectrum.dtype() != DType::F32 {
        return Err(format!(
            "Op::WelchPeaks spectrum must be F32, got {:?}",
            spectrum.dtype()
        ));
    }
    if spectrum.rank() != 2 {
        return Err("Op::WelchPeaks spectrum must be [outer, 2*n_fft]".into());
    }
    if n_segments == 0 || k == 0 {
        return Err("Op::WelchPeaks requires k >= 1 and n_segments >= 1".into());
    }
    let n_fft2 = spectrum.dim(1).unwrap_static();
    if !n_fft2.is_multiple_of(2) {
        return Err(format!(
            "Op::WelchPeaks spectrum last dim must be 2*n_fft, got {n_fft2}"
        ));
    }
    let n_fft = n_fft2 / 2;
    let outer = spectrum.dim(0).unwrap_static();
    if !outer.is_multiple_of(n_segments) {
        return Err(format!(
            "Op::WelchPeaks outer {outer} not divisible by n_segments {n_segments}"
        ));
    }
    Ok(WelchPeaksMeta {
        welch_batch: outer / n_segments,
        n_fft,
        n_bins: n_fft / 2 + 1,
        n_segments,
        k,
    })
}

pub fn welch_peaks_output_shape(
    spectrum: &Shape,
    k: usize,
    n_segments: usize,
) -> Result<Shape, String> {
    let meta = welch_peaks_meta(spectrum, k, n_segments)?;
    Ok(Shape::new(&[meta.welch_batch, meta.k * 2], DType::F32))
}

/// Max one-sided bins for native GPU WelchPeaks kernels (WGSL/CUDA).
pub const WELCH_PEAKS_GPU_MAX_BINS: usize = 512;

/// Max top-K for native GPU WelchPeaks kernels.
pub const WELCH_PEAKS_GPU_MAX_K: usize = 64;

/// True when backends may use in-arena GPU WelchPeaks instead of tail-host CPU.
pub fn welch_peaks_gpu_native_eligible(
    spectrum: &Shape,
    k: usize,
    n_segments: usize,
) -> Result<bool, String> {
    let meta = welch_peaks_meta(spectrum, k, n_segments)?;
    Ok(spectrum.dtype() == DType::F32
        && meta.n_bins <= WELCH_PEAKS_GPU_MAX_BINS
        && meta.k <= WELCH_PEAKS_GPU_MAX_K)
}

fn accumulate_block_power_row(row: &mut [f32], block: &[f32], n_fft: usize, scale: f32) {
    let n_bins = n_fft / 2 + 1;
    debug_assert!(block.len() >= n_fft * 2);
    row[0] += scale * (block[0] * block[0] + block[n_fft] * block[n_fft]);
    for bin in 1..n_bins.saturating_sub(1) {
        let re = block[bin];
        let im = block[n_fft + bin];
        row[bin] += scale * 2.0 * (re * re + im * im);
    }
    if n_bins > 1 {
        let bin = n_bins - 1;
        row[bin] += scale * (block[bin] * block[bin] + block[n_fft + bin] * block[n_fft + bin]);
    }
}

fn topk_peaks_one(psd: &[f32], k: usize) -> Vec<(usize, f32)> {
    use std::cmp::Ordering;
    let n_bins = psd.len();
    let k = k.min(n_bins).max(1);
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k);
    for (bin, &power) in psd.iter().enumerate() {
        if top.len() < k {
            top.push((bin, power));
            if top.len() == k {
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            }
            continue;
        }
        if power <= top[k - 1].1 {
            continue;
        }
        top[k - 1] = (bin, power);
        let mut i = k - 1;
        while i > 0 && top[i].1 > top[i - 1].1 {
            top.swap(i, i - 1);
            i -= 1;
        }
    }
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    top
}

/// Block-layout segment spectra `[outer, 2*n_fft]` → packed peaks `[batch, k*2]`.
pub fn welch_peaks_block_f32(
    spectrum: &[f32],
    welch_batch: usize,
    n_fft: usize,
    n_segments: usize,
    k: usize,
    out: &mut [f32],
) {
    let n_bins = n_fft / 2 + 1;
    let row_len = n_fft * 2;
    let inv = 1.0 / n_segments as f32;
    let mut psd = vec![0f32; n_bins];
    for b in 0..welch_batch {
        psd.fill(0.0);
        for s in 0..n_segments {
            let base = (b * n_segments + s) * row_len;
            accumulate_block_power_row(&mut psd, &spectrum[base..base + row_len], n_fft, inv);
        }
        let peaks = topk_peaks_one(&psd, k);
        for (i, &(bin, power)) in peaks.iter().enumerate().take(k) {
            let dst = (b * k + i) * 2;
            out[dst] = bin as f32;
            out[dst + 1] = power;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_mel_meta_shapes() {
        let spec = Shape::new(&[4, 256], DType::F32);
        let filt = Shape::new(&[80, 65], DType::F32);
        let meta = log_mel_meta(&spec, &filt).unwrap();
        assert_eq!(meta.outer, 4);
        assert_eq!(meta.n_fft, 128);
        assert_eq!(meta.n_bins, 65);
        assert_eq!(meta.n_mels, 80);
    }

    #[test]
    fn log_mel_vjp_nonzero() {
        let n_fft = 32;
        let n_mels = 4;
        let n_bins = n_fft / 2 + 1;
        let filters: Vec<f32> = (0..n_mels * n_bins)
            .map(|i| (i % 5) as f32 * 0.05 + 0.02)
            .collect();
        let mut spec = vec![0f32; n_fft * 2];
        for k in 0..n_bins {
            spec[k] = 0.2 * (k as f32 + 1.0);
            spec[n_fft + k] = -0.1 * k as f32;
        }
        let dy = vec![1.0f32; n_mels];
        let mut d_spec = vec![0f32; n_fft * 2];
        log_mel_block_vjp(&spec, &filters, &dy, 1, n_fft, n_bins, n_mels, &mut d_spec);
        assert!(d_spec.iter().any(|v| v.abs() > 1e-6));
        assert!(d_spec[0].abs() < 1.0);
    }

    #[test]
    fn block_and_interleaved_mel_match() {
        let n_fft = 64;
        let n_mels = 8;
        let n_bins = n_fft / 2 + 1;
        let filters: Vec<f32> = (0..n_mels * n_bins)
            .map(|i| (i % 7) as f32 * 0.03 + 0.01)
            .collect();
        let mut block = vec![0f32; n_fft * 2];
        for k in 0..n_bins {
            block[k] = (k as f32 * 0.11).sin();
            block[n_fft + k] = (k as f32 * 0.07).cos();
        }
        let mut interleaved = vec![0f32; n_fft * 2];
        for k in 0..n_fft {
            interleaved[k * 2] = block[k];
            interleaved[k * 2 + 1] = block[n_fft + k];
        }
        let mut out_block = vec![0f32; n_mels];
        let mut out_int = vec![0f32; n_mels];
        log_mel_block_f32(&block, &filters, 1, n_fft, n_bins, n_mels, &mut out_block);
        log_mel_interleaved_f32(
            &interleaved,
            &filters,
            1,
            n_fft,
            n_bins,
            n_mels,
            &mut out_int,
        );
        for (a, b) in out_block.iter().zip(out_int.iter()) {
            assert!((a - b).abs() < 1e-6, "block={a} interleaved={b}");
        }
    }

    #[test]
    fn welch_peaks_meta_and_topk() {
        let spec = Shape::new(&[8, 256], DType::F32);
        let meta = welch_peaks_meta(&spec, 4, 2).unwrap();
        assert_eq!(meta.welch_batch, 4);
        assert_eq!(meta.n_fft, 128);
        assert_eq!(meta.k, 4);
        let out_shape = welch_peaks_output_shape(&spec, 4, 2).unwrap();
        assert_eq!(out_shape.dims()[0], crate::shape::Dim::Static(4));
        assert_eq!(out_shape.dims()[1], crate::shape::Dim::Static(8));

        let n_fft = 32;
        let n_bins = n_fft / 2 + 1;
        let row_len = n_fft * 2;
        let n_segments = 2;
        let welch_batch = 2;
        let mut spectrum = vec![0f32; welch_batch * n_segments * row_len];
        for s in 0..welch_batch * n_segments {
            let base = s * row_len;
            for k in 0..n_bins {
                let amp = if k == 7 { 3.0 } else { 0.01 * k as f32 };
                spectrum[base + k] = amp;
                spectrum[base + n_fft + k] = 0.0;
            }
        }
        let mut out = vec![0f32; welch_batch * 2 * 2];
        welch_peaks_block_f32(&spectrum, welch_batch, n_fft, n_segments, 2, &mut out);
        assert!((out[0] - 7.0).abs() < 1e-5);
        assert!(out[1] > 1.0);
    }
}
