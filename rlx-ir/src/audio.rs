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
}
