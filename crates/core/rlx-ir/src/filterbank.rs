// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side constructors for the audio frontend: analysis windows, signal
//! framing, and the filterbanks `Op::LogMel` / a plain `mm` consume.
//!
//! Layouts match [`crate::audio`]: a filterbank is row-major
//! `[n_filters, n_bins]` with `n_bins = n_fft / 2 + 1`, so it can be fed
//! directly as the `filters` input of [`crate::Graph::log_mel`], or
//! transposed into `[n_bins, n_filters]` for `mm` against a power spectrum.
//!
//! Framing on the host produces a `[n_frames, frame_len]` tensor, which one
//! batched `rfft` transforms in a single op — cheaper to build than
//! [`crate::Graph::stft`], which emits a slice per frame and grows the graph
//! with signal length.

/// Periodic Hann window of length `n` (the STFT convention).
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = 2.0 * core::f64::consts::PI * i as f64 / n as f64;
            (0.5 - 0.5 * x.cos()) as f32
        })
        .collect()
}

/// Number of frames [`frame_signal`] produces.
pub fn frame_count(len: usize, frame_len: usize, hop: usize) -> usize {
    if len < frame_len || hop == 0 {
        if len == 0 { 0 } else { 1 }
    } else {
        1 + (len - frame_len) / hop
    }
}

/// Cut `signal` into overlapping frames, multiplied by `window` (if given),
/// as a row-major `[n_frames, frame_len]` buffer. A signal shorter than one
/// frame yields one zero-padded frame.
pub fn frame_signal(
    signal: &[f32],
    frame_len: usize,
    hop: usize,
    window: Option<&[f32]>,
) -> Vec<f32> {
    assert!(
        frame_len > 0 && hop > 0,
        "frame_signal: frame_len and hop must be positive"
    );
    if let Some(w) = window {
        assert_eq!(
            w.len(),
            frame_len,
            "frame_signal: window length must equal frame_len"
        );
    }
    let n = frame_count(signal.len(), frame_len, hop);
    let mut out = vec![0f32; n * frame_len];
    for f in 0..n {
        let start = f * hop;
        for i in 0..frame_len {
            let x = signal.get(start + i).copied().unwrap_or(0.0);
            out[f * frame_len + i] = match window {
                Some(w) => x * w[i],
                None => x,
            };
        }
    }
    out
}

/// Hz → mel on the HTK scale: `2595 · log10(1 + f/700)`.
pub fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Inverse of [`hz_to_mel`].
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// Triangular mel filterbank, `[n_mels, n_fft/2 + 1]` row-major.
///
/// Filters are spaced evenly on the HTK mel scale between `f_min` and
/// `f_max` (clamped to Nyquist), peak at 1, and are not area-normalised —
/// the same shape as `librosa.filters.mel(htk=True, norm=None)`.
pub fn mel_filterbank(
    sample_rate: f64,
    n_fft: usize,
    n_mels: usize,
    f_min: f64,
    f_max: f64,
) -> Vec<f32> {
    assert!(
        n_fft >= 2 && n_mels >= 1,
        "mel_filterbank: n_fft >= 2 and n_mels >= 1 required"
    );
    let n_bins = n_fft / 2 + 1;
    let f_max = f_max.min(sample_rate / 2.0);
    let (m_lo, m_hi) = (hz_to_mel(f_min.max(0.0)), hz_to_mel(f_max));
    // n_mels + 2 edge frequencies.
    let edges: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m_lo + (m_hi - m_lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let bin_hz = |k: usize| k as f64 * sample_rate / n_fft as f64;
    let mut fb = vec![0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let (lo, c, hi) = (edges[m], edges[m + 1], edges[m + 2]);
        for k in 0..n_bins {
            let f = bin_hz(k);
            let w = if f <= lo || f >= hi {
                0.0
            } else if f <= c {
                (f - lo) / (c - lo).max(1e-12)
            } else {
                (hi - f) / (hi - c).max(1e-12)
            };
            fb[m * n_bins + k] = w as f32;
        }
    }
    fb
}

/// Chroma filterbank, `[12, n_fft/2 + 1]` row-major: each bin's energy is
/// assigned to its nearest pitch class (row 0 = C, A4 = `tuning_hz`), with a
/// Gaussian weight in semitones so bins between semitones split their
/// energy. The Gaussian is at least as wide as the bin itself, so low bins
/// that span several semitones spread their energy instead of crediting one
/// class. Bins outside `[f_min, f_max]` are ignored, and weights fade in
/// over the half octave above `f_min` (raised cosine) so a partial near the
/// edge does not switch abruptly in or out.
pub fn chroma_filterbank(
    sample_rate: f64,
    n_fft: usize,
    tuning_hz: f64,
    f_min: f64,
    f_max: f64,
) -> Vec<f32> {
    assert!(n_fft >= 2, "chroma_filterbank: n_fft >= 2 required");
    let n_bins = n_fft / 2 + 1;
    let mut fb = vec![0f32; 12 * n_bins];
    let df = sample_rate / n_fft as f64;
    for k in 1..n_bins {
        let f = k as f64 * df;
        if f < f_min || f > f_max.min(sample_rate / 2.0) {
            continue;
        }
        // Width of the per-class weight, in semitones: at least the bin's
        // own span.
        let span = 12.0 * df / (f * std::f64::consts::LN_2);
        let sigma = span.max(0.5);
        // MIDI note number (continuous).
        let midi = 69.0 + 12.0 * (f / tuning_hz).log2();
        for pc in 0..12 {
            // Distance to the nearest note of this class, in semitones.
            let d = (midi - pc as f64).rem_euclid(12.0);
            let d = d.min(12.0 - d);
            fb[pc * n_bins + k] = (-0.5 * (d / sigma).powi(2)).exp() as f32;
        }
    }
    // Each bin distributes one unit of energy across classes, faded in
    // above f_min.
    for k in 0..n_bins {
        let s: f32 = (0..12).map(|pc| fb[pc * n_bins + k]).sum();
        if s > 0.0 {
            let f = k as f64 * df;
            let t = if f_min > 0.0 {
                ((f / f_min).log2() / 0.5).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let fade = (0.5 - 0.5 * (std::f64::consts::PI * t).cos()) as f32;
            for pc in 0..12 {
                fb[pc * n_bins + k] *= fade / s;
            }
        }
    }
    fb
}

/// Fractional-octave band centres (base-2, referenced to 1 kHz, as in
/// IEC 61260): `1000 · 2^(k / per_octave)` for every `k` whose centre lies in
/// `[f_min, f_max]`.
pub fn octave_band_centres(per_octave: usize, f_min: f64, f_max: f64) -> Vec<f64> {
    assert!(
        per_octave >= 1,
        "octave_band_centres: per_octave >= 1 required"
    );
    let b = per_octave as f64;
    let k_lo = (b * (f_min.max(1e-9) / 1000.0).log2()).ceil() as i64;
    let k_hi = (b * (f_max / 1000.0).log2()).floor() as i64;
    (k_lo..=k_hi)
        .map(|k| 1000.0 * 2f64.powf(k as f64 / b))
        .collect()
}

/// Fractional-octave filterbank (1/`per_octave` octave; 3 = third-octave),
/// `[bands, n_fft/2 + 1]` row-major, bands as [`octave_band_centres`].
///
/// Band edges sit a half band either side of each centre. Each FFT bin
/// covers `[f - df/2, f + df/2]` and is shared between bands in proportion
/// to the overlap, so the bands partition the spectrum exactly: summed over
/// bands, every bin inside the covered range weighs 1, and a band narrower
/// than a bin still gets its share instead of nothing. `mm(power, fbᵀ)`
/// gives band power — the long-term spectrum used for noise colour and
/// environmental-sound comparisons.
pub fn fractional_octave_filterbank(
    sample_rate: f64,
    n_fft: usize,
    per_octave: usize,
    f_min: f64,
    f_max: f64,
) -> Vec<f32> {
    assert!(
        n_fft >= 2,
        "fractional_octave_filterbank: n_fft >= 2 required"
    );
    let n_bins = n_fft / 2 + 1;
    let df = sample_rate / n_fft as f64;
    let half = 2f64.powf(0.5 / per_octave as f64);
    let centres = octave_band_centres(per_octave, f_min, f_max.min(sample_rate / 2.0 / half));
    let mut fb = vec![0f32; centres.len() * n_bins];
    for (m, fc) in centres.iter().enumerate() {
        let (lo, hi) = (fc / half, fc * half);
        let (k0, k1) = (
            ((lo / df) - 0.5).floor().max(0.0) as usize,
            (((hi / df) + 0.5).ceil() as usize).min(n_bins - 1),
        );
        for k in k0..=k1 {
            let (a, b) = (k as f64 * df - df / 2.0, k as f64 * df + df / 2.0);
            let overlap = (b.min(hi) - a.max(lo)).max(0.0);
            fb[m * n_bins + k] = (overlap / df) as f32;
        }
    }
    fb
}

/// Transpose a row-major `[rows, cols]` matrix — for feeding a filterbank
/// to `mm(power[frames, bins], fbᵀ[bins, filters])`.
pub fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        m.len(),
        rows * cols,
        "transpose: length must equal rows*cols"
    );
    let mut t = vec![0f32; m.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_scale_reference_points() {
        // HTK: 1000 Hz ≈ 1000 mel; round trip.
        assert!((hz_to_mel(1000.0) - 999.985).abs() < 0.01);
        for hz in [0.0, 440.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1e-6);
        }
    }

    #[test]
    fn mel_filters_peak_at_their_centres_and_cover_the_band() {
        let (sr, n_fft, n_mels) = (16_000.0, 512, 40);
        let fb = mel_filterbank(sr, n_fft, n_mels, 0.0, 8000.0);
        let n_bins = n_fft / 2 + 1;
        assert_eq!(fb.len(), n_mels * n_bins);
        for m in 0..n_mels {
            let row = &fb[m * n_bins..(m + 1) * n_bins];
            let peak = row.iter().copied().fold(0f32, f32::max);
            assert!(peak > 0.5 && peak <= 1.0, "filter {m} peak {peak}");
            assert!(row.iter().all(|w| *w >= 0.0));
        }
        // Centres rise monotonically.
        let centre = |m: usize| {
            (0..n_bins)
                .max_by(|a, b| fb[m * n_bins + a].total_cmp(&fb[m * n_bins + b]))
                .unwrap()
        };
        assert!((1..n_mels).all(|m| centre(m) >= centre(m - 1)));
    }

    #[test]
    fn chroma_maps_a440_to_a() {
        let (sr, n_fft) = (48_000.0, 8192);
        let fb = chroma_filterbank(sr, n_fft, 440.0, 30.0, 5000.0);
        let n_bins = n_fft / 2 + 1;
        let bin = (440.0 * n_fft as f64 / sr).round() as usize;
        let col: Vec<f32> = (0..12).map(|pc| fb[pc * n_bins + bin]).collect();
        let best = (0..12).max_by(|a, b| col[*a].total_cmp(&col[*b])).unwrap();
        assert_eq!(best, 9, "{col:?}");
        assert!((col.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn octave_bands_partition_the_spectrum() {
        let (sr, n) = (48_000.0, 8192);
        let fb = fractional_octave_filterbank(sr, n, 3, 45.0, 16_000.0);
        let centres = octave_band_centres(3, 45.0, 16_000.0);
        let bins = n / 2 + 1;
        assert_eq!(fb.len(), centres.len() * bins);
        assert!(
            (centres[0] - 50.0).abs() < 1.0 && (centres.last().unwrap() - 16_000.0).abs() < 200.0,
            "{centres:?}"
        );
        // Inside the covered range every bin weighs exactly 1 in total.
        let df = sr / n as f64;
        for k in ((100.0 / df) as usize)..((12_000.0 / df) as usize) {
            let s: f32 = (0..centres.len()).map(|m| fb[m * bins + k]).sum();
            assert!((s - 1.0).abs() < 1e-4, "bin {k}: {s}");
        }
        // The 50 Hz band is narrower than two bins and still gets weight.
        let low: f32 = fb[..bins].iter().sum();
        assert!(low > 1.0 && low < 3.0, "50 Hz band weight {low}");
        // 1 kHz lands in the 1 kHz band.
        let k = (1000.0 / df).round() as usize;
        let m = centres
            .iter()
            .position(|c| (c - 1000.0).abs() < 1.0)
            .unwrap();
        assert!((fb[m * bins + k] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn chroma_spreads_coarse_bins_and_fades_in_at_f_min() {
        let (sr, n_fft) = (16_000.0, 2048);
        let n_bins = n_fft / 2 + 1;
        let fb = chroma_filterbank(sr, n_fft, 440.0, 50.0, 5000.0);
        let col = |k: usize| -> Vec<f32> { (0..12).map(|pc| fb[pc * n_bins + k]).collect() };
        // ~55 Hz: one bin spans over two semitones, so no class dominates.
        let low = col(7);
        let peak = low.iter().copied().fold(0.0, f32::max);
        assert!(peak < 0.5 * low.iter().sum::<f32>(), "{low:?}");
        // Fade: inside the half octave above f_min the column sums to < 1,
        // above it to 1.
        let near = col((55.0 * n_fft as f64 / sr).round() as usize)
            .iter()
            .sum::<f32>();
        let far = col((440.0 * n_fft as f64 / sr).round() as usize)
            .iter()
            .sum::<f32>();
        assert!(near > 0.0 && near < 0.5, "{near}");
        assert!((far - 1.0).abs() < 1e-5, "{far}");
        // A 440 Hz bin still reads as A.
        let a = col((440.0 * n_fft as f64 / sr).round() as usize);
        assert_eq!((0..12).max_by(|x, y| a[*x].total_cmp(&a[*y])).unwrap(), 9);
    }

    #[test]
    fn framing_and_windows() {
        let w = hann_window(8);
        assert_eq!(w[0], 0.0);
        assert!((w[4] - 1.0).abs() < 1e-6);
        let x: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let f = frame_signal(&x, 8, 4, None);
        assert_eq!(frame_count(20, 8, 4), 4);
        assert_eq!(f.len(), 4 * 8);
        assert_eq!(&f[8..12], &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(frame_signal(&x[..3], 8, 4, None).len(), 8);
        let t = transpose(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
