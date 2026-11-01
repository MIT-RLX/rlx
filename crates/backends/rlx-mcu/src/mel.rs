// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NeMo `AudioToMelSpectrogramPreprocessor` (Conformer-CTC, Citrinet, …),
//! with the model's own window and filterbank: 0.97 pre-emphasis, centered zero-padded STFT, power
//! spectrum, mel filterbank, `ln(x + 2^-24)`, then `per_feature`
//! normalization over the frames that hold audio.
//!
//! The model supplies its window and banded filterbank through
//! [`MelSource`].
//!
//! The FFT is an in-place radix-2 over fixed arrays, so the frontend needs no
//! allocator; `n_fft` up to [`MAX_FFT`].

use crate::Pcm;

/// What [`Frontend::log_mel`] needs from a model.
pub trait MelSource {
    /// STFT hop in samples.
    fn hop(&self) -> usize;
    /// Mel bins.
    fn n_mels(&self) -> usize;
    /// STFT window, centered in `n_fft`.
    fn window(&self) -> &[f32];
    /// Mel filter `m` as `(first FFT bin, weights)`.
    fn mel_band(&self, m: usize) -> (usize, &[f32]);
}

/// Largest supported FFT size.
pub const MAX_FFT: usize = 512;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8;
const NORM_EPS: f32 = 1e-5;

/// FFT tables for one `n_fft`. About 4 KiB; build once.
pub struct Frontend {
    n_fft: usize,
    cos: [f32; MAX_FFT / 2],
    sin: [f32; MAX_FFT / 2],
    bitrev: [u16; MAX_FFT],
}

impl Frontend {
    /// Tables for `n_fft`, which must be a power of two `<= MAX_FFT`.
    pub fn new(n_fft: usize) -> Self {
        assert!(n_fft.is_power_of_two() && n_fft <= MAX_FFT);
        let mut f = Self {
            n_fft,
            cos: [0.0; MAX_FFT / 2],
            sin: [0.0; MAX_FFT / 2],
            bitrev: [0; MAX_FFT],
        };
        for k in 0..n_fft / 2 {
            let a = -2.0 * core::f64::consts::PI * k as f64 / n_fft as f64;
            f.cos[k] = libm::cos(a) as f32;
            f.sin[k] = libm::sin(a) as f32;
        }
        let bits = n_fft.trailing_zeros();
        for i in 0..n_fft {
            f.bitrev[i] = ((i as u32).reverse_bits() >> (32 - bits)) as u16;
        }
        f
    }

    /// In-place complex FFT of length `n_fft` (radix-2, bit-reversed tables).
    pub fn fft(&self, re: &mut [f32], im: &mut [f32]) {
        let n = self.n_fft;
        for i in 0..n {
            let j = self.bitrev[i] as usize;
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let (wr, wi) = (self.cos[k * step], self.sin[k * step]);
                    let (a, b) = (start + k, start + k + half);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len *= 2;
        }
    }

    /// Mel frames for `n_samples` of audio (`1 + n / hop`, center padding).
    pub fn frames<M: MelSource + ?Sized>(model: &M, n_samples: usize) -> usize {
        1 + n_samples / model.hop()
    }

    /// Scratch `log_mel` needs besides its output: `2 * n_fft` values.
    pub fn work_len(&self) -> usize {
        2 * self.n_fft
    }

    /// Log-mel features, **time-major** `[frames, n_mels]`, into `out`.
    /// Returns the frame count.
    pub fn log_mel<M: MelSource + ?Sized, P: Pcm + ?Sized>(
        &self,
        model: &M,
        pcm: &P,
        out: &mut [f32],
        work: &mut [f32],
    ) -> usize {
        let (n_fft, hop, n_mels) = (self.n_fft, model.hop(), model.n_mels());
        let n = pcm.len();
        let frames = Self::frames(model, n);
        let window = model.window();
        let (re, im) = work[..2 * n_fft].split_at_mut(n_fft);
        let half = n_fft / 2;

        // The pre-emphasized signal, `y[t] = x[t] - 0.97 x[t-1]`, at
        // `t = i - n_fft/2` (center padding), zero outside the audio.
        let sample = |i: usize| -> f32 {
            if i < half || i - half >= n {
                return 0.0;
            }
            let t = i - half;
            let x = pcm.at(t);
            if t == 0 {
                x
            } else {
                x - PREEMPH * pcm.at(t - 1)
            }
        };

        for f in 0..frames {
            let start = f * hop;
            for j in 0..n_fft {
                re[j] = sample(start + j) * window[j];
                im[j] = 0.0;
            }
            self.fft(re, im);
            // Power spectrum into the first n_fft/2+1 slots of `re`.
            for b in 0..=half {
                re[b] = re[b] * re[b] + im[b] * im[b];
            }
            let row = &mut out[f * n_mels..(f + 1) * n_mels];
            for (m, v) in row.iter_mut().enumerate() {
                let (start, w) = model.mel_band(m);
                let mut acc = 0.0f32;
                for (k, &wk) in w.iter().enumerate() {
                    acc += wk * re[start + k];
                }
                *v = libm::logf(acc + LOG_GUARD);
            }
        }

        // Statistics over the frames that hold audio (NeMo `get_seq_len`),
        // applied to all; the rest are zeroed, as the featurizer masks them.
        let valid = (n / hop).clamp(1, frames);
        for m in 0..n_mels {
            let mut sum = 0.0f32;
            for f in 0..valid {
                sum += out[f * n_mels + m];
            }
            let mean = sum / valid as f32;
            let mut var = 0.0f32;
            if valid > 1 {
                for f in 0..valid {
                    let d = out[f * n_mels + m] - mean;
                    var += d * d;
                }
                var /= valid as f32 - 1.0;
            }
            let denom = libm::sqrtf(var) + NORM_EPS;
            for f in 0..frames {
                let v = &mut out[f * n_mels + m];
                *v = if f < valid { (*v - mean) / denom } else { 0.0 };
            }
        }
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_dft() {
        let fe = Frontend::new(16);
        let x: [f32; 16] = core::array::from_fn(|i| ((i * 7 % 5) as f32) - 2.0);
        let mut re = x;
        let mut im = [0.0f32; 16];
        fe.fft(&mut re, &mut im);
        for k in 0..16 {
            let (mut r, mut i) = (0.0f64, 0.0f64);
            for (t, &v) in x.iter().enumerate() {
                let a = -2.0 * core::f64::consts::PI * (k * t) as f64 / 16.0;
                r += v as f64 * a.cos();
                i += v as f64 * a.sin();
            }
            assert!((re[k] as f64 - r).abs() < 1e-4 && (im[k] as f64 - i).abs() < 1e-4);
        }
    }
}
