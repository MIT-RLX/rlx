//! Mel scales and triangular filterbanks (`[n_mels, n_fft/2 + 1]`, mel-major),
//! built in f64 and rounded once into the working precision.

use crate::float::Float;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MelScale {
    /// `2595·log10(1 + f/700)`
    Htk,
    /// Kaldi: `1127·ln(1 + f/700)`
    Kaldi,
    /// Slaney / librosa (`htk=False`): linear below 1 kHz, logarithmic above.
    Slaney,
}

impl MelScale {
    pub fn hz_to_mel(self, hz: f64) -> f64 {
        match self {
            MelScale::Htk => 2595.0 * libm::log10(1.0 + hz / 700.0),
            MelScale::Kaldi => 1127.0 * libm::log(1.0 + hz / 700.0),
            MelScale::Slaney => {
                let f_sp = 200.0 / 3.0;
                let min_log_hz = 1000.0;
                let min_log_mel = min_log_hz / f_sp;
                let logstep = libm::log(6.4) / 27.0;
                if hz >= min_log_hz {
                    min_log_mel + libm::log(hz / min_log_hz) / logstep
                } else {
                    hz / f_sp
                }
            }
        }
    }

    pub fn mel_to_hz(self, mel: f64) -> f64 {
        match self {
            MelScale::Htk => 700.0 * (libm::pow(10.0, mel / 2595.0) - 1.0),
            MelScale::Kaldi => 700.0 * (libm::exp(mel / 1127.0) - 1.0),
            MelScale::Slaney => {
                let f_sp = 200.0 / 3.0;
                let min_log_hz = 1000.0;
                let min_log_mel = min_log_hz / f_sp;
                let logstep = libm::log(6.4) / 27.0;
                if mel >= min_log_mel {
                    min_log_hz * libm::exp(logstep * (mel - min_log_mel))
                } else {
                    mel * f_sp
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MelNorm {
    None,
    /// Divide each triangle by its width in Hz (`2 / (f[m+2] − f[m])`).
    Slaney,
}

/// A mel filterbank definition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MelBank {
    pub sample_rate: f64,
    pub n_fft: usize,
    pub n_mels: usize,
    pub f_min: f64,
    pub f_max: f64,
    pub scale: MelScale,
    pub norm: MelNorm,
    /// Triangles in mel space (Kaldi / HF `triangularize_in_mel_space`)
    /// rather than over Hz (librosa).
    pub in_mel_space: bool,
}

impl MelBank {
    /// Whisper / librosa: Slaney scale and norm, triangles over Hz, 0 Hz..Nyquist.
    pub fn whisper(sample_rate: f64, n_fft: usize, n_mels: usize) -> Self {
        MelBank {
            sample_rate,
            n_fft,
            n_mels,
            f_min: 0.0,
            f_max: sample_rate / 2.0,
            scale: MelScale::Slaney,
            norm: MelNorm::Slaney,
            in_mel_space: false,
        }
    }

    /// Kaldi fbank / SeamlessM4T: Kaldi scale, no norm, triangles in mel space.
    pub fn kaldi(sample_rate: f64, n_fft: usize, n_mels: usize, f_min: f64) -> Self {
        MelBank {
            sample_rate,
            n_fft,
            n_mels,
            f_min,
            f_max: sample_rate / 2.0,
            scale: MelScale::Kaldi,
            norm: MelNorm::None,
            in_mel_space: true,
        }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Dense weights `[n_mels, bins]`.
    pub fn weights<T: Float>(&self) -> Vec<T> {
        let nb = self.bins();
        let (lo, hi) = (
            self.scale.hz_to_mel(self.f_min),
            self.scale.hz_to_mel(self.f_max),
        );
        // n_mels + 2 equally spaced mel points (numpy linspace: lo + i·step, end exact)
        let pts: Vec<f64> = (0..self.n_mels + 2)
            .map(|i| {
                if i == self.n_mels + 1 {
                    hi
                } else {
                    lo + (hi - lo) * i as f64 / (self.n_mels + 1) as f64
                }
            })
            .collect();
        let hz_pts: Vec<f64> = pts.iter().map(|&m| self.scale.mel_to_hz(m)).collect();
        let bin_hz = self.sample_rate / self.n_fft as f64;
        let mut w = vec![T::ZERO; self.n_mels * nb];
        for m in 0..self.n_mels {
            let (l, c, r) = if self.in_mel_space {
                (pts[m], pts[m + 1], pts[m + 2])
            } else {
                (hz_pts[m], hz_pts[m + 1], hz_pts[m + 2])
            };
            let enorm = match self.norm {
                MelNorm::None => 1.0,
                MelNorm::Slaney => 2.0 / (hz_pts[m + 2] - hz_pts[m]),
            };
            for b in 0..nb {
                let f = b as f64 * bin_hz;
                let x = if self.in_mel_space {
                    self.scale.hz_to_mel(f)
                } else {
                    f
                };
                let down = (x - l) / (c - l);
                let up = (r - x) / (r - c);
                let v = down.min(up).max(0.0) * enorm;
                w[m * nb + b] = T::from_f64(v);
            }
        }
        w
    }
}

/// A filterbank with each row stored only over its nonzero span: applying it
/// costs the bank's support, not `n_mels × bins`.
#[derive(Clone, Debug)]
pub struct SparseBank<T> {
    pub n_mels: usize,
    pub bins: usize,
    offset: Vec<usize>,
    span: Vec<usize>,
    weight: Vec<T>,
}

impl<T: Float> SparseBank<T> {
    pub fn from_dense(w: &[T], n_mels: usize, bins: usize) -> Self {
        let (mut offset, mut span, mut weight) = (Vec::with_capacity(n_mels), vec![0], Vec::new());
        for m in 0..n_mels {
            let row = &w[m * bins..(m + 1) * bins];
            let lo = row.iter().position(|v| *v != T::ZERO).unwrap_or(bins);
            let hi = row
                .iter()
                .rposition(|v| *v != T::ZERO)
                .map_or(lo, |i| i + 1);
            offset.push(lo);
            weight.extend_from_slice(&row[lo..hi]);
            span.push(weight.len());
        }
        SparseBank {
            n_mels,
            bins,
            offset,
            span,
            weight,
        }
    }

    pub fn new(bank: &MelBank) -> Self {
        Self::from_dense(&bank.weights::<T>(), bank.n_mels, bank.bins())
    }

    /// `out[m] = Σ_b power[b]·w[m][b]`, in bin order.
    pub fn apply(&self, power: &[T], out: &mut [T]) {
        for (m, o) in out.iter_mut().enumerate() {
            let w = &self.weight[self.span[m]..self.span[m + 1]];
            let p = &power[self.offset[m]..self.offset[m] + w.len()];
            let mut acc = T::ZERO;
            for (&pk, &wk) in p.iter().zip(w) {
                acc = pk.mul_add(wk, acc);
            }
            *o = acc;
        }
    }
}
