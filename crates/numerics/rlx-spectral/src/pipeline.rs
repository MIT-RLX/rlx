//! Complete front-ends.

use crate::float::Float;
use crate::mel::{MelBank, SparseBank};
use crate::stft::{FramePrep, Pad, Stft};
use crate::window::Window;

/// Whisper's log-mel spectrogram (`whisper.audio.log_mel_spectrogram`):
/// 16 kHz, n_fft 400, hop 160, periodic Hann, centered reflect padding,
/// power, Slaney mel, `log10(max(·, 1e-10))`, floor at `max − 8`,
/// `(x + 4) / 4`. Returns `[n_mels, frames]` row-major and `frames`
/// (`1 + len/160`; whisper itself drops the last frame).
#[derive(Clone, Debug)]
pub struct WhisperLogMel<T> {
    stft: Stft<T>,
    bank: SparseBank<T>,
    pub n_mels: usize,
}

impl<T: Float> WhisperLogMel<T> {
    pub const N_FFT: usize = 400;
    pub const HOP: usize = 160;
    pub const SAMPLE_RATE: f64 = 16_000.0;

    pub fn new(n_mels: usize) -> Self {
        let stft = Stft::new(
            Window::HannPeriodic.build(Self::N_FFT),
            Self::HOP,
            Self::N_FFT,
            Pad::CenterReflect,
            FramePrep::default(),
        );
        let bank = SparseBank::new(&MelBank::whisper(Self::SAMPLE_RATE, Self::N_FFT, n_mels));
        WhisperLogMel { stft, bank, n_mels }
    }

    /// The dense Slaney filters, `[n_mels, 201]`.
    pub fn filters(&self) -> Vec<T> {
        MelBank::whisper(Self::SAMPLE_RATE, Self::N_FFT, self.n_mels).weights()
    }

    pub fn compute(&self, pcm: &[T]) -> (Vec<T>, usize) {
        let nf = self.stft.n_frames(pcm.len());
        (self.compute_frames(pcm, nf), nf)
    }

    /// Exactly `n_frames` columns: frames past the signal have zero power,
    /// frames past `n_frames` are dropped, and the `max − 8` floor is taken
    /// over these columns (what whisper does for its padded 30 s window).
    pub fn compute_frames(&self, pcm: &[T], n_frames: usize) -> Vec<T> {
        let n_mels = self.n_mels;
        let nf = n_frames;
        let tiny = T::from_f64(1e-10);
        let mut out = vec![tiny; n_mels * nf];
        let mut row = vec![T::ZERO; n_mels];
        self.stft.power_frames(pcm, |f, power| {
            if f >= nf {
                return;
            }
            self.bank.apply(power, &mut row);
            for (m, &v) in row.iter().enumerate() {
                out[m * nf + f] = v.max(tiny);
            }
        });
        T::log10_slice(&mut out);
        let mut mx = T::from_f64(f64::NEG_INFINITY);
        for &v in &out {
            mx = mx.max(v);
        }
        let floor = mx - T::from_f64(8.0);
        let (four, quarter) = (T::from_f64(4.0), T::from_f64(0.25));
        for v in out.iter_mut() {
            *v = (v.max(floor) + four) * quarter;
        }
        out
    }
}

/// Kaldi-style log-mel filterbank as HF's `SeamlessM4TFeatureExtractor`
/// computes it (w2v-BERT 2.0): int16 scale, 25 ms frames / 10 ms hop
/// (`snip_edges`), DC removal, preemphasis 0.97, Povey window, 512-point FFT,
/// power, 80 Kaldi mel bins from 20 Hz, `ln(max(·, f32::EPSILON))`.
#[derive(Clone, Debug)]
pub struct KaldiFbank<T> {
    stft: Stft<T>,
    bank: SparseBank<T>,
    pub n_mels: usize,
}

impl<T: Float> KaldiFbank<T> {
    /// SeamlessM4T's configuration: 80 mel bins.
    pub fn seamless_m4t(sample_rate: f64) -> Self {
        Self::new(sample_rate, 80)
    }

    pub fn new(sample_rate: f64, n_mels: usize) -> Self {
        let frame = (0.025 * sample_rate).round() as usize;
        let hop = (0.010 * sample_rate).round() as usize;
        let n_fft = frame.next_power_of_two();
        let prep = FramePrep {
            remove_dc: true,
            preemphasis: Some(0.97),
            input_scale: Some(32768.0),
        };
        let stft = Stft::new(Window::Povey.build(frame), hop, n_fft, Pad::None, prep);
        let bank = SparseBank::new(&MelBank::kaldi(sample_rate, n_fft, n_mels, 20.0));
        KaldiFbank { stft, bank, n_mels }
    }

    /// `[frames, n_mels]` row-major and `frames`.
    pub fn compute(&self, wave: &[T]) -> (Vec<T>, usize) {
        let nf = self.stft.n_frames(wave.len());
        let n_mels = self.n_mels;
        let mut out = vec![T::ZERO; nf * n_mels];
        let floor = T::from_f64(f32::EPSILON as f64);
        self.stft.power_frames(wave, |f, power| {
            let row = &mut out[f * n_mels..(f + 1) * n_mels];
            self.bank.apply(power, row);
            row.iter_mut().for_each(|v| *v = v.max(floor));
        });
        T::ln_slice(&mut out);
        (out, nf)
    }

    /// SeamlessM4T `input_features`: per-bin `(x − mean)/sqrt(var(ddof=1) + 1e-7)`,
    /// zero-padded to an even frame count, frames stacked in pairs:
    /// `[ceil(frames/2), 2·n_mels]` and its row count.
    pub fn input_features(&self, wave: &[T]) -> (Vec<T>, usize) {
        self.input_features_stacked(wave, 2)
    }

    /// [`input_features`](Self::input_features) with `stride` frames stacked
    /// per row: `[ceil(frames/stride), stride·n_mels]`.
    pub fn input_features_stacked(&self, wave: &[T], stride: usize) -> (Vec<T>, usize) {
        let stride = stride.max(1);
        let (mut x, t) = self.compute(wave);
        if t == 0 {
            return (Vec::new(), 0);
        }
        let m = self.n_mels;
        for b in 0..m {
            let mut s = T::ZERO;
            for f in 0..t {
                s += x[f * m + b];
            }
            let mean = s / T::from_f64(t as f64);
            let mut ss = T::ZERO;
            for f in 0..t {
                let d = x[f * m + b] - mean;
                ss = d.mul_add(d, ss);
            }
            let var = ss / T::from_f64((t - 1).max(1) as f64);
            let inv = T::ONE / (var + T::from_f64(1e-7)).sqrt();
            for f in 0..t {
                let v = &mut x[f * m + b];
                *v = (*v - mean) * inv;
            }
        }
        let padded = t.div_ceil(stride) * stride;
        x.resize(padded * m, T::ZERO);
        (x, padded / stride)
    }
}
