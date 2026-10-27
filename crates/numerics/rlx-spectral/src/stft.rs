//! Framing a signal for a short-time transform, and power spectrograms.

use crate::fft::RealFftPlan;
use crate::float::Float;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pad {
    /// `center=True`, `np.pad(mode="reflect")` by `n_fft/2` on both sides.
    CenterReflect,
    /// No padding; frames of `frame_len` fully inside the signal (Kaldi
    /// `snip_edges=True`).
    None,
}

/// Per-frame conditioning before the window (Kaldi order: DC removal, then
/// preemphasis `x[i] −= k·x[i−1]`, `x[0] *= 1 − k`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct FramePrep {
    pub remove_dc: bool,
    pub preemphasis: Option<f64>,
    /// Multiply the signal by this first (Kaldi's int16 scale, 2^15).
    pub input_scale: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Stft<T> {
    pub frame_len: usize,
    pub hop: usize,
    pub n_fft: usize,
    pub pad: Pad,
    pub prep: FramePrep,
    window: Vec<T>,
    plan: RealFftPlan<T>,
}

impl<T: Float> Stft<T> {
    /// `window.len()` is the frame length (zero-padded to `n_fft`).
    pub fn new(window: Vec<T>, hop: usize, n_fft: usize, pad: Pad, prep: FramePrep) -> Self {
        assert!(window.len() <= n_fft && hop > 0);
        Stft {
            frame_len: window.len(),
            hop,
            n_fft,
            pad,
            prep,
            window,
            plan: RealFftPlan::new(n_fft),
        }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// The padded signal the frames are read from.
    fn padded(&self, x: &[T]) -> Vec<T> {
        let scale = self.prep.input_scale.map(T::from_f64);
        let sc = |v: T| scale.map_or(v, |s| v * s);
        match self.pad {
            Pad::None => x.iter().map(|&v| sc(v)).collect(),
            Pad::CenterReflect => {
                let p = self.n_fft / 2;
                let last = x.len().saturating_sub(1);
                let mut out = Vec::with_capacity(x.len() + 2 * p);
                for i in (1..=p).rev() {
                    out.push(sc(x[i.min(last)]));
                }
                out.extend(x.iter().map(|&v| sc(v)));
                for i in 1..=p {
                    out.push(sc(x[last.saturating_sub(i)]));
                }
                out
            }
        }
    }

    pub fn n_frames(&self, signal_len: usize) -> usize {
        let len = match self.pad {
            Pad::None => signal_len,
            Pad::CenterReflect => signal_len + 2 * (self.n_fft / 2),
        };
        if len < self.frame_len {
            0
        } else {
            1 + (len - self.frame_len) / self.hop
        }
    }

    /// Power spectrogram `|X|²`, `[frames, bins]` row-major, handed to `sink`
    /// one frame at a time (no full spectrogram buffer).
    pub fn power_frames(&self, x: &[T], mut sink: impl FnMut(usize, &[T])) {
        let sig = self.padded(x);
        let nf = self.n_frames(x.len());
        let mut frame = vec![T::ZERO; self.n_fft];
        let mut spec = vec![Default::default(); self.bins()];
        let mut scratch = self.plan.make_scratch();
        let mut power = vec![T::ZERO; self.bins()];
        let fl = self.frame_len;
        for f in 0..nf {
            let src = &sig[f * self.hop..f * self.hop + fl];
            frame[..fl].copy_from_slice(src);
            if self.prep.remove_dc {
                let mut s = T::ZERO;
                for &v in &frame[..fl] {
                    s += v;
                }
                let mean = s / T::from_f64(fl as f64);
                frame[..fl].iter_mut().for_each(|v| *v -= mean);
            }
            if let Some(k) = self.prep.preemphasis {
                let k = T::from_f64(k);
                for i in (1..fl).rev() {
                    frame[i] = (-k).mul_add(frame[i - 1], frame[i]);
                }
                frame[0] *= T::ONE - k;
            }
            for (v, &w) in frame[..fl].iter_mut().zip(&self.window) {
                *v *= w;
            }
            frame[fl..].iter_mut().for_each(|v| *v = T::ZERO);
            self.plan.forward_with(&frame, &mut spec, &mut scratch);
            for (p, c) in power.iter_mut().zip(&spec) {
                *p = c.norm_sqr();
            }
            sink(f, &power);
        }
    }
}
