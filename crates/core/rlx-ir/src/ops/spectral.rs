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

//! Spectral front-ends for EEG models — windowed spectrogram, per-band power,
//! and per-band differential entropy — composed on the existing `rfft` helper.
//!
//! These are the tokenizer/feature stages that BrainBERT, BIOT,
//! SleepTransformer, CBraMod, Uni-NTFM, FoME and MAET currently compute
//! host-side in plain Rust. Expressed as graph subtrees they run on-device and
//! are differentiable.
//!
//! Because `rfft` zero-pads the frame to the next power of two, the frequency
//! axis has `next_pow2(frame_len)/2 + 1` bins.

use crate::infer::GraphExt as _;
use crate::op::Activation;
use crate::{DType, Graph, NodeId, fft::FftNorm};

/// Analysis window applied to each frame before the FFT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowKind {
    /// No window (boxcar).
    Rectangular,
    /// Hann (a.k.a. Hanning) — the usual EEG default.
    Hann,
    /// Hamming.
    Hamming,
    /// Blackman.
    Blackman,
}

impl WindowKind {
    fn coeffs(self, n: usize) -> Vec<f32> {
        use std::f32::consts::PI;
        if n == 1 {
            return vec![1.0];
        }
        let d = (n - 1) as f32;
        (0..n)
            .map(|i| {
                let t = i as f32 / d;
                match self {
                    WindowKind::Rectangular => 1.0,
                    WindowKind::Hann => 0.5 - 0.5 * (2.0 * PI * t).cos(),
                    WindowKind::Hamming => 0.54 - 0.46 * (2.0 * PI * t).cos(),
                    WindowKind::Blackman => {
                        0.42 - 0.5 * (2.0 * PI * t).cos() + 0.08 * (4.0 * PI * t).cos()
                    }
                }
            })
            .collect()
    }
}

impl Graph {
    /// Windowed short-time spectrogram of the last axis.
    ///
    /// `[..., T]` → `[n_frames, ..., n_bins]` where
    /// `n_frames = 1 + (T − frame_len)/hop` and `n_bins = next_pow2(frame_len)/2 + 1`.
    ///
    /// * `power` — return `|X|²` (true) or magnitude `|X|` (false).
    /// * `log`   — apply `log(· + 1e-8)` to the result (log-power / log-magnitude).
    pub fn spectrogram(
        &mut self,
        x: NodeId,
        frame_len: usize,
        hop: usize,
        window: WindowKind,
        power: bool,
        log: bool,
    ) -> NodeId {
        assert!(frame_len > 0 && hop > 0, "spectrogram: frame_len/hop > 0");
        let shape = self.shape(x).clone();
        let last = shape.rank() - 1;
        let t = shape.dim(last).unwrap_static();
        assert!(t >= frame_len, "spectrogram: T {t} < frame_len {frame_len}");
        let n_frames = 1 + (t - frame_len) / hop;

        // Stack time-domain frames into [n_frames, ..., frame_len].
        let mut rows = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let start = f * hop;
            let frame = self.narrow_(x, last, start, frame_len);
            let mut dims: Vec<i64> = self
                .shape(frame)
                .dims()
                .iter()
                .map(|d| d.unwrap_static() as i64)
                .collect();
            dims.insert(0, 1);
            rows.push(self.reshape_(frame, dims));
        }
        let framed = if rows.len() == 1 {
            rows.pop().unwrap()
        } else {
            self.concat_(rows, 0)
        };

        // Apply the analysis window (broadcast over the frame axis).
        let framed = if window == WindowKind::Rectangular {
            framed
        } else {
            let w = self.const_f32_tensor(window.coeffs(frame_len), &[frame_len]);
            self.mul(framed, w)
        };

        let (re, im) = self.rfft(framed, FftNorm::Backward);
        let re2 = self.mul(re, re);
        let im2 = self.mul(im, im);
        let mag2 = self.add(re2, im2);
        let out = if power { mag2 } else { self.sqrt(mag2) };
        if log { self.log_eps(out, 1e-8) } else { out }
    }

    /// Per-band power of the last axis via a single `rfft`.
    ///
    /// `[..., T]` → `[..., n_bands]`. Each band `(lo, hi)` in Hz sums the
    /// (unnormalized) power `|X|²` over the rFFT bins whose center frequency
    /// falls in `[lo, hi]`, using `sample_rate` to map Hz → bin index.
    pub fn band_power(&mut self, x: NodeId, sample_rate: f32, bands: &[(f32, f32)]) -> NodeId {
        assert!(!bands.is_empty(), "band_power: need ≥1 band");
        let shape = self.shape(x).clone();
        let last = shape.rank() - 1;
        let t = shape.dim(last).unwrap_static();
        let n_pad = crate::fft::next_pow2(t);
        let n_bins = n_pad / 2 + 1;

        let (re, im) = self.rfft(x, FftNorm::Backward);
        let re2 = self.mul(re, re);
        let im2 = self.mul(im, im);
        let power = self.add(re2, im2); // [.., n_bins]
        let plast = self.shape(power).rank() - 1;

        let hz_per_bin = sample_rate / n_pad as f32;
        let mut cols = Vec::with_capacity(bands.len());
        for &(lo, hi) in bands {
            let mut b0 = (lo / hz_per_bin).ceil() as isize;
            let mut b1 = (hi / hz_per_bin).floor() as isize;
            b0 = b0.clamp(0, n_bins as isize - 1);
            b1 = b1.clamp(0, n_bins as isize - 1);
            let (b0, b1) = if b1 < b0 { (b0, b0) } else { (b0, b1) };
            let seg = self.narrow_(power, plast, b0 as usize, (b1 - b0 + 1) as usize);
            cols.push(self.sum(seg, vec![plast], true)); // [.., 1]
        }
        if cols.len() == 1 {
            cols.pop().unwrap()
        } else {
            self.concat_(cols, plast)
        }
    }

    /// Per-band differential entropy: `0.5·log(2πe · band_power + 1e-8)`.
    ///
    /// For a band-limited Gaussian signal this equals its differential entropy
    /// up to the constant — the classic EEG "DE" feature (SEED, MAET, FoME).
    /// Returns `[..., n_bands]`.
    pub fn differential_entropy(
        &mut self,
        x: NodeId,
        sample_rate: f32,
        bands: &[(f32, f32)],
    ) -> NodeId {
        let bp = self.band_power(x, sample_rate, bands);
        // 2πe
        let c = self.constant((2.0 * std::f64::consts::PI * std::f64::consts::E) as f64, DType::F32);
        let scaled = self.mul(bp, c);
        let logv = self.log_eps(scaled, 1e-8);
        let half = self.constant(0.5, DType::F32);
        self.mul(logv, half)
    }

    fn log_eps(&mut self, x: NodeId, eps: f32) -> NodeId {
        let e = self.constant(eps as f64, DType::F32);
        let shifted = self.add(x, e);
        let s = crate::shape::unary_shape(self.shape(shifted));
        self.activation(Activation::Log, shifted, s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    fn dims(g: &Graph, id: NodeId) -> Vec<usize> {
        g.shape(id).dims().iter().map(|d| d.unwrap_static()).collect()
    }
    fn input(g: &mut Graph, shape: &[usize]) -> NodeId {
        g.input("x", Shape::new(shape, DType::F32))
    }

    #[test]
    fn spectrogram_shape() {
        let mut g = Graph::new("spec");
        let x = input(&mut g, &[2, 256]); // [C, T]
        // frame_len=64 (pow2) → n_bins = 64/2+1 = 33; n_frames = 1+(256-64)/32 = 7
        let y = g.spectrogram(x, 64, 32, WindowKind::Hann, true, true);
        assert_eq!(dims(&g, y), vec![7, 2, 33]);
    }

    #[test]
    fn band_power_shape() {
        let mut g = Graph::new("bp");
        let x = input(&mut g, &[4, 512]); // [C, T]
        let bands = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)];
        let y = g.band_power(x, 128.0, &bands);
        assert_eq!(dims(&g, y), vec![4, 5]);
    }

    #[test]
    fn differential_entropy_shape() {
        let mut g = Graph::new("de");
        let x = input(&mut g, &[4, 512]);
        let bands = [(1.0, 4.0), (4.0, 8.0), (8.0, 14.0)];
        let y = g.differential_entropy(x, 200.0, &bands);
        assert_eq!(dims(&g, y), vec![4, 3]);
    }
}
