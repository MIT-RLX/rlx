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

//! Analytic-signal and zero-phase-filtering DSP helpers, composed on the
//! existing FFT primitives.
//!
//! * [`Graph::hilbert`] — the analytic signal `x + i·H(x)` (scipy
//!   `signal.hilbert`), the basis for amplitude envelope and instantaneous
//!   phase used in EEG band-power / phase-amplitude-coupling features.
//! * [`Graph::envelope`] / [`Graph::instantaneous_phase`] — magnitude and
//!   angle of the analytic signal.
//! * [`Graph::fir_filtfilt`] — zero-phase FIR filtering (forward + reversed
//!   pass), the parity-preserving band-pass used in the `exg` preprocessor.
//!
//! `hilbert` zero-pads the last axis to the next power of two (matching
//! `rfft`) and truncates back, so for non-pow2 lengths it is the analytic
//! signal of the zero-padded frame — the same approximation as calling
//! `scipy.signal.hilbert` on a padded buffer.

use crate::infer::GraphExt as _;
use crate::op::Activation;
use crate::{DType, Graph, NodeId, Op, Shape, fft::FftNorm};

impl Graph {
    /// Analytic signal of the last axis. Returns `(real, imag)`, each the same
    /// shape as `x`; `real ≈ x` and `imag = H(x)` (the Hilbert transform).
    pub fn hilbert(&mut self, x: NodeId) -> (NodeId, NodeId) {
        let shape = self.shape(x).clone();
        let last = shape.rank() - 1;
        let l = shape.dim(last).unwrap_static();
        let n = crate::fft::next_pow2(l);

        // Unnormalized forward DFT of the (real) signal, full n-bin spectrum.
        let (re, im) = self.fft_real(x, FftNorm::Backward); // each [.., n]

        // Hilbert multiplier h: DC×1, positive freqs ×2, Nyquist×1, negatives ×0.
        let mut h = vec![0f32; n];
        h[0] = 1.0;
        if n >= 2 {
            h[n / 2] = 1.0;
            for hk in h.iter_mut().take(n / 2).skip(1) {
                *hk = 2.0;
            }
        }
        let h_node = self.const_f32_tensor(h, &[n]);
        let re_h = self.mul(re, h_node);
        let im_h = self.mul(im, h_node);

        // Inverse DFT with 1/N scaling (numpy `ifft`), keeping both parts.
        let block = self.concat_(vec![re_h, im_h], last);
        let full = self.fft_norm(block, true, FftNorm::Forward); // [.., 2n]
        let a_re = self.narrow_(full, last, 0, n);
        let a_im = self.narrow_(full, last, n, n);
        // Truncate the pow2 padding back to the original length.
        (
            self.narrow_(a_re, last, 0, l),
            self.narrow_(a_im, last, 0, l),
        )
    }

    /// Amplitude envelope `|x + i·H(x)|` of the last axis (same shape as `x`).
    pub fn envelope(&mut self, x: NodeId) -> NodeId {
        let (re, im) = self.hilbert(x);
        self.complex_abs(re, im)
    }

    /// Instantaneous phase `atan2(H(x), x)` of the last axis, in radians.
    ///
    /// Uses the branch-cut-free identity
    /// `atan2(y, x) = 2·atan(y / (√(x²+y²) + x))`, exact everywhere except the
    /// negative real axis (a measure-zero set).
    pub fn instantaneous_phase(&mut self, x: NodeId) -> NodeId {
        let (re, im) = self.hilbert(x);
        let r = self.complex_abs(re, im);
        let denom = self.add(r, re);
        let eps = self.constant(1e-12, DType::F32);
        let denom = self.add(denom, eps);
        let ratio = self.div(im, denom);
        let s = crate::shape::unary_shape(self.shape(ratio));
        let at = self.activation(Activation::Atan, ratio, s);
        let two = self.constant(2.0, DType::F32);
        self.mul(at, two)
    }

    /// Zero-phase FIR filtering: `reverse(fir(reverse(fir(x))))`.
    ///
    /// `taps` are the FIR coefficients (host constant). Output has the same
    /// last-axis length as `x`; the two passes cancel the filter's phase, so
    /// the effective magnitude response is `|H(f)|²`. This is the zero-phase
    /// band-pass the `exg` preprocessor uses for MNE parity.
    pub fn fir_filtfilt(&mut self, x: NodeId, taps: &[f32]) -> NodeId {
        let k = taps.len();
        assert!(k > 0, "fir_filtfilt: need ≥1 tap");
        let last = self.shape(x).rank() - 1;
        let l = self.shape(x).dim(last).unwrap_static();
        let h = self.const_f32_tensor(taps.to_vec(), &[k]);
        let y1 = self.fir_conv_same(x, h, k, l);
        let y2 = self.reverse(y1, vec![last]);
        let y3 = self.fir_conv_same(y2, h, k, l);
        self.reverse(y3, vec![last])
    }

    /// Single-pass "same"-length FIR convolution along the last axis.
    ///
    /// `FftNorm::Forward` normalizes the inverse by `1/N`, so the FFT
    /// convolution theorem yields the un-scaled linear convolution (an
    /// unnormalized round-trip would scale by `n_fft`).
    fn fir_conv_same(&mut self, x: NodeId, taps: NodeId, k: usize, l: usize) -> NodeId {
        let full = self.fft_conv1d(x, taps, 0, FftNorm::Forward); // [.., l+k-1]
        let start = (k - 1) / 2;
        let last = self.shape(full).rank() - 1;
        self.narrow_(full, last, start, l)
    }

    /// One biquad (2nd-order IIR) section applied along the last axis,
    /// Direct-Form II Transposed. `b = [b0, b1, b2]`, `a = [a0, a1, a2]`
    /// (coefficients are normalized by `a0`). This is a true recurrence —
    /// `y[n] = b0·x[n] + z1[n−1]`, etc. — evaluated with an `Op::Scan` over
    /// time, so it currently lowers on the CPU backend (where `Op::Scan`
    /// runs); GPU support tracks `Op::Scan` GPU lowering.
    ///
    /// Any rank ≥ 1 is accepted; leading axes are independent channels.
    pub fn biquad(&mut self, x: NodeId, b: [f32; 3], a: [f32; 3]) -> NodeId {
        assert!(a[0] != 0.0, "biquad: a0 must be non-zero");
        let (b0, b1, b2) = (b[0] / a[0], b[1] / a[0], b[2] / a[0]);
        let (a1, a2) = (a[1] / a[0], a[2] / a[0]);

        let xs_shape = self.shape(x).clone();
        let orig_dims: Vec<i64> = xs_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        let rank = xs_shape.rank();
        let n = xs_shape.dim(rank - 1).unwrap_static();
        let p: usize = (0..rank - 1).map(|i| xs_shape.dim(i).unwrap_static()).product();

        // Reorder to [N, P] so time is the scan axis, channels are per-step.
        let x_pn = self.reshape_(x, vec![p as i64, n as i64]);
        let xs = self.transpose_(x_pn, vec![1, 0]); // [N, P]

        // Scan body: carry = [P, 3] holding (y, z1, z2); x_t = [P].
        let mut body = Graph::new("biquad_body");
        let carry = body.input("carry", Shape::new(&[p, 3], DType::F32));
        let x_t = body.input("x_t", Shape::new(&[p], DType::F32));
        let z1p = body.narrow_(carry, 1, 1, 1);
        let z1p = body.reshape_(z1p, vec![p as i64]);
        let z2p = body.narrow_(carry, 1, 2, 1);
        let z2p = body.reshape_(z2p, vec![p as i64]);
        let cb0 = body.constant(b0 as f64, DType::F32);
        let cb1 = body.constant(b1 as f64, DType::F32);
        let cb2 = body.constant(b2 as f64, DType::F32);
        let ca1 = body.constant(a1 as f64, DType::F32);
        let ca2 = body.constant(a2 as f64, DType::F32);
        let b0x = body.mul(x_t, cb0);
        let y = body.add(b0x, z1p); // y = b0·x + z1
        let b1x = body.mul(x_t, cb1);
        let a1y = body.mul(y, ca1);
        let z1_tmp = body.sub(b1x, a1y);
        let z1 = body.add(z1_tmp, z2p); // z1 = b1·x − a1·y + z2
        let b2x = body.mul(x_t, cb2);
        let a2y = body.mul(y, ca2);
        let z2 = body.sub(b2x, a2y); // z2 = b2·x − a2·y
        let yc = body.reshape_(y, vec![p as i64, 1]);
        let z1c = body.reshape_(z1, vec![p as i64, 1]);
        let z2c = body.reshape_(z2, vec![p as i64, 1]);
        let next = body.concat_(vec![yc, z1c, z2c], 1); // [P, 3]
        body.set_outputs(vec![next]);

        // Trajectory scan with per-step xs → [N, P, 3].
        let init = self.const_f32_tensor(vec![0.0; p * 3], &[p, 3]);
        let traj = self.add_node(
            Op::Scan {
                body: Box::new(body),
                length: n as u32,
                save_trajectory: true,
                num_bcast: 0,
                num_xs: 1,
                num_checkpoints: 0,
            },
            vec![init, xs],
            Shape::new(&[n, p, 3], DType::F32),
        );
        // Extract the y column → [N, P] → [P, N] → original shape.
        let y_np = self.narrow_(traj, 2, 0, 1); // [N, P, 1]
        let y_np = self.reshape_(y_np, vec![n as i64, p as i64]);
        let y_pn = self.transpose_(y_np, vec![1, 0]); // [P, N]
        self.reshape_(y_pn, orig_dims)
    }

    /// Cascade of second-order sections (scipy `sosfilt`): apply each
    /// `(b, a)` biquad in series along the last axis.
    pub fn sosfilt(&mut self, x: NodeId, sections: &[([f32; 3], [f32; 3])]) -> NodeId {
        let mut y = x;
        for &(b, a) in sections {
            y = self.biquad(y, b, a);
        }
        y
    }

    /// Zero-phase IIR filtering (scipy `sosfiltfilt`): forward pass, reverse,
    /// forward pass, reverse. Cancels the SOS cascade's phase, leaving the
    /// squared magnitude response.
    pub fn sosfiltfilt(&mut self, x: NodeId, sections: &[([f32; 3], [f32; 3])]) -> NodeId {
        let last = self.shape(x).rank() - 1;
        let f1 = self.sosfilt(x, sections);
        let r1 = self.reverse(f1, vec![last]);
        let f2 = self.sosfilt(r1, sections);
        self.reverse(f2, vec![last])
    }

    /// Polyphase rational resampling of the last axis by `up`/`down`.
    ///
    /// Zero-stuffs by `up`, applies the FIR anti-alias filter `taps` (scaled
    /// by `up` to preserve amplitude), then decimates by `down`. Output
    /// last-axis length is `ceil(N·up / down)`. Composed from
    /// reshape/concat/FFT-conv/narrow, so it runs on every backend that
    /// supports those (all Apple-Silicon backends).
    pub fn resample_poly(&mut self, x: NodeId, up: usize, down: usize, taps: &[f32]) -> NodeId {
        assert!(up >= 1 && down >= 1, "resample_poly: up/down must be ≥1");
        assert!(!taps.is_empty(), "resample_poly: need ≥1 tap");
        let shape = self.shape(x).clone();
        let rank = shape.rank();
        let last = rank - 1;
        let n = shape.dim(last).unwrap_static();
        let lead: Vec<i64> = (0..last).map(|i| shape.dim(i).unwrap_static() as i64).collect();

        // Zero-stuff by `up`: [.., N] → [.., N, 1] ⊕ zeros[.., N, up-1] → [.., N·up].
        let up_sig = if up == 1 {
            x
        } else {
            let mut d1: Vec<i64> = lead.clone();
            d1.push(n as i64);
            d1.push(1);
            let xr = self.reshape_(x, d1);
            let mut zdims: Vec<usize> = shape
                .dims()
                .iter()
                .take(last)
                .map(|d| d.unwrap_static())
                .collect();
            zdims.push(n);
            zdims.push(up - 1);
            let zeros = self.const_f32_tensor(vec![0.0; zdims.iter().product()], &zdims);
            let stacked = self.concat_(vec![xr, zeros], last + 1); // [.., N, up]
            let mut flat: Vec<i64> = lead.clone();
            flat.push((n * up) as i64);
            self.reshape_(stacked, flat)
        };

        // FIR anti-alias filter (gain `up` compensates the zero-stuffing).
        let l = n * up;
        let scaled: Vec<f32> = taps.iter().map(|&t| t * up as f32).collect();
        let h = self.const_f32_tensor(scaled, &[taps.len()]);
        let filtered = self.fir_conv_same(up_sig, h, taps.len(), l);

        // Decimate by `down`: pad to a multiple of `down`, reshape [.., M, down],
        // keep phase-0 column.
        if down == 1 {
            return filtered;
        }
        let out_len = l.div_ceil(down);
        let padded = out_len * down;
        let filtered = if padded > l {
            let mut zdims: Vec<usize> = shape
                .dims()
                .iter()
                .take(last)
                .map(|d| d.unwrap_static())
                .collect();
            zdims.push(padded - l);
            let zeros = self.const_f32_tensor(vec![0.0; zdims.iter().product()], &zdims);
            self.concat_(vec![filtered, zeros], last)
        } else {
            filtered
        };
        let mut rdims: Vec<i64> = lead.clone();
        rdims.push(out_len as i64);
        rdims.push(down as i64);
        let reshaped = self.reshape_(filtered, rdims); // [.., out_len, down]
        let col0 = self.narrow_(reshaped, last + 1, 0, 1); // [.., out_len, 1]
        let mut odims: Vec<i64> = lead;
        odims.push(out_len as i64);
        self.reshape_(col0, odims)
    }

    fn complex_abs(&mut self, re: NodeId, im: NodeId) -> NodeId {
        let re2 = self.mul(re, re);
        let im2 = self.mul(im, im);
        let sum = self.add(re2, im2);
        self.sqrt(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(g: &Graph, id: NodeId) -> Vec<usize> {
        g.shape(id).dims().iter().map(|d| d.unwrap_static()).collect()
    }
    fn input(g: &mut Graph, shape: &[usize]) -> NodeId {
        g.input("x", Shape::new(shape, DType::F32))
    }

    #[test]
    fn hilbert_preserves_shape() {
        let mut g = Graph::new("hil");
        let x = input(&mut g, &[3, 128]);
        let (re, im) = g.hilbert(x);
        assert_eq!(dims(&g, re), vec![3, 128]);
        assert_eq!(dims(&g, im), vec![3, 128]);
    }

    #[test]
    fn envelope_and_phase_shape() {
        let mut g = Graph::new("env");
        let x = input(&mut g, &[2, 100]);
        let e = g.envelope(x);
        let p = g.instantaneous_phase(x);
        assert_eq!(dims(&g, e), vec![2, 100]);
        assert_eq!(dims(&g, p), vec![2, 100]);
    }

    #[test]
    fn filtfilt_same_length() {
        let mut g = Graph::new("ff");
        let x = input(&mut g, &[2, 200]);
        let taps: Vec<f32> = (0..15).map(|i| 1.0 / (i as f32 + 1.0)).collect();
        let y = g.fir_filtfilt(x, &taps);
        assert_eq!(dims(&g, y), vec![2, 200]);
    }

    #[test]
    fn biquad_and_sosfilt_same_length() {
        let mut g = Graph::new("iir");
        let x = input(&mut g, &[3, 128]);
        let b = [0.2, 0.4, 0.2];
        let a = [1.0, -0.3, 0.1];
        let y = g.biquad(x, b, a);
        assert_eq!(dims(&g, y), vec![3, 128]);
        let sos = [(b, a), ([0.5, 0.0, 0.5], [1.0, 0.2, 0.05])];
        let z = g.sosfiltfilt(x, &sos);
        assert_eq!(dims(&g, z), vec![3, 128]);
    }

    #[test]
    fn resample_poly_length() {
        let mut g = Graph::new("rs");
        let x = input(&mut g, &[2, 100]);
        let taps: Vec<f32> = (0..13).map(|i| 1.0 / (i as f32 + 1.0)).collect();
        // up=3, down=2 → ceil(100*3/2) = 150.
        let y = g.resample_poly(x, 3, 2, &taps);
        assert_eq!(dims(&g, y), vec![2, 150]);
    }
}
