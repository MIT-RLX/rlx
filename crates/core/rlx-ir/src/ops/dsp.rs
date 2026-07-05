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

/// FIR filters with at most this many taps use the direct time-domain
/// (shift-and-add) path in [`Graph::fir_conv1d`]; longer filters use the FFT
/// convolution theorem. Direct form avoids two FFTs and is exact for short
/// kernels; the crossover is deliberately conservative.
const FIR_DIRECT_MAX_TAPS: usize = 64;

/// Output-length convention for [`Graph::fir_conv1d`] — mirrors NumPy / SciPy
/// convolution modes, plus a causal-filter mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FirMode {
    /// Full linear convolution, last-axis length `L + K − 1`.
    Full,
    /// Centered same-length output, length `L` (NumPy `mode="same"`).
    Same,
    /// Only fully-overlapping positions, length `L − K + 1` (NumPy `mode="valid"`).
    Valid,
    /// Causal FIR `y[n] = Σ_k h[k]·x[n−k]`, length `L` (the first `L` samples of
    /// `Full`) — the convention of `scipy.signal.lfilter` with an FIR kernel.
    Causal,
}

/// Truncated impulse response of the IIR filter `(b, a)` (host, `f32`).
///
/// Runs a Direct-Form-II-Transposed recurrence on a unit impulse for `n`
/// samples: `h[t]` for `t ∈ [0, n)`. `a[0]` need not be normalized. Used by
/// [`Graph::iir_as_fir`] to turn a stable IIR into an FIR that lowers natively
/// on every backend.
pub fn iir_impulse_response(b: &[f32], a: &[f32], n: usize) -> Vec<f32> {
    assert!(
        !b.is_empty() && !a.is_empty(),
        "iir_impulse_response: empty coeffs"
    );
    assert!(a[0] != 0.0, "iir_impulse_response: a0 must be non-zero");
    let m = b.len().max(a.len());
    let a0 = a[0];
    let bn: Vec<f32> = (0..m)
        .map(|i| b.get(i).copied().unwrap_or(0.0) / a0)
        .collect();
    let an: Vec<f32> = (0..m)
        .map(|i| a.get(i).copied().unwrap_or(0.0) / a0)
        .collect();
    let s = m - 1; // number of state variables
    let mut w = vec![0f32; s];
    let mut out = Vec::with_capacity(n);
    for t in 0..n {
        let x = if t == 0 { 1.0 } else { 0.0 };
        let y = bn[0] * x + if s > 0 { w[0] } else { 0.0 };
        let mut new_w = vec![0f32; s];
        for i in 0..s {
            let mut wi = bn[i + 1] * x - an[i + 1] * y;
            if i + 1 < s {
                wi += w[i + 1];
            }
            new_w[i] = wi;
        }
        w = new_w;
        out.push(y);
    }
    out
}

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
        let p: usize = (0..rank - 1)
            .map(|i| xs_shape.dim(i).unwrap_static())
            .product();

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
        let lead: Vec<i64> = (0..last)
            .map(|i| shape.dim(i).unwrap_static() as i64)
            .collect();

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

    /// Single-pass 1-D FIR convolution of the last axis with host `taps`,
    /// with a NumPy/SciPy-style output [`FirMode`].
    ///
    /// Short filters (`≤ FIR_DIRECT_MAX_TAPS`) use a direct time-domain
    /// shift-and-add (exact, no FFT); longer filters use the FFT convolution
    /// theorem. Both are pure compositions of `mul`/`add`/`concat`/`Op::Fft`,
    /// so this lowers on **every** backend and stays differentiable. Leading
    /// axes are independent channels/batches.
    pub fn fir_conv1d(&mut self, x: NodeId, taps: &[f32], mode: FirMode) -> NodeId {
        let k = taps.len();
        assert!(k > 0, "fir_conv1d: need ≥1 tap");
        let last = self.shape(x).rank() - 1;
        let l = self.shape(x).dim(last).unwrap_static();
        // `full` has last-axis length l + k - 1.
        let full = if k <= FIR_DIRECT_MAX_TAPS {
            self.fir_full_direct(x, taps)
        } else {
            let h = self.const_f32_tensor(taps.to_vec(), &[k]);
            self.fft_conv1d(x, h, 0, FftNorm::Forward)
        };
        match mode {
            FirMode::Full => full,
            FirMode::Causal => self.narrow_(full, last, 0, l),
            FirMode::Same => {
                let start = (k - 1) / 2;
                self.narrow_(full, last, start, l)
            }
            FirMode::Valid => {
                assert!(
                    l + 1 > k,
                    "fir_conv1d(Valid): signal ({l}) shorter than taps ({k})"
                );
                self.narrow_(full, last, k - 1, l - k + 1)
            }
        }
    }

    /// Direct-form (time-domain) full FIR convolution — exact shift-and-add,
    /// used by [`Self::fir_conv1d`] for short filters. Output length `L + K − 1`.
    fn fir_full_direct(&mut self, x: NodeId, taps: &[f32]) -> NodeId {
        let k = taps.len();
        let last = self.shape(x).rank() - 1;
        let l = self.shape(x).dim(last).unwrap_static();
        let mut acc: Option<NodeId> = None;
        for (j, &h) in taps.iter().enumerate() {
            if h == 0.0 {
                continue;
            }
            let c = self.constant(h as f64, DType::F32);
            let scaled = self.mul(x, c);
            let term = self.fir_pad_last(scaled, j, k - 1 - j);
            acc = Some(match acc {
                None => term,
                Some(a) => self.add(a, term),
            });
        }
        match acc {
            Some(a) => a,
            None => {
                let mut dims: Vec<usize> = self
                    .shape(x)
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                dims[last] = l + k - 1;
                self.fir_zeros_dims(&dims)
            }
        }
    }

    /// Long 1-D convolution via **uniform-partitioned overlap-save (UPOLS)** —
    /// the GPU-friendly algorithm for room-impulse-response (RIR) / convolution
    /// reverb, where the impulse response is thousands to millions of taps.
    ///
    /// The impulse response `ir` (rank-1, length `M`) is split into
    /// `P = ⌈M/B⌉` partitions of `B = next_pow2(block)`; each is zero-padded to
    /// `2B` and pre-transformed to `H_p`. The signal `x` (`[.., L]`) is streamed
    /// as overlap-save frames of `2B`; the per-partition spectra are summed in
    /// the frequency domain (a frequency-domain delay line,
    /// `Y[m] = Σ_p H_p · X[m−p]`) and inverse-transformed, keeping the valid `B`
    /// samples per frame.
    ///
    /// Every FFT is a fixed `2B` points — small enough to stay on the native
    /// Metal/WGPU FFT kernels (unlike a single `L+M−1` transform) — and the
    /// whole thing is a pure composition of `Op::Fft` + elementwise + shape ops,
    /// so it lowers on **all** backends. Output is the full linear convolution,
    /// last-axis length `L + M − 1`.
    pub fn partitioned_conv1d(&mut self, x: NodeId, ir: NodeId, block: usize) -> NodeId {
        let xs = self.shape(x).clone();
        let rank = xs.rank();
        let last = rank - 1;
        let l = xs.dim(last).unwrap_static();
        let ir_shape = self.shape(ir).clone();
        assert_eq!(ir_shape.rank(), 1, "partitioned_conv1d: ir must be rank-1");
        let m = ir_shape.dim(0).unwrap_static();
        assert!(m >= 1 && l >= 1, "partitioned_conv1d: empty input/ir");

        let b = crate::fft::next_pow2(block.max(1));
        let n = 2 * b; // FFT size (power of two)
        let kbins = n / 2 + 1;
        let p = m.div_ceil(b);
        let mf = (l + m - 1).div_ceil(b);

        // IR partition spectra: [M] → [P, B] → [P, 2B] → rfft → [P, K].
        let ir_pad = self.fir_pad_last(ir, 0, p * b - m); // [P*B]
        let ir_blocks = self.reshape_(ir_pad, vec![p as i64, b as i64]); // [P, B]
        let ir_zeros = self.fir_zeros_dims(&[p, b]); // [P, B]
        let ir_frames = self.concat_(vec![ir_blocks, ir_zeros], 1); // [P, 2B]
        let (h_re, h_im) = self.rfft(ir_frames, FftNorm::Forward); // [P, K]

        // Frame x with overlap-save: prepend B zeros, pad tail to (Mf+1)·B.
        let back = mf * b - l; // ≥ 0 since Mf·B ≥ L
        let x_pad = self.fir_pad_last(x, b, back); // [.., (Mf+1)·B]
        let mut blk_dims: Vec<i64> = (0..last)
            .map(|i| xs.dim(i).unwrap_static() as i64)
            .collect();
        blk_dims.push((mf + 1) as i64);
        blk_dims.push(b as i64);
        let blocks = self.reshape_(x_pad, blk_dims); // [.., Mf+1, B]
        let brank = self.shape(blocks).rank();
        let baxis = brank - 2; // block axis
        let blast = brank - 1; // B axis
        let left = self.narrow_(blocks, baxis, 0, mf); // [.., Mf, B]
        let right = self.narrow_(blocks, baxis, 1, mf); // [.., Mf, B]
        let frames = self.concat_(vec![left, right], blast); // [.., Mf, 2B]
        let (x_re, x_im) = self.rfft(frames, FftNorm::Forward); // [.., Mf, K]

        // Frequency-domain delay line: Y[m] = Σ_p H[p] · X[m−p].
        let mf_axis = self.shape(x_re).rank() - 2;
        let mut acc: Option<(NodeId, NodeId)> = None;
        for pp in 0..p.min(mf) {
            let hp_re = self.narrow_(h_re, 0, pp, 1);
            let hp_re = self.reshape_(hp_re, vec![kbins as i64]); // [K]
            let hp_im = self.narrow_(h_im, 0, pp, 1);
            let hp_im = self.reshape_(hp_im, vec![kbins as i64]); // [K]
            // X shifted down by pp along the frame axis (zero-fill first pp frames).
            let (xr_s, xi_s) = if pp == 0 {
                (x_re, x_im)
            } else {
                let keep = mf - pp;
                let mut zdims: Vec<usize> = self
                    .shape(x_re)
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                zdims[mf_axis] = pp;
                let zre = self.fir_zeros_dims(&zdims);
                let zim = self.fir_zeros_dims(&zdims);
                let hr = self.narrow_(x_re, mf_axis, 0, keep);
                let hi = self.narrow_(x_im, mf_axis, 0, keep);
                (
                    self.concat_(vec![zre, hr], mf_axis),
                    self.concat_(vec![zim, hi], mf_axis),
                )
            };
            // Complex multiply (H_re + iH_im)·(X_re + iX_im), broadcasting [K].
            let rr = self.mul(hp_re, xr_s);
            let ii = self.mul(hp_im, xi_s);
            let t_re = self.sub(rr, ii);
            let ri = self.mul(hp_re, xi_s);
            let ir2 = self.mul(hp_im, xr_s);
            let t_im = self.add(ri, ir2);
            acc = Some(match acc {
                None => (t_re, t_im),
                Some((ar, ai)) => (self.add(ar, t_re), self.add(ai, t_im)),
            });
        }
        let (y_re, y_im) = acc.expect("partitioned_conv1d: no partitions");

        // Inverse transform, overlap-save trim (keep last B), reassemble.
        let y_frames = self.irfft(y_re, y_im, n, FftNorm::Forward); // [.., Mf, 2B]
        let ylast = self.shape(y_frames).rank() - 1;
        let valid = self.narrow_(y_frames, ylast, b, b); // [.., Mf, B]
        let mut out_dims: Vec<i64> = (0..last)
            .map(|i| xs.dim(i).unwrap_static() as i64)
            .collect();
        out_dims.push((mf * b) as i64);
        let flat = self.reshape_(valid, out_dims); // [.., Mf·B]
        let flast = self.shape(flat).rank() - 1;
        self.narrow_(flat, flast, 0, l + m - 1) // [.., L+M-1]
    }

    /// Convolution reverb: apply a room impulse response `ir` (host samples) to
    /// `x` via [`Self::partitioned_conv1d`]. Output is the full wet signal,
    /// last-axis length `L + len(ir) − 1` (the reverb tail extends past `x`).
    ///
    /// `block` sets the partition / FFT size: `512` keeps every FFT at `1024`
    /// (within the WGPU native cap), `1024` keeps it at `2048` (Metal cap).
    pub fn conv_reverb(&mut self, x: NodeId, ir: &[f32], block: usize) -> NodeId {
        assert!(!ir.is_empty(), "conv_reverb: empty impulse response");
        let ir_node = self.const_f32_tensor(ir.to_vec(), &[ir.len()]);
        self.partitioned_conv1d(x, ir_node, block)
    }

    /// Fused [`crate::Op::PartitionedConv`] node — the single-op form of
    /// [`Self::partitioned_conv1d`]. A backend may lower it natively; otherwise
    /// `unfuse` decomposes it to the batched-GEMM frequency-domain path
    /// ([`Self::partitioned_conv1d_gemm`]). Output is the full linear
    /// convolution, last-axis length `L + M − 1`.
    pub fn partitioned_conv(&mut self, x: NodeId, ir: NodeId, block: usize) -> NodeId {
        let xs = self.shape(x).clone();
        let last = xs.rank() - 1;
        let l = xs.dim(last).unwrap_static();
        let ir_shape = self.shape(ir).clone();
        assert_eq!(ir_shape.rank(), 1, "partitioned_conv: ir must be rank-1");
        let m = ir_shape.dim(0).unwrap_static();
        let mut dims: Vec<usize> = xs.dims().iter().map(|d| d.unwrap_static()).collect();
        dims[last] = l + m - 1;
        let out_shape = Shape::new(&dims, xs.dtype());
        self.push(Op::PartitionedConv { block }, vec![x, ir], out_shape, None)
    }

    /// GEMM-path variant of [`Self::partitioned_conv1d`]: the frequency-domain
    /// delay-line sum `Y[m,k] = Σ_p H[p,k]·X[m−p,k]` is evaluated as a **batched
    /// complex matmul** contracting the partition axis (batched over FFT bin
    /// `k`) instead of `P` elementwise multiply-accumulates. On CUDA/ROCm/Metal
    /// this routes the accumulation through the native batched-GEMM kernels
    /// (cuBLAS / rocBLAS / MPS — the tensor-core path) while staying a pure op
    /// composition that lowers on every backend. Numerically identical to
    /// [`Self::partitioned_conv1d`]; this is the decomposition behind
    /// [`crate::Op::PartitionedConv`].
    pub fn partitioned_conv1d_gemm(&mut self, x: NodeId, ir: NodeId, block: usize) -> NodeId {
        let xs = self.shape(x).clone();
        let rank = xs.rank();
        let last = rank - 1;
        let l = xs.dim(last).unwrap_static();
        let ir_shape = self.shape(ir).clone();
        assert_eq!(
            ir_shape.rank(),
            1,
            "partitioned_conv1d_gemm: ir must be rank-1"
        );
        let m = ir_shape.dim(0).unwrap_static();
        assert!(m >= 1 && l >= 1, "partitioned_conv1d_gemm: empty input/ir");

        let b = crate::fft::next_pow2(block.max(1));
        let n = 2 * b;
        let kbins = n / 2 + 1;
        let p = m.div_ceil(b);
        let mf = (l + m - 1).div_ceil(b);

        // IR partition spectra: [M] → [P, 2B] → rfft → [P, K].
        let ir_pad = self.fir_pad_last(ir, 0, p * b - m);
        let ir_blocks = self.reshape_(ir_pad, vec![p as i64, b as i64]);
        let ir_zeros = self.fir_zeros_dims(&[p, b]);
        let ir_frames = self.concat_(vec![ir_blocks, ir_zeros], 1);
        let (h_re, h_im) = self.rfft(ir_frames, FftNorm::Forward); // [P, K]

        // Overlap-save frames of x → rfft → X [.., Mf, K].
        let back = mf * b - l;
        let x_pad = self.fir_pad_last(x, b, back);
        let mut blk_dims: Vec<i64> = (0..last)
            .map(|i| xs.dim(i).unwrap_static() as i64)
            .collect();
        blk_dims.push((mf + 1) as i64);
        blk_dims.push(b as i64);
        let blocks = self.reshape_(x_pad, blk_dims);
        let brank = self.shape(blocks).rank();
        let baxis = brank - 2;
        let blast = brank - 1;
        let left = self.narrow_(blocks, baxis, 0, mf);
        let right = self.narrow_(blocks, baxis, 1, mf);
        let frames = self.concat_(vec![left, right], blast);
        let (x_re, x_im) = self.rfft(frames, FftNorm::Forward); // [.., Mf, K]

        // Build the delay-line tensor U[.., Mf, P, K] = stack_p shift(X, p),
        // then contract the P axis with H via one batched complex matmul.
        let xr = self.shape(x_re).rank();
        let mf_axis = xr - 2;
        let lead: Vec<usize> = (0..mf_axis)
            .map(|i| self.shape(x_re).dim(i).unwrap_static())
            .collect();
        let rows: usize = lead.iter().product::<usize>() * mf;

        let mut u_re_parts = Vec::with_capacity(p);
        let mut u_im_parts = Vec::with_capacity(p);
        for pp in 0..p {
            let (sre, sim) = if pp == 0 {
                (x_re, x_im)
            } else {
                let keep = mf - pp;
                let mut zd: Vec<usize> = self
                    .shape(x_re)
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                zd[mf_axis] = pp;
                let zre = self.fir_zeros_dims(&zd);
                let zim = self.fir_zeros_dims(&zd);
                let hr = self.narrow_(x_re, mf_axis, 0, keep);
                let hi = self.narrow_(x_im, mf_axis, 0, keep);
                (
                    self.concat_(vec![zre, hr], mf_axis),
                    self.concat_(vec![zim, hi], mf_axis),
                )
            };
            // insert a size-1 P slot before K: [.., Mf, K] → [.., Mf, 1, K]
            let mut d: Vec<i64> = self
                .shape(sre)
                .dims()
                .iter()
                .map(|x| x.unwrap_static() as i64)
                .collect();
            d.insert(d.len() - 1, 1);
            u_re_parts.push(self.reshape_(sre, d.clone()));
            u_im_parts.push(self.reshape_(sim, d));
        }
        let u_re = self.concat_(u_re_parts, mf_axis + 1); // [.., Mf, P, K]
        let u_im = self.concat_(u_im_parts, mf_axis + 1);

        // Flatten to [R, P, K], move to [K, R, P] for a batch-over-K matmul.
        let u_re = self.reshape_(u_re, vec![rows as i64, p as i64, kbins as i64]);
        let u_im = self.reshape_(u_im, vec![rows as i64, p as i64, kbins as i64]);
        let u_re = self.transpose_(u_re, vec![2, 0, 1]); // [K, R, P]
        let u_im = self.transpose_(u_im, vec![2, 0, 1]);
        let h_re_t = self.transpose_(h_re, vec![1, 0]); // [K, P]
        let h_im_t = self.transpose_(h_im, vec![1, 0]);
        let h_re_c = self.reshape_(h_re_t, vec![kbins as i64, p as i64, 1]); // [K, P, 1]
        let h_im_c = self.reshape_(h_im_t, vec![kbins as i64, p as i64, 1]);

        let out_krp = Shape::new(&[kbins, rows, 1], DType::F32);
        let rr = self.matmul(u_re, h_re_c, out_krp.clone());
        let ii = self.matmul(u_im, h_im_c, out_krp.clone());
        let y_re_k = self.sub(rr, ii); // Re = Σ (Xre·Hre − Xim·Him)
        let ri = self.matmul(u_re, h_im_c, out_krp.clone());
        let ir2 = self.matmul(u_im, h_re_c, out_krp);
        let y_im_k = self.add(ri, ir2); // Im = Σ (Xre·Him + Xim·Hre)

        // [K, R, 1] → [K, R] → [R, K] → [.., Mf, K].
        let y_re = self.reshape_(y_re_k, vec![kbins as i64, rows as i64]);
        let y_re = self.transpose_(y_re, vec![1, 0]);
        let y_im = self.reshape_(y_im_k, vec![kbins as i64, rows as i64]);
        let y_im = self.transpose_(y_im, vec![1, 0]);
        let mut yd: Vec<i64> = lead.iter().map(|&d| d as i64).collect();
        yd.push(mf as i64);
        yd.push(kbins as i64);
        let y_re = self.reshape_(y_re, yd.clone());
        let y_im = self.reshape_(y_im, yd);

        // Inverse transform, overlap-save trim, reassemble → [.., L+M−1].
        let y_frames = self.irfft(y_re, y_im, n, FftNorm::Forward);
        let ylast = self.shape(y_frames).rank() - 1;
        let valid = self.narrow_(y_frames, ylast, b, b);
        let mut out_dims: Vec<i64> = (0..last)
            .map(|i| xs.dim(i).unwrap_static() as i64)
            .collect();
        out_dims.push((mf * b) as i64);
        let flat = self.reshape_(valid, out_dims);
        let flast = self.shape(flat).rank() - 1;
        self.narrow_(flat, flast, 0, l + m - 1)
    }

    /// General direct-form-II-transposed IIR filter of arbitrary order applied
    /// along the last axis:
    /// `a[0]·y[n] = Σ_i b[i]·x[n−i] − Σ_{i≥1} a[i]·y[n−i]`.
    ///
    /// A true recurrence evaluated with `Op::Scan` over time — [`Self::biquad`]
    /// is the order-2 case — so it lowers wherever `Op::Scan` runs (CPU
    /// natively; Metal/WGPU/oneAPI via host-fallback). For an IIR that runs
    /// **natively on every** backend, see [`Self::iir_as_fir`]. Leading axes are
    /// independent channels.
    pub fn iirfilt(&mut self, x: NodeId, b: &[f32], a: &[f32]) -> NodeId {
        assert!(
            !b.is_empty() && !a.is_empty(),
            "iirfilt: empty coefficients"
        );
        assert!(a[0] != 0.0, "iirfilt: a0 must be non-zero");
        let m = b.len().max(a.len());
        let a0 = a[0];
        let bn: Vec<f32> = (0..m)
            .map(|i| b.get(i).copied().unwrap_or(0.0) / a0)
            .collect();
        let an: Vec<f32> = (0..m)
            .map(|i| a.get(i).copied().unwrap_or(0.0) / a0)
            .collect();
        let s = m - 1; // number of state variables

        let xs_shape = self.shape(x).clone();
        let orig_dims: Vec<i64> = xs_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        let rank = xs_shape.rank();
        let nlen = xs_shape.dim(rank - 1).unwrap_static();
        let pch: usize = (0..rank - 1)
            .map(|i| xs_shape.dim(i).unwrap_static())
            .product();

        // [.., N] → [P, N] → [N, P] so time is the scan axis, channels per step.
        let x_pn = self.reshape_(x, vec![pch as i64, nlen as i64]);
        let xs = self.transpose_(x_pn, vec![1, 0]); // [N, P]

        // Scan body: carry [P, 1+S] = (y, w[0..S]); x_t [P].
        let cw = 1 + s;
        let mut body = Graph::new("iir_body");
        let carry = body.input("carry", Shape::new(&[pch, cw], DType::F32));
        let x_t = body.input("x_t", Shape::new(&[pch], DType::F32));
        let w: Vec<NodeId> = (0..s)
            .map(|i| {
                let wi = body.narrow_(carry, 1, 1 + i, 1);
                body.reshape_(wi, vec![pch as i64])
            })
            .collect();
        let cbn: Vec<NodeId> = (0..m)
            .map(|i| body.constant(bn[i] as f64, DType::F32))
            .collect();
        let can: Vec<NodeId> = (0..m)
            .map(|i| body.constant(an[i] as f64, DType::F32))
            .collect();
        // y = b0·x + w0
        let b0x = body.mul(x_t, cbn[0]);
        let y = if s > 0 { body.add(b0x, w[0]) } else { b0x };
        // new_w[i] = b[i+1]·x − a[i+1]·y + w[i+1]   (last term absent for i = S−1)
        let mut new_w = Vec::with_capacity(s);
        for i in 0..s {
            let bx = body.mul(x_t, cbn[i + 1]);
            let ay = body.mul(y, can[i + 1]);
            let mut wi = body.sub(bx, ay);
            if i + 1 < s {
                wi = body.add(wi, w[i + 1]);
            }
            new_w.push(wi);
        }
        // next carry = concat([y], new_w) → [P, 1+S]
        let mut cols = Vec::with_capacity(cw);
        cols.push(body.reshape_(y, vec![pch as i64, 1]));
        for wi in new_w {
            cols.push(body.reshape_(wi, vec![pch as i64, 1]));
        }
        let next = body.concat_(cols, 1);
        body.set_outputs(vec![next]);

        let init = self.const_f32_tensor(vec![0.0; pch * cw], &[pch, cw]);
        let traj = self.add_node(
            Op::Scan {
                body: Box::new(body),
                length: nlen as u32,
                save_trajectory: true,
                num_bcast: 0,
                num_xs: 1,
                num_checkpoints: 0,
            },
            vec![init, xs],
            Shape::new(&[nlen, pch, cw], DType::F32),
        );
        // Extract y column [N, P, 1] → [N, P] → [P, N] → original shape.
        let y_np = self.narrow_(traj, 2, 0, 1);
        let y_np = self.reshape_(y_np, vec![nlen as i64, pch as i64]);
        let y_pn = self.transpose_(y_np, vec![1, 0]);
        self.reshape_(y_pn, orig_dims)
    }

    /// Stable IIR filtering that runs **natively on every backend**: reduce the
    /// IIR `(b, a)` to its truncated impulse response (`taps` samples, computed
    /// on the host via [`iir_impulse_response`]) and apply it as an FIR through
    /// [`Self::fir_conv1d`] with the given `mode` (`FirMode::Causal` reproduces
    /// [`Self::iirfilt`] up to truncation error).
    ///
    /// Exact for FIR filters (`a = [a0]`); for IIR it is exact up to the
    /// truncation tail, so choose `taps` past the impulse response's decay.
    pub fn iir_as_fir(
        &mut self,
        x: NodeId,
        b: &[f32],
        a: &[f32],
        taps: usize,
        mode: FirMode,
    ) -> NodeId {
        let h = iir_impulse_response(b, a, taps);
        self.fir_conv1d(x, &h, mode)
    }

    /// Zero constant tensor of the given static dims (filter-builder helper).
    fn fir_zeros_dims(&mut self, dims: &[usize]) -> NodeId {
        self.const_f32_tensor(vec![0.0; dims.iter().product()], dims)
    }

    /// Pad the last axis of `x` with `front` leading and `back` trailing zeros.
    fn fir_pad_last(&mut self, x: NodeId, front: usize, back: usize) -> NodeId {
        if front == 0 && back == 0 {
            return x;
        }
        let last = self.shape(x).rank() - 1;
        let mut zdims: Vec<usize> = self
            .shape(x)
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        let mut parts = Vec::new();
        if front > 0 {
            zdims[last] = front;
            let z = self.fir_zeros_dims(&zdims);
            parts.push(z);
        }
        parts.push(x);
        if back > 0 {
            zdims[last] = back;
            let z = self.fir_zeros_dims(&zdims);
            parts.push(z);
        }
        self.concat_(parts, last)
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
        g.shape(id)
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect()
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

    #[test]
    fn fir_conv1d_modes_lengths() {
        let taps: Vec<f32> = (0..9).map(|i| 1.0 / (i as f32 + 1.0)).collect();
        for &(mode, want) in &[
            (FirMode::Full, 64 + 9 - 1),
            (FirMode::Same, 64),
            (FirMode::Valid, 64 - 9 + 1),
            (FirMode::Causal, 64),
        ] {
            let mut g = Graph::new("fir");
            let x = input(&mut g, &[2, 64]);
            let y = g.fir_conv1d(x, &taps, mode);
            assert_eq!(dims(&g, y), vec![2, want], "mode {mode:?}");
        }
    }

    #[test]
    fn fir_conv1d_long_uses_fft_path() {
        // > FIR_DIRECT_MAX_TAPS taps exercises the FFT branch.
        let taps: Vec<f32> = (0..100).map(|i| ((i as f32) * 0.01).cos()).collect();
        let mut g = Graph::new("firfft");
        let x = input(&mut g, &[130]);
        let y = g.fir_conv1d(x, &taps, FirMode::Full);
        assert_eq!(dims(&g, y), vec![130 + 100 - 1]);
    }

    #[test]
    fn conv_reverb_full_length() {
        let mut g = Graph::new("rir");
        let x = input(&mut g, &[2, 100]);
        let ir: Vec<f32> = (0..300).map(|i| (-(i as f32) * 0.02).exp()).collect();
        let y = g.conv_reverb(x, &ir, 64);
        assert_eq!(dims(&g, y), vec![2, 100 + 300 - 1]);
    }

    #[test]
    fn partitioned_conv1d_node_ir() {
        let mut g = Graph::new("pconv");
        let x = input(&mut g, &[50]);
        let ir = g.const_f32_tensor((0..70).map(|i| i as f32 * 0.1).collect(), &[70]);
        let y = g.partitioned_conv1d(x, ir, 32);
        assert_eq!(dims(&g, y), vec![50 + 70 - 1]);
    }

    #[test]
    fn partitioned_conv1d_gemm_shape() {
        let mut g = Graph::new("pgemm");
        let x = input(&mut g, &[2, 50]);
        let ir = g.const_f32_tensor((0..70).map(|i| i as f32 * 0.1).collect(), &[70]);
        let y = g.partitioned_conv1d_gemm(x, ir, 32);
        assert_eq!(dims(&g, y), vec![2, 50 + 70 - 1]);
    }

    #[test]
    fn partitioned_conv_op_shape() {
        let mut g = Graph::new("pop");
        let x = input(&mut g, &[3, 40]);
        let ir = g.const_f32_tensor((0..100).map(|i| i as f32 * 0.01).collect(), &[100]);
        let y = g.partitioned_conv(x, ir, 32);
        // Op::PartitionedConv carries the L+M-1 output shape directly.
        assert_eq!(dims(&g, y), vec![3, 40 + 100 - 1]);
    }

    #[test]
    fn iirfilt_high_order_same_length() {
        let mut g = Graph::new("iirN");
        let x = input(&mut g, &[3, 128]);
        // Order-3 IIR (4 b/a coeffs).
        let b = [0.1, 0.2, 0.2, 0.1];
        let a = [1.0, -0.5, 0.3, -0.1];
        let y = g.iirfilt(x, &b, &a);
        assert_eq!(dims(&g, y), vec![3, 128]);
    }

    #[test]
    fn iir_as_fir_causal_same_length() {
        let mut g = Graph::new("iirfir");
        let x = input(&mut g, &[2, 100]);
        let b = [0.2, 0.4, 0.2];
        let a = [1.0, -0.3, 0.1];
        let y = g.iir_as_fir(x, &b, &a, 48, FirMode::Causal);
        assert_eq!(dims(&g, y), vec![2, 100]);
    }
}
