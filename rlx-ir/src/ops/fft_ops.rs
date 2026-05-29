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

//! gpu-fft-shaped graph helpers: real-input FFT, spectrum split/merge, PSD, STFT.
//!
//! These methods compose primitive ops (`Op::Fft`, `narrow`, `concat`, …) into
//! NumPy/JAX-style signal-processing building blocks. Backends that cannot lower
//! the full subgraph (for example Metal MPSGraph on `Op::Fft`) still execute
//! the underlying `Op::Fft` nodes via thunks or host fallback.

use crate::infer::GraphExt as _;
use crate::{DType, Graph, NodeId, Op, Shape, fft::FftNorm};

impl Graph {
    /// Zero-pad the last axis to the next power of two (no-op when already pow2).
    pub fn pad_last_axis_to_pow2(&mut self, x: NodeId) -> NodeId {
        let shape = self.shape(x).clone();
        let rank = shape.rank();
        let last = rank - 1;
        let n = shape.dim(last).unwrap_static();
        let n_pad = crate::fft::next_pow2(n);
        if n_pad == n {
            return x;
        }
        let pad_len = n_pad - n;
        let mut pad_dims: Vec<usize> = shape.dims().iter().map(|d| d.unwrap_static()).collect();
        pad_dims[last] = pad_len;
        let pad_shape = Shape::new(&pad_dims, shape.dtype());
        let zeros = self.zeros_tensor(&pad_shape);
        self.concat_(vec![x, zeros], last)
    }

    /// Split a 2N real-block spectrum into separate real / imag tensors.
    pub fn split_spectrum(&mut self, spectrum: NodeId) -> (NodeId, NodeId) {
        let shape = self.shape(spectrum).clone();
        let meta = crate::fft::fft_meta(&shape);
        let last = shape.rank() - 1;
        let n = meta.n_complex;
        let re = self.narrow_(spectrum, last, 0, n);
        let im = self.narrow_(spectrum, last, n, n);
        (re, im)
    }

    /// Real-input FFT (gpu-fft `fft`): auto zero-pads to pow2, returns `(re, im)`.
    pub fn fft_real(&mut self, x: NodeId, norm: FftNorm) -> (NodeId, NodeId) {
        assert_eq!(
            self.shape(x).dtype(),
            DType::F32,
            "fft_real: requires F32 real input"
        );
        let padded = self.pad_last_axis_to_pow2(x);
        let shape = self.shape(padded).clone();
        let rank = shape.rank();
        let last = rank - 1;
        let n = shape.dim(last).unwrap_static();
        let mut im_dims: Vec<usize> = shape.dims().iter().map(|d| d.unwrap_static()).collect();
        im_dims[last] = n;
        let im_shape = Shape::new(&im_dims, DType::F32);
        let zero_im = self.zeros_tensor(&im_shape);
        let block = self.concat_(vec![padded, zero_im], last);
        let spectrum = self.fft_norm(block, false, norm);
        self.split_spectrum(spectrum)
    }

    /// Batched real-input FFT — same as `fft_real` when the last axis is signal
    /// length; leading axes are independent batch dimensions.
    pub fn fft_batch_real(&mut self, x: NodeId, norm: FftNorm) -> (NodeId, NodeId) {
        self.fft_real(x, norm)
    }

    /// Real-input FFT with half-spectrum output (`n_pad/2 + 1` complex bins).
    ///
    /// The input is zero-padded to the next power of two along the last axis
    /// before the transform, matching NumPy `rfft` padding semantics.
    pub fn rfft(&mut self, x: NodeId, norm: FftNorm) -> (NodeId, NodeId) {
        let (re, im) = self.fft_real(x, norm);
        let rank = self.shape(re).rank();
        let last = rank - 1;
        let n = self.shape(re).dim(last).unwrap_static();
        let half = n / 2 + 1;
        (
            self.narrow_(re, last, 0, half),
            self.narrow_(im, last, 0, half),
        )
    }

    /// Inverse real FFT from half-spectrum `(re, im)` with Hermitian symmetry.
    ///
    /// Mirrors the conjugate half of the spectrum (excluding DC and Nyquist) before
    /// calling [`Self::ifft_spectrum`], then truncates to length `n`.
    pub fn irfft(&mut self, re_half: NodeId, im_half: NodeId, n: usize, norm: FftNorm) -> NodeId {
        assert_eq!(
            *self.shape(re_half),
            *self.shape(im_half),
            "irfft: re/im shape mismatch"
        );
        let n_pad = crate::fft::next_pow2(n);
        let half = n_pad / 2 + 1;
        let rank = self.shape(re_half).rank();
        let last = rank - 1;
        assert_eq!(
            self.shape(re_half).dim(last).unwrap_static(),
            half,
            "irfft: expected half-spectrum length {half}, got {}",
            self.shape(re_half).dim(last).unwrap_static()
        );
        let (re_full, im_full) = if half > 2 {
            let mirror_len = half - 2;
            let mirror_re = self.narrow_(re_half, last, 1, mirror_len);
            let mirror_im = self.narrow_(im_half, last, 1, mirror_len);
            let mirror_re_rev = self.reverse_last_axis(mirror_re);
            let mirror_im_rev = self.reverse_last_axis(mirror_im);
            let neg = self.scalar_f32(-1.0);
            let mirror_im_neg = self.mul(mirror_im_rev, neg);
            (
                self.concat_(vec![re_half, mirror_re_rev], last),
                self.concat_(vec![im_half, mirror_im_neg], last),
            )
        } else {
            (re_half, im_half)
        };
        let recovered = self.ifft_spectrum(re_full, im_full, norm);
        self.narrow_(recovered, last, 0, n)
    }

    /// Short-time Fourier transform: `[..., T]` → `[frames, ..., 2·half]` (re/im block per frame).
    ///
    /// Each frame is `rfft`'d with length `frame_len` and hop `hop` along the last axis.
    pub fn stft(&mut self, x: NodeId, frame_len: usize, hop: usize, norm: FftNorm) -> NodeId {
        assert!(
            frame_len > 0 && hop > 0,
            "stft: frame_len and hop must be positive"
        );
        let shape = self.shape(x).clone();
        let rank = shape.rank();
        let last = rank - 1;
        let t = shape.dim(last).unwrap_static();
        assert!(
            t >= frame_len,
            "stft: signal length {t} < frame_len {frame_len}"
        );
        let n_frames = 1 + (t - frame_len) / hop;
        let mut frames = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let start = f * hop;
            let frame = self.narrow_(x, last, start, frame_len);
            let (re, im) = self.rfft(frame, norm);
            let block = self.concat_(vec![re, im], last);
            frames.push(block);
        }
        if frames.len() == 1 {
            let f = frames[0];
            let mut dims: Vec<i64> = self
                .shape(f)
                .dims()
                .iter()
                .map(|d| d.unwrap_static() as i64)
                .collect();
            dims.insert(0, 1);
            return self.reshape_(f, dims);
        }
        let mut rows = Vec::new();
        for f in frames {
            let mut dims: Vec<i64> = self
                .shape(f)
                .dims()
                .iter()
                .map(|d| d.unwrap_static() as i64)
                .collect();
            dims.insert(0, 1);
            rows.push(self.reshape_(f, dims));
        }
        self.concat_(rows, 0)
    }

    /// 1D convolution via the convolution theorem (`rfft` → complex multiply → `irfft`).
    ///
    /// Both inputs are zero-padded to at least `n_fft` (or the next power of two covering
    /// `len(a) + len(b) - 1` when `n_fft` is small).
    pub fn fft_conv1d(&mut self, a: NodeId, b: NodeId, n_fft: usize, norm: FftNorm) -> NodeId {
        let n_fft = n_fft.max(crate::fft::next_pow2(
            self.shape(a).dim(self.shape(a).rank() - 1).unwrap_static()
                + self.shape(b).dim(self.shape(b).rank() - 1).unwrap_static()
                - 1,
        ));
        let pad_a = self.pad_axis_to_len(a, n_fft);
        let pad_b = self.pad_axis_to_len(b, n_fft);
        let (a_re, a_im) = self.rfft(pad_a, norm);
        let (b_re, b_im) = self.rfft(pad_b, norm);
        let ar_br = self.mul(a_re, b_re);
        let ai_bi = self.mul(a_im, b_im);
        let prod_re = self.sub(ar_br, ai_bi);
        let ar_bi = self.mul(a_re, b_im);
        let ai_br = self.mul(a_im, b_re);
        let prod_im = self.add(ar_bi, ai_br);
        let out_len = self.shape(a).dim(self.shape(a).rank() - 1).unwrap_static()
            + self.shape(b).dim(self.shape(b).rank() - 1).unwrap_static()
            - 1;
        self.irfft(prod_re, prod_im, out_len.max(1), norm)
    }

    /// Constant tensor of FFT sample frequencies (length `n`, f64).
    pub fn fftfreq_tensor(&mut self, n: usize) -> NodeId {
        let xs = crate::fft::fftfreq(n);
        let mut bytes = Vec::with_capacity(n * 8);
        for x in &xs {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        self.add_node(
            Op::Constant { data: bytes },
            vec![],
            Shape::new(&[n], DType::F64),
        )
    }

    /// Constant tensor of rFFT sample frequencies (length `n/2 + 1`, f64).
    pub fn rfftfreq_tensor(&mut self, n: usize) -> NodeId {
        let xs = crate::fft::rfftfreq(n);
        let half = xs.len();
        let mut bytes = Vec::with_capacity(half * 8);
        for x in &xs {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        self.add_node(
            Op::Constant { data: bytes },
            vec![],
            Shape::new(&[half], DType::F64),
        )
    }

    /// Power spectral density from real input: `rfft` → `psd`.
    pub fn psd_real(&mut self, x: NodeId, norm: FftNorm) -> NodeId {
        let (re, im) = self.rfft(x, norm);
        self.psd(re, im)
    }

    /// Inverse FFT from separate real / imag spectra (gpu-fft `ifft` real part).
    pub fn ifft_spectrum(&mut self, re: NodeId, im: NodeId, norm: FftNorm) -> NodeId {
        let re_shape = self.shape(re).clone();
        assert_eq!(
            re_shape,
            *self.shape(im),
            "ifft_spectrum: re/im shape mismatch"
        );
        let rank = re_shape.rank();
        let last = rank - 1;
        let n = re_shape.dim(last).unwrap_static();
        let block = self.concat_(vec![re, im], last);
        let full = self.fft_norm(block, true, norm);
        self.narrow_(full, last, 0, n)
    }

    /// Power spectral density: `(re² + im²) / N` (gpu-fft `psd::psd`).
    pub fn psd(&mut self, re: NodeId, im: NodeId) -> NodeId {
        let n = self
            .shape(re)
            .dim(self.shape(re).rank() - 1)
            .unwrap_static();
        let re2 = self.mul(re, re);
        let im2 = self.mul(im, im);
        let power = self.add(re2, im2);
        let inv_n = self.scalar_f32(1.0 / n as f32);
        self.mul(power, inv_n)
    }

    fn reverse_last_axis(&mut self, x: NodeId) -> NodeId {
        let shape = self.shape(x).clone();
        let rank = shape.rank();
        let last = rank - 1;
        let len = shape.dim(last).unwrap_static();
        if len <= 1 {
            return x;
        }
        let prefix_elems: usize = shape
            .dims()
            .iter()
            .take(last)
            .map(|d| d.unwrap_static())
            .product();
        let mut idx_bytes = Vec::with_capacity(prefix_elems * len * 4);
        for _ in 0..prefix_elems.max(1) {
            for i in (0..len).rev() {
                idx_bytes.extend_from_slice(&(i as i32).to_le_bytes());
            }
        }
        let idx_dims: Vec<usize> = shape.dims().iter().map(|d| d.unwrap_static()).collect();
        let idx = self.add_node(
            Op::Constant { data: idx_bytes },
            vec![],
            Shape::new(&idx_dims, DType::I32),
        );
        self.gather_(x, idx, last)
    }

    fn pad_axis_to_len(&mut self, x: NodeId, len: usize) -> NodeId {
        let shape = self.shape(x).clone();
        let last = shape.rank() - 1;
        let n = shape.dim(last).unwrap_static();
        if n >= len {
            return self.narrow_(x, last, 0, len);
        }
        let pad_len = len - n;
        let mut pad_dims: Vec<usize> = shape.dims().iter().map(|d| d.unwrap_static()).collect();
        pad_dims[last] = pad_len;
        let zeros = self.zeros_tensor(&Shape::new(&pad_dims, shape.dtype()));
        self.concat_(vec![x, zeros], last)
    }

    fn zeros_tensor(&mut self, shape: &Shape) -> NodeId {
        let n = shape.num_elements().unwrap();
        let bytes = vec![0u8; n * shape.dtype().size_bytes()];
        self.add_node(Op::Constant { data: bytes }, vec![], shape.clone())
    }

    fn scalar_f32(&mut self, v: f32) -> NodeId {
        self.add_node(
            Op::Constant {
                data: v.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::scalar(DType::F32),
        )
    }
}
