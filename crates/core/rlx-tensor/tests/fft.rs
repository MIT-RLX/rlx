// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! FFT — `[..., 2N]` block layout (real plane then imag plane). ndarray has
//! no FFT at all. Run: `cargo test -p rlx-tensor --features eval`.
#![cfg(feature = "eval")]

use rlx_tensor::Tensor;

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-4, "{a:?} != {b:?}");
    }
}

#[test]
fn dft_n2() {
    // x = [1, 2] (real), imag [0, 0] -> [re=[1,2], im=[0,0]] = [1,2,0,0]
    // DFT: X[0] = 3, X[1] = -1 -> [re=[3,-1], im=[0,0]] = [3,-1,0,0]
    let x = Tensor::from_vec(vec![1.0, 2.0, 0.0, 0.0], [4]);
    approx(&x.fft().to_vec(), &[3.0, -1.0, 0.0, 0.0]);
}

#[test]
fn dft_n4_impulse() {
    // impulse x = [1,0,0,0] -> flat spectrum (all ones, zero imag).
    // layout [2N=8]: re=[1,0,0,0], im=[0,0,0,0]
    let x = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [8]);
    let y = x.fft().to_vec();
    approx(&y[0..4], &[1.0, 1.0, 1.0, 1.0]); // real part all ones
    approx(&y[4..8], &[0.0, 0.0, 0.0, 0.0]); // imag part zero
}

#[test]
fn ifft_of_fft_is_n_times_input() {
    // Unnormalized: ifft(fft(x)) = N * x. N = 4.
    let re = [0.5_f32, -1.0, 2.0, 0.25];
    let mut data = re.to_vec();
    data.extend_from_slice(&[0.0; 4]); // imag plane
    let x = Tensor::from_vec(data, [8]);
    let round = x.fft().ifft().to_vec();
    let expect: Vec<f32> = re
        .iter()
        .map(|v| v * 4.0)
        .chain(std::iter::repeat_n(0.0, 4))
        .collect();
    approx(&round, &expect);
}
