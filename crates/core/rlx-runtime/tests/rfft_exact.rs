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

//! `Graph::rfft_exact` — exact arbitrary-`n` real FFT (DFT-matrix decomposition)
//! executed on CPU and cross-checked two ways:
//!   * vs a nested-loop O(n²) real DFT reference at the non-pow2 lengths the EEG
//!     tokenizers need (`n = 200` for CBraMod, `n = 400` for BrainBERT);
//!   * vs the crate's radix-2 [`Graph::rfft`] at a power-of-two length (where the
//!     zero-pad is a no-op, so the two paths must agree).

#![cfg(feature = "cpu")]

use rlx_ir::{DType, FftNorm, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

fn const_f32(g: &mut Graph, xs: &[f32]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[xs.len()], DType::F32),
    )
}

fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Deterministic bounded white-noise signal (LCG), values in ~[-0.05, 0.05] so no
/// single bin dominates and the f32 vs f64 absolute error stays well below 1e-5.
fn signal(n: usize) -> Vec<f32> {
    let mut s = 0x1234_5678u32;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.1
        })
        .collect()
}

/// Nested-loop O(n²) one-sided real DFT (n/2+1 bins), f64 accumulation.
/// re[k] = Σ x[t] cos(2π k t / n);  im[k] = −Σ x[t] sin(2π k t / n).
fn rfft_reference(x: &[f32], n: usize) -> (Vec<f64>, Vec<f64>) {
    let nf = n / 2 + 1;
    let mut re = vec![0f64; nf];
    let mut im = vec![0f64; nf];
    let two_pi = 2.0 * std::f64::consts::PI;
    for k in 0..nf {
        for t in 0..n {
            let ang = two_pi * (k as f64) * (t as f64) / (n as f64);
            re[k] += x[t] as f64 * ang.cos();
            im[k] += -(x[t] as f64) * ang.sin();
        }
    }
    (re, im)
}

/// Compile + run `rfft_exact` on CPU, returning `(re, im)` half-spectra.
fn run_rfft_exact(x: &[f32], n: usize, norm: FftNorm) -> (Vec<f32>, Vec<f32>) {
    let mut g = Graph::new("rfft_exact_test");
    let xin = const_f32(&mut g, x);
    let (re, im) = g.rfft_exact(xin, n, norm);
    g.set_outputs(vec![re, im]);
    let outs = Session::new(Device::Cpu).compile(g).run_typed(&[]);
    (bytes_to_f32s(&outs[0].0), bytes_to_f32s(&outs[1].0))
}

fn max_abs(a: &[f32], b: &[f64]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y as f32).abs())
        .fold(0f32, f32::max)
}

#[test]
fn rfft_exact_matches_nested_loop_dft_non_pow2() {
    for &n in &[200usize, 400] {
        let x = signal(n);
        let (re, im) = run_rfft_exact(&x, n, FftNorm::Backward);
        let nf = n / 2 + 1;
        assert_eq!(re.len(), nf, "n={n}: re bin count");
        assert_eq!(im.len(), nf, "n={n}: im bin count");
        let (ref_re, ref_im) = rfft_reference(&x, n);
        let e_re = max_abs(&re, &ref_re);
        let e_im = max_abs(&im, &ref_im);
        println!("rfft_exact n={n}: max_abs re={e_re:.3e} im={e_im:.3e}");
        assert!(e_re < 1e-5, "n={n}: re max_abs {e_re:.3e} >= 1e-5");
        assert!(e_im < 1e-5, "n={n}: im max_abs {e_im:.3e} >= 1e-5");
    }
}

#[test]
fn rfft_exact_matches_radix2_rfft_pow2() {
    // At a power of two the radix-2 `rfft` zero-pad is a no-op, so `rfft_exact`
    // (DFT-matrix matmul) and `rfft` (Op::Fft) must agree.
    let n = 256usize;
    let x = signal(n);

    let (re_e, im_e) = run_rfft_exact(&x, n, FftNorm::Backward);

    let mut g = Graph::new("radix2_rfft_test");
    let xin = const_f32(&mut g, &x);
    let (re_r, im_r) = g.rfft(xin, FftNorm::Backward);
    g.set_outputs(vec![re_r, im_r]);
    let outs = Session::new(Device::Cpu).compile(g).run_typed(&[]);
    let re_r = bytes_to_f32s(&outs[0].0);
    let im_r = bytes_to_f32s(&outs[1].0);

    assert_eq!(re_e.len(), re_r.len(), "bin count re");
    assert_eq!(im_e.len(), im_r.len(), "bin count im");
    let e_re = re_e
        .iter()
        .zip(&re_r)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let e_im = im_e
        .iter()
        .zip(&im_r)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("rfft_exact vs radix-2 rfft n={n}: max_abs re={e_re:.3e} im={e_im:.3e}");
    assert!(e_re < 1e-5, "re max_abs {e_re:.3e} >= 1e-5");
    assert!(e_im < 1e-5, "im max_abs {e_im:.3e} >= 1e-5");
}
