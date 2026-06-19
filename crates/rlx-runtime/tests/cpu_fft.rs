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

//! FFT primitive (`Op::Fft`) end-to-end tests on CPU.
//!
//! 2N real-block convention: complex `[..., N]` is stored as real
//! `[..., 2N]` with first N real, second N imag along the last axis.
//! Both forward and inverse are unnormalized — `ifft(fft(x)) = N·x`.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_opt::autodiff_fwd::jvp;
use rlx_runtime::{Device, Session};

fn f32s_to_bytes(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn f64s_to_bytes(xs: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 8);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
fn bytes_to_f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn const_f64(g: &mut Graph, xs: &[f64]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 8);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[xs.len()], DType::F64),
    )
}

/// Reference: naive O(N²) DFT for cross-checking.
fn dft_reference(re: &[f64], im: &[f64], inverse: bool) -> (Vec<f64>, Vec<f64>) {
    let n = re.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut out_re = vec![0f64; n];
    let mut out_im = vec![0f64; n];
    for k in 0..n {
        for nn in 0..n {
            let theta = sign * 2.0 * std::f64::consts::PI * (nn as f64) * (k as f64) / (n as f64);
            let c = theta.cos();
            let s = theta.sin();
            // (re[n] + i im[n]) * (c + i s)
            out_re[k] += re[nn] * c - im[nn] * s;
            out_im[k] += re[nn] * s + im[nn] * c;
        }
    }
    (out_re, out_im)
}

#[test]
fn fft_forward_matches_naive_dft() {
    let n: usize = 8;
    // Block layout: first N real, then N imag.
    let re = [1.0_f64, 0.5, -2.0, 0.25, 0.0, 1.5, -0.75, 3.0];
    let im = [0.5_f64, -1.0, 0.0, 2.0, -0.5, 0.25, 1.0, -1.5];
    let mut x_block = Vec::with_capacity(2 * n);
    x_block.extend_from_slice(&re);
    x_block.extend_from_slice(&im);

    let mut g = Graph::new("fft_fwd");
    let x = const_f64(&mut g, &x_block);
    let y = g.fft(x, false);
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[]);
    assert_eq!(outs.len(), 1);
    let y_got = bytes_to_f64s(&outs[0].0);
    assert_eq!(y_got.len(), 2 * n);

    let (exp_re, exp_im) = dft_reference(&re, &im, false);
    for k in 0..n {
        assert!(
            (y_got[k] - exp_re[k]).abs() < 1e-9,
            "fft re[{k}]: got {} vs ref {}",
            y_got[k],
            exp_re[k]
        );
        assert!(
            (y_got[n + k] - exp_im[k]).abs() < 1e-9,
            "fft im[{k}]: got {} vs ref {}",
            y_got[n + k],
            exp_im[k]
        );
    }
}

#[test]
fn fft_inverse_round_trip_recovers_n_times_input() {
    // ifft(fft(x)) = N·x  (unnormalized convention).
    let n: usize = 16;
    let re: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1 - 0.5).collect();
    let im: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.3).sin()).collect();
    let mut x_block = re.clone();
    x_block.extend_from_slice(&im);

    let mut g = Graph::new("fft_round_trip");
    let x = const_f64(&mut g, &x_block);
    let y = g.fft(x, false);
    let z = g.fft(y, true);
    g.set_outputs(vec![z]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[]);
    let z_got = bytes_to_f64s(&outs[0].0);

    for k in 0..n {
        let want_re = (n as f64) * re[k];
        let want_im = (n as f64) * im[k];
        assert!(
            (z_got[k] - want_re).abs() < 1e-9,
            "round-trip re[{k}]: got {} vs N·x = {}",
            z_got[k],
            want_re
        );
        assert!(
            (z_got[n + k] - want_im).abs() < 1e-9,
            "round-trip im[{k}]: got {} vs N·x = {}",
            z_got[n + k],
            want_im
        );
    }
}

#[test]
fn fft_vjp_is_inverse_fft() {
    // For y = fft(x), VJP gives dx = ifft(upstream). Build a graph
    // that computes loss = sum of squared real parts of fft(x), and
    // compare the autodiff gradient against a closed-form check
    // routed through the inverse-FFT identity.
    //
    // Concrete: L = sum(y_re²)
    //   dL/dy_re[k] = 2·y_re[k]
    //   dL/dy_im[k] = 0
    //   dL/dx = ifft(dL/dy)  (real-block view)
    let n: usize = 8;
    let re = [1.0_f64, 0.5, -2.0, 0.25, 0.0, 1.5, -0.75, 3.0];
    let im = [0.5_f64, -1.0, 0.0, 2.0, -0.5, 0.25, 1.0, -1.5];
    let mut x_block: Vec<f64> = re.iter().chain(im.iter()).copied().collect();
    let _ = &mut x_block;

    let mut g = Graph::new("fft_vjp");
    let x = g.input("x", Shape::new(&[2 * n], DType::F64));
    let y = g.fft(x, false);
    // L = sum(y_re²) — slice off the first N (real part), square,
    // sum. We need a real-only mask: build y_re via Narrow.
    let y_re = g.narrow_(y, 0, 0, n);
    let y_re_sq = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        y_re,
        y_re,
        Shape::new(&[n], DType::F64),
    );
    let loss = g.sum(y_re_sq, vec![0], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x]);
    assert_eq!(bwd.outputs.len(), 2, "[loss, dL/dx]");

    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let outs = compiled.run_typed(&[
        ("x", &f64s_to_bytes(&x_block), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let dx = bytes_to_f64s(&outs[1].0);
    assert_eq!(dx.len(), 2 * n);

    // Reference: dL/dy_re = 2·y_re, dL/dy_im = 0; dL/dx = ifft(dL/dy).
    let (y_re_ref, _y_im_ref) = dft_reference(&re, &im, false);
    let dl_dy_re: Vec<f64> = y_re_ref.iter().map(|v| 2.0 * v).collect();
    let dl_dy_im: Vec<f64> = vec![0.0; n];
    let (dl_dx_re_ref, dl_dx_im_ref) = dft_reference(&dl_dy_re, &dl_dy_im, true);

    for k in 0..n {
        assert!(
            (dx[k] - dl_dx_re_ref[k]).abs() < 5e-8,
            "dL/dx_re[{k}]: VJP={} vs ref={}",
            dx[k],
            dl_dx_re_ref[k]
        );
        assert!(
            (dx[n + k] - dl_dx_im_ref[k]).abs() < 5e-8,
            "dL/dx_im[{k}]: VJP={} vs ref={}",
            dx[n + k],
            dl_dx_im_ref[k]
        );
    }
}

#[test]
fn fft_f32_round_trip_recovers_n_times_input() {
    // f32 path parity check: same unnormalized convention, same
    // 2N-real-block layout, just lower precision. Tolerance loosened
    // to match radix-2 accumulation in single precision.
    let n: usize = 16;
    let re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
    let im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos() * 0.5).collect();
    let mut x_block: Vec<f32> = re.iter().chain(im.iter()).copied().collect();
    let _ = &mut x_block;

    let mut g = Graph::new("fft_f32_round_trip");
    let x = g.input("x", Shape::new(&[2 * n], DType::F32));
    let y = g.fft(x, false);
    let z = g.fft(y, true);
    g.set_outputs(vec![z]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[("x", &f32s_to_bytes(&x_block), DType::F32)]);
    let z_got = bytes_to_f32s(&outs[0].0);
    assert_eq!(z_got.len(), 2 * n);

    let nf = n as f32;
    for k in 0..n {
        let want_re = nf * re[k];
        let want_im = nf * im[k];
        assert!(
            (z_got[k] - want_re).abs() < 1e-3,
            "f32 round-trip re[{k}]: got {} vs N·x = {}",
            z_got[k],
            want_re
        );
        assert!(
            (z_got[n + k] - want_im).abs() < 1e-3,
            "f32 round-trip im[{k}]: got {} vs N·x = {}",
            z_got[n + k],
            want_im
        );
    }
}

#[test]
fn fft_bluestein_forward_matches_naive_dft_non_pow2() {
    // Bluestein path: N=6 (not a power of two) must still match the
    // naive DFT to the same tolerance as the radix-2 path.
    let n: usize = 6;
    let re = [1.0_f64, 0.5, -2.0, 0.25, 0.0, 1.5];
    let im = [0.5_f64, -1.0, 0.0, 2.0, -0.5, 0.25];
    let mut x_block: Vec<f64> = re.iter().chain(im.iter()).copied().collect();
    let _ = &mut x_block;

    let mut g = Graph::new("fft_bluestein_fwd");
    let x = g.input("x", Shape::new(&[2 * n], DType::F64));
    let y = g.fft(x, false);
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[("x", &f64s_to_bytes(&x_block), DType::F64)]);
    let y_got = bytes_to_f64s(&outs[0].0);

    let (y_re_ref, y_im_ref) = dft_reference(&re, &im, false);
    for k in 0..n {
        assert!(
            (y_got[k] - y_re_ref[k]).abs() < 1e-9,
            "bluestein fft re[{k}]: got {} vs ref {}",
            y_got[k],
            y_re_ref[k]
        );
        assert!(
            (y_got[n + k] - y_im_ref[k]).abs() < 1e-9,
            "bluestein fft im[{k}]: got {} vs ref {}",
            y_got[n + k],
            y_im_ref[k]
        );
    }
}

#[test]
fn fft_bluestein_round_trip_non_pow2() {
    // Round-trip identity ifft(fft(x)) = N·x must hold for non-pow2
    // sizes too. Cover several N values to exercise different
    // padding lengths M = next_pow2(2N-1).
    for &n in &[3usize, 5, 6, 7, 10, 12, 13, 15] {
        let re: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3 - 1.0).sin()).collect();
        let im: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos() * 0.5).collect();
        let x_block: Vec<f64> = re.iter().chain(im.iter()).copied().collect();

        let mut g = Graph::new("fft_bluestein_round_trip");
        let x = g.input("x", Shape::new(&[2 * n], DType::F64));
        let y = g.fft(x, false);
        let z = g.fft(y, true);
        g.set_outputs(vec![z]);

        let mut compiled = Session::new(Device::Cpu).compile(g);
        let outs = compiled.run_typed(&[("x", &f64s_to_bytes(&x_block), DType::F64)]);
        let z_got = bytes_to_f64s(&outs[0].0);

        let nf = n as f64;
        for k in 0..n {
            let want_re = nf * re[k];
            let want_im = nf * im[k];
            assert!(
                (z_got[k] - want_re).abs() < 1e-9,
                "N={n} round-trip re[{k}]: got {} vs N·x = {}",
                z_got[k],
                want_re
            );
            assert!(
                (z_got[n + k] - want_im).abs() < 1e-9,
                "N={n} round-trip im[{k}]: got {} vs N·x = {}",
                z_got[n + k],
                want_im
            );
        }
    }
}

#[test]
fn fft_bluestein_vjp_is_inverse_fft_non_pow2() {
    // VJP rule (VJP(fft)=ifft) must hold for non-pow2 N — the AD
    // doesn't know which kernel runs underneath. Mirrors
    // `fft_vjp_is_inverse_fft` but with N=6.
    let n: usize = 6;
    let re = [1.0_f64, 0.5, -2.0, 0.25, 0.0, 1.5];
    let im = [0.5_f64, -1.0, 0.0, 2.0, -0.5, 0.25];
    let mut x_block: Vec<f64> = re.iter().chain(im.iter()).copied().collect();
    let _ = &mut x_block;

    let mut g = Graph::new("fft_bluestein_vjp");
    let x = g.input("x", Shape::new(&[2 * n], DType::F64));
    let y = g.fft(x, false);
    let y_re = g.narrow_(y, 0, 0, n);
    let y_re_sq = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        y_re,
        y_re,
        Shape::new(&[n], DType::F64),
    );
    let loss = g.sum(y_re_sq, vec![0], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let outs = compiled.run_typed(&[
        ("x", &f64s_to_bytes(&x_block), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let dx = bytes_to_f64s(&outs[1].0);

    let (y_re_ref, _y_im_ref) = dft_reference(&re, &im, false);
    let dl_dy_re: Vec<f64> = y_re_ref.iter().map(|v| 2.0 * v).collect();
    let dl_dy_im: Vec<f64> = vec![0.0; n];
    let (dl_dx_re_ref, dl_dx_im_ref) = dft_reference(&dl_dy_re, &dl_dy_im, true);

    for k in 0..n {
        assert!(
            (dx[k] - dl_dx_re_ref[k]).abs() < 1e-8,
            "bluestein VJP re[{k}]: got {} vs ref={}",
            dx[k],
            dl_dx_re_ref[k]
        );
        assert!(
            (dx[n + k] - dl_dx_im_ref[k]).abs() < 1e-8,
            "bluestein VJP im[{k}]: got {} vs ref={}",
            dx[n + k],
            dl_dx_im_ref[k]
        );
    }
}

#[test]
fn fft_bluestein_f32_round_trip_non_pow2() {
    // f32 Bluestein path: looser tolerance than f64, same identity.
    let n: usize = 10;
    let re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
    let im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos() * 0.5).collect();
    let x_block: Vec<f32> = re.iter().chain(im.iter()).copied().collect();

    let mut g = Graph::new("fft_bluestein_f32");
    let x = g.input("x", Shape::new(&[2 * n], DType::F32));
    let y = g.fft(x, false);
    let z = g.fft(y, true);
    g.set_outputs(vec![z]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[("x", &f32s_to_bytes(&x_block), DType::F32)]);
    let z_got = bytes_to_f32s(&outs[0].0);

    let nf = n as f32;
    for k in 0..n {
        let want_re = nf * re[k];
        let want_im = nf * im[k];
        assert!(
            (z_got[k] - want_re).abs() < 5e-3,
            "f32 bluestein round-trip re[{k}]: got {} vs N·x = {}",
            z_got[k],
            want_re
        );
        assert!(
            (z_got[n + k] - want_im).abs() < 5e-3,
            "f32 bluestein round-trip im[{k}]: got {} vs N·x = {}",
            z_got[n + k],
            want_im
        );
    }
}

#[test]
fn fft_axis_non_last_matches_naive_dft() {
    // fft_axis(x, axis=0) should equal a manual transpose+fft+transpose.
    // Set up a 2D real-block tensor where the FFT axis (axis 0) carries
    // the 2N split: shape [2N, B], with B independent rows.
    let n: usize = 4;
    let b: usize = 3;
    // Per batch column b, the complex sequence is x[k, b] = re[k] + i·im[k].
    let mut re = vec![vec![0f64; b]; n];
    let mut im = vec![vec![0f64; b]; n];
    for bi in 0..b {
        for k in 0..n {
            re[k][bi] = (k as f64 + bi as f64 * 0.5).sin();
            im[k][bi] = (k as f64 * 0.3 - bi as f64).cos();
        }
    }
    // Pack into [2N, B] row-major: first N rows real, next N rows imag.
    let mut x_block = vec![0f64; 2 * n * b];
    for k in 0..n {
        for bi in 0..b {
            x_block[k * b + bi] = re[k][bi];
            x_block[(n + k) * b + bi] = im[k][bi];
        }
    }

    let mut g = Graph::new("fft_axis_0");
    let x = g.input("x", Shape::new(&[2 * n, b], DType::F64));
    let y = g.fft_axis(x, 0, false);
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[("x", &f64s_to_bytes(&x_block), DType::F64)]);
    let y_got = bytes_to_f64s(&outs[0].0);
    assert_eq!(y_got.len(), 2 * n * b);

    // Per batch column, compare against naive DFT.
    for bi in 0..b {
        let re_col: Vec<f64> = (0..n).map(|k| re[k][bi]).collect();
        let im_col: Vec<f64> = (0..n).map(|k| im[k][bi]).collect();
        let (y_re_ref, y_im_ref) = dft_reference(&re_col, &im_col, false);
        for k in 0..n {
            let got_re = y_got[k * b + bi];
            let got_im = y_got[(n + k) * b + bi];
            assert!(
                (got_re - y_re_ref[k]).abs() < 1e-9,
                "axis=0 batch={bi} re[{k}]: got {} vs ref {}",
                got_re,
                y_re_ref[k]
            );
            assert!(
                (got_im - y_im_ref[k]).abs() < 1e-9,
                "axis=0 batch={bi} im[{k}]: got {} vs ref {}",
                got_im,
                y_im_ref[k]
            );
        }
    }
}

#[test]
fn fft_axis_last_is_alias_for_fft() {
    // When axis == rank-1, fft_axis should be identical to fft (no
    // transposes inserted, same numerical result).
    let n: usize = 8;
    let re = [1.0_f64, 0.5, -2.0, 0.25, 0.0, 1.5, -0.75, 3.0];
    let im = [0.5_f64, -1.0, 0.0, 2.0, -0.5, 0.25, 1.0, -1.5];
    let x_block: Vec<f64> = re.iter().chain(im.iter()).copied().collect();

    let mut g1 = Graph::new("via_fft");
    let x1 = g1.input("x", Shape::new(&[2 * n], DType::F64));
    let y1 = g1.fft(x1, false);
    g1.set_outputs(vec![y1]);

    let mut g2 = Graph::new("via_fft_axis_last");
    let x2 = g2.input("x", Shape::new(&[2 * n], DType::F64));
    let y2 = g2.fft_axis(x2, 0, false); // rank=1, only axis is 0=last
    g2.set_outputs(vec![y2]);

    let mut c1 = Session::new(Device::Cpu).compile(g1);
    let mut c2 = Session::new(Device::Cpu).compile(g2);
    let o1 = c1.run_typed(&[("x", &f64s_to_bytes(&x_block), DType::F64)]);
    let o2 = c2.run_typed(&[("x", &f64s_to_bytes(&x_block), DType::F64)]);
    let v1 = bytes_to_f64s(&o1[0].0);
    let v2 = bytes_to_f64s(&o2[0].0);
    assert_eq!(v1, v2, "fft_axis(last) must be bit-identical to fft()");
}

#[test]
fn fft_jvp_matches_forward_of_tangent() {
    // FFT is linear, so JVP(fft(x), dx) = fft(dx) with the same
    // direction. Build a graph y = fft(x), JVP-transform it with x
    // seeded as the tangent input, and check the emitted tangent
    // output equals fft(dx) computed independently via the naive DFT.
    let n: usize = 8;
    let re = [1.0_f64, 0.5, -2.0, 0.25, 0.0, 1.5, -0.75, 3.0];
    let im = [0.5_f64, -1.0, 0.0, 2.0, -0.5, 0.25, 1.0, -1.5];
    let dx_re = [0.1_f64, -0.2, 0.05, 0.3, -0.4, 0.0, 0.25, -0.15];
    let dx_im = [0.0_f64, 0.5, -0.5, 0.1, 0.2, -0.3, 0.4, -0.1];
    let x_block: Vec<f64> = re.iter().chain(im.iter()).copied().collect();
    let dx_block: Vec<f64> = dx_re.iter().chain(dx_im.iter()).copied().collect();

    let mut g = Graph::new("fft_jvp");
    let x = g.input("x", Shape::new(&[2 * n], DType::F64));
    let y = g.fft(x, false);
    g.set_outputs(vec![y]);

    // JVP transform: emits a graph with outputs [primal, tangent]
    // and an extra `tangent_x` input alongside the original `x`.
    let jvp_graph = jvp(&g, &[x]);

    let mut compiled = Session::new(Device::Cpu).compile(jvp_graph);
    let outs = compiled.run_typed(&[
        ("x", &f64s_to_bytes(&x_block), DType::F64),
        ("tangent_x", &f64s_to_bytes(&dx_block), DType::F64),
    ]);
    let tangent = bytes_to_f64s(&outs[1].0);
    assert_eq!(tangent.len(), 2 * n);

    // Reference: tangent should equal fft(dx) (same direction).
    let (t_re_ref, t_im_ref) = dft_reference(&dx_re, &dx_im, false);
    for k in 0..n {
        assert!(
            (tangent[k] - t_re_ref[k]).abs() < 1e-9,
            "JVP re[{k}]: got {} vs fft(dx) = {}",
            tangent[k],
            t_re_ref[k]
        );
        assert!(
            (tangent[n + k] - t_im_ref[k]).abs() < 1e-9,
            "JVP im[{k}]: got {} vs fft(dx) = {}",
            tangent[n + k],
            t_im_ref[k]
        );
    }
}

fn bytes_to_c64_pairs(bytes: &[u8]) -> Vec<(f32, f32)> {
    bytes
        .chunks_exact(8)
        .map(|c| {
            (
                f32::from_le_bytes(c[..4].try_into().unwrap()),
                f32::from_le_bytes(c[4..].try_into().unwrap()),
            )
        })
        .collect()
}

#[test]
fn fft2_c64_round_trip() {
    // 2×2 complex field stored as C64 [H, W].
    let h = 2usize;
    let w = 2usize;
    let mut re_vals = Vec::new();
    let mut im_vals = Vec::new();
    for row in 0..h {
        for col in 0..w {
            re_vals.push((row + 1) as f32 * 0.5 + col as f32 * 0.1);
            im_vals.push((col + 1) as f32 * 0.25 - row as f32 * 0.05);
        }
    }

    let mut g = Graph::new("fft2_c64");
    let x = {
        let mut bytes = Vec::with_capacity(h * w * 8);
        for i in 0..(h * w) {
            bytes.extend_from_slice(&re_vals[i].to_le_bytes());
            bytes.extend_from_slice(&im_vals[i].to_le_bytes());
        }
        g.add_node(
            Op::Constant { data: bytes },
            vec![],
            Shape::new(&[h, w], DType::C64),
        )
    };
    let y = g.fftn(x, &[0, 1], false);
    let z = g.fftn(y, &[0, 1], true);
    g.set_outputs(vec![z]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run(&[]);
    let out_bytes: Vec<u8> = outs[0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let got = bytes_to_c64_pairs(&out_bytes);
    let scale = (h * w) as f32;
    for (i, ((gr, gi), (&xr, &xi))) in got
        .iter()
        .zip(re_vals.iter().zip(im_vals.iter()))
        .enumerate()
    {
        assert!(
            (gr - xr * scale).abs() < 1e-4,
            "re[{i}]: got {gr} vs expected {}",
            xr * scale
        );
        assert!(
            (gi - xi * scale).abs() < 1e-4,
            "im[{i}]: got {gi} vs expected {}",
            xi * scale
        );
    }
}

#[test]
fn fftn_single_axis_matches_fft() {
    let n = 8usize;
    let re = [1.0_f64, 0.5, -2.0, 0.25, 0.0, 1.5, -0.75, 3.0];
    let im = [0.5_f64, -1.0, 0.0, 2.0, -0.5, 0.25, 1.0, -1.5];
    let x_block: Vec<f64> = re.iter().chain(im.iter()).copied().collect();

    let mut g1 = Graph::new("fft");
    let x1 = g1.input("x", Shape::new(&[2 * n], DType::F64));
    let y1 = g1.fft(x1, false);
    g1.set_outputs(vec![y1]);

    let mut g2 = Graph::new("fftn");
    let x2 = g2.input("x", Shape::new(&[2 * n], DType::F64));
    let y2 = g2.fftn(x2, &[0], false);
    g2.set_outputs(vec![y2]);

    let mut c1 = Session::new(Device::Cpu).compile(g1);
    let mut c2 = Session::new(Device::Cpu).compile(g2);
    let o1 = c1.run_typed(&[("x", &f64s_to_bytes(&x_block), DType::F64)]);
    let o2 = c2.run_typed(&[("x", &f64s_to_bytes(&x_block), DType::F64)]);
    assert_eq!(bytes_to_f64s(&o1[0].0), bytes_to_f64s(&o2[0].0));
}

fn const_f32(g: &mut Graph, xs: &[f32]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[xs.len()], DType::F32),
    )
}

#[test]
fn fft_real_pads_to_pow2_and_splits_spectrum() {
    let signal = [1.0_f32, 2.0, 3.0]; // pads to N=4
    let mut g = Graph::new("fft_real");
    let x = const_f32(&mut g, &signal);
    let (re, im) = g.fft_real(x, rlx_ir::FftNorm::Backward);
    g.set_outputs(vec![re, im]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[]);
    let re_out = bytes_to_f32s(&outs[0].0);
    let im_out = bytes_to_f32s(&outs[1].0);
    assert_eq!(re_out.len(), 4);
    assert_eq!(im_out.len(), 4);

    let (exp_re, exp_im) = {
        let mut re = [0.0_f32; 4];
        re[0] = 1.0;
        re[1] = 2.0;
        re[2] = 3.0;
        let im = [0.0_f32; 4];
        let mut out_re = [0.0_f32; 4];
        let mut out_im = [0.0_f32; 4];
        for k in 0..4 {
            for nn in 0..4 {
                let theta = -2.0 * std::f64::consts::PI * (nn as f64) * (k as f64) / 4.0;
                let c = theta.cos() as f32;
                let s = theta.sin() as f32;
                out_re[k] += re[nn] * c - im[nn] * s;
                out_im[k] += re[nn] * s + im[nn] * c;
            }
        }
        (out_re, out_im)
    };
    for k in 0..4 {
        assert!((re_out[k] - exp_re[k]).abs() < 1e-5, "re[{k}]");
        assert!((im_out[k] - exp_im[k]).abs() < 1e-5, "im[{k}]");
    }
}

#[test]
fn ifft_spectrum_forward_norm_round_trip() {
    let signal = [1.0_f32, 0.0, 0.0, 0.0];
    let mut g = Graph::new("ifft_real");
    let x = const_f32(&mut g, &signal);
    let (re, im) = g.fft_real(x, rlx_ir::FftNorm::Forward);
    let recovered = g.ifft_spectrum(re, im, rlx_ir::FftNorm::Forward);
    g.set_outputs(vec![recovered]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[]);
    let got = bytes_to_f32s(&outs[0].0);
    for (i, (&g, &s)) in got.iter().zip(signal.iter()).enumerate() {
        assert!((g - s).abs() < 1e-5, "sample {i}: got {g} vs {s}");
    }
}

#[test]
fn psd_matches_squared_magnitude_over_n() {
    let re = [3.0_f32, 0.0, 4.0, 0.0];
    let im = [0.0_f32, 1.0, 0.0, 2.0];
    let mut g = Graph::new("psd");
    let re_n = const_f32(&mut g, &re);
    let im_n = const_f32(&mut g, &im);
    let p = g.psd(re_n, im_n);
    g.set_outputs(vec![p]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[]);
    let got = bytes_to_f32s(&outs[0].0);
    let n = 4.0_f32;
    for k in 0..4 {
        let want = (re[k] * re[k] + im[k] * im[k]) / n;
        assert!((got[k] - want).abs() < 1e-6, "psd[{k}]");
    }
}

#[test]
fn fft_batch_real_matches_sequential() {
    let batch = [[1.0_f32, 0.0, 0.0, 0.0], [0.0_f32, 1.0, 0.0, 0.0]];
    let flat: Vec<f32> = batch.iter().flat_map(|s| s.iter().copied()).collect();

    let mut g_batch = Graph::new("batch");
    let x = const_f32(&mut g_batch, &flat);
    let x = g_batch.reshape(x, vec![2, 4], Shape::new(&[2, 4], DType::F32));
    let (re_b, im_b) = g_batch.fft_batch_real(x, rlx_ir::FftNorm::Backward);
    g_batch.set_outputs(vec![re_b, im_b]);

    let mut compiled = Session::new(Device::Cpu).compile(g_batch);
    let outs = compiled.run_typed(&[]);
    let re_all = bytes_to_f32s(&outs[0].0);
    let im_all = bytes_to_f32s(&outs[1].0);
    assert_eq!(re_all.len(), 8);
    assert_eq!(im_all.len(), 8);

    for row in 0..2 {
        let mut g1 = Graph::new("one");
        let row_sig = const_f32(&mut g1, &batch[row]);
        let (re, im) = g1.fft_real(row_sig, rlx_ir::FftNorm::Backward);
        g1.set_outputs(vec![re, im]);
        let mut c1 = Session::new(Device::Cpu).compile(g1);
        let o1 = c1.run_typed(&[]);
        let re1 = bytes_to_f32s(&o1[0].0);
        let im1 = bytes_to_f32s(&o1[1].0);
        for k in 0..4 {
            assert!((re_all[row * 4 + k] - re1[k]).abs() < 1e-5);
            assert!((im_all[row * 4 + k] - im1[k]).abs() < 1e-5);
        }
    }
}

#[test]
fn fft_forward_norm_round_trip_is_identity() {
    let n: usize = 16;
    let re: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3 - 1.0).sin()).collect();
    let im: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos() * 0.5).collect();
    let x_block: Vec<f64> = re.iter().chain(im.iter()).copied().collect();

    let mut g = Graph::new("fft_forward_norm");
    let x = g.input("x", Shape::new(&[2 * n], DType::F64));
    let y = g.fft_norm(x, false, rlx_ir::FftNorm::Forward);
    let z = g.fft_norm(y, true, rlx_ir::FftNorm::Forward);
    g.set_outputs(vec![z]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[("x", &f64s_to_bytes(&x_block), DType::F64)]);
    let z_got = bytes_to_f64s(&outs[0].0);
    for k in 0..n {
        assert!(
            (z_got[k] - re[k]).abs() < 1e-9,
            "Forward round-trip re[{k}]: got {} vs {}",
            z_got[k],
            re[k]
        );
        assert!(
            (z_got[n + k] - im[k]).abs() < 1e-9,
            "Forward round-trip im[{k}]: got {} vs {}",
            z_got[n + k],
            im[k]
        );
    }
}

#[test]
fn fft_ortho_norm_round_trip_is_identity() {
    let n: usize = 16;
    let re: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
    let im: Vec<f64> = (0..n).map(|i| (i as f64 * 0.4).cos()).collect();
    let x_block: Vec<f64> = re.iter().chain(im.iter()).copied().collect();

    let mut g = Graph::new("fft_ortho_norm");
    let x = g.input("x", Shape::new(&[2 * n], DType::F64));
    let y = g.fft_norm(x, false, rlx_ir::FftNorm::Ortho);
    let z = g.fft_norm(y, true, rlx_ir::FftNorm::Ortho);
    g.set_outputs(vec![z]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[("x", &f64s_to_bytes(&x_block), DType::F64)]);
    let z_got = bytes_to_f64s(&outs[0].0);
    for k in 0..n {
        assert!((z_got[k] - re[k]).abs() < 1e-9, "Ortho round-trip re[{k}]");
        assert!(
            (z_got[n + k] - im[k]).abs() < 1e-9,
            "Ortho round-trip im[{k}]"
        );
    }
}

#[test]
fn fft_forward_norm_vjp_matches_scaled_inverse() {
    let n: usize = 8;
    let re = [1.0_f64, 0.5, -2.0, 0.25, 0.0, 1.5, -0.75, 3.0];
    let im = [0.5_f64, -1.0, 0.0, 2.0, -0.5, 0.25, 1.0, -1.5];
    let x_block: Vec<f64> = re.iter().chain(im.iter()).copied().collect();

    let mut g = Graph::new("fft_forward_vjp");
    let x = g.input("x", Shape::new(&[2 * n], DType::F64));
    let y = g.fft_norm(x, false, rlx_ir::FftNorm::Forward);
    let y_re = g.narrow_(y, 0, 0, n);
    let y_re_sq = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        y_re,
        y_re,
        Shape::new(&[n], DType::F64),
    );
    let loss = g.sum(y_re_sq, vec![0], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let outs = compiled.run_typed(&[
        ("x", &f64s_to_bytes(&x_block), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let dx = bytes_to_f64s(&outs[1].0);

    let (y_re_ref, _y_im_ref) = dft_reference(&re, &im, false);
    let dl_dy_re: Vec<f64> = y_re_ref.iter().map(|v| 2.0 * v).collect();
    let dl_dy_im: Vec<f64> = vec![0.0; n];
    let scale = rlx_ir::FftNorm::Forward.output_scale(n, false);
    let mut dl_dy_block = dl_dy_re.clone();
    dl_dy_block.extend(dl_dy_im.iter().copied());
    for v in dl_dy_block.iter_mut() {
        *v *= scale;
    }
    let (dl_dx_re_ref, dl_dx_im_ref) = dft_reference(&dl_dy_block[..n], &dl_dy_block[n..], true);
    for k in 0..n {
        assert!(
            (dx[k] - dl_dx_re_ref[k]).abs() < 5e-8,
            "Forward VJP re[{k}]: got {} vs ref={}",
            dx[k],
            dl_dx_re_ref[k]
        );
        assert!(
            (dx[n + k] - dl_dx_im_ref[k]).abs() < 5e-8,
            "Forward VJP im[{k}]: got {} vs ref={}",
            dx[n + k],
            dl_dx_im_ref[k]
        );
    }
}

#[test]
fn fft_ortho_norm_jvp_matches_forward_of_tangent() {
    let n: usize = 8;
    let re = [1.0_f64, 0.5, -2.0, 0.25, 0.0, 1.5, -0.75, 3.0];
    let im = [0.5_f64, -1.0, 0.0, 2.0, -0.5, 0.25, 1.0, -1.5];
    let dx_re = [0.1_f64, -0.2, 0.05, 0.3, -0.4, 0.0, 0.25, -0.15];
    let dx_im = [0.0_f64, 0.5, -0.5, 0.1, 0.2, -0.3, 0.4, -0.1];
    let x_block: Vec<f64> = re.iter().chain(im.iter()).copied().collect();
    let dx_block: Vec<f64> = dx_re.iter().chain(dx_im.iter()).copied().collect();

    let mut g = Graph::new("fft_ortho_jvp");
    let x = g.input("x", Shape::new(&[2 * n], DType::F64));
    let y = g.fft_norm(x, false, rlx_ir::FftNorm::Ortho);
    g.set_outputs(vec![y]);

    let jvp_graph = jvp(&g, &[x]);
    let mut compiled = Session::new(Device::Cpu).compile(jvp_graph);
    let outs = compiled.run_typed(&[
        ("x", &f64s_to_bytes(&x_block), DType::F64),
        ("tangent_x", &f64s_to_bytes(&dx_block), DType::F64),
    ]);
    let tangent = bytes_to_f64s(&outs[1].0);

    let (t_re_ref, t_im_ref) = dft_reference(&dx_re, &dx_im, false);
    let scale = rlx_ir::FftNorm::Ortho.output_scale(n, false);
    for k in 0..n {
        assert!(
            (tangent[k] - t_re_ref[k] * scale).abs() < 1e-6,
            "Ortho JVP re[{k}]: got {} vs {}",
            tangent[k],
            t_re_ref[k] * scale
        );
        assert!(
            (tangent[n + k] - t_im_ref[k] * scale).abs() < 1e-6,
            "Ortho JVP im[{k}]: got {} vs {}",
            tangent[n + k],
            t_im_ref[k] * scale
        );
    }
}

#[test]
fn rfft_irfft_forward_round_trip() {
    let signal = [1.0_f32, 2.0, 3.0, 0.5];
    let mut g = Graph::new("rfft_round_trip");
    let x = const_f32(&mut g, &signal);
    let (re, im) = g.rfft(x, rlx_ir::FftNorm::Forward);
    let recovered = g.irfft(re, im, signal.len(), rlx_ir::FftNorm::Forward);
    g.set_outputs(vec![recovered]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[]);
    let got = bytes_to_f32s(&outs[0].0);
    for (i, (&g, &s)) in got.iter().zip(signal.iter()).enumerate() {
        assert!((g - s).abs() < 1e-4, "sample {i}: got {g} vs {s}");
    }
}

#[test]
fn stft_single_frame_matches_rfft() {
    let frame = [0.0_f32, 1.0, 0.0, -1.0];
    let mut g1 = Graph::new("stft");
    let x = const_f32(&mut g1, &frame);
    let st = g1.stft(x, 4, 4, rlx_ir::FftNorm::Forward);
    g1.set_outputs(vec![st]);

    let mut g2 = Graph::new("rfft_only");
    let x2 = const_f32(&mut g2, &frame);
    let (re, im) = g2.rfft(x2, rlx_ir::FftNorm::Forward);
    let block = g2.concat_(vec![re, im], 0);
    let block = g2.reshape_(block, vec![1, 6]);
    g2.set_outputs(vec![block]);

    let mut c1 = Session::new(Device::Cpu).compile(g1);
    let mut c2 = Session::new(Device::Cpu).compile(g2);
    let o1 = bytes_to_f32s(&c1.run_typed(&[])[0].0);
    let o2 = bytes_to_f32s(&c2.run_typed(&[])[0].0);
    assert_eq!(o1.len(), o2.len());
    for (a, b) in o1.iter().zip(o2.iter()) {
        assert!((a - b).abs() < 1e-4, "stft vs rfft mismatch: {a} vs {b}");
    }
}

#[test]
fn fft_prime_factors_small_composite() {
    assert_eq!(rlx_ir::fft::prime_factors(12), vec![2, 2, 3]);
    assert_eq!(rlx_ir::fft::prime_factors(7), vec![7]);
}

#[test]
fn rfft_half_spectrum_length() {
    let signal = [1.0_f32, 2.0, 3.0, 0.5];
    let mut g = Graph::new("rfft_len");
    let x = const_f32(&mut g, &signal);
    let (re, im) = g.rfft(x, rlx_ir::FftNorm::Forward);
    g.set_outputs(vec![re, im]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[]);
    let re_out = bytes_to_f32s(&outs[0].0);
    let im_out = bytes_to_f32s(&outs[1].0);
    assert_eq!(re_out.len(), 3);
    assert_eq!(im_out.len(), 3);
}

#[test]
fn fftfreq_matches_numpy_convention() {
    let n = 8usize;
    let freqs = rlx_ir::fft::fftfreq(n);
    assert_eq!(freqs.len(), n);
    assert!((freqs[0] - 0.0).abs() < 1e-12);
    assert!((freqs[1] - 0.125).abs() < 1e-12);
    assert!((freqs[n - 1] + 0.125).abs() < 1e-12);

    let rf = rlx_ir::fft::rfftfreq(n);
    assert_eq!(rf.len(), n / 2 + 1);
    assert!((rf[0] - 0.0).abs() < 1e-12);
    assert!((rf[1] - 0.125).abs() < 1e-12);
}
