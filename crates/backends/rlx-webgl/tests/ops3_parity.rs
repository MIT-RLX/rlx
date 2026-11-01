// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Parity for LayerNorm2d, Interpolate3d, FFT (matmul by a DFT matrix), RNG
// (precomputed at plan time) and host-staged linalg vs RLX's CPU backend.

use rlx_ir::fft::FftNorm;
use rlx_ir::{DType, Graph, Op, Shape};

fn f32s(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(f).collect()
}
fn sh(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

fn parity_tol(g: Graph, inputs: &[(&str, &[f32])], tol: f32) {
    let mut sess = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g.clone());
    let reference = sess.run(inputs);
    let plan = rlx_webgl::build_plan(&g).expect("plan");
    let got = rlx_webgl::run_cpu(&plan, inputs).expect("run_cpu");
    assert_eq!(reference.len(), got.len(), "output count");
    for (oi, (r, gv)) in reference.iter().zip(&got).enumerate() {
        assert_eq!(r.len(), gv.len(), "output {oi} length");
        for (i, (a, b)) in r.iter().zip(gv).enumerate() {
            assert!(
                (a - b).abs() <= tol * (1.0 + a.abs()),
                "output {oi}[{i}]: reference={a} webgl={b}"
            );
        }
    }
}

fn parity(g: Graph, inputs: &[(&str, &[f32])]) {
    parity_tol(g, inputs, 1e-4);
}

#[test]
fn layer_norm2d_over_channels() {
    let xs = f32s(2 * 4 * 3 * 3, |i| (i as f32 * 0.43).sin() * 2.0);
    let gs = f32s(4, |i| 0.5 + i as f32 * 0.25);
    let bs = f32s(4, |i| i as f32 * 0.1 - 0.2);
    let mut g = Graph::new("ln2d");
    let x = g.input("x", sh(&[2, 4, 3, 3]));
    let ga = g.input("g", sh(&[4]));
    let be = g.input("b", sh(&[4]));
    let y = g.add_node(
        Op::LayerNorm2d { eps: 1e-5 },
        vec![x, ga, be],
        sh(&[2, 4, 3, 3]),
    );
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("g", &gs), ("b", &bs)]);
}

#[test]
fn interpolate3d_nearest_up_and_down() {
    let xs = f32s(2 * 3 * 4 * 5, |i| i as f32);
    for size in [[4usize, 7, 9], [1, 2, 3]] {
        let mut g = Graph::new("interp3d");
        let x = g.input("x", sh(&[1, 2, 3, 4, 5]));
        let y = g.add_node(
            Op::Interpolate3d {
                size: size.to_vec(),
            },
            vec![x],
            sh(&[1, 2, size[0], size[1], size[2]]),
        );
        g.set_outputs(vec![y]);
        parity(g, &[("x", &xs)]);
    }
}

/// F32 2N-block layout: every norm, both directions; power-of-two lengths
/// (Stockham passes) and others (Bluestein), up to 1024.
#[test]
fn fft_block_layout_every_norm() {
    for n in [1usize, 2, 8, 6, 5, 12, 1000, 1024] {
        let xs = f32s(3 * 2 * n, |i| {
            (i as f32 * 0.37).sin() + 0.1 * (i % 17) as f32
        });
        for norm in [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho] {
            for inverse in [false, true] {
                let mut g = Graph::new("fft");
                let x = g.input("x", sh(&[3, 2 * n]));
                let y = g.fft_norm(x, inverse, norm);
                g.set_outputs(vec![y]);
                // Error grows with the transform's magnitude: compare against
                // the largest output of the reference, not element-wise.
                fft_parity(g, &[("x", &xs)], n);
            }
        }
    }
}

/// FFT parity with an error bound relative to the output's largest magnitude
/// (an O(n log n) f32 transform and the reference round differently; small
/// outputs next to large ones carry the large ones' absolute error).
fn fft_parity(g: Graph, inputs: &[(&str, &[f32])], n: usize) {
    let mut sess = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g.clone());
    let reference = sess.run(inputs);
    let plan = rlx_webgl::build_plan(&g).expect("plan");
    // No dense DFT matrix: every texture stays O(n) wide.
    assert!(
        plan.max_texture_side() <= 4 * (2 * n).next_power_of_two().max(8),
        "n={n}: texture side {}",
        plan.max_texture_side()
    );
    let got = rlx_webgl::run_cpu(&plan, inputs).expect("run_cpu");
    for (r, gv) in reference.iter().zip(&got) {
        assert_eq!(r.len(), gv.len());
        let peak = r.iter().fold(1f32, |m, v| m.max(v.abs()));
        for (i, (a, b)) in r.iter().zip(gv).enumerate() {
            assert!(
                (a - b).abs() <= 2e-5 * peak * (n as f32).log2().max(1.0),
                "n={n} [{i}]: reference={a} webgl={b} (peak {peak})"
            );
        }
    }
}

/// C64 interleaved layout: real → C64 → FFT → C64 → real part, plus the
/// round trip through the inverse.
#[test]
fn fft_complex_layout() {
    let n = 8usize;
    let xs = f32s(2 * n, |i| (i as f32 * 0.61).cos());
    let mut g = Graph::new("fft_c64");
    let x = g.input("x", sh(&[2, n]));
    let c = g.add_node(
        Op::Cast { to: DType::C64 },
        vec![x],
        Shape::new(&[2, n], DType::C64),
    );
    let f = g.fft(c, false);
    let back = g.fft_norm(f, true, FftNorm::Forward);
    let re_f = g.add_node(Op::Cast { to: DType::F32 }, vec![f], sh(&[2, n]));
    let re_b = g.add_node(Op::Cast { to: DType::F32 }, vec![back], sh(&[2, n]));
    g.set_outputs(vec![re_f, re_b]);
    parity_tol(g, &[("x", &xs)], 2e-4);
}

/// C64 interleaved layout with a non-power-of-two length (Bluestein).
#[test]
fn fft_complex_layout_bluestein() {
    let n = 10usize;
    let xs = f32s(2 * n, |i| (i as f32 * 0.29).sin());
    let mut g = Graph::new("fft_c64_b");
    let x = g.input("x", sh(&[2, n]));
    let c = g.add_node(
        Op::Cast { to: DType::C64 },
        vec![x],
        Shape::new(&[2, n], DType::C64),
    );
    let f = g.fft_norm(c, true, FftNorm::Ortho);
    let re = g.add_node(Op::Cast { to: DType::F32 }, vec![f], sh(&[2, n]));
    g.set_outputs(vec![re]);
    fft_parity(g, &[("x", &xs)], n);
}

/// Counter-based RNG: fixed by the graph, so WebGL precomputes it.
#[test]
fn rng_uniform_and_normal() {
    let mut g = Graph::new("rng");
    let x = g.input("x", sh(&[4, 6]));
    let u = g.add_node(
        Op::RngUniform {
            low: -2.0,
            high: 3.0,
            key: 11,
            op_seed: None,
        },
        vec![x],
        sh(&[4, 6]),
    );
    let nrm = g.add_node(
        Op::RngNormal {
            mean: 1.0,
            scale: 0.5,
            key: 12,
            op_seed: None,
        },
        vec![x],
        sh(&[4, 6]),
    );
    g.set_outputs(vec![u, nrm]);
    parity(g, &[("x", &[0.0; 24])]);
}

/// SPD `[n, n]`: diagonally dominant with small symmetric off-diagonals.
fn spd(n: usize) -> Vec<f32> {
    f32s(n * n, |i| {
        let (r, c) = (i / n, i % n);
        if r == c {
            3.0 + r as f32
        } else {
            0.4 / (1.0 + (r + c) as f32)
        }
    })
}

#[test]
fn cholesky_det_logdet() {
    let n = 5;
    let a = spd(n);
    // A general (non-symmetric, negative-det) matrix for Det / LogDet.
    let m = f32s(n * n, |i| {
        ((i * 37 % 11) as f32 - 5.0) * 0.3 + if i % 6 == 0 { 2.0 } else { 0.0 }
    });
    let mut g = Graph::new("chol");
    let ai = g.input("a", sh(&[n, n]));
    let mi = g.input("m", sh(&[n, n]));
    let l = g.cholesky(ai, sh(&[n, n]));
    let d = g.det(mi, sh(&[]));
    let ld = g.logdet(mi, sh(&[]));
    g.set_outputs(vec![l, d, ld]);
    parity(g, &[("a", &a), ("m", &m)]);
}

#[test]
fn triangular_solve_all_forms() {
    let n = 4;
    let a = f32s(n * n, |i| {
        if i % (n + 1) == 0 {
            2.0 + i as f32 * 0.1
        } else {
            0.3 * ((i % 5) as f32 - 2.0)
        }
    });
    let b = f32s(n * 3, |i| (i as f32 * 0.7).sin());
    for lower in [true, false] {
        for transpose in [false, true] {
            let mut g = Graph::new("trsm");
            let ai = g.input("a", sh(&[n, n]));
            let bi = g.input("b", sh(&[n, 3]));
            let x = g.triangular_solve(ai, bi, lower, transpose, sh(&[n, 3]));
            g.set_outputs(vec![x]);
            parity(g, &[("a", &a), ("b", &b)]);
        }
    }
}

#[test]
fn dense_and_batched_solve() {
    let n = 4;
    // Needs pivoting: a zero leading entry.
    let a = f32s(n * n, |i| {
        if i == 0 {
            0.0
        } else {
            ((i * 13 % 7) as f32 - 3.0) * 0.5 + if i % 5 == 0 { 3.0 } else { 0.0 }
        }
    });
    let bv = f32s(n, |i| i as f32 - 1.5);
    let bm = f32s(n * 2, |i| (i as f32 * 0.9).cos());
    let mut g = Graph::new("solve");
    let ai = g.input("a", sh(&[n, n]));
    let b1 = g.input("bv", sh(&[n]));
    let b2 = g.input("bm", sh(&[n, 2]));
    let x1 = g.dense_solve(ai, b1, sh(&[n]));
    let x2 = g.dense_solve(ai, b2, sh(&[n, 2]));
    g.set_outputs(vec![x1, x2]);
    parity(g, &[("a", &a), ("bv", &bv), ("bm", &bm)]);

    let ab: Vec<f32> = [spd(n), a.clone(), spd(n)].concat();
    let bb = f32s(3 * n * 2, |i| i as f32 * 0.25 - 1.0);
    let mut g = Graph::new("bsolve");
    let ai = g.input("a", sh(&[3, n, n]));
    let bi = g.input("b", sh(&[3, n, 2]));
    let x = g.batched_dense_solve(ai, bi, sh(&[3, n, 2]));
    g.set_outputs(vec![x]);
    parity(g, &[("a", &ab), ("b", &bb)]);
}
