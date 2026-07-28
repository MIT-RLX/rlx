// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end CPU numeric tests for the exg-derived signal-processing graph
//! helpers (`vector_quantize`, `interpolate1d`, `conv_transpose1d`,
//! `spectrogram`, `band_power`, `envelope`, `fir_filtfilt`). Each builds a
//! constant-fed graph, runs it on the CPU backend, and checks the numbers.

#![cfg(feature = "cpu")]

use rlx_ir::ops::spectral::WindowKind;
use rlx_ir::ops::upsample::InterpMode;
use rlx_ir::ops::vq::VqMetric;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

fn const_f32(g: &mut Graph, xs: &[f32], dims: &[usize]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(dims, DType::F32),
    )
}

fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Run a compiled constant graph and return each output as an f32 vector.
fn run(g: Graph) -> Vec<Vec<f32>> {
    Session::new(Device::Cpu)
        .compile(g)
        .run_typed(&[])
        .iter()
        .map(|o| bytes_to_f32s(&o.0))
        .collect()
}

#[test]
fn vector_quantize_l2_selects_nearest_code() {
    let mut g = Graph::new("vq_l2");
    let cb = const_f32(&mut g, &[0.0, 0.0, 10.0, 0.0, 0.0, 10.0], &[3, 2]);
    let x = const_f32(&mut g, &[9.0, 1.0, 1.0, 9.0, 0.2, 0.1], &[3, 2]);
    let (idx, q) = g.vector_quantize(x, cb, VqMetric::L2);
    g.set_outputs(vec![idx, q]);
    let outs = run(g);

    assert_eq!(outs[0], vec![1.0, 2.0, 0.0], "nearest code ids");
    assert_eq!(
        outs[1],
        vec![10.0, 0.0, 0.0, 10.0, 0.0, 0.0],
        "gathered code vectors"
    );
}

#[test]
fn interpolate1d_linear_exact_ramp() {
    let mut g = Graph::new("interp");
    let x = const_f32(&mut g, &[0.0, 1.0, 2.0, 3.0], &[4]);
    let y = g.interpolate1d(x, 7, InterpMode::Linear);
    g.set_outputs(vec![y]);
    let out = &run(g)[0];
    let expect = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    assert_eq!(out.len(), 7);
    for (g, e) in out.iter().zip(expect.iter()) {
        assert!((g - e).abs() < 1e-5, "interp {g} vs {e}");
    }
}

#[test]
fn conv_transpose1d_overlap_add() {
    // input [1,1,2] = [1,1], kernel [1,1,2] = [1,1], stride 1.
    // out[t] = Σ_i in[i]·w[t−i] = [1, 2, 1].
    let mut g = Graph::new("ct1d");
    let x = const_f32(&mut g, &[1.0, 1.0], &[1, 1, 2]);
    let w = const_f32(&mut g, &[1.0, 1.0], &[1, 1, 2]);
    let y = g.conv_transpose1d(x, w, 2, 1, 0, 1, 0, 1);
    g.set_outputs(vec![y]);
    let out = &run(g)[0];
    assert_eq!(out.len(), 3);
    let expect = [1.0, 2.0, 1.0];
    for (g, e) in out.iter().zip(expect.iter()) {
        assert!((g - e).abs() < 1e-5, "ct1d {g} vs {e}");
    }
}

#[test]
fn spectrogram_dc_concentrates_in_bin0() {
    // All-ones frame → unnormalized rFFT DC bin = Σx = 64, power = 64².
    let mut g = Graph::new("spec_dc");
    let x = const_f32(&mut g, &[1.0; 64], &[1, 64]);
    let y = g.spectrogram(x, 64, 64, WindowKind::Rectangular, true, false);
    g.set_outputs(vec![y]);
    let out = &run(g)[0]; // [1 frame, 1 ch, 33 bins]
    assert_eq!(out.len(), 33);
    assert!((out[0] - 4096.0).abs() < 1.0, "DC power {} != 4096", out[0]);
    for (i, &p) in out.iter().enumerate().skip(1) {
        assert!(p < 1.0, "bin {i} power {p} should be ~0");
    }
}

#[test]
fn band_power_peaks_in_the_signal_band() {
    // 10 Hz sine, sr=64, T=64 → 1 Hz/bin, energy at bin 10 (band 8–12 Hz).
    let sr = 64.0f32;
    let x: Vec<f32> = (0..64)
        .map(|t| (2.0 * std::f32::consts::PI * 10.0 * t as f32 / sr).sin())
        .collect();
    let mut g = Graph::new("bp");
    let xn = const_f32(&mut g, &x, &[1, 64]);
    let bands = [(0.0, 5.0), (8.0, 12.0), (20.0, 32.0)];
    let y = g.band_power(xn, sr, &bands);
    g.set_outputs(vec![y]);
    let out = &run(g)[0]; // [1 ch, 3 bands]
    assert_eq!(out.len(), 3);
    assert!(
        out[1] > out[0] && out[1] > out[2],
        "band powers {out:?} — middle band should dominate"
    );
}

#[test]
fn envelope_of_cosine_recovers_amplitude() {
    // 2·cos(2π·8·t/64): analytic amplitude is 2 across the interior.
    let x: Vec<f32> = (0..64)
        .map(|t| 2.0 * (2.0 * std::f32::consts::PI * 8.0 * t as f32 / 64.0).cos())
        .collect();
    let mut g = Graph::new("env");
    let xn = const_f32(&mut g, &x, &[64]);
    let e = g.envelope(xn);
    g.set_outputs(vec![e]);
    let out = &run(g)[0];
    assert_eq!(out.len(), 64);
    // Interior mean envelope ≈ 2 (edges have FFT-boundary ripple).
    let interior: Vec<f32> = out[16..48].to_vec();
    let mean = interior.iter().sum::<f32>() / interior.len() as f32;
    assert!((mean - 2.0).abs() < 0.2, "envelope mean {mean} != 2");
}

#[test]
fn fir_filtfilt_preserves_dc() {
    // Unit-DC-gain taps on a constant signal → interior unchanged.
    let mut g = Graph::new("ff");
    let x = const_f32(&mut g, &[3.0; 128], &[128]);
    let taps = [0.25f32, 0.5, 0.25];
    let y = g.fir_filtfilt(x, &taps);
    g.set_outputs(vec![y]);
    let out = &run(g)[0];
    assert_eq!(out.len(), 128);
    assert!((out[64] - 3.0).abs() < 0.05, "filtfilt DC {} != 3", out[64]);
}

#[test]
fn biquad_matches_reference_recurrence() {
    // DF2T biquad vs a scalar reference loop.
    let xin = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [0.2f32, 0.4, 0.2];
    let a = [1.0f32, -0.3, 0.1];
    let mut g = Graph::new("biquad");
    let x = const_f32(&mut g, &xin, &[1, 8]);
    let y = g.biquad(x, b, a);
    g.set_outputs(vec![y]);
    let out = &run(g)[0];

    // Reference Direct-Form II Transposed.
    let (mut z1, mut z2) = (0.0f32, 0.0f32);
    let mut want = [0.0f32; 8];
    for (n, &xn) in xin.iter().enumerate() {
        let yn = b[0] * xn + z1;
        z1 = b[1] * xn - a[1] * yn + z2;
        z2 = b[2] * xn - a[2] * yn;
        want[n] = yn;
    }
    assert_eq!(out.len(), 8);
    for (g, e) in out.iter().zip(want.iter()) {
        assert!((g - e).abs() < 1e-5, "biquad {g} vs ref {e}");
    }
}

/// The `Op::Scan` → unroll decomposition (used to lower Scan for HLO/codegen
/// backends without a native scan: TPU/QNN/Cerebras/WebGL) must match the
/// native scan. Validated here on CPU so the untestable backends inherit a
/// checked lowering.
#[test]
fn scan_unroll_matches_native_biquad() {
    let xin: Vec<f32> = (0..32).map(|i| (i as f32 * 0.2).sin() + 0.1).collect();
    let b = [0.2f32, 0.4, 0.2];
    let a = [1.0f32, -0.3, 0.1];
    let build = || {
        let mut g = Graph::new("bq");
        let x = const_f32(&mut g, &xin, &[2, 16]); // [channels, samples]
        let y = g.biquad(x, b, a); // emits Op::Scan
        g.set_outputs(vec![y]);
        g
    };
    let native = run(build());
    // Force the unroll (as a Scan-less backend's legalization would).
    let unrolled_graph = rlx_fusion::control_flow::unroll_scan(build());
    assert!(
        !unrolled_graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Scan { .. })),
        "unroll must eliminate Op::Scan"
    );
    let unrolled = run(unrolled_graph);

    assert_eq!(native[0].len(), unrolled[0].len());
    for (a, b) in native[0].iter().zip(unrolled[0].iter()) {
        assert!((a - b).abs() < 1e-4, "unrolled {a} vs native {b}");
    }
}

#[test]
fn resample_poly_decimates_exactly() {
    // up=1, down=2, taps=[1] → keep every 2nd sample (phase 0).
    let xin: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mut g = Graph::new("rs");
    let x = const_f32(&mut g, &xin, &[8]);
    let y = g.resample_poly(x, 1, 2, &[1.0]);
    g.set_outputs(vec![y]);
    let out = &run(g)[0];
    assert_eq!(out, &vec![0.0, 2.0, 4.0, 6.0], "phase-0 decimation by 2");
}
