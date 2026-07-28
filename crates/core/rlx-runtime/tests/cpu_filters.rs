// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end CPU numeric tests for the FIR / RIR / IIR filter builders
//! (`fir_conv1d`, `conv_reverb` / `partitioned_conv1d`, `iirfilt`,
//! `iir_as_fir`). Each builds a constant-fed graph, runs it on the CPU backend,
//! and checks the numbers against a straightforward host reference.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, FirMode, Graph, NodeId, Op, Shape, iir_impulse_response};
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

fn run(g: Graph) -> Vec<f32> {
    let outs = Session::new(Device::Cpu).compile(g).run_typed(&[]);
    bytes_to_f32s(&outs[0].0)
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, name: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: len {} != {}",
        got.len(),
        want.len()
    );
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let d = (a - b).abs();
        let rel = d / a.abs().max(b.abs()).max(1e-4);
        assert!(
            d < tol || rel < tol,
            "{name}[{i}]: got={a} want={b} (Δ={d})"
        );
    }
}

/// Full linear convolution reference, length `x.len() + h.len() - 1`.
fn conv_full(x: &[f32], h: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len() + h.len() - 1];
    for (i, &xi) in x.iter().enumerate() {
        for (j, &hj) in h.iter().enumerate() {
            y[i + j] += xi * hj;
        }
    }
    y
}

/// Slice `conv_full` for a given FIR mode.
fn fir_ref(x: &[f32], h: &[f32], mode: FirMode) -> Vec<f32> {
    let full = conv_full(x, h);
    let (l, k) = (x.len(), h.len());
    match mode {
        FirMode::Full => full,
        FirMode::Causal => full[0..l].to_vec(),
        FirMode::Same => {
            let s = (k - 1) / 2;
            full[s..s + l].to_vec()
        }
        FirMode::Valid => full[k - 1..k - 1 + (l - k + 1)].to_vec(),
    }
}

/// Direct-Form-II-Transposed IIR reference (matches `iirfilt` / `biquad`).
fn iir_ref(x: &[f32], b: &[f32], a: &[f32]) -> Vec<f32> {
    let m = b.len().max(a.len());
    let a0 = a[0];
    let bn: Vec<f32> = (0..m)
        .map(|i| b.get(i).copied().unwrap_or(0.0) / a0)
        .collect();
    let an: Vec<f32> = (0..m)
        .map(|i| a.get(i).copied().unwrap_or(0.0) / a0)
        .collect();
    let s = m - 1;
    let mut w = vec![0.0f32; s];
    let mut out = Vec::with_capacity(x.len());
    for &xn in x {
        let y = bn[0] * xn + if s > 0 { w[0] } else { 0.0 };
        let mut nw = vec![0.0f32; s];
        for i in 0..s {
            let mut wi = bn[i + 1] * xn - an[i + 1] * y;
            if i + 1 < s {
                wi += w[i + 1];
            }
            nw[i] = wi;
        }
        w = nw;
        out.push(y);
    }
    out
}

fn signal(n: usize) -> Vec<f32> {
    (0..n)
        .map(|t| (t as f32 * 0.3).sin() + 0.4 * (t as f32 * 0.11).cos())
        .collect()
}

// ---------------------------------------------------------------- FIR

#[test]
fn fir_direct_all_modes_match_reference() {
    let x = signal(64);
    let taps: Vec<f32> = (0..9).map(|i| 1.0 / (i as f32 + 1.0)).collect();
    for &mode in &[
        FirMode::Full,
        FirMode::Same,
        FirMode::Valid,
        FirMode::Causal,
    ] {
        let mut g = Graph::new("fir");
        let xn = const_f32(&mut g, &x, &[x.len()]);
        let y = g.fir_conv1d(xn, &taps, mode);
        g.set_outputs(vec![y]);
        assert_close(
            &run(g),
            &fir_ref(&x, &taps, mode),
            1e-4,
            &format!("fir_direct {mode:?}"),
        );
    }
}

#[test]
fn fir_direct_multichannel() {
    let r0 = signal(48);
    let r1: Vec<f32> = signal(48).iter().map(|v| 0.5 - v).collect();
    let mut flat = r0.clone();
    flat.extend_from_slice(&r1);
    let taps = [0.25f32, 0.5, 0.25];
    let mut g = Graph::new("firmc");
    let xn = const_f32(&mut g, &flat, &[2, 48]);
    let y = g.fir_conv1d(xn, &taps, FirMode::Causal);
    g.set_outputs(vec![y]);
    let mut want = fir_ref(&r0, &taps, FirMode::Causal);
    want.extend(fir_ref(&r1, &taps, FirMode::Causal));
    assert_close(&run(g), &want, 1e-4, "fir_direct_mc");
}

#[test]
fn fir_fft_path_matches_reference() {
    // 100 > FIR_DIRECT_MAX_TAPS → exercises the FFT convolution branch.
    let x = signal(130);
    let taps: Vec<f32> = (0..100)
        .map(|i| ((i as f32) * 0.02).cos() / (i as f32 + 1.0))
        .collect();
    for &mode in &[FirMode::Full, FirMode::Causal] {
        let mut g = Graph::new("firfft");
        let xn = const_f32(&mut g, &x, &[x.len()]);
        let y = g.fir_conv1d(xn, &taps, mode);
        g.set_outputs(vec![y]);
        assert_close(
            &run(g),
            &fir_ref(&x, &taps, mode),
            3e-3,
            &format!("fir_fft {mode:?}"),
        );
    }
}

// ---------------------------------------------------------------- RIR

#[test]
fn conv_reverb_multi_partition_matches_direct() {
    // ir 300 with block 64 → B=64, P=5 partitions, L not a multiple of B.
    let x = signal(220);
    let ir: Vec<f32> = (0..300)
        .map(|i| (-(i as f32) * 0.02).exp() * (i as f32 * 0.2).sin())
        .collect();
    let mut g = Graph::new("rir");
    let xn = const_f32(&mut g, &x, &[x.len()]);
    let y = g.conv_reverb(xn, &ir, 64);
    g.set_outputs(vec![y]);
    assert_close(&run(g), &conv_full(&x, &ir), 3e-3, "conv_reverb");
}

#[test]
fn conv_reverb_single_partition() {
    // block ≥ ir length → single partition (degenerate overlap-save).
    let x = signal(100);
    let ir: Vec<f32> = (0..40).map(|i| (-(i as f32) * 0.1).exp()).collect();
    let mut g = Graph::new("rir1");
    let xn = const_f32(&mut g, &x, &[x.len()]);
    let y = g.conv_reverb(xn, &ir, 64);
    g.set_outputs(vec![y]);
    assert_close(&run(g), &conv_full(&x, &ir), 3e-3, "conv_reverb_1part");
}

#[test]
fn conv_reverb_multichannel() {
    let r0 = signal(150);
    let r1: Vec<f32> = signal(150).iter().map(|v| 0.3 * v).collect();
    let mut flat = r0.clone();
    flat.extend_from_slice(&r1);
    let ir: Vec<f32> = (0..200).map(|i| (-(i as f32) * 0.03).exp()).collect();
    let mut g = Graph::new("rirmc");
    let xn = const_f32(&mut g, &flat, &[2, 150]);
    let y = g.conv_reverb(xn, &ir, 128);
    g.set_outputs(vec![y]);
    let mut want = conv_full(&r0, &ir);
    want.extend(conv_full(&r1, &ir));
    assert_close(&run(g), &want, 3e-3, "conv_reverb_mc");
}

// ---------------------------------------------------------------- IIR

#[test]
fn iirfilt_order2_matches_reference() {
    let x = signal(96);
    let b = [0.2f32, 0.4, 0.2];
    let a = [1.0f32, -0.3, 0.1];
    let mut g = Graph::new("iir2");
    let xn = const_f32(&mut g, &x, &[x.len()]);
    let y = g.iirfilt(xn, &b, &a);
    g.set_outputs(vec![y]);
    assert_close(&run(g), &iir_ref(&x, &b, &a), 1e-4, "iirfilt2");
}

#[test]
fn iirfilt_high_order_matches_reference() {
    let x = signal(128);
    let b = [0.05f32, 0.1, 0.1, 0.05];
    let a = [1.0f32, -0.7, 0.4, -0.1];
    let mut g = Graph::new("iir3");
    let xn = const_f32(&mut g, &x, &[x.len()]);
    let y = g.iirfilt(xn, &b, &a);
    g.set_outputs(vec![y]);
    assert_close(&run(g), &iir_ref(&x, &b, &a), 1e-4, "iirfilt3");
}

#[test]
fn iir_impulse_response_matches_recurrence() {
    let b = [0.2f32, 0.4, 0.2];
    let a = [1.0f32, -0.3, 0.1];
    let n = 40;
    let mut imp = vec![0.0f32; n];
    imp[0] = 1.0;
    assert_close(
        &iir_impulse_response(&b, &a, n),
        &iir_ref(&imp, &b, &a),
        1e-6,
        "iir_impulse",
    );
}

#[test]
fn iir_as_fir_approximates_iirfilt() {
    // Well-decayed IIR: truncated-IR FIR should match the true recurrence.
    let x = signal(120);
    let b = [0.15f32, 0.3, 0.15];
    let a = [1.0f32, -0.5, 0.2];
    let mut g = Graph::new("iirfir");
    let xn = const_f32(&mut g, &x, &[x.len()]);
    let y = g.iir_as_fir(xn, &b, &a, 96, FirMode::Causal);
    g.set_outputs(vec![y]);
    assert_close(&run(g), &iir_ref(&x, &b, &a), 3e-3, "iir_as_fir");
}

// ------------------------------------------------ RIR: GEMM path + Op::PartitionedConv

fn ir_node(g: &mut Graph, ir: &[f32]) -> NodeId {
    const_f32(g, ir, &[ir.len()])
}

#[test]
fn partitioned_conv1d_gemm_matches_direct() {
    let x = signal(220);
    let ir: Vec<f32> = (0..300)
        .map(|i| (-(i as f32) * 0.02).exp() * (i as f32 * 0.2).sin())
        .collect();
    let mut g = Graph::new("gemm");
    let xn = const_f32(&mut g, &x, &[x.len()]);
    let irn = ir_node(&mut g, &ir);
    let y = g.partitioned_conv1d_gemm(xn, irn, 64);
    g.set_outputs(vec![y]);
    assert_close(&run(g), &conv_full(&x, &ir), 3e-3, "gemm_mono");
}

#[test]
fn partitioned_conv1d_gemm_multichannel() {
    let r0 = signal(150);
    let r1: Vec<f32> = signal(150).iter().map(|v| 0.3 * v).collect();
    let mut flat = r0.clone();
    flat.extend_from_slice(&r1);
    let ir: Vec<f32> = (0..200).map(|i| (-(i as f32) * 0.03).exp()).collect();
    let mut g = Graph::new("gemmmc");
    let xn = const_f32(&mut g, &flat, &[2, 150]);
    let irn = ir_node(&mut g, &ir);
    let y = g.partitioned_conv1d_gemm(xn, irn, 128);
    g.set_outputs(vec![y]);
    let mut want = conv_full(&r0, &ir);
    want.extend(conv_full(&r1, &ir));
    assert_close(&run(g), &want, 3e-3, "gemm_mc");
}

#[test]
fn partitioned_conv_op_decomposes_and_matches_direct() {
    // Op::PartitionedConv node → unfuse → GEMM decomposition → CPU run.
    let x = signal(220);
    let ir: Vec<f32> = (0..300).map(|i| (-(i as f32) * 0.02).exp()).collect();
    let mut g = Graph::new("op");
    let xn = const_f32(&mut g, &x, &[x.len()]);
    let irn = ir_node(&mut g, &ir);
    let y = g.partitioned_conv(xn, irn, 64);
    g.set_outputs(vec![y]);
    assert_close(&run(g), &conv_full(&x, &ir), 3e-3, "op_partitioned_conv");
}

#[test]
fn partitioned_conv_op_equals_shift_builder() {
    // The fused op (GEMM decomposition) must agree with the eager shift-and-add
    // builder to tight tolerance (same algorithm, different accumulation order).
    let x = signal(180);
    let ir: Vec<f32> = (0..120).map(|i| (-(i as f32) * 0.05).exp()).collect();
    let mut g = Graph::new("op2");
    let xn = const_f32(&mut g, &x, &[x.len()]);
    let irn = ir_node(&mut g, &ir);
    let y = g.partitioned_conv(xn, irn, 64);
    g.set_outputs(vec![y]);
    let op_out = run(g);

    let mut g2 = Graph::new("shift");
    let xn2 = const_f32(&mut g2, &x, &[x.len()]);
    let irn2 = ir_node(&mut g2, &ir);
    let y2 = g2.partitioned_conv1d(xn2, irn2, 64);
    g2.set_outputs(vec![y2]);
    let shift_out = run(g2);

    assert_close(&op_out, &shift_out, 1e-4, "op_vs_shift");
}
