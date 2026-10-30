// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Fft` on CoreML (`fft_lower`): with the Neural Engine allowed — the
//! default for an FFT graph — it runs there as an f16 1×1-conv DFT; on CPU /
//! GPU it stays a host FFT (faster there). Checks placement and accuracy
//! against an f64 reference, and reports timings.
#![cfg(all(target_vendor = "apple", not(target_os = "watchos")))]

use std::time::{Duration, Instant};

use rlx_coreml::{ComputeUnits, CoremlExecutable, ane_available};
use rlx_ir::fft::FftNorm;
use rlx_ir::{DType, Graph, Op, Shape};

fn fft_graph(batch: usize, n: usize, inverse: bool, norm: FftNorm) -> Graph {
    let mut g = Graph::new("fft");
    let s = Shape::new(&[batch, 2 * n], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.add_node(
        Op::Fft {
            inverse,
            norm,
            algo: Default::default(),
        },
        vec![x],
        s,
    );
    g.set_outputs(vec![y]);
    g
}

fn reference(x: &[f32], batch: usize, n: usize, inverse: bool, norm: FftNorm) -> Vec<f64> {
    let s = if inverse { 1.0 } else { -1.0 };
    let scale = norm.output_scale(n, inverse);
    let tw: Vec<(f64, f64)> = (0..n)
        .map(|j| (2.0 * std::f64::consts::PI * j as f64 / n as f64).sin_cos())
        .collect();
    let mut out = vec![0f64; batch * 2 * n];
    for b in 0..batch {
        let row = &x[b * 2 * n..(b + 1) * 2 * n];
        for k in 0..n {
            let (mut re, mut im) = (0f64, 0f64);
            for t in 0..n {
                let (si, c) = tw[(t * k) % n];
                let (a, bb) = (row[t] as f64, row[n + t] as f64);
                re += a * c - bb * s * si;
                im += a * s * si + bb * c;
            }
            out[b * 2 * n + k] = re * scale;
            out[b * 2 * n + n + k] = im * scale;
        }
    }
    out
}

fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-12);
    got.iter()
        .zip(want)
        .map(|(g, w)| (*g as f64 - w).abs())
        .fold(0f64, f64::max)
        / scale
}

/// (output, median time, placement [cpu, gpu, ane, unknown] or None if host-only)
fn measure(mut ex: CoremlExecutable, x: &[f32]) -> (Vec<f32>, Duration, Option<[i32; 4]>) {
    let mut out = ex.run(&[("x", x)]).expect("run").remove(0);
    let plan = ex.compute_plan().ok().flatten();
    let mut t: Vec<Duration> = (0..20)
        .map(|_| {
            let s = Instant::now();
            out = ex.run(&[("x", x)]).expect("run").remove(0);
            s.elapsed()
        })
        .collect();
    t.sort();
    (out, t[10], plan)
}

fn eeg_rows(batch: usize, n: usize) -> Vec<f32> {
    (0..batch * 2 * n)
        .map(|i| {
            if i % (2 * n) < n {
                ((i * 7 + 1) as f32 * 0.03).sin() * 50.0
            } else {
                0.0
            }
        })
        .collect()
}

#[test]
fn fft_runs_on_the_neural_engine_by_default_and_on_the_host_for_cpu_gpu() {
    let (batch, n) = (32, 1024); // 32 EEG channels × 1024 samples
    let x = eeg_rows(batch, n);
    let want = reference(&x, batch, n, false, FftNorm::Backward);

    // Explicit CPU / GPU: the FFT stays a host op — exact, and faster there.
    for units in [ComputeUnits::CpuOnly, ComputeUnits::CpuAndGpu] {
        let (out, t, plan) = measure(
            CoremlExecutable::compile_with_units(
                fft_graph(batch, n, false, FftNorm::Backward),
                units,
            ),
            &x,
        );
        let e = rel_err(&out, &want);
        eprintln!("{units:?}: host FFT {t:>9.2?}  rel err {e:.2e}  placement {plan:?}");
        assert!(
            plan.is_none(),
            "{units:?}: expected a host-only FFT, got CoreML placement {plan:?}"
        );
        assert!(e < 1e-5, "{units:?}: rel err {e:.2e}");
    }

    // Default compile (what Device::Ane gives): the f16 DFT island, on the ANE.
    let (out, t, plan) = measure(
        CoremlExecutable::compile(fft_graph(batch, n, false, FftNorm::Backward)),
        &x,
    );
    let e = rel_err(&out, &want);
    eprintln!(
        "default (ANE): DFT conv f16 {t:>9.2?}  rel err {e:.2e}  placement [cpu,gpu,ane,?]={plan:?}"
    );
    assert!(e < 1e-3, "default: rel err {e:.2e}");
    let plan = plan.expect("the FFT must run inside CoreML, not on the host");
    if ane_available() {
        assert!(
            plan[2] > 0,
            "expected ops on the Neural Engine, placement {plan:?}"
        );
    }

    // Inverse / other norms through the same path, non-power-of-two length.
    for (inverse, norm) in [
        (true, FftNorm::Forward),
        (false, FftNorm::Ortho),
        (true, FftNorm::Backward),
    ] {
        let (b, n) = (8, 768);
        let x = eeg_rows(b, n);
        let (out, _, _) = measure(
            CoremlExecutable::compile(fft_graph(b, n, inverse, norm)),
            &x,
        );
        let e = rel_err(&out, &reference(&x, b, n, inverse, norm));
        eprintln!("default inverse={inverse} norm={norm:?}: rel err {e:.2e}");
        assert!(e < 1e-3, "inverse={inverse} norm={norm:?}: rel err {e:.2e}");
    }
}

/// The graph skill's EEG pipeline builds (`fft_batch_real`: concat zeros → FFT
/// → narrow re/im), through the public `Session` API on `Device::Ane`.
#[test]
fn eeg_fft_batch_real_through_session_on_ane() {
    use rlx_runtime::{Device, Session};
    let (batch, n) = (32usize, 1024usize);
    let mut g = Graph::new("eeg_fft");
    let x = g.input("x", Shape::new(&[batch, n], DType::F32));
    let (re, im) = g.fft_batch_real(x, FftNorm::Forward);
    g.set_outputs(vec![re, im]);
    let sig: Vec<f32> = (0..batch * n)
        .map(|i| ((i * 7 + 1) as f32 * 0.03).sin() * 50.0)
        .collect();
    let mut c = Session::new(Device::Ane).compile(g);
    let out = c.run(&[("x", &sig)]);
    // Reference: same rows in the 2N block layout (imaginary part zero).
    let mut block = vec![0f32; batch * 2 * n];
    for b in 0..batch {
        block[b * 2 * n..b * 2 * n + n].copy_from_slice(&sig[b * n..(b + 1) * n]);
    }
    let want = reference(&block, batch, n, false, FftNorm::Forward);
    let mut got = vec![0f32; batch * 2 * n];
    for b in 0..batch {
        got[b * 2 * n..b * 2 * n + n].copy_from_slice(&out[0][b * n..(b + 1) * n]);
        got[b * 2 * n + n..(b + 1) * 2 * n].copy_from_slice(&out[1][b * n..(b + 1) * n]);
    }
    let e = rel_err(&got, &want);
    eprintln!("Session(Device::Ane) fft_batch_real 32×1024: rel err {e:.2e}");
    assert!(e < 1e-3, "rel err {e:.2e}");
}
