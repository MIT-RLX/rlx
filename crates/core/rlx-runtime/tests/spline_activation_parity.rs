// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::SplineActivation` (KAN Gaussian-RBF spline) forward parity:
//!   1. native CPU kernel vs a hand-written RBF reference,
//!   2. the decompose oracle (Reshape/Expand/Sub/Mul/Exp + ReduceSum) vs native,
//!   3. Metal vs CPU — the oracle is all-f32, so (unlike SynthMatMul's U8-index
//!      decompose) it runs correctly on Metal even before a native kernel.

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const ROWS: usize = 4;
const CH: usize = 3;
const NB: u32 = 6;
const GMIN: f32 = -2.0;
const GMAX: f32 = 2.0;

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn x_data() -> Vec<f32> {
    (0..ROWS * CH)
        .map(|i| (i as f32 * 0.7).sin() * 1.3)
        .collect()
}
fn coeff_data() -> Vec<f32> {
    (0..CH * NB as usize)
        .map(|i| (i as f32 * 0.37).cos() * 0.9)
        .collect()
}

fn reference(x: &[f32], coeff: &[f32]) -> Vec<f32> {
    let nb = NB as usize;
    let step = (GMAX - GMIN) / (nb as f32 - 1.0);
    let inv_h = 1.0 / step;
    let mut out = vec![0f32; ROWS * CH];
    for r in 0..ROWS {
        for c in 0..CH {
            let xv = x[r * CH + c];
            let mut acc = 0f32;
            for gi in 0..nb {
                let center = GMIN + gi as f32 * step;
                let z = (xv - center) * inv_h;
                acc += coeff[c * nb + gi] * (-(z * z)).exp();
            }
            out[r * CH + c] = acc;
        }
    }
    out
}

fn build() -> Graph {
    let mut g = Graph::new("spline");
    let x = g.input("x", Shape::new(&[ROWS, CH], DType::F32));
    let coeff = g.input("coeff", Shape::new(&[CH, NB as usize], DType::F32));
    let y = g.spline_activation(x, coeff, NB, GMIN, GMAX);
    g.set_outputs(vec![y]);
    g
}

fn run(device: Device, x: &[f32], coeff: &[f32]) -> Vec<f32> {
    Session::new(device)
        .compile(build())
        .run(&[("x", x), ("coeff", coeff)])
        .into_iter()
        .next()
        .unwrap()
}

#[test]
fn cpu_native_matches_reference() {
    let (x, coeff) = (x_data(), coeff_data());
    let out = run(Device::Cpu, &x, &coeff);
    let want = reference(&x, &coeff);
    let err = max_abs_err(&out, &want);
    eprintln!("spline native vs reference: err={err:e}");
    assert!(err < 1e-5, "native diverges from reference: err {err}");
}

#[test]
fn decompose_matches_native() {
    use rlx_fusion::LowerSplineActivation;
    use rlx_fusion::pass::Pass;

    let (x, coeff) = (x_data(), coeff_data());
    let native = run(Device::Cpu, &x, &coeff);

    let lowered = LowerSplineActivation.run(build());
    assert!(
        !lowered
            .nodes()
            .iter()
            .any(|nd| matches!(nd.op, rlx_ir::Op::SplineActivation { .. })),
        "decompose left a SplineActivation node"
    );
    let decomp = Session::new(Device::Cpu)
        .compile(lowered)
        .run(&[("x", &x), ("coeff", &coeff)])
        .into_iter()
        .next()
        .unwrap();
    let err = max_abs_err(&native, &decomp);
    eprintln!("spline decompose vs native: err={err:e}");
    assert!(err < 1e-4, "decompose diverges from native: err {err}");
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let (x, coeff) = (x_data(), coeff_data());
    let cpu = run(Device::Cpu, &x, &coeff);
    let met = run(Device::Metal, &x, &coeff);
    let err = max_abs_err(&cpu, &met);
    eprintln!("spline metal vs cpu: err={err:e}");
    assert!(err < 1e-4, "metal diverges from cpu: err {err}");
}
