// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A fused wgpu `ElementwiseRegion` must compute every region-fusable
//! `Activation` exactly as the standalone `unary` kernel does.
//!
//! The region kernel had its own hand-written activation switch. It covered 17
//! of the 18 region-fusable activations: `Round` and `Recip` fell through to
//! `return x`, so a kept region silently skipped them, and `Sqrt`/`Rsqrt` were
//! WGSL's uncorrected forms. Region steps now run the generated rlxsl manifest
//! math under the region's own opcode numbering.
//!
//! Each chain is `act(x) * z`, which fuses into one region when regions are
//! kept (`RLX_KEEP_ELEMENTWISE_REGIONS`, read from the process environment;
//! this file holds one test so the flag cannot leak).

#![cfg(all(feature = "gpu", feature = "cpu"))]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

const N: usize = 4096;

fn chain(act: Activation) -> Graph {
    let s = Shape::new(&[N], DType::F32);
    let mut g = Graph::new("region_activation");
    let x = g.input("x", s.clone());
    let z = g.input("z", s.clone());
    let a = g.activation(act, x, s.clone());
    let y = g.binary(BinaryOp::Mul, a, z, s);
    g.set_outputs(vec![y]);
    g
}

/// Operands inside each activation's domain: positive for `Log`/`Sqrt`/
/// `Rsqrt`, away from zero for `Recip`, near ±π/2 avoided for `Tan`.
fn operands(act: Activation) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let u = ((i as f32 + 0.5) * 0.618_034).fract() * 2.0 - 1.0;
            match act {
                Activation::Log | Activation::Sqrt | Activation::Rsqrt => (u.abs() + 0.01) * 40.0,
                Activation::Recip => u.signum() * (u.abs() + 0.05) * 9.0,
                Activation::Tan => u * 1.2,
                Activation::Exp => u * 20.0,
                _ => u * 6.0,
            }
        })
        .collect()
}

/// ULPs between two floats of the same sign (`u32::MAX` otherwise).
fn ulps(a: f32, b: f32) -> u32 {
    if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
        return 0;
    }
    if !a.is_finite() || !b.is_finite() || (a < 0.0) != (b < 0.0) {
        return u32::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

/// Activations whose WGSL form is exact (pure arithmetic, `round`, and the
/// correctly-rounded `sqrt`/`1/sqrt`/`1/x`): bit-equal to rlx-cpu and to the
/// unfused kernel. The rest are transcendental — WGSL `exp`/`tanh`/`sin`/...
/// are not libm, and on discrete NVIDIA the *unfused* op runs on the host, on
/// libm — so they are held to 1e-5 of the output scale against both.
fn exact_on_wgpu(act: Activation) -> bool {
    matches!(
        act,
        Activation::Relu
            | Activation::Neg
            | Activation::Abs
            | Activation::Round
            | Activation::Recip
            | Activation::Sqrt
            | Activation::Rsqrt
    )
}

#[test]
fn every_region_fusable_activation_matches_the_unfused_kernel() {
    let Some(_gpu) = common::serialize_gpu() else {
        return;
    };
    let z: Vec<f32> = (0..N).map(|i| 1.0 + (i % 5) as f32 * 0.25).collect();
    let mut bad = Vec::new();
    for act in Activation::ALL.into_iter().filter(|a| a.region_fusable()) {
        let x = operands(act);
        let run = |d: Device, keep: bool| {
            if keep {
                rlx_ir::env::set("RLX_KEEP_ELEMENTWISE_REGIONS", "1");
            }
            let out = Session::new(d)
                .compile(chain(act))
                .run(&[("x", &x), ("z", &z)])
                .pop()
                .expect("one output");
            rlx_ir::env::remove("RLX_KEEP_ELEMENTWISE_REGIONS");
            out
        };
        let kept = run(Device::Gpu, true);
        let (unfused, cpu) = (run(Device::Gpu, false), run(Device::Cpu, false));
        // Transcendentals: error scaled by the output's magnitude, not ULPs —
        // `GeluApprox` near x≈-4 is `0.5·x·(1 + tanh)` with `1 + tanh ≈ 0`, a
        // cancellation where the unfused kernel is just as many ULPs away.
        let scale = cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
        let close = |a: f32, b: f32, tol: u32| {
            if tol == u32::MAX {
                (a - b).abs() <= 1e-5 * scale || ulps(a, b) == 0
            } else {
                ulps(a, b) <= tol
            }
        };
        let tol = if exact_on_wgpu(act) { 0 } else { u32::MAX };
        let mut check = |what: &str, want: &[f32], tol: u32| {
            let wrong: Vec<usize> = (0..N).filter(|&i| !close(kept[i], want[i], tol)).collect();
            if let Some(&i) = wrong.first() {
                bad.push(format!(
                    "{act:?} vs {what}: {} of {N} off (tol {tol}), e.g. x={} -> {:e} vs region {:e}",
                    wrong.len(),
                    x[i],
                    want[i],
                    kept[i]
                ));
            }
        };
        check("unfused wgpu", &unfused, tol);
        check("cpu", &cpu, tol);
    }
    assert!(bad.is_empty(), "{bad:#?}");
}
