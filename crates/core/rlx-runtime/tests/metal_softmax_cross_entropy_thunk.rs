// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Same parity as `metal_softmax_cross_entropy`, but with MPSGraph
//! disabled so the graph is forced down the thunk path — this pins the
//! native MSL `softmax_cross_entropy_dense` kernel specifically (the
//! default test now lowers through MPSGraph). Own test binary so the
//! env override stays process-local.

#![cfg(all(target_os = "macos", feature = "metal"))]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const N: usize = 6;
const C: usize = 7;

fn build() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("metal_sce_thunk");
    let logits = g.input("logits", Shape::new(&[N, C], f));
    let targets = g.input("targets", Shape::new(&[N, C], f));
    let loss = g.softmax_cross_entropy(logits, targets);
    g.set_outputs(vec![loss]);
    g
}

fn sample_logits() -> Vec<f32> {
    (0..N * C)
        .map(|i| 0.4 * (i as f32) - 1.3 * ((i % 3) as f32) - 0.7)
        .collect()
}

fn sample_targets() -> Vec<f32> {
    let mut t = vec![0f32; N * C];
    for n in 0..N {
        let raw: Vec<f32> = (0..C).map(|c| 0.2 + 0.5 * (((n + c) % 4) as f32)).collect();
        let s: f32 = raw.iter().sum();
        for c in 0..C {
            t[n * C + c] = raw[c] / s;
        }
    }
    t
}

fn ref_loss(logits: &[f32], targets: &[f32]) -> Vec<f32> {
    (0..N)
        .map(|n| {
            let row = &logits[n * C..(n + 1) * C];
            let trow = &targets[n * C..(n + 1) * C];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&v| (v - m).exp()).sum();
            let lse = m + sum.ln();
            let dot: f32 = (0..C).map(|c| trow[c] * row[c]).sum();
            lse - dot
        })
        .collect()
}

#[test]
fn metal_sce_msl_thunk_kernel_matches_reference() {
    // Force the thunk path so the native MSL kernel runs (not MPSGraph).
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let logits = sample_logits();
    let targets = sample_targets();
    let inputs: &[(&str, &[f32])] = &[("logits", &logits), ("targets", &targets)];

    let metal = Session::new(Device::Metal).compile(build()).run(inputs)[0].clone();
    let want = ref_loss(&logits, &targets);

    assert_eq!(metal.len(), N);
    for n in 0..N {
        assert!(
            (metal[n] - want[n]).abs() < 1e-4,
            "row {n}: metal-msl {} vs ref {}",
            metal[n],
            want[n]
        );
    }
}
