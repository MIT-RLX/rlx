// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device-resident training on wgpu, step by step against CPU.
//!
//! `rlx_runtime::train` is `#[cfg(feature = "training")]` and the run needs a
//! wgpu device, so the whole target is gated on both. Without the gate this file
//! fails to compile under any feature set lacking `training` — which takes the
//! entire test binary down, not just this case.
#![cfg(all(feature = "training", feature = "gpu"))]
//!
//! The fused update appends `m'`, `v'` and `p'` to the backward graph and keeps
//! parameters and moments in device buffers across steps — the whole point is
//! that nothing round-trips to the host. That makes it a read-after-write
//! question: step N writes `p'`, step N+1 must read what step N wrote.
//!
//! It did not. Parameters after ONE step matched CPU to 3e-8, and the
//! trajectory then diverged from step 2 by ~3e-3 — the signature of the next
//! step reading stale values rather than of arithmetic drift. Checking only the
//! final parameters would show a vague mismatch; checking every step says
//! exactly which one first disagrees, which is the difference between "wgpu is
//! inaccurate" and "the write-back does not land".

use std::collections::HashMap;

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::train::{AdamSpec, ResidentTrainer, TrainableParam};
use rlx_runtime::{Device, is_available};

const DIM: usize = 4;
const BATCH: usize = 2;

fn regression() -> (Graph, Vec<TrainableParam>) {
    let mut g = Graph::new("mse");
    let x = g.input("x", Shape::new(&[BATCH, DIM], DType::F32));
    let t = g.input("t", Shape::new(&[BATCH, DIM], DType::F32));
    let w = g.param("w", Shape::new(&[DIM, DIM], DType::F32));
    let b = g.param("b", Shape::new(&[DIM], DType::F32));
    let xw = g.matmul(x, w, Shape::new(&[BATCH, DIM], DType::F32));
    let y = g.add(xw, b);
    let d = g.sub(y, t);
    let sq = g.mul(d, d);
    let loss = g.mean(sq, vec![0, 1], false);
    g.set_outputs(vec![loss]);
    let wrt = vec![
        TrainableParam {
            name: "w".into(),
            node: w,
        },
        TrainableParam {
            name: "b".into(),
            node: b,
        },
    ];
    (g, wrt)
}

fn initial() -> HashMap<String, Vec<f32>> {
    let w: Vec<f32> = (0..DIM * DIM).map(|i| 0.1 * (i as f32).sin()).collect();
    let b: Vec<f32> = (0..DIM).map(|i| 0.05 * (i as f32).cos()).collect();
    HashMap::from([("w".to_string(), w), ("b".to_string(), b)])
}

#[test]
fn wgpu_resident_training_tracks_cpu_step_by_step() {
    if rlx_ir::env::skip_unless_device("wgpu", true, is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let spec = AdamSpec {
        lr: 0.05,
        weight_decay: 0.01,
        ..AdamSpec::default()
    };
    let x: Vec<f32> = vec![1.0, 0.5, -0.5, 2.0, 0.25, -1.0, 1.5, 0.0];
    let t: Vec<f32> = vec![1.0, -1.0, 0.5, 0.0, 0.0, 2.0, -1.0, 1.0];

    let mk = |device| {
        let (fwd, wrt) = regression();
        ResidentTrainer::new(&fwd, &wrt, &initial(), &spec, device).expect("fuses")
    };
    let mut cpu = mk(Device::Cpu);
    let mut gpu = mk(Device::Gpu);
    assert!(
        gpu.is_resident(),
        "wgpu trainer did not bind device handles"
    );

    let mut first_bad: Option<(usize, String, f32)> = None;
    for step in 1..=8 {
        let lc = cpu.step(&[("x", &x), ("t", &t)])[0][0];
        let lg = gpu.step(&[("x", &x), ("t", &t)])[0][0];
        let (pc, pg) = (cpu.params(), gpu.params());
        let mut worst = (lc - lg).abs();
        let mut who = "loss".to_string();
        let mut per: Vec<(String, f32)> = Vec::new();
        for (name, c) in &pc {
            let g = &pg[name];
            let d = c
                .iter()
                .zip(g)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            per.push((name.clone(), d));
            if d > worst {
                worst = d;
                who = name.clone();
            }
        }
        // p, m and v are three separate write-backs; compare each.
        let (mc, mg) = (cpu.moments(), gpu.moments());
        for ((n, m1, v1), (_, m2, v2)) in mc.iter().zip(&mg) {
            let dm = m1
                .iter()
                .zip(m2)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let dv = v1
                .iter()
                .zip(v2)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            per.push((format!("{n}.m"), dm));
            per.push((format!("{n}.v"), dv));
        }
        per.sort_by(|a, b| a.0.cmp(&b.0));
        let detail: Vec<String> = per.iter().map(|(n, d)| format!("{n} {d:.2e}")).collect();
        eprintln!(
            "step {step}: w0 cpu {:.6} gpu {:.6} | b0 cpu {:.6} gpu {:.6} | {}.m0 cpu {:.6} gpu {:.6} | {}.v0 cpu {:.8} gpu {:.8} | worst {worst:.2e} ({who})",
            pc["w"][0],
            pg["w"][0],
            pc["b"][0],
            pg["b"][0],
            mc[0].0,
            mc[0].1[0],
            mg[0].1[0],
            mc[0].0,
            mc[0].2[0],
            mg[0].2[0]
        );
        let _ = &detail;
        if worst > 1e-5 && first_bad.is_none() {
            first_bad = Some((step, who, worst));
        }
    }
    if let Some((step, who, worst)) = first_bad {
        panic!(
            "wgpu resident training first diverges from CPU at STEP {step} \
             (worst |Δ| {worst:.3e} in `{who}`). Divergence that starts at a \
             specific step, rather than growing from step 1, means the previous \
             step's write-back did not reach the next step's read."
        );
    }
}
