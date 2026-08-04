// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-seq` — a decode-like stepped driver. A single matmul site is run
//! over `T` successive steps with a **fixed weight** and **drifting
//! activations** (a random walk). Records `step_idx` so the miner can see the
//! cross-call time series: the weight's sketches never change (→ stationary →
//! precompute/prepack) while the activation's drift slowly (→ temporal
//! coherence across steps).
//!
//! Usage: `opscope-seq [out.csv] [steps]`  (defaults: `opscope_seq.csv`, 16)

use rlx_ir::{DType, Graph, Philox4x32, Shape};
use rlx_opscope::{Recorder, StatConfig, inject_matmul_stats};
use rlx_runtime::{Device, Session};

fn matmul_graph(b: usize, d: usize) -> Graph {
    let mut g = Graph::new("decode_matmul");
    let a = g.input("A", Shape::new(&[b, d], DType::F32));
    let w = g.input("W", Shape::new(&[d, d], DType::F32));
    let c = g.matmul(a, w, Shape::new(&[b, d], DType::F32));
    g.set_outputs(vec![c]);
    g
}

fn main() -> std::io::Result<()> {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "opscope_seq.csv".into());
    let steps: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let (b, d) = (64usize, 128usize);

    let (ginj, specs) = inject_matmul_stats(&matmul_graph(b, d), &StatConfig::default());
    let mut compiled = Session::new(Device::Cpu).compile(ginj);

    // Fixed weight (stationary across the whole decode).
    let mut w = vec![0f32; d * d];
    Philox4x32::new(777).fill_normal(&mut w);

    // Activation state + a per-step drift stream (random walk).
    let mut a = vec![0f32; b * d];
    Philox4x32::new(1).fill_normal(&mut a);
    let mut drift_rng = Philox4x32::new(2);
    let mut drift = vec![0f32; b * d];

    let mut rec = Recorder::create(&out_path)?;
    for step in 0..steps {
        if step > 0 {
            drift_rng.fill_normal(&mut drift);
            for (ai, di) in a.iter_mut().zip(&drift) {
                *ai += 0.05 * di; // slow random walk
            }
        }
        let outs = compiled.run(&[("A", &a), ("W", &w)]);
        // Single sequence (run_id = 0), ordered by step.
        rec.record(0, step, "cpu", "decode", b, d, d, &specs, &outs)?;
    }
    rec.flush()?;
    eprintln!("[opscope] wrote {steps} decode steps → {out_path}");
    Ok(())
}
