// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-live` — the sampled production-recorder path. Runs a model over a
//! `requests`-long stream (a small pool of inputs replayed, like a serving
//! trace), samples 1-in-N, and folds each site's fingerprint into bounded online
//! sketches. Reports the streaming recurrence estimate at constant memory —
//! no per-row CSV.
//!
//! Usage: `opscope-live [requests] [pool] [sample_every]`

use rlx_ir::{Op, Philox4x32};
use rlx_opscope::demo::{D, S, build};
use rlx_opscope::live::LiveSampler;
use rlx_opscope::{StatConfig, inject_matmul_stats};
use rlx_runtime::{Device, Session};

fn main() {
    let requests: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let pool: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let sample_every: u64 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let g = build("mlp", 4);
    let (gi, specs) = inject_matmul_stats(&g, &StatConfig::default());
    let mut c = Session::new(Device::Cpu).compile(gi);

    let mut rng = Philox4x32::new(0x11FE);
    for node in g.nodes() {
        if let Op::Param { name } = &node.op {
            let numel: usize = (0..node.shape.rank())
                .map(|i| node.shape.dim(i).unwrap_static())
                .product();
            let mut d = vec![0f32; numel];
            rng.fill_normal(&mut d);
            c.set_param(name, &d);
        }
    }
    let inputs: Vec<Vec<f32>> = (0..pool)
        .map(|_| {
            let mut v = vec![0f32; S * D];
            rng.fill_normal(&mut v);
            v
        })
        .collect();

    let mut sampler = LiveSampler::new(sample_every);
    for _ in 0..requests {
        let p = (rng.next_u32() as usize) % pool;
        let outs = c.run(&[("x", inputs[p].as_slice())]);
        sampler.record(&specs, &outs);
    }
    sampler.report();
    println!(
        "\n(ground truth: {pool} distinct inputs → ~{:.0}% of calls repeat an earlier one)",
        (1.0 - pool as f32 / requests as f32) * 100.0
    );
}
