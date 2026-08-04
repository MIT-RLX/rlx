// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-replay` — self-contained demo of the optimization-opportunity loop.
//! Runs a varying-width MLP over `T` steps against a **serving-trace-like**
//! workload: a small pool of distinct inputs is replayed, so most calls repeat
//! an earlier input (cache-hittable). Records per-step sketches + FLOPs, then
//! mines opportunities: the expensive matmuls whose input *repeats over time*
//! rank top as memoize/delta-compute targets.
//!
//! Usage: `opscope-replay [steps] [pool] [out.csv]`

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Philox4x32, Shape};
use rlx_opscope::optimize::{mine_opportunities, report};
use rlx_opscope::{Recorder, StatConfig, inject_matmul_stats, write_site_costs};
use rlx_runtime::{Device, Session};

const S: usize = 32; // rows (tokens)

fn sh(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// A chain of matmul→relu with the given (in,out) widths → varying op costs.
fn pyramid_mlp(widths: &[(usize, usize)]) -> Graph {
    let mut g = Graph::new("pyramid");
    let mut x = g.input("x", sh(&[S, widths[0].0]));
    for (i, &(din, dout)) in widths.iter().enumerate() {
        let w = g.param(format!("W{i}"), sh(&[din, dout]));
        let h = g.matmul(x, w, sh(&[S, dout]));
        x = g.activation(Activation::Relu, h, sh(&[S, dout]));
    }
    g.set_outputs(vec![x]);
    g
}

fn main() -> std::io::Result<()> {
    let steps: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let pool: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let out = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "opscope_replay.csv".into());

    // Widths of increasing/decreasing cost — the 512-wide matmuls dominate FLOPs.
    let widths = [(64usize, 128usize), (128, 64), (64, 512), (512, 64)];
    let d0 = widths[0].0;
    let (g, specs) = inject_matmul_stats(&pyramid_mlp(&widths), &StatConfig::default());
    let mut c = Session::new(Device::Cpu).compile(g);

    // Random fixed weights.
    let mut rng = Philox4x32::new(0xA11CE);
    for (i, &(din, dout)) in widths.iter().enumerate() {
        let mut w = vec![0f32; din * dout];
        rng.fill_normal(&mut w);
        let scale = (2.0 / din as f32).sqrt();
        for v in &mut w {
            *v *= scale;
        }
        c.set_param(&format!("W{i}"), &w);
    }

    // Two workloads: pool>0 replays a small pool of distinct inputs (→ exact
    // repeats over time → memoize); pool==0 is a slow random walk (consecutive
    // inputs barely change → delta-compute).
    let walk = pool == 0;
    let inputs: Vec<Vec<f32>> = if walk {
        Vec::new()
    } else {
        (0..pool)
            .map(|_| {
                let mut v = vec![0f32; S * d0];
                rng.fill_normal(&mut v);
                v
            })
            .collect()
    };
    let mut cur = vec![0f32; S * d0];
    rng.fill_normal(&mut cur);

    let mut rec = Recorder::create(&out)?;
    for step in 0..steps {
        if walk && step > 0 {
            let mut n = vec![0f32; S * d0];
            rng.fill_normal(&mut n);
            for (c, dn) in cur.iter_mut().zip(&n) {
                *c += 0.03 * dn; // small drift
            }
        }
        let x: &[f32] = if walk {
            &cur
        } else {
            &inputs[(rng.next_u32() as usize) % pool]
        };
        let outs = c.run(&[("x", x)]);
        rec.record(0, step, "cpu", "replay", S, d0, 0, &specs, &outs)?;
    }
    rec.flush()?;
    let sidecar = format!("{out}.sites");
    write_site_costs(&sidecar, &specs)?;
    eprintln!(
        "[opscope] replayed {steps} steps over a pool of {pool} inputs → {out} (+{sidecar})\n"
    );

    // Mine + report inline.
    let opps = mine_opportunities(&out, &sidecar)?;
    report(&opps);
    Ok(())
}
