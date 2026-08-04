// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-sweep` — drive a matmul across shapes × synthetic distributions
//! with stat-injection on, and record every tensor sketch to a tidy CSV.
//!
//! Usage: `opscope-sweep [out.csv]`  (default: `opscope.csv`)

use rlx_ir::{DType, Graph, Shape};
use rlx_opscope::{Dist, Recorder, StatConfig, inject_matmul_stats, sample};
use rlx_runtime::{Device, Session};

/// Build `C = A @ B` with `A:[M,K]`, `B:[K,N]` as named inputs.
fn matmul_graph(m: usize, k: usize, n: usize) -> Graph {
    let mut g = Graph::new("matmul");
    let a = g.input("A", Shape::new(&[m, k], DType::F32));
    let b = g.input("B", Shape::new(&[k, n], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![c]);
    g
}

fn approx_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 1e-3 * (1.0 + x.abs().max(y.abs())))
}

fn main() -> std::io::Result<()> {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "opscope.csv".into());
    let cfg = StatConfig::default();
    let shapes = [
        (128usize, 128usize, 128usize),
        (256, 64, 256),
        (64, 256, 128),
    ];
    let seeds = [1u64, 2, 3];
    let backend = "cpu";

    let mut rec = Recorder::create(&out_path)?;
    let mut run_id = 0u64;
    let mut checked = false;

    for &(m, k, n) in &shapes {
        // Compile the injected graph once per shape (shapes are static).
        let (ginj, specs) = inject_matmul_stats(&matmul_graph(m, k, n), &cfg);
        let mut compiled = Session::new(Device::Cpu).compile(ginj);

        for &dist in &Dist::ALL {
            for &seed in &seeds {
                let a = sample(dist, m, k, seed);
                let b = sample(dist, k, n, seed + 1000);
                let outs = compiled.run(&[("A", &a), ("B", &b)]);

                // One-time correctness gate: the injected graph's primary
                // output must equal the un-injected matmul.
                if !checked {
                    let mut base = Session::new(Device::Cpu).compile(matmul_graph(m, k, n));
                    let base_out = base.run(&[("A", &a), ("B", &b)]);
                    assert!(
                        approx_eq(&outs[0], &base_out[0]),
                        "stat injection changed the matmul output!"
                    );
                    eprintln!(
                        "[opscope] correctness gate passed: injected output == matmul output"
                    );
                    checked = true;
                }

                rec.record(run_id, 0, backend, dist.name(), m, k, n, &specs, &outs)?;
                run_id += 1;
            }
        }
        eprintln!(
            "[opscope] swept shape {m}x{k}x{n} ({} sketches/run)",
            specs.len()
        );
    }

    rec.flush()?;
    eprintln!("[opscope] wrote {run_id} runs → {out_path}");
    Ok(())
}
