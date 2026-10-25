// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Wall time of a decode-shaped schedule with independent branches, for an
//! A/B of wave execution (`wave-exec`):
//!
//! ```sh
//! RLX_CPU_WAVES=0 cargo run -p rlx-cpu --release --features wave-exec --example wave_bench
//! RLX_CPU_WAVES=1 cargo run -p rlx-cpu --release --features wave-exec --example wave_bench
//! ```
//!
//! `M`, `K`, `BRANCHES` and `ITERS` override the shape.

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{WavePlan, compile_thunks, execute_thunks};
use rlx_ir::{DType, Graph, Op, Shape, Tick};

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let (m, k, branches, iters) = (
        env("M", 1),
        env("K", 1024),
        env("BRANCHES", 8),
        env("ITERS", 2000),
    );
    let f32s = |d: &[usize]| Shape::new(d, DType::F32);
    let mut g = Graph::new("wave_bench");
    let x = g.input("x", f32s(&[m, k]));
    let gamma = g.input("g", f32s(&[k]));
    let beta = g.input("b", f32s(&[k]));
    let mut outs = Vec::new();
    for i in 0..branches {
        let w = g.input(format!("w{i}"), f32s(&[k, k]));
        let h = g.matmul(x, w, f32s(&[m, k]));
        outs.push(g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![h, gamma, beta],
            f32s(&[m, k]),
        ));
    }
    g.set_outputs(outs);

    let mut arena = Arena::from_plan(rlx_opt::memory::plan_memory(&g));
    let sched = compile_thunks(&g, &arena);
    for node in g.nodes() {
        if let Op::Input { .. } = node.op {
            for (i, v) in arena.slice_mut(node.id).iter_mut().enumerate() {
                *v = ((i % 97) as f32 - 48.0) / 97.0;
            }
        }
    }
    for _ in 0..50 {
        execute_thunks(&sched, arena.raw_buf_mut());
    }
    let mut samples: Vec<u64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Tick::now();
        execute_thunks(&sched, arena.raw_buf_mut());
        samples.push(Tick::now().elapsed_ns(t0));
    }
    samples.sort_unstable();
    println!(
        "m={m} k={k} branches={branches} waves={} parallel_thunks={} p50={:.1}us p10={:.1}us",
        rlx_ir::env::var("RLX_CPU_WAVES").unwrap_or_else(|| "1".into()),
        WavePlan::build(&sched.thunks).parallel_thunks(),
        samples[iters / 2] as f64 / 1e3,
        samples[iters / 10] as f64 / 1e3,
    );
}
