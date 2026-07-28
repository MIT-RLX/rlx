// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Same graph, run on CPU and GPU **in parallel** (data parallelism). `Func`
//! owns a plain `Graph` (Send), so it can be moved to another thread; each
//! thread has its own compile cache + device. Run:
//! `cargo test -p rlx-tensor --features eval-metal -- --nocapture`.
#![cfg(feature = "eval-metal")]

use std::thread;

use rlx_tensor::{Device, Func, shape};

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-3, "{a:?} != {b:?}");
    }
}

/// Per-shard model: y = relu(x @ w + b), x:[2,3], w:[3,2], b:[2].
fn model() -> Func {
    Func::new("shard", |s| {
        let x = s.input("x", shape![2, 3]);
        let w = s.param("w", shape![3, 2]);
        let b = s.param("b", shape![2]);
        (&x.matmul(&w) + &b).relu()
    })
    .with_param("w", vec![0.5, -1.0, 2.0, 0.0, 1.0, 1.0])
    .with_param("b", vec![0.1, -0.2])
}

#[test]
fn same_graph_cpu_and_gpu_in_parallel() {
    // Batch of 4 rows, split into two shards of 2.
    let shard_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // rows 0,1
    let shard_b: Vec<f32> = vec![-1.0, 0.5, 2.0, 3.0, -2.0, 1.0]; // rows 2,3

    // GPU shard runs on its own thread; CPU shard runs on this thread —
    // they execute concurrently.
    let f_gpu = model();
    let b_for_gpu = shard_b.clone();
    let gpu = thread::spawn(move || {
        let out = f_gpu.run_on(Device::Metal, &[("x", &b_for_gpu)]);
        rlx_tensor::clear_cache(); // drop GPU CompiledGraph while TLS alive
        out
    });

    let f_cpu = model();
    let out_a = f_cpu.run_on(Device::Cpu, &[("x", &shard_a)]);
    let out_b = gpu.join().unwrap();

    // Reference: run both shards on CPU sequentially.
    let r = model();
    let ref_a = r.run_on(Device::Cpu, &[("x", &shard_a)]);
    let ref_b = r.run_on(Device::Cpu, &[("x", &shard_b)]);

    approx(&out_a[0], &ref_a[0]);
    approx(&out_b[0], &ref_b[0]); // GPU shard matches CPU reference
    eprintln!(
        "data-parallel OK: shard A on CPU = {:?}, shard B on Metal = {:?}",
        out_a[0], out_b[0]
    );
}
