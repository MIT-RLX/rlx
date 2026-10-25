// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Wall time of an affine `Scan` (`h = sigmoid(x)·h + y`, a gated linear
//! recurrence) through `Session`, for an A/B of the `assoc-scan` rewrite:
//!
//! ```sh
//! cargo run -p rlx-runtime --release --features cpu --example assoc_scan_bench
//! cargo run -p rlx-runtime --release --features cpu,assoc-scan --example assoc_scan_bench
//! DEVICE=metal cargo run -p rlx-runtime --release --features metal,assoc-scan --example assoc_scan_bench
//! ```
//!
//! `T`, `D`, `ITERS` override the shape. Prints the output checksum so the two
//! builds can be compared.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Op, Shape, Tick};
use rlx_runtime::{CompileOptions, Session, parse_device};

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let (t, d, iters) = (env("T", 1024), env("D", 256), env("ITERS", 20));
    let dev =
        parse_device(&std::env::var("DEVICE").unwrap_or_else(|_| "cpu".into())).expect("DEVICE");
    let sh = |dims: &[usize]| Shape::new(dims, DType::F32);

    let mut body = Graph::new("body");
    let h = body.input("h", sh(&[d]));
    let x = body.input("x", sh(&[d]));
    let y = body.input("y", sh(&[d]));
    let a = body.activation(Activation::Sigmoid, x, sh(&[d]));
    let ah = body.binary(BinaryOp::Mul, a, h, sh(&[d]));
    let next = body.binary(BinaryOp::Add, ah, y, sh(&[d]));
    body.set_outputs(vec![next]);

    let mut g = Graph::new("assoc_scan_bench");
    let h0 = g.input("h0", sh(&[d]));
    let xs = g.input("xs", sh(&[t, d]));
    let ys = g.input("ys", sh(&[t, d]));
    let scan = g.add_node(
        Op::Scan {
            body: Box::new(body),
            length: t as u32,
            save_trajectory: true,
            num_bcast: 0,
            num_xs: 2,
            num_checkpoints: 0,
        },
        vec![h0, xs, ys],
        sh(&[t, d]),
    );
    g.set_outputs(vec![scan]);

    let wave = |n: usize, s: f32| -> Vec<f32> { (0..n).map(|i| ((i as f32) * s).sin()).collect() };
    let (h0v, xv, yv) = (wave(d, 0.37), wave(t * d, 0.011), wave(t * d, 0.023));
    let opts = CompileOptions::new().scan_unroll_max_length(0);
    let t0 = Tick::now();
    let mut sess = Session::new(dev).compile_with(g, &opts);
    let compile_ms = Tick::now().elapsed_ns(t0) as f64 / 1e6;
    let feeds = [("h0", &h0v[..]), ("xs", &xv[..]), ("ys", &yv[..])];
    let out = sess.run(&feeds);
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Tick::now();
        let _ = sess.run(&feeds);
        samples.push(Tick::now().elapsed_ns(t0));
    }
    samples.sort_unstable();
    let checksum: f64 = out[0].iter().map(|&v| v as f64).sum();
    println!(
        "{dev:?} T={t} D={d} assoc-scan={} compile={compile_ms:.1}ms p50={:.3}ms checksum={checksum:.6}",
        cfg!(feature = "assoc-scan"),
        samples[iters / 2] as f64 / 1e6,
    );
}
