// SPDX-License-Identifier: GPL-3.0-only
//! A-B microbenchmark: `FusedConvBiasAct` (cuDNN fused / conv+epilogue) vs the
//! unfused `conv → expand → add → relu` path, on CUDA.
//!
//! Same graph both runs; `RLX_DISABLE_CONV_BIAS_ACT_FUSION` (set per invocation)
//! selects the path, so run it twice and compare the printed ms/iter:
//!
//!   # fused (default)
//!   cargo test -p rlx-runtime --features cpu,cuda --test bench_conv_bias_act \
//!     -- --ignored --nocapture
//!   # unfused baseline
//!   RLX_DISABLE_CONV_BIAS_ACT_FUSION=1 cargo test ... -- --ignored --nocapture
//!
//! `#[ignore]` so it stays out of the normal suite. Rig-only (needs a CUDA GPU).

#![cfg(feature = "cuda")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::*;
use rlx_runtime::{Device, Session, is_available};
use std::time::Instant;

struct Shape5 {
    name: &'static str,
    b: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    k: usize,
    p: usize,
    groups: usize,
}

fn build(s: &Shape5) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv_bias_relu");
    let x = g.input("x", Shape::new(&[s.b, s.c_in, s.h, s.w], f));
    let weight = g.input("w", Shape::new(&[s.c_out, s.c_in / s.groups, s.k, s.k], f));
    let bias = g.input("b", Shape::new(&[s.c_out], f));
    let y = g.conv2d(x, weight, [s.k, s.k], [1, 1], [s.p, s.p], [1, 1], s.groups);
    let out_s = Shape::new(&[s.b, s.c_out, s.h, s.w], f);
    let bias4 = g.reshape(
        bias,
        vec![1, s.c_out as i64, 1, 1],
        Shape::new(&[1, s.c_out, 1, 1], f),
    );
    let exp = g.add_node(
        Op::Expand {
            target_shape: vec![s.b as i64, s.c_out as i64, s.h as i64, s.w as i64],
        },
        vec![bias4],
        out_s.clone(),
    );
    let add = g.binary(BinaryOp::Add, y, exp, out_s.clone());
    let out = g.add_node(Op::Activation(Activation::Relu), vec![add], out_s);
    g.set_outputs(vec![out]);
    g
}

#[test]
#[ignore = "rig-only conv-bias-act microbenchmark; run with --ignored --nocapture"]
fn bench_conv_bias_act_cuda() {
    if !is_available(Device::Cuda) {
        eprintln!("skip: no CUDA device");
        return;
    }
    let fused = !rlx_ir::env::flag("RLX_DISABLE_CONV_BIAS_ACT_FUSION");
    let label = if fused { "FUSED" } else { "UNFUSED" };
    eprintln!("\n=== conv+bias+relu CUDA bench — {label} ===");

    let shapes = [
        Shape5 {
            name: "3x3   b1  64->64  56x56",
            b: 1,
            c_in: 64,
            c_out: 64,
            h: 56,
            w: 56,
            k: 3,
            p: 1,
            groups: 1,
        },
        Shape5 {
            name: "3x3   b8  64->64  56x56",
            b: 8,
            c_in: 64,
            c_out: 64,
            h: 56,
            w: 56,
            k: 3,
            p: 1,
            groups: 1,
        },
        Shape5 {
            name: "3x3   b1 128->128 28x28",
            b: 1,
            c_in: 128,
            c_out: 128,
            h: 28,
            w: 28,
            k: 3,
            p: 1,
            groups: 1,
        },
        Shape5 {
            name: "1x1   b1 256->256 28x28",
            b: 1,
            c_in: 256,
            c_out: 256,
            h: 28,
            w: 28,
            k: 1,
            p: 0,
            groups: 1,
        },
        Shape5 {
            name: "dw3x3 b1 128->128 28x28",
            b: 1,
            c_in: 128,
            c_out: 128,
            h: 28,
            w: 28,
            k: 3,
            p: 1,
            groups: 128,
        },
    ];

    let warmup = 20;
    let iters = 100;

    for s in &shapes {
        let xn = s.b * s.c_in * s.h * s.w;
        let wn = s.c_out * (s.c_in / s.groups) * s.k * s.k;
        let x: Vec<f32> = (0..xn).map(|i| (i * 7 % 23) as f32 / 23.0 - 0.5).collect();
        let w: Vec<f32> = (0..wn).map(|i| (i * 5 % 17) as f32 / 17.0 - 0.5).collect();
        let b: Vec<f32> = (0..s.c_out).map(|i| i as f32 * 0.05 - 0.2).collect();

        let mut c = Session::new(Device::Cuda).compile(build(s));
        let feeds: [(&str, &[f32]); 3] = [
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("b", b.as_slice()),
        ];

        for _ in 0..warmup {
            let _ = c.run(&feeds);
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = c.run(&feeds);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // conv MACs → GFLOP/s (2 flops per MAC).
        let macs = (s.b * s.c_out * s.h * s.w) as f64 * (s.c_in / s.groups * s.k * s.k) as f64;
        let gflops = 2.0 * macs / (ms * 1e6);
        eprintln!(
            "  {label:7} {}  {ms:8.4} ms/iter   {gflops:8.1} GFLOP/s",
            s.name
        );
    }
}
