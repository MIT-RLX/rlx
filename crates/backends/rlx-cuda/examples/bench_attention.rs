// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal CUDA-only flash-attention micro-bench, for A/B-ing kernel edits
//! (e.g. the `__launch_bounds__` occupancy hint). Twin of the rlx-rocm one.
//!
//! cargo run --release -p rlx-cuda --example bench_attention

use std::time::Instant;

use rlx_cuda::backend::CudaExecutable;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};

// BSHD layout [batch, seq, heads, head_dim] — matches the flash kernel path.
fn bench(b: usize, s: usize, h: usize, d: usize, warmup: usize, iters: usize) {
    let mut g = Graph::new("attn");
    let q = g.input("q", Shape::new(&[b, s, h, d], DType::F32));
    let k = g.input("k", Shape::new(&[b, s, h, d], DType::F32));
    let v = g.input("v", Shape::new(&[b, s, h, d], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: h,
            head_dim: d,
            v_head_dim: None,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, s, h, d], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut exe = CudaExecutable::compile(g);
    let n = b * h * s * d;
    let qv: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 1e-3).collect();
    let kv: Vec<f32> = (0..n).map(|i| ((i % 89) as f32) * 1e-3).collect();
    let vv: Vec<f32> = (0..n).map(|i| ((i % 83) as f32) * 1e-3).collect();

    for _ in 0..warmup {
        let _ = exe.run(&[("q", &qv), ("k", &kv), ("v", &vv)]);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = exe.run(&[("q", &qv), ("k", &kv), ("v", &vv)]);
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    println!("  B={b} S={s:>5} H={h:>2} D={d:>3}   {:>8.3} ms", dt * 1e3);
}

fn main() {
    if !rlx_cuda::is_available() {
        println!("CUDA not available on this host — exiting.");
        return;
    }
    println!("rlx-cuda flash-attention bench (BSHD, D<=128)");
    println!("--------------------------------------------");
    let cases: &[(usize, usize, usize, usize)] = &[
        (1, 512, 8, 64),
        (1, 1024, 8, 64),
        (1, 1024, 16, 64),
        (1, 2048, 8, 128),
        (4, 1024, 8, 64),
    ];
    for &(b, s, h, d) in cases {
        bench(b, s, h, d, /*warmup*/ 5, /*iters*/ 50);
    }
}
