// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Achieved bandwidth of the packed decode GEMV kernels (`m == 1`).
//!
//! Decode is weight-streaming: every token reads the whole model once, so a
//! GEMV kernel's achieved GB/s *is* the token rate. This measures each K-quant
//! kernel directly at a realistic 27B FFN shape, which a whole-model benchmark
//! cannot do — at model scale the number is confounded by paging, and on a
//! small model everything is launch-bound instead.
//!
//! `REPS` independent matmuls share one input inside a single graph so the
//! per-`run()` encode/commit/wait overhead (~0.2 ms) is amortised across
//! ~400 MB of weight traffic. Reported figure is the **min** over iterations —
//! the uncontended cost, which is what a kernel change moves.
//!
//! Run: `cargo test -p rlx-metal --test gemv_bandwidth -- --nocapture`

#![cfg(target_os = "macos")]

use rlx_ir::op::BinaryOp;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const K: usize = 5120; // Qwen3.8-27B hidden
const REPS: usize = 8;
const ITERS: usize = 12;

// Two widths: the FFN shape (plenty of parallelism, so a kernel can afford more
// rows per simdgroup) and a narrow projection (where too many rows per
// simdgroup starves the GPU). A tiling change must be checked against BOTH.
const N_WIDE: usize = 17408; // Qwen3.8-27B ffn intermediate
const N_NARROW: usize = 1024;

/// Row-major `[n, k]`: output column `c` owns `k` contiguous values.
fn weight(n: usize, seed: f32, ggml: rlx_gguf::GgmlType) -> Vec<u8> {
    let w: Vec<f32> = (0..K * n)
        .map(|i| ((i as f32) * seed).sin() * 0.5)
        .collect();
    rlx_gguf::quantize(&w, ggml).expect("quantize")
}

fn build(n: usize, scheme: QuantScheme, packed_len: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("gemv_bw");
    let x = g.input("x", Shape::new(&[1, K], f));
    let mut acc = None;
    for r in 0..REPS {
        let w = g.param(format!("w{r}"), Shape::new(&[packed_len], DType::U8));
        let y = g.add_node(
            Op::DequantMatMul { scheme },
            vec![x, w],
            Shape::new(&[1, n], f),
        );
        acc = Some(match acc {
            None => y,
            Some(a) => g.add_node(
                Op::Binary(BinaryOp::Add),
                vec![a, y],
                Shape::new(&[1, n], f),
            ),
        });
    }
    g.set_outputs(vec![acc.unwrap()]);
    g
}

fn bench(label: &str, n: usize, ggml: rlx_gguf::GgmlType, scheme: QuantScheme) {
    let packed = weight(n, 0.011, ggml);
    let bytes_per_run = (packed.len() * REPS) as f64;
    let g = build(n, scheme, packed.len());
    let mut s = Session::new(Device::Metal).compile(g);
    for r in 0..REPS {
        s.set_param_typed(&format!("w{r}"), &packed, DType::U8);
    }
    let x: Vec<f32> = (0..K).map(|i| ((i as f32) * 0.03).sin()).collect();

    let mut best = f64::MAX;
    for _ in 0..ITERS {
        let t = std::time::Instant::now();
        let out = s.run(&[("x", &x)]).remove(0);
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(&out);
        best = best.min(dt);
    }
    let gbs = bytes_per_run / best / 1e9;
    println!(
        "{label:>8} n={n:<6}: {:6.1} GB/s   ({:5.2} ms for {:5.0} MB, {:4.1}% of peak)",
        gbs,
        best * 1e3,
        bytes_per_run / 1e6,
        gbs / 273.0 * 100.0
    );
}

#[test]
fn packed_gemv_achieved_bandwidth() {
    println!("\nDecode GEMV bandwidth, m=1, K={K}  (M4 Pro peak ~273 GB/s)");
    for n in [N_WIDE, N_NARROW] {
        bench("Q4_K", n, rlx_gguf::GgmlType::Q4K, QuantScheme::GgufQ4K);
        bench("Q6_K", n, rlx_gguf::GgmlType::Q6K, QuantScheme::GgufQ6K);
        bench("Q5_K", n, rlx_gguf::GgmlType::Q5K, QuantScheme::GgufQ5K);
        bench("Q8_0", n, rlx_gguf::GgmlType::Q8_0, QuantScheme::GgufQ8_0);
        // Ternary-Bonsai / Pestle-27B. `RLX_METAL_Q2_0_SCALAR=1` reverts the
        // inner loop to the byte-load version for an A/B on the same run.
        bench("Q2_0", n, rlx_gguf::GgmlType::Q2_0, QuantScheme::GgufQ2_0);
    }
}
