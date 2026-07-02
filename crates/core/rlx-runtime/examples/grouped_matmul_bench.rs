// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Bench the optimised CPU GroupedMatMul (sort-by-expert + BLAS sgemm)
//! against an inlined naive scalar implementation, across MoE-realistic
//! shapes.
//!
//! cargo run --release -p rlx-runtime --example grouped_matmul_bench \
//!     --features blas-accelerate

use rlx_ir::*;
use rlx_runtime::{Device, Session};
use std::time::Instant;

fn det(seed: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) * 7 + 11) % 31) as f32 / 31.0 * 0.4 - 0.2)
        .collect()
}

// Naive scalar GroupedMatMul — what was in the thunk before this commit.
// Inlined here so the bench can compare against the optimised path.
fn naive_grouped_matmul(
    input: &[f32],
    weight: &[f32],
    ids: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    _num_experts: usize,
) {
    let expert_stride = k * n;
    for i in 0..m {
        let e = ids[i] as usize;
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += input[i * k + kk] * weight[e * expert_stride + kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

fn time_optimised(
    input: &[f32],
    weight: &[f32],
    ids: &[f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    iters: usize,
) -> f64 {
    // Build a one-op graph and let the rlx-cpu thunk runtime drive it.
    // This is the same code path real callers hit, just measured in
    // isolation.
    let build = || {
        let f = DType::F32;
        let mut g = Graph::new("gmm");
        let x = g.input("x", Shape::new(&[m, k], f));
        let w = g.param("w", Shape::new(&[num_experts, k, n], f));
        let idx = g.input("idx", Shape::new(&[m], f));
        let y = g.add_node(Op::GroupedMatMul, vec![x, w, idx], Shape::new(&[m, n], f));
        g.set_outputs(vec![y]);
        g
    };
    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(build());
    compiled.set_param("w", weight);
    // Warmup
    for _ in 0..3 {
        let _ = compiled.run(&[("x", input), ("idx", ids)]);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = compiled.run(&[("x", input), ("idx", ids)]);
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64 // ms per iter
}

fn time_naive(
    input: &[f32],
    weight: &[f32],
    ids: &[f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    iters: usize,
) -> f64 {
    let mut out = vec![0f32; m * n];
    // Warmup
    for _ in 0..3 {
        naive_grouped_matmul(input, weight, ids, &mut out, m, k, n, num_experts);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        naive_grouped_matmul(input, weight, ids, &mut out, m, k, n, num_experts);
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    println!(
        "{:<28} {:>11} {:>11} {:>10}",
        "Shape (M, K, N, E)", "naive (ms)", "opt (ms)", "speedup"
    );
    println!("{:-<64}", "");

    // Realistic MoE shapes. K=N=768 mirrors a Nomic-class hidden dim;
    // sweep M/E ratios from "many tokens, few experts" (well-batched per
    // expert) down to "few tokens per expert" (each expert gets ~1 token).
    let cases = vec![
        // (M, K, N, num_experts, iters)
        (32, 768, 768, 8, 100),  // 4 tokens/expert
        (64, 768, 768, 8, 100),  // 8 tokens/expert
        (128, 768, 768, 8, 50),  // 16 tokens/expert
        (256, 768, 768, 8, 30),  // 32 tokens/expert
        (512, 768, 768, 8, 20),  // 64 tokens/expert
        (256, 768, 768, 16, 30), // 16 tokens/expert (more experts)
        (256, 768, 768, 32, 30), // 8 tokens/expert
        (256, 768, 768, 64, 30), // 4 tokens/expert (worst case for sgemm)
    ];

    for (m, k, n, e, iters) in cases {
        let input = det(0, m * k);
        let weight = det(1, e * k * n);
        // Round-robin token-to-expert assignment so every expert gets work.
        let ids: Vec<f32> = (0..m).map(|i| (i % e) as f32).collect();
        let naive_ms = time_naive(&input, &weight, &ids, m, k, n, e, iters);
        let opt_ms = time_optimised(&input, &weight, &ids, m, k, n, e, iters);
        let speedup = naive_ms / opt_ms;
        println!(
            "({:>4}, {:>4}, {:>4}, {:>3})    {:>11.2} {:>11.2} {:>9.2}x",
            m, k, n, e, naive_ms, opt_ms, speedup
        );
    }

    println!();
    println!("Optimised path: counting-sort tokens by expert, run one BLAS sgemm");
    println!("per expert on its packed token slab, then unpermute outputs back.");
    println!("The win grows with tokens-per-expert — bigger sgemms amortize the");
    println!("permute/unpermute overhead. Below ~4 tokens/expert the per-expert");
    println!("BLAS call cost catches up to the scalar loop.");
}
