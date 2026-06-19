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

//! Transformer-block microbenchmark via real PJRT compile + execute.
//!
//! Builds a single self-attention + FFN block at BERT-base shape
//! (B=1, S=128, H=384, FFN=1536) and runs it N times, reporting
//! compile time, per-iteration execute time, and a rough GFLOPs
//! figure based on the matmul work alone (the dominant cost).
//!
//! Gated on both `LIBTPU_PATH` (skips on hosts without a plugin)
//! and `RLX_TPU_BENCH=1` (so it doesn't run on every `cargo test`
//! by default — bench output bloats the test stream).
//!
//! Run inside the Docker harness with:
//!
//!   RLX_TPU_BENCH=1 ./rlx-tpu/docker/validate.sh --numerical
//!
//! This benches the XLA CPU plugin built from source, *not* a real
//! TPU. It's useful for spotting compile/exec regressions in our HLO
//! emission and tracking IR optimization deltas; absolute numbers
//! shouldn't be read as TPU performance.

use std::time::Instant;

use rlx_ir::op::{Activation, BinaryOp, MaskKind};
use rlx_ir::{DType, Graph, Shape};

const BENCH_ITERS: usize = 25;
const WARMUP_ITERS: usize = 3;

fn skip_unless_bench() -> bool {
    if std::env::var("LIBTPU_PATH").is_err() {
        eprintln!("[pjrt_bench] LIBTPU_PATH not set — skipping");
        return true;
    }
    if rlx_ir::env::is_unset("RLX_TPU_BENCH") {
        eprintln!(
            "[pjrt_bench] RLX_TPU_BENCH not set — skipping. \
                   set RLX_TPU_BENCH=1 to run."
        );
        return true;
    }
    if !rlx_tpu::is_available() {
        eprintln!("[pjrt_bench] PJRT plugin failed to initialize — skipping");
        return true;
    }
    false
}

fn build_block(b: usize, s: usize, h: usize, n_heads: usize, ffn: usize) -> Graph {
    assert_eq!(h % n_heads, 0, "H must be divisible by n_heads");
    let d_head = h / n_heads;
    let f = DType::F32;
    let mut g = Graph::new("transformer_block");

    // reshape() takes Vec<i64>; Shape::new takes &[usize]. Cast helper.
    let i64v = |dims: &[usize]| -> Vec<i64> { dims.iter().map(|&d| d as i64).collect() };

    let x = g.input("x", Shape::new(&[b, s, h], f));
    let w_q = g.param("w_q", Shape::new(&[h, h], f));
    let w_k = g.param("w_k", Shape::new(&[h, h], f));
    let w_v = g.param("w_v", Shape::new(&[h, h], f));
    let w_o = g.param("w_o", Shape::new(&[h, h], f));
    let ln1_g = g.param("ln1_g", Shape::new(&[h], f));
    let ln1_b = g.param("ln1_b", Shape::new(&[h], f));
    let ln2_g = g.param("ln2_g", Shape::new(&[h], f));
    let ln2_b = g.param("ln2_b", Shape::new(&[h], f));
    let w_ff1 = g.param("w_ff1", Shape::new(&[h, ffn], f));
    let w_ff2 = g.param("w_ff2", Shape::new(&[ffn, h], f));

    let xn = g.layer_norm(x, ln1_g, ln1_b, -1, 1e-5, Shape::new(&[b, s, h], f));
    let bs = b * s;
    let xn_2d = g.reshape(xn, i64v(&[bs, h]), Shape::new(&[bs, h], f));
    let q_2d = g.matmul(xn_2d, w_q, Shape::new(&[bs, h], f));
    let k_2d = g.matmul(xn_2d, w_k, Shape::new(&[bs, h], f));
    let v_2d = g.matmul(xn_2d, w_v, Shape::new(&[bs, h], f));
    let q = g.reshape(
        q_2d,
        i64v(&[b, n_heads, s, d_head]),
        Shape::new(&[b, n_heads, s, d_head], f),
    );
    let k = g.reshape(
        k_2d,
        i64v(&[b, n_heads, s, d_head]),
        Shape::new(&[b, n_heads, s, d_head], f),
    );
    let v = g.reshape(
        v_2d,
        i64v(&[b, n_heads, s, d_head]),
        Shape::new(&[b, n_heads, s, d_head], f),
    );
    let attn = g.attention_kind(
        q,
        k,
        v,
        n_heads,
        d_head,
        MaskKind::Causal,
        Shape::new(&[b, n_heads, s, d_head], f),
    );
    let attn_2d = g.reshape(attn, i64v(&[bs, h]), Shape::new(&[bs, h], f));
    let proj_2d = g.matmul(attn_2d, w_o, Shape::new(&[bs, h], f));
    let proj = g.reshape(proj_2d, i64v(&[b, s, h]), Shape::new(&[b, s, h], f));
    let resid1 = g.binary(BinaryOp::Add, x, proj, Shape::new(&[b, s, h], f));

    let r1n = g.layer_norm(resid1, ln2_g, ln2_b, -1, 1e-5, Shape::new(&[b, s, h], f));
    let r1n_2d = g.reshape(r1n, i64v(&[bs, h]), Shape::new(&[bs, h], f));
    let ff1 = g.matmul(r1n_2d, w_ff1, Shape::new(&[bs, ffn], f));
    let act = g.activation(Activation::Gelu, ff1, Shape::new(&[bs, ffn], f));
    let ff2 = g.matmul(act, w_ff2, Shape::new(&[bs, h], f));
    let ff2_3d = g.reshape(ff2, i64v(&[b, s, h]), Shape::new(&[b, s, h], f));
    let out = g.binary(BinaryOp::Add, resid1, ff2_3d, Shape::new(&[b, s, h], f));
    g.set_outputs(vec![out]);
    g
}

fn pct(times: &mut [u128], p: f64) -> u128 {
    times.sort_unstable();
    let i = ((times.len() as f64 - 1.0) * p).round() as usize;
    times[i]
}

/// One bench result row.
struct BenchRow {
    b: usize,
    s: usize,
    h: usize,
    n_heads: usize,
    ffn: usize,
    compile_ms: u128,
    mean_us: u128,
    p50_us: u128,
    p95_us: u128,
    gflops: f64,
}

fn run_one(b: usize, s: usize, h: usize, n_heads: usize, ffn: usize) -> BenchRow {
    let g = build_block(b, s, h, n_heads, ffn);

    let t_compile = Instant::now();
    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    let compile_ms = t_compile.elapsed().as_millis();

    let n_h = h;
    let n_ffn = ffn;
    let mut rng = 1234_u64;
    let mut next = || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 33) as f32 / u32::MAX as f32) * 0.02 - 0.01
    };
    let w_h2: Vec<f32> = (0..(n_h * n_h)).map(|_| next()).collect();
    let w_h_ffn: Vec<f32> = (0..(n_h * n_ffn)).map(|_| next()).collect();
    let w_ffn_h: Vec<f32> = (0..(n_ffn * n_h)).map(|_| next()).collect();
    let ln_g: Vec<f32> = vec![1.0; n_h];
    let ln_b: Vec<f32> = vec![0.0; n_h];
    exec.set_param("w_q", &w_h2);
    exec.set_param("w_k", &w_h2);
    exec.set_param("w_v", &w_h2);
    exec.set_param("w_o", &w_h2);
    exec.set_param("w_ff1", &w_h_ffn);
    exec.set_param("w_ff2", &w_ffn_h);
    exec.set_param("ln1_g", &ln_g);
    exec.set_param("ln1_b", &ln_b);
    exec.set_param("ln2_g", &ln_g);
    exec.set_param("ln2_b", &ln_b);

    let xs: Vec<f32> = (0..(b * s * h)).map(|i| (i as f32) * 1e-3).collect();

    for _ in 0..WARMUP_ITERS {
        let _ = exec.run(&[("x", &xs)]);
    }

    let mut times_us: Vec<u128> = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let t = Instant::now();
        let _ = exec.run(&[("x", &xs)]);
        times_us.push(t.elapsed().as_micros());
    }
    let total_us: u128 = times_us.iter().sum();
    let mean_us = total_us / BENCH_ITERS as u128;
    let p50_us = pct(&mut times_us.clone(), 0.50);
    let p95_us = pct(&mut times_us.clone(), 0.95);

    let bs = (b * s) as f64;
    let h_f = h as f64;
    let ffn_f = ffn as f64;
    let mm_flops_per_iter = 2.0 * bs * h_f * h_f * 4.0
        + 2.0 * bs * h_f * ffn_f
        + 2.0 * bs * ffn_f * h_f
        + 2.0 * bs * (s as f64) * h_f * 2.0;
    let gflops = mm_flops_per_iter / (mean_us as f64 * 1e3);

    BenchRow {
        b,
        s,
        h,
        n_heads,
        ffn,
        compile_ms,
        mean_us,
        p50_us,
        p95_us,
        gflops,
    }
}

/// Single canonical configuration — quick check for the bench
/// machinery itself. Always runs when `RLX_TPU_BENCH=1`.
#[test]
fn transformer_block_bench() {
    if skip_unless_bench() {
        return;
    }
    let r = run_one(1, 64, 192, 4, 768);
    eprintln!(
        "[pjrt_bench] transformer_block \
               B={} S={} H={} FFN={} heads={}",
        r.b, r.s, r.h, r.ffn, r.n_heads
    );
    eprintln!("[pjrt_bench]   compile = {} ms", r.compile_ms);
    eprintln!(
        "[pjrt_bench]   exec    = mean {} µs   p50 {} µs   p95 {} µs",
        r.mean_us, r.p50_us, r.p95_us
    );
    eprintln!(
        "[pjrt_bench]   ~       = {:.2} GFLOPs (matmul-only estimate)",
        r.gflops
    );
}

/// Shape sweep — runs a small (S, H) matrix to spot perf cliffs.
/// Gated on `RLX_TPU_BENCH_SWEEP=1` (slower; typically run by
/// hand, not in CI).
#[test]
fn transformer_block_bench_sweep() {
    if skip_unless_bench() {
        return;
    }
    if rlx_ir::env::is_unset("RLX_TPU_BENCH_SWEEP") {
        eprintln!(
            "[pjrt_bench] sweep skipped — set \
                   RLX_TPU_BENCH_SWEEP=1 to enable"
        );
        return;
    }
    // (S, H, n_heads, FFN) covering: small (32×128), bert-tiny
    // (64×192), bert-mini (64×256), bert-small (128×384).
    let configs = [
        (32_usize, 128_usize, 4_usize, 512_usize),
        (64, 192, 4, 768),
        (64, 256, 4, 1024),
        (128, 384, 8, 1536),
    ];
    eprintln!(
        "[pjrt_bench_sweep] {:>4} {:>4} {:>4} {:>5} | \
               {:>7} {:>7} {:>7} {:>7} {:>9}",
        "B", "S", "H", "FFN", "compile", "mean", "p50", "p95", "GFLOPs"
    );
    eprintln!("[pjrt_bench_sweep] {}", "-".repeat(70));
    for &(s, h, nh, ffn) in &configs {
        let r = run_one(1, s, h, nh, ffn);
        eprintln!(
            "[pjrt_bench_sweep] {:>4} {:>4} {:>4} {:>5} | \
                   {:>5}ms {:>5}µs {:>5}µs {:>5}µs {:>9.2}",
            r.b, r.s, r.h, r.ffn, r.compile_ms, r.mean_us, r.p50_us, r.p95_us, r.gflops
        );
    }
}
