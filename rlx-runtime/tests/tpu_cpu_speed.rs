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

//! Side-by-side speed comparison: Device::Cpu vs Device::Tpu on
//! the same graph. Companion to `tpu_real_model_parity.rs` —
//! parity proves correctness, this proves competitive throughput.
//!
//! Important caveats:
//!   * The TPU-side numbers come from the **XLA CPU PJRT plugin**
//!     in the Docker harness, not real TPU silicon. They reflect
//!     XLA-on-CPU performance, not what TPU hardware would give.
//!   * The CPU-side numbers come from rlx-cpu's NEON + Accelerate
//!     thunk path on whatever host runs the test (typically Apple
//!     Silicon or Linux/x86 in Docker).
//!   * Both compilations apply the same precision (`AlwaysF32`) so
//!     the comparison is "two compilers, same arithmetic
//!     precision".
//!
//! Gated on the `tpu` feature, `LIBTPU_PATH`, and `RLX_TPU_BENCH=1`
//! (so it doesn't slow the default validate run).

#![cfg(feature = "tpu")]

use std::time::Instant;

use rlx_driver::Device;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{PrecisionPolicy, Session};

const BENCH_ITERS: usize = 25;
const WARMUP_ITERS: usize = 3;

fn skip_unless_bench() -> bool {
    if std::env::var("LIBTPU_PATH").is_err() {
        eprintln!("[tpu_cpu_speed] LIBTPU_PATH not set — skipping");
        return true;
    }
    if std::env::var("RLX_TPU_BENCH").is_err() {
        eprintln!("[tpu_cpu_speed] RLX_TPU_BENCH not set — skipping");
        return true;
    }
    false
}

fn build_ffn(b: usize, s: usize, h: usize, ffn: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("speed_ffn");
    let i64v = |dims: &[usize]| -> Vec<i64> { dims.iter().map(|&d| d as i64).collect() };
    let bs = b * s;
    let x = g.input("x", Shape::new(&[b, s, h], f));
    let ln_g = g.param("ln_g", Shape::new(&[h], f));
    let ln_b = g.param("ln_b", Shape::new(&[h], f));
    let w_up = g.param("w_up", Shape::new(&[h, ffn], f));
    let w_down = g.param("w_down", Shape::new(&[ffn, h], f));
    let xn = g.layer_norm(x, ln_g, ln_b, -1, 1e-5, Shape::new(&[b, s, h], f));
    let xn_2d = g.reshape(xn, i64v(&[bs, h]), Shape::new(&[bs, h], f));
    let up = g.matmul(xn_2d, w_up, Shape::new(&[bs, ffn], f));
    let act = g.activation(Activation::Gelu, up, Shape::new(&[bs, ffn], f));
    let down = g.matmul(act, w_down, Shape::new(&[bs, h], f));
    let down_3d = g.reshape(down, i64v(&[b, s, h]), Shape::new(&[b, s, h], f));
    let out = g.binary(BinaryOp::Add, x, down_3d, Shape::new(&[b, s, h], f));
    g.set_outputs(vec![out]);
    g
}

fn det_random(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / u32::MAX as f32) * scale - scale * 0.5
        })
        .collect()
}

fn upload(exec: &mut rlx_runtime::CompiledGraph, h: usize, ffn: usize) {
    let w_up: Vec<f32> = det_random(11, h * ffn, 0.04);
    let w_down: Vec<f32> = det_random(22, ffn * h, 0.04);
    let ln_g: Vec<f32> = vec![1.0; h];
    let ln_b: Vec<f32> = vec![0.0; h];
    exec.set_param("w_up", &w_up);
    exec.set_param("w_down", &w_down);
    exec.set_param("ln_g", &ln_g);
    exec.set_param("ln_b", &ln_b);
}

fn pct(times: &mut [u128], p: f64) -> u128 {
    times.sort_unstable();
    let i = ((times.len() as f64 - 1.0) * p).round() as usize;
    times[i]
}

struct BenchRow {
    label: &'static str,
    compile_ms: u128,
    p50_us: u128,
    p95_us: u128,
    gflops: f64,
}

fn run_one(
    label: &'static str,
    device: Device,
    policy: PrecisionPolicy,
    b: usize,
    s: usize,
    h: usize,
    ffn: usize,
    xs: &[f32],
) -> BenchRow {
    let t0 = Instant::now();
    let mut exec = Session::new(device)
        .with_policy(policy)
        .compile(build_ffn(b, s, h, ffn));
    let compile_ms = t0.elapsed().as_millis();
    upload(&mut exec, h, ffn);

    for _ in 0..WARMUP_ITERS {
        let _ = exec.run(&[("x", xs)]);
    }
    let mut times: Vec<u128> = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let t = Instant::now();
        let _ = exec.run(&[("x", xs)]);
        times.push(t.elapsed().as_micros());
    }
    let p50 = pct(&mut times.clone(), 0.50);
    let p95 = pct(&mut times.clone(), 0.95);

    // FFN flops: 2 * BS * (H*FFN + FFN*H) = 4 * BS * H * FFN.
    let bs = (b * s) as f64;
    let flops = 4.0 * bs * (h as f64) * (ffn as f64);
    let gflops = flops / (p50 as f64 * 1e3);

    BenchRow {
        label,
        compile_ms,
        p50_us: p50,
        p95_us: p95,
        gflops,
    }
}

#[test]
fn ffn_cpu_vs_tpu_bench() {
    if skip_unless_bench() {
        return;
    }

    let configs: &[(usize, usize, usize, usize)] = &[
        (1, 32, 128, 512),
        (1, 64, 192, 768),
        (1, 64, 256, 1024),
        (1, 128, 384, 1536),
    ];

    eprintln!(
        "[ffn_speed] {:>9} {:>4} {:>4} {:>4} {:>5} | \
               {:>8} {:>8} {:>8} {:>9}",
        "device", "B", "S", "H", "FFN", "compile", "p50", "p95", "GFLOPs"
    );
    eprintln!("[ffn_speed] {}", "-".repeat(85));
    for &(b, s, h, ffn) in configs {
        let xs: Vec<f32> = det_random(7, b * s * h, 0.1);
        // Three rows per shape:
        //   cpu        — rlx-cpu (NEON + OpenBLAS / Accelerate), F32
        //   tpu (f32)  — XLA via PJRT, F32 throughout
        //   tpu (bf16) — XLA via PJRT, AutoMixedBf16 (TPU's native
        //                compute dtype; ~2× the f32 throughput on
        //                silicon, smaller win on the CPU plugin)
        let cpu_row = run_one(
            "cpu",
            Device::Cpu,
            PrecisionPolicy::AlwaysF32,
            b,
            s,
            h,
            ffn,
            &xs,
        );
        let tpu_f32 = run_one(
            "tpu/f32",
            Device::Tpu,
            PrecisionPolicy::AlwaysF32,
            b,
            s,
            h,
            ffn,
            &xs,
        );
        let tpu_bf16 = run_one(
            "tpu/bf16",
            Device::Tpu,
            PrecisionPolicy::AutoMixedBf16,
            b,
            s,
            h,
            ffn,
            &xs,
        );
        for r in [cpu_row, tpu_f32, tpu_bf16] {
            eprintln!(
                "[ffn_speed] {:>9} {:>4} {:>4} {:>4} {:>5} | \
                       {:>5}ms {:>5}µs {:>5}µs {:>9.2}",
                r.label, b, s, h, ffn, r.compile_ms, r.p50_us, r.p95_us, r.gflops
            );
        }
    }
}
