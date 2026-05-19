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

//! Diagnostic micro-bench for the CPU-side perf cliff seen in
//! `tpu_cpu_speed`. Runs three increasingly-fused sub-graphs at a
//! single shape, reports GFLOPs for each, so we can localize the
//! slowdown:
//!
//!   1. matmul-only        — pure cblas_sgemm work
//!   2. matmul + GELU      — adds the activation
//!   3. matmul + GELU + LN — adds the norm + residual
//!
//! If matmul-only is ~50+ GFLOPs but the full FFN drops to single
//! digits, the bottleneck is the glue. If matmul-only is already
//! slow, it's the BLAS link / threading config.
//!
//! Gated on RLX_TPU_BENCH=1 (this is a perf probe, not correctness).

use std::time::Instant;

use rlx_driver::Device;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{PrecisionPolicy, Session};

const ITERS: usize = 25;
const WARMUP: usize = 3;

/// Whether to run TPU rows. Skips cleanly on hosts without a PJRT
/// plugin so the test still produces useful CPU-only output (e.g.
/// when run on macOS host to capture Apple Accelerate numbers).
#[allow(dead_code)]
fn tpu_available() -> bool {
    std::env::var("LIBTPU_PATH").is_ok()
}

fn skip_unless_bench() -> bool {
    std::env::var("RLX_TPU_BENCH").is_err()
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

fn time_run(
    label: &'static str,
    exec: &mut rlx_runtime::CompiledGraph,
    xs: &[f32],
    flops: f64,
) -> u128 {
    for _ in 0..WARMUP {
        let _ = exec.run(&[("x", xs)]);
    }
    let mut times: Vec<u128> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let _ = exec.run(&[("x", xs)]);
        times.push(t.elapsed().as_micros());
    }
    times.sort_unstable();
    let p50 = times[times.len() / 2];
    let gflops = flops / (p50 as f64 * 1e3);
    eprintln!(
        "[cpu_perf_diag] {:<22} p50={}µs  {:.2} GFLOPs",
        label, p50, gflops
    );
    p50
}

fn build_matmul(b: usize, s: usize, h: usize, ffn: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("mm_only");
    let i64v = |dims: &[usize]| -> Vec<i64> { dims.iter().map(|&d| d as i64).collect() };
    let bs = b * s;
    let x = g.input("x", Shape::new(&[b, s, h], f));
    let w = g.param("w", Shape::new(&[h, ffn], f));
    let x_2d = g.reshape(x, i64v(&[bs, h]), Shape::new(&[bs, h], f));
    let y_2d = g.matmul(x_2d, w, Shape::new(&[bs, ffn], f));
    let out = g.reshape(y_2d, i64v(&[b, s, ffn]), Shape::new(&[b, s, ffn], f));
    g.set_outputs(vec![out]);
    g
}

fn build_matmul_act(
    b: usize,
    s: usize,
    h: usize,
    ffn: usize,
    name: &str,
    act: Activation,
) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new(name);
    let i64v = |dims: &[usize]| -> Vec<i64> { dims.iter().map(|&d| d as i64).collect() };
    let bs = b * s;
    let x = g.input("x", Shape::new(&[b, s, h], f));
    let w = g.param("w", Shape::new(&[h, ffn], f));
    let x_2d = g.reshape(x, i64v(&[bs, h]), Shape::new(&[bs, h], f));
    let y_2d = g.matmul(x_2d, w, Shape::new(&[bs, ffn], f));
    let activated = g.activation(act, y_2d, Shape::new(&[bs, ffn], f));
    let out = g.reshape(activated, i64v(&[b, s, ffn]), Shape::new(&[b, s, ffn], f));
    g.set_outputs(vec![out]);
    g
}

fn build_full_ffn(b: usize, s: usize, h: usize, ffn: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("full_ffn");
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

fn run_at_shape(b: usize, s: usize, h: usize, ffn: usize) {
    let bs = (b * s) as f64;
    let mm_flops = 2.0 * bs * (h as f64) * (ffn as f64);

    let xs = det_random(7, b * s * h, 0.1);
    let w_h_ffn = det_random(11, h * ffn, 0.04);
    let w_ffn_h = det_random(22, ffn * h, 0.04);
    let ln_g = vec![1.0_f32; h];
    let ln_b = vec![0.0_f32; h];

    eprintln!("[cpu_perf_diag] === shape (B={b}, S={s}, H={h}, FFN={ffn}) ===");

    // CPU side, several configurations.
    {
        let mut e = Session::new(Device::Cpu)
            .with_policy(PrecisionPolicy::AlwaysF32)
            .compile(build_matmul(b, s, h, ffn));
        e.set_param("w", &w_h_ffn);
        time_run("cpu mm-only", &mut e, &xs, mm_flops);
    }
    for (label, act) in [
        ("cpu mm+relu", Activation::Relu),
        ("cpu mm+sigmoid", Activation::Sigmoid),
        ("cpu mm+tanh", Activation::Tanh),
        ("cpu mm+gelu", Activation::Gelu),
    ] {
        let mut e = Session::new(Device::Cpu)
            .with_policy(PrecisionPolicy::AlwaysF32)
            .compile(build_matmul_act(b, s, h, ffn, &format!("mm_{label}"), act));
        e.set_param("w", &w_h_ffn);
        time_run(
            Box::leak(label.to_string().into_boxed_str()),
            &mut e,
            &xs,
            mm_flops,
        );
    }
    {
        let mut e = Session::new(Device::Cpu)
            .with_policy(PrecisionPolicy::AlwaysF32)
            .compile(build_full_ffn(b, s, h, ffn));
        e.set_param("w_up", &w_h_ffn);
        e.set_param("w_down", &w_ffn_h);
        e.set_param("ln_g", &ln_g);
        e.set_param("ln_b", &ln_b);
        // Two matmuls in the FFN; report combined work.
        time_run("cpu full ffn (2 mm)", &mut e, &xs, mm_flops * 2.0);
    }

    // TPU side, full FFN for comparison. Skip silently when the
    // PJRT plugin isn't available (so the test is useful on macOS
    // host for capturing Accelerate-only numbers).
    #[cfg(feature = "tpu")]
    if tpu_available() {
        let mut e = Session::new(Device::Tpu)
            .with_policy(PrecisionPolicy::AlwaysF32)
            .compile(build_full_ffn(b, s, h, ffn));
        e.set_param("w_up", &w_h_ffn);
        e.set_param("w_down", &w_ffn_h);
        e.set_param("ln_g", &ln_g);
        e.set_param("ln_b", &ln_b);
        time_run("tpu full ffn (2 mm)", &mut e, &xs, mm_flops * 2.0);
    }
}

#[test]
fn cpu_perf_breakdown() {
    if skip_unless_bench() {
        return;
    }

    let openblas_threads =
        std::env::var("OPENBLAS_NUM_THREADS").unwrap_or_else(|_| "<unset>".into());
    eprintln!("[cpu_perf_diag] OPENBLAS_NUM_THREADS = {openblas_threads}");
    eprintln!(
        "[cpu_perf_diag] available_parallelism = {}",
        std::thread::available_parallelism()
            .map(|n| n.get().to_string())
            .unwrap_or_else(|_| "?".into())
    );
    let target = if cfg!(target_os = "macos") {
        "macos (Accelerate / AMX)"
    } else if cfg!(target_arch = "aarch64") {
        "linux/aarch64 (OpenBLAS)"
    } else {
        "x86_64 (OpenBLAS or MKL)"
    };
    eprintln!("[cpu_perf_diag] BLAS target = {target}");

    // Sweep across the speed-bench's shapes, plus the H=384 shape
    // we care about diagnosing.
    let shapes: &[(usize, usize, usize, usize)] =
        &[(1, 64, 192, 768), (1, 64, 256, 1024), (1, 128, 384, 1536)];
    for &(b, s, h, ffn) in shapes {
        run_at_shape(b, s, h, ffn);
    }
}
