// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-backend bench for the newly-added ops (SelectiveScan, Sample,
//! GroupNorm, LayerNorm2d, ResizeNearest2x). For each (op × device) it reports:
//!
//!   * **validity** — max abs diff of the output vs the CPU reference (PASS/FAIL)
//!   * **latency**  — median wall-clock per `run()` (synchronous; GPU readback
//!                    forces completion before the timer stops)
//!   * **throughput** — total I/O elements moved per second
//!   * **bandwidth**   — effective GB/s = (input + output bytes) / latency
//!   * **RAM limit**   — largest problem size that runs without OOM, + its
//!                       working-set bytes (sum of tensor sizes)
//!
//! ```sh
//! cargo run -p rlx-bench --release --example bench_new_ops                 # CPU only
//! cargo run -p rlx-bench --release --example bench_new_ops --features metal
//! cargo run -p rlx-bench --release --example bench_new_ops --features mlx
//! cargo run -p rlx-bench --release --example bench_new_ops --features gpu  # wgpu
//! cargo run -p rlx-bench --release --example bench_new_ops --features metal,mlx,gpu
//! ```

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 25;
const VALID_TOL: f32 = 1e-3;

/// All devices compiled into this binary. CPU is always present and is the
/// numerical reference.
fn devices() -> Vec<(Device, &'static str)> {
    let v = vec![(Device::Cpu, "cpu")];
    #[cfg(feature = "metal")]
    v.push((Device::Metal, "metal"));
    #[cfg(feature = "mlx")]
    v.push((Device::Mlx, "mlx"));
    #[cfg(feature = "gpu")]
    v.push((Device::Gpu, "wgpu"));
    v
}

/// One benchmarkable op instance: a graph + named inputs + an I/O byte/elem count.
struct Case {
    op: &'static str,
    shape: String,
    graph: Graph,
    inputs: Vec<(&'static str, Vec<f32>)>,
    io_elems: usize, // total input + output elements moved
    io_bytes: usize, // total input + output bytes moved (f32)
    /// Stochastic ops (Sample) use per-backend RNG, so an exact match vs CPU is
    /// the wrong criterion — validity is instead "every output is an in-range
    /// index `[0, range)`". `None` = deterministic (exact-match vs CPU).
    stochastic_range: Option<usize>,
}

fn median(mut xs: Vec<u128>) -> u128 {
    xs.sort_unstable();
    xs[xs.len() / 2]
}

/// Compile + run `case` on `device`. Returns (output, median_latency_ns) or
/// `None` if the op is unsupported / the run fails on that backend.
fn run_case(device: Device, case: &Case) -> Option<(Vec<f32>, u128)> {
    let refs: Vec<(&str, &[f32])> = case
        .inputs
        .iter()
        .map(|(n, v)| (*n, v.as_slice()))
        .collect();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut exe = Session::new(device).compile(case.graph.clone());
        for _ in 0..WARMUP {
            let _ = exe.run(&refs);
        }
        let mut samples = Vec::with_capacity(ITERS);
        let mut last = Vec::new();
        for _ in 0..ITERS {
            let t = Instant::now();
            last = exe.run(&refs).pop().unwrap_or_default();
            samples.push(t.elapsed().as_nanos());
        }
        (last, median(samples))
    }));
    result.ok()
}

fn fmt_ns(ns: u128) -> String {
    if ns < 1_000 {
        format!("{ns} ns")
    } else if ns < 1_000_000 {
        format!("{:.2} µs", ns as f64 / 1e3)
    } else {
        format!("{:.2} ms", ns as f64 / 1e6)
    }
}

fn bench_case(case: &Case, cpu_ref: &[f32]) {
    for (device, label) in devices() {
        let Some((out, lat_ns)) = run_case(device, case) else {
            println!("  {:<6} {:<22} unsupported / failed", label, "");
            continue;
        };
        // Validity. Stochastic ops (Sample) use per-backend RNG, so check
        // in-range indices instead of an exact CPU match.
        let (max_abs, valid) = if let Some(range) = case.stochastic_range {
            let ok = !out.is_empty()
                && out
                    .iter()
                    .all(|&x| x >= 0.0 && (x as usize) < range && x.fract() == 0.0);
            (f32::NAN, ok)
        } else if out.len() == cpu_ref.len() {
            let m = out
                .iter()
                .zip(cpu_ref)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            (m, m <= VALID_TOL)
        } else {
            (f32::INFINITY, false)
        };
        let secs = lat_ns as f64 / 1e9;
        let gbps = case.io_bytes as f64 / secs / 1e9;
        let gelems = case.io_elems as f64 / secs / 1e9;
        let status = if valid { "PASS" } else { "FAIL" };
        println!(
            "  {:<6} lat {:<10} {:>8.2} GB/s {:>8.3} Gelem/s  valid {} (max|Δ|={:.2e})",
            label,
            fmt_ns(lat_ns),
            gbps,
            gelems,
            status,
            max_abs
        );
    }
}

// ── Graph builders ──────────────────────────────────────────────────────────

fn selective_scan_case(b: usize, s: usize, h: usize, n: usize) -> Case {
    let mut g = Graph::new("ssm");
    let bsh = Shape::new(&[b, s, h], DType::F32);
    let x = g.input("x", bsh.clone());
    let delta = g.input("delta", bsh.clone());
    let a = g.input("a", Shape::new(&[h, n], DType::F32));
    let bb = g.input("b", Shape::new(&[b, s, n], DType::F32));
    let c = g.input("c", Shape::new(&[b, s, n], DType::F32));
    let y = g.selective_scan(x, delta, a, bb, c, n, bsh);
    g.set_outputs(vec![y]);
    let mk = |len, seed: usize| {
        (0..len)
            .map(|i| 0.05 + 0.01 * ((i + seed) % 11) as f32)
            .collect()
    };
    let io_elems = 3 * b * s * h + h * n + 2 * b * s * n + b * s * h;
    Case {
        op: "SelectiveScan",
        stochastic_range: None,
        shape: format!("b{b}·s{s}·h{h}·n{n}"),
        graph: g,
        inputs: vec![
            ("x", mk(b * s * h, 1)),
            ("delta", mk(b * s * h, 2)),
            (
                "a",
                (0..h * n).map(|i| -0.5 + 0.05 * (i % 7) as f32).collect(),
            ),
            ("b", mk(b * s * n, 3)),
            ("c", mk(b * s * n, 4)),
        ],
        io_elems,
        io_bytes: io_elems * 4,
    }
}

fn sample_case(b: usize, v: usize) -> Case {
    let mut g = Graph::new("sample");
    let logits = g.input("logits", Shape::new(&[b, v], DType::F32));
    let y = g.sample(logits, 40, 0.95, 0.8, 1234, Shape::new(&[b], DType::F32));
    g.set_outputs(vec![y]);
    let data = (0..b * v)
        .map(|i| ((i % v) as f32 * 0.37).sin() * 3.0)
        .collect();
    let io_elems = b * v + b;
    Case {
        op: "Sample",
        stochastic_range: Some(v),
        shape: format!("b{b}·vocab{v}"),
        graph: g,
        inputs: vec![("logits", data)],
        io_elems,
        io_bytes: io_elems * 4,
    }
}

fn group_norm_case(n: usize, c: usize, h: usize, w: usize, groups: usize) -> Case {
    let mut g = Graph::new("gn");
    let x = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let gamma = g.input("gamma", Shape::new(&[c], DType::F32));
    let beta = g.input("beta", Shape::new(&[c], DType::F32));
    let y = g.group_norm(x, gamma, beta, groups, 1e-5);
    g.set_outputs(vec![y]);
    let xs = (0..n * c * h * w)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let io_elems = 2 * n * c * h * w + 2 * c;
    Case {
        op: "GroupNorm",
        stochastic_range: None,
        shape: format!("[{n},{c},{h},{w}] g{groups}"),
        graph: g,
        inputs: vec![
            ("x", xs),
            ("gamma", (0..c).map(|i| 1.0 + 0.01 * i as f32).collect()),
            ("beta", (0..c).map(|i| 0.01 * i as f32).collect()),
        ],
        io_elems,
        io_bytes: io_elems * 4,
    }
}

fn layer_norm2d_case(n: usize, c: usize, h: usize, w: usize) -> Case {
    let mut g = Graph::new("ln2d");
    let x = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let gamma = g.input("gamma", Shape::new(&[c], DType::F32));
    let beta = g.input("beta", Shape::new(&[c], DType::F32));
    let y = g.layer_norm2d(x, gamma, beta, 1e-6);
    g.set_outputs(vec![y]);
    let xs = (0..n * c * h * w)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let io_elems = 2 * n * c * h * w + 2 * c;
    Case {
        op: "LayerNorm2d",
        stochastic_range: None,
        shape: format!("[{n},{c},{h},{w}]"),
        graph: g,
        inputs: vec![
            ("x", xs),
            ("gamma", (0..c).map(|i| 1.0 + 0.01 * i as f32).collect()),
            ("beta", (0..c).map(|i| 0.01 * i as f32).collect()),
        ],
        io_elems,
        io_bytes: io_elems * 4,
    }
}

fn im2col_case(n: usize, c: usize, h: usize, w: usize, k: usize, s: usize, p: usize) -> Case {
    let mut g = Graph::new("im2col");
    let x = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let y = g.im2col(x, [k, k], [s, s], [p, p], [1, 1]);
    g.set_outputs(vec![y]);
    let xs = (0..n * c * h * w).map(|i| (i % 29) as f32 * 0.05).collect();
    let h_out = (h + 2 * p - k) / s + 1;
    let w_out = (w + 2 * p - k) / s + 1;
    let out_elems = n * h_out * w_out * c * k * k;
    let io_elems = n * c * h * w + out_elems;
    Case {
        op: "Im2Col",
        stochastic_range: None,
        shape: format!("[{n},{c},{h},{w}] k{k}s{s}p{p}"),
        graph: g,
        inputs: vec![("x", xs)],
        io_elems,
        io_bytes: io_elems * 4,
    }
}

fn resize_case(n: usize, c: usize, h: usize, w: usize) -> Case {
    let mut g = Graph::new("resize");
    let x = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let y = g.add_node(
        Op::ResizeNearest2x,
        vec![x],
        Shape::new(&[n, c, h * 2, w * 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let xs = (0..n * c * h * w).map(|i| (i % 23) as f32 * 0.05).collect();
    let io_elems = n * c * h * w + n * c * h * 2 * w * 2;
    Case {
        op: "ResizeNearest2x",
        stochastic_range: None,
        shape: format!("[{n},{c},{h},{w}]→2×"),
        graph: g,
        inputs: vec![("x", xs)],
        io_elems,
        io_bytes: io_elems * 4,
    }
}

/// Single compile+run probe (no timing loop) — used by the size-limit sweep,
/// where we only care whether a given size fits, not how fast it is.
fn probe_once(device: Device, case: &Case) -> bool {
    let refs: Vec<(&str, &[f32])> = case
        .inputs
        .iter()
        .map(|(n, v)| (*n, v.as_slice()))
        .collect();
    catch_unwind(AssertUnwindSafe(|| {
        let mut exe = Session::new(device).compile(case.graph.clone());
        let _ = exe.run(&refs);
    }))
    .is_ok()
}

/// Grow `mk(scale)`'s working set until a run fails (OOM / device limit).
/// Reports the largest size that ran on each device.
fn ram_sweep(name: &str, mk: impl Fn(usize) -> Case, scales: &[usize]) {
    println!("\n  RAM / size limit sweep — {name}");
    for (device, label) in devices() {
        let mut last_ok: Option<(String, usize)> = None;
        for &scale in scales {
            let case = mk(scale);
            let bytes = case.io_bytes;
            match probe_once(device, &case) {
                true => last_ok = Some((case.shape.clone(), bytes)),
                false => break,
            }
        }
        match last_ok {
            Some((shape, bytes)) => println!(
                "    {:<6} max ok: {:<22} working set {:>8.1} MB",
                label,
                shape,
                bytes as f64 / 1e6
            ),
            None => println!("    {:<6} no size ran", label),
        }
    }
}

fn main() {
    let devs: Vec<&str> = devices().iter().map(|(_, l)| *l).collect();
    println!(
        "rlx new-op cross-backend bench — devices: {}",
        devs.join(", ")
    );
    println!("(warmup {WARMUP}, {ITERS} timed iters, median latency; CPU = numerical reference)\n");

    let cases: Vec<Case> = vec![
        selective_scan_case(2, 256, 256, 16), // Mamba-ish decode-prefill
        selective_scan_case(1, 1024, 512, 16),
        sample_case(1, 32000),              // single-stream LLM decode
        sample_case(8, 32000),              // batched decode
        group_norm_case(2, 64, 56, 56, 32), // vision backbone stage
        group_norm_case(1, 256, 28, 28, 32),
        layer_norm2d_case(2, 64, 56, 56),
        layer_norm2d_case(1, 256, 28, 28),
        resize_case(1, 64, 128, 128), // U-Net decoder upsample
        resize_case(2, 32, 256, 256),
        im2col_case(1, 64, 56, 56, 3, 1, 1), // conv-as-matmul unfold
        im2col_case(2, 128, 28, 28, 3, 1, 1),
    ];

    let mut cur = "";
    for case in &cases {
        if case.op != cur {
            println!("\n{} ({})", case.op, case.shape);
            cur = case.op;
        } else {
            println!("\n ({})", case.shape);
        }
        // CPU reference (always available).
        let cpu_ref = run_case(Device::Cpu, case)
            .map(|(o, _)| o)
            .unwrap_or_default();
        bench_case(case, &cpu_ref);
    }

    // RAM / size-limit sweeps — scale the dominant dimension upward.
    // Scales kept modest so the slow sequential MLX SelectiveScan leg finishes;
    // bump them when probing a specific device's true ceiling.
    ram_sweep(
        "SelectiveScan (grow seq)",
        |s| selective_scan_case(2, 512 * s, 256, 16),
        &[1, 2, 4, 8],
    );
    ram_sweep(
        "GroupNorm (grow spatial)",
        |s| group_norm_case(2, 64, 56 * s, 56 * s, 32),
        &[1, 2, 4, 6],
    );
    ram_sweep(
        "ResizeNearest2x (grow spatial)",
        |s| resize_case(1, 64, 128 * s, 128 * s),
        &[1, 2, 4, 6],
    );

    println!("\ndone.");
}
