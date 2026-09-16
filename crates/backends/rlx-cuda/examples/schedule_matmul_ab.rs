// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Does generating `matmul` from its schedule actually beat the hand-written
//! kernel?**
//!
//! `rlx_gpu_kernels::kernel_schedule_emit` builds the `matmul` entry point from
//! a `rlx_ir::kernel_schedule::KernelSchedule` instead of templating
//! `kernels/matmul.cu`. That is only worth having if a *declaration* can buy
//! something the hand-written text cannot express — here, a `stages`-deep
//! `cp.async` rotation with one block sync per K iteration instead of two.
//!
//! This example is the measurement, and it is written so that it can say no.
//!
//! # Three arms, because two cannot attribute a win
//!
//! | arm | source | stages | syncs / K iter |
//! |---|---|---|---|
//! | `default` | `kernels/matmul.cu` templated by `#define` | 1 | 2 |
//! | `serial` | **emitted** from the schedule describing that kernel | 1 | 2 |
//! | `pipelined:N` | **emitted** from `matmul_pipelined_schedule` | N | 1 |
//!
//! `serial` is the arm that makes the result interpretable. It emits the same
//! physical schedule as the baseline, so it should land on top of it; if it
//! does not, the emitter itself is costing (or gaining) something and any
//! `pipelined` delta cannot be attributed to the pipeline. Reporting only
//! `default` vs `pipelined` would confound the two.
//!
//! # Protocol
//!
//! Copied from `tune_dispatch` deliberately, so the two are comparable:
//!
//! * **Bit-exactness gates every arm.** All three compute the same `kk` order
//!   over the same tile contents, so a differing bit is a bug in the schedule,
//!   not a faster kernel. An arm that fails this is reported and excluded — it
//!   never gets a speedup number.
//! * **L2 flushed before each timed sample**, each sample timed individually,
//!   median of 30 reported. A single timer around 30 iterations measures a
//!   warm-L2 best case.
//! * **The GPU must be idle.** A contended device produced a 25x spread and a
//!   different winner on this rig before the gate existed.
//! * **cuBLAS is out of the way** (`no_cublas`), or dense f32 GEMM never
//!   reaches rlx's own kernel at all and all three arms measure cuBLAS.
//!
//! # What this does not claim
//!
//! The reference is rlx's own default schedule, not an external library. "1.2x
//! the default tile" says the pipeline helped; it says nothing about the
//! distance to cuBLAS or to a tuned CUTLASS kernel — `reference_perf` is the
//! example for that question.
//!
//! ```sh
//! cargo run --release -p rlx-cuda --features schedule-codegen \
//!     --example schedule_matmul_ab
//! cargo run --release -p rlx-cuda --features schedule-codegen \
//!     --example schedule_matmul_ab -- --stages 2,3,4
//! ```

use std::time::Instant;

use rlx_cuda::backend::CudaExecutable;
use rlx_cuda::config::{MatmulScheduleSource, install_runtime_config, runtime_config};
use rlx_gpu_kernels::dispatch::{self, Choice, GpuArch, Workload};
use rlx_gpu_kernels::tiles::TileParams;
use rlx_ir::{DType, Graph, Shape};

const WARMUP: usize = 5;
const ITERS: usize = 30;
/// Sized past any current NVIDIA L2 so one pass evicts the working set.
const L2_FLUSH_BYTES: usize = 128 << 20;
/// Above this sustained utilization the numbers are contention, not measurement.
const MAX_BUSY_PERCENT: f32 = 15.0;

/// The workload contract, stated in one place (CAKE §4).
struct Contract {
    workload: &'static str,
    oracle: &'static str,
    tolerance: &'static str,
    hardware: &'static str,
    references: &'static str,
    search: &'static str,
}

const CONTRACT: Contract = Contract {
    workload: "dense f32 GEMM through rlx's own tiled `matmul`, cuBLAS disabled",
    oracle: "the `default` arm's output on the same device, same tile, same K order",
    tolerance: "bit-exact; any differing element disqualifies the arm",
    hardware: "GPU idle (<=15% util over 5 NVML samples) and not at its power cap",
    references: "rlx's own default schedule only — NOT cuBLAS, NOT an external \
                 library (see the `reference_perf` example for that)",
    search: "the schedules named on the command line, at the tile the dispatch \
             table resolves for each shape",
};

/// Shapes to measure, spanning the regimes where the answer should differ.
///
/// The pipeline overlaps a global load with compute and needs enough K tiles to
/// fill its stages, so it should help most where K is long and the block is
/// full, and help least at decode (`m = 1`), where most of the block is masked
/// off and the kernel is bandwidth-bound on B regardless. Both are included
/// because an experiment that only measures where it expects to win is not one.
const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("decode, LM hidden", 1, 4096, 4096),
    ("decode, mlp up", 1, 2048, 8192),
    ("tiny batch", 4, 4096, 4096),
    ("small batch", 32, 4096, 4096),
    ("medium batch", 128, 4096, 4096),
    ("short prefill", 256, 2048, 2048),
    ("medium prefill", 512, 2048, 2048),
    ("prefill, wide hidden", 512, 4096, 4096),
    ("large prefill", 2048, 2048, 2048),
    ("large square", 4096, 4096, 4096),
    ("fat k, small mn", 192, 8192, 512),
    ("tall, short k (dW)", 4096, 512, 512),
];

/// Whether the emitted kernel's pipelined path is reachable at this shape.
///
/// Mirrors the `pipe_ok` predicate in the generated CUDA. It matters because a
/// shape where the pipeline never executes still runs, still passes
/// bit-exactness, and still produces a timing — one that measures the serial
/// fallback plus a branch. Reporting `1.00x` for those without saying the
/// treatment never applied is how a null result gets read as a real one.
///
/// The predicate is per-block in the kernel; this is the whole-shape version,
/// so it answers "does any block take it", which is what a `—` in the table
/// should mean.
fn pipe_reachable(m: usize, k: usize, n: usize, tile: TileParams) -> bool {
    let (bm, bn, bk) = (tile.bm as usize, tile.bn as usize, tile.bk as usize);
    m >= bm
        && n >= bn
        && k.is_multiple_of(bk)
        && k.is_multiple_of(4)
        && n.is_multiple_of(4)
        && k.div_ceil(bk) >= 2
}

fn build(m: usize, k: usize, n: usize) -> Graph {
    let mut g = Graph::new("sched_ab");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[k, n], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);
    g
}

/// Deterministic inputs — a fixed LCG, so a rerun compares like with like and a
/// bit-exactness failure is reproducible rather than a one-off draw.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn flush_l2() {
    static mut SCRATCH: Option<Vec<u8>> = None;
    // A host-side memset is enough to evict the device's working set here only
    // because each `exe.run` re-uploads its inputs; the device-side path is the
    // arena traffic that follows. Kept identical to `tune_dispatch::flush_l2`
    // so the two examples' absolute numbers stay comparable.
    let buf = unsafe {
        let p = &raw mut SCRATCH;
        (*p).get_or_insert_with(|| vec![0u8; L2_FLUSH_BYTES])
    };
    for chunk in buf.chunks_mut(4096) {
        chunk[0] = chunk[0].wrapping_add(1);
    }
}

fn contention_check() -> Result<String, String> {
    if rlx_ir::env::flag("RLX_ALLOW_THROTTLE") {
        return Ok("RLX_ALLOW_THROTTLE=1 — contention gate bypassed".into());
    }
    const SAMPLES: usize = 5;
    let mut min_util: Option<f32> = None;
    let mut last = None;
    for i in 0..SAMPLES {
        let Some(s) = rlx_cuda::nvml::sample(0) else {
            return Ok("NVML unavailable — cannot verify the GPU is idle".into());
        };
        if let Some(u) = s.util_percent {
            min_util = Some(min_util.map_or(u, |m: f32| m.min(u)));
        }
        last = Some(s);
        if i + 1 < SAMPLES {
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }
    let Some(s) = last else {
        return Ok("NVML unavailable".into());
    };
    let mut problems = Vec::new();
    if let Some(u) = min_util
        && u > MAX_BUSY_PERCENT
    {
        problems.push(format!(
            "{u:.0}% utilization sustained over {SAMPLES} samples"
        ));
    }
    if let (Some(p), Some(cap)) = (s.power_w, s.power_cap_w)
        && p >= cap * 0.98
    {
        problems.push(format!("at the {cap:.0} W power cap ({p:.0} W)"));
    }
    if problems.is_empty() {
        Ok(format!(
            "idle: {}% util (min of {SAMPLES}), {} °C, {} MHz",
            min_util.map(|u| format!("{u:.0}")).unwrap_or("?".into()),
            s.temp_c.map(|t| format!("{t:.0}")).unwrap_or("?".into()),
            s.sm_clock_mhz.map(|c| c.to_string()).unwrap_or("?".into()),
        ))
    } else {
        Err(problems.join("; "))
    }
}

/// Install `schedule` and run one shape, returning (output, median ms).
///
/// `iters == 1` with no warmup is the correctness-only mode: bit-exactness does
/// not depend on the device being idle, so the two halves of this experiment can
/// be run at different times. The returned time is then meaningless and the
/// caller must not print it — which is why `--correctness` suppresses the
/// timing columns entirely rather than labelling them.
fn measure_n(
    schedule: MatmulScheduleSource,
    m: usize,
    k: usize,
    n: usize,
    xv: &[f32],
    wv: &[f32],
    warmup: usize,
    iters: usize,
) -> (Vec<f32>, f64) {
    let mut cfg = runtime_config();
    cfg.schedule_matmul = schedule;
    // cuBLAS in front of the tiled kernel would make all three arms measure the
    // same vendor GEMM.
    cfg.no_cublas = true;
    install_runtime_config(cfg);

    let mut exe = CudaExecutable::compile(build(m, k, n));
    exe.set_param("w", wv);
    for _ in 0..warmup {
        let _ = exe.run(&[("x", xv)]);
    }
    let mut samples = Vec::with_capacity(iters);
    let mut out = Vec::new();
    for _ in 0..iters {
        flush_l2();
        let t0 = Instant::now();
        let r = exe.run(&[("x", xv)]);
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
        out = r[0].clone();
    }
    samples.sort_by(|a, b| a.partial_cmp(b).expect("timings are finite"));
    (out, samples[samples.len() / 2])
}

/// Index and values of the first differing element, if any.
fn first_diff(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    if a.len() != b.len() {
        return Some((usize::MAX, a.len() as f32, b.len() as f32));
    }
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
        .map(|i| (i, a[i], b[i]))
}

struct Row {
    shape: &'static str,
    tile: String,
    default_ms: f64,
    /// Whether the emitted pipelined path executes at all at this shape.
    pipe_reachable: bool,
    /// Per-arm (label, ms, bit-exact) — `None` when the arm did not run.
    arms: Vec<(String, Option<f64>, bool)>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let stages: Vec<usize> = args
        .iter()
        .position(|a| a == "--stages")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 3]);

    println!("schedule_matmul_ab — is the emitted kernel faster than matmul.cu?\n");
    println!("  workload   : {}", CONTRACT.workload);
    println!("  oracle     : {}", CONTRACT.oracle);
    println!("  tolerance  : {}", CONTRACT.tolerance);
    println!("  hardware   : {}", CONTRACT.hardware);
    println!("  references : {}", CONTRACT.references);
    println!("  search     : {}", CONTRACT.search);
    // Correctness and performance are separable questions, and only one of them
    // needs an idle GPU. Splitting them means a busy rig can still answer "does
    // the emitted kernel compute the right thing" today.
    let correctness_only = args.iter().any(|a| a == "--correctness");
    // Sample count is tunable so a short sweep is possible on a rig that is
    // only briefly free. Fewer samples is a noisier median, not a wrong one —
    // and the `serial` control arm reports exactly how much noise it bought,
    // so a quick run is readable rather than merely fast.
    let cli_iters: Option<usize> = args
        .iter()
        .position(|a| a == "--iters")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.trim().parse().ok());
    let (warmup, iters) = if correctness_only {
        (0, 1)
    } else {
        let it = cli_iters.unwrap_or(ITERS).max(1);
        (WARMUP.min(it), it)
    };

    if correctness_only {
        println!("  mode       : CORRECTNESS ONLY — one run per arm, no timings reported");
        println!("  protocol   : bit-exactness vs the default arm; device state irrelevant\n");
    } else {
        println!("  protocol   : L2 flush per sample, median of {iters}, {warmup} warmup\n");
        match contention_check() {
            Ok(state) => println!("  device     : {state}\n"),
            Err(why) => {
                eprintln!("REFUSED: the GPU is not in a measurable state — {why}");
                eprintln!(
                    "Re-run when it is idle, pass --correctness to check the arms without \
                     timing them,\nor set RLX_ALLOW_THROTTLE=1 to measure anyway (the numbers \
                     will not mean much)."
                );
                std::process::exit(1);
            }
        }
    }

    // Any context names the arch; compile a trivial graph to get one.
    let arch = {
        let mut g = Graph::new("probe");
        let x = g.input("x", Shape::new(&[1], DType::F32));
        g.set_outputs(vec![x]);
        let _ = CudaExecutable::compile(g);
        rlx_cuda::device::cuda_context()
            .and_then(|ctx| ctx.compute_capability().ok())
            .map(|(maj, min)| GpuArch::cuda(maj as u32, min as u32))
            .unwrap_or_else(GpuArch::unknown)
    };
    // Load the persisted tuning cache NOW, then clear it. It otherwise loads
    // lazily on the first matmul dispatch — inside the first `measure()` — and
    // would reinstall a previous tuning run's tile mid-experiment, so the arms
    // would not all be compiling the same tile. (`tune_dispatch` was measuring
    // against itself for exactly this reason.)
    rlx_cuda::tuning::ensure_tuning_cache_loaded();
    dispatch::clear_overrides();
    println!("  arch       : {}\n", arch.as_str());

    let mut arms: Vec<(String, MatmulScheduleSource)> =
        vec![("serial".into(), MatmulScheduleSource::Serial)];
    for s in &stages {
        arms.push((format!("pipe:{s}"), MatmulScheduleSource::Pipelined(*s)));
    }

    let mut rows = Vec::new();
    for (shape, m, k, n) in SHAPES {
        let (m, k, n) = (*m, *k, *n);
        // Pin the tile the table would pick, so every arm compiles the SAME
        // tile and the only variable is the schedule.
        let tile = match dispatch::resolve_matmul(&arch, m, k, n) {
            Choice::MatmulTiled(t) => t,
            _ => TileParams::DEFAULT_MATMUL,
        };
        let _ = dispatch::set_override(
            Workload::Matmul { m, k, n }.key(&arch),
            Choice::MatmulTiled(tile),
        );

        let xv = fill(m * k, 0x5eed_1234);
        let wv = fill(k * n, 0xbeef_9876);

        let (reference, default_ms) = measure_n(
            MatmulScheduleSource::Default,
            m,
            k,
            n,
            &xv,
            &wv,
            warmup,
            iters,
        );

        let mut measured = Vec::new();
        for (label, sched) in &arms {
            let (out, ms) = measure_n(*sched, m, k, n, &xv, &wv, warmup, iters);
            match first_diff(&reference, &out) {
                None => measured.push((label.clone(), Some(ms), true)),
                Some((i, a, b)) => {
                    eprintln!(
                        "  {shape}: arm `{label}` is NOT bit-exact (element {i}: {a:e} vs {b:e}) \
                         — excluded from the ranking"
                    );
                    measured.push((label.clone(), Some(ms), false));
                }
            }
        }
        rows.push(Row {
            shape,
            tile: tile.label(),
            default_ms,
            pipe_reachable: pipe_reachable(m, k, n, tile),
            arms: measured,
        });
    }

    // ── Report ──────────────────────────────────────────────────────────
    let headers: Vec<&str> = rows
        .first()
        .map(|r| r.arms.iter().map(|(l, _, _)| l.as_str()).collect())
        .unwrap_or_default();
    if correctness_only {
        println!("\n{:<22} {:<20}", "shape", "tile");
        let mut wrong = 0usize;
        for r in &rows {
            print!(
                "{:<22} {:<20} {:<10}",
                r.shape,
                r.tile,
                if r.pipe_reachable {
                    "pipe:yes"
                } else {
                    "pipe:NO"
                }
            );
            for (label, _, exact) in &r.arms {
                if *exact {
                    print!("  {label}=bit-exact");
                } else {
                    wrong += 1;
                    print!("  {label}=WRONG");
                }
            }
            println!();
        }
        println!(
            "\n{} arm-shape pair(s) differed from the default arm's output.",
            wrong
        );
        if wrong == 0 {
            println!(
                "Every emitted schedule reproduces `matmul.cu` bit-for-bit. Timings still \
                 need an idle GPU:\n  cargo run --release -p rlx-cuda --features \
                 schedule-codegen --example schedule_matmul_ab"
            );
        }
        return;
    }

    print!(
        "\n{:<22} {:<20} {:<9} {:>10}",
        "shape", "tile", "pipe?", "default ms"
    );
    for h in &headers {
        print!(" {h:>12}");
    }
    println!();
    println!("{}", "-".repeat(64 + 13 * headers.len()));

    // Geometric mean per arm, over the shapes where that arm was bit-exact.
    // A second geomean, over only the shapes where the pipeline is reachable,
    // is reported for the pipelined arms below — averaging in shapes where the
    // treatment provably did not apply drags every result toward 1.00x and
    // makes a real effect look like noise.
    let mut logsum = vec![0.0f64; headers.len()];
    let mut counted = vec![0usize; headers.len()];
    let mut logsum_reach = vec![0.0f64; headers.len()];
    let mut counted_reach = vec![0usize; headers.len()];
    for r in &rows {
        print!(
            "{:<22} {:<20} {:<9} {:>10.4}",
            r.shape,
            r.tile,
            if r.pipe_reachable { "yes" } else { "NO" },
            r.default_ms
        );
        for (i, (_, ms, exact)) in r.arms.iter().enumerate() {
            match (ms, exact) {
                (Some(ms), true) => {
                    let speedup = r.default_ms / ms;
                    print!(" {speedup:>11.3}x");
                    logsum[i] += speedup.ln();
                    counted[i] += 1;
                    if r.pipe_reachable {
                        logsum_reach[i] += speedup.ln();
                        counted_reach[i] += 1;
                    }
                }
                (Some(_), false) => print!(" {:>12}", "WRONG"),
                (None, _) => print!(" {:>12}", "—"),
            }
        }
        println!();
    }
    println!("{}", "-".repeat(64 + 13 * headers.len()));
    let geo = |sum: f64, n: usize| {
        if n == 0 {
            "n/a".to_string()
        } else {
            format!("{:.3}x", (sum / n as f64).exp())
        }
    };
    print!(
        "{:<22} {:<20} {:<9} {:>10}",
        "geomean, all shapes", "", "", "1.000"
    );
    for i in 0..headers.len() {
        print!(" {:>12}", geo(logsum[i], counted[i]));
    }
    println!();
    print!(
        "{:<22} {:<20} {:<9} {:>10}",
        "geomean, pipe reachable", "", "", "1.000"
    );
    for i in 0..headers.len() {
        print!(" {:>12}", geo(logsum_reach[i], counted_reach[i]));
    }
    println!();
    // ── Noise floor, measured rather than assumed ───────────────────────
    //
    // `serial` is the same physical schedule as the baseline, so every row's
    // deviation of that column from 1.000 is this machine's noise on that row.
    // It is the only honest scale to read the other columns against.
    //
    // This matters most when the device is NOT idle. `RLX_ALLOW_THROTTLE=1`
    // makes the contention gate advisory, and without a measured floor the
    // resulting table looks exactly like a clean one — which is how the ROCm
    // sweep first produced a `serial` column ranging 0.79x-1.25x that every
    // ratio beside it was silently read against.
    const CONTROL_TOL: f64 = 0.10;
    if let Some(ci) = headers.iter().position(|h| *h == "serial") {
        let mut untrustworthy = Vec::new();
        let mut worst = 0.0f64;
        for r in &rows {
            if let Some((_, Some(ms), true)) = r.arms.get(ci) {
                let dev = (r.default_ms / ms - 1.0).abs();
                worst = worst.max(dev);
                if dev > CONTROL_TOL {
                    untrustworthy.push((r.shape, r.default_ms / ms));
                }
            }
        }
        println!(
            "\nNoise floor from the `serial` control (same kernel as the baseline, so it \
             must read 1.000):\n  worst deviation {:.1}%, tolerance {:.0}%",
            worst * 100.0,
            CONTROL_TOL * 100.0
        );
        if untrustworthy.is_empty() {
            println!("  every row is inside the floor — the ratios above are readable.");
        } else {
            println!(
                "  {} row(s) exceed it; their ratios are NOT measurements:",
                untrustworthy.len()
            );
            for (shape, got) in &untrustworthy {
                println!("    {shape:<22} serial={got:.3}x");
            }
            println!(
                "  A `pipe:*` number on those rows counts only if it falls well outside \
                 this\n  floor. Re-run on an idle device for a clean read."
            );
        }
    }

    let unreachable = rows.iter().filter(|r| !r.pipe_reachable).count();
    if unreachable > 0 {
        println!(
            "\n{unreachable}/{} shapes cannot reach the pipelined path (m < BM, or K not a \
             whole\nnumber of BK tiles), so there the `pipe:*` arms run the same serial \
             fallback as\n`serial` and a ~1.00x reading means the treatment did not apply, \
             not that it\nfailed. The second geomean excludes them.",
            rows.len()
        );
    }

    for (i, h) in headers.iter().enumerate() {
        if counted[i] < rows.len() {
            println!(
                "\nNOTE: arm `{h}` contributed {}/{} shapes to its geomean; the rest \
                 were excluded (not bit-exact or did not run).",
                counted[i],
                rows.len()
            );
        }
    }
    println!(
        "\nReference is rlx's own default schedule. A number above 1.00x means the \
         emitted\nschedule beat `matmul.cu` at that shape — not that it is near \
         cuBLAS or peak."
    );
}
