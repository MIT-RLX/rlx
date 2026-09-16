// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Does generating `matmul` from its schedule beat the hand-written kernel on
//! AMD?**
//!
//! The ROCm arm of the experiment in `rlx-cuda/examples/schedule_matmul_ab.rs`,
//! `rlx-metal/examples/schedule_sgemm_ab.rs` and
//! `rlx-wgpu/examples/wgpu_schedule_matmul_ab.rs`.
//!
//! # What differs from the CUDA arm, and why it matters
//!
//! ROCm compiles the **same `matmul.cu` text** through hipRTC, so the tile
//! space, the dispatch table and now the emitter are literally shared. But the
//! CUDA treatment is two changes at once — a rotating buffer *and* `cp.async`
//! staging — and only the first is portable. `cp.async` is an NVIDIA
//! instruction; handing it to hipRTC is a compile error, so
//! `rlx_gpu_kernels::kernel_schedule_emit::Lang::Hip` refuses it by name and
//! the pipelined schedule here declares no `Feature::AsyncCopy` at all.
//!
//! That makes this arm the **isolation experiment**. Metal and wgpu already
//! showed the rotation alone losing on Apple silicon. If it also loses here,
//! the rotation is simply a bad trade and CUDA's result (whatever it turns out
//! to be) is carried by `cp.async`. If it wins here, the Apple loss is a vendor
//! property. Either way the arm that is *missing* is what the reading turns on,
//! so `EmitFacts.async_copy` is printed rather than assumed.
//!
//! | arm | source | stages | syncs / K iter | async copy |
//! |---|---|---|---|---|
//! | `default` | `matmul.cu` via `#define` | 1 | 2 | no |
//! | `serial` | **emitted**, describing that kernel | 1 | 2 | no |
//! | `pipe:N` | **emitted**, rotation only | N | 1 | **no** |
//!
//! # Protocol
//!
//! Matches the CUDA sweep so the two are comparable: bit-exactness against the
//! `default` arm gates every result, all arms are warmed before any is timed,
//! samples are taken round-robin, and the median of 30 is reported. The
//! round-robin is not decoration — on Metal, timing arms back-to-back made the
//! first absorb the buffers' first-touch cost and every later arm read ~20%
//! fast.
//!
//! `RLX_ROCM_NO_VENDOR_GEMM=1` is set so dense f32 GEMM reaches rlx's own
//! tiled kernel; with rocBLAS in front, all three arms would measure rocBLAS.
//!
//! ```sh
//! RLX_ROCM_NO_VENDOR_GEMM=1 cargo run --release -p rlx-rocm \
//!     --features schedule-codegen --example rocm_schedule_matmul_ab
//! ```

#[cfg(not(feature = "schedule-codegen"))]
fn main() {
    eprintln!(
        "this example needs `--features schedule-codegen`:\n  \
         cargo run --release -p rlx-rocm --features schedule-codegen \
         --example rocm_schedule_matmul_ab"
    );
}

#[cfg(feature = "schedule-codegen")]
fn main() {
    use std::time::Instant;

    use rlx_gpu_kernels::MatmulSchedule;
    use rlx_gpu_kernels::dispatch::{self, Choice, Workload};
    use rlx_gpu_kernels::tiles::TileParams;
    use rlx_ir::{DType, Graph, Shape};
    use rlx_rocm::backend::RocmExecutable;

    const WARMUP: usize = 5;
    const ITERS: usize = 30;
    /// Above this sustained utilization the numbers are contention.
    const MAX_BUSY_PERCENT: f32 = 15.0;

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

    /// Device-memory a single arm's executable needs for this shape.
    ///
    /// One arena per arm holds x, w and y. Every arm is compiled up front (so
    /// compilation is not charged to whichever arm runs first), which means the
    /// peak is `arms * arena`, not `arena`.
    fn arena_bytes(m: usize, k: usize, n: usize) -> usize {
        (m * k + k * n + m * n) * std::mem::size_of::<f32>()
    }

    fn build(m: usize, k: usize, n: usize) -> Graph {
        let mut g = Graph::new("rocm_sched_ab");
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let w = g.param("w", Shape::new(&[k, n], DType::F32));
        let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);
        g
    }

    /// Which rocm-smi device index to poll.
    ///
    /// **Not hardcoded to 0.** This rig is mixed-arch (a gfx908 MI100 and a
    /// gfx1103 780M) and either card can be the busy one. Reading index 0 while
    /// running on device 1 refuses a perfectly idle GPU — or worse, passes a
    /// contended one. `HIP_VISIBLE_DEVICES` is what selects the device the run
    /// actually uses, so the gate follows it; `RLX_ROCM_SMI_INDEX` overrides
    /// when the mapping is not the identity.
    fn smi_index() -> u32 {
        if let Some(v) = rlx_ir::env::var("RLX_ROCM_SMI_INDEX")
            && let Ok(i) = v.trim().parse()
        {
            return i;
        }
        std::env::var("HIP_VISIBLE_DEVICES")
            .ok()
            .and_then(|v| v.split(',').next()?.trim().parse().ok())
            .unwrap_or(0)
    }

    /// Refuse to time a GPU somebody else is using.
    ///
    /// The AMD rig is shared and mixed-arch, so this reports which device it
    /// actually read rather than assuming index 0 is the one in play.
    fn contention_check() -> Result<String, String> {
        if rlx_ir::env::flag("RLX_ALLOW_THROTTLE") {
            return Ok("RLX_ALLOW_THROTTLE=1 — contention gate bypassed".into());
        }
        const SAMPLES: usize = 5;
        let mut min_util: Option<f32> = None;
        let mut last = None;
        for i in 0..SAMPLES {
            let Some(s) = rlx_rocm::rsmi::sample(smi_index()) else {
                // No rocm-smi is a coverage limitation, not a pass.
                return Ok("rocm-smi unavailable — cannot verify the GPU is idle".into());
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
            return Ok("rocm-smi unavailable".into());
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
                "[smi {}] {} idle: {}% util (min of {SAMPLES}), {} °C",
                smi_index(),
                s.name.clone().unwrap_or_else(|| "?".into()),
                min_util.map(|u| format!("{u:.0}")).unwrap_or("?".into()),
                s.temp_edge_c
                    .map(|t| format!("{t:.0}"))
                    .unwrap_or("?".into()),
            ))
        } else {
            Err(problems.join("; "))
        }
    }

    // rocBLAS in front of the tiled kernel would make every arm measure rocBLAS.
    rlx_ir::env::set("RLX_ROCM_NO_VENDOR_GEMM", "1");

    let args: Vec<String> = std::env::args().collect();
    let stages: Vec<usize> = args
        .iter()
        .position(|a| a == "--stages")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 3]);
    let correctness_only = args.iter().any(|a| a == "--correctness");
    // Device-memory ceiling for the whole arm set at one shape.
    //
    // Not decoration: this rig pairs a 32 GiB MI100 with a 2 GiB 780M iGPU, and
    // the shape list was written for the big card. On the small one the run
    // died with `HipError(719)` on the FIRST shape — a launch failure with no
    // hint that memory was the cause, and no indication of which shapes were
    // even attemptable. Budgeting up front turns that into a reported skip.
    let budget_mb: usize = args
        .iter()
        .position(|a| a == "--max-arena-mb")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(usize::MAX);
    // Sample count is tunable because the HSA runtime leaks a signal per
    // dispatch (`SharedSignalPool ... Signals leaked` on exit) and a small
    // device runs out: on the 2 GiB gfx1103 iGPU, 4 arms x 35 runs x 11 shapes
    // died with `HipError(719)` while the 4-run correctness pass over the same
    // shapes was clean. Fewer samples is a noisier median, not a wrong one, and
    // the `serial` control arm reports exactly how much noise that bought.
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

    println!("rocm_schedule_matmul_ab — is the emitted HIP faster than matmul.cu?\n");
    println!("  workload   : dense f32 GEMM via rlx's own tiled `matmul`, rocBLAS disabled");
    println!("  oracle     : the `default` arm's output, same device, same tile, same K order");
    println!("  tolerance  : bit-exact; any differing element disqualifies the arm");
    println!("  references : rlx's own default schedule only — NOT rocBLAS");
    println!(
        "  target     : {}",
        rlx_rocm::kernels::schedule_target().name
    );
    println!(
        "  arch       : {}",
        rlx_rocm::hip::rocm_target_arch().unwrap_or_else(|| "unknown".into())
    );
    if correctness_only {
        println!("  mode       : CORRECTNESS ONLY — no timings reported\n");
    } else {
        println!(
            "  protocol   : warm all arms, round-robin samples, median of {iters} \
             ({warmup} warmup)\n"
        );
        match contention_check() {
            Ok(state) => println!("  device     : {state}\n"),
            Err(why) => {
                eprintln!("REFUSED: the GPU is not in a measurable state — {why}");
                eprintln!(
                    "Re-run when idle, pass --correctness to check the arms without timing,\n\
                     or set RLX_ALLOW_THROTTLE=1 (the numbers will not mean much)."
                );
                std::process::exit(1);
            }
        }
    }

    let arch = rlx_rocm::kernels::gpu_arch().clone();
    let mut arms: Vec<(String, MatmulSchedule)> =
        vec![("serial".into(), MatmulSchedule::EmittedSerial)];
    for s in &stages {
        arms.push((
            format!("pipe:{s}"),
            MatmulSchedule::EmittedPipelined { stages: *s },
        ));
    }

    struct Row {
        shape: &'static str,
        tile: String,
        base_ms: f64,
        spread: f64,
        cols: Vec<(String, f64, bool)>,
    }
    let mut rows: Vec<Row> = Vec::new();

    let n_arms = arms.len() + 1; // + the default arm
    let mut skipped: Vec<(&str, usize)> = Vec::new();
    for (shape, m, k, n) in SHAPES {
        let (m, k, n) = (*m, *k, *n);
        let need_mb = arena_bytes(m, k, n) * n_arms / (1 << 20);
        if need_mb > budget_mb {
            skipped.push((shape, need_mb));
            continue;
        }
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

        let run_once = |s: MatmulSchedule, exe: &mut RocmExecutable| -> (Vec<f32>, f64) {
            rlx_rocm::kernels::set_matmul_schedule(s);
            let t0 = Instant::now();
            let r = exe.run(&[("x", &xv)]);
            (r[0].clone(), t0.elapsed().as_secs_f64() * 1e3)
        };

        // One executable per arm: compilation is per-schedule, and folding it
        // into the first timed sample would charge it to whichever arm ran
        // first.
        let mut exes: Vec<(String, MatmulSchedule, RocmExecutable)> = Vec::new();
        for (label, s) in
            std::iter::once(&("default".to_string(), MatmulSchedule::Default)).chain(arms.iter())
        {
            rlx_rocm::kernels::set_matmul_schedule(*s);
            let mut e = RocmExecutable::compile(build(m, k, n));
            e.set_param("w", &wv);
            exes.push((label.clone(), *s, e));
        }

        // Phase 1: correctness.
        let mut reference: Vec<f32> = Vec::new();
        let mut exact: Vec<bool> = Vec::new();
        for (i, (label, s, e)) in exes.iter_mut().enumerate() {
            let (out, _) = run_once(*s, e);
            if i == 0 {
                reference = out;
            } else {
                let ok = out.len() == reference.len()
                    && out
                        .iter()
                        .zip(&reference)
                        .all(|(x, y)| x.to_bits() == y.to_bits());
                if !ok {
                    let bad = out
                        .iter()
                        .zip(&reference)
                        .position(|(x, y)| x.to_bits() != y.to_bits())
                        .unwrap_or(0);
                    eprintln!(
                        "  {shape}: arm `{label}` NOT bit-exact (element {bad}: {:e} vs {:e})",
                        out[bad], reference[bad]
                    );
                }
                exact.push(ok);
            }
        }

        if correctness_only {
            rows.push(Row {
                shape,
                tile: tile.label(),
                base_ms: 0.0,
                spread: 0.0,
                cols: exes
                    .iter()
                    .skip(1)
                    .enumerate()
                    .map(|(i, (l, _, _))| (l.clone(), 0.0, exact[i]))
                    .collect(),
            });
            continue;
        }

        // Phase 2: warm EVERY arm before any is timed.
        for (_, s, e) in exes.iter_mut() {
            for _ in 0..warmup {
                run_once(*s, e);
            }
        }
        // Phase 3: round-robin.
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(iters); exes.len()];
        for _ in 0..iters {
            for (i, (_, s, e)) in exes.iter_mut().enumerate() {
                let (_, ms) = run_once(*s, e);
                samples[i].push(ms);
            }
        }
        let stat = |v: &mut Vec<f64>| -> (f64, f64) {
            v.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
            (v[v.len() / 2], v[v.len() - 1] / v[0].max(f64::MIN_POSITIVE))
        };
        let (base_ms, spread) = stat(&mut samples[0]);
        let cols = (1..exes.len())
            .map(|i| {
                let (med, _) = stat(&mut samples[i]);
                (exes[i].0.clone(), med, exact[i - 1])
            })
            .collect();
        rows.push(Row {
            shape,
            tile: tile.label(),
            base_ms,
            spread,
            cols,
        });
    }

    // Silent truncation reads exactly like full coverage. Say what was dropped
    // before saying anything about what was measured.
    if !skipped.is_empty() {
        println!(
            "\nSKIPPED {} of {} shapes: they need more than the {budget_mb} MiB arena \
             budget\n(--max-arena-mb). These are NOT in any number below.",
            skipped.len(),
            SHAPES.len()
        );
        for (shape, mb) in &skipped {
            println!("  {shape:<22} needs ~{mb} MiB across {n_arms} arms");
        }
        println!();
    }

    // ── Report ──────────────────────────────────────────────────────────
    let headers: Vec<String> = rows
        .first()
        .map(|r| r.cols.iter().map(|(l, _, _)| l.clone()).collect())
        .unwrap_or_default();

    if correctness_only {
        let mut wrong = 0usize;
        for r in &rows {
            print!("{:<22} {:<20}", r.shape, r.tile);
            for (l, _, ok) in &r.cols {
                if *ok {
                    print!("  {l}=bit-exact");
                } else {
                    wrong += 1;
                    print!("  {l}=WRONG");
                }
            }
            println!();
        }
        println!("\n{wrong} arm-shape pair(s) differed from the default arm's output.");
        return;
    }

    print!(
        "{:<22} {:<20} {:>10} {:>7}",
        "shape", "tile", "default ms", "spread"
    );
    for h in &headers {
        print!(" {h:>12}");
    }
    println!();
    println!("{}", "-".repeat(62 + 13 * headers.len()));
    let mut logsum = vec![0.0f64; headers.len()];
    let mut counted = vec![0usize; headers.len()];
    for r in &rows {
        print!(
            "{:<22} {:<20} {:>10.4} {:>6.2}x",
            r.shape, r.tile, r.base_ms, r.spread
        );
        for (i, (_, ms, ok)) in r.cols.iter().enumerate() {
            if *ok {
                let sp = r.base_ms / ms;
                print!(" {sp:>11.3}x");
                logsum[i] += sp.ln();
                counted[i] += 1;
            } else {
                print!(" {:>12}", "WRONG");
            }
        }
        println!();
    }
    println!("{}", "-".repeat(62 + 13 * headers.len()));
    print!(
        "{:<22} {:<20} {:>10} {:>7}",
        "geomean (bit-exact)", "", "1.000", ""
    );
    for i in 0..headers.len() {
        if counted[i] == 0 {
            print!(" {:>12}", "n/a");
        } else {
            print!(" {:>11.3}x", (logsum[i] / counted[i] as f64).exp());
        }
    }
    println!();

    // ── Noise floor, measured rather than assumed ───────────────────────
    //
    // `serial` is the same physical schedule as the baseline, so every row's
    // deviation of that column from 1.000 is this rig's noise on that row. It
    // is the only honest scale to read the other columns against, and on a
    // shared machine it can be large enough that a printed ratio means nothing.
    // Reporting it beats printing three decimal places over a +/-25% floor.
    const CONTROL_TOL: f64 = 0.10;
    if let Some(ci) = headers.iter().position(|h| h == "serial") {
        let mut untrustworthy = Vec::new();
        let mut worst = 0.0f64;
        for r in &rows {
            if let Some((_, ms, true)) = r.cols.get(ci) {
                let dev = (r.base_ms / ms - 1.0).abs();
                worst = worst.max(dev);
                if dev > CONTROL_TOL {
                    untrustworthy.push((r.shape, r.base_ms / ms));
                }
            }
        }
        println!(
            "\nNoise floor from the `serial` control (same kernel as the baseline, so it \
             must read 1.000):\n  worst deviation {:.1}%, tolerance {:.0}%",
            worst * 100.0,
            CONTROL_TOL * 100.0
        );
        if !untrustworthy.is_empty() {
            println!(
                "  {} row(s) exceed it and their ratios should NOT be read as measurements:",
                untrustworthy.len()
            );
            for (shape, got) in &untrustworthy {
                println!("    {shape:<22} serial={got:.3}x");
            }
            println!(
                "  A `pipe:*` number on those rows is only meaningful if it falls well \
                 outside\n  this floor. Re-run on an idle machine for a clean read."
            );
        }
    }
    println!(
        "\nThe `pipe:*` arms here carry NO async copy — HIP gets the rotation only. That is\n\
         the whole point of this arm: it isolates the half of the CUDA treatment that is\n\
         portable. Compare against the CUDA sweep to see what cp.async is worth."
    );
}
