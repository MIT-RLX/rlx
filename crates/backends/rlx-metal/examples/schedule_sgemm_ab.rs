// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Does generating `sgemm_tiled` from its schedule beat the hand-written MSL?**
//!
//! The Apple-GPU half of the experiment in
//! `rlx-cuda/examples/schedule_matmul_ab.rs`. Same question, different machine:
//! `rlx_metal::kernel_schedule_emit` builds a tiled sgemm from a
//! `rlx_ir::kernel_schedule::KernelSchedule`, and the treatment is a
//! `stages`-deep rotation of the threadgroup tiles that reduces the K loop from
//! **two** `threadgroup_barrier` calls to **one**.
//!
//! Metal has no `cp.async`, so the async-copy half of the CUDA experiment does
//! not transfer and the emitter refuses `Feature::AsyncCopy` outright. What
//! does transfer is the structural claim, and it transfers *better*: the CUDA
//! pipeline needs 16-byte-aligned bulk copies and so only runs on full blocks,
//! whereas this keeps `sgemm_tiled`'s bounds-checked loads and therefore runs
//! at every shape.
//!
//! # Three arms
//!
//! | arm | source | stages | barriers / K iter |
//! |---|---|---|---|
//! | `shipping` | `sgemm_tiled` from `kernels.rs`, the real library | 1 | 2 |
//! | `serial` | **emitted** from the schedule describing it | 1 | 2 |
//! | `pipe:N` | **emitted** from the pipelined schedule | N | 1 |
//!
//! `serial` is what makes a `pipe:N` delta attributable. If the emitter itself
//! costs or gains anything, it shows up there and not in the pipeline column.
//!
//! # What this is NOT a claim about
//!
//! `sgemm_tiled` is **not** Metal's default GEMM. `rlx_metal::cost::pick_sgemm`
//! prefers MPS and the `simdgroup_matrix` variants and only falls back to the
//! scalar tiled kernel when those are ineligible. A win here is a win on the
//! scalar-tiled fallback path — it says nothing about rlx's fastest Metal GEMM,
//! and nothing about MPS.
//!
//! # Protocol
//!
//! * **Bit-exactness gates every arm** against the shipping kernel's own
//!   output. All three accumulate in the same `k` order over the same tile
//!   contents, so a differing bit is a bug in the schedule, not a faster
//!   kernel.
//! * **GPU time, not wall time.** Each arm's cost is taken from the command
//!   buffer's own `GPUStartTime`/`GPUEndTime`, so host-side encode and `objc`
//!   bridging are excluded. Median of 30 after 5 warmups.
//! * **Every arm is warmed before any is timed, and samples are taken
//!   round-robin.** Timing arms back-to-back in order made the first one absorb
//!   the fresh buffers' first-touch cost — which the `serial` arm exposed by
//!   reporting 1.24x against a kernel it is byte-for-byte equivalent to. Round
//!   robin also spreads thermal drift and outside GPU load across all arms
//!   rather than concentrating it on one.
//! * **No L2 flush.** The CUDA sweep flushes L2 before each sample; Apple's
//!   unified memory has no equivalent host-side lever, and pretending otherwise
//!   would be protocol theatre. Absolute numbers here are therefore warm-cache
//!   and not comparable to the CUDA example's — the *ratios* are what this
//!   measures, and every arm gets identical treatment.
//! * **Thermals.** `scripts/check-throttle.sh` is the repo's Apple gate; this
//!   example reports the run's spread instead of refusing, and a wide spread is
//!   a reason to distrust the row.
//!
//! ```sh
//! cargo run --release -p rlx-metal --features schedule-codegen \
//!     --example schedule_sgemm_ab
//! cargo run --release -p rlx-metal --features schedule-codegen \
//!     --example schedule_sgemm_ab -- --stages 2,3,4
//! ```

#[cfg(not(all(target_os = "macos", feature = "schedule-codegen")))]
fn main() {
    eprintln!(
        "this example needs macOS and `--features schedule-codegen`:\n  \
         cargo run --release -p rlx-metal --features schedule-codegen \
         --example schedule_sgemm_ab"
    );
}

#[cfg(all(target_os = "macos", feature = "schedule-codegen"))]
fn main() {
    use rlx_metal::apple_params::AppleKernelParams;
    use rlx_metal::kernel_schedule_emit::{
        EMITTED_ENTRY, EmitFacts, SHIPPING_TILE, emit_for, emit_msl,
        sgemm_tiled_pipelined_schedule, sgemm_tiled_schedule, shipping_msl,
    };
    use rlx_metal::kernel_schedule_port::METAL_TARGET;
    use rlx_metal::mtl::{ComputePipelineState, MTLSize};
    use rlx_metal::occupancy::AppleGpuFamily;

    const WARMUP: usize = 5;
    const ITERS: usize = 30;

    /// Shapes spanning decode through prefill. Unlike the CUDA sweep, every one
    /// of these reaches the treatment — the staging is bounds-checked, so there
    /// is no alignment gate to fall out of.
    const SHAPES: &[(&str, usize, usize, usize)] = &[
        ("decode, LM hidden", 1, 4096, 4096),
        ("decode, mlp up", 1, 2048, 8192),
        ("tiny batch", 4, 4096, 4096),
        ("small batch", 32, 4096, 4096),
        ("medium batch", 128, 4096, 4096),
        ("short prefill", 256, 2048, 2048),
        ("medium prefill", 512, 2048, 2048),
        ("prefill, wide hidden", 512, 4096, 4096),
        ("large prefill", 1024, 2048, 2048),
        ("large square", 2048, 2048, 2048),
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

    let Some(dev) = rlx_metal::device::metal_device() else {
        eprintln!("no Metal device");
        std::process::exit(1);
    };

    let args: Vec<String> = std::env::args().collect();
    let stages: Vec<usize> = args
        .iter()
        .position(|a| a == "--stages")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 3]);

    // ── Build every pipeline up front ───────────────────────────────────
    //
    // Compiling inside the timing loop would fold Metal's shader compile into
    // the first sample; doing it here means every arm is timed the same way.
    // (label, pipeline, facts, tile edge). The tile edge is carried per arm
    // because the threadgroup size and grid MUST match the kernel that was
    // emitted: dispatching a tile=32 kernel at 16x16 gives it half the threads
    // it needs, so it accumulates half of K and returns ~half the answer. The
    // bit-exactness gate caught that; a tolerance-based check would not have.
    let mut pipelines: Vec<(String, ComputePipelineState, Option<EmitFacts>, usize)> = Vec::new();
    let mut params_arms: Vec<(String, AppleKernelParams)> = Vec::new();
    let chip = AppleGpuFamily::from_name(dev.device.name());

    let ship_lib = rlx_metal::pipeline_cache::load_or_compile_library(&dev.device, &shipping_msl());
    let ship_fn = ship_lib
        .get_function("sgemm_tiled", None)
        .expect("sgemm_tiled not found in the shipping library");
    pipelines.push((
        "shipping".into(),
        dev.device
            .new_compute_pipeline_state_with_function(&ship_fn)
            .expect("shipping pipeline"),
        None,
        SHIPPING_TILE,
    ));

    let mut add_emitted = |label: String, sched: &rlx_ir::kernel_schedule::KernelSchedule| {
        match emit_msl(sched, SHIPPING_TILE, METAL_TARGET) {
            Ok((src, facts)) => {
                let lib = rlx_metal::pipeline_cache::load_or_compile_library(&dev.device, &src);
                let f = lib
                    .get_function(EMITTED_ENTRY, None)
                    .unwrap_or_else(|e| panic!("{EMITTED_ENTRY} missing from `{label}`: {e}"));
                let p = dev
                    .device
                    .new_compute_pipeline_state_with_function(&f)
                    .unwrap_or_else(|e| panic!("pipeline for `{label}`: {e}"));
                pipelines.push((label, p, Some(facts), SHIPPING_TILE));
            }
            // Refused, never silently replaced by the baseline: an arm that
            // quietly ran the shipping kernel would report 1.00x and read as
            // "no difference" rather than "did not run".
            Err(e) => eprintln!("arm `{label}` could not be emitted: {e} — EXCLUDED"),
        }
    };
    add_emitted("serial".into(), &sgemm_tiled_schedule(SHIPPING_TILE));
    for s in &stages {
        add_emitted(
            format!("pipe:{s}"),
            &sgemm_tiled_pipelined_schedule(SHIPPING_TILE, *s),
        );
    }
    // NLL ends `add_emitted`'s borrow of `pipelines` at its last use above, so
    // the next closure can take it. No explicit drop needed.

    // Parameter-driven arms. `--params "stages=2,precision=f16" --params "tile=8"`
    // adds one arm per spec, using the same parser the config and builder use,
    // so an arm in this table is a configuration someone can paste back.
    let mut add_params = |spec: &str| {
        let p = match AppleKernelParams::parse(spec) {
            Ok(p) => p,
            Err(why) => {
                eprintln!("arm `{spec}`: {why} — EXCLUDED");
                return;
            }
        };
        match emit_for(&p, METAL_TARGET) {
            Ok((src, facts)) => {
                let lib = rlx_metal::pipeline_cache::load_or_compile_library(&dev.device, &src);
                let label = p.spec();
                let f = lib
                    .get_function(EMITTED_ENTRY, None)
                    .unwrap_or_else(|e| panic!("{EMITTED_ENTRY} missing from `{label}`: {e}"));
                let st = dev
                    .device
                    .new_compute_pipeline_state_with_function(&f)
                    .unwrap_or_else(|e| panic!("pipeline for `{label}`: {e}"));
                // Print what the cost model expected, next to what will be
                // measured. A model nobody scores is a model nobody trusts.
                println!(
                    "  arm {label}\n      tile_edge={} tg_bytes={} predicted={}",
                    p.tile_value(),
                    p.threadgroup_bytes(),
                    rlx_metal::occupancy::explain(
                        chip,
                        AppleKernelParams::default().threadgroup_bytes(),
                        p.threadgroup_bytes()
                    )
                );
                params_arms.push((label.clone(), p));
                pipelines.push((label, st, Some(facts), p.tile_value()));
            }
            Err(e) => eprintln!("arm `{spec}` could not be emitted: {e} — EXCLUDED"),
        }
    };

    // Extra arms from `--params <spec>`, repeatable.
    let specs: Vec<String> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| a.as_str() == "--params")
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect();
    for spec in &specs {
        add_params(spec);
    }

    println!("schedule_sgemm_ab — is the emitted MSL faster than kernels.rs?\n");
    println!("  device     : {}", dev.device.name());
    println!("  workload   : f32 GEMM via the scalar tiled sgemm (NOT MPS, NOT simdgroup)");
    println!("  oracle     : the shipping `sgemm_tiled` output, same shape, same k order");
    println!("  tolerance  : bit-exact; any differing element disqualifies the arm");
    println!("  timing     : command-buffer GPU span, median of {ITERS}, {WARMUP} warmup");
    println!("  references : rlx's own shipping kernel only — not MPS, not peak\n");
    for (label, p, facts, _) in &pipelines {
        match facts {
            Some(f) => println!(
                "  arm {label:<9}: stages={} barriers/iter={} tg_bytes={} threads={} \
                 max_tg={}",
                f.stages,
                f.barriers_per_k_iter,
                f.threadgroup_bytes,
                f.threads,
                p.max_total_threads_per_threadgroup()
            ),
            None => println!(
                "  arm {label:<9}: hand-written MSL (baseline), max_tg={}",
                p.max_total_threads_per_threadgroup()
            ),
        }
    }
    println!();

    // ── Measure ─────────────────────────────────────────────────────────
    struct Row {
        shape: &'static str,
        base_ms: f64,
        base_spread: f64,
        arms: Vec<(String, f64, bool)>,
    }
    let mut rows: Vec<Row> = Vec::new();

    for (shape, m, k, n) in SHAPES {
        let (m, k, n) = (*m, *k, *n);
        let a = fill(m * k, 0x5eed_1234);
        let b = fill(k * n, 0xbeef_9876);

        let a_buf = dev.alloc_shared(m * k * 4);
        let b_buf = dev.alloc_shared(k * n * 4);
        let c_buf = dev.alloc_shared(m * n * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(a.as_ptr(), a_buf.contents() as *mut f32, m * k);
            std::ptr::copy_nonoverlapping(b.as_ptr(), b_buf.contents() as *mut f32, k * n);
        }
        let (mu, ku, nu) = (m as u32, k as u32, n as u32);

        // One dispatch; returns the command buffer's own GPU span in ms.
        let run = |pipe: &ComputePipelineState, tile: usize| -> f64 {
            let cb = dev.queue.new_command_buffer();
            let enc = cb.compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&a_buf), 0);
            enc.set_buffer(1, Some(&b_buf), 0);
            enc.set_buffer(2, Some(&c_buf), 0);
            enc.set_bytes(3, 4, (&mu as *const u32).cast());
            enc.set_bytes(4, 4, (&ku as *const u32).cast());
            enc.set_bytes(5, 4, (&nu as *const u32).cast());
            let t = tile as u64;
            enc.dispatch_thread_groups(
                MTLSize {
                    width: (n as u64).div_ceil(t),
                    height: (m as u64).div_ceil(t),
                    depth: 1,
                },
                MTLSize {
                    width: t,
                    height: t,
                    depth: 1,
                },
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            (cb.gpu_end_time() - cb.gpu_start_time()) * 1e3
        };

        let read_c = || -> Vec<f32> {
            let mut out = vec![0f32; m * n];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    c_buf.contents() as *const f32,
                    out.as_mut_ptr(),
                    m * n,
                );
            }
            out
        };

        // ── Phase 1: correctness, one run per arm ───────────────────────
        let mut reference: Vec<f32> = Vec::new();
        let mut exact_by_arm: Vec<bool> = Vec::new();
        for (i, (label, pipe, _, tile)) in pipelines.iter().enumerate() {
            run(pipe, *tile);
            let out = read_c();
            if i == 0 {
                reference = out;
            } else {
                let exact = out.len() == reference.len()
                    && out
                        .iter()
                        .zip(&reference)
                        .all(|(x, y)| x.to_bits() == y.to_bits());
                if !exact {
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
                exact_by_arm.push(exact);
            }
        }

        // ── Phase 2: warm EVERY arm before any of them is timed ──────────
        //
        // Running arms back-to-back in declaration order made the first one pay
        // the fresh buffers' first-touch and residency cost, which showed up as
        // the baseline being slow and every later arm "winning". It was visible
        // because the `serial` arm — the same kernel as the baseline — came out
        // 1.24x, and only on the shapes with the widest spread.
        for (_, pipe, _, tile) in &pipelines {
            for _ in 0..WARMUP {
                run(pipe, *tile);
            }
        }

        // ── Phase 3: timing, ROUND-ROBIN across arms ─────────────────────
        //
        // One sample of each arm per round, so thermal drift and interference
        // from whatever else is on the GPU land on all arms alike instead of
        // on whichever one happened to run during a bad stretch.
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(ITERS); pipelines.len()];
        for _ in 0..ITERS {
            for (i, (_, pipe, _, tile)) in pipelines.iter().enumerate() {
                samples[i].push(run(pipe, *tile));
            }
        }
        let stat = |v: &mut Vec<f64>| -> (f64, f64) {
            v.sort_by(|x, y| x.partial_cmp(y).expect("finite timings"));
            (v[v.len() / 2], v[v.len() - 1] / v[0].max(f64::MIN_POSITIVE))
        };
        let (base_ms, base_spread) = stat(&mut samples[0]);
        let mut arms = Vec::new();
        for i in 1..pipelines.len() {
            let (med, _) = stat(&mut samples[i]);
            arms.push((pipelines[i].0.clone(), med, exact_by_arm[i - 1]));
        }
        rows.push(Row {
            shape,
            base_ms,
            base_spread,
            arms,
        });
    }

    // ── Report ──────────────────────────────────────────────────────────
    let headers: Vec<String> = rows
        .first()
        .map(|r| r.arms.iter().map(|(l, _, _)| l.clone()).collect())
        .unwrap_or_default();
    print!("{:<22} {:>11} {:>7}", "shape", "shipping ms", "spread");
    for h in &headers {
        print!(" {h:>12}");
    }
    println!();
    println!("{}", "-".repeat(42 + 13 * headers.len()));

    let mut logsum = vec![0.0f64; headers.len()];
    let mut counted = vec![0usize; headers.len()];
    for r in &rows {
        print!(
            "{:<22} {:>11.4} {:>6.2}x",
            r.shape, r.base_ms, r.base_spread
        );
        for (i, (_, ms, exact)) in r.arms.iter().enumerate() {
            if *exact {
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
    println!("{}", "-".repeat(42 + 13 * headers.len()));
    print!("{:<22} {:>11} {:>7}", "geomean (bit-exact)", "1.000", "");
    for i in 0..headers.len() {
        if counted[i] == 0 {
            print!(" {:>12}", "n/a");
        } else {
            print!(" {:>11.3}x", (logsum[i] / counted[i] as f64).exp());
        }
    }
    println!();

    for (i, h) in headers.iter().enumerate() {
        if counted[i] < rows.len() {
            println!(
                "\nNOTE: arm `{h}` contributed {}/{} shapes; the rest were not bit-exact.",
                counted[i],
                rows.len()
            );
        }
    }
    let worst = rows.iter().map(|r| r.base_spread).fold(0.0f64, f64::max);
    if worst > 1.5 {
        println!(
            "\nWARNING: worst within-arm spread was {worst:.2}x (max/min of {ITERS} samples). \
             Rows\nthat noisy are not a measurement — check thermals \
             (`scripts/check-throttle.sh`) and rerun."
        );
    }
    println!(
        "\nReference is rlx's own shipping `sgemm_tiled`, which is the scalar FALLBACK \
         path on\nMetal — `pick_sgemm` prefers MPS and the simdgroup variants. A number \
         above 1.00x\nmeans the emitted schedule beat that kernel, not that it is near \
         Metal's best."
    );
}
