// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fill the GPU dispatch table by **measurement**, then persist it.
//!
//! The table in `rlx_gpu_kernels::dispatch` has compile-time defaults that
//! reproduce the historical hand-written routing exactly. This is the other
//! half: for each shape bucket, compile every legal physical schedule, check it
//! against the default's output, time it, and record the winner. What lands in
//! the cache is a measured fact about *this* GPU, not a constant someone picked
//! once on different silicon.
//!
//! Three properties matter more than the speedup:
//!
//! * **Generalization gates the win.** An override is stored per *bucket*, and a
//!   bucket holds many shapes — so measuring one shape does not entitle you to
//!   claim the bucket. Each winner is re-checked on held-out shapes from the same
//!   bucket that the search never saw, and discarded if it regresses there. This
//!   is CAKE §6's rule: the shape domain is declared before tuning, and coverage
//!   is claimed only over shards the tuner did not fit on.
//! * **Correctness gates the win.** A candidate is only eligible if its output
//!   matches the default schedule's bit-for-bit. Both compute the same f32
//!   reduction over the same K in the same order, so a tile change must not move
//!   a single bit; anything that does is a bug in the tile, not a faster kernel,
//!   and "fast but wrong" must never win a tuning run.
//! * **Nothing is silently dropped.** Candidates that fail to compile, fail
//!   correctness, or lose are all reported. A tuner that prints only its winners
//!   reads as "covered everything" when it did not.
//!
//! Timing protocol: L2 flushed before every sample, each sample timed
//! individually, **median** of 30 reported. See `flush_l2`.
//!
//! ```sh
//! cargo run --release -p rlx-cuda --example tune_dispatch
//! cargo run --release -p rlx-cuda --example tune_dispatch -- --dry-run
//! RLX_GPU_TUNING_CACHE=/tmp/t.tsv cargo run --release -p rlx-cuda --example tune_dispatch
//! ```

use std::time::Instant;

use rlx_cuda::backend::CudaExecutable;
use rlx_gpu_dispatch::cost::TileCostModel;
use rlx_gpu_dispatch::dispatch::{DecisionOutcome, DecisionRecord};
use rlx_gpu_kernels::dispatch::{self, Choice, GpuArch, OverrideRecord, Workload};
use rlx_gpu_kernels::tiles::{MATMUL_TILE_CANDIDATES, TileParams};
use rlx_ir::{DType, Graph, Shape};

const WARMUP: usize = 5;
const ITERS: usize = 30;

/// Bytes written to a scratch buffer between timed samples to evict the working
/// set from L2. Sized well past any current NVIDIA L2 (Ampere laptop ~6 MB,
/// H100 50 MB, B200 larger) so one pass is enough.
const L2_FLUSH_BYTES: usize = 128 << 20;

/// A candidate must beat the default by this much to displace it. Wall-clock
/// timing on a shared GPU is noisy at the percent level, and a tuner that
/// installs a 1.00× "win" is fitting noise into a persisted cache — churn that
/// looks like tuning. Anything under this margin leaves the default in place.
const MIN_SPEEDUP: f64 = 1.02;

/// Above this GPU utilization, something else is already using the device and
/// every timing below is contention, not measurement.
const MAX_BUSY_PERCENT: f32 = 15.0;

/// A winner must also not be SLOWER than the default on any held-out shape by
/// more than this. Held-out shapes are different shapes, so exact parity is not
/// the bar; a real regression is.
const MAX_HOLDOUT_REGRESSION: f64 = 0.98;

/// How many candidates the analytical cost model is allowed to narrow each
/// bucket to, read from `RLX_TUNE_PREFILTER`.
///
/// **Unset means no prefilter: measure every candidate.** That is the right
/// default while the candidate list is small enough to measure exhaustively —
/// the tuner's output is the ground truth the model is calibrated against, and
/// a tuner that quietly stops measuring is a tuner whose data stops being able
/// to contradict the model.
///
/// Set it when the candidate cross product outgrows the GPU time available. The
/// skipped tiles are always named in the run's disclosure line.
fn prefilter_keep() -> Option<usize> {
    rlx_ir::env::var("RLX_TUNE_PREFILTER")?
        .parse::<usize>()
        .ok()
        .filter(|n| *n > 0)
}

/// One bucket's declared shape domain: the shape tuning measures against, and
/// unseen shapes from the same bucket used only to check that the winner
/// generalizes.
///
/// **The `holdout` field is the point.** A bucket covers a `floor(log2)` band per
/// axis, so it contains many shapes, and measuring one of them does not entitle
/// you to claim the bucket. CAKE §6: *"The valid shape domain is declared before
/// tuning. Dispatcher predicates may partition that domain, but they may not
/// introduce convenient new evaluation rows... This separation prevents the
/// dispatcher from being tuned on the set used to claim generalization."* Without
/// a held-out check this tuner was doing exactly that — 5 measurements, 5 buckets
/// claimed, zero evidence about the rest of each bucket.
struct Domain {
    label: &'static str,
    tune: (usize, usize, usize),
    holdout: &'static [(usize, usize, usize)],
}

/// The **workload contract** for this tuner, stated in one place.
///
/// CAKE §4: *"The workload contract fixes the shapes, oracle, tolerances,
/// hardware, and permitted references."* Those five were previously scattered —
/// shapes in `DOMAIN`, the oracle implicit in `measure`'s first call, the
/// tolerance implicit in an `out != reference` comparison, the hardware implicit
/// in whatever rig you happened to run on. Naming them makes the run
/// reproducible and makes a change to any of them visible in a diff.
pub struct Contract {
    /// What is being tuned.
    pub workload: &'static str,
    /// Ground truth every candidate is compared against.
    pub oracle: &'static str,
    /// Accepted deviation from the oracle.
    pub tolerance: &'static str,
    /// Required device state for a measurement to count.
    pub hardware: &'static str,
    /// What a candidate is allowed to be compared to.
    pub references: &'static str,
    /// Which candidates were actually measured, and by whose authority any were
    /// not. A contract that names the oracle but not the search space lets a
    /// narrowed sweep read exactly like an exhaustive one.
    pub search: &'static str,
}

/// The contract this sweep runs under.
pub const CONTRACT: Contract = Contract {
    workload: "dense f32 GEMM via rlx's own tiled `matmul` kernel (RLX_CUDA_NO_CUBLAS=1)",
    oracle: "the DEFAULT tile's output on the same device — same math, same K order",
    tolerance: "bit-exact; any differing element disqualifies the candidate",
    hardware: "GPU idle (<=15% util over 5 NVML samples) and not at its power cap",
    references: "the compile-time default tile only; no external library baseline \
                 (see the `reference_perf` example for cuBLAS-anchored ratios)",
    search: "every tile in MATMUL_TILE_CANDIDATES, unless RLX_TUNE_PREFILTER=N \
             narrows it — in which case each bucket prints what was skipped",
};

/// The declared shape domain, one entry per bucket.
///
/// Held-out shapes must land in the SAME bucket as their `tune` shape — a check
/// asserts it at run time, because a holdout in a different bucket validates
/// nothing about the override being installed.
const DOMAIN: &[Domain] = &[
    // ── decode (m = 1) ──────────────────────────────────────────────────
    Domain {
        label: "decode, small hidden",
        tune: (1, 1024, 1024),
        holdout: &[(1, 1536, 1200), (1, 1024, 2047)],
    },
    Domain {
        label: "decode, LM hidden",
        tune: (1, 4096, 4096),
        holdout: &[(1, 5120, 6144), (1, 4096, 7000)],
    },
    Domain {
        label: "decode, narrow k",
        tune: (1, 512, 4096),
        holdout: &[(1, 600, 5000), (1, 512, 7000)],
    },
    Domain {
        label: "decode, wide n (vocab)",
        tune: (1, 4096, 32768),
        holdout: &[(1, 5000, 40000), (1, 4096, 50000)],
    },
    Domain {
        label: "decode, mlp up",
        tune: (1, 2048, 8192),
        holdout: &[(1, 3000, 9000), (1, 2048, 15000)],
    },
    // ── small batch / speculative decode ────────────────────────────────
    Domain {
        label: "small batch",
        tune: (32, 4096, 4096),
        holdout: &[(48, 5000, 4096), (63, 4096, 6000)],
    },
    Domain {
        label: "tiny batch",
        tune: (4, 4096, 4096),
        holdout: &[(6, 5000, 4096), (7, 4096, 6000)],
    },
    Domain {
        label: "medium batch",
        tune: (128, 4096, 4096),
        holdout: &[(200, 5000, 4096), (250, 4096, 6000)],
    },
    // ── prefill ─────────────────────────────────────────────────────────
    Domain {
        label: "short prefill",
        tune: (256, 2048, 2048),
        holdout: &[(300, 3000, 2048), (400, 2048, 3500)],
    },
    Domain {
        label: "medium prefill",
        tune: (512, 2048, 2048),
        holdout: &[(768, 3000, 2048), (600, 2048, 3500)],
    },
    Domain {
        label: "large prefill",
        tune: (2048, 2048, 2048),
        holdout: &[(3000, 2048, 2500), (2048, 3900, 2048)],
    },
    Domain {
        label: "prefill, wide hidden",
        tune: (512, 4096, 4096),
        holdout: &[(700, 5000, 4096), (600, 4096, 6000)],
    },
    // ── shapes with a distinctive aspect ratio ──────────────────────────
    Domain {
        label: "tall, short k (dW)",
        tune: (4096, 512, 512),
        holdout: &[(5000, 600, 512), (6000, 512, 700)],
    },
    Domain {
        label: "fat k, small mn",
        tune: (192, 8192, 512),
        holdout: &[(200, 9000, 600), (250, 12000, 512)],
    },
    Domain {
        label: "large square",
        tune: (4096, 4096, 4096),
        holdout: &[(5000, 5000, 4096), (4096, 6000, 5000)],
    },
];

struct Timing {
    tile: TileParams,
    ms: f64,
}

fn build(m: usize, k: usize, n: usize) -> Graph {
    let mut g = Graph::new("tune_mm");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[k, n], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);
    g
}

/// Refuse to tune on a GPU that is already busy or clock-limited.
///
/// This is not paranoia — it is the failure this tuner actually hit. A run on the
/// shared CUDA rig produced absolute times ~25× slower than a clean run and
/// picked a *different* winner, because an unrelated Python process was holding
/// the GPU at 100% utilization with SM clocks capped at 892 MHz. Nothing in the
/// output said so; the numbers simply looked like measurements and would have
/// been written into a persisted cache that every later process reads.
///
/// The repo's `scripts/check-throttle.sh` guards exactly this for benchmarks, but
/// it is Apple-Silicon-only (`pmset -g therm`) and has no NVIDIA path. Reading
/// NVML in-process is better here anyway: the tuner is the thing whose output
/// gets *persisted*, so it should be the thing that refuses.
///
/// Honours `RLX_ALLOW_THROTTLE=1`, the same bypass the shell gate uses.
fn contention_check() -> Result<String, String> {
    if rlx_ir::env::flag("RLX_ALLOW_THROTTLE") {
        return Ok("RLX_ALLOW_THROTTLE=1 — contention gate bypassed".to_string());
    }
    // Sample repeatedly and keep the MINIMUM utilization. NVML reports a recent
    // window, so one instantaneous read catches transients — an idle GPU already
    // produced a spurious 17% here and the gate refused a perfectly good run.
    // A genuinely busy GPU stays busy, so the floor across several samples is the
    // honest statistic; a single reading is not.
    const SAMPLES: usize = 5;
    let mut min_util: Option<f32> = None;
    let mut last: Option<rlx_cuda::nvml::NvmlSample> = None;
    for i in 0..SAMPLES {
        let Some(s) = rlx_cuda::nvml::sample(0) else {
            // No NVML is a coverage limitation, not a pass. Say which.
            return Ok("NVML unavailable — cannot verify the GPU is idle".to_string());
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
        return Ok("NVML unavailable — cannot verify the GPU is idle".to_string());
    };
    let mut problems = Vec::new();
    if let Some(util) = min_util
        && util > MAX_BUSY_PERCENT
    {
        problems.push(format!(
            "{util:.0}% utilization sustained over {SAMPLES} samples \
             (another process is running)"
        ));
    }
    if let (Some(power), Some(cap)) = (s.power_w, s.power_cap_w)
        && power >= cap * 0.98
    {
        problems.push(format!("at the {cap:.0} W power cap ({power:.0} W)"));
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

/// Run one shape under one pinned tile: returns (output, median-ish mean ms).
fn measure(
    arch: &GpuArch,
    m: usize,
    k: usize,
    n: usize,
    tile: TileParams,
    xv: &[f32],
    wv: &[f32],
) -> Option<(Vec<f32>, f64)> {
    let workload = Workload::Matmul { m, k, n };
    dispatch::set_override(workload.key(arch), Choice::MatmulTiled(tile)).ok()?;

    let mut exe = CudaExecutable::compile(build(m, k, n));
    exe.set_param("w", wv);
    for _ in 0..WARMUP {
        let _ = exe.run(&[("x", xv)]);
    }

    // Time each sample SEPARATELY, flushing L2 in between, and report the
    // MEDIAN.
    //
    // The previous form — one timer around 30 back-to-back iterations, divided
    // by 30 — measured a warm-L2 best case. That matters most for exactly the
    // shapes that win here: m=1, k=n=1024 has a ~4 MB working set, which fits
    // inside this GPU's ~6 MB L2, so iterations 2..30 re-read cache rather than
    // memory. The ranking survived (every candidate got the same treatment) but
    // the absolute numbers were optimistic and candidates with different
    // footprints could reorder. CAKE's protocol is explicit about this: "CUPTI
    // timing ... with the L2 cache flushed before each timed sample", and it
    // reports a median rather than a mean, which is also far less sensitive to a
    // single scheduling hiccup.
    let mut samples: Vec<f64> = Vec::with_capacity(ITERS);
    let mut out = Vec::new();
    for _ in 0..ITERS {
        flush_l2();
        let t0 = Instant::now();
        let r = exe.run(&[("x", xv)]);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        out = r[0].clone();
        samples.push(dt);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).expect("timings are finite"));
    Some((out, samples[samples.len() / 2]))
}

/// Evict the previous sample's working set from L2 by streaming a large buffer
/// through it.
///
/// Deliberately a device-side memset rather than a host round-trip: a host sync
/// would add its own latency to the *next* sample's timer. The buffer is
/// allocated once and reused, so the flush costs a fixed bandwidth-bound write
/// and never re-enters the allocator mid-sweep.
fn flush_l2() {
    use std::sync::{Mutex, OnceLock};
    static SCRATCH: OnceLock<Mutex<Option<cudarc::driver::CudaSlice<u8>>>> = OnceLock::new();
    let Some(ctx) = rlx_cuda::device::cuda_context() else {
        return;
    };
    let cell = SCRATCH
        .get_or_init(|| Mutex::new(ctx.default_stream().alloc_zeros::<u8>(L2_FLUSH_BYTES).ok()));
    let mut guard = cell.lock().expect("l2 scratch poisoned");
    if let Some(buf) = guard.as_mut() {
        // memset the whole buffer: a write stream large enough to displace any
        // resident lines. Errors are ignored — a failed flush degrades the
        // measurement, it does not invalidate the run, and reporting it per
        // sample would drown the sweep.
        let _ = ctx.default_stream().memset_zeros(buf);
        let _ = ctx.default_stream().synchronize();
    }
}

fn main() {
    if !rlx_cuda::is_available() {
        println!("CUDA not available on this host — nothing to tune.");
        return;
    }
    let dry_run = std::env::args().any(|a| a == "--dry-run");

    // The tiled kernel is behind cuBLAS in the dispatch order, so without this
    // the sweep would time cuBLAS five times and "discover" that every tile is
    // identical. Set before the first compile so the config snapshot sees it.
    // SAFETY: set once, before any GPU work, single-threaded.
    unsafe { std::env::set_var("RLX_CUDA_NO_CUBLAS", "1") };

    // Any context works to name the arch; compile a trivial graph to get one.
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
    // Force the persisted cache to load NOW, then clear it.
    //
    // `ensure_tuning_cache_loaded` is a `OnceLock` that fires lazily from
    // `backend::gpu_arch()`, i.e. on the first matmul dispatch — which is INSIDE
    // the first `measure()` call, *after* the sweep's `clear_overrides()`. So on
    // a non-empty cache file the previous run's overrides were re-installed
    // mid-measurement and the "default tile" baseline silently ran a cached tile
    // instead. That is why a re-run against an existing cache reported 1 improved
    // bucket where a fresh cache reported 3: the tuner was measuring against
    // itself. Loading eagerly makes the subsequent clear actually stick.
    rlx_cuda::tuning::ensure_tuning_cache_loaded();
    dispatch::clear_overrides();

    println!("rlx-cuda dispatch tuner — arch {}", arch.as_str());
    println!("  workload:   {}", CONTRACT.workload);
    println!("  oracle:     {}", CONTRACT.oracle);
    println!("  tolerance:  {}", CONTRACT.tolerance);
    println!("  hardware:   {}", CONTRACT.hardware);
    println!("  references: {}", CONTRACT.references);
    println!("  search:     {}", CONTRACT.search);
    match contention_check() {
        Ok(note) => println!("GPU state: {note}"),
        Err(why) => {
            eprintln!(
                "REFUSING to tune: {why}.\n\
                 Timings taken now would be contention, not measurement, and this \
                 tuner writes a persisted cache. Wait for the GPU to go idle, or set \
                 RLX_ALLOW_THROTTLE=1 to override (results will be unreliable)."
            );
            std::process::exit(1);
        }
    }
    println!(
        "{} tile candidates × {} buckets, each validated on {} held-out shape(s)\n",
        MATMUL_TILE_CANDIDATES.len(),
        DOMAIN.len(),
        DOMAIN.iter().map(|d| d.holdout.len()).sum::<usize>() / DOMAIN.len().max(1),
    );

    // Winners accumulate here rather than in the live table: the sweep clears
    // the table between shapes so each measurement starts from the default, and
    // installing as we go would wipe every earlier winner.
    let mut winners: Vec<(dispatch::DispatchKey, OverrideRecord)> = Vec::new();
    let mut skipped = 0usize;
    let mut rejected_by_holdout = 0usize;
    let mut decisions: Vec<DecisionRecord> = Vec::new();
    // Fresh evidence for (or against) the analytical cost model, from shapes it
    // was not fitted on.
    let mut model_agreements = 0usize;
    let mut model_scored = 0usize;
    let mut model_extrapolated = 0usize;
    let mut model_misses: Vec<String> = Vec::new();
    let stamp = rlx_cuda::tuning::now_unix_s();

    for dom in DOMAIN {
        let (m, k, n) = dom.tune;
        let workload = Workload::Matmul { m, k, n };
        let bucket = workload.bucket();
        let xv: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 1e-2 - 0.5).collect();
        let wv: Vec<f32> = (0..k * n).map(|i| ((i % 89) as f32) * 1e-2 - 0.5).collect();

        println!(
            "── {}: m={m} k={k} n={n}  (bucket {})",
            dom.label,
            bucket.encode()
        );

        // A holdout in a different bucket validates nothing about the override
        // being installed for THIS bucket. Fail loudly rather than report a
        // meaningless pass.
        for &(hm, hk, hn) in dom.holdout {
            let hb = Workload::Matmul {
                m: hm,
                k: hk,
                n: hn,
            }
            .bucket();
            assert_eq!(
                hb,
                bucket,
                "holdout ({hm},{hk},{hn}) is in bucket {} but tunes bucket {} — \
                 fix DOMAIN, it would validate the wrong key",
                hb.encode(),
                bucket.encode()
            );
        }

        // The default tile is both the baseline time and the correctness oracle.
        dispatch::clear_overrides();
        let Some((reference, base_ms)) =
            measure(&arch, m, k, n, TileParams::DEFAULT_MATMUL, &xv, &wv)
        else {
            println!("   default tile failed to run — skipping this shape");
            skipped += 1;
            continue;
        };
        println!(
            "   {:>22}  {:>8.3} ms   (baseline)",
            TileParams::DEFAULT_MATMUL.label(),
            base_ms
        );

        let mut best = Timing {
            tile: TileParams::DEFAULT_MATMUL,
            ms: base_ms,
        };

        // The analytical model runs on every bucket whether or not it is
        // allowed to skip anything. When it is not, its ranking is still
        // printed beside the measurements and scored at the end — so each
        // tuning run is also a fresh, un-fitted check on the model, and the
        // evidence for trusting it as a prefilter accumulates from data it
        // never saw.
        let model = TileCostModel::sm86();
        let workload = Workload::Matmul { m, k, n };
        let keep = prefilter_keep().unwrap_or(MATMUL_TILE_CANDIDATES.len());
        let pf = model.prefilter(&workload, MATMUL_TILE_CANDIDATES, keep);
        if prefilter_keep().is_some() {
            println!("   {}", pf.disclosure());
        }
        // Buckets outside the model's calibrated box are where its agreement
        // with measurement is worth the most: it is the only evidence that
        // would justify widening that box.
        let extrapolated = !model.provenance().covers(&workload);
        if extrapolated {
            println!("   (outside the cost model's calibrated shapes — its rank here is a guess)");
        }
        let model_order = pf.tiles();
        let model_rank = |t: &TileParams| model_order.iter().position(|c| c == t);
        let measured: Vec<TileParams> = pf.tiles();

        for &tile in &measured {
            if tile == TileParams::DEFAULT_MATMUL {
                continue;
            }
            let Some((out, ms)) = measure(&arch, m, k, n, tile, &xv, &wv) else {
                println!("   {:>22}  REJECTED (illegal tile)", tile.label());
                skipped += 1;
                continue;
            };
            // Same math, same K order — a tile change must not move a bit.
            if out != reference {
                let worst = out
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "   {:>22}  {:>8.3} ms   WRONG (max |Δ| {worst:.3e}) — not eligible",
                    tile.label(),
                    ms
                );
                skipped += 1;
                continue;
            }
            let mark = if ms < best.ms { " <-- best" } else { "" };
            let rank =
                model_rank(&tile).map_or_else(|| "  m?".to_string(), |r| format!("m#{}", r + 1));
            println!(
                "   {:>22}  {:>8.3} ms   {:>6.2}×  [{rank}]{mark}",
                tile.label(),
                ms,
                base_ms / ms
            );
            if ms < best.ms {
                best = Timing { tile, ms };
            }
        }

        // Score the model against this bucket's ground truth. `m#1` next to the
        // measured winner means the prefilter would have kept it at keep=1.
        let tag = if extrapolated { " [extrapolated]" } else { "" };
        match model_rank(&best.tile) {
            Some(0) => model_agreements += 1,
            Some(r) => {
                model_misses.push(format!(
                    "{}: winner {} was model rank #{}{tag}",
                    bucket.encode(),
                    best.tile.label(),
                    r + 1
                ));
            }
            None => model_misses.push(format!(
                "{}: winner {} was not scored by the model{tag}",
                bucket.encode(),
                best.tile.label()
            )),
        }
        model_scored += 1;
        if extrapolated {
            model_extrapolated += 1;
        }

        dispatch::clear_overrides();
        let speedup = base_ms / best.ms;
        let record = |outcome: DecisionOutcome| DecisionRecord {
            at_unix_s: stamp,
            arch: arch.clone(),
            op: rlx_gpu_dispatch::dispatch::GpuOp::Matmul,
            bucket,
            outcome,
            tuned_shape: format!("{m}x{k}x{n}"),
        };
        if best.tile == TileParams::DEFAULT_MATMUL {
            decisions.push(record(DecisionOutcome::DefaultHeld {
                best_speedup: speedup as f32,
            }));
            println!("   => default tile holds\n");
            continue;
        }
        if speedup < MIN_SPEEDUP {
            decisions.push(record(DecisionOutcome::DefaultHeld {
                best_speedup: speedup as f32,
            }));
            println!(
                "   => {} led at {speedup:.2}× but is under the {MIN_SPEEDUP:.2}× margin \
                 — default holds\n",
                best.tile.label()
            );
            continue;
        }

        // ── Generalization check on UNSEEN shapes in the same bucket ────────
        //
        // The override about to be installed applies to the whole bucket, not to
        // the shape that was measured. Confirm on shapes the search never saw.
        let mut regressions: Vec<String> = Vec::new();
        for &(hm, hk, hn) in dom.holdout {
            let hxv: Vec<f32> = (0..hm * hk)
                .map(|i| ((i % 97) as f32) * 1e-2 - 0.5)
                .collect();
            let hwv: Vec<f32> = (0..hk * hn)
                .map(|i| ((i % 89) as f32) * 1e-2 - 0.5)
                .collect();
            let Some((h_ref, h_base)) =
                measure(&arch, hm, hk, hn, TileParams::DEFAULT_MATMUL, &hxv, &hwv)
            else {
                regressions.push(format!("({hm},{hk},{hn}) default failed to run"));
                continue;
            };
            let Some((h_out, h_ms)) = measure(&arch, hm, hk, hn, best.tile, &hxv, &hwv) else {
                regressions.push(format!("({hm},{hk},{hn}) winner not launchable"));
                continue;
            };
            if h_out != h_ref {
                regressions.push(format!("({hm},{hk},{hn}) WRONG output"));
                continue;
            }
            let h_speedup = h_base / h_ms;
            let verdict = if h_speedup < MAX_HOLDOUT_REGRESSION {
                regressions.push(format!("({hm},{hk},{hn}) {h_speedup:.2}×"));
                "REGRESSION"
            } else {
                "ok"
            };
            println!(
                "     holdout m={hm} k={hk} n={hn}   {h_ms:>8.3} ms   {h_speedup:>6.2}×  {verdict}"
            );
        }

        dispatch::clear_overrides();
        if regressions.is_empty() {
            // Store the evidence, not just the verdict — a route report has to be
            // able to say what this override bought and how well it was checked.
            winners.push((
                workload.key(&arch),
                OverrideRecord {
                    choice: Choice::MatmulTiled(best.tile),
                    speedup: Some(speedup as f32),
                    holdouts: dom.holdout.len() as u16,
                },
            ));
            decisions.push(record(DecisionOutcome::Installed {
                choice: Choice::MatmulTiled(best.tile),
                speedup: speedup as f32,
                holdouts: dom.holdout.len() as u16,
            }));
            println!(
                "   => {} wins ({speedup:.2}× tuned, holds on {} holdout shape(s))\n",
                best.tile.label(),
                dom.holdout.len()
            );
        } else {
            rejected_by_holdout += 1;
            // A rejection is the MORE informative event: it says the bucket is
            // heterogeneous, which is a fact about the key rather than about the
            // candidate. Journal it.
            decisions.push(record(DecisionOutcome::RejectedByHoldout {
                choice: Choice::MatmulTiled(best.tile),
                speedup: speedup as f32,
                detail: regressions.join(", "),
            }));
            println!(
                "   => {} won the search at {speedup:.2}× but FAILED generalization \
                 [{}] — default holds\n",
                best.tile.label(),
                regressions.join(", ")
            );
        }
    }

    // Install every winner at once, now that no further measurement will clear
    // the table.
    dispatch::clear_overrides();
    for (key, record) in &winners {
        dispatch::set_override_measured(key.clone(), *record)
            .expect("winner was validated when it was measured");
    }

    println!(
        "{} bucket(s) improved, {rejected_by_holdout} winner(s) rejected by the \
         held-out check, {skipped} candidate(s) skipped or rejected",
        winners.len()
    );

    // What this run says about the analytical model. Reported whether or not
    // the prefilter was enabled: this is the evidence that decides whether
    // enabling it later would be safe, and it is worth more when the model had
    // no say in what got measured.
    if model_scored > 0 {
        println!(
            "\ncost model: picked the measured winner in {model_agreements}/{model_scored} \
             bucket(s){}",
            if prefilter_keep().is_some() {
                " (prefilter ACTIVE — the model chose what was measured, so this is not \
                 independent evidence)"
            } else {
                " (prefilter off — every candidate was measured, so this is independent)"
            }
        );
        for miss in &model_misses {
            // Named individually: a miss is the case where a prefilter at
            // keep=1 would have skipped the tile that actually won.
            println!("   miss: {miss}");
        }
        if model_extrapolated > 0 {
            println!(
                "   {model_extrapolated} of those bucket(s) sit outside the model's calibrated \n   \
                 shapes ({}) — agreement there is new evidence, disagreement is expected",
                match TileCostModel::sm86().provenance() {
                    rlx_gpu_dispatch::cost::CostProvenance::Measured { domain, .. } =>
                        domain.to_string(),
                    rlx_gpu_dispatch::cost::CostProvenance::Structural => "none".to_string(),
                }
            );
        }
    }

    // Per-route breakdown over the whole declared domain — tuned shapes AND
    // holdouts, so routes still on the default are listed rather than omitted.
    // An aggregate speedup would hide which routes actually moved.
    let all: Vec<Workload> = DOMAIN
        .iter()
        .flat_map(|d| {
            std::iter::once(d.tune)
                .chain(d.holdout.iter().copied())
                .map(|(m, k, n)| Workload::Matmul { m, k, n })
        })
        .collect();
    println!();
    print!(
        "{}",
        dispatch::render_routes(&arch, &dispatch::explain(&arch, &all))
    );
    if dry_run {
        println!("--dry-run: not writing the cache. Table would be:\n");
        print!("{}", dispatch::save_overrides());
        return;
    }
    match rlx_cuda::tuning::save_tuning_cache() {
        Some(p) => println!("wrote {}", p.display()),
        None => println!("tuning cache is disabled or unwritable — nothing persisted"),
    }
    match rlx_cuda::tuning::append_decisions(&decisions) {
        Some(p) => println!(
            "appended {} decision(s) to {}",
            decisions.len(),
            p.display()
        ),
        None => println!("decision journal disabled or unwritable"),
    }
}
