// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Reference-anchored performance** — how far is rlx's own MSL sgemm from
//! Apple's tuned one, and does the routing cascade actually pick the winner?
//!
//! Every speedup rlx measures on Metal today is against *rlx*. That says
//! nothing about whether the hand-written path is at 40% or 105% of what the
//! chip gives a good implementation. CAKE's Table 4 reports the other number —
//! relative performance against the best-known library per kernel — and treats
//! "below reference" as a signal to act on rather than a fact to omit.
//!
//! Metal makes this a sharper question than CUDA does, because
//! [`SgemmVariant`] already carries a *claim*: `Simd64` is documented as
//! "~1.8× Simd4x4 and beats MPS on TALL / short-K aligned shapes (measured)".
//! That is falsifiable, and unlike the CUDA case the reference here is not a
//! separate library call — it is a variant the cascade can already choose. So
//! this example answers two things:
//!
//! 1. **The ratio.** rlx's own kernels vs `MPSMatrixMultiplication`, same
//!    process, same device state, so the comparison cannot drift on thermals
//!    or clocks between runs.
//! 2. **The routing.** Whether [`pick_sgemm_default`] selects the faster of the
//!    two at each shape. A dispatcher that confidently picks the slower path is
//!    worse than no dispatcher, and only a measurement can tell you which it
//!    is doing.
//!
//! MPSGraph is disabled throughout. It is a *different* fast path (whole-graph
//! capture, not sgemm variant selection), it would silently replace the matmul
//! being measured, and its executable init is a known ~2% SIGSEGV risk that has
//! nothing to do with this comparison.
//!
//! ```sh
//! cargo run --release -p rlx-metal --example reference_perf
//! ```

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("Metal is macOS-only — nothing to anchor here.");
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}

#[cfg(target_os = "macos")]
mod macos {
    use std::time::Instant;

    use rlx_ir::{DType, Graph, Shape};
    use rlx_runtime::{Device, Session};

    /// Load average above which timing on this host is contention, not
    /// measurement.
    ///
    /// The CUDA twin refuses to run when NVML reports the GPU busy. macOS
    /// exposes no equivalent exclusive-access signal for the GPU, but the
    /// symptom is the same and load average catches it: this harness was
    /// developed at load 51 on an 8-core machine, where the SAME pinned variant
    /// timed 1.19 ms and 10.94 ms on consecutive runs. Every ratio taken there
    /// was noise wearing a table's clothes.
    const MAX_LOAD_AVG: f64 = 4.0;

    /// 1-minute load average, or `None` if it cannot be read.
    fn load_avg_1m() -> Option<f64> {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .ok()?;
        // "{ 51.19 45.93 42.83 }"
        String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<f64>().ok())
    }

    /// `Err` when the machine is too busy for the numbers to mean anything.
    fn contention_ok() -> Result<String, String> {
        if rlx_ir::env::flag("RLX_ALLOW_THROTTLE") {
            return Ok("RLX_ALLOW_THROTTLE=1 — gate bypassed".into());
        }
        match load_avg_1m() {
            Some(l) if l > MAX_LOAD_AVG => Err(format!(
                "1-minute load average is {l:.1} (limit {MAX_LOAD_AVG:.1})"
            )),
            Some(l) => Ok(format!("load average {l:.2}")),
            None => Ok("load average unreadable — proceeding unverified".into()),
        }
    }

    const WARMUP: usize = 10;
    const ITERS: usize = 50;

    /// Relative tolerance between the two paths.
    ///
    /// NOT bit-exact, deliberately: MPS and the hand-written kernels accumulate
    /// in a different order, so identical results would be the surprise. A
    /// ratio is only meaningful if both sides computed the same thing, which
    /// makes this a correctness gate on the comparison itself.
    const REL_TOL: f32 = 2e-3;

    /// p50/min above which a shape's numbers are contention rather than
    /// measurement, and the ratio should not be leaned on.
    const NOISY_SPREAD: f64 = 1.15;

    /// Shapes spanning the regimes the cascade routes differently: decode
    /// (m=1), the tall/short-K shapes `Simd64` claims to win, square prefill,
    /// and the fat-K/small-MN backward shape the occupancy gate excludes.
    const SHAPES: &[(usize, usize, usize, &str)] = &[
        (1, 2048, 2048, "decode"),
        (1, 4096, 4096, "decode, LM hidden"),
        (64, 1024, 1024, "tall, short K"),
        (256, 512, 1024, "tall, short K (Simd64 claim)"),
        (512, 512, 512, "square, small"),
        (1024, 1024, 1024, "square"),
        (2048, 2048, 2048, "square, large"),
        (192, 4096, 512, "fat K, small MN (dW shape)"),
        (4096, 512, 512, "very tall"),
    ];

    /// `tag` distinguishes the graph per routing configuration.
    ///
    /// Load-bearing, not cosmetic: compiling the *same* graph under different
    /// env settings returned one shared executable, so all three configurations
    /// measured whichever variant was selected first and every ratio came out
    /// 1.00x. `pick_sgemm` was answering correctly the whole time
    /// (SimdPadded / Mps / Simd64) — the compile never re-ran.
    fn build_tagged(tag: &str, m: usize, k: usize, n: usize) -> Graph {
        let mut g = Graph::new(format!("ref_mm_{tag}"));
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let w = g.param("w", Shape::new(&[k, n], DType::F32));
        let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);
        g
    }

    /// Compile one executable per routing configuration, then run them
    /// **interleaved**, and report each one's minimum.
    ///
    /// Both halves of that are corrections to a first version that measured
    /// each configuration in its own block and took the median. On this machine
    /// that produced a 35-50% spread between two runs of the *same*
    /// configuration — larger than every effect it was trying to detect — and
    /// it manufactured a confident "the cascade picks the slower path at 8 of 9
    /// shapes" result that was entirely measurement order.
    ///
    /// * **Interleaving** puts every configuration under the same thermal and
    ///   contention conditions. Block measurement charges whatever drift
    ///   happens during the run to whichever block ran last.
    /// * **Minimum, not median.** Contention and thermal throttling are
    ///   one-sided: they can only make a sample slower. The fastest observed
    ///   time is therefore the closest estimate of the uncontended cost, and it
    ///   does not move when an unrelated process wakes up. A median does.
    ///
    /// This is a shared desktop, not a reserved rig, so the spread is reported
    /// alongside: a ratio taken while the numbers are still swinging is not a
    /// ratio, and the reader needs to see which it is.
    struct Sampled {
        out: Vec<f32>,
        min_ms: f64,
        spread: f64,
    }

    fn measure_interleaved(
        m: usize,
        k: usize,
        n: usize,
        xv: &[f32],
        wv: &[f32],
        configs: &[(&str, fn())],
    ) -> Vec<Sampled> {
        // Compile each configuration first: variant selection happens at
        // compile time, so the env must be set before `compile`, not before
        // `run`.
        let mut exes: Vec<_> = configs
            .iter()
            .map(|(tag, set)| {
                set();
                let mut e = Session::new(Device::Metal).compile(build_tagged(tag, m, k, n));
                e.set_param("w", wv);
                e
            })
            .collect();

        for e in exes.iter_mut() {
            for _ in 0..WARMUP {
                let _ = e.run(&[("x", xv)]);
            }
        }

        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(ITERS); configs.len()];
        let mut outs: Vec<Vec<f32>> = vec![Vec::new(); configs.len()];
        for _ in 0..ITERS {
            for (i, e) in exes.iter_mut().enumerate() {
                // Device span when the backend reports one, wall clock only as
                // a fallback. A wall clock around `run()` measures host encode +
                // objc bridging + queue wait as well as the kernel, and on a
                // loaded host those dominate — this example withheld its own
                // aggregate at 24-47% spread for exactly that reason. The
                // command buffer's own GPUStartTime/GPUEndTime isolate the part
                // a kernel change can move.
                rlx_metal::gpu_span::reset();
                let t0 = Instant::now();
                let r = e.run(&[("x", xv)]);
                let wall = t0.elapsed().as_secs_f64() * 1e3;
                samples[i].push(rlx_metal::gpu_span::last_ms().unwrap_or(wall));
                outs[i] = r[0].clone();
            }
        }

        samples
            .into_iter()
            .zip(outs)
            .map(|(mut s, out)| {
                s.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
                let min_ms = s[0];
                // p50/min: how far the typical sample sits above the floor. Near
                // 1.0 means a quiet machine; well above means these numbers are
                // contention, not measurement.
                let spread = s[s.len() / 2] / min_ms;
                Sampled {
                    out,
                    min_ms,
                    spread,
                }
            })
            .collect()
    }

    /// Force MPS on, or force rlx's own kernels, for the next compile.
    fn use_mps(enabled: bool) {
        if enabled {
            rlx_ir::env::unset("RLX_DISABLE_MPS");
            rlx_ir::env::set("RLX_METAL_SGEMM_MPS", "1");
        } else {
            rlx_ir::env::set("RLX_DISABLE_MPS", "1");
            rlx_ir::env::unset("RLX_METAL_SGEMM_MPS");
        }
    }

    /// Let the cascade decide, as a real model would.
    fn use_default_routing() {
        rlx_ir::env::unset("RLX_DISABLE_MPS");
        rlx_ir::env::unset("RLX_METAL_SGEMM_MPS");
    }

    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
            .fold(0.0f32, f32::max)
    }

    pub fn run() {
        if !rlx_metal::is_available() {
            println!("Metal not available on this host — nothing to anchor.");
            return;
        }
        match contention_ok() {
            Ok(note) => println!("host state: {note}"),
            Err(why) => {
                eprintln!(
                    "REFUSING to measure: {why}.\n\
                     A timing taken under contention is not a timing. Free the machine, or set\n\
                     RLX_ALLOW_THROTTLE=1 if you accept the numbers are indicative only."
                );
                std::process::exit(1);
            }
        }

        // MPSGraph would replace the matmul being measured and is a separate
        // fast path with its own known init crash. Out of scope here.
        rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
        rlx_ir::env::set("RLX_DISABLE_MPSGRAPH_EXECUTABLE", "1");

        println!("\nrlx's own MSL sgemm vs Apple MPS — same process, same device state");
        println!("reference = MPSMatrixMultiplication (a real tuned library, NOT peak)\n");
        println!(
            "  {:<18} {:>30}  {:>9} {:>9}  {:>7}  {:>7}  {:>8}",
            "shape", "case", "rlx ms", "MPS ms", "rel", "noise", "routing"
        );

        let mut noisy = 0usize;
        let mut ratios: Vec<f64> = Vec::new();
        let mut mis_routed: Vec<String> = Vec::new();
        let mut worst: Option<(String, f64)> = None;

        for &(m, k, n, label) in SHAPES {
            let xv: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 1e-2 - 0.5).collect();
            let wv: Vec<f32> = (0..k * n).map(|i| ((i % 89) as f32) * 1e-2 - 0.5).collect();

            let r = measure_interleaved(
                m,
                k,
                n,
                &xv,
                &wv,
                &[
                    ("rlx", || use_mps(false)),
                    ("mps", || use_mps(true)),
                    ("default", use_default_routing),
                ],
            );
            let (rlx, mps, dflt) = (&r[0], &r[1], &r[2]);

            // A ratio between two different computations is meaningless.
            let rel_err = max_rel(&rlx.out, &mps.out);
            if rel_err > REL_TOL {
                println!(
                    "  {:<18} {:>30}  MISMATCH (max rel {rel_err:.2e}) — ratio withheld",
                    format!("{m}x{k}x{n}"),
                    label
                );
                continue;
            }

            let worst_spread = rlx.spread.max(mps.spread).max(dflt.spread);
            let best_ms = rlx.min_ms.min(mps.min_ms);
            // Only call a routing miss when the gap clears the noise floor this
            // shape actually showed. A fixed slack would report drift as a bug.
            let miss_bar = 1.03_f64.max(worst_spread);
            let routing = if dflt.min_ms <= best_ms * miss_bar {
                "ok"
            } else {
                let faster = if rlx.min_ms < mps.min_ms {
                    "rlx"
                } else {
                    "MPS"
                };
                mis_routed.push(format!(
                    "{m}x{k}x{n} ({label}): default {:.3} ms, but {faster} alone is {best_ms:.3} \
                     ms (noise floor {:.0}%)",
                    dflt.min_ms,
                    (worst_spread - 1.0) * 100.0
                ));
                "SLOWER"
            };

            let rel = mps.min_ms / rlx.min_ms; // >1 = rlx's own kernel is faster
            ratios.push(rel);
            if worst.as_ref().is_none_or(|(_, w)| rel < *w) {
                worst = Some((format!("{m}x{k}x{n} ({label})"), rel));
            }
            if worst_spread > NOISY_SPREAD {
                noisy += 1;
            }
            println!(
                "  {:<18} {:>30}  {:>9.3} {:>9.3}  {rel:>6.2}x  {:>6.0}%  {routing:>8}",
                format!("{m}x{k}x{n}"),
                label,
                rlx.min_ms,
                mps.min_ms,
                (worst_spread - 1.0) * 100.0
            );
        }

        if ratios.is_empty() {
            println!("\nno comparable shapes — every case mismatched the reference.");
            return;
        }
        // Geometric mean: ratios compose multiplicatively, so an arithmetic
        // mean would let one 3x outlier hide several 0.5x shapes.
        let gm = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
        // Withhold the aggregate when most rows are contention. The CUDA twin
        // refuses to measure at all on a busy GPU; macOS has no equivalent
        // exclusive-access signal, so the check moves after the fact — but the
        // conclusion is the same. A number printed here gets quoted, and
        // "1.05x of MPS" derived from rows that swing 20-90% run to run is not
        // a measurement, it is an average of noise.
        if noisy * 2 > ratios.len() {
            println!(
                "\n  AGGREGATE WITHHELD: {noisy} of {} shape(s) are contention-dominated.\n                   Per-shape ratios here do not reproduce (observed spans of 0.80x-1.24x for the\n                   same shape across runs). Re-run on an idle machine.",
                ratios.len()
            );
        } else {
            println!(
                "\n  geometric mean over {} shape(s): {gm:.2}x of MPS",
                ratios.len()
            );
            if let Some((shape, r)) = &worst {
                // Name the worst case explicitly. An aggregate that hides where
                // the implementation is weakest is the failure this exists to avoid.
                println!("  worst shape: {shape} at {r:.2}x");
            }
        }
        if noisy > 0 {
            println!(
                "  CAUTION: {noisy} of {} shape(s) showed >{:.0}% spread between the median and\n                   the fastest sample — those rows are contention, not measurement. Re-run on an\n                   otherwise idle machine before quoting them.",
                ratios.len(),
                (NOISY_SPREAD - 1.0) * 100.0
            );
        }
        if mis_routed.is_empty() {
            println!("  routing: the cascade picked the faster path at every shape");
        } else {
            println!(
                "\n  ROUTING MISSES ({}) — the cascade chose the slower path:",
                mis_routed.len()
            );
            for m in &mis_routed {
                println!("    {m}");
            }
        }
        println!(
            "\n  Read this as distance from a tuned library, not from peak. Above 1.00x\n  \
             means MPS is not tuned for that shape, not that rlx is optimal."
        );
    }
}
