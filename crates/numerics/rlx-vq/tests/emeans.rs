// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! e-means (arXiv:2611.09501) across every rlx backend and assignment precision.
//!
//! The step graph runs on each device in `rlx_runtime::available_devices()`
//! at each precision in [`PRECISIONS`], in two forms:
//!
//! - `nf:<fmt>` — the whole graph under `CompileOptions::number_format`
//!   (f64, double-single f64x2, f32, f16, bf16, f8 e4m3), rlx's cross-backend
//!   precision mode;
//! - `cast:<dtype>` — native casts in the graph: the assignment runs in that
//!   dtype ([`step_graph_with`]), statistics and EMA in f32.
//!
//! Each combination is checked against an independent host computation; a
//! full graph-driven run is checked against the host trainer's loss; and the
//! host trainer is checked against Lloyd's k-means.
//!
//! Backends come from `rlx-runtime` features, enabled on the command line:
//!
//! ```sh
//! cargo test -p rlx-vq --features emeans,metal --release --test emeans -- --nocapture
//! cargo test -p rlx-vq --features emeans,rlx-runtime/cuda,rlx-runtime/vulkan ...
//! cargo test -p rlx-vq --features emeans,rlx-runtime/rocm,rlx-runtime/xdna ...
//! ```
//!
//! A combination the backend cannot compile or run is reported as
//! `unsupported` (with the reason) and does not fail the test; wrong numbers
//! always fail it. Set `RLX_REQUIRE_DEVICE=1` to fail on `unsupported` too.

#![cfg(feature = "emeans")]

use std::panic::{AssertUnwindSafe, catch_unwind};

use rlx_ir::{DType, NumberFormat};
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};
use rlx_vq::emeans::{
    BatchStats, Emeans, EmeansConfig, Init, assign_l2, epoch_seed, step_graph_with, step_io,
};

/// A device's own matmul precision when it is coarser than f32: the XDNA NPU
/// multiplies in INT8 per channel (bf16 at best), so its sums carry that error
/// whatever the requested format.
fn device_floor(dev: Device) -> f32 {
    if dev == Device::Xdna { 1e-2 } else { 0.0 }
}

/// One precision setting of the matrix.
#[derive(Clone, Copy)]
enum Prec {
    /// Whole graph under `CompileOptions::number_format`.
    Format(&'static str),
    /// Assignment in this dtype via graph casts.
    Cast(DType),
}

const PRECISIONS: [Prec; 10] = [
    Prec::Format("f64"),
    Prec::Format("f64x2"),
    Prec::Format("f32"),
    Prec::Format("f16"),
    Prec::Format("bf16"),
    Prec::Format("f8"),
    Prec::Cast(DType::F64),
    Prec::Cast(DType::F32),
    Prec::Cast(DType::F16),
    Prec::Cast(DType::BF16),
];

impl Prec {
    fn name(self) -> String {
        match self {
            Prec::Format(f) => format!("nf:{f}"),
            Prec::Cast(d) => format!("cast:{}", dtype_name(d)),
        }
    }

    /// Relative error this setting may introduce on `dev`: the format's
    /// mantissa, or the device's own matmul precision when that is coarser
    /// (the XDNA NPU multiplies in INT8 per channel, bf16 at best).
    fn eps_on(self, dev: Device) -> f32 {
        let eps = self.eps().max(device_floor(dev));
        // CoreML has no bf16 (it demotes to f16) and runs half-precision
        // graphs on the Neural Engine, whose f16 runs land like bf16 ones
        // elsewhere: a cast:f16 run ended 2.4e-2 above the host, where every
        // other backend's ended within 1e-4. Its f32 runs are exact.
        if dev == Device::Ane && eps > 1e-5 {
            eps.max(Prec::Cast(DType::BF16).eps())
        } else {
            eps
        }
    }

    /// Mantissa-scale relative error this setting may introduce.
    fn eps(self) -> f32 {
        let which = match self {
            Prec::Format(f) => f,
            Prec::Cast(d) => dtype_name(d),
        };
        match which {
            "f64" | "f64x2" | "f32" => 1e-5,
            "f16" => 2e-3,
            "bf16" => 1.6e-2,
            _ => 1.3e-1, // f8 e4m3: 3 mantissa bits
        }
    }

    /// Whether everything (statistics and EMA too) is rounded, not just the
    /// assignment.
    fn rounds_statistics(self) -> bool {
        matches!(self, Prec::Format(f) if !matches!(f, "f64" | "f64x2" | "f32"))
    }

    /// The device trainer's compile options on `dev` and assignment dtype for
    /// this setting.
    #[cfg(feature = "emeans-device")]
    fn options(self, dev: Device) -> (CompileOptions, DType) {
        let base = rlx_vq::emeans::device::device_options(dev);
        match self {
            Prec::Format(f) => (
                base.number_format(f.parse().expect("number format")),
                DType::F32,
            ),
            Prec::Cast(dt) => (base, dt),
        }
    }

    fn compile(self, dev: Device, b: usize, d: usize, k: usize) -> CompiledGraph {
        let session = Session::new(dev);
        match self {
            Prec::Format(f) => {
                let format: NumberFormat = f.parse().expect("number format");
                session.compile_with(
                    step_graph_with(b, d, k, DType::F32),
                    &CompileOptions::new().number_format(format),
                )
            }
            Prec::Cast(dt) => session.compile(step_graph_with(b, d, k, dt)),
        }
    }
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform(state: &mut u64) -> f32 {
    (splitmix(state) >> 40) as f32 / (1u64 << 24) as f32
}

/// `n` rows around `blobs` centres with unequal spreads and sizes.
fn blobs(n: usize, d: usize, blobs: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let centres: Vec<f32> = (0..blobs * d)
        .map(|_| uniform(&mut s) * 20.0 - 10.0)
        .collect();
    let mut x = Vec::with_capacity(n * d);
    for _ in 0..n {
        // Skewed blob choice so some regions are dense and some sparse.
        let u = uniform(&mut s);
        let b = ((u * u) * blobs as f32) as usize % blobs;
        let spread = 0.3 + (b % 5) as f32 * 0.2;
        for t in 0..d {
            let noise = (uniform(&mut s) + uniform(&mut s) + uniform(&mut s) - 1.5) * spread;
            x.push(centres[b * d + t] + noise);
        }
    }
    x
}

fn seeded_init(x: &[f32], d: usize, k: usize, seed: u32) -> Vec<f32> {
    let perm = rlx_rng::numpy::RandomState::new(seed).permutation(x.len() / d);
    perm[..k]
        .iter()
        .flat_map(|&i| x[i * d..(i + 1) * d].iter().copied())
        .collect()
}

fn mse(x: &[f32], c: &[f32], d: usize) -> f64 {
    let (dist, _) = assign_l2(x, c, d);
    dist.iter().map(|&v| v as f64).sum::<f64>() / (x.len() / d) as f64
}

fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Max absolute difference over the reference's max magnitude.
fn rel_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    diff / b.iter().map(|v| v.abs()).fold(f32::MIN_POSITIVE, f32::max)
}

fn dtype_name(p: DType) -> &'static str {
    match p {
        DType::F64 => "f64",
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => "?",
    }
}

enum Outcome {
    Pass(String),
    Unsupported(String),
    Fail(String),
}

fn panic_text(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "panic".into())
        .lines()
        .next()
        .unwrap_or("")
        .chars()
        .take(140)
        .collect()
}

/// Set in a child process to the one `Device/precision` case it runs.
const CASE_ENV: &str = "EMEANS_PARITY_CASE";
/// Prefix of the child's result line on stdout.
const RESULT_MARK: &str = "@@emeans-case ";

fn run_case(
    body: &dyn Fn(Device, Prec) -> Result<String, String>,
    dev: Device,
    p: Prec,
) -> Outcome {
    match catch_unwind(AssertUnwindSafe(|| body(dev, p))) {
        Ok(Ok(msg)) => Outcome::Pass(msg),
        Ok(Err(msg)) => Outcome::Fail(msg),
        Err(e) => Outcome::Unsupported(panic_text(e)),
    }
}

/// Run `body` per (device, precision) and print a matrix. Each case runs in
/// its own process (this test binary re-run on `test` with [`CASE_ENV`]), so
/// a backend that aborts or corrupts memory fails its own row instead of
/// taking the matrix down. Compile/run panics are `unsupported`; a returned
/// `Err` or a crash is a failure.
fn matrix(test: &str, title: &str, body: &dyn Fn(Device, Prec) -> Result<String, String>) {
    if let Ok(case) = std::env::var(CASE_ENV) {
        let (d, pn) = case.split_once('/').expect("case is Device/precision");
        let dev = rlx_runtime::available_devices()
            .into_iter()
            .find(|x| format!("{x:?}") == d)
            .expect("case device is available");
        let p = PRECISIONS
            .into_iter()
            .find(|p| p.name() == pn)
            .expect("case precision");
        let (tag, msg) = match run_case(body, dev, p) {
            Outcome::Pass(m) if m.starts_with("n/a") => ("n/a", m),
            Outcome::Pass(m) => ("ok", m),
            Outcome::Unsupported(m) => ("unsupported", m),
            Outcome::Fail(m) => ("FAIL", m),
        };
        println!("{RESULT_MARK}{tag}\t{}", msg.replace('\n', " "));
        return;
    }
    let require = std::env::var("RLX_REQUIRE_DEVICE").is_ok_and(|v| v == "1");
    let exe = std::env::current_exe().expect("test binary path");
    let mut failures = Vec::new();
    eprintln!("\n== {title}");
    for (dev, why) in rlx_runtime::detected_unavailable_devices() {
        let why: String = why.lines().next().unwrap_or("").chars().take(140).collect();
        eprintln!("{:<8} {:<10} not runnable: {why}", format!("{dev:?}"), "-");
        if require {
            failures.push(format!("{dev:?}: detected but not runnable: {why}"));
        }
    }
    for dev in rlx_runtime::available_devices() {
        for p in PRECISIONS {
            let out = std::process::Command::new(&exe)
                .args([test, "--exact", "--nocapture", "--test-threads=1"])
                .env(CASE_ENV, format!("{dev:?}/{}", p.name()))
                .output()
                .expect("spawn case");
            let stdout = String::from_utf8_lossy(&out.stdout);
            let reported = stdout
                .lines()
                .find_map(|l| l.find(RESULT_MARK).map(|i| &l[i + RESULT_MARK.len()..]))
                .and_then(|r| r.split_once('\t'));
            let outcome = match reported {
                Some(("ok" | "n/a", m)) => Outcome::Pass(m.to_string()),
                Some(("unsupported", m)) => Outcome::Unsupported(m.to_string()),
                Some((_, m)) => Outcome::Fail(m.to_string()),
                None => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let last = stderr
                        .lines()
                        .rev()
                        .find(|l| !l.trim().is_empty())
                        .unwrap_or("");
                    Outcome::Fail(format!("CRASH ({}): {last}", out.status))
                }
            };
            let (tag, msg) = match &outcome {
                Outcome::Pass(m) if m.starts_with("n/a") => ("n/a        ", m),
                Outcome::Pass(m) => ("ok         ", m),
                Outcome::Unsupported(m) => ("unsupported", m),
                Outcome::Fail(m) => ("FAIL       ", m),
            };
            eprintln!("{:<8} {:<10} {tag} {msg}", format!("{dev:?}"), p.name());
            match outcome {
                Outcome::Fail(m) => failures.push(format!("{dev:?}/{}: {m}", p.name())),
                Outcome::Unsupported(m)
                    if require || (dev == Device::Cpu && matches!(p, Prec::Format("f32"))) =>
                {
                    failures.push(format!("{dev:?}/{}: unsupported: {m}", p.name()))
                }
                _ => {}
            }
        }
    }
    assert!(failures.is_empty(), "{title}:\n{}", failures.join("\n"));
}

/// Equation 2 from the definition, in f64, for the labels a backend chose.
fn ema_reference(
    batch: &[f32],
    labels: &[u32],
    c0: &[f32],
    u0: &[f32],
    alpha: f64,
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let k = u0.len();
    let mut sums = vec![0f64; k * d];
    let mut counts = vec![0f64; k];
    for (row, &l) in batch.chunks_exact(d).zip(labels) {
        let l = l as usize;
        counts[l] += 1.0;
        for t in 0..d {
            sums[l * d + t] += row[t] as f64;
        }
    }
    let mut c = c0.to_vec();
    let mut u = vec![0f32; k];
    for j in 0..k {
        let mass = (1.0 - alpha) * u0[j] as f64 + alpha * counts[j];
        u[j] = mass as f32;
        if counts[j] > 0.0 {
            let w = alpha * counts[j] / mass;
            for t in 0..d {
                let mean = sums[j * d + t] / counts[j];
                let cv = c0[j * d + t] as f64;
                c[j * d + t] = (cv + w * (mean - cv)) as f32;
            }
        }
    }
    let mut stats = vec![0f32; k * (d + 1)];
    for j in 0..k {
        for t in 0..d {
            stats[j * (d + 1) + t] = sums[j * d + t] as f32;
        }
        stats[j * (d + 1) + d] = counts[j] as f32;
    }
    (c, u, stats)
}

/// One graph step per (device, precision) against an independent host
/// computation: the chosen labels are near-optimal for that precision, and the
/// statistics, EMA and objective are exactly what those labels imply.
#[test]
fn step_graph_parity_across_backends_and_precisions() {
    let (n, d, k, b) = (4096, 16, 32, 512);
    let x = blobs(n, d, 12, 7);
    let cfg = EmeansConfig {
        batch_size: Some(b),
        unused_threshold: Some(0.0),
        ..EmeansConfig::default()
    };
    // Advance the host model a few steps so the state is not the initial one.
    let mut host = Emeans::new(d, k, cfg);
    host.begin(n, &x[..k * d]);
    for i in 0..3 {
        let mut st = BatchStats::new(k, d);
        host.accumulate(&x[i * b * d..(i + 1) * b * d], &mut st);
        host.update(&st);
    }
    let c0 = host.centroids().to_vec();
    let u0 = host.usage().to_vec();
    let alpha64 = host.current_alpha();
    let alpha = [alpha64 as f32];
    let batch = &x[3 * b * d..4 * b * d];
    let (_, host_labels) = assign_l2(batch, &c0, d);

    matrix(
        "step_graph_parity_across_backends_and_precisions",
        "e-means step graph vs host",
        &|dev, p| {
            let outs = p.compile(dev, b, d, k).run(&[
                (step_io::X, batch),
                (step_io::CENTROIDS, &c0),
                (step_io::USAGE, &u0),
                (step_io::ALPHA, &alpha),
            ]);
            let labels: Vec<u32> = outs[2].iter().map(|&v| v as u32).collect();
            if let Some(bad) = labels.iter().find(|&&l| l as usize >= k) {
                return Err(format!("label {bad} out of range"));
            }
            // Near-optimal ranking: the chosen centroid is within the precision's
            // slack of the f32-exact nearest one.
            let tol = p.eps_on(dev);
            let mut worst = 0f32;
            for (i, row) in batch.chunks_exact(d).enumerate() {
                let chosen = sq_dist(row, &c0[labels[i] as usize * d..][..d]);
                let best = c0
                    .chunks_exact(d)
                    .map(|c| sq_dist(row, c))
                    .fold(f32::INFINITY, f32::min);
                let scale = row.iter().map(|v| v * v).sum::<f32>()
                    + c0[labels[i] as usize * d..][..d]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f32>();
                worst = worst.max((chosen - best) / scale);
            }
            if worst > tol {
                return Err(format!(
                    "chosen centroid {worst:.2e} past nearest (tol {tol:.0e})"
                ));
            }
            let agree = labels
                .iter()
                .zip(&host_labels)
                .filter(|(a, b)| a == b)
                .count() as f64
                / labels.len() as f64;
            if p.eps_on(dev) <= 1e-5 && agree < 0.99 {
                return Err(format!("label agreement {agree:.4} with the f32 host"));
            }
            if p.rounds_statistics() && p.eps() > 2e-2 {
                // f8 cannot hold per-cluster sums (max 448) or counts past 16:
                // only the ranking is meaningful at this format.
                return Ok(format!(
                    "agree={agree:.4} rank={worst:.1e} (statistics not representable)"
                ));
            }
            // Everything downstream of the labels must be what those labels
            // imply: exactly (f32 rounding) when only the assignment is reduced,
            // within the format's rounding when the statistics are rounded too.
            let (c_ref, u_ref, stats_ref) = ema_reference(batch, &labels, &c0, &u0, alpha64, d);
            let dc = rel_diff(&outs[0], &c_ref);
            let du = rel_diff(&outs[1], &u_ref);
            let ds = rel_diff(&outs[3], &stats_ref);
            let obj_ref: f64 = batch
                .chunks_exact(d)
                .zip(&labels)
                .map(|(row, &l)| sq_dist(row, &c0[l as usize * d..][..d]) as f64)
                .sum();
            let dobj = (outs[4][0] as f64 - obj_ref).abs() / obj_ref;
            let msg = format!(
                "agree={agree:.4} rank={worst:.1e} Δc={dc:.1e} Δusage={du:.1e} Δstats={ds:.1e} Δobj={dobj:.1e}"
            );
            let tol = if p.rounds_statistics() || device_floor(dev) > 0.0 {
                4.0 * p.eps_on(dev)
            } else {
                1e-5
            };
            if dc > tol || du > tol || ds > tol || dobj as f32 > tol.max(1e-5) {
                return Err(msg);
            }
            Ok(msg)
        },
    );
}

/// The step's assignment does not depend on where the data sits: shifting the
/// batch and the centroids by the same vector keeps the labels. Computed raw,
/// ‖c‖² and x·c grow with the offset and their rounding swamps the gaps
/// between nearby centroids (an XDNA INT8 or bf16 matmul labelled 8% of rows
/// right at offset 300), and in f16 ‖c‖² overflows outright.
#[test]
fn step_graph_assignment_is_translation_invariant() {
    let (n, d, k, b) = (4096, 16, 32, 512);
    let offset = 100.0;
    let x = blobs(n, d, 12, 7);
    let c0 = x[..k * d].to_vec();
    let batch = &x[k * d..(k + b) * d];
    let shift = |v: &[f32]| v.iter().map(|x| x + offset).collect::<Vec<f32>>();
    let (c1, batch1) = (shift(&c0), shift(batch));
    let usage = vec![1f32; k];
    let alpha = [0.5f32];

    matrix(
        "step_graph_assignment_is_translation_invariant",
        "step labels at offset 0 vs offset 100",
        &|dev, p| {
            if p.rounds_statistics() && p.eps() > 2e-3 {
                return Ok("n/a: the format rounds the shifted inputs past the blob spread".into());
            }
            let mut graph = p.compile(dev, b, d, k);
            let mut labels = |rows: &[f32], c: &[f32]| -> Vec<f32> {
                graph
                    .run(&[
                        (step_io::X, rows),
                        (step_io::CENTROIDS, c),
                        (step_io::USAGE, &usage),
                        (step_io::ALPHA, &alpha),
                    ])
                    .swap_remove(2)
            };
            let (l0, l1) = (labels(batch, &c0), labels(&batch1, &c1));
            let agree = l0.iter().zip(&l1).filter(|(a, b)| a == b).count() as f64 / b as f64;
            // Excess distance of the shifted labels over the exact nearest
            // centroid, relative to the nearest distance.
            let (mut excess, mut nearest) = (0f64, 0f64);
            for (row, &l) in batch.chunks_exact(d).zip(&l1) {
                let best = c0
                    .chunks_exact(d)
                    .map(|c| sq_dist(row, c))
                    .fold(f32::INFINITY, f32::min);
                excess += (sq_dist(row, &c0[l as usize * d..][..d]) - best) as f64;
                nearest += best as f64;
            }
            let regret = excess / nearest;
            let msg = format!("agree={agree:.4} regret={regret:.1e}");
            // A number format rounds the shifted inputs themselves (f16 spacing
            // at 100 is 1/16), which moves labels without any cancellation:
            // there only near-optimality is required.
            let ok = if p.rounds_statistics() {
                regret < 2e-2
            } else {
                agree >= 0.97
            };
            if ok { Ok(msg) } else { Err(msg) }
        },
    );
}

/// A whole run driven through the compiled step graph (host schedule and
/// revival) on every backend and precision ends near the host trainer's loss.
#[test]
fn graph_driven_run_across_backends_and_precisions() {
    let (n, d, k) = (8192, 16, 48);
    let x = blobs(n, d, 20, 11);
    let cfg = EmeansConfig {
        nepochs: 3.0,
        ..EmeansConfig::default()
    };
    let mut host = Emeans::new(d, k, cfg.clone());
    let (host_c, _) = host.train(&x, None);
    let host_mse = mse(&x, &host_c, d);
    let init = seeded_init(&x, d, k, cfg.seed);

    matrix(
        "graph_driven_run_across_backends_and_precisions",
        "graph-driven run vs host trainer",
        &|dev, p| {
            let mut em = Emeans::new(d, k, cfg.clone());
            em.begin(n, &init);
            let recipe = em.recipe().unwrap().clone();
            if p.rounds_statistics() && p.eps() > 2e-3 {
                // bf16 holds integers exactly only to 256 and f8 to 16: rounded
                // per-cluster counts no longer cover the batch, so a run is not
                // e-means any more.
                return Ok("n/a: statistics rounded past their exact-integer range".into());
            }
            let mut full = p.compile(dev, recipe.batch_size, d, k);
            let mut tail = None;
            let mut batch = Vec::new();
            for (epoch, &rows) in recipe.epoch_rows.iter().enumerate() {
                let order =
                    rlx_rng::numpy::RandomState::new(epoch_seed(cfg.seed, epoch)).permutation(n);
                for idx in order[..rows].chunks(recipe.batch_size) {
                    batch.clear();
                    for &i in idx {
                        batch.extend_from_slice(&x[i * d..(i + 1) * d]);
                    }
                    let alpha = [em.current_alpha() as f32];
                    let graph = if idx.len() == recipe.batch_size {
                        &mut full
                    } else {
                        tail.get_or_insert_with(|| p.compile(dev, idx.len(), d, k))
                    };
                    let outs = graph.run(&[
                        (step_io::X, &batch),
                        (step_io::CENTROIDS, em.centroids()),
                        (step_io::USAGE, em.usage()),
                        (step_io::ALPHA, &alpha),
                    ]);
                    em.update_from_step(&outs[0], &outs[1], &outs[3], outs[4][0] as f64);
                }
                em.end_epoch();
            }
            let got = mse(&x, em.finalize(), d);
            let rel = (got - host_mse) / host_mse;
            let msg = format!(
                "mse={got:.5} host={host_mse:.5} rel={rel:+.2e} splits={} (host {})",
                em.n_splits(),
                host.n_splits()
            );
            // Lower precision may cost a little loss, never a lot.
            let slack = if p.eps_on(dev) <= 1e-5 {
                0.01
            } else if p.eps_on(dev) <= 2e-3 {
                0.02
            } else {
                0.05
            };
            if rel.abs() > slack && rel > 0.0 {
                return Err(msg);
            }
            Ok(msg)
        },
    );
}

/// Plain Lloyd iterations from the same seeded initialization.
fn lloyd(x: &[f32], init: &[f32], d: usize, iters: usize) -> Vec<f32> {
    let k = init.len() / d;
    let mut c = init.to_vec();
    for _ in 0..iters {
        let (_, labels) = assign_l2(x, &c, d);
        let mut sums = vec![0f64; k * d];
        let mut counts = vec![0usize; k];
        for (row, &l) in x.chunks_exact(d).zip(&labels) {
            let l = l as usize;
            counts[l] += 1;
            for t in 0..d {
                sums[l * d + t] += row[t] as f64;
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                for t in 0..d {
                    c[j * d + t] = (sums[j * d + t] / counts[j] as f64) as f32;
                }
            }
        }
    }
    c
}

/// The paper's headline claim at small scale: 5 e-means epochs reach a lower
/// training MSE than 25 Lloyd iterations from the same initialization.
#[test]
fn five_epochs_beat_lloyd_at_25_iterations() {
    let (n, d, k) = (30_000, 24, 128);
    let x = blobs(n, d, 300, 3);
    let cfg = EmeansConfig::default();
    let init = seeded_init(&x, d, k, cfg.seed);

    let mut em = Emeans::new(d, k, cfg);
    let (c, labels) = em.train(&x, Some(&init));
    let e = mse(&x, &c, d);
    let l = mse(&x, &lloyd(&x, &init, d, 25), d);
    let r = em.recipe().unwrap();
    eprintln!(
        "e-means MSE {e:.5} (B={}, α₀={:.3}, {} steps, {} splits) vs Lloyd-25 {l:.5}",
        r.batch_size,
        r.alpha0,
        r.total_steps,
        em.n_splits()
    );
    assert!(labels.unwrap().iter().all(|&v| (v as usize) < k));
    assert!(e <= l, "e-means {e} worse than Lloyd {l}");
}

/// Random-row initialization can strand a blob without a centroid, which
/// revival does not repair; k-means++ does not. On these 20 blobs seed 1 of
/// the reference's initialization ends near 4.8× the other seeds' MSE.
#[test]
fn kmeans_plus_plus_init_avoids_stranded_blobs() {
    let (n, d, k) = (8192, 16, 48);
    let x = blobs(n, d, 20, 11);
    let run = |seed: u32, init: Init| {
        let cfg = EmeansConfig {
            nepochs: 3.0,
            seed,
            init,
            ..EmeansConfig::default()
        };
        let (c, _) = Emeans::new(d, k, cfg).train(&x, None);
        mse(&x, &c, d)
    };
    let random: Vec<f64> = (1..=4).map(|s| run(s, Init::RandomRows)).collect();
    let pp: Vec<f64> = (1..=4).map(|s| run(s, Init::KMeansPlusPlus)).collect();
    eprintln!("random rows {random:.4?}  k-means++ {pp:.4?}");
    let best = random.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        random[0] > 2.0 * best,
        "seed 1 no longer strands a blob: {random:?}"
    );
    for (seed, &m) in (1..).zip(&pp) {
        assert!(
            m < 1.05 * best,
            "k-means++ seed {seed}: {m:.4} vs best random {best:.4}"
        );
    }
    #[cfg(feature = "emeans-device")]
    {
        // The device trainer seeds the same way.
        let cfg = EmeansConfig {
            nepochs: 3.0,
            seed: 1,
            init: Init::KMeansPlusPlus,
            ..EmeansConfig::default()
        };
        let (c, _) =
            rlx_vq::emeans::device::DeviceEmeans::new(Device::Cpu, d, k, cfg).train(&x, None);
        let m = mse(&x, &c, d);
        assert!(
            m < 1.05 * best,
            "device k-means++ seed 1: {m:.4} vs best random {best:.4}"
        );
    }
}

/// Phase 2: the device-resident trainer on every backend and precision ends
/// at the host trainer's loss with the same revivals, and keeps its state on
/// the device wherever the backend has GPU handles.
#[cfg(feature = "emeans-device")]
#[test]
fn device_resident_run_across_backends_and_precisions() {
    use rlx_vq::emeans::device::DeviceEmeans;
    let (n, d, k) = (8192, 16, 48);
    let x = blobs(n, d, 20, 11);
    let cfg = EmeansConfig {
        nepochs: 3.0,
        ..EmeansConfig::default()
    };
    let mut host = Emeans::new(d, k, cfg.clone());
    let t = std::time::Instant::now();
    let (host_c, host_labels) = host.train(&x, None);
    let host_secs = t.elapsed().as_secs_f64();
    let host_mse = mse(&x, &host_c, d);
    let host_labels = host_labels.unwrap();

    matrix(
        "device_resident_run_across_backends_and_precisions",
        "device-resident trainer vs host trainer",
        &|dev, p| {
            if p.rounds_statistics() && p.eps() > 2e-3 {
                return Ok("n/a: statistics rounded past their exact-integer range".into());
            }
            let (options, assign) = p.options(dev);
            let mut em = DeviceEmeans::with_options(dev, d, k, cfg.clone(), options, assign);
            let t = std::time::Instant::now();
            let (c, labels) = em.train(&x, None);
            let secs = t.elapsed().as_secs_f64();
            let got = mse(&x, &c, d);
            let rel = (got - host_mse) / host_mse;
            let labels = labels.expect("labels captured");
            let agree = labels
                .iter()
                .zip(&host_labels)
                .filter(|(a, b)| a == b)
                .count() as f64
                / n as f64;
            let splits = em.emeans().n_splits();
            let msg = format!(
                "resident={} mse={got:.5} host={host_mse:.5} rel={rel:+.2e} labels={agree:.4} \
                 splits={splits} (host {}) {secs:.2}s (host {host_secs:.2}s)",
                em.resident().unwrap(),
                host.n_splits()
            );
            if labels.iter().any(|&l| l as usize >= k) {
                return Err(format!("unlabelled rows: {msg}"));
            }
            let slack = if p.eps_on(dev) <= 1e-5 {
                0.01
            } else if p.eps_on(dev) <= 2e-3 {
                0.02
            } else {
                0.05
            };
            if rel > slack {
                return Err(msg);
            }
            // A backend that has GPU handles must keep the state on them.
            let handles = Session::new(dev)
                .compile(step_graph_with(8, d, k, DType::F32))
                .capabilities()
                .gpu_handles;
            if handles && !em.resident().unwrap() {
                return Err(format!("state not resident: {msg}"));
            }
            Ok(msg)
        },
    );
}
