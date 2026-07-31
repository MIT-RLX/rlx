// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PLAN L5 — uniform benchmark harness across RLX backends + patterns.
//!
//! Replaces the bespoke per-backend bench files (`bench_cuda_modes`,
//! `bench_rocm_modes`, `bench_metal_full`, …) with one structure:
//!
//! - **`BenchmarkPattern`** — describes *what to measure* (a graph
//!   builder + sample-input shape). Implementations are tier-tagged:
//!   `Tier::L1` for single-op micro-benches (matmul, layernorm,
//!   softmax), `Tier::L2` for composite patterns (FFN sub-graph,
//!   attention block).
//!
//! - **`run_benchmark`** — *where to run* abstracted via
//!   `Device`. Compiles the pattern's graph for the device, runs
//!   warm-ups, then `n_runs` timed iterations. Uses `rlx_ir::Tick`
//!   (CNTVCT_EL0 directly on Apple Silicon) for sub-µs precision.
//!
//! - **`BenchResult`** — per-run timings + aggregate stats.
//!
//! ## Adding a backend
//!
//! Backends are addressed by `rlx_driver::Device`; registration is
//! handled by the existing `rlx-runtime` machinery. Enable the
//! relevant feature (`cargo run --features metal`) and pass
//! `Device::Metal`. No per-backend adapter file in this crate.
//!
//! ## Adding a pattern
//!
//! Implement `BenchmarkPattern` for your workload type. The trait
//! gives you a graph builder + sample-input layout; the harness
//! handles compilation, warm-up, timing, and stats.

use rlx_driver::Device;
use rlx_ir::{Graph, Tick};
use rlx_runtime::{GpuThermal, Session, device_thermal};

pub mod patterns;

/// Coarse grouping for benchmark patterns. Mirrors luminal's L1/L2
/// distinction:
/// - **`L1`** — single-op micro-bench (matmul alone, softmax alone, …).
///   Captures the kernel's raw throughput.
/// - **`L2`** — composite pattern (matmul→bias→activation, FFN block,
///   attention block, …). Captures fusion/scheduling effects on top
///   of raw kernel throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    L1,
    L2,
}

/// What to measure. Implement on a workload type to make it
/// benchmarkable. Each call produces a fresh graph because backends
/// can mutate compile state per-graph; the harness compiles once
/// per `run_benchmark` call.
pub trait BenchmarkPattern {
    /// Stable name for this pattern (used in result labels).
    fn name(&self) -> &str;

    /// Tier classification (L1 micro-op vs L2 composite).
    fn tier(&self) -> Tier;

    /// Build a fresh graph that performs one iteration of this pattern.
    /// Includes inputs, params (set via `set_param` after compile if
    /// `param_data` returns Some), and exactly one output node.
    fn build_graph(&self) -> Graph;

    /// Per-input data the harness will pass to `compiled.run()`.
    /// Returned in graph-input declaration order.
    fn input_data(&self) -> Vec<(String, Vec<f32>)>;

    /// Optional: per-param data the harness will set via `set_param`
    /// before the first run. Defaults to none — pattern providers
    /// using `Op::Input` for everything (no `Op::Param` weights) need
    /// nothing.
    fn param_data(&self) -> Vec<(String, Vec<f32>)> {
        Vec::new()
    }
}

/// Aggregate timing data from a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Pattern name (mirrors `BenchmarkPattern::name()`).
    pub pattern: String,
    /// Tier (L1 / L2).
    pub tier: Tier,
    /// Device the benchmark ran on.
    pub device: Device,
    /// Number of timed iterations.
    pub n_runs: usize,
    /// Per-iteration nanoseconds (length == `n_runs`).
    pub samples_ns: Vec<u64>,
    /// Peak GPU telemetry observed around the timed loop, when the bench
    /// ran on a GPU backend with a readable sensor. `None` on CPU / Apple
    /// backends or hosts without a GPU management library. Wall-clock
    /// timing silently absorbs thermal throttling — this makes a
    /// heat-degraded sample visible.
    pub gpu_peak: Option<GpuPeak>,
}

/// Peak GPU readings captured across a timed benchmark. Read-only and
/// best-effort — each field is `None` when the board doesn't expose it.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuPeak {
    /// GPU product name, if the driver reports one.
    pub name: Option<String>,
    /// Hottest GPU-die temperature seen, °C.
    pub temp_c: Option<f32>,
    /// Highest board power draw seen, W.
    pub power_w: Option<f32>,
}

impl BenchResult {
    pub fn min_ns(&self) -> u64 {
        self.samples_ns.iter().copied().min().unwrap_or(0)
    }
    pub fn max_ns(&self) -> u64 {
        self.samples_ns.iter().copied().max().unwrap_or(0)
    }
    pub fn mean_ns(&self) -> u64 {
        if self.samples_ns.is_empty() {
            return 0;
        }
        let sum: u128 = self.samples_ns.iter().map(|&v| v as u128).sum();
        (sum / self.samples_ns.len() as u128) as u64
    }
    /// Median (cheap O(n log n) sort — for a few hundred samples this
    /// is irrelevant). Useful when an outlier pulls the mean.
    pub fn median_ns(&self) -> u64 {
        if self.samples_ns.is_empty() {
            return 0;
        }
        let mut sorted = self.samples_ns.clone();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    }
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let to_us = |ns: u64| ns as f64 / 1000.0;
        write!(
            f,
            "{:?}/{:?} {:?}: n={} mean={:.2}µs median={:.2}µs min={:.2}µs max={:.2}µs",
            self.tier,
            self.device,
            self.pattern,
            self.n_runs,
            to_us(self.mean_ns()),
            to_us(self.median_ns()),
            to_us(self.min_ns()),
            to_us(self.max_ns()),
        )?;
        if let Some(g) = &self.gpu_peak {
            if let Some(t) = g.temp_c {
                write!(f, " gpu_peak={t:.0}°C")?;
            }
            if let Some(p) = g.power_w {
                write!(f, " {p:.0}W")?;
            }
        }
        Ok(())
    }
}

/// Read the backing GPU's telemetry for a bench on `device`. Index 0
/// matches the context both rlx-cuda and rlx-rocm bind (device 0). No-op
/// (`None`) on non-GPU backends or hosts without a management library.
fn gpu_bench_sample(device: Device) -> Option<GpuThermal> {
    match device {
        Device::Cuda | Device::Rocm => device_thermal(device, 0),
        _ => None,
    }
}

fn max_opt(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

/// Fold pre- and post-run GPU samples into a peak reading.
fn merge_gpu_peak(a: Option<GpuThermal>, b: Option<GpuThermal>) -> Option<GpuPeak> {
    if a.is_none() && b.is_none() {
        return None;
    }
    let name = b
        .as_ref()
        .and_then(|x| x.name.clone())
        .or_else(|| a.as_ref().and_then(|x| x.name.clone()));
    let temp_c = max_opt(
        a.as_ref().and_then(|x| x.temp_c),
        b.as_ref().and_then(|x| x.temp_c),
    );
    let power_w = max_opt(
        a.as_ref().and_then(|x| x.power_w),
        b.as_ref().and_then(|x| x.power_w),
    );
    Some(GpuPeak {
        name,
        temp_c,
        power_w,
    })
}

/// Compile `pattern` for `device`, run `n_warmup` un-timed iterations,
/// then `n_runs` timed iterations. Returns per-iteration ns + stats.
///
/// **Throttle gating**: callers running real benches on Apple Silicon
/// should invoke `scripts/check-throttle.sh` first — `rlx_ir::Tick`
/// measures wall-clock, so thermal throttling silently bloats every
/// sample.
pub fn run_benchmark<P: BenchmarkPattern>(
    pattern: &P,
    device: Device,
    n_warmup: usize,
    n_runs: usize,
) -> BenchResult {
    let graph = pattern.build_graph();
    let mut compiled = Session::new(device).compile(graph);

    for (name, data) in pattern.param_data() {
        compiled.set_param(&name, &data);
    }

    // Pre-build the input list once so warm-ups + timed runs share it.
    let inputs_owned = pattern.input_data();
    let inputs: Vec<(&str, &[f32])> = inputs_owned
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_slice()))
        .collect();

    // Warm-ups (let JIT / kernel cache / arena warm up).
    for _ in 0..n_warmup {
        let _ = compiled.run(&inputs);
    }

    // GPU thermal watchdog: sample the backing GPU just before and after
    // the timed loop (post-warm-up, so the "before" reading already
    // reflects a hot device). Read-only + outside the timed region.
    let gpu_before = gpu_bench_sample(device);
    let mut samples_ns = Vec::with_capacity(n_runs);
    let tick = Tick::now();
    for _ in 0..n_runs {
        let t0 = Tick::now();
        let _ = compiled.run(&inputs);
        let elapsed = Tick::now().elapsed_ns(t0);
        samples_ns.push(elapsed);
    }
    let _ = tick;
    let gpu_after = gpu_bench_sample(device);

    BenchResult {
        pattern: pattern.name().to_string(),
        tier: pattern.tier(),
        device,
        n_runs,
        samples_ns,
        gpu_peak: merge_gpu_peak(gpu_before, gpu_after),
    }
}

/// Like [`run_benchmark`] but sets `RLX_BENCH_DISPATCH_ONLY=1` so wgpu skips
/// output readback (measures dispatch + kernel time more closely).
pub fn run_benchmark_dispatch_only<P: BenchmarkPattern>(
    pattern: &P,
    device: Device,
    n_warmup: usize,
    n_runs: usize,
) -> BenchResult {
    // SAFETY: bench harness toggles process-local env for the duration of this call.
    unsafe {
        std::env::set_var("RLX_BENCH_DISPATCH_ONLY", "1");
    }
    let r = run_benchmark(pattern, device, n_warmup, n_runs);
    unsafe {
        std::env::remove_var("RLX_BENCH_DISPATCH_ONLY");
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::*;

    #[test]
    fn matmul_pattern_runs_on_cpu() {
        // CPU's NEON 4-wide matmul kernel hardcodes 8-row accumulators
        // (kernels.rs line ~414). Use m ≤ 8 to stay in the small-shape
        // dispatch path; larger shapes go through BLAS sgemm.
        let p = MatmulPattern { m: 8, k: 16, n: 8 };
        let r = run_benchmark(&p, Device::Cpu, 1, 3);
        assert_eq!(r.pattern, "matmul");
        assert_eq!(r.tier, Tier::L1);
        assert_eq!(r.device, Device::Cpu);
        assert_eq!(r.n_runs, 3);
        assert_eq!(r.samples_ns.len(), 3);
        // Every sample should be > 0 ns (the kernel actually executed).
        assert!(r.samples_ns.iter().all(|&v| v > 0));
    }

    #[test]
    fn layernorm_pattern_runs_on_cpu() {
        let p = LayerNormPattern {
            rows: 8,
            hidden: 16,
        };
        let r = run_benchmark(&p, Device::Cpu, 1, 3);
        assert_eq!(r.pattern, "layer_norm");
        assert_eq!(r.tier, Tier::L1);
        assert_eq!(r.samples_ns.len(), 3);
    }

    #[test]
    fn matmul_bias_relu_l2_pattern_runs_on_cpu() {
        let p = MatmulBiasReluPattern { m: 8, k: 16, n: 8 };
        let r = run_benchmark(&p, Device::Cpu, 1, 3);
        assert_eq!(r.pattern, "matmul_bias_relu");
        assert_eq!(r.tier, Tier::L2);
        assert_eq!(r.samples_ns.len(), 3);
    }

    #[test]
    fn bench_result_stats_are_sane() {
        let r = BenchResult {
            pattern: "x".into(),
            tier: Tier::L1,
            device: Device::Cpu,
            n_runs: 5,
            samples_ns: vec![100, 200, 300, 400, 500],
            gpu_peak: None,
        };
        assert_eq!(r.min_ns(), 100);
        assert_eq!(r.max_ns(), 500);
        assert_eq!(r.mean_ns(), 300);
        assert_eq!(r.median_ns(), 300);
    }

    #[test]
    fn bench_result_display_is_human_readable() {
        let r = BenchResult {
            pattern: "matmul".into(),
            tier: Tier::L1,
            device: Device::Cpu,
            n_runs: 2,
            samples_ns: vec![1000, 2000],
            gpu_peak: None,
        };
        let s = format!("{r}");
        assert!(s.contains("matmul"));
        assert!(s.contains("L1"));
        assert!(s.contains("Cpu"));
        assert!(s.contains("n=2"));
    }
}
