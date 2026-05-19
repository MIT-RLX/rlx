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
use rlx_runtime::Session;

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
        )
    }
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

    let mut samples_ns = Vec::with_capacity(n_runs);
    let tick = Tick::now();
    for _ in 0..n_runs {
        let t0 = Tick::now();
        let _ = compiled.run(&inputs);
        let elapsed = Tick::now().elapsed_ns(t0);
        samples_ns.push(elapsed);
    }
    let _ = tick;

    BenchResult {
        pattern: pattern.name().to_string(),
        tier: pattern.tier(),
        device,
        n_runs,
        samples_ns,
    }
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
        };
        let s = format!("{r}");
        assert!(s.contains("matmul"));
        assert!(s.contains("L1"));
        assert!(s.contains("Cpu"));
        assert!(s.contains("n=2"));
    }
}
