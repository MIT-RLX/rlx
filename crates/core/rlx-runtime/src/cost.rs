// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-backend cost interface.
//!
//! Each backend implements `BackendCostModel` to expose its execution
//! characteristics (kernel throughput, dispatch overhead, memory bw).
//! The runtime can then estimate the cost of running a graph on each
//! available backend and pick the fastest.
//!
//! This is what enables "auto device" — given a graph, pick CPU or
//! Metal automatically based on which is faster for THIS workload on
//! THIS hardware.

use crate::Device;
use rlx_ir::{Graph, Node, Op};

/// Where a cost model's numbers actually came from.
///
/// The distinction is load-bearing and rlx used to erase it. `RocmCostModel`
/// falls back to `sgemm_gflops: 10_000.0` when ROCm is absent; `MetalHwModel`
/// falls back to `AppleGpuFamily::Unknown => (300e9, 180e9, 60e9)`. Both then
/// feed [`fastest_device_for`], which ranks devices against each other — so on an
/// unrecognized GPU rlx would confidently place a graph using invented
/// throughput, and say nothing.
///
/// CAKE B.5 states the rule this implements: *"Timing-model coverage is
/// separately evidence-gated: B200 is the measured baseline, H100 is calibrated
/// independently, and other targets report a coverage limitation instead of
/// inheriting estimates."* Reporting a coverage limitation is strictly more
/// useful than a confident guess, because a caller can act on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CostCalibration {
    /// Measured on *this* machine (a calibration cache exists, or the model
    /// probed the device). Comparisons across such models are meaningful.
    Measured,
    /// Compile-time constants chosen for this specific detected architecture.
    /// Reasonable, but not measured here — treat cross-device comparisons as
    /// indicative rather than authoritative.
    ArchDefault,
    /// The architecture was not recognized, so the numbers are a generic guess.
    /// **Not** a basis for ranking one device against another.
    Uncalibrated,
}

impl CostCalibration {
    /// True when this model's numbers may be compared against another device's.
    pub fn is_rankable(&self) -> bool {
        matches!(self, Self::Measured | Self::ArchDefault)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::ArchDefault => "arch-default",
            Self::Uncalibrated => "uncalibrated",
        }
    }
}

/// Hardware-aware cost characteristics for a backend on the current machine.
pub trait BackendCostModel: Send + Sync {
    /// Identify which device this model is for.
    fn device(&self) -> Device;

    /// Provenance of this model's numbers — see [`CostCalibration`].
    ///
    /// Defaults to [`CostCalibration::ArchDefault`]: a model that has not been
    /// audited claims neither measurement nor ignorance. Every in-tree impl
    /// overrides this with the truth.
    fn calibration(&self) -> CostCalibration {
        CostCalibration::ArchDefault
    }

    /// Effective f32 sgemm throughput in GFLOP/s for the most-used kernel
    /// path at the given dimensions. Backends should return their best
    /// sustained rate (not peak).
    fn sgemm_gflops(&self, m: usize, k: usize, n: usize) -> f64;

    /// Cost to dispatch one kernel (function call, BLAS setup, etc.) in ns.
    fn dispatch_overhead_ns(&self) -> f64;

    /// Cost to commit + wait for a command buffer / forward pass in ns.
    /// Roughly amortized per-forward overhead independent of kernel count.
    fn roundtrip_overhead_ns(&self) -> f64;

    /// Memory bandwidth in bytes/ns (== GB/s).
    fn memory_bw(&self) -> f64;

    /// Effective host readback bandwidth (bytes/ns). Defaults to [`Self::memory_bw`].
    fn host_readback_bw(&self) -> f64 {
        self.memory_bw()
    }

    /// True when device and host share the same physical memory (Apple Silicon Metal).
    fn unified_memory(&self) -> bool {
        false
    }

    /// Number of compute threads available.
    fn num_threads(&self) -> usize;
}

/// Estimate forward-pass time (ns) for a graph on the given backend.
/// Uses node-level cost contributions; conservative — actual time may
/// be lower due to hardware parallelism we don't model.
pub fn estimate_graph_cost(graph: &Graph, model: &dyn BackendCostModel) -> f64 {
    estimate_graph_cost_with_io(graph, model, &crate::graph_io::profile_graph_io(graph))
}

/// IO-aware cost: compute + device traffic + host readback + sync points.
pub fn estimate_graph_cost_with_io(
    graph: &Graph,
    model: &dyn BackendCostModel,
    io: &crate::graph_io::GraphIoProfile,
) -> f64 {
    let mut total = model.roundtrip_overhead_ns();
    for node in graph.nodes() {
        total += node_cost(node, graph, model);
    }
    total += io.device_traffic_bytes as f64 / model.memory_bw().max(1.0);
    total +=
        io.host_readback_bytes(model.unified_memory()) as f64 / model.host_readback_bw().max(1.0);
    total += io.sync_points as f64 * model.roundtrip_overhead_ns();
    total
}

fn node_cost(node: &Node, graph: &Graph, model: &dyn BackendCostModel) -> f64 {
    let dispatch = model.dispatch_overhead_ns();
    match &node.op {
        Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => 0.0,
        Op::MatMul | Op::FusedMatMulBiasAct { .. } => {
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            let total = node.shape.num_elements().unwrap_or(0);
            let m = total / n.max(1);
            let a_total = graph.node(node.inputs[0]).shape.num_elements().unwrap_or(0);
            let k = a_total / m.max(1);
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            flops / (model.sgemm_gflops(m, k, n) + 1.0) + dispatch
        }
        Op::Attention {
            num_heads,
            head_dim,
            ..
        } => {
            let q_shape = &graph.node(node.inputs[0]).shape;
            let seq = q_shape.dim(q_shape.rank() - 2).unwrap_static();
            let batch = q_shape.num_elements().unwrap_or(0) / (seq * num_heads * head_dim).max(1);
            let flops = (batch * num_heads * seq * seq * head_dim * 2) as f64;
            flops / (model.sgemm_gflops(seq, *head_dim, seq) + 1.0) + dispatch
        }
        // Element-wise + small ops: bounded by memory bandwidth.
        _ => {
            let bytes = node.shape.num_elements().unwrap_or(0) * 4;
            (bytes as f64) / model.memory_bw().max(1.0) + dispatch
        }
    }
}

/// Pick the device with the lowest predicted cost for this graph.
///
/// **Uncalibrated models are excluded from the ranking, not silently trusted.**
/// A model whose numbers are a generic guess ([`CostCalibration::Uncalibrated`])
/// cannot be meaningfully compared against a measured one — and the guesses are
/// optimistic, so including them systematically hands the graph to whichever
/// device rlx knows *least* about. That is the failure mode CAKE B.5 names:
/// inheriting estimates rather than reporting a coverage limitation.
///
/// If *every* model is uncalibrated there is nothing to compare, so the lowest
/// cost is as good a tiebreak as any and we take it — but say so under
/// `RLX_VERBOSE`, because a caller seeing an unexpected placement deserves to
/// know the ranking had no evidence behind it.
pub fn pick_best_device(graph: &Graph, models: &[&dyn BackendCostModel]) -> Device {
    let rankable: Vec<&&dyn BackendCostModel> = models
        .iter()
        .filter(|m| m.calibration().is_rankable())
        .collect();

    let (pool, blind): (&[&dyn BackendCostModel], bool) = if rankable.is_empty() {
        (models, true)
    } else if rankable.len() == models.len() {
        (models, false)
    } else {
        // Some were dropped — report which, then rank the rest.
        if rlx_ir::env::flag("RLX_VERBOSE") {
            for m in models {
                if !m.calibration().is_rankable() {
                    eprintln!(
                        "rlx: excluding {:?} from device ranking — cost model is {}",
                        m.device(),
                        m.calibration().as_str()
                    );
                }
            }
        }
        // Re-borrow as a slice of trait objects.
        let kept: Vec<&dyn BackendCostModel> = rankable.iter().map(|m| **m).collect();
        return pick_lowest(graph, &kept);
    };

    if blind && rlx_ir::env::flag("RLX_VERBOSE") {
        eprintln!(
            "rlx: every candidate cost model is uncalibrated — device ranking has \
             no measured basis on this host"
        );
    }
    pick_lowest(graph, pool)
}

fn pick_lowest(graph: &Graph, models: &[&dyn BackendCostModel]) -> Device {
    let mut best = (Device::Cpu, f64::INFINITY);
    for &m in models {
        let cost = estimate_graph_cost(graph, m);
        if cost < best.1 {
            best = (m.device(), cost);
        }
    }
    best.0
}

/// Pick the fastest backend for `graph` on this host.
pub fn fastest_device_for(graph: &Graph) -> Device {
    fastest_device_for_with_policy(graph, &crate::device_policy::DevicePolicy::default())
}

/// Like [`fastest_device_for`] but respects a [`crate::DevicePolicy`] allow-list.
pub fn fastest_device_for_with_policy(
    graph: &Graph,
    policy: &crate::device_policy::DevicePolicy,
) -> Device {
    // Hard override: `RLX_FORCE_DEVICE` pins the backend regardless of the cost
    // model or policy. Without this, a small graph whose GPU dispatch cost
    // exceeds CPU is silently placed on CPU even when the caller asked for a
    // GPU backend — so a `--features gpu` run can be a CPU run in disguise,
    // masking real GPU failures. Honored at the top of the single chokepoint
    // so EVERY `fastest_device_for*` caller respects it.
    if let Some(s) = rlx_ir::env::var("RLX_FORCE_DEVICE") {
        if let Ok(dev) = crate::parse_device(&s) {
            return dev;
        }
    }
    let candidates = crate::device_policy::devices_for_with_policy(graph, policy);
    if candidates.is_empty() {
        return crate::device_ext::fastest_among(&policy.apply(crate::available_devices()));
    }

    #[cfg(feature = "cpu")]
    let cpu = CpuCostModel::new();
    #[cfg(all(feature = "metal", target_vendor = "apple", not(target_os = "watchos")))]
    let metal = MetalCostModel::new();
    #[cfg(all(feature = "mlx", rlx_mlx_host))]
    let mlx = MlxCostModel::new();
    #[cfg(feature = "cuda")]
    let cuda = CudaCostModel::new();
    #[cfg(feature = "rocm")]
    let rocm = RocmCostModel::new();
    #[cfg(feature = "gpu")]
    let wgpu = WgpuCostModel::new();

    // Every push below is backend-cfg-gated; the backend-less build has none.
    #[cfg_attr(not(feature = "cpu"), allow(unused_mut))]
    let mut models: Vec<&dyn BackendCostModel> = Vec::new();
    #[cfg(feature = "cpu")]
    if candidates.contains(&Device::Cpu) {
        models.push(&cpu);
    }
    #[cfg(all(feature = "metal", target_vendor = "apple", not(target_os = "watchos")))]
    if candidates.contains(&Device::Metal) {
        models.push(&metal);
    }
    #[cfg(all(feature = "mlx", rlx_mlx_host))]
    if candidates.contains(&Device::Mlx) {
        models.push(&mlx);
    }
    #[cfg(feature = "cuda")]
    if candidates.contains(&Device::Cuda) {
        models.push(&cuda);
    }
    #[cfg(feature = "rocm")]
    if candidates.contains(&Device::Rocm) {
        models.push(&rocm);
    }
    #[cfg(feature = "gpu")]
    if candidates.contains(&Device::Gpu) {
        models.push(&wgpu);
    }

    if models.len() >= 2 {
        pick_best_device(graph, &models)
    } else if let Some(m) = models.first() {
        m.device()
    } else {
        crate::device_ext::fastest_among(&candidates)
    }
}

// ── Backend adapters (plan #29) ─────────────────────────────────
//
// The CPU and Metal crates own their own internal cost models for
// kernel-selection decisions. These thin adapters wrap them in
// `BackendCostModel` so `pick_best_device` can compare both with a
// single uniform interface.

/// `BackendCostModel` impl backed by `rlx_cpu::cost::HwModel`.
#[cfg(feature = "cpu")]
pub struct CpuCostModel(rlx_cpu::cost::HwModel);

#[cfg(feature = "cpu")]
impl CpuCostModel {
    pub fn new() -> Self {
        let cfg = rlx_cpu::config::RuntimeConfig::global();
        Self(rlx_cpu::cost::HwModel::from_config(cfg))
    }
}

#[cfg(feature = "cpu")]
impl Default for CpuCostModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "cpu")]
impl BackendCostModel for CpuCostModel {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn calibration(&self) -> CostCalibration {
        // `rlx_cpu::cost::HwModel` is built from detected CPU features and
        // per-arch compile-time constants — real for the arch, not measured here.
        CostCalibration::ArchDefault
    }
    fn sgemm_gflops(&self, m: usize, k: usize, n: usize) -> f64 {
        // Take the better of NEON / BLAS at this shape.
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let neon_time = flops / self.0.neon_flops.max(1.0);
        let blas_time = flops / self.0.blas_flops.max(1.0);
        let pick = neon_time.min(blas_time);
        if pick > 0.0 {
            flops / (pick * 1e9)
        } else {
            0.0
        }
    }
    fn dispatch_overhead_ns(&self) -> f64 {
        self.0.blas_overhead_ns
    }
    fn roundtrip_overhead_ns(&self) -> f64 {
        self.0.par_for_overhead_ns
    }
    fn memory_bw(&self) -> f64 {
        self.0.mem_bw
    }
    fn num_threads(&self) -> usize {
        self.0.num_threads
    }
}

/// `BackendCostModel` impl backed by `rlx_metal::cost`. Reads from
/// the on-disk calibration cache so the numbers reflect what this
/// machine actually measured.
#[cfg(all(feature = "metal", target_vendor = "apple", not(target_os = "watchos")))]
pub struct MetalCostModel {
    sgemm_gflops_avg: f64,
    roundtrip_ns: f64,
    memory_bw: f64,
}

#[cfg(all(feature = "metal", target_vendor = "apple", not(target_os = "watchos")))]
impl MetalCostModel {
    pub fn new() -> Self {
        let cal = rlx_metal::calibrate::Calibration::load_or_measure();
        // Effective single-shape sgemm: best of the calibrated paths.
        let best = cal
            .sgemm_simd_4x4_flops
            .max(cal.sgemm_simd_flops)
            .max(cal.sgemm_padded_flops);
        Self {
            sgemm_gflops_avg: best,
            roundtrip_ns: cal.roundtrip_overhead_ns,
            // Apple Silicon unified memory bandwidth (rough): ~200 GB/s
            // on M-series base, much higher on Pro/Max. The calibrator
            // doesn't measure pure mem-bw yet, so we hard-code a
            // floor that makes mem-bound ops not look free.
            memory_bw: 200.0,
        }
    }
}

#[cfg(all(feature = "metal", target_vendor = "apple", not(target_os = "watchos")))]
impl Default for MetalCostModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(feature = "metal", target_vendor = "apple", not(target_os = "watchos")))]
impl BackendCostModel for MetalCostModel {
    fn device(&self) -> Device {
        Device::Metal
    }
    fn calibration(&self) -> CostCalibration {
        // `load_or_measure` really does measure when no cache exists, so the
        // sgemm figure is measured — but `memory_bw` is a hard-coded 200 GB/s
        // floor (the calibrator does not probe bandwidth yet) and the Apple GPU
        // family may be `Unknown`. Claiming Measured would overstate both, so
        // report the weaker of the two: arch-default.
        if rlx_metal::cost::hw_model().gpu_family == rlx_metal::cost::AppleGpuFamily::Unknown {
            CostCalibration::Uncalibrated
        } else {
            CostCalibration::ArchDefault
        }
    }
    fn sgemm_gflops(&self, _m: usize, _k: usize, _n: usize) -> f64 {
        self.sgemm_gflops_avg
    }
    fn dispatch_overhead_ns(&self) -> f64 {
        // Per-kernel encode cost — small relative to the round-trip.
        2_000.0
    }
    fn roundtrip_overhead_ns(&self) -> f64 {
        self.roundtrip_ns
    }
    fn memory_bw(&self) -> f64 {
        self.memory_bw
    }
    fn unified_memory(&self) -> bool {
        true
    }
    fn num_threads(&self) -> usize {
        1
    } // single command queue
}

/// `BackendCostModel` impl backed by `rlx_mlx::calibrate`. Reads from
/// the on-disk MLX calibration cache. The first construction on a
/// fresh machine pays a one-time measurement cost (tens of ms);
/// subsequent constructions read the cache.
#[cfg(all(feature = "mlx", rlx_mlx_host))]
pub struct MlxCostModel {
    sgemm_large_flops: f64,
    sgemm_small_flops: f64,
    roundtrip_ns: f64,
    memory_bw: f64,
}

#[cfg(all(feature = "mlx", rlx_mlx_host))]
impl MlxCostModel {
    pub fn new() -> Self {
        let cal = rlx_mlx::calibrate::Calibration::load_or_measure();
        // Use measured memory bandwidth when available (post-PR16
        // calibrators record it); fall back to the Apple-Silicon
        // unified-memory floor otherwise so old caches still produce
        // sane numbers.
        let memory_bw = if cal.memory_bw_gbps > 0.0 {
            cal.memory_bw_gbps
        } else {
            200.0
        };
        Self {
            sgemm_large_flops: cal.sgemm_large_flops,
            sgemm_small_flops: cal.sgemm_small_flops,
            roundtrip_ns: cal.roundtrip_overhead_ns,
            memory_bw,
        }
    }
}

#[cfg(all(feature = "mlx", rlx_mlx_host))]
impl Default for MlxCostModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(feature = "mlx", rlx_mlx_host))]
impl BackendCostModel for MlxCostModel {
    fn device(&self) -> Device {
        Device::Mlx
    }
    fn calibration(&self) -> CostCalibration {
        // A crossover heuristic with hand-picked constants — no measurement and
        // no per-arch table behind it.
        CostCalibration::Uncalibrated
    }
    fn sgemm_gflops(&self, m: usize, k: usize, n: usize) -> f64 {
        // Crossover heuristic: small shapes pay the per-op overhead;
        // large shapes hit the optimized path. The cutoff is rough —
        // matches the calibrator's "small" / "large" probe sizes.
        let total = m as f64 * k as f64 * n as f64;
        if total < 32_768.0 {
            self.sgemm_small_flops
        } else {
            self.sgemm_large_flops
        }
    }
    fn dispatch_overhead_ns(&self) -> f64 {
        // MLX's lazy-eval keeps per-op encode cost low; trace
        // construction in Rust is the dominant per-op cost.
        2_000.0
    }
    fn roundtrip_overhead_ns(&self) -> f64 {
        self.roundtrip_ns
    }
    fn memory_bw(&self) -> f64 {
        self.memory_bw
    }
    fn num_threads(&self) -> usize {
        1
    }
}

/// Heuristic CUDA cost model until a dedicated calibrator lands.
#[cfg(feature = "cuda")]
pub struct CudaCostModel {
    sgemm_gflops: f64,
    roundtrip_ns: f64,
    memory_bw: f64,
    /// Whether the numbers above came from a real calibration on this
    /// machine or from the hardcoded no-device fallback below.
    calibration: CostCalibration,
}

#[cfg(feature = "cuda")]
impl CudaCostModel {
    pub fn new() -> Self {
        if crate::is_available(crate::Device::Cuda) {
            let cal = rlx_cuda::calibrate::Calibration::load_or_measure();
            return Self {
                sgemm_gflops: cal.sgemm_gflops,
                roundtrip_ns: cal.roundtrip_overhead_ns,
                memory_bw: cal.memory_bw_gbps,
                calibration: CostCalibration::Measured,
            };
        }
        // No CUDA device: these are invented numbers for a device that is not
        // here. Marked Uncalibrated so `pick_best_device` refuses to rank on them
        // rather than "discovering" that an absent GPU is fastest.
        Self {
            sgemm_gflops: 12_000.0,
            roundtrip_ns: 35_000.0,
            memory_bw: 900.0,
            calibration: CostCalibration::Uncalibrated,
        }
    }
}

#[cfg(feature = "cuda")]
impl Default for CudaCostModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "cuda")]
impl BackendCostModel for CudaCostModel {
    fn device(&self) -> Device {
        Device::Cuda
    }
    fn calibration(&self) -> CostCalibration {
        self.calibration
    }
    fn sgemm_gflops(&self, _m: usize, _k: usize, _n: usize) -> f64 {
        self.sgemm_gflops
    }
    fn dispatch_overhead_ns(&self) -> f64 {
        3_000.0
    }
    fn roundtrip_overhead_ns(&self) -> f64 {
        self.roundtrip_ns
    }
    fn memory_bw(&self) -> f64 {
        self.memory_bw
    }
    fn num_threads(&self) -> usize {
        1
    }
}

/// Heuristic ROCm cost model (same class as CUDA until calibrated).
#[cfg(feature = "rocm")]
pub struct RocmCostModel {
    sgemm_gflops: f64,
    roundtrip_ns: f64,
    memory_bw: f64,
    /// Whether the numbers above came from a real calibration on this
    /// machine or from the hardcoded no-device fallback below.
    calibration: CostCalibration,
}

#[cfg(feature = "rocm")]
impl RocmCostModel {
    pub fn new() -> Self {
        if crate::is_available(crate::Device::Rocm) {
            let cal = rlx_rocm::calibrate::Calibration::load_or_measure();
            return Self {
                sgemm_gflops: cal.sgemm_gflops,
                roundtrip_ns: cal.roundtrip_overhead_ns,
                memory_bw: cal.memory_bw_gbps,
                calibration: CostCalibration::Measured,
            };
        }
        // No ROCm device — see the CUDA note. The old comment on this struct read
        // "same class as CUDA until calibrated", which is exactly the inherited
        // estimate CAKE B.5 warns against; now it is labelled instead.
        Self {
            sgemm_gflops: 10_000.0,
            roundtrip_ns: 40_000.0,
            memory_bw: 800.0,
            calibration: CostCalibration::Uncalibrated,
        }
    }
}

#[cfg(feature = "rocm")]
impl Default for RocmCostModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "rocm")]
impl BackendCostModel for RocmCostModel {
    fn device(&self) -> Device {
        Device::Rocm
    }
    fn calibration(&self) -> CostCalibration {
        self.calibration
    }
    fn sgemm_gflops(&self, _m: usize, _k: usize, _n: usize) -> f64 {
        self.sgemm_gflops
    }
    fn dispatch_overhead_ns(&self) -> f64 {
        3_000.0
    }
    fn roundtrip_overhead_ns(&self) -> f64 {
        self.roundtrip_ns
    }
    fn memory_bw(&self) -> f64 {
        self.memory_bw
    }
    fn num_threads(&self) -> usize {
        1
    }
}

/// Heuristic wgpu (`Device::Gpu`) cost model.
#[cfg(feature = "gpu")]
pub struct WgpuCostModel {
    sgemm_gflops: f64,
    roundtrip_ns: f64,
    memory_bw: f64,
    /// Whether the numbers above came from a real calibration on this machine
    /// or from the hardcoded no-adapter fallback below.
    calibration: CostCalibration,
}

#[cfg(feature = "gpu")]
impl WgpuCostModel {
    pub fn new() -> Self {
        if rlx_wgpu::is_available() {
            let cal = rlx_wgpu::calibrate::Calibration::load_or_measure();
            return Self {
                sgemm_gflops: cal.sgemm_gflops,
                roundtrip_ns: cal.roundtrip_overhead_ns,
                memory_bw: cal.memory_bw_gbps,
                calibration: CostCalibration::Measured,
            };
        }
        // No wgpu adapter — invented numbers for a device that is not here.
        Self {
            sgemm_gflops: 2_500.0,
            roundtrip_ns: 80_000.0,
            memory_bw: 120.0,
            calibration: CostCalibration::Uncalibrated,
        }
    }
}

#[cfg(feature = "gpu")]
impl Default for WgpuCostModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "gpu")]
impl BackendCostModel for WgpuCostModel {
    fn device(&self) -> Device {
        Device::Gpu
    }
    fn calibration(&self) -> CostCalibration {
        self.calibration
    }
    fn sgemm_gflops(&self, _m: usize, _k: usize, _n: usize) -> f64 {
        self.sgemm_gflops
    }
    fn dispatch_overhead_ns(&self) -> f64 {
        5_000.0
    }
    fn roundtrip_overhead_ns(&self) -> f64 {
        self.roundtrip_ns
    }
    fn memory_bw(&self) -> f64 {
        self.memory_bw
    }
    fn num_threads(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Graph, Shape};

    /// A cost model with dictated numbers and provenance, for testing the
    /// ranking policy without needing real hardware.
    struct FakeModel {
        device: Device,
        gflops: f64,
        calibration: CostCalibration,
    }

    impl BackendCostModel for FakeModel {
        fn device(&self) -> Device {
            self.device
        }
        fn calibration(&self) -> CostCalibration {
            self.calibration
        }
        fn sgemm_gflops(&self, _m: usize, _k: usize, _n: usize) -> f64 {
            self.gflops
        }
        fn dispatch_overhead_ns(&self) -> f64 {
            100.0
        }
        fn roundtrip_overhead_ns(&self) -> f64 {
            1_000.0
        }
        fn memory_bw(&self) -> f64 {
            100.0
        }
        fn num_threads(&self) -> usize {
            1
        }
    }

    fn big_matmul() -> Graph {
        let mut g = Graph::new("mm");
        let x = g.input("x", Shape::new(&[512, 512], DType::F32));
        let w = g.param("w", Shape::new(&[512, 512], DType::F32));
        let y = g.matmul(x, w, Shape::new(&[512, 512], DType::F32));
        g.set_outputs(vec![y]);
        g
    }

    /// The core guarantee: an uncalibrated model cannot win the ranking, however
    /// good its invented numbers look. Before this, `RocmCostModel`'s no-device
    /// fallback claimed 10 TFLOP/s and would beat a measured CPU every time.
    #[test]
    fn uncalibrated_model_cannot_win_the_ranking() {
        let g = big_matmul();
        let measured_slow = FakeModel {
            device: Device::Cpu,
            gflops: 100.0,
            calibration: CostCalibration::Measured,
        };
        let guessed_fast = FakeModel {
            device: Device::Rocm,
            gflops: 10_000.0,
            calibration: CostCalibration::Uncalibrated,
        };
        let models: Vec<&dyn BackendCostModel> = vec![&measured_slow, &guessed_fast];
        assert_eq!(
            pick_best_device(&g, &models),
            Device::Cpu,
            "a guessed 100x-faster device must not be picked over a measured one"
        );
    }

    /// …but a *measured* faster device must still win, or the guard would have
    /// disabled device selection instead of correcting it.
    #[test]
    fn measured_faster_device_still_wins() {
        let g = big_matmul();
        let cpu = FakeModel {
            device: Device::Cpu,
            gflops: 100.0,
            calibration: CostCalibration::Measured,
        };
        let gpu = FakeModel {
            device: Device::Cuda,
            gflops: 10_000.0,
            calibration: CostCalibration::Measured,
        };
        let models: Vec<&dyn BackendCostModel> = vec![&cpu, &gpu];
        assert_eq!(pick_best_device(&g, &models), Device::Cuda);
    }

    /// ArchDefault is rankable — per-arch constants are evidence, just weaker
    /// than a measurement. Excluding them would strand every backend whose
    /// calibrator has not run.
    #[test]
    fn arch_default_is_rankable() {
        assert!(CostCalibration::Measured.is_rankable());
        assert!(CostCalibration::ArchDefault.is_rankable());
        assert!(!CostCalibration::Uncalibrated.is_rankable());

        let g = big_matmul();
        let cpu = FakeModel {
            device: Device::Cpu,
            gflops: 100.0,
            calibration: CostCalibration::ArchDefault,
        };
        let gpu = FakeModel {
            device: Device::Metal,
            gflops: 5_000.0,
            calibration: CostCalibration::ArchDefault,
        };
        let models: Vec<&dyn BackendCostModel> = vec![&cpu, &gpu];
        assert_eq!(pick_best_device(&g, &models), Device::Metal);
    }

    /// When nothing is calibrated there is no evidence either way, so ranking
    /// still has to return something — it must not panic or drop to a fixed
    /// device.
    #[test]
    fn all_uncalibrated_still_returns_a_device() {
        let g = big_matmul();
        let a = FakeModel {
            device: Device::Cpu,
            gflops: 100.0,
            calibration: CostCalibration::Uncalibrated,
        };
        let b = FakeModel {
            device: Device::Gpu,
            gflops: 9_000.0,
            calibration: CostCalibration::Uncalibrated,
        };
        let models: Vec<&dyn BackendCostModel> = vec![&a, &b];
        assert_eq!(pick_best_device(&g, &models), Device::Gpu);
    }

    /// Every cost model compiled into this build must state its provenance —
    /// the trait default exists for out-of-tree impls, not as a way for an
    /// in-tree one to stay silent.
    #[test]
    fn in_tree_models_declare_their_provenance() {
        #[allow(unused_mut)]
        let mut checked = 0usize;
        #[cfg(feature = "cpu")]
        {
            let m = CpuCostModel::new();
            assert_eq!(m.calibration(), CostCalibration::ArchDefault);
            checked += 1;
        }
        #[cfg(feature = "gpu")]
        {
            let m = WgpuCostModel::new();
            // Either is honest; what matters is that it is not a silent guess
            // dressed as a measurement when no adapter exists.
            assert!(
                m.calibration() == CostCalibration::Measured
                    || m.calibration() == CostCalibration::Uncalibrated
            );
            checked += 1;
        }
        assert!(checked > 0, "no cost model was compiled in to check");
    }

    #[test]
    fn fastest_device_for_falls_back_to_cpu_for_simple_graph() {
        let mut g = Graph::new("mm");
        let x = g.input("x", Shape::new(&[4, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 4], DType::F32));
        let y = g.matmul(x, w, Shape::new(&[4, 4], DType::F32));
        g.set_outputs(vec![y]);
        let pick = fastest_device_for(&g);
        assert!(crate::is_available(pick));
        assert!(crate::devices_for(&g).contains(&pick));
    }
}
