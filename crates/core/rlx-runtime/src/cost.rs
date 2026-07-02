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

/// Hardware-aware cost characteristics for a backend on the current machine.
pub trait BackendCostModel: Send + Sync {
    /// Identify which device this model is for.
    fn device(&self) -> Device;

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
pub fn pick_best_device(graph: &Graph, models: &[&dyn BackendCostModel]) -> Device {
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
            };
        }
        Self {
            sgemm_gflops: 12_000.0,
            roundtrip_ns: 35_000.0,
            memory_bw: 900.0,
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
            };
        }
        Self {
            sgemm_gflops: 10_000.0,
            roundtrip_ns: 40_000.0,
            memory_bw: 800.0,
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
            };
        }
        Self {
            sgemm_gflops: 2_500.0,
            roundtrip_ns: 80_000.0,
            memory_bw: 120.0,
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
