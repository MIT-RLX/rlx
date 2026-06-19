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

//! Cost model — estimates execution time for kernel dispatch decisions.
//!
//! Instead of hardcoded thresholds scattered across thunk.rs, all dispatch
//! decisions are routed through this cost model. The model considers:
//! - Hardware: cache sizes, core count, AMX availability, NEON throughput
//! - Workload: matrix dimensions, batch size, sequence length
//! - Strategy: BLAS vs NEON, parallel vs sequential, fused vs individual
//!
//! The model is calibrated at compile time (platform-specific constants)
//! and refined at runtime (detected hardware + optional autotune).

use crate::config::RuntimeConfig;

/// Estimated cost in nanoseconds for a kernel execution strategy.
#[derive(Debug, Clone, Copy)]
pub struct Cost(pub f64);

impl Cost {
    pub fn ns(self) -> f64 {
        self.0
    }
}

/// Hardware model — derived from RuntimeConfig + platform detection.
pub struct HwModel {
    /// NEON throughput: FLOP/s for element-wise (FMA chains)
    pub neon_flops: f64,
    /// BLAS throughput: FLOP/s for sgemm (AMX or optimized NEON)
    pub blas_flops: f64,
    /// BLAS call overhead in nanoseconds (function call + AMX sync)
    pub blas_overhead_ns: f64,
    /// par_for dispatch overhead in nanoseconds
    pub par_for_overhead_ns: f64,
    /// L1 data cache size in bytes
    pub l1_bytes: usize,
    /// L2 cache size in bytes
    pub l2_bytes: usize,
    /// Memory bandwidth (L2 → registers) in bytes/ns
    pub mem_bw: f64,
    /// Number of worker threads
    pub num_threads: usize,
}

impl HwModel {
    /// Build from runtime config and platform defaults.
    pub fn from_config(cfg: &RuntimeConfig) -> Self {
        // Platform-calibrated constants
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        let model = HwModel {
            neon_flops: 72e9,            // ~72 GFLOP/s NEON FMA throughput (M4 Pro P-core)
            blas_flops: 1000e9,          // ~1 TFLOP/s AMX peak (effective varies with tile fill)
            blas_overhead_ns: 500.0,     // ~0.5µs per cblas_sgemm call
            par_for_overhead_ns: 5000.0, // ~5µs spin-wait dispatch
            l1_bytes: 65536,             // 64KB L1d (refined by sysctl at runtime)
            l2_bytes: 4 * 1024 * 1024,   // 4MB L2 per core
            mem_bw: 50.0,                // ~50 GB/s = 50 B/ns
            num_threads: cfg.pool_workers + 1,
        };

        #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
        let model = HwModel {
            neon_flops: 32e9,
            blas_flops: 200e9,
            blas_overhead_ns: 300.0,
            par_for_overhead_ns: 3000.0,
            l1_bytes: 32768,
            l2_bytes: 1024 * 1024,
            mem_bw: 30.0,
            num_threads: cfg.pool_workers + 1,
        };

        model
    }

    // ── Dispatch decisions ──────────────────────────────────────────

    /// Should we use NEON sgemm instead of BLAS for this matrix multiply?
    /// Returns true when BLAS overhead dominates the compute.
    pub fn prefer_neon_sgemm(&self, m: usize, k: usize, n: usize) -> bool {
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let blas_time = flops / self.blas_flops + self.blas_overhead_ns * 1e-9;
        let neon_time = flops / self.neon_flops;
        neon_time < blas_time
    }

    /// Should we use par_for for this element-wise operation?
    /// Returns true when parallelism benefit exceeds dispatch overhead.
    pub fn prefer_parallel(&self, total_elements: usize, cost_per_element_ns: f64) -> bool {
        let seq_time = total_elements as f64 * cost_per_element_ns;
        let par_time = seq_time / self.num_threads as f64 + self.par_for_overhead_ns;
        par_time < seq_time
    }

    /// Should we use strided BLAS for SDPA, or sequential NEON dots?
    pub fn prefer_blas_sdpa(
        &self,
        batch: usize,
        seq: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> bool {
        let total_heads = batch * num_heads;
        // Two sgemm per head (Q@K^T + scores@V)
        let per_head_flops = 2.0 * seq as f64 * seq as f64 * head_dim as f64 * 2.0;
        let blas_per_head = per_head_flops / self.blas_flops + 2.0 * self.blas_overhead_ns * 1e-9;
        let neon_per_head = per_head_flops / self.neon_flops;

        // With par_for, BLAS heads run in parallel
        let blas_total = blas_per_head * total_heads as f64 / self.num_threads as f64
            + self.par_for_overhead_ns * 1e-9;
        let neon_total = neon_per_head * total_heads as f64; // sequential

        blas_total < neon_total
    }

    /// Should we fuse the entire transformer layer into one thunk?
    /// True when intermediates fit in L1 and per-thunk overhead dominates.
    pub fn prefer_fused_layer(
        &self,
        batch: usize,
        seq: usize,
        hidden: usize,
        intermediate: usize,
    ) -> bool {
        let m = batch * seq;
        // Estimate intermediate buffer sizes
        let qkv_bytes = m * 3 * hidden * 4;
        let attn_bytes = m * hidden * 4;
        let ffn_bytes = m * intermediate * 4;
        let total_bytes = qkv_bytes + 2 * attn_bytes + ffn_bytes;
        // Fuse if total intermediates fit in L2 (L1 would be ideal but tight)
        total_bytes <= self.l2_bytes / 2
    }
}

/// Global hardware model singleton.
pub fn hw_model() -> &'static HwModel {
    use std::sync::OnceLock;
    static MODEL: OnceLock<HwModel> = OnceLock::new();
    MODEL.get_or_init(|| HwModel::from_config(RuntimeConfig::global()))
}
