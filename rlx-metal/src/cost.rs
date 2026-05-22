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

//! Metal cost model — analytical kernel selection for GPU.
//!
//! Mirrors rlx-cpu/src/cost.rs. Centralizes all dispatch decisions so
//! kernel selection is data-driven (hardware specs + matrix dims) rather
//! than scattered hardcoded thresholds.

use crate::device::metal_device;
use std::sync::OnceLock;

/// Apple GPU family — different memory bandwidth and tensor unit characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppleGpuFamily {
    Unknown,
    M1,    // M1 (8-core GPU baseline)
    M1Pro, // M1 Pro/Max (16-32 core)
    M2,    // M2 family
    M3,    // M3 family — added dynamic caching
    M4,    // M4 family — improved tensor units
}

impl AppleGpuFamily {
    fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("m4") {
            Self::M4
        } else if lower.contains("m3") {
            Self::M3
        } else if lower.contains("m2") {
            Self::M2
        } else if lower.contains("m1 pro") || lower.contains("m1 max") || lower.contains("m1 ultra")
        {
            Self::M1Pro
        } else if lower.contains("m1") {
            Self::M1
        } else {
            Self::Unknown
        }
    }
}

/// Variant picked by the cost model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgemmVariant {
    /// MPSMatrixMultiplication — Apple's per-chip-tuned matmul. Wins for
    /// large matmuls (M·K·N above ~16 MFLOPs) where the ~5–20µs objc
    /// bridging cost amortizes against compute time.
    Mps,
    /// 32×32 output per threadgroup; 16 simdgroups cooperate via threadgroup memory.
    /// Best throughput for our hand-rolled path. Requires M%32==K%32==N%32==0.
    Simd4x4,
    /// 8×8 output per threadgroup. Requires M%8==K%8==N%8==0.
    Simd,
    /// simdgroup tensor units with bounds-checked partial-tile load/store.
    SimdPadded,
    /// Threadgroup-memory-tiled scalar fp32 (16x16 tiles).
    Tiled,
    /// One thread per output element; for very small dims.
    Naive,
}

/// Metal hardware model — built once at startup from device properties.
pub struct MetalHwModel {
    pub gpu_family: AppleGpuFamily,
    pub gpu_name: String,
    /// Effective fp32 throughput for simdgroup_matrix sgemm (GFLOP/s).
    pub sgemm_simd_flops: f64,
    /// Effective throughput for 32×32 tiled simdgroup matmul (GFLOP/s).
    pub sgemm_simd_4x4_flops: f64,
    /// Effective throughput for padded simdgroup variant (GFLOP/s).
    pub sgemm_padded_flops: f64,
    /// Effective throughput for scalar tiled fp32 (GFLOP/s).
    pub sgemm_tiled_flops: f64,
    /// Per-kernel dispatch overhead (ns).
    pub dispatch_overhead_ns: f64,
    /// Per-command-buffer commit + wait_until_completed (ns).
    pub roundtrip_overhead_ns: f64,
    /// Threadgroup memory budget per group (bytes).
    pub threadgroup_mem_bytes: usize,
    /// Has unified memory (zero-copy CPU↔GPU).
    pub unified_memory: bool,
    /// Minimum M·K·N (FLOP/2 ≈ MAC count) above which routing through
    /// MPSMatrixMultiplication wins despite per-call objc bridging cost.
    /// Below this we use our in-encoder MSL kernels.
    pub mps_threshold_flop: u64,
}

impl MetalHwModel {
    fn detect() -> Self {
        let dev = metal_device();
        let (name, unified) = match dev {
            Some(d) => (d.name.clone(), d.has_unified_memory),
            None => ("unknown".to_string(), false),
        };
        let family = AppleGpuFamily::from_name(&name);

        // Tier 1 — compile-time platform defaults (per Apple GPU family).
        // These are last-resort fallbacks when calibration cache + measurement
        // are both unavailable.
        let (simd_flops, padded_flops, tiled_flops) = match family {
            AppleGpuFamily::M4 => (600e9, 350e9, 100e9),
            AppleGpuFamily::M3 => (500e9, 300e9, 90e9),
            AppleGpuFamily::M2 => (400e9, 240e9, 75e9),
            AppleGpuFamily::M1Pro => (350e9, 200e9, 65e9),
            AppleGpuFamily::M1 => (200e9, 110e9, 40e9),
            AppleGpuFamily::Unknown => (300e9, 180e9, 60e9),
        };
        let mut simd_4x4_flops = simd_flops * 3.5;
        let mut simd_flops = simd_flops;
        let mut padded_flops = padded_flops;
        let mut tiled_flops = tiled_flops;
        let mut roundtrip_ns = 800_000.0_f64;

        // Tier 2 — only load if a calibration cache file already exists.
        // We never measure at startup — that's done by `cargo run --example
        // metal_calibrate` (or via `Calibration::measure()` directly).
        // The cache file is keyed by GPU registry ID, so it's portable across runs.
        let dev_id = dev.map(|d| d.registry_id).unwrap_or(0);
        if let Some(cal) = crate::calibrate::Calibration::load(dev_id) {
            simd_4x4_flops = cal.sgemm_simd_4x4_flops;
            simd_flops = cal.sgemm_simd_flops;
            padded_flops = cal.sgemm_padded_flops;
            tiled_flops = cal.sgemm_tiled_flops;
            roundtrip_ns = cal.roundtrip_overhead_ns;
        }

        // MPS pays ~5–20µs objc/encoder overhead; we want compute to be at
        // least ~5× that to net win. With our 32×32 simd kernel running near
        // 1 TFLOPS, that's M·K·N ≥ ~25M FLOPs (≈ 256×256×768). Use 16M as a
        // conservative cutoff; tune with RLX_MPS_THRESHOLD_FLOP env var.
        let mps_threshold_flop = rlx_ir::env::var("RLX_MPS_THRESHOLD_FLOP")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(16_000_000);

        Self {
            gpu_family: family,
            gpu_name: name,
            sgemm_simd_flops: simd_flops,
            sgemm_simd_4x4_flops: simd_4x4_flops,
            sgemm_padded_flops: padded_flops,
            sgemm_tiled_flops: tiled_flops,
            dispatch_overhead_ns: 8_000.0,
            roundtrip_overhead_ns: roundtrip_ns,
            threadgroup_mem_bytes: 32 * 1024,
            unified_memory: unified,
            mps_threshold_flop,
        }
    }

    // ── Dispatch decisions ──────────────────────────────────────────

    /// Pick the best sgemm variant for these dimensions.
    /// Higher-throughput variants have stricter alignment requirements.
    pub fn pick_sgemm(&self, m: usize, k: usize, n: usize) -> SgemmVariant {
        let aligned_32 = m.is_multiple_of(32) && k.is_multiple_of(32) && n.is_multiple_of(32);
        let aligned_8 = m.is_multiple_of(8) && k.is_multiple_of(8) && n.is_multiple_of(8);

        // MPS path: Apple's tuned matmul wins above ~16 MFLOPs because their
        // private knowledge of the tensor-unit fabric beats our hand-rolled
        // 32×32 simd kernel. Below the threshold the ~5–20µs objc bridging
        // cost dominates and we stay on the in-encoder MSL path.
        // Override via env var for A/B benchmarking.
        let mps_disabled = rlx_ir::env::var("RLX_DISABLE_MPS")
            .map(|v| v == "1")
            .unwrap_or(false);
        let flop = (m as u64) * (k as u64) * (n as u64);
        if !mps_disabled
            && crate::mps_blas::mps_supports_matmul()
            && flop >= self.mps_threshold_flop
        {
            return SgemmVariant::Mps;
        }

        if aligned_32 && m >= 32 && n >= 32 {
            SgemmVariant::Simd4x4
        } else if aligned_8 && m >= 8 && n >= 8 {
            SgemmVariant::Simd
        } else if k.is_multiple_of(8) && n >= 8 && m >= 1 {
            SgemmVariant::SimdPadded
        } else if m >= 16 && n >= 16 {
            SgemmVariant::Tiled
        } else {
            SgemmVariant::Naive
        }
    }

    /// Estimate execution time in nanoseconds for an sgemm of given dims.
    pub fn sgemm_cost_ns(&self, m: usize, k: usize, n: usize) -> f64 {
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let throughput = match self.pick_sgemm(m, k, n) {
            // MPS hits roughly 1.5–2.5× our hand-rolled simd_4x4 throughput
            // on M3/M4 once it's past the bridging-cost threshold.
            SgemmVariant::Mps => self.sgemm_simd_4x4_flops * 2.0,
            SgemmVariant::Simd4x4 => self.sgemm_simd_4x4_flops,
            SgemmVariant::Simd => self.sgemm_simd_flops,
            SgemmVariant::SimdPadded => self.sgemm_padded_flops,
            SgemmVariant::Tiled => self.sgemm_tiled_flops,
            SgemmVariant::Naive => self.sgemm_tiled_flops * 0.3,
        };
        let compute_ns = flops / throughput;
        compute_ns + self.dispatch_overhead_ns
    }

    /// Should we fuse matmul + bias + activation into a single kernel?
    /// Yes — saves dispatch overhead. Only skip if any kernel is unsupported.
    pub fn prefer_fused_matmul_bias(&self, _m: usize, _k: usize, _n: usize) -> bool {
        // Always fuse — fused kernels never lose compared to separate calls.
        true
    }

    /// Can the entire transformer layer's intermediates fit in threadgroup memory?
    /// If yes, a monolithic FusedTransformerLayer shader is viable.
    pub fn fits_threadgroup_mem(
        &self,
        batch: usize,
        seq: usize,
        hidden: usize,
        intermediate: usize,
    ) -> bool {
        // Per-row stack: hidden * 4 bytes (residual) + 3*hidden*4 (qkv) + intermediate*4 (ffn)
        // Per row × batch×seq rows
        let m = batch * seq;
        let bytes = m * (hidden + 3 * hidden + hidden + intermediate) * 4;
        bytes <= self.threadgroup_mem_bytes
    }

    /// Estimate total forward time for a transformer of given shape.
    /// Used to predict batch-size crossover where Metal beats CPU.
    pub fn estimate_transformer_forward_ns(
        &self,
        batch: usize,
        seq: usize,
        hidden: usize,
        intermediate: usize,
        num_heads: usize,
        num_layers: usize,
    ) -> f64 {
        let m = batch * seq;
        let _ = num_heads;

        // Per layer: QKV proj + out proj + FC1 + FC2 + element-wise ops
        let qkv = self.sgemm_cost_ns(m, hidden, 3 * hidden);
        let out = self.sgemm_cost_ns(m, hidden, hidden);
        let fc1 = self.sgemm_cost_ns(m, hidden, intermediate);
        let fc2 = self.sgemm_cost_ns(m, intermediate, hidden);
        // Approx attention: O(seq^2 * hidden) — usually small for embeddings.
        let attn = (seq * seq * hidden) as f64 / self.sgemm_simd_flops + self.dispatch_overhead_ns;
        // Element-wise + LN: dominated by dispatch overhead at small sizes.
        let elem = 4.0 * self.dispatch_overhead_ns;

        let per_layer = qkv + out + fc1 + fc2 + attn + elem;
        per_layer * num_layers as f64 + self.roundtrip_overhead_ns
    }
}

/// Global hardware model singleton.
pub fn hw_model() -> &'static MetalHwModel {
    static MODEL: OnceLock<MetalHwModel> = OnceLock::new();
    MODEL.get_or_init(MetalHwModel::detect)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_some_gpu() {
        let hw = hw_model();
        assert!(!hw.gpu_name.is_empty());
        assert!(hw.sgemm_simd_flops > 0.0);
    }

    #[test]
    fn picks_simd_for_aligned() {
        // Force the in-encoder MSL path so the threshold logic doesn't shadow
        // the alignment routing (these dims would otherwise hit Mps).
        rlx_ir::env::set("RLX_DISABLE_MPS", "1");
        let hw = MetalHwModel::detect();
        assert_eq!(hw.pick_sgemm(64, 768, 2304), SgemmVariant::Simd4x4);
        assert_eq!(hw.pick_sgemm(8, 16, 16), SgemmVariant::Simd);
        assert_eq!(hw.pick_sgemm(6, 768, 2304), SgemmVariant::SimdPadded);
        assert_eq!(hw.pick_sgemm(6, 7, 7), SgemmVariant::Naive);
        unsafe {
            rlx_ir::env::unset("RLX_DISABLE_MPS");
        }
    }
}
