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

//! Runtime configuration — compile-time platform defaults + runtime hardware detection.
//!
//! Compile-time: target arch/OS sets optimal defaults (cache line, SIMD strategy).
//! Runtime: sysctl/cpuid refines values (P-core count, L1/L2 sizes).
//! Env vars: `RLX_*` overrides for manual tuning — or set the same keys in
//! code via [`rlx_ir::env::set`] / [`RuntimeConfig::install`].
//!
//! ```bash
//! RLX_WORKERS=8           # thread pool size (0 = auto)
//! RLX_PAR_THRESHOLD=20000 # min elements for parallel dispatch
//! RLX_SDPA_THRESHOLD=32   # seq len: NEON dots (≤) vs BLAS sgemm (>)
//! RLX_ARENA_ALIGN=128     # arena buffer alignment in bytes
//! RLX_VERBOSE=0           # 0=quiet, 1=fusion passes, 2=full graph dump
//! ```

use std::sync::OnceLock;

// ── Compile-time platform defaults ──────────────────────────────────────

/// Cache line size — known at compile time per platform.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
const PLATFORM_CACHE_LINE: usize = 128; // Apple Silicon: 128-byte L1 lines

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
const PLATFORM_CACHE_LINE: usize = 64; // ARM servers (Graviton, Ampere): typically 64

#[cfg(not(target_arch = "aarch64"))]
const PLATFORM_CACHE_LINE: usize = 64; // x86_64: 64-byte cache lines

/// Default parallel threshold — tuned per platform.
/// Apple Silicon AMX handles BLAS internally; our par_for is for element-wise ops.
/// Lower threshold = more parallelism for LayerNorm/GELU on small tensors.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
const PLATFORM_PAR_THRESHOLD: usize = 16_384; // Apple Silicon: AMX does BLAS, parallelize rest earlier

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
const PLATFORM_PAR_THRESHOLD: usize = 30_000;

// ── Runtime hardware detection ──────────────────────────────────────────

/// Detect hardware properties at runtime.
struct HwInfo {
    total_cpus: usize,
    perf_cores: usize, // P-cores (0 = unknown, use total_cpus)
    l1d_cache: usize,  // L1 data cache bytes (0 = unknown)
    l2_cache: usize,   // L2 cache bytes (0 = unknown)
    cache_line: usize, // actual cache line from OS (0 = use compile-time default)
}

impl HwInfo {
    fn detect() -> Self {
        let total = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);

        let mut info = HwInfo {
            total_cpus: total,
            perf_cores: 0,
            l1d_cache: 0,
            l2_cache: 0,
            cache_line: 0,
        };

        #[cfg(target_os = "macos")]
        {
            info.perf_cores = sysctl_usize("hw.perflevel0.physicalcpu").unwrap_or(0);
            info.l1d_cache = sysctl_usize("hw.l1dcachesize").unwrap_or(0);
            info.l2_cache = sysctl_usize("hw.l2cachesize").unwrap_or(0);
            info.cache_line = sysctl_usize("hw.cachelinesize").unwrap_or(0);
        }

        #[cfg(target_os = "linux")]
        {
            // /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size
            if let Ok(v) = std::fs::read_to_string(
                "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size",
            ) {
                info.cache_line = v.trim().parse().unwrap_or(0);
            }
            if let Ok(v) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/size")
            {
                // Parse "32K" or "32768"
                let s = v.trim().to_uppercase();
                if s.ends_with('K') {
                    info.l1d_cache = s[..s.len() - 1].parse::<usize>().unwrap_or(0) * 1024;
                } else {
                    info.l1d_cache = s.parse().unwrap_or(0);
                }
            }
        }

        info
    }

    /// Optimal worker count: P-cores/2 (avoids E-cores + AMX cache thrashing).
    fn optimal_workers(&self) -> usize {
        let base = if self.perf_cores > 0 {
            self.perf_cores / 2 // Use half of P-cores
        } else {
            self.total_cpus / 2 // Fallback: half of all CPUs
        };
        base.clamp(1, 15)
    }

    /// Cache line: prefer runtime-detected, fall back to compile-time.
    fn cache_line(&self) -> usize {
        if self.cache_line > 0 {
            self.cache_line
        } else {
            PLATFORM_CACHE_LINE
        }
    }

    /// Fusion threshold: intermediates must fit in L1 for monolithic kernels.
    #[allow(dead_code)]
    fn fuse_attn_threshold(&self) -> usize {
        if self.l1d_cache > 0 {
            // L1 budget: ~60% for intermediates (rest for weights being streamed)
            // Each fused layer needs: qkv(m×3h) + attn(m×h) + res(m×h) + normed(m×h) + ffn(m×int)
            // ≈ m × 7h floats for BERT. Must fit in 60% of L1.
            // Solve for m: m = 0.6 * L1 / (7 * 768 * 4) ≈ L1/36000
            // For L1=64KB: m ≈ 1.8 → batch*seq ≤ ~2 → threshold ~64 is about right
            64
        } else {
            64
        }
    }
}

#[cfg(target_os = "macos")]
fn sysctl_usize(name: &str) -> Option<usize> {
    use std::ffi::CString;
    let cname = CString::new(name).ok()?;
    let mut val: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    unsafe {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const i8,
                oldp: *mut u8,
                oldlenp: *mut usize,
                newp: *const u8,
                newlen: usize,
            ) -> i32;
        }
        let ret = sysctlbyname(
            cname.as_ptr(),
            &mut val as *mut u64 as *mut u8,
            &mut len,
            std::ptr::null(),
            0,
        );
        if ret == 0 { Some(val as usize) } else { None }
    }
}

/// Runtime configuration for the RLX CPU backend.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    // ── Thread pool ─────────────────────────────────────────
    pub pool_workers: usize,

    // ── Parallelization ─────────────────────────────────────
    pub par_threshold: usize,
    pub min_rows_per_thread: usize,

    // ── SDPA dispatch ───────────────────────────────────────
    pub sdpa_seq_threshold: usize,

    // ── Memory planning ─────────────────────────────────────
    pub arena_alignment: usize,

    // ── Numerical constants ─────────────────────────────────
    pub ln_eps_default: f32,
    pub attn_mask_neg_inf: f32,
    pub score_skip_threshold: f32,
    pub mask_binary_threshold: f32,

    // ── Diagnostics ─────────────────────────────────────────
    pub verbose: u8,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::auto_detect()
    }
}

impl RuntimeConfig {
    /// Auto-detect hardware and apply optimal defaults.
    pub fn auto_detect() -> Self {
        let hw = HwInfo::detect();

        Self {
            pool_workers: hw.optimal_workers(),
            par_threshold: PLATFORM_PAR_THRESHOLD,
            min_rows_per_thread: 4,
            sdpa_seq_threshold: 32,
            arena_alignment: hw.cache_line(),
            ln_eps_default: 1e-12,
            attn_mask_neg_inf: -1e9,
            score_skip_threshold: 1e-8,
            mask_binary_threshold: 0.5,
            verbose: 0,
        }
    }

    /// Auto-detect then override from `RLX_*` environment variables.
    pub fn from_env() -> Self {
        let mut cfg = Self::auto_detect();

        if let Some(v) = rlx_ir::env::var("RLX_WORKERS")
            && let Ok(n) = v.parse::<usize>()
        {
            cfg.pool_workers = if n == 0 { cfg.pool_workers } else { n.min(15) };
        }
        if let Some(v) = rlx_ir::env::var("RLX_PAR_THRESHOLD")
            && let Ok(n) = v.parse()
        {
            cfg.par_threshold = n;
        }
        if let Some(v) = rlx_ir::env::var("RLX_SDPA_THRESHOLD")
            && let Ok(n) = v.parse()
        {
            cfg.sdpa_seq_threshold = n;
        }
        if let Some(v) = rlx_ir::env::var("RLX_ARENA_ALIGN")
            && let Ok(n) = v.parse()
        {
            cfg.arena_alignment = n;
        }
        if let Some(v) = rlx_ir::env::var("RLX_VERBOSE")
            && let Ok(n) = v.parse()
        {
            cfg.verbose = n;
        }

        if cfg.verbose >= 1 {
            let hw = HwInfo::detect();
            eprintln!(
                "[rlx] hw: {} CPUs ({} P-cores), L1={}KB, L2={}KB, cacheline={}B",
                hw.total_cpus,
                hw.perf_cores,
                hw.l1d_cache / 1024,
                hw.l2_cache / 1024,
                hw.cache_line()
            );
            eprintln!(
                "[rlx] config: workers={}, par_thr={}, sdpa_thr={}, align={}",
                cfg.pool_workers, cfg.par_threshold, cfg.sdpa_seq_threshold, cfg.arena_alignment
            );
        }

        cfg
    }

    /// Push this config into the global [`rlx_ir::env`] override map so all
    /// RLX backends see the same knobs without setting process env vars.
    pub fn install(&self) {
        rlx_ir::env::set("RLX_WORKERS", self.pool_workers.to_string());
        rlx_ir::env::set("RLX_PAR_THRESHOLD", self.par_threshold.to_string());
        rlx_ir::env::set("RLX_SDPA_THRESHOLD", self.sdpa_seq_threshold.to_string());
        rlx_ir::env::set("RLX_ARENA_ALIGN", self.arena_alignment.to_string());
        rlx_ir::env::set("RLX_VERBOSE", self.verbose.to_string());
    }

    /// Get or initialize the global singleton config.
    pub fn global() -> &'static RuntimeConfig {
        static CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();
        CONFIG.get_or_init(RuntimeConfig::from_env)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_detect_sane_defaults() {
        let cfg = RuntimeConfig::auto_detect();
        assert!(cfg.pool_workers >= 1);
        assert!(cfg.pool_workers <= 15);
        // Platform-appropriate cache line
        assert!(cfg.arena_alignment >= 64);
        assert!(cfg.verbose == 0);
    }

    #[test]
    fn global_is_consistent() {
        let a = RuntimeConfig::global();
        let b = RuntimeConfig::global();
        assert_eq!(a.pool_workers, b.pool_workers);
    }

    #[test]
    fn hw_detection() {
        let hw = HwInfo::detect();
        assert!(hw.total_cpus >= 1);
        // On macOS with sysctl, we should detect cache line
        #[cfg(target_os = "macos")]
        assert!(
            hw.cache_line > 0,
            "expected sysctl to return cache line size"
        );
    }
}
