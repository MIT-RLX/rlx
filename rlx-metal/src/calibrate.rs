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

//! Calibration cache — measures real GPU throughput on this hardware,
//! persists results to disk, replaces hardcoded `sgemm_*_flops` defaults.
//!
//! Strategy:
//!   1. Look for cache file `~/.cache/rlx/metal-calib-<hwid>.json`
//!   2. If found and valid: use measured values
//!   3. Otherwise: run quick benchmark (~50ms total), save results, use them
//!
//! The cache is keyed by GPU registry ID, so it stays valid across runs
//! on the same machine and is invalidated automatically if hardware changes.

use rlx_ir::Tick;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::blas::metal_sgemm;
use crate::device::metal_device;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub gpu_name: String,
    pub registry_id: u64,
    /// Measured GFLOP/s for sgemm_simd_4x4 at large M (best case).
    pub sgemm_simd_4x4_flops: f64,
    /// Measured GFLOP/s for sgemm_simd at small-aligned M.
    pub sgemm_simd_flops: f64,
    /// Measured GFLOP/s for sgemm_simd_padded.
    pub sgemm_padded_flops: f64,
    /// Measured GFLOP/s for sgemm_tiled (scalar fp32).
    pub sgemm_tiled_flops: f64,
    /// Measured baseline command-buffer round-trip (ns).
    pub roundtrip_overhead_ns: f64,
}

fn cache_path(registry_id: u64) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let dir = PathBuf::from(home).join(".cache").join("rlx");
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("metal-calib-{:x}.json", registry_id))
}

impl Calibration {
    pub fn load(registry_id: u64) -> Option<Self> {
        let path = cache_path(registry_id);
        let raw = std::fs::read_to_string(&path).ok()?;
        let cal: Calibration = serde_json::from_str(&raw).ok()?;
        if cal.registry_id == registry_id {
            Some(cal)
        } else {
            None
        }
    }

    pub fn save(&self) -> std::io::Result<()> {
        let path = cache_path(self.registry_id);
        let raw = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, raw)
    }

    /// Measure throughput for each kernel variant by running representative
    /// matmul shapes. Total cost ~50ms; runs once per machine.
    pub fn measure() -> Self {
        let dev = metal_device().expect("Metal device required for calibration");

        let measure = |m: usize, k: usize, n: usize| -> f64 {
            // Allocate three buffers in one (m*k + k*n + m*n) * 4 bytes
            let total = (m * k + k * n + m * n) * 4;
            let buffer = dev.alloc_shared(total);
            unsafe {
                let ptr = buffer.contents() as *mut f32;
                for i in 0..(m * k + k * n) {
                    *ptr.add(i) = ((i * 13 + 7) % 257) as f32 / 257.0;
                }
            }
            let a_off = 0;
            let b_off = m * k * 4;
            let c_off = (m * k + k * n) * 4;

            // Warmup (kernels JIT on first dispatch)
            {
                let cb = dev.queue.new_command_buffer();
                let enc =
                    cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
                for _ in 0..2 {
                    metal_sgemm(enc, &buffer, a_off, b_off, c_off, m, k, n);
                }
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
            // Batch many sgemm calls into ONE command buffer so compute
            // dominates the single wait_until_completed (~0.8ms baseline).
            // 50 iterations × ~50µs compute = ~2.5ms, dwarfing dispatch.
            let n_iter = 50;
            let cb = dev.queue.new_command_buffer();
            let enc = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
            let t0 = Tick::now();
            for _ in 0..n_iter {
                metal_sgemm(enc, &buffer, a_off, b_off, c_off, m, k, n);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let total_s = Tick::now().elapsed_ns(t0) as f64 / 1e9;
            2.0 * (m * k * n) as f64 * (n_iter as f64) / total_s
        };

        // Probe shapes — sized to match production BERT FFN matmul.
        // 50 iterations per probe → enough compute to dominate dispatch cost.
        //   Simd4x4   : 256×768×3072  (BERT FFN-up at batch=16, seq=16-ish)
        //   Simd      : 8×512×512     (8-aligned, m<32; small variant)
        //   SimdPadded: 6×768×768     (batch=1 attention-out)
        //   Tiled     : 64×128×17     (n%8 != 0 fallback)
        let simd_4x4 = measure(256, 768, 3072);
        let simd = measure(8, 512, 512);
        let padded = measure(6, 768, 768);
        let tiled = measure(64, 128, 17);

        // Round-trip baseline: empty command buffer commit+wait
        let roundtrip_ns = {
            let n_iter = 10;
            let t0 = Tick::now();
            for _ in 0..n_iter {
                let cb = dev.queue.new_command_buffer();
                cb.commit();
                cb.wait_until_completed();
            }
            Tick::now().elapsed_ns(t0) as f64 / n_iter as f64
        };

        Calibration {
            gpu_name: dev.name.clone(),
            registry_id: dev.registry_id,
            sgemm_simd_4x4_flops: simd_4x4,
            sgemm_simd_flops: simd,
            sgemm_padded_flops: padded,
            sgemm_tiled_flops: tiled,
            roundtrip_overhead_ns: roundtrip_ns,
        }
    }

    /// Load from cache, or measure and save. Idempotent.
    pub fn load_or_measure() -> Self {
        let dev = metal_device().expect("Metal device required");
        if let Some(cal) = Self::load(dev.registry_id) {
            return cal;
        }
        let verbose = std::env::var("RLX_VERBOSE")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(0)
            >= 1;
        if verbose {
            eprintln!(
                "[rlx-metal] no calibration cache for {}; measuring...",
                dev.name
            );
        }
        let cal = Self::measure();
        if verbose {
            eprintln!(
                "[rlx-metal] calibrated: simd_4x4={:.0} GF/s, simd={:.0} GF/s, padded={:.0} GF/s, tiled={:.0} GF/s, rt={:.0}µs",
                cal.sgemm_simd_4x4_flops / 1e9,
                cal.sgemm_simd_flops / 1e9,
                cal.sgemm_padded_flops / 1e9,
                cal.sgemm_tiled_flops / 1e9,
                cal.roundtrip_overhead_ns / 1000.0
            );
        }
        let _ = cal.save();
        cal
    }
}
