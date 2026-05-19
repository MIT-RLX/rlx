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

//! Calibration cache — measures real MLX matmul throughput on this
//! device, persists results to disk, and feeds the runtime's cost
//! model. Mirrors the rlx-metal calibration shape.
//!
//! Strategy:
//!   1. Look for cache file `~/.cache/rlx/mlx-calib-<sanitized-name>.json`.
//!   2. If found and valid: use measured values.
//!   3. Otherwise: run a quick benchmark, save results, use them.
//!
//! Cache key is the MLX device name string (e.g. "Apple M2 Pro").
//! That keeps the cache valid across runs on the same machine and
//! invalidates automatically if the device the user runs on changes.

use std::path::PathBuf;

use rlx_ir::Tick;
use serde::{Deserialize, Serialize};

use crate::array::{Array, MlxError, device_name, eval};
use crate::ffi::{MlxMask, MlxReduce};
use crate::ops;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    /// Device name string. Used as the cache key + sanity check on load.
    pub device_name: String,
    /// Measured GFLOP/s for a representative large matmul (best case).
    pub sgemm_large_flops: f64,
    /// Measured GFLOP/s for a small matmul (where overhead dominates).
    pub sgemm_small_flops: f64,
    /// Synchronous round-trip overhead (ns) — cost of build-graph +
    /// `eval` against a near-empty graph. Captures host+sync cost.
    pub roundtrip_overhead_ns: f64,
    /// Memory bandwidth in GB/s, measured by timing a contiguous
    /// copy of a large F32 array. Lower-bound on real bandwidth
    /// because MLX's lazy eval batches the op with allocator setup.
    #[serde(default)]
    pub memory_bw_gbps: f64,
    /// Throughput for SDPA at a representative shape (`B=1, H=4,
    /// S=128, D=64`). Reported as effective FLOP/s using
    /// `4 * B * H * S * S * D` as the work estimate (Q@K^T plus
    /// attn@V dominant terms).
    #[serde(default)]
    pub attention_flops: f64,
    /// Reduction throughput in GB/s — bytes-read per second for a
    /// sum reduction over the last axis of a large array. Mostly
    /// memory-bound, so doubles as a sanity check on memory_bw_gbps.
    #[serde(default)]
    pub reduce_gbps: f64,
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn cache_path(device_name: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let dir = PathBuf::from(home).join(".cache").join("rlx");
    let _ = std::fs::create_dir_all(&dir);
    let key = if device_name.is_empty() {
        "default".into()
    } else {
        sanitize(device_name)
    };
    dir.join(format!("mlx-calib-{key}.json"))
}

impl Calibration {
    pub fn load(name: &str) -> Option<Self> {
        let raw = std::fs::read_to_string(cache_path(name)).ok()?;
        let cal: Calibration = serde_json::from_str(&raw).ok()?;
        if cal.device_name == name {
            Some(cal)
        } else {
            None
        }
    }

    pub fn save(&self) -> std::io::Result<()> {
        let path = cache_path(&self.device_name);
        let raw = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, raw)
    }

    /// Run two representative matmul shapes (large + small) plus a
    /// near-empty round-trip. ~tens of ms total; runs once per machine
    /// per device-name change.
    pub fn measure() -> Result<Self, MlxError> {
        let name = device_name();

        let measure_matmul = |m: usize, k: usize, n: usize| -> Result<f64, MlxError> {
            // Reusable input arrays — outside the timed loop so we
            // don't time host buffer construction.
            let lhs_data = (0..m * k).map(|i| (i as f32) / 257.0).collect::<Vec<_>>();
            let rhs_data = (0..k * n)
                .map(|i| ((i + 7) as f32) / 257.0)
                .collect::<Vec<_>>();
            let lhs = Array::from_f32_slice(&lhs_data, &[m, k], rlx_ir::DType::F32)?;
            let rhs = Array::from_f32_slice(&rhs_data, &[k, n], rlx_ir::DType::F32)?;

            // Warmup: matmul + eval to JIT/compile any one-time paths.
            let warm = ops::matmul(&lhs, &rhs)?;
            eval(&[&warm])?;

            // Timed loop: chain N matmuls in one trace, eval all.
            const N_ITER: usize = 50;
            let t0 = Tick::now();
            let mut outs: Vec<Array> = Vec::with_capacity(N_ITER);
            for _ in 0..N_ITER {
                outs.push(ops::matmul(&lhs, &rhs)?);
            }
            let refs: Vec<&Array> = outs.iter().collect();
            eval(&refs)?;
            let total_ns = Tick::now().elapsed_ns(t0) as f64;
            let total_s = total_ns / 1e9;
            let flops = 2.0 * m as f64 * k as f64 * n as f64 * N_ITER as f64;
            Ok(flops / total_s)
        };

        let sgemm_large = measure_matmul(256, 768, 3072)?;
        let sgemm_small = measure_matmul(8, 64, 64)?;

        // Round-trip baseline: matmul on a tiny shape; the dominant
        // cost is the synchronous eval/sync, not compute.
        let roundtrip_ns = {
            let lhs = Array::from_f32_slice(&[1.0, 2.0], &[1, 2], rlx_ir::DType::F32)?;
            let rhs = Array::from_f32_slice(&[3.0, 4.0], &[2, 1], rlx_ir::DType::F32)?;
            const N_ITER: usize = 10;
            let t0 = Tick::now();
            for _ in 0..N_ITER {
                let y = ops::matmul(&lhs, &rhs)?;
                eval(&[&y])?;
            }
            Tick::now().elapsed_ns(t0) as f64 / N_ITER as f64
        };

        // Memory-bw probe: time a contiguous copy of a 4 MB array
        // (1M f32). cast(_, F32) on an already-F32 array degenerates
        // to a no-op, so use add(_, 0) which has the same byte
        // movement profile but produces a real fresh buffer.
        let memory_bw_gbps = {
            const N: usize = 1024 * 1024;
            let data: Vec<f32> = (0..N).map(|i| i as f32 * 0.001).collect();
            let a = Array::from_f32_slice(&data, &[N], rlx_ir::DType::F32)?;
            let zero = Array::from_f32_slice(&[0.0], &[1], rlx_ir::DType::F32)?;
            // Warmup
            let warm = ops::add(&a, &zero)?;
            eval(&[&warm])?;
            const N_ITER: usize = 20;
            let t0 = Tick::now();
            let mut outs = Vec::with_capacity(N_ITER);
            for _ in 0..N_ITER {
                outs.push(ops::add(&a, &zero)?);
            }
            let refs: Vec<&Array> = outs.iter().collect();
            eval(&refs)?;
            let total_ns = Tick::now().elapsed_ns(t0) as f64;
            // Per iteration: read N*4 bytes + write N*4 bytes.
            let bytes_per_iter = (N * 4 * 2) as f64;
            (bytes_per_iter * N_ITER as f64) / total_ns
        };

        // Attention probe at a small-but-realistic shape.
        let attention_flops = {
            const B: usize = 1;
            const H: usize = 4;
            const S: usize = 128;
            const D: usize = 64;
            let q_data: Vec<f32> = (0..B * H * S * D)
                .map(|i| (i as f32 % 17.0) * 0.01)
                .collect();
            let q = Array::from_f32_slice(&q_data, &[B, H, S, D], rlx_ir::DType::F32)?;
            // Reuse same data for k, v — calibration just times work.
            let k = Array::from_f32_slice(&q_data, &[B, H, S, D], rlx_ir::DType::F32)?;
            let v = Array::from_f32_slice(&q_data, &[B, H, S, D], rlx_ir::DType::F32)?;
            let scale = 1.0 / (D as f32).sqrt();
            // Warmup
            let warm = ops::attention(&q, &k, &v, scale, MlxMask::None, None)?;
            eval(&[&warm])?;
            const N_ITER: usize = 20;
            let t0 = Tick::now();
            let mut outs = Vec::with_capacity(N_ITER);
            for _ in 0..N_ITER {
                outs.push(ops::attention(&q, &k, &v, scale, MlxMask::None, None)?);
            }
            let refs: Vec<&Array> = outs.iter().collect();
            eval(&refs)?;
            let total_ns = Tick::now().elapsed_ns(t0) as f64;
            let total_s = total_ns / 1e9;
            // Q@K^T: 2*B*H*S*S*D, scores@V: 2*B*H*S*S*D (≈ ignoring softmax)
            let flops_per_iter = 4.0 * B as f64 * H as f64 * S as f64 * S as f64 * D as f64;
            (flops_per_iter * N_ITER as f64) / total_s
        };

        // Reduce probe (sum over last axis of a large 2D array).
        let reduce_gbps = {
            const M: usize = 1024;
            const N: usize = 1024;
            let data: Vec<f32> = (0..M * N).map(|i| i as f32 * 0.001).collect();
            let a = Array::from_f32_slice(&data, &[M, N], rlx_ir::DType::F32)?;
            let warm = ops::reduce(&a, MlxReduce::Sum, &[1], false)?;
            eval(&[&warm])?;
            const N_ITER: usize = 20;
            let t0 = Tick::now();
            let mut outs = Vec::with_capacity(N_ITER);
            for _ in 0..N_ITER {
                outs.push(ops::reduce(&a, MlxReduce::Sum, &[1], false)?);
            }
            let refs: Vec<&Array> = outs.iter().collect();
            eval(&refs)?;
            let total_ns = Tick::now().elapsed_ns(t0) as f64;
            let bytes_per_iter = (M * N * 4) as f64;
            (bytes_per_iter * N_ITER as f64) / total_ns
        };

        Ok(Calibration {
            device_name: name,
            sgemm_large_flops: sgemm_large,
            sgemm_small_flops: sgemm_small,
            roundtrip_overhead_ns: roundtrip_ns,
            memory_bw_gbps,
            attention_flops,
            reduce_gbps,
        })
    }

    /// Load from cache, or measure and save. Idempotent. Falls back to
    /// a conservative default Calibration on measurement failure (so
    /// the cost model never panics on a backend that's just missing).
    pub fn load_or_measure() -> Self {
        let name = device_name();
        if let Some(cal) = Self::load(&name) {
            return cal;
        }
        let verbose = std::env::var("RLX_VERBOSE")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(0)
            >= 1;
        if verbose {
            eprintln!("[rlx-mlx] no calibration cache for '{name}'; measuring...");
        }
        let cal = Self::measure().unwrap_or_else(|e| {
            if verbose {
                eprintln!("[rlx-mlx] calibration failed: {e}; using conservative defaults");
            }
            Calibration {
                device_name: name.clone(),
                // Conservative numbers: assume MLX is roughly Metal-tier
                // (~1 TFLOP/s on M-series base, lower on small shapes).
                sgemm_large_flops: 1.0e12,
                sgemm_small_flops: 5.0e10,
                roundtrip_overhead_ns: 200_000.0,
                memory_bw_gbps: 200.0,
                attention_flops: 5.0e11,
                reduce_gbps: 150.0,
            }
        });
        if verbose {
            eprintln!(
                "[rlx-mlx] calibrated: large={:.0} GF/s, small={:.0} GF/s, \
                       rt={:.0}µs, mem={:.0} GB/s, attn={:.0} GF/s, reduce={:.0} GB/s",
                cal.sgemm_large_flops / 1e9,
                cal.sgemm_small_flops / 1e9,
                cal.roundtrip_overhead_ns / 1000.0,
                cal.memory_bw_gbps,
                cal.attention_flops / 1e9,
                cal.reduce_gbps
            );
        }
        let _ = cal.save();
        cal
    }
}
