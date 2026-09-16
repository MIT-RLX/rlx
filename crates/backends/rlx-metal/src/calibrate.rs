// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
    /// Measured GFLOP/s for the causal attention thunk.
    ///
    /// Attention does not run at GEMM rate — softmax, masking and
    /// materializing the S x S scores are work no FLOP count sees — so the cost
    /// model needs its own number rather than a fudge factor applied to sgemm.
    /// It was a hardcoded `ATTENTION_EFFICIENCY` constant until this field
    /// existed, and the constant was wrong by 5x because it had been derived
    /// from a measurement that accidentally included Q/K/V upload.
    ///
    /// `serde(default)` so an older cache file still loads; a missing value
    /// falls back to the arch default rather than deserialising to zero and
    /// producing an infinite predicted cost.
    #[serde(default)]
    pub attention_flops: f64,
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
                let enc = cb.compute_command_encoder_with_dispatch_type(
                    crate::mtl::MTLDispatchType::Serial,
                );
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
            let enc =
                cb.compute_command_encoder_with_dispatch_type(crate::mtl::MTLDispatchType::Serial);
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

        // Attention, measured the same way the sgemm variants are: run the real
        // thunk and divide FLOPs by device time.
        //
        // B=1, H=16, S=512, D=64 — a mid prefill, big enough that the kernel
        // dominates the dispatch and small enough to calibrate quickly. Causal,
        // so the FLOP count is 2*B*H*S^2*D (QK^T and PV are S^2*D MACs each;
        // the mask skips about half).
        let attention = {
            use rlx_ir::op::MaskKind;
            use rlx_ir::{DType, Graph, Shape};
            let (b, h, sq, d) = (1usize, 16usize, 512usize, 64usize);
            let f = DType::F32;
            let mut g = Graph::new("calib_attn");
            let q = g.input("q", Shape::new(&[b, h, sq, d], f));
            let k = g.input("k", Shape::new(&[b, h, sq, d], f));
            let v = g.input("v", Shape::new(&[b, h, sq, d], f));
            let y = g.add_node(
                rlx_ir::Op::Attention {
                    num_heads: h,
                    head_dim: d,
                    v_head_dim: None,
                    mask_kind: MaskKind::Causal,
                    score_scale: None,
                    attn_logit_softcap: None,
                },
                vec![q, k, v],
                Shape::new(&[b, h, sq, d], f),
            );
            g.set_outputs(vec![y]);

            let n = b * h * sq * d;
            let data: Vec<f32> = (0..n)
                .map(|i| ((i * 13 + 7) % 257) as f32 / 257.0)
                .collect();
            let feeds: Vec<(&str, &[f32])> = vec![("q", &data), ("k", &data), ("v", &data)];

            // Per-thunk profiling isolates the KERNEL from the Q/K/V upload.
            // Timing the whole run instead is what produced the 5x-wrong
            // constant this field replaces.
            rlx_ir::env::set("RLX_METAL_THUNK_PROFILE", "1");
            let mut exe = crate::backend::MetalExecutable::compile(g);
            let _ = exe.run(&feeds);
            crate::thunk_profile::reset();
            const N: usize = 10;
            for _ in 0..N {
                let _ = exe.run(&feeds);
            }
            let ms = crate::thunk_profile::total_ms().unwrap_or(0.0) / N as f64;
            rlx_ir::env::unset("RLX_METAL_THUNK_PROFILE");

            let flops = 2.0 * (b * h) as f64 * (sq * sq) as f64 * d as f64;
            if ms > 0.0 { flops / (ms / 1e3) } else { 0.0 }
        };

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
            attention_flops: attention,
        }
    }

    /// Load from cache, or measure and save. Idempotent.
    pub fn load_or_measure() -> Self {
        let dev = metal_device().expect("Metal device required");
        if let Some(cal) = Self::load(dev.registry_id) {
            return cal;
        }
        let verbose = rlx_ir::env::var("RLX_VERBOSE")
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
