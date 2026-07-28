// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Multi-kernel f32 FFT dispatch for Metal (mirrors rlx-cuda/src/fft_dispatch.rs).

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::kernels::Kernels;

const WG: u64 = 256;

fn grid_1d(n: u32) -> u64 {
    (n as u64).div_ceil(WG)
}

/// native-gpu-fft: largest `n` the single-kernel on-chip path handles. Shared =
/// 8·n bytes (sre+sim f32); 4096 → 32 KB, Apple Silicon's threadgroup max.
#[cfg(feature = "native-gpu-fft")]
const BIG_FFT_MAX_N: u32 = 4096;

/// native-gpu-fft runtime gate. Default on; `RLX_FFT_FAST=0/off/false` routes
/// back to the multi-kernel path (for A/B benchmarking in one process).
#[cfg(feature = "native-gpu-fft")]
fn fast_fft_enabled() -> bool {
    !rlx_ir::env::var("RLX_FFT_FAST").is_some_and(|v| {
        v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")
    })
}

/// Max butterfly radix for the on-chip kernel: higher radix = fewer stages /
/// barriers. Default 8 (best applicable per size is chosen at dispatch).
/// `RLX_FFT_RADIX=2|4|8` caps it for A/B benchmarking.
#[cfg(feature = "native-gpu-fft")]
fn fft_max_radix() -> u32 {
    match rlx_ir::env::var("RLX_FFT_RADIX").as_deref() {
        Some("2") => 2,
        Some("4") => 4,
        Some("16") => 16,
        // Default 8 (measured sweet spot); radix-16 is opt-in (heavy register
        // pressure usually loses the occupancy it costs).
        _ => 8,
    }
}

/// Run native multi-kernel FFT on the unified-memory arena (f32, pow-2 `n`).
///
/// `real_input` (native-gpu-fft only): `src` is an `n`-wide real signal (row
/// stride `n`, imaginary part implicitly 0) rather than the 2N complex block —
/// the real→complex zero-pad fused into the on-chip kernel's load. Only the
/// radix-4/8 kernels read it; callers set it only for `n` in (1024, 4096].
#[allow(clippy::too_many_arguments)]
pub fn run_fft_gpu(
    k: &Kernels,
    enc: &ComputeCommandEncoderRef,
    arena: &metal::Buffer,
    src_off: u32,
    dst_off: u32,
    outer: u32,
    n: u32,
    inverse: bool,
    norm_scale: f32,
    real_input: bool,
) {
    if outer == 0 {
        return;
    }
    let plan = rlx_ir::fft::FftGpuPlan::new(n as usize).expect("run_fft_gpu: n must be pow2");
    let inv = if inverse { 1u32 } else { 0u32 };
    let log2n = n.trailing_zeros();
    let _ = real_input;

    // native-gpu-fft: keep the whole transform on-chip for n in (1024, 4096]
    // (the sizes that otherwise use the multi-kernel DRAM round-trip path). One
    // threadgroup per row reads src, runs every stage in threadgroup memory, and
    // writes dst — so no pre-copy and no per-stage DRAM traffic.
    #[cfg(feature = "native-gpu-fft")]
    if fast_fft_enabled() && n > rlx_ir::fft::FFT_TILE_SIZE as u32 && n <= BIG_FFT_MAX_N {
        // Pick the highest applicable radix (fewest stages/barriers): radix-8
        // for powers of 8 (n=4096), radix-4 for the rest (n=2048 = 2·4^5 via a
        // leading radix-2 stage), radix-2 as the floor / A/B baseline.
        let m = n.trailing_zeros();
        let max_radix = fft_max_radix();
        // Only radix-4/8 read `real_input`; force one of them for the fused
        // real path. Otherwise pick the highest applicable radix.
        let pipe = if real_input {
            if max_radix >= 8 && m % 3 == 0 {
                &k.fft_radix8_full_f32
            } else {
                &k.fft_radix4_full_f32
            }
        } else if max_radix >= 16 && m % 4 == 0 {
            &k.fft_radix16_full_f32
        } else if max_radix >= 8 && m % 3 == 0 {
            &k.fft_radix8_full_f32
        } else if max_radix >= 4 {
            &k.fft_radix4_full_f32
        } else {
            &k.fft_radix2_full_big_f32
        };
        let real_u = u32::from(real_input);
        enc.set_buffer(0, Some(arena), 0);
        enc.set_compute_pipeline_state(pipe);
        enc.set_bytes(1, 4, &src_off as *const u32 as *const _);
        enc.set_bytes(2, 4, &dst_off as *const u32 as *const _);
        enc.set_bytes(3, 4, &n as *const u32 as *const _);
        enc.set_bytes(4, 4, &log2n as *const u32 as *const _);
        enc.set_bytes(5, 4, &inv as *const u32 as *const _);
        enc.set_bytes(6, 4, &norm_scale as *const f32 as *const _);
        enc.set_bytes(7, 4, &outer as *const u32 as *const _);
        // buffer(8) is read only by radix-4/8; harmless for radix-2/16.
        enc.set_bytes(8, 4, &real_u as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize {
                width: outer as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: WG,
                height: 1,
                depth: 1,
            },
        );
        return;
    }

    // NOTE: no host-side src→dst copy here. The bit-reverse kernel gathers
    // src→dst on the GPU (see fft_bit_reverse_f32), so `src` may be produced by
    // a preceding GPU op in the same command buffer (e.g. rfft's Concat). A host
    // memcpy would read stale data before that op runs — it broke rfft n>4096.
    let off = dst_off;

    enc.set_buffer(0, Some(arena), 0);

    if plan.single_inner_only() {
        enc.set_compute_pipeline_state(&k.fft_radix2_full_f32);
        enc.set_bytes(1, 4, &src_off as *const u32 as *const _);
        enc.set_bytes(2, 4, &dst_off as *const u32 as *const _);
        enc.set_bytes(3, 4, &n as *const u32 as *const _);
        enc.set_bytes(4, 4, &log2n as *const u32 as *const _);
        enc.set_bytes(5, 4, &inv as *const u32 as *const _);
        enc.set_bytes(6, 4, &norm_scale as *const f32 as *const _);
        enc.set_bytes(7, 4, &outer as *const u32 as *const _);
        let tg_w = 256u64.min(n as u64);
        enc.dispatch_thread_groups(
            MTLSize {
                width: outer as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }

    // Bit-reverse gathers src→dst on the GPU (dst[k] = src[rev(k)] when
    // src != dst; in-place swap when equal), so no host pre-copy is needed.
    enc.set_compute_pipeline_state(&k.fft_bit_reverse_f32);
    enc.set_bytes(1, 4, &src_off as *const u32 as *const _);
    enc.set_bytes(2, 4, &dst_off as *const u32 as *const _);
    enc.set_bytes(3, 4, &n as *const u32 as *const _);
    enc.set_bytes(4, 4, &log2n as *const u32 as *const _);
    enc.set_bytes(5, 4, &outer as *const u32 as *const _);
    enc.dispatch_thread_groups(
        MTLSize {
            width: grid_1d(n),
            height: outer as u64,
            depth: 1,
        },
        MTLSize {
            width: WG,
            height: 1,
            depth: 1,
        },
    );
    let tile = rlx_ir::fft::FFT_TILE_SIZE.min(n as usize) as u32;
    let inner_stages = plan.inner_stages as u32;
    let num_tiles = (n / tile).max(1);
    // MSL inner kernel uses flat tg_id = row * num_tiles + tile_id; width is
    // num_tiles * outer (see fft_gpu.msl), equivalent to CUDA's (num_tiles, outer).
    let wg_threads = (n / 2).min(tile / 2);
    let scale1 = 1.0f32;

    enc.set_compute_pipeline_state(&k.fft_inner_f32);
    enc.set_bytes(1, 4, &off as *const u32 as *const _);
    enc.set_bytes(2, 4, &n as *const u32 as *const _);
    enc.set_bytes(3, 4, &tile as *const u32 as *const _);
    enc.set_bytes(4, 4, &inner_stages as *const u32 as *const _);
    enc.set_bytes(5, 4, &inv as *const u32 as *const _);
    enc.set_bytes(6, 4, &scale1 as *const f32 as *const _);
    enc.set_bytes(7, 4, &outer as *const u32 as *const _);
    enc.dispatch_thread_groups(
        MTLSize {
            width: (num_tiles * outer) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: wg_threads as u64,
            height: 1,
            depth: 1,
        },
    );

    let r4_count = plan.outer_rad4_q.len();
    for (i, q) in plan.outer_rad4_q.iter().enumerate() {
        let q_u = *q as u32;
        let stage_scale = if plan.outer_r2_hs.is_none() && i + 1 == r4_count {
            norm_scale
        } else {
            1.0f32
        };
        enc.set_compute_pipeline_state(&k.fft_outer_r4_f32);
        enc.set_bytes(1, 4, &off as *const u32 as *const _);
        enc.set_bytes(2, 4, &n as *const u32 as *const _);
        enc.set_bytes(3, 4, &q_u as *const u32 as *const _);
        enc.set_bytes(4, 4, &inv as *const u32 as *const _);
        enc.set_bytes(5, 4, &stage_scale as *const f32 as *const _);
        enc.set_bytes(6, 4, &outer as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize {
                width: grid_1d((n / 4).max(1)),
                height: outer as u64,
                depth: 1,
            },
            MTLSize {
                width: WG,
                height: 1,
                depth: 1,
            },
        );
    }

    if let Some(hs) = plan.outer_r2_hs {
        let hs_u = hs as u32;
        enc.set_compute_pipeline_state(&k.fft_outer_r2_f32);
        enc.set_bytes(1, 4, &off as *const u32 as *const _);
        enc.set_bytes(2, 4, &n as *const u32 as *const _);
        enc.set_bytes(3, 4, &hs_u as *const u32 as *const _);
        enc.set_bytes(4, 4, &inv as *const u32 as *const _);
        enc.set_bytes(5, 4, &norm_scale as *const f32 as *const _);
        enc.set_bytes(6, 4, &outer as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize {
                width: grid_1d(n / 2),
                height: outer as u64,
                depth: 1,
            },
            MTLSize {
                width: WG,
                height: 1,
                depth: 1,
            },
        );
    }

    let _ = dst_off;
}
