// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! On-device Philox / Zero RNG fills for CUDA arenas, plus a warm-path cache
//! for Ort / Bnns (host-generated, device-resident) so Kitten's default Ort
//! vocoder noise is not re-H2D'd every infer.

use crate::kernels::{
    dispatch_grid_1d, rng_fill_zero_kernel, rng_normal_philox_kernel, rng_uniform_philox_kernel,
};
use crate::rng_host;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use rlx_ir::{RngBackend, RngOptions, combine_seed};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

type NormalKey = (u64, u64, u32, u32, u32, Option<u32>); // seed,key,len,mean,scale,op_seed_bits
type UniformKey = (u64, u64, u32, u32, u32, Option<u32>);

fn normal_cache() -> &'static Mutex<HashMap<NormalKey, CudaSlice<f32>>> {
    static C: OnceLock<Mutex<HashMap<NormalKey, CudaSlice<f32>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn uniform_cache() -> &'static Mutex<HashMap<UniformKey, CudaSlice<f32>>> {
    static C: OnceLock<Mutex<HashMap<UniformKey, CudaSlice<f32>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn op_seed_bits(op_seed: Option<f32>) -> Option<u32> {
    op_seed.map(f32::to_bits)
}

/// Fill normals on-device (Philox / Zero) or via a warm Ort/Bnns device cache.
/// Always returns `true` (Ort/Bnns cold path runs host fill inside).
#[allow(clippy::too_many_arguments)]
pub fn try_rng_normal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    arena: &mut CudaSlice<f32>,
    dst_byte_off: usize,
    len: usize,
    mean: f32,
    scale: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: RngOptions,
) -> bool {
    if len == 0 {
        return true;
    }
    debug_assert_eq!(dst_byte_off % 4, 0);
    let dst_off = (dst_byte_off / 4) as u32;
    let len_u = len as u32;
    match opts.backend {
        RngBackend::Zero => {
            launch_zero(ctx, stream, arena, dst_off, len_u);
            true
        }
        RngBackend::Philox => {
            let seed = combine_seed(opts.seed, key);
            let seed_lo = (seed & 0xFFFF_FFFF) as u32;
            let seed_hi = (seed >> 32) as u32;
            let kernel = rng_normal_philox_kernel(ctx);
            let (grid, block) = dispatch_grid_1d(len_u, 256);
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launcher = stream.launch_builder(&kernel.function);
            launcher
                .arg(&mut *arena)
                .arg(&dst_off)
                .arg(&len_u)
                .arg(&mean)
                .arg(&scale)
                .arg(&seed_lo)
                .arg(&seed_hi);
            unsafe {
                launcher
                    .launch(cfg)
                    .expect("rlx-cuda: rng_normal_philox launch failed");
            }
            true
        }
        RngBackend::Ort | RngBackend::Bnns => {
            // Fixed seed → identical stream every infer. Cache on device after the
            // first host fill so warm Kitten waves skip the ~8 ms H2D.
            let ck: NormalKey = (
                opts.seed,
                key,
                len_u,
                mean.to_bits(),
                scale.to_bits(),
                op_seed_bits(op_seed),
            );
            {
                let cache = normal_cache().lock().unwrap();
                if let Some(cached) = cache.get(&ck) {
                    let src = cached.slice(0..len);
                    let mut dst = arena.slice_mut(dst_off as usize..(dst_off as usize + len));
                    stream
                        .memcpy_dtod(&src, &mut dst)
                        .expect("rlx-cuda: rng normal cache dtod");
                    return true;
                }
            }
            rng_host::run_rng_normal(
                stream,
                arena,
                dst_byte_off,
                len,
                mean,
                scale,
                key,
                op_seed,
                opts,
            );
            let mut copy = stream
                .alloc_zeros::<f32>(len.max(1))
                .expect("rlx-cuda: rng normal cache alloc");
            {
                let src = arena.slice(dst_off as usize..(dst_off as usize + len));
                stream
                    .memcpy_dtod(&src, &mut copy)
                    .expect("rlx-cuda: rng normal cache capture");
            }
            normal_cache().lock().unwrap().insert(ck, copy);
            true
        }
    }
}

/// Fill uniforms on-device / via Ort·Bnns warm cache. Always returns `true`.
#[allow(clippy::too_many_arguments)]
pub fn try_rng_uniform(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    arena: &mut CudaSlice<f32>,
    dst_byte_off: usize,
    len: usize,
    low: f32,
    high: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: RngOptions,
) -> bool {
    if len == 0 {
        return true;
    }
    debug_assert_eq!(dst_byte_off % 4, 0);
    let dst_off = (dst_byte_off / 4) as u32;
    let len_u = len as u32;
    match opts.backend {
        RngBackend::Zero => {
            launch_zero(ctx, stream, arena, dst_off, len_u);
            true
        }
        RngBackend::Philox => {
            let seed = combine_seed(opts.seed, key);
            let seed_lo = (seed & 0xFFFF_FFFF) as u32;
            let seed_hi = (seed >> 32) as u32;
            let kernel = rng_uniform_philox_kernel(ctx);
            let (grid, block) = dispatch_grid_1d(len_u, 256);
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launcher = stream.launch_builder(&kernel.function);
            launcher
                .arg(&mut *arena)
                .arg(&dst_off)
                .arg(&len_u)
                .arg(&low)
                .arg(&high)
                .arg(&seed_lo)
                .arg(&seed_hi);
            unsafe {
                launcher
                    .launch(cfg)
                    .expect("rlx-cuda: rng_uniform_philox launch failed");
            }
            true
        }
        RngBackend::Ort | RngBackend::Bnns => {
            let ck: UniformKey = (
                opts.seed,
                key,
                len_u,
                low.to_bits(),
                high.to_bits(),
                op_seed_bits(op_seed),
            );
            {
                let cache = uniform_cache().lock().unwrap();
                if let Some(cached) = cache.get(&ck) {
                    let src = cached.slice(0..len);
                    let mut dst = arena.slice_mut(dst_off as usize..(dst_off as usize + len));
                    stream
                        .memcpy_dtod(&src, &mut dst)
                        .expect("rlx-cuda: rng uniform cache dtod");
                    return true;
                }
            }
            rng_host::run_rng_uniform(
                stream,
                arena,
                dst_byte_off,
                len,
                low,
                high,
                key,
                op_seed,
                opts,
            );
            let mut copy = stream
                .alloc_zeros::<f32>(len.max(1))
                .expect("rlx-cuda: rng uniform cache alloc");
            {
                let src = arena.slice(dst_off as usize..(dst_off as usize + len));
                stream
                    .memcpy_dtod(&src, &mut copy)
                    .expect("rlx-cuda: rng uniform cache capture");
            }
            uniform_cache().lock().unwrap().insert(ck, copy);
            true
        }
    }
}

fn launch_zero(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    arena: &mut CudaSlice<f32>,
    dst_off: u32,
    len: u32,
) {
    let kernel = rng_fill_zero_kernel(ctx);
    let (grid, block) = dispatch_grid_1d(len.max(1), 256);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher.arg(&mut *arena).arg(&dst_off).arg(&len);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: rng_fill_zero launch failed");
    }
}
