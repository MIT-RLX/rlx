// RLX — versatile ML compiler + runtime.
// Multi-kernel f32 FFT dispatch (gpu-fft strategy, RLX 2N layout).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::kernels::{
    copy_kernel, dispatch_grid_1d, fft_bit_reverse_kernel, fft_inner_kernel, fft_outer_r2_kernel,
    fft_outer_r4_kernel, fft_radix2_full_kernel,
};

const WG: u32 = 256;

/// Run native GPU FFT on the device arena (f32, pow-2 `n`).
pub fn run_fft_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    src_off: u32,
    dst_off: u32,
    outer: u32,
    n: u32,
    inverse: bool,
    norm_scale: f32,
) {
    let plan = rlx_ir::fft::FftGpuPlan::new(n as usize).expect("run_fft_gpu: n must be pow2");
    let inv = if inverse { 1u32 } else { 0u32 };
    let log2n = n.trailing_zeros();
    if src_off != dst_off && !plan.single_inner_only() {
        let count = outer * n * 2;
        let kernel = copy_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(count, 64);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&count)
            .arg(&src_off)
            .arg(&dst_off);
        unsafe {
            launcher
                .launch(cfg)
                .expect("run_fft_gpu: copy kernel launch failed");
        }
    }
    let off = dst_off;
    let scale1 = 1.0f32;

    if plan.single_inner_only() {
        let kernel = fft_radix2_full_kernel(ctx);
        let block = n.min(256);
        let cfg = LaunchConfig {
            grid_dim: (1, outer, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 8192,
        };
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&src_off)
            .arg(&dst_off)
            .arg(&n)
            .arg(&log2n)
            .arg(&inv)
            .arg(&norm_scale)
            .arg(&outer);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: fft_radix2_full launch failed");
        }
        return;
    }

    {
        let kernel = fft_bit_reverse_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(n, WG);
        let cfg = LaunchConfig {
            grid_dim: (grid, outer, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&off)
            .arg(&n)
            .arg(&log2n)
            .arg(&outer);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: fft_bit_reverse launch failed");
        }
    }

    let tile = rlx_ir::fft::FFT_TILE_SIZE.min(n as usize) as u32;
    let inner_stages = plan.inner_stages as u32;
    let num_tiles = (n / tile).max(1);
    let wg_threads = (n / 2).min(tile / 2);

    {
        let kernel = fft_inner_kernel(ctx);
        let cfg = LaunchConfig {
            grid_dim: (num_tiles, outer, 1),
            block_dim: (wg_threads, 1, 1),
            shared_mem_bytes: tile * 8,
        };
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&off)
            .arg(&n)
            .arg(&tile)
            .arg(&inner_stages)
            .arg(&inv)
            .arg(&scale1)
            .arg(&outer);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: fft_inner launch failed");
        }
    }

    let r4_count = plan.outer_rad4_q.len();
    for (i, q) in plan.outer_rad4_q.iter().enumerate() {
        let q_u = *q as u32;
        let stage_scale = if plan.outer_r2_hs.is_none() && i + 1 == r4_count {
            norm_scale
        } else {
            1.0f32
        };
        let kernel = fft_outer_r4_kernel(ctx);
        let (grid, block) = dispatch_grid_1d((n / 4).max(1), WG);
        let cfg = LaunchConfig {
            grid_dim: (grid, outer, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&off)
            .arg(&n)
            .arg(&q_u)
            .arg(&inv)
            .arg(&stage_scale)
            .arg(&outer);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: fft_outer_r4 launch failed");
        }
    }

    if let Some(hs) = plan.outer_r2_hs {
        let hs_u = hs as u32;
        let kernel = fft_outer_r2_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(n / 2, WG);
        let cfg = LaunchConfig {
            grid_dim: (grid, outer, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&off)
            .arg(&n)
            .arg(&hs_u)
            .arg(&inv)
            .arg(&norm_scale)
            .arg(&outer);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: fft_outer_r2 launch failed");
        }
    }
}
