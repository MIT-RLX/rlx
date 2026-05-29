// RLX — versatile ML compiler + runtime.
// Multi-kernel f32 FFT dispatch (gpu-fft strategy, RLX 2N layout).

use std::sync::Arc;

use crate::device::RocmContext;
use crate::hip::HipDeviceptr;
use crate::kernels::{
    dispatch_grid_1d, fft_bit_reverse_kernel, fft_inner_kernel, fft_outer_r2_kernel,
    fft_outer_r4_kernel, fft_radix2_full_kernel,
};

const WG: u32 = 256;

fn launch_fft(
    kernel: &crate::hip::HipKernel,
    stream: crate::hip::HipStream,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem_bytes: u32,
    args: &mut [*mut core::ffi::c_void],
) {
    unsafe {
        let _ = kernel.launch(stream, grid, block, shared_mem_bytes, args.as_mut_ptr());
    }
}

/// Run native GPU FFT on the device arena (f32, pow-2 `n`).
pub fn run_fft_gpu(
    ctx: &Arc<RocmContext>,
    stream: crate::hip::HipStream,
    arena_ptr: HipDeviceptr,
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
        let byte_src = (src_off as u64) * 4;
        let byte_dst = (dst_off as u64) * 4;
        let bytes = ((outer as u64) * (n as u64) * 2 * 4) as usize;
        unsafe {
            let _ =
                (ctx.runtime.hip_memcpy_dtod)(arena_ptr + byte_dst, arena_ptr + byte_src, bytes);
        }
    }
    let off = dst_off;
    let scale1 = 1.0f32;
    let mut arena_ptr_mut = arena_ptr;

    if plan.single_inner_only() {
        let kernel = fft_radix2_full_kernel(ctx);
        let block = n.min(256);
        let mut params = [
            &mut arena_ptr_mut as *const _ as *mut core::ffi::c_void,
            &src_off as *const _ as *mut core::ffi::c_void,
            &dst_off as *const _ as *mut core::ffi::c_void,
            &n as *const _ as *mut core::ffi::c_void,
            &log2n as *const _ as *mut core::ffi::c_void,
            &inv as *const _ as *mut core::ffi::c_void,
            &norm_scale as *const _ as *mut core::ffi::c_void,
            &outer as *const _ as *mut core::ffi::c_void,
        ];
        launch_fft(
            kernel,
            stream,
            (1, outer, 1),
            (block, 1, 1),
            8192,
            &mut params,
        );
        return;
    }

    {
        let kernel = fft_bit_reverse_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(n, WG);
        let mut params = [
            &mut arena_ptr_mut as *const _ as *mut core::ffi::c_void,
            &off as *const _ as *mut core::ffi::c_void,
            &n as *const _ as *mut core::ffi::c_void,
            &log2n as *const _ as *mut core::ffi::c_void,
            &outer as *const _ as *mut core::ffi::c_void,
        ];
        launch_fft(
            kernel,
            stream,
            (grid, outer, 1),
            (block, 1, 1),
            0,
            &mut params,
        );
    }

    let tile = rlx_ir::fft::FFT_TILE_SIZE.min(n as usize) as u32;
    let inner_stages = plan.inner_stages as u32;
    let num_tiles = (n / tile).max(1);
    let wg_threads = (n / 2).min(tile / 2);

    {
        let kernel = fft_inner_kernel(ctx);
        let mut params = [
            &mut arena_ptr_mut as *const _ as *mut core::ffi::c_void,
            &off as *const _ as *mut core::ffi::c_void,
            &n as *const _ as *mut core::ffi::c_void,
            &tile as *const _ as *mut core::ffi::c_void,
            &inner_stages as *const _ as *mut core::ffi::c_void,
            &inv as *const _ as *mut core::ffi::c_void,
            &scale1 as *const _ as *mut core::ffi::c_void,
            &outer as *const _ as *mut core::ffi::c_void,
        ];
        launch_fft(
            kernel,
            stream,
            (num_tiles, outer, 1),
            (wg_threads, 1, 1),
            tile * 8,
            &mut params,
        );
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
        let mut params = [
            &mut arena_ptr_mut as *const _ as *mut core::ffi::c_void,
            &off as *const _ as *mut core::ffi::c_void,
            &n as *const _ as *mut core::ffi::c_void,
            &q_u as *const _ as *mut core::ffi::c_void,
            &inv as *const _ as *mut core::ffi::c_void,
            &stage_scale as *const _ as *mut core::ffi::c_void,
            &outer as *const _ as *mut core::ffi::c_void,
        ];
        launch_fft(
            kernel,
            stream,
            (grid, outer, 1),
            (block, 1, 1),
            0,
            &mut params,
        );
    }

    if let Some(hs) = plan.outer_r2_hs {
        let hs_u = hs as u32;
        let kernel = fft_outer_r2_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(n / 2, WG);
        let mut params = [
            &mut arena_ptr_mut as *const _ as *mut core::ffi::c_void,
            &off as *const _ as *mut core::ffi::c_void,
            &n as *const _ as *mut core::ffi::c_void,
            &hs_u as *const _ as *mut core::ffi::c_void,
            &inv as *const _ as *mut core::ffi::c_void,
            &norm_scale as *const _ as *mut core::ffi::c_void,
            &outer as *const _ as *mut core::ffi::c_void,
        ];
        launch_fft(
            kernel,
            stream,
            (grid, outer, 1),
            (block, 1, 1),
            0,
            &mut params,
        );
    }
}
