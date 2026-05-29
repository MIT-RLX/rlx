// Multi-kernel f32 FFT dispatch for Metal (mirrors rlx-cuda/src/fft_dispatch.rs).

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::kernels::Kernels;

const WG: u64 = 256;

fn grid_1d(n: u32) -> u64 {
    (n as u64).div_ceil(WG)
}

/// Run native multi-kernel FFT on the unified-memory arena (f32, pow-2 `n`).
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
) {
    if outer == 0 {
        return;
    }
    let plan = rlx_ir::fft::FftGpuPlan::new(n as usize).expect("run_fft_gpu: n must be pow2");
    let inv = if inverse { 1u32 } else { 0u32 };
    let log2n = n.trailing_zeros();
    if src_off != dst_off && !plan.single_inner_only() {
        let count = outer as usize * n as usize * 2;
        let ptr = arena.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.add(src_off as usize),
                ptr.add(dst_off as usize),
                count,
            );
        }
    }
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

    enc.set_compute_pipeline_state(&k.fft_bit_reverse_f32);
    enc.set_bytes(1, 4, &off as *const u32 as *const _);
    enc.set_bytes(2, 4, &n as *const u32 as *const _);
    enc.set_bytes(3, 4, &log2n as *const u32 as *const _);
    enc.set_bytes(4, 4, &outer as *const u32 as *const _);
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
