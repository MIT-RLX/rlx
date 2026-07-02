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

// Multi-kernel f32 FFT dispatch for wgpu (mirrors rlx-cuda/src/fft_dispatch.rs).

use crate::buffer::Arena;
use crate::kernels::{
    CopyParams, FftGpuParams, Kernel, copy_kernel, fft_gpu_bit_reverse_kernel,
    fft_gpu_inner_kernel, fft_gpu_outer_r2_kernel, fft_gpu_outer_r4_kernel,
    fft_gpu_radix2_full_kernel,
};
#[cfg(feature = "native-gpu-fft")]
use crate::kernels::{
    fft_gpu_big_r2_kernel, fft_gpu_big_r4_kernel, fft_gpu_big_r8_kernel, fft_gpu_multirow_kernel,
    fft_gpu_r4_16k_kernel, fft_gpu_radix2_full_big_kernel,
};

const WG: u32 = 256;

/// native-gpu-fft: largest `n` the portable (16 KB) single-kernel path handles
/// (radix-2, sh = 8·n bytes; 2048 → 16 KB = WebGPU's workgroup-storage floor).
#[cfg(feature = "native-gpu-fft")]
const BIG_FFT_MAX_N: u32 = 2048;

/// native-gpu-fft: largest `n` the 32 KB on-chip path handles (radix-2/4/8;
/// sh = 4096·vec2<f32> = 32 KB). Used only when the device reports >=32 KB
/// workgroup storage (Apple Silicon / desktop GPUs).
#[cfg(feature = "native-gpu-fft")]
const BIG32_FFT_MAX_N: u32 = 4096;

/// Workgroup-storage bytes the 32 KB on-chip kernels need.
#[cfg(feature = "native-gpu-fft")]
const WG_STORAGE_32K: u32 = 4096 * 8;

/// native-gpu-fft runtime gate. Default on; `RLX_FFT_FAST=0/off/false` routes
/// back to the multi-kernel path (for A/B benchmarking in one process).
#[cfg(feature = "native-gpu-fft")]
fn fast_fft_enabled() -> bool {
    !rlx_ir::env::var("RLX_FFT_FAST").is_some_and(|v| {
        v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")
    })
}

/// Max butterfly radix for the 32 KB on-chip path (best applicable per size is
/// chosen at dispatch). Default 8; `RLX_FFT_RADIX=2|4` caps it for A/B.
#[cfg(feature = "native-gpu-fft")]
fn fft_max_radix() -> u32 {
    match rlx_ir::env::var("RLX_FFT_RADIX").as_deref() {
        Some("2") => 2,
        Some("4") => 4,
        _ => 8,
    }
}

/// All wgpu on-chip (single-kernel) FFT is OPT-IN (default off): measured a net
/// regression that worsens with batch — a single big workgroup pins wgpu to few
/// resident workgroups per core, so FFT rows serialize, whereas the multi-kernel
/// path parallelizes across batch. (The same kernels win ~4× on native Metal,
/// whose scheduler tolerates large threadgroup memory.) The 16 KB radix-4 path
/// (n<=2048) is enabled by `RLX_FFT_WGPU_ONCHIP=1`; the 32 KB radix-8 path
/// (n<=4096) additionally needs `RLX_FFT_WGPU_BIG=1`.
#[cfg(feature = "native-gpu-fft")]
fn env_on(key: &str) -> bool {
    rlx_ir::env::var(key)
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true"))
}

#[cfg(feature = "native-gpu-fft")]
fn wgpu_onchip_enabled() -> bool {
    env_on("RLX_FFT_WGPU_ONCHIP") || env_on("RLX_FFT_WGPU_BIG")
}

#[cfg(feature = "native-gpu-fft")]
fn wgpu_big_enabled() -> bool {
    env_on("RLX_FFT_WGPU_BIG")
}

/// Multi-row small-n path (default OFF, opt-in via `RLX_FFT_MULTIROW=1`): packs
/// floor(2048/n) FFT rows per workgroup into a 16 KB shared buffer. Measured a
/// consistent 1.5–5× REGRESSION on wgpu (bench_fft_wgpu_multirow) — same
/// occupancy collapse as the 16/32 KB on-chip kernels: the large workgroup
/// buffer caps resident groups and packing serializes rows a group at a time,
/// while the default one-workgroup-per-row path spreads the batch across the
/// whole GPU. Kept behind the flag for other backends / future tuning.
#[cfg(feature = "native-gpu-fft")]
fn multirow_enabled() -> bool {
    rlx_ir::env::var("RLX_FFT_MULTIROW").is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("on"))
}

fn grid_1d(n: u32) -> u32 {
    n.div_ceil(WG)
}

/// wgpu caps every dispatch dimension at 65535 (`maxComputeWorkgroupsPerDimension`).
/// A flat copy of `outer·n·2` floats can far exceed that in the x dimension
/// (e.g. batch 512 × n 4096 → 262 144 groups), so spill the overflow into y.
/// The copy shader recovers the linear index as `gid.x + gid.y·num_workgroups.x·wg`.
fn dispatch_dims(n: u32, wg: u32) -> (u32, u32, u32) {
    let groups = n.div_ceil(wg).max(1);
    if groups <= 65535 {
        (groups, 1, 1)
    } else {
        let gx = 65535u32;
        (gx, groups.div_ceil(gx), 1)
    }
}

/// FFT rows (`outer`) can exceed 65535 (small-n huge-batch STFT). Split the row
/// count across the y and z dispatch dimensions; the per-row FFT kernels recover
/// `row = wgid.y + wgid.z · num_workgroups.y`. Returns `(grid_y, grid_z)`.
fn row_grid(outer: u32) -> (u32, u32) {
    let y = outer.clamp(1, 65535);
    (y, outer.div_ceil(y))
}

/// Per-stage uniform slot stride (must be ≥ `min_uniform_buffer_offset_alignment`,
/// 256 on all targets) and the max multi-kernel stages one FFT can emit
/// (bit-reverse + inner + up to ~log₄(n) radix-4 outer + 1 radix-2 ≈ 2 + 11).
const STAGE_STRIDE: u64 = 256;
const MAX_FFT_STAGES: u64 = 24;

/// Pre-built uniform buffers + bind groups for FFT stages (per executable).
pub struct FftGpuResources {
    pub uniform: wgpu::Buffer,
    pub copy_uniform: wgpu::Buffer,
    pub bg_radix2_full: wgpu::BindGroup,
    #[cfg(feature = "native-gpu-fft")]
    pub bg_radix2_full_big: wgpu::BindGroup,
    /// Portable 16 KB radix-4 (n<=2048) — the default on-chip path on wgpu.
    #[cfg(feature = "native-gpu-fft")]
    pub bg_r4_16k: wgpu::BindGroup,
    /// Multi-row small-n FFT (16 KB) — packs rows/workgroup for batched small n.
    #[cfg(feature = "native-gpu-fft")]
    pub bg_multirow: wgpu::BindGroup,
    /// 32 KB on-chip radix-2/4/8 bind groups — `Some` only when the device has
    /// >=32 KB workgroup storage (else those pipelines exceed the limit).
    #[cfg(feature = "native-gpu-fft")]
    pub bg_big_r2: Option<wgpu::BindGroup>,
    #[cfg(feature = "native-gpu-fft")]
    pub bg_big_r4: Option<wgpu::BindGroup>,
    #[cfg(feature = "native-gpu-fft")]
    pub bg_big_r8: Option<wgpu::BindGroup>,
    /// Per-stage uniform pool: each multi-kernel FFT stage writes its params to a
    /// DISTINCT slot. `queue.write_buffer` calls to one buffer all land before the
    /// command buffer runs, so a *shared* uniform makes every stage in the pass
    /// read the LAST stage's params — silently wrong once there are ≥2 outer
    /// stages (n ≥ 8192). Distinct slots (bind groups below index the same pool at
    /// different offsets) give each stage its own params.
    pub stage_uniform: wgpu::Buffer,
    pub bg_bit_reverse: Vec<wgpu::BindGroup>,
    pub bg_inner: Vec<wgpu::BindGroup>,
    pub bg_outer_r4: Vec<wgpu::BindGroup>,
    pub bg_outer_r2: Vec<wgpu::BindGroup>,
    pub bg_copy: wgpu::BindGroup,
}

impl FftGpuResources {
    pub fn new(device: &wgpu::Device, arena: &wgpu::Buffer) -> Self {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu fft uniform"),
            size: std::mem::size_of::<FftGpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let copy_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu fft copy uniform"),
            size: std::mem::size_of::<CopyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mk_bg = |k: &Kernel| k.bind_two(device, arena, &uniform);
        // Per-stage uniform pool + slot-indexed bind groups (see struct doc).
        let stage_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu fft stage uniform pool"),
            size: STAGE_STRIDE * MAX_FFT_STAGES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let param_size =
            std::num::NonZeroU64::new(std::mem::size_of::<FftGpuParams>() as u64).unwrap();
        let mk_slots = |k: &Kernel| -> Vec<wgpu::BindGroup> {
            (0..MAX_FFT_STAGES)
                .map(|s| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("rlx-wgpu fft stage bg"),
                        layout: &k.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: arena.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &stage_uniform,
                                    offset: s * STAGE_STRIDE,
                                    size: Some(param_size),
                                }),
                            },
                        ],
                    })
                })
                .collect()
        };
        // 32 KB on-chip kernels: only when opted in (they regress on wgpu) AND
        // the device actually has the workgroup storage.
        #[cfg(feature = "native-gpu-fft")]
        let big32 = wgpu_big_enabled()
            && device.limits().max_compute_workgroup_storage_size >= WG_STORAGE_32K;
        Self {
            bg_radix2_full: mk_bg(fft_gpu_radix2_full_kernel(device)),
            #[cfg(feature = "native-gpu-fft")]
            bg_radix2_full_big: mk_bg(fft_gpu_radix2_full_big_kernel(device)),
            #[cfg(feature = "native-gpu-fft")]
            bg_r4_16k: mk_bg(fft_gpu_r4_16k_kernel(device)),
            #[cfg(feature = "native-gpu-fft")]
            bg_multirow: mk_bg(fft_gpu_multirow_kernel(device)),
            #[cfg(feature = "native-gpu-fft")]
            bg_big_r2: big32.then(|| mk_bg(fft_gpu_big_r2_kernel(device))),
            #[cfg(feature = "native-gpu-fft")]
            bg_big_r4: big32.then(|| mk_bg(fft_gpu_big_r4_kernel(device))),
            #[cfg(feature = "native-gpu-fft")]
            bg_big_r8: big32.then(|| mk_bg(fft_gpu_big_r8_kernel(device))),
            bg_bit_reverse: mk_slots(fft_gpu_bit_reverse_kernel(device)),
            bg_inner: mk_slots(fft_gpu_inner_kernel(device)),
            bg_outer_r4: mk_slots(fft_gpu_outer_r4_kernel(device)),
            bg_outer_r2: mk_slots(fft_gpu_outer_r2_kernel(device)),
            bg_copy: copy_kernel(device).bind_two(device, arena, &copy_uniform),
            stage_uniform,
            uniform,
            copy_uniform,
        }
    }
}

fn dispatch_with_bg(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    gx: u32,
    gy: u32,
    gz: u32,
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bg, &[]);
    pass.dispatch_workgroups(gx, gy, gz);
}

/// Run FFT stages inside an existing compute pass (no extra submit/poll).
pub fn dispatch_fft_gpu_in_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &mut wgpu::ComputePass<'_>,
    res: &FftGpuResources,
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

    // native-gpu-fft (>=32 KB devices): whole transform on-chip for n in
    // (1024, 4096], highest applicable radix (8 for pow-8, 4 for the rest incl.
    // the 2·4^m mixed case, 2 as the floor). One workgroup per row, no DRAM
    // round-trips. `bg_big_r4.is_some()` gates on the device's workgroup storage.
    #[cfg(feature = "native-gpu-fft")]
    if fast_fft_enabled()
        && n > rlx_ir::fft::FFT_TILE_SIZE as u32
        && n <= BIG32_FFT_MAX_N
        && res.bg_big_r4.is_some()
    {
        let m = n.trailing_zeros();
        let max_radix = fft_max_radix();
        let (kernel, bg) = if max_radix >= 8 && m % 3 == 0 {
            (
                fft_gpu_big_r8_kernel(device),
                res.bg_big_r8.as_ref().unwrap(),
            )
        } else if max_radix >= 4 {
            (
                fft_gpu_big_r4_kernel(device),
                res.bg_big_r4.as_ref().unwrap(),
            )
        } else {
            (
                fft_gpu_big_r2_kernel(device),
                res.bg_big_r2.as_ref().unwrap(),
            )
        };
        let p = FftGpuParams {
            off: src_off,
            dst_off,
            n,
            log2n,
            inverse: inv,
            norm_scale,
            outer,
            tile: 0,
            inner_stages: 0,
            q_or_hs: 0,
        };
        queue.write_buffer(&res.uniform, 0, bytemuck::bytes_of(&p));
        let (gy, gz) = row_grid(outer);
        dispatch_with_bg(pass, &kernel.pipeline, bg, 1, gy, gz);
        return;
    }

    // native-gpu-fft (portable 16 KB, OPT-IN via RLX_FFT_WGPU_ONCHIP): whole
    // transform on-chip for n in (1024, 2048] via the right-sized radix-4 kernel.
    // Off by default — it regresses at high batch on wgpu (low occupancy); the
    // multi-kernel path parallelizes across batch and wins. Kept for the
    // GPU-resident/chained case where the dispatch-count reduction can help.
    #[cfg(feature = "native-gpu-fft")]
    if fast_fft_enabled()
        && wgpu_onchip_enabled()
        && n > rlx_ir::fft::FFT_TILE_SIZE as u32
        && n <= BIG_FFT_MAX_N
    {
        let p = FftGpuParams {
            off: src_off,
            dst_off,
            n,
            log2n,
            inverse: inv,
            norm_scale,
            outer,
            tile: 0,
            inner_stages: 0,
            q_or_hs: 0,
        };
        queue.write_buffer(&res.uniform, 0, bytemuck::bytes_of(&p));
        let (gy, gz) = row_grid(outer);
        dispatch_with_bg(
            pass,
            &fft_gpu_r4_16k_kernel(device).pipeline,
            &res.bg_r4_16k,
            1,
            gy,
            gz,
        );
        return;
    }

    if src_off != dst_off && !plan.single_inner_only() {
        let count = outer * n * 2;
        let cp = CopyParams {
            n: count,
            in_off: src_off,
            out_off: dst_off,
            _p0: 0,
            _p1: 0,
            _p2: 0,
            _p3: 0,
            _p4: 0,
        };
        queue.write_buffer(&res.copy_uniform, 0, bytemuck::bytes_of(&cp));
        let (gx, gy, gz) = dispatch_dims(count, 64);
        dispatch_with_bg(
            pass,
            &copy_kernel(device).pipeline,
            &res.bg_copy,
            gx,
            gy,
            gz,
        );
    }
    let off = dst_off;

    // native-gpu-fft: multi-row on-chip for SMALL n (single_inner_only, n<=1024).
    // The default small-n path launches one workgroup per row (256 threads for
    // n/2 butterflies) — for n<=512 that under-fills the workgroup. Pack
    // rows=floor(2048/n) rows per workgroup so a 256-thread group does useful
    // work across several rows (MLX's small-n threadgroup batching). Only helps
    // when rows>=2, i.e. n<=1024 with enough batch.
    #[cfg(feature = "native-gpu-fft")]
    if fast_fft_enabled() && multirow_enabled() && plan.single_inner_only() && outer >= 2 {
        let rows = (2048u32 / n).min(outer).max(1);
        // multirow packs its grid into x; if that would exceed 65535 fall through
        // to the single-inner path (which spills batch across y/z via row_grid).
        if rows >= 2 && outer.div_ceil(rows) <= 65535 {
            let p = FftGpuParams {
                off: src_off,
                dst_off,
                n,
                log2n,
                inverse: inv,
                norm_scale,
                outer,
                tile: rows,
                inner_stages: 0,
                q_or_hs: 0,
            };
            queue.write_buffer(&res.uniform, 0, bytemuck::bytes_of(&p));
            dispatch_with_bg(
                pass,
                &fft_gpu_multirow_kernel(device).pipeline,
                &res.bg_multirow,
                outer.div_ceil(rows),
                1,
                1,
            );
            return;
        }
    }

    if plan.single_inner_only() {
        let p = FftGpuParams {
            off: src_off,
            dst_off,
            n,
            log2n,
            inverse: inv,
            norm_scale,
            outer,
            tile: 0,
            inner_stages: 0,
            q_or_hs: 0,
        };
        queue.write_buffer(&res.uniform, 0, bytemuck::bytes_of(&p));
        let (gy, gz) = row_grid(outer);
        dispatch_with_bg(
            pass,
            &fft_gpu_radix2_full_kernel(device).pipeline,
            &res.bg_radix2_full,
            1,
            gy,
            gz,
        );
        return;
    }

    // Multi-kernel stages: batch (`outer`) spilled across y/z (row_grid), so
    // every per-row kernel recovers row = wgid.y + wgid.z·num_workgroups.y.
    // Each stage writes its params to its OWN slot of `stage_uniform` and binds
    // that slot — a shared uniform would alias (see FftGpuResources doc): every
    // `queue.write_buffer` lands before the pass runs, so a reused uniform makes
    // all stages read the last stage's params (wrong once there are ≥2 outer).
    let (gy, gz) = row_grid(outer);
    let mut p = FftGpuParams {
        off,
        dst_off,
        n,
        log2n,
        inverse: inv,
        norm_scale: 1.0,
        outer,
        tile: 0,
        inner_stages: 0,
        q_or_hs: 0,
    };
    let mut slot = 0usize;
    let write_slot = |p: &FftGpuParams, slot: usize| {
        debug_assert!((slot as u64) < MAX_FFT_STAGES, "fft: too many stages");
        queue.write_buffer(
            &res.stage_uniform,
            slot as u64 * STAGE_STRIDE,
            bytemuck::bytes_of(p),
        );
    };

    write_slot(&p, slot);
    dispatch_with_bg(
        pass,
        &fft_gpu_bit_reverse_kernel(device).pipeline,
        &res.bg_bit_reverse[slot],
        grid_1d(n),
        gy,
        gz,
    );
    slot += 1;

    let tile = rlx_ir::fft::FFT_TILE_SIZE.min(n as usize) as u32;
    let inner_stages = plan.inner_stages as u32;
    let num_tiles = (n / tile).max(1);
    p.tile = tile;
    p.inner_stages = inner_stages;
    p.norm_scale = 1.0;
    write_slot(&p, slot);
    dispatch_with_bg(
        pass,
        &fft_gpu_inner_kernel(device).pipeline,
        &res.bg_inner[slot],
        num_tiles,
        gy,
        gz,
    );
    slot += 1;

    let r4_count = plan.outer_rad4_q.len();
    for (i, q) in plan.outer_rad4_q.iter().enumerate() {
        p.q_or_hs = *q as u32;
        p.norm_scale = if plan.outer_r2_hs.is_none() && i + 1 == r4_count {
            norm_scale
        } else {
            1.0
        };
        write_slot(&p, slot);
        dispatch_with_bg(
            pass,
            &fft_gpu_outer_r4_kernel(device).pipeline,
            &res.bg_outer_r4[slot],
            grid_1d((n / 4).max(1)),
            gy,
            gz,
        );
        slot += 1;
    }

    if let Some(hs) = plan.outer_r2_hs {
        p.q_or_hs = hs as u32;
        p.norm_scale = norm_scale;
        write_slot(&p, slot);
        dispatch_with_bg(
            pass,
            &fft_gpu_outer_r2_kernel(device).pipeline,
            &res.bg_outer_r2[slot],
            grid_1d(n / 2),
            gy,
            gz,
        );
    }
}

/// Standalone FFT dispatch using compile-time cached resources.
pub fn run_fft_gpu_cached(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    _arena: &Arena,
    res: &FftGpuResources,
    src_off: u32,
    dst_off: u32,
    outer: u32,
    n: u32,
    inverse: bool,
    norm_scale: f32,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu fft gpu"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu fft gpu pass"),
            timestamp_writes: None,
        });
        dispatch_fft_gpu_in_pass(
            device, queue, &mut pass, res, src_off, dst_off, outer, n, inverse, norm_scale,
        );
    }
    queue.submit(std::iter::once(encoder.finish()));
}

/// Standalone FFT dispatch (legacy callers).
pub fn run_fft_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    arena: &Arena,
    src_off: u32,
    dst_off: u32,
    outer: u32,
    n: u32,
    inverse: bool,
    norm_scale: f32,
) {
    let res = FftGpuResources::new(device, &arena.buffer);
    run_fft_gpu_cached(
        device, queue, arena, &res, src_off, dst_off, outer, n, inverse, norm_scale,
    );
}
