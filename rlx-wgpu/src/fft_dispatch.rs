// Multi-kernel f32 FFT dispatch for wgpu (mirrors rlx-cuda/src/fft_dispatch.rs).

use crate::buffer::Arena;
use crate::kernels::{
    CopyParams, FftGpuParams, Kernel, copy_kernel, fft_gpu_bit_reverse_kernel,
    fft_gpu_inner_kernel, fft_gpu_outer_r2_kernel, fft_gpu_outer_r4_kernel,
    fft_gpu_radix2_full_kernel,
};

const WG: u32 = 256;

fn grid_1d(n: u32) -> u32 {
    n.div_ceil(WG)
}

fn dispatch_dims(n: u32, wg: u32) -> (u32, u32, u32) {
    (n.div_ceil(wg).max(1), 1, 1)
}

/// Pre-built uniform buffers + bind groups for FFT stages (per executable).
pub struct FftGpuResources {
    pub uniform: wgpu::Buffer,
    pub copy_uniform: wgpu::Buffer,
    pub bg_radix2_full: wgpu::BindGroup,
    pub bg_bit_reverse: wgpu::BindGroup,
    pub bg_inner: wgpu::BindGroup,
    pub bg_outer_r4: wgpu::BindGroup,
    pub bg_outer_r2: wgpu::BindGroup,
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
        Self {
            bg_radix2_full: mk_bg(fft_gpu_radix2_full_kernel(device)),
            bg_bit_reverse: mk_bg(fft_gpu_bit_reverse_kernel(device)),
            bg_inner: mk_bg(fft_gpu_inner_kernel(device)),
            bg_outer_r4: mk_bg(fft_gpu_outer_r4_kernel(device)),
            bg_outer_r2: mk_bg(fft_gpu_outer_r2_kernel(device)),
            bg_copy: copy_kernel(device).bind_two(device, arena, &copy_uniform),
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
        dispatch_with_bg(
            pass,
            &fft_gpu_radix2_full_kernel(device).pipeline,
            &res.bg_radix2_full,
            1,
            outer,
            1,
        );
        return;
    }

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

    queue.write_buffer(&res.uniform, 0, bytemuck::bytes_of(&p));
    dispatch_with_bg(
        pass,
        &fft_gpu_bit_reverse_kernel(device).pipeline,
        &res.bg_bit_reverse,
        grid_1d(n),
        outer,
        1,
    );

    let tile = rlx_ir::fft::FFT_TILE_SIZE.min(n as usize) as u32;
    let inner_stages = plan.inner_stages as u32;
    let num_tiles = (n / tile).max(1);
    p.tile = tile;
    p.inner_stages = inner_stages;
    p.norm_scale = 1.0;
    queue.write_buffer(&res.uniform, 0, bytemuck::bytes_of(&p));
    dispatch_with_bg(
        pass,
        &fft_gpu_inner_kernel(device).pipeline,
        &res.bg_inner,
        num_tiles,
        outer,
        1,
    );

    let r4_count = plan.outer_rad4_q.len();
    for (i, q) in plan.outer_rad4_q.iter().enumerate() {
        p.q_or_hs = *q as u32;
        p.norm_scale = if plan.outer_r2_hs.is_none() && i + 1 == r4_count {
            norm_scale
        } else {
            1.0
        };
        queue.write_buffer(&res.uniform, 0, bytemuck::bytes_of(&p));
        dispatch_with_bg(
            pass,
            &fft_gpu_outer_r4_kernel(device).pipeline,
            &res.bg_outer_r4,
            grid_1d((n / 4).max(1)),
            outer,
            1,
        );
    }

    if let Some(hs) = plan.outer_r2_hs {
        p.q_or_hs = hs as u32;
        p.norm_scale = norm_scale;
        queue.write_buffer(&res.uniform, 0, bytemuck::bytes_of(&p));
        dispatch_with_bg(
            pass,
            &fft_gpu_outer_r2_kernel(device).pipeline,
            &res.bg_outer_r2,
            grid_1d(n / 2),
            outer,
            1,
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
