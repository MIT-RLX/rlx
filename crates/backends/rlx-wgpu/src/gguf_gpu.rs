// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// GPL-3.0-only. See LICENSE.

//! GPU GGUF dequant + wgpu matmul for `Op::DequantMatMul`.
//!
//! Flow: `dequant_gguf` compute (scheme ids 0–23, shared with Metal/CUDA) writes
//! f32 `[n,k]` into arena scratch, then `matmul_bt` (`C = X @ W^T`).
//!
//! **When the GPU path runs:** `dequant_scratch_off > 0` after arena planning
//! (scratch must fit `device.limits().max_buffer_size`). Otherwise
//! [`crate::gguf_host`] handles dequant + matmul on CPU.
//!
//! **Grouped MoE:** [`run_dequant_grouped_matmul_gguf_gpu`] when scratch fits;
//! otherwise [`crate::gguf_host::run_dequant_grouped_matmul_gguf`].
//!
//! **Limits:** arena byte offsets are u32; IQ-family branches mirror Metal/CUDA
//! but are not covered by dedicated WGPU parity tests yet.
//!
//! Full matrix: [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md).

use rlx_ir::{Graph, Op};

use crate::buffer::Arena;
use crate::gguf_host::scheme_from_id;
use crate::kernels::{
    DequantGemvGgufParams, DequantGgufParams, Kernel, MatmulParams, dequant_gemv_gguf_kernel,
    dequant_gguf_kernel, matmul_bt_kernel,
};

/// Schemes the fused decode GEMV ([`run_dequant_matmul_gguf_gemv`]) handles
/// on-GPU without f32 scratch. Q4_K (0) + Q6_K (2) cover Llama Q4_K_M GGUFs
/// (q/k/o/gate/up are Q4_K; v/down/embed are Q6_K).
pub fn gemv_supports_scheme(scheme_id: u32) -> bool {
    matches!(scheme_id, 0 | 2)
}

/// Max f32 scratch for dequantized weights `[n, k]` across all GGUF ops.
pub fn dequant_gguf_scratch_bytes(graph: &Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        if let Op::DequantMatMul { scheme } = &node.op
            && scheme.is_gguf()
        {
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            let total = node.shape.num_elements().unwrap();
            let m = total / n.max(1);
            // Decode GEMV (m==1, Q4_K/Q6_K) runs scratch-free via
            // `run_dequant_matmul_gguf_gemv`, so it needs no f32 slab. Skipping
            // it here keeps the arena from reserving the (multi-GiB) dequant
            // scratch on pure decode graphs — and avoids binding >4 GiB.
            if m == 1 && gemv_supports_scheme(crate::gguf_host::gguf_scheme_id(*scheme)) {
                continue;
            }
            let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
            let k = x_total / m.max(1);
            max = max.max(k * n * std::mem::size_of::<f32>());
        }
        if let Op::DequantGroupedMatMul { scheme: _ } = &node.op {
            let in_shape = &graph.node(node.inputs[0]).shape;
            let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
            let k = in_shape.dim(in_shape.rank() - 1).unwrap_static();
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            max = max.max(k * n * 4 + m * k * 4 + m * n * 4);
        }
    }
    max
}

fn slab_bytes_for(scheme: rlx_ir::quant::QuantScheme, k: usize, n: usize) -> usize {
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    (k * n) / block_elems * block_bytes
}

fn launch_dequant_gguf(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    w_byte_off: usize,
    scratch_byte_off: usize,
    scheme_id: u32,
    num_blocks: usize,
) {
    let dk = dequant_gguf_kernel(device);
    let lut = crate::iq_grid::wgpu_iq_grid_buffer(device, queue);
    let p = DequantGgufParams {
        w_byte_off: w_byte_off as u32,
        dst_f32_off: (scratch_byte_off / 4) as u32,
        scheme_id,
        num_blocks: num_blocks as u32,
    };
    let u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu dequant_gguf uniform"),
        size: std::mem::size_of::<DequantGgufParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
    let bg = bind_dequant_gguf(device, dk, &arena.buffer, &u, &lut);

    let block = 256u32.min(num_blocks as u32).max(1);
    let grid = num_blocks.div_ceil(block as usize) as u32;

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu dequant_gguf"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu dequant_gguf pass"),
            ..Default::default()
        });
        pass.set_pipeline(&dk.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(grid, 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));
}

fn bind_dequant_gguf(
    device: &wgpu::Device,
    kernel: &Kernel,
    arena: &wgpu::Buffer,
    uniform: &wgpu::Buffer,
    lut: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu dequant_gguf bg"),
        layout: &kernel.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: arena.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: lut.as_entire_binding(),
            },
        ],
    })
}

fn dispatch_matmul_bt(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    arena: &wgpu::Buffer,
    m: u32,
    k: u32,
    n: u32,
    x_off_f32: u32,
    w_off_f32: u32,
    out_off_f32: u32,
) {
    let mm = matmul_bt_kernel(device);
    let p = MatmulParams {
        m,
        k,
        n,
        a_off: x_off_f32,
        b_off: w_off_f32,
        c_off: out_off_f32,
        batch: 1,
        a_batch_stride: m * k,
        b_batch_stride: 0,
        c_batch_stride: m * n,
        has_bias: 0,
        bias_off: 0,
        act_id: 0xFFFF,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu gguf matmul_bt uniform"),
        size: std::mem::size_of::<MatmulParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu gguf matmul_bt bg"),
        layout: &mm.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: arena.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu gguf matmul_bt"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu gguf matmul_bt pass"),
            ..Default::default()
        });
        pass.set_pipeline(&mm.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(32), m.div_ceil(32), 1);
    }
    queue.submit(std::iter::once(enc.finish()));
}

/// Launch `dequant_gguf` into arena scratch, then `C = X @ W^T` via matmul_bt.
pub fn run_dequant_matmul_gguf_gpu(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    m: usize,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    scratch_byte_off: usize,
    out_byte_off: usize,
) {
    let scheme = scheme_from_id(scheme_id);
    let block_elems = scheme.gguf_block_size() as usize;
    let num_blocks = (k * n) / block_elems.max(1);

    launch_dequant_gguf(
        arena,
        device,
        queue,
        w_byte_off,
        scratch_byte_off,
        scheme_id,
        num_blocks,
    );

    dispatch_matmul_bt(
        device,
        queue,
        &arena.buffer,
        m as u32,
        k as u32,
        n as u32,
        (x_byte_off / 4) as u32,
        (scratch_byte_off / 4) as u32,
        (out_byte_off / 4) as u32,
    );
}

const STORAGE_ALIGN: u64 = 256;

/// Fused decode GEMV: `y[1,n] = x[1,k] @ W^T` with `W` GGUF-packed `[n,k]`,
/// dequantizing each weight block on the fly (no f32 scratch).
///
/// `x` and `weight` are bound as separate **read-only windows** of the arena
/// (each < 4 GiB; the whole-arena `as_entire_binding` overruns the binding
/// limit on multi-GiB models). `y` goes to a small **separate** output buffer —
/// the arena cannot also be bound read-write in the same dispatch (wgpu treats
/// STORAGE_READ_WRITE as exclusive) — which is then copied back into the arena.
///
/// Caller guarantees `m == 1` and [`gemv_supports_scheme`].
#[allow(clippy::too_many_arguments)]
pub fn run_dequant_matmul_gguf_gemv(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) {
    let scheme = scheme_from_id(scheme_id);
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let w_total_bytes = (k * n) / block_elems.max(1) * block_bytes;
    let arena_size = arena.size as u64;

    // x window: cover [x_byte_off, +k*4).
    let x0 = x_byte_off as u64;
    let x_base = (x0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let x_size = ((x0 + (k * 4) as u64 - x_base).div_ceil(16) * 16).min(arena_size - x_base);

    // weight window: cover [w_byte_off, +w_total_bytes).
    let w0 = w_byte_off as u64;
    let w_base = (w0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let w_size = ((w0 + w_total_bytes as u64 - w_base).div_ceil(16) * 16).min(arena_size - w_base);

    let max_bind = device.limits().max_storage_buffer_binding_size;
    assert!(
        x_size <= max_bind && w_size <= max_bind,
        "rlx-wgpu gguf gemv: window too large (x={x_size}, w={w_size}, max={max_bind})"
    );

    // Separate output buffer (rw) — copied into the arena after the dispatch.
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu dequant_gemv_gguf out"),
        size: ((n * 4).max(4) as u64).div_ceil(16) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let p = DequantGemvGgufParams {
        k: k as u32,
        n: n as u32,
        scheme_id,
        x_f32_off: ((x0 - x_base) / 4) as u32,
        w_byte_off: (w0 - w_base) as u32,
        out_f32_off: 0,
        _p0: 0,
        _p1: 0,
    };

    let dk = dequant_gemv_gguf_kernel(device);
    let u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu dequant_gemv_gguf uniform"),
        size: std::mem::size_of::<DequantGemvGgufParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu dequant_gemv_gguf bg"),
        layout: &dk.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &arena.buffer,
                    offset: x_base,
                    size: wgpu::BufferSize::new(x_size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &arena.buffer,
                    offset: w_base,
                    size: wgpu::BufferSize::new(w_size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu dequant_gemv_gguf"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu dequant_gemv_gguf pass"),
            ..Default::default()
        });
        pass.set_pipeline(&dk.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    // Copy y back into the arena (distinct buffers → buffer-to-buffer is legal).
    enc.copy_buffer_to_buffer(
        &out_buf,
        0,
        &arena.buffer,
        out_byte_off as u64,
        (n * 4) as u64,
    );
    queue.submit(std::iter::once(enc.finish()));
}

/// GPU dequant + grouped matmul for MoE packed expert stacks.
///
/// Scratch layout at `scratch_byte_off` (f32 bytes):
///   `[0 .. k*n)`: dequantized expert slab
///   `[k*n .. k*n+m*k)`: sorted token inputs
///   `[k*n+m*k .. k*n+m*k+m*n)`: sorted outputs before unpermute
pub fn run_dequant_grouped_matmul_gguf_gpu(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    idx_byte_off: usize,
    scratch_byte_off: usize,
    out_byte_off: usize,
) {
    let scheme = scheme_from_id(scheme_id);
    let slab_bytes = slab_bytes_for(scheme, k, n);
    let num_blocks = (k * n) / scheme.gguf_block_size() as usize;

    let x_bytes = arena.read_bytes_range(device, queue, x_byte_off, m * k * 4);
    let x_host: Vec<f32> = x_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let idx_bytes = arena.read_bytes_range(device, queue, idx_byte_off, m * 4);
    let idx_host: Vec<f32> = idx_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let (packed_in, original_pos, offsets) =
        rlx_cpu::gguf_matmul::grouped_moe_sort_plan(&x_host, &idx_host, m, k, num_experts);

    let dequant_off = scratch_byte_off;
    let pack_in_off = scratch_byte_off + k * n * 4;
    let pack_out_off = scratch_byte_off + (k * n + m * k) * 4;

    let pack_in_bytes: Vec<u8> = packed_in.iter().flat_map(|v| v.to_le_bytes()).collect();
    arena.write_bytes_range(queue, pack_in_off, &pack_in_bytes);

    for e in 0..num_experts {
        let count = offsets[e + 1] - offsets[e];
        if count == 0 {
            continue;
        }
        let w_off = w_byte_off + e * slab_bytes;
        launch_dequant_gguf(
            arena,
            device,
            queue,
            w_off,
            dequant_off,
            scheme_id,
            num_blocks,
        );
        let in_start = offsets[e];
        dispatch_matmul_bt(
            device,
            queue,
            &arena.buffer,
            count as u32,
            k as u32,
            n as u32,
            (pack_in_off / 4 + in_start * k) as u32,
            (dequant_off / 4) as u32,
            (pack_out_off / 4 + in_start * n) as u32,
        );
    }

    let pack_out_bytes = arena.read_bytes_range(device, queue, pack_out_off, m * n * 4);
    let packed_out: Vec<f32> = pack_out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut out_host = vec![0f32; m * n];
    rlx_cpu::gguf_matmul::grouped_moe_unpermute_out(
        &packed_out,
        &original_pos,
        &mut out_host,
        m,
        n,
    );

    let out_bytes: Vec<u8> = out_host.iter().flat_map(|v| v.to_le_bytes()).collect();
    arena.write_bytes_range(queue, out_byte_off, &out_bytes);
}
