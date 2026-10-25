// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU GGUF dequant + wgpu matmul for `Op::DequantMatMul`.
//!
//! Preferred path: scratch-free windowed GEMV ([`run_dequant_matmul_gguf_gemv_rows`])
//! for Q4_K / Q6_K / Q1_0 (decode `m=1` and prefill `m>1`). Older scratch+`matmul_bt`
//! path still exists for schemes without a GEMV kernel.
//!
//! **Host fallback:** [`crate::gguf_host`] when the scheme is unsupported on GPU
//! and no dequant scratch was reserved.
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
use crate::kernels::GemvEntry;
use crate::kernels::{
    DequantGemmQ10Params, DequantGemvGgufParams, DequantGgufParams, Kernel, MatmulParams,
    dequant_gemm_q1_0_kernel, dequant_gemm_q8_0_kernel, dequant_gemv_gguf_kernel,
    dequant_gguf_kernel, matmul_bt_kernel,
};

/// Schemes the fused decode GEMV ([`run_dequant_matmul_gguf_gemv`]) handles
/// on-GPU without f32 scratch. Q4_K (0) + Q6_K (2) cover Llama Q4_K_M GGUFs
/// (q/k/o/gate/up are Q4_K; v/down/embed are Q6_K). Q1_0 (24) is the prism-ml
/// Bonsai-27B 1-bit scheme — scratch-free here avoids re-materializing the whole
/// 27B to f32 every token (~200s/tok → fused) AND avoids the multi-GiB dequant
/// scratch that blew wgpu's storage-buffer binding limit.
pub fn gemv_supports_scheme(scheme_id: u32) -> bool {
    matches!(scheme_id, 0 | 1 | 2 | 20 | 24)
        || b32_scheme(scheme_id)
        || scheme_id == crate::gguf_host::BF16_SCHEME_ID
}

/// The 32-weight block formats `dequant_gemv_b32x16` reads: IQ4_NL (6),
/// IQ4_XS (7, as 32-weight sub-blocks), Q4_0 (19), Q8_0 (20), Q4_1 (21),
/// Q5_0 (22), Q5_1 (23), Q2_0 (25, as 32-weight sub-blocks of its 128).
///
/// Q2_0 used to fall to the one-thread-per-column `dequant_gemv` — and before
/// that to no fused GEMV at all, so every ternary projection of a decode step
/// (197 on KittenTTS 2's Qwen3-1.7B) took the dequant-to-scratch path with its
/// own submit and host sync: 3.4 s per token on Apple wgpu.
fn b32_scheme(scheme_id: u32) -> bool {
    matches!(scheme_id, 6 | 7 | 19 | 20 | 21 | 22 | 23 | 25)
}

/// Whether `graph` has packed (U8/I8) params and every one of them is only
/// the weight operand of a GGUF matmul the fused GEMV runs on the GPU — the
/// one consumer that binds the weight buffer in place. Anything else reading
/// a packed param (a view, a gather, a host-dequant scheme) would have to
/// stage it through the host.
pub fn packed_weights_bind_in_place(graph: &Graph) -> bool {
    use std::collections::HashSet;
    let packed: HashSet<rlx_ir::NodeId> = graph
        .nodes()
        .iter()
        .filter(|n| {
            matches!(n.op, Op::Param { .. })
                && matches!(n.shape.dtype(), rlx_ir::DType::U8 | rlx_ir::DType::I8)
        })
        .map(|n| n.id)
        .collect();
    if packed.is_empty() || graph.outputs.iter().any(|o| packed.contains(o)) {
        return false;
    }
    graph.nodes().iter().all(|node| {
        let reads_packed = node.inputs.iter().any(|i| packed.contains(i));
        if !reads_packed {
            return true;
        }
        let Op::DequantMatMul { scheme } = &node.op else {
            return false;
        };
        if !scheme.is_gguf() {
            return false;
        }
        let id = crate::gguf_host::gguf_scheme_id(*scheme);
        // Q1_0 (24) prefers a tiled GEMM for m > 1 that is not weight-buffer aware.
        gemv_supports_scheme(id)
            && id != 24
            && node.inputs.len() >= 2
            && !packed.contains(&node.inputs[0])
            && node.inputs[2..].iter().all(|i| !packed.contains(i))
    })
}

/// Scheme ids the WGSL `dequant_gguf` kernel actually has a branch for.
///
/// The kernel is an `if (scheme_id == N) { … return; }` chain with **no
/// default arm**: an id that reaches it without a matching branch writes
/// nothing, so the weights read as zeros and the model degrades silently
/// instead of failing. Any scheme outside this set is therefore routed to
/// the host dequant path ([`crate::gguf_host`]) — slower, but correct.
///
/// Every registered scheme is branched today, so this currently never
/// diverts — it is the seam that keeps the *next* unbranched scheme correct.
/// Kept in sync with the shader by
/// `tests/gguf_kernel_scheme_coverage.rs`, which parses the branches out of
/// `DEQUANT_GGUF_WGSL` and requires the two to agree in both directions — so
/// adding a scheme id without a branch (or a branch without listing it here)
/// fails the tests rather than shipping a silent mis-decode.
pub fn kernel_supports_scheme(scheme_id: u32) -> bool {
    matches!(scheme_id, 0..=29)
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
            // Q4_K/Q6_K GGUF matmuls run scratch-free via the windowed GEMV path
            // (`run_dequant_matmul_gguf_gemv`), looped over rows for m>1. Skipping
            // them here keeps the arena from reserving the (multi-GiB) dequant
            // scratch — the LM-head slab (~0.6 GiB for a 0.6B model) is what pushed
            // the arena over wgpu's 2 GiB storage-buffer binding limit and made the
            // prefill dequant bind fail (Validation Error: binding range > limit).
            if gemv_supports_scheme(crate::gguf_host::gguf_scheme_id(*scheme)) {
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
    // Windowed GEMV covers the sharded / split-weight case; this scratch+matmul_bt
    // path still assumes one contiguous arena bind.
    if arena.is_sharded() || crate::buffer::is_weight_off(w_byte_off) {
        crate::gguf_host::run_dequant_matmul_gguf(
            arena,
            device,
            queue,
            m,
            k,
            n,
            scheme_id,
            x_byte_off,
            w_byte_off,
            out_byte_off,
        );
        return;
    }
    // Q4_K/Q6_K/Q1_0: scratch-free windowed GEMV (batched rows → one submit).
    if gemv_supports_scheme(scheme_id) {
        run_dequant_matmul_gguf_gemv_rows(
            arena,
            device,
            queue,
            m,
            k,
            n,
            scheme_id,
            x_byte_off,
            w_byte_off,
            out_byte_off,
        );
        return;
    }

    // No WGSL branch for this scheme → host dequant. Without this the kernel
    // would run, match nothing, and leave the scratch untouched (= zeros).
    if !kernel_supports_scheme(scheme_id) {
        crate::gguf_host::run_dequant_matmul_gguf(
            arena,
            device,
            queue,
            m,
            k,
            n,
            scheme_id,
            x_byte_off,
            w_byte_off,
            out_byte_off,
        );
        return;
    }

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

/// Reused GEMM/GEMV output slabs — creating a fresh buffer per matmul was a
/// large fraction of Bonsai-27B prefill time (hundreds of allocs/frame).
/// Safe across dispatches in one encoder because each dispatch is followed by
/// a `copy_buffer_to_buffer` into the arena before the next write.
/// Pooled output buffer + its capacity. Named (vs a tuple) so it can assert
/// `Send` on the browser WebGPU backend where `wgpu::Buffer` is `!Send`; wasm is
/// single-threaded, so the process-global pool is never sent cross-thread.
struct OutPool(wgpu::Buffer, u64);
#[cfg(target_arch = "wasm32")]
unsafe impl Send for OutPool {}

fn with_pooled_out_buf<R>(
    device: &wgpu::Device,
    need_bytes: u64,
    f: impl FnOnce(&wgpu::Buffer) -> R,
) -> R {
    use std::sync::Mutex;
    static POOL: Mutex<Option<OutPool>> = Mutex::new(None);
    let need = need_bytes.max(16).div_ceil(16) * 16;
    let mut slot = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let recreate = match slot.as_ref() {
        Some(OutPool(_, cap)) => *cap < need,
        None => true,
    };
    if recreate {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu dequant_gemm/gemv out pool"),
            size: need,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        *slot = Some(OutPool(buf, need));
    }
    let OutPool(buf, _) = slot.as_ref().expect("out pool");
    f(buf)
}

/// Encode fused GGUF DequantMatMul into `enc` (no `queue.submit`).
///
/// Used by the main wgpu run loop so Q1_0 GEMMs share one submission with
/// surrounding GPU ops instead of flushing ~1× per weight matrix.
/// Uniforms are unique per dispatch (`queue.write_buffer` is not ordered
/// inside an encoder the way copies are).
#[allow(clippy::too_many_arguments)]
pub fn encode_dequant_matmul_gguf_into(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    enc: &mut wgpu::CommandEncoder,
    m: usize,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) {
    if m == 0 {
        return;
    }
    if scheme_id == 24
        && m > 1
        && !rlx_ir::env::flag("RLX_WGPU_Q1_0_GEMM_DISABLE")
        && encode_dequant_matmul_gguf_gemm(
            arena,
            device,
            queue,
            enc,
            gemm_q1_0(device),
            m,
            k,
            n,
            x_byte_off,
            w_byte_off,
            out_byte_off,
        )
    {
        return;
    }
    if q8_0_gemm_applies(scheme_id, m, k)
        && encode_dequant_matmul_gguf_gemm(
            arena,
            device,
            queue,
            enc,
            gemm_q8_0(device),
            m,
            k,
            n,
            x_byte_off,
            w_byte_off,
            out_byte_off,
        )
    {
        return;
    }
    // Rows go in one dispatch (grid y); chunked only so the bound activation
    // window and the pooled output stay well inside one storage binding.
    let max_bind = device.limits().max_storage_buffer_binding_size as usize;
    let rows_per = (max_bind / 2 / (k.max(n) * 4).max(1)).clamp(1, 65_535);
    let mut row = 0;
    while row < m {
        let rows = (m - row).min(rows_per);
        encode_dequant_matmul_gguf_gemv_one(
            arena,
            device,
            queue,
            enc,
            rows,
            k,
            n,
            scheme_id,
            x_byte_off + row * k * 4,
            w_byte_off,
            out_byte_off + row * n * 4,
        );
        row += rows;
    }
}

/// Fused decode/prefill path: `Y[m,n] = X[m,k] @ W^T` with packed GGUF `W`.
///
/// * Q1_0 + `m > 1` → tiled GEMM (Metal `q1_0_mm_f32` parity) — one dispatch,
///   unless `RLX_WGPU_Q1_0_GEMM_DISABLE=1` or X spans shards.
/// * Otherwise → per-row GEMV encoded into one command buffer (one submit).
#[allow(clippy::too_many_arguments)]
pub fn run_dequant_matmul_gguf_gemv_rows(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    m: usize,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu dequant_gguf rows"),
    });
    encode_dequant_matmul_gguf_into(
        arena,
        device,
        queue,
        &mut enc,
        m,
        k,
        n,
        scheme_id,
        x_byte_off,
        w_byte_off,
        out_byte_off,
    );
    queue.submit(std::iter::once(enc.finish()));
}

/// Q8_0 with several rows takes the tiled GEMM ([`gemm_q8_0`]): it reads each
/// weight block once per 64-row tile instead of once per row, as the per-row
/// GEMV does. The in-arena GEMV plan declines these so they reach it.
/// Off-switch: `RLX_WGPU_Q8_0_GEMM_DISABLE`.
fn q8_0_gemm_applies(scheme_id: u32, m: usize, k: usize) -> bool {
    scheme_id == 20
        && m >= 8
        && k.is_multiple_of(32)
        && !rlx_ir::env::flag("RLX_WGPU_Q8_0_GEMM_DISABLE")
}

/// A fused GGUF GEMM kernel (`DequantGemmQ10Params` layout, bindings
/// `x / params / w / out`) and the grid it wants.
struct TiledGemm {
    kernel: &'static Kernel,
    label: &'static str,
    block_elems: usize,
    block_bytes: usize,
    /// Output columns per workgroup (grid x).
    tile_n: u32,
    /// Output rows per workgroup (grid y).
    tile_m: u32,
}

/// Q1_0: one thread per (column, 8-row tile), 64 columns per workgroup.
fn gemm_q1_0(device: &wgpu::Device) -> TiledGemm {
    TiledGemm {
        kernel: dequant_gemm_q1_0_kernel(device),
        label: "dequant_gemm_q1_0",
        block_elems: 128,
        block_bytes: 18,
        tile_n: 64,
        tile_m: 8,
    }
}

/// Q8_0: 64×64 output tiles staged in workgroup memory
/// (`dequant_gemm_q8_0.wgsl`).
fn gemm_q8_0(device: &wgpu::Device) -> TiledGemm {
    TiledGemm {
        kernel: dequant_gemm_q8_0_kernel(device),
        label: "dequant_gemm_q8_0",
        block_elems: 32,
        block_bytes: 34,
        tile_n: 64,
        tile_m: 64,
    }
}

/// Tiled GGUF GEMM into `enc`. Returns `false` when X/W cannot fit a shard window.
#[allow(clippy::too_many_arguments)]
fn encode_dequant_matmul_gguf_gemm(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    enc: &mut wgpu::CommandEncoder,
    gemm: TiledGemm,
    m: usize,
    k: usize,
    n: usize,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) -> bool {
    debug_assert!(
        k.is_multiple_of(gemm.block_elems),
        "{} needs whole blocks",
        gemm.label
    );
    let (block_elems, block_bytes) = (gemm.block_elems, gemm.block_bytes);
    let w_total_bytes = (k * n) / block_elems * block_bytes;

    let x0 = x_byte_off as u64;
    let x_bytes = (m * k * 4) as u64;
    let x_base = (x0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let (x_buf, x_local) = arena.resolve_act(x_base as usize);
    let x_local = x_local as u64;
    let x_need = x0 + x_bytes - x_base;
    let x_size = (x_need.div_ceil(16) * 16).min(x_buf.size().saturating_sub(x_local));
    if x_size < x_need {
        return false; // X spans shards — GEMV-per-row is safe
    }

    let (w_buf, w_raw) = arena.resolve_w(w_byte_off);
    let w_buf_size = w_buf.size();
    let w0 = w_raw as u64;
    let w_base = (w0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let w_need = w0 + w_total_bytes as u64 - w_base;
    let w_size = (w_need.div_ceil(16) * 16).min(w_buf_size.saturating_sub(w_base));
    if w_size < w_need {
        return false;
    }

    let max_bind = device.limits().max_storage_buffer_binding_size;
    if x_size > max_bind || w_size > max_bind {
        return false;
    }

    let out_elems = m * n;
    let out_bytes = (out_elems * 4) as u64;
    with_pooled_out_buf(device, out_bytes, |out_buf| {
        let p = DequantGemmQ10Params {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            x_f32_off: ((x0 - x_base) / 4) as u32,
            w_byte_off: (w0 - w_base) as u32,
            out_f32_off: 0,
            _p0: 0,
            _p1: 0,
        };
        let dk = gemm.kernel;
        let u = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(gemm.label),
            size: std::mem::size_of::<DequantGemmQ10Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(gemm.label),
            layout: &dk.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: x_buf,
                        offset: x_local,
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
                        buffer: w_buf,
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

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(gemm.label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&dk.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                (n as u32).div_ceil(gemm.tile_n),
                (m as u32).div_ceil(gemm.tile_m),
                1,
            );
        }

        let (dst0, dst0_off) = arena.resolve_act(out_byte_off);
        let row_bytes = (n * 4) as u64;
        let contiguous = (dst0_off as u64) + out_bytes <= dst0.size();
        if contiguous {
            enc.copy_buffer_to_buffer(out_buf, 0, dst0, dst0_off as u64, out_bytes);
        } else {
            for r in 0..m {
                let (dst, dst_off) = arena.resolve_act(out_byte_off + r * n * 4);
                enc.copy_buffer_to_buffer(
                    out_buf,
                    (r as u64) * row_bytes,
                    dst,
                    dst_off as u64,
                    row_bytes,
                );
            }
        }
    });
    true
}

/// Fused decode GEMV: `y[1,n] = x[1,k] @ W^T` with `W` GGUF-packed `[n,k]`,
/// dequantizing each weight block on the fly (no f32 scratch).
///
/// `x` and `weight` are bound as separate **read-only windows** of the arena
/// (each < 4 GiB; the whole-arena `as_entire_binding` overruns the binding
/// limit on multi-GiB models). `y` goes to a small **separate** output buffer —
/// the arena cannot also be bound read-write in the same dispatch (wgpu treats
/// STORAGE_READ_WRITE as exclusive) — which is then copied back into the arena.
///
/// Caller guarantees [`gemv_supports_scheme`]. Prefer
/// [`run_dequant_matmul_gguf_gemv_rows`] when `m > 1` (this helper is m=1 only).
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
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu dequant_gemv_gguf"),
    });
    encode_dequant_matmul_gguf_gemv_one(
        arena,
        device,
        queue,
        &mut enc,
        1,
        k,
        n,
        scheme_id,
        x_byte_off,
        w_byte_off,
        out_byte_off,
    );
    queue.submit(std::iter::once(enc.finish()));
}

fn encode_dequant_matmul_gguf_gemv_one(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    enc: &mut wgpu::CommandEncoder,
    m: usize,
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

    let x0 = x_byte_off as u64;
    let x_base = (x0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let (x_buf, x_local) = arena.resolve_act(x_base as usize);
    let x_local = x_local as u64;
    let x_size = ((x0 + (m * k * 4) as u64 - x_base).div_ceil(16) * 16)
        .min(x_buf.size().saturating_sub(x_local));

    let (w_buf, w_raw) = arena.resolve_w(w_byte_off);
    let w_buf_size = w_buf.size();
    let w0_all = w_raw as u64;

    // Packed-weight bytes per output row. Output rows are independent (each is a
    // separate dot product over k), so when the full weight window would exceed
    // `max_storage_buffer_binding_size` (128 MiB minimum on wgpu — hit by llvmpipe /
    // min-spec adapters, or large models on any GPU) we split the GEMV along N and
    // bind only each chunk's rows. `w_total_bytes / n` is the exact per-row stride
    // (matches the shader's `j * nbk * blk` indexing).
    let row_bytes = (w_total_bytes / n.max(1)) as u64;

    // `RLX_WGPU_MAX_BIND_MB` artificially caps the binding size to exercise the
    // split path on GPUs whose real limit is large (validation/testing).
    let max_bind = rlx_ir::env::var("RLX_WGPU_MAX_BIND_MB")
        .and_then(|s| s.parse::<u64>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(device.limits().max_storage_buffer_binding_size);
    assert!(
        x_size <= max_bind,
        "rlx-wgpu gguf gemv: x window {x_size} > max_bind {max_bind}"
    );
    // Rows per chunk so the bound weight window — including the align-down base
    // slack (< STORAGE_ALIGN) and the 16-byte round-up — stays within max_bind.
    let budget = max_bind.saturating_sub(STORAGE_ALIGN + 16).max(row_bytes);
    let rows_per_chunk = ((budget / row_bytes.max(1)) as usize).max(1);

    let out_bytes = ((m * n * 4).max(4) as u64).div_ceil(16) * 16;
    let entry = gemv_entry(scheme_id);
    with_pooled_out_buf(device, out_bytes, |out_buf| {
        let dk = match entry {
            GemvEntry::Q4k16 => crate::kernels::dequant_gemv_gguf_q4k16_kernel(device),
            GemvEntry::Q5k16 => crate::kernels::dequant_gemv_gguf_q5k16_kernel(device),
            GemvEntry::Q6k16 => crate::kernels::dequant_gemv_gguf_q6k16_kernel(device),
            GemvEntry::B32x16 => crate::kernels::dequant_gemv_gguf_b32x16_kernel(device),
            GemvEntry::Rows8 => crate::kernels::dequant_gemv_gguf_rows8_kernel(device),
            GemvEntry::Serial => dequant_gemv_gguf_kernel(device),
        };
        let mut n0 = 0usize;
        while n0 < n {
            let n_chunk = (n - n0).min(rows_per_chunk);
            let w0 = w0_all + n0 as u64 * row_bytes;
            let w_base = (w0 / STORAGE_ALIGN) * STORAGE_ALIGN;
            let chunk_bytes = n_chunk as u64 * row_bytes;
            let w_size = ((w0 + chunk_bytes - w_base).div_ceil(16) * 16).min(w_buf_size - w_base);
            debug_assert!(
                w_size <= max_bind,
                "rlx-wgpu gguf gemv chunk window {w_size} > max_bind {max_bind}"
            );

            let p = DequantGemvGgufParams {
                k: k as u32,
                n: n_chunk as u32,
                scheme_id,
                x_f32_off: ((x0 - x_base) / 4) as u32,
                w_byte_off: (w0 - w_base) as u32,
                out_f32_off: n0 as u32,
                out_stride: n as u32,
                m_rows: m as u32,
            };

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
                            buffer: x_buf,
                            offset: x_local,
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
                            buffer: w_buf,
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

            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rlx-wgpu dequant_gemv_gguf pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&dk.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                let [gx, gy, gz] = gemv_dispatch(entry, m, n_chunk);
                pass.dispatch_workgroups(gx, gy, gz);
            }
            n0 += n_chunk;
        }
        let (dst, dst_off) = arena.resolve_act(out_byte_off);
        enc.copy_buffer_to_buffer(out_buf, 0, dst, dst_off as u64, (m * n * 4) as u64);
    });
}

/// The GEMV entry point for a scheme. Lanes share an output column where the
/// shader has that path, one thread per column otherwise; the serial kernel
/// has no branch for the 32-weight formats, so they always take theirs.
fn gemv_entry(scheme_id: u32) -> GemvEntry {
    let serial = rlx_ir::env::flag("RLX_WGPU_GGUF_GEMV_SERIAL");
    match scheme_id {
        0 if !serial => GemvEntry::Q4k16,
        1 if !serial => GemvEntry::Q5k16,
        2 if !serial => GemvEntry::Q6k16,
        id if b32_scheme(id) && (id != 20 || !serial) => GemvEntry::B32x16,
        id if id == crate::gguf_host::BF16_SCHEME_ID && !serial => GemvEntry::Rows8,
        _ => GemvEntry::Serial,
    }
}

/// 16 columns × up to 8 rows per workgroup (the K-quants and 32-weight
/// formats), against 8 columns × one row (bf16) or a thread per column.
fn gemv_entry_is_wide(entry: GemvEntry) -> bool {
    !matches!(entry, GemvEntry::Serial | GemvEntry::Rows8)
}

/// Workgroups for `m` rows × `n` columns: column groups over x (and z past one
/// dimension's 65535 limit), row groups over y.
fn gemv_dispatch(entry: GemvEntry, m: usize, n: usize) -> [u32; 3] {
    if entry == GemvEntry::Serial {
        return [(n as u32).div_ceil(64), m as u32, 1];
    }
    let wide = gemv_entry_is_wide(entry);
    let groups = (n as u32).div_ceil(if wide { 16 } else { 8 });
    let gx = groups.min(32_768);
    let gy = if wide {
        (m as u32).div_ceil(8)
    } else {
        m as u32
    };
    [gx, gy, groups.div_ceil(gx)]
}

/// A GGUF GEMV step made ready to encode: bind groups, params and workgroup
/// counts built once and reused by every later run of the step.
pub struct GemvPlan {
    kernel: &'static crate::kernels::Kernel,
    bind_group: wgpu::BindGroup,
    dispatch: [u32; 3],
    _params: wgpu::Buffer,
}

impl GemvPlan {
    /// One dispatch inside an open compute pass.
    pub fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.kernel.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let [x, y, z] = self.dispatch;
        pass.dispatch_workgroups(x, y, z);
    }

    pub fn encode(&self, enc: &mut wgpu::CommandEncoder) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu dequant_gemv_gguf in-arena pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.kernel.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let [x, y, z] = self.dispatch;
        pass.dispatch_workgroups(x, y, z);
    }
}

/// [`encode_dequant_matmul_gguf_into`] as a reusable [`GemvPlan`] that reads
/// the activations and writes the output in place in the arena — no output
/// buffer, no copy, no per-run bind groups. Only for a weight held in the
/// weight buffer (the arena is then free to bind read-write) and an
/// activation/output pair that fits one storage binding without
/// overlapping; `None` sends the caller to the general path.
#[allow(clippy::too_many_arguments)]
pub fn plan_dequant_gemv_in_arena(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    m: usize,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) -> Option<GemvPlan> {
    use crate::buffer::is_weight_off;
    if m == 0
        || scheme_id == 24
        || q8_0_gemm_applies(scheme_id, m, k)
        || !is_weight_off(w_byte_off)
        || is_weight_off(x_byte_off)
        || is_weight_off(out_byte_off)
        || arena.is_sharded()
        || rlx_ir::env::flag("RLX_WGPU_GGUF_GEMV_COPY_OUT")
    {
        return None;
    }
    let max_bind = device.limits().max_storage_buffer_binding_size;
    let (x0, x1) = (x_byte_off as u64, (x_byte_off + m * k * 4) as u64);
    let (o0, o1) = (out_byte_off as u64, (out_byte_off + m * n * 4) as u64);
    if x0 < o1 && o0 < x1 {
        return None;
    }
    let base = (x0.min(o0) / STORAGE_ALIGN) * STORAGE_ALIGN;
    let (a_buf, a_local) = arena.resolve_act(base as usize);
    let a_size = (x1.max(o1) - base).div_ceil(16) * 16;
    if a_size > max_bind || a_local as u64 + a_size > a_buf.size() {
        return None;
    }

    let scheme = scheme_from_id(scheme_id);
    let w_total =
        (k * n) / (scheme.gguf_block_size() as usize).max(1) * scheme.gguf_block_bytes() as usize;
    let (w_buf, w_raw) = arena.resolve_w(w_byte_off);
    let w0 = w_raw as u64;
    let w_base = (w0 / STORAGE_ALIGN) * STORAGE_ALIGN;
    let w_size = ((w0 + w_total as u64 - w_base).div_ceil(16) * 16).min(w_buf.size() - w_base);
    if w_size > max_bind || w0 + w_total as u64 > w_buf.size() {
        return None;
    }

    let entry = gemv_entry(scheme_id);
    let kernel = crate::kernels::dequant_gemv_gguf_in_arena_kernel(device, entry);
    let p = DequantGemvGgufParams {
        k: k as u32,
        n: n as u32,
        scheme_id,
        x_f32_off: ((x0 - base) / 4) as u32,
        w_byte_off: (w0 - w_base) as u32,
        out_f32_off: ((o0 - base) / 4) as u32,
        out_stride: n as u32,
        m_rows: m as u32,
    };
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu dequant_gemv_gguf in-arena params"),
        size: std::mem::size_of::<DequantGemvGgufParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params, 0, bytemuck::bytes_of(&p));
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu dequant_gemv_gguf in-arena bg"),
        layout: &kernel.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: a_buf,
                    offset: a_local as u64,
                    size: wgpu::BufferSize::new(a_size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: w_buf,
                    offset: w_base,
                    size: wgpu::BufferSize::new(w_size),
                }),
            },
        ],
    });
    Some(GemvPlan {
        kernel,
        bind_group,
        dispatch: gemv_dispatch(entry, m, n),
        _params: params,
    })
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
    if arena.is_sharded() {
        crate::gguf_host::run_dequant_grouped_matmul_gguf(
            arena,
            device,
            queue,
            m,
            k,
            n,
            num_experts,
            scheme_id,
            x_byte_off,
            w_byte_off,
            idx_byte_off,
            out_byte_off,
        );
        return;
    }
    let scheme = scheme_from_id(scheme_id);
    let slab_bytes = slab_bytes_for(scheme, k, n);
    let num_blocks = (k * n) / scheme.gguf_block_size() as usize;

    let x_bytes = arena.read_bytes_range(device, queue, x_byte_off, m * k * 4);
    let x_host: Vec<f32> = x_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let idx_bytes = arena.read_bytes_range(device, queue, idx_byte_off, m * 4);
    let idx_host: Vec<f32> = idx_bytes
        .as_chunks::<4>()
        .0
        .iter()
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
        .as_chunks::<4>()
        .0
        .iter()
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
