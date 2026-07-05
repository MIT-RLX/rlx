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
//! GPU GGUF dequant + cuBLAS matmul for `Op::DequantMatMul` and grouped MoE
//! `Op::DequantGroupedMatMul`.
//!
//! Flow: launch `dequant_gguf` into arena scratch, then `C = X @ W^T` via
//! the shared `matmul_bt` kernel (GGUF row-major `[n,k]` weights).
//!
//! See [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md).

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use rlx_ir::{Graph, Op};
use std::sync::Arc;

use crate::gguf_host::scheme_from_id;
use crate::kernels::{dequant_gguf_kernel, dequant_matmul_gguf_kernel, matmul_bt_kernel};

fn slab_bytes_for(scheme: rlx_ir::quant::QuantScheme, k: usize, n: usize) -> usize {
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    (k * n) / block_elems * block_bytes
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
            let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
            let k = x_total / m.max(1);
            // Decode GEMV (m==1, fused, scratch-free — see backend.rs) needs no
            // f32 dequant slab. Skipping it avoids reserving the multi-GiB
            // tied-embed dequant scratch (Gemma 4 12B lm_head: 3840*262144*4 =
            // 4 GiB) that would push the arena past 16 GiB VRAM and OOM.
            if m == 1
                && gguf_fused_gemv_m1_supported(crate::gguf_host::gguf_scheme_id(*scheme), m, k)
            {
                continue;
            }
            max = max.max(k * n * std::mem::size_of::<f32>());
        }
        if let Op::DequantGroupedMatMul { scheme } = &node.op {
            let in_shape = &graph.node(node.inputs[0]).shape;
            let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
            let k = in_shape.dim(in_shape.rank() - 1).unwrap_static();
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            // dequant slab + packed input + packed output (bytes).
            max = max.max(k * n * 4 + m * k * 4 + m * n * 4);
            let _ = scheme;
        }
    }
    max
}

/// Fused on-device GEMV for decode (`m == 1`) on Q4_K / Q6_K — matches
/// rlx-vulkan `dequant_matmul` and rlx-cpu `gguf_matmul_bt`.
pub fn gguf_fused_gemv_m1_supported(scheme_id: u32, m: usize, k: usize) -> bool {
    m == 1 && k.is_multiple_of(256) && matches!(scheme_id, 0 | 2)
}

/// Launch [`dequant_matmul_gguf`] — one thread per output column.
pub fn run_dequant_matmul_gguf_gemv_m1(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    n: usize,
    k: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) {
    debug_assert!(gguf_fused_gemv_m1_supported(scheme_id, 1, k));
    let kernel = dequant_matmul_gguf_kernel(ctx);
    let (grid, block) = crate::kernels::dispatch_grid_1d(n as u32, 64);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let x_off = (x_byte_off / 4) as u32;
    let out_off = (out_byte_off / 4) as u32;
    let w_off = w_byte_off as u32;
    let n_u = n as u32;
    let k_u = k as u32;
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&n_u)
        .arg(&k_u)
        .arg(&x_off)
        .arg(&w_off)
        .arg(&out_off)
        .arg(&scheme_id);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: dequant_matmul_gguf launch failed");
    }
}

/// Launch `matmul_bt`: `C[m,n] = A[m,k] @ W[n,k]^T` with row-major arena offsets.
fn run_matmul_bt(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) {
    let kernel = matmul_bt_kernel(ctx);
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(32) as u32, m.div_ceil(32) as u32, 1),
        block_dim: (8, 8, 1),
        shared_mem_bytes: 0,
    };
    let m_u = m as u32;
    let k_u = k as u32;
    let n_u = n as u32;
    let a_off = (x_byte_off / 4) as u32;
    let b_off = (w_byte_off / 4) as u32;
    let c_off = (out_byte_off / 4) as u32;
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&m_u)
        .arg(&k_u)
        .arg(&n_u)
        .arg(&a_off)
        .arg(&b_off)
        .arg(&c_off);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: matmul_bt launch failed");
    }
}

/// Launch `dequant_gguf` into arena scratch, then `C = X @ W^T` via `matmul_bt`.
pub fn run_dequant_matmul_gguf_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
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
    let total = k * n;
    let num_blocks = total / block_elems.max(1);
    if num_blocks == 0 {
        panic!(
            "rlx-cuda: dequant_gguf num_blocks=0 (m={m}, k={k}, n={n}, scheme_id={scheme_id}, block_elems={block_elems})"
        );
    }

    let kernel = dequant_gguf_kernel(ctx);
    let threads = 256u32.min(num_blocks as u32).max(1);
    let grid = num_blocks.div_ceil(threads as usize) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let dst_f32_off = (scratch_byte_off / 4) as u32;
    let w_off_u32 = w_byte_off as u32;
    let nb_u32 = num_blocks as u32;
    // Materialise the IQ grid LUT on this context once (cached). Bound
    // as a kernel arg unconditionally — non-IQ schemes ignore the pointer.
    let lut = crate::iq_grid::cuda_iq_grid_buffer(ctx, stream);
    use cudarc::driver::DevicePtr;
    let (lut_ptr, _lut_rec) = lut.device_ptr(stream);
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&w_off_u32)
        .arg(&dst_f32_off)
        .arg(&scheme_id)
        .arg(&nb_u32)
        .arg(&lut_ptr);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: dequant_gguf launch failed");
    }

    run_matmul_bt(
        ctx,
        stream,
        buffer,
        m,
        k,
        n,
        x_byte_off,
        scratch_byte_off,
        out_byte_off,
    );
}

/// GPU dequant + grouped matmul for MoE packed expert stacks.
///
/// Scratch layout at `scratch_byte_off` (f32 bytes):
///   `[0 .. k*n)`: dequantized expert slab
///   `[k*n .. k*n+m*k)`: sorted token inputs
///   `[k*n+m*k .. k*n+m*k+m*n)`: sorted outputs before unpermute
pub fn run_dequant_grouped_matmul_gguf_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
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

    stream
        .synchronize()
        .expect("rlx-cuda: grouped gguf pre-sync failed");

    let x_f32_off = x_byte_off / 4;
    let mut x_host = vec![0f32; m * k];
    stream
        .memcpy_dtoh(&buffer.slice(x_f32_off..x_f32_off + m * k), &mut x_host)
        .expect("rlx-cuda: grouped gguf x dtoh failed");

    let idx_f32_off = idx_byte_off / 4;
    let mut idx_host = vec![0f32; m];
    stream
        .memcpy_dtoh(&buffer.slice(idx_f32_off..idx_f32_off + m), &mut idx_host)
        .expect("rlx-cuda: grouped gguf idx dtoh failed");

    let (packed_in, original_pos, offsets) =
        rlx_cpu::gguf_matmul::grouped_moe_sort_plan(&x_host, &idx_host, m, k, num_experts);

    let dequant_off = scratch_byte_off;
    let pack_in_off = scratch_byte_off + k * n * 4;
    let pack_out_off = scratch_byte_off + (k * n + m * k) * 4;

    stream
        .memcpy_htod(
            &packed_in,
            &mut buffer.slice_mut(pack_in_off / 4..pack_in_off / 4 + m * k),
        )
        .expect("rlx-cuda: grouped gguf pack_in htod failed");

    let kernel = dequant_gguf_kernel(ctx);
    let block = 256u32.min(num_blocks as u32).max(1);
    let grid = num_blocks.div_ceil(block as usize) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let dst_f32_off = (dequant_off / 4) as u32;
    let nb_u32 = num_blocks as u32;

    let lut = crate::iq_grid::cuda_iq_grid_buffer(ctx, stream);
    use cudarc::driver::DevicePtr;
    let (lut_ptr, _lut_rec) = lut.device_ptr(stream);
    for e in 0..num_experts {
        let count = offsets[e + 1] - offsets[e];
        if count == 0 {
            continue;
        }
        let w_off = w_byte_off + e * slab_bytes;
        let w_off_u32 = w_off as u32;
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&w_off_u32)
            .arg(&dst_f32_off)
            .arg(&scheme_id)
            .arg(&nb_u32)
            .arg(&lut_ptr);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: grouped dequant_gguf launch failed");
        }

        let in_start = offsets[e];
        run_matmul_bt(
            ctx,
            stream,
            buffer,
            count,
            k,
            n,
            pack_in_off + in_start * k * 4,
            dequant_off,
            pack_out_off + in_start * n * 4,
        );
    }

    let mut packed_out = vec![0f32; m * n];
    stream
        .memcpy_dtoh(
            &buffer.slice(pack_out_off / 4..pack_out_off / 4 + m * n),
            &mut packed_out,
        )
        .expect("rlx-cuda: grouped gguf pack_out dtoh failed");

    let mut out_host = vec![0f32; m * n];
    rlx_cpu::gguf_matmul::grouped_moe_unpermute_out(
        &packed_out,
        &original_pos,
        &mut out_host,
        m,
        n,
    );

    let out_f32_off = out_byte_off / 4;
    stream
        .memcpy_htod(
            &out_host,
            &mut buffer.slice_mut(out_f32_off..out_f32_off + m * n),
        )
        .expect("rlx-cuda: grouped gguf out htod failed");
}
