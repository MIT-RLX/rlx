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
//! GPU GGUF K-quant dequant + cuBLAS matmul for `Op::DequantMatMul` and
//! grouped MoE `Op::DequantGroupedMatMul`.

use cudarc::cublas::{CudaBlas, result as cublas_result, sys as cublas_sys};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use rlx_ir::{Graph, Op};
use std::sync::{Arc, Mutex};

use crate::gguf_host::scheme_from_id;
use crate::kernels::dequant_gguf_kernel;

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

/// Launch `dequant_gguf` into arena scratch, then `C = X @ W^T` via cuBLAS.
pub fn run_dequant_matmul_gguf_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    blas: &Arc<Mutex<CudaBlas>>,
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

    let kernel = dequant_gguf_kernel(ctx);
    let block = 256u32.min(num_blocks as u32).max(1);
    let grid = num_blocks.div_ceil(block as usize) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let dst_f32_off = (scratch_byte_off / 4) as u32;
    let w_off_u32 = w_byte_off as u32;
    let nb_u32 = num_blocks as u32;
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&w_off_u32)
        .arg(&dst_f32_off)
        .arg(&scheme_id)
        .arg(&nb_u32);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: dequant_gguf launch failed");
    }

    let x_off_f32 = x_byte_off / 4;
    let w_off_f32 = scratch_byte_off / 4;
    let out_off_f32 = out_byte_off / 4;
    let (arena_ptr_u64, _record) = buffer.device_ptr_mut(stream);
    let a_dev = arena_ptr_u64 + (x_off_f32 as u64) * 4;
    let b_dev = arena_ptr_u64 + (w_off_f32 as u64) * 4;
    let c_dev = arena_ptr_u64 + (out_off_f32 as u64) * 4;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let blas = blas.lock().unwrap();
    unsafe {
        cublas_result::sgemm(
            *blas.handle(),
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            &alpha as *const f32,
            b_dev as *const f32,
            n as i32,
            a_dev as *const f32,
            k as i32,
            &beta as *const f32,
            c_dev as *mut f32,
            n as i32,
        )
        .expect("rlx-cuda: gguf matmul cublasSgemm failed");
    }
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
    blas: &Arc<Mutex<CudaBlas>>,
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

    let blas = blas.lock().unwrap();
    let arena_ptr_u64 = {
        let (ptr, _record) = buffer.device_ptr_mut(stream);
        ptr
    };
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

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
            .arg(&nb_u32);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: grouped dequant_gguf launch failed");
        }

        let in_start = offsets[e];
        let a_dev = arena_ptr_u64 + ((pack_in_off / 4 + in_start * k) as u64) * 4;
        let b_dev = arena_ptr_u64 + (dequant_off as u64);
        let c_dev = arena_ptr_u64 + ((pack_out_off / 4 + in_start * n) as u64) * 4;
        unsafe {
            cublas_result::sgemm(
                *blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                count as i32,
                k as i32,
                &alpha as *const f32,
                b_dev as *const f32,
                n as i32,
                a_dev as *const f32,
                k as i32,
                &beta as *const f32,
                c_dev as *mut f32,
                n as i32,
            )
            .expect("rlx-cuda: grouped gguf cublasSgemm failed");
        }
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
