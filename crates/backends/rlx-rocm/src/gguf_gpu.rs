// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU GGUF K-quant dequant + hipBLAS matmul for `Op::DequantMatMul` and
//! grouped MoE `Op::DequantGroupedMatMul`.

use rlx_ir::{Graph, Op};
use std::sync::{Arc, Mutex};

use crate::device::RocmContext;
use crate::gguf_host::scheme_from_id;
use crate::hip::HipBuffer;
use crate::hipblas::{HipblasContext, HipblasOperation};
use crate::kernels::dequant_gguf_kernel;

fn launch_dequant_gguf(
    ctx: &Arc<RocmContext>,
    stream: crate::hip::HipStream,
    buffer: &HipBuffer<f32>,
    w_byte_off: u64,
    dst_f32_off: u64,
    scheme_id: u32,
    nb_u32: u32,
) {
    use core::ffi::c_void;
    let kernel = dequant_gguf_kernel(ctx);
    let block = 256u32.min(nb_u32).max(1);
    let grid = nb_u32.div_ceil(block);
    // The kernel takes 6 params with **64-bit** arena offsets and an IQ grid
    // LUT pointer (see `dequant_gguf.cu`). Passing 5 u32-offset params — as this
    // did — makes hipModuleLaunchKernel read a 6th arg past the array and
    // misread the offsets, so a >4 GB arena overflows u32 → SIGSEGV / garbage.
    // IQ-family schemes index `iq_lut`; non-IQ schemes ignore it but the arg
    // must still be bound. Mirrors `rlx_cuda::gguf_gpu`.
    let lut = crate::iq_grid::rocm_iq_grid_buffer(ctx);
    let mut dev = buffer.ptr;
    let mut lut_ptr = lut.ptr;
    let mut params: [*mut c_void; 6] = [
        &mut dev as *mut _ as *mut c_void,
        &w_byte_off as *const _ as *mut c_void,
        &dst_f32_off as *const _ as *mut c_void,
        &scheme_id as *const _ as *mut c_void,
        &nb_u32 as *const _ as *mut c_void,
        &mut lut_ptr as *mut _ as *mut c_void,
    ];
    unsafe {
        kernel
            .launch(stream, (grid, 1, 1), (block, 1, 1), 0, params.as_mut_ptr())
            .expect("rlx-rocm: dequant_gguf launch failed");
    }
}

fn slab_bytes_for(scheme: rlx_ir::quant::QuantScheme, k: usize, n: usize) -> usize {
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    (k * n) / block_elems * block_bytes
}

/// Max f32 scratch for dequantized weights `[n, k]` across all GGUF ops.
pub fn dequant_gguf_scratch_bytes(graph: &Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        // GGUF and MxFp4x2 both decode the weight into an f32 [k,n] scratch
        // before the sgemm, so they size the scratch identically.
        if let Op::DequantMatMul { scheme } = &node.op
            && (scheme.is_gguf() || scheme.mxfp4x2_config().is_some())
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
            max = max.max(k * n * 4 + m * k * 4 + m * n * 4);
            let _ = scheme;
        }
    }
    max
}

/// Launch `dequant_gguf` into arena scratch, then `C = X @ W^T` via hipBLAS.
pub fn run_dequant_matmul_gguf_gpu(
    ctx: &Arc<RocmContext>,
    stream: crate::hip::HipStream,
    buffer: &HipBuffer<f32>,
    blas: &Arc<Mutex<HipblasContext>>,
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

    let dst_f32_off = (scratch_byte_off / 4) as u64;
    let w_off_u64 = w_byte_off as u64;
    let nb_u32 = num_blocks as u32;
    launch_dequant_gguf(
        ctx,
        stream,
        buffer,
        w_off_u64,
        dst_f32_off,
        scheme_id,
        nb_u32,
    );

    let x_dev = buffer.ptr + (x_byte_off as u64);
    let w_dev = buffer.ptr + (scratch_byte_off as u64);
    let c_dev = buffer.ptr + (out_byte_off as u64);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let blas = blas.lock().unwrap();
    let result = unsafe {
        (blas.runtime.sgemm)(
            blas.handle,
            HipblasOperation::N,
            HipblasOperation::N,
            n as i32,
            m as i32,
            k as i32,
            &alpha as *const f32,
            w_dev as *const f32,
            n as i32,
            x_dev as *const f32,
            k as i32,
            &beta as *const f32,
            c_dev as *mut f32,
            n as i32,
        )
    };
    result
        .ok()
        .expect("rlx-rocm: gguf matmul hipblasSgemm failed");
}

/// MxFp4x2 two-level residual E2M1 `DequantMatMul` on GPU: decode the packed
/// `[plane0|plane1]` weight + `[s0|s1]` scales into f32 `[k,n]` scratch, then
/// `C = X @ W` via hipBLAS. Layout matches `rlx_cpu::dequant_matmul_mxfp4x2` so
/// the decoded scratch feeds the same sgemm as the GGUF path.
#[allow(clippy::too_many_arguments)]
pub fn run_dequant_matmul_mxfp4x2_gpu(
    ctx: &Arc<RocmContext>,
    stream: crate::hip::HipStream,
    buffer: &HipBuffer<f32>,
    blas: &Arc<Mutex<HipblasContext>>,
    m: usize,
    k: usize,
    n: usize,
    group: usize,
    x_byte_off: usize,
    w_byte_off: usize,
    scale_byte_off: usize,
    scratch_byte_off: usize,
    out_byte_off: usize,
) {
    use core::ffi::c_void;
    let kernel = crate::kernels::mxfp4x2_dequant_kernel(ctx);
    let total = (k * n) as u32;
    let block = 256u32.min(total).max(1);
    let grid = total.div_ceil(block);
    let w_off_u64 = w_byte_off as u64;
    let s_f32_off = (scale_byte_off / 4) as u64;
    let dst_f32_off = (scratch_byte_off / 4) as u64;
    let (k_u, n_u, g_u) = (k as u32, n as u32, group as u32);
    let mut dev = buffer.ptr;
    let mut params: [*mut c_void; 7] = [
        &mut dev as *mut _ as *mut c_void,
        &w_off_u64 as *const _ as *mut c_void,
        &s_f32_off as *const _ as *mut c_void,
        &dst_f32_off as *const _ as *mut c_void,
        &k_u as *const _ as *mut c_void,
        &n_u as *const _ as *mut c_void,
        &g_u as *const _ as *mut c_void,
    ];
    unsafe {
        kernel
            .launch(stream, (grid, 1, 1), (block, 1, 1), 0, params.as_mut_ptr())
            .expect("rlx-rocm: mxfp4x2_dequant launch failed");
    }

    let x_dev = buffer.ptr + (x_byte_off as u64);
    let w_dev = buffer.ptr + (scratch_byte_off as u64);
    let c_dev = buffer.ptr + (out_byte_off as u64);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let blas = blas.lock().unwrap();
    let result = unsafe {
        (blas.runtime.sgemm)(
            blas.handle,
            HipblasOperation::N,
            HipblasOperation::N,
            n as i32,
            m as i32,
            k as i32,
            &alpha as *const f32,
            w_dev as *const f32,
            n as i32,
            x_dev as *const f32,
            k as i32,
            &beta as *const f32,
            c_dev as *mut f32,
            n as i32,
        )
    };
    result
        .ok()
        .expect("rlx-rocm: mxfp4x2 matmul hipblasSgemm failed");
}

/// GPU dequant + grouped matmul for MoE packed expert stacks.
pub unsafe fn run_dequant_grouped_matmul_gguf_gpu(
    ctx: &Arc<RocmContext>,
    stream: crate::hip::HipStream,
    buffer: &HipBuffer<f32>,
    blas: &Arc<Mutex<HipblasContext>>,
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

    let rt = &ctx.runtime;
    unsafe {
        let _ = (rt.hip_stream_sync)(stream);
    }

    let x_f32_off = x_byte_off / 4;
    let mut x_host = vec![0f32; m * k];
    unsafe {
        let _ = (rt.hip_memcpy_dtoh)(
            x_host.as_mut_ptr() as *mut _,
            buffer.ptr + (x_f32_off as u64) * 4,
            m * k * 4,
        );
    }

    let idx_f32_off = idx_byte_off / 4;
    let mut idx_host = vec![0f32; m];
    unsafe {
        let _ = (rt.hip_memcpy_dtoh)(
            idx_host.as_mut_ptr() as *mut _,
            buffer.ptr + (idx_f32_off as u64) * 4,
            m * 4,
        );
    }

    let (packed_in, original_pos, offsets) =
        rlx_cpu::gguf_matmul::grouped_moe_sort_plan(&x_host, &idx_host, m, k, num_experts);

    let dequant_off = scratch_byte_off;
    let pack_in_off = scratch_byte_off + k * n * 4;
    let pack_out_off = scratch_byte_off + (k * n + m * k) * 4;

    unsafe {
        let _ = (rt.hip_memcpy_htod)(
            buffer.ptr + (pack_in_off as u64),
            packed_in.as_ptr() as *const _,
            m * k * 4,
        );
    }

    let dst_f32_off = (dequant_off / 4) as u64;
    let nb_u32 = num_blocks as u32;

    let blas = blas.lock().unwrap();
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    for e in 0..num_experts {
        let count = offsets[e + 1] - offsets[e];
        if count == 0 {
            continue;
        }
        let w_off = w_byte_off + e * slab_bytes;
        let w_off_u64 = w_off as u64;
        launch_dequant_gguf(
            ctx,
            stream,
            buffer,
            w_off_u64,
            dst_f32_off,
            scheme_id,
            nb_u32,
        );

        let in_start = offsets[e];
        let a_dev = buffer.ptr + ((pack_in_off / 4 + in_start * k) as u64) * 4;
        let b_dev = buffer.ptr + (dequant_off as u64);
        let c_dev = buffer.ptr + ((pack_out_off / 4 + in_start * n) as u64) * 4;
        let result = unsafe {
            (blas.runtime.sgemm)(
                blas.handle,
                HipblasOperation::N,
                HipblasOperation::N,
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
        };
        result
            .ok()
            .expect("rlx-rocm: grouped gguf hipblasSgemm failed");
    }

    let mut packed_out = vec![0f32; m * n];
    unsafe {
        let _ = (rt.hip_memcpy_dtoh)(
            packed_out.as_mut_ptr() as *mut _,
            buffer.ptr + (pack_out_off as u64),
            m * n * 4,
        );
    }

    let mut out_host = vec![0f32; m * n];
    rlx_cpu::gguf_matmul::grouped_moe_unpermute_out(
        &packed_out,
        &original_pos,
        &mut out_host,
        m,
        n,
    );

    let out_f32_off = out_byte_off / 4;
    unsafe {
        let _ = (rt.hip_memcpy_htod)(
            buffer.ptr + (out_f32_off as u64) * 4,
            out_host.as_ptr() as *const _,
            m * n * 4,
        );
    }
}
