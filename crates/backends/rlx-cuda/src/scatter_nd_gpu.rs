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

//! On-device ONNX ScatterND (reduction=none) for CUDA f32-uniform arenas.
//!
//! Avoids the host D2H of large update tensors (Kitten wave: ~40 MiB).

use crate::kernels::{dispatch_grid_1d, scatter_nd_kernel};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use rlx_cpu::thunk::{IndexingThunk, Thunk};
use rlx_ir::ScatterNdReduction;
use std::sync::Arc;

fn row_major_strides(shape: &[u32]) -> Vec<u32> {
    let mut strides = vec![1u32; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].saturating_mul(shape[i + 1].max(1));
    }
    strides
}

/// Try to run ScatterND on-device. Returns `false` to fall back to host indexing.
pub fn try_run(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    arena: &mut CudaSlice<f32>,
    thunk: &IndexingThunk,
) -> bool {
    if rlx_ir::env::flag("RLX_CUDA_SCATTER_ND_HOST") {
        return false;
    }
    let Thunk::ScatterNd {
        data,
        indices,
        updates,
        dst,
        data_shape,
        indices_shape,
        data_len,
        updates_len,
        indices_len: _,
        indices_i64,
        reduction,
    } = thunk.inner()
    else {
        if rlx_ir::env::flag("RLX_CUDA_SCATTER_ND_TRACE") {
            eprintln!("[scatter_nd_gpu] skip: not ScatterNd");
        }
        return false;
    };
    // f32-uniform arenas store indices as f32 slots (force_indices_f32).
    // Raw i64 (8 B/elem) stays on host for now.
    if *indices_i64 != 0 {
        if rlx_ir::env::flag("RLX_CUDA_SCATTER_ND_TRACE") {
            eprintln!("[scatter_nd_gpu] skip: i64 indices");
        }
        return false;
    }
    if !matches!(reduction, ScatterNdReduction::None) {
        if rlx_ir::env::flag("RLX_CUDA_SCATTER_ND_TRACE") {
            eprintln!("[scatter_nd_gpu] skip: reduction={reduction:?}");
        }
        return false;
    }
    if data_shape.is_empty() || indices_shape.is_empty() {
        return false;
    }
    let k = *indices_shape.last().unwrap_or(&0) as usize;
    if k == 0 || k > 4 {
        if rlx_ir::env::flag("RLX_CUDA_SCATTER_ND_TRACE") {
            eprintln!("[scatter_nd_gpu] skip: k={k}");
        }
        return false;
    }
    let num_updates: usize = indices_shape[..indices_shape.len() - 1]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let slice: usize = data_shape
        .get(k..)
        .map(|s| s.iter().map(|&d| d as usize).product::<usize>())
        .unwrap_or(1)
        .max(1);
    if num_updates.saturating_mul(slice) > *updates_len as usize {
        if rlx_ir::env::flag("RLX_CUDA_SCATTER_ND_TRACE") {
            eprintln!(
                "[scatter_nd_gpu] skip: updates too small num={num_updates} slice={slice} updates_len={} \
                 data_shape={data_shape:?} indices_shape={indices_shape:?}",
                *updates_len
            );
        }
        return false;
    }
    if *data_len as usize != data_shape.iter().map(|&d| d as usize).product::<usize>() {
        return false;
    }

    let data_off = (*data / 4) as u32;
    let idx_off = (*indices / 4) as u32;
    let upd_off = (*updates / 4) as u32;
    let dst_off = (*dst / 4) as u32;
    let data_n = *data_len as usize;

    // Copy data → dst when not aliased (matches CPU scatter_nd_into_f32).
    if *data != *dst && data_n > 0 {
        let (ptr, rec) = arena.device_ptr_mut(stream);
        let src = ptr + *data as u64;
        let dst_p = ptr + *dst as u64;
        // SAFETY: non-overlapping f32 regions in the same arena buffer.
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                dst_p,
                src,
                data_n * 4,
                stream.cu_stream() as _,
            )
            .expect("rlx-cuda: scatter_nd data→dst");
        }
        drop(rec);
    }

    let strides = row_major_strides(data_shape);
    let mut s = [0u32; 4];
    let mut d = [0u32; 4];
    for i in 0..k {
        s[i] = strides[i];
        d[i] = data_shape[i].max(1);
    }

    let total = (num_updates * slice) as u32;
    if total == 0 {
        return true;
    }
    let kernel = scatter_nd_kernel(ctx);
    let (grid, block) = dispatch_grid_1d(total, 256);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let num_u = num_updates as u32;
    let slice_u = slice as u32;
    let k_u = k as u32;
    let zero = 0u32;
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *arena)
        .arg(&data_off)
        .arg(&idx_off)
        .arg(&upd_off)
        .arg(&dst_off)
        .arg(&num_u)
        .arg(&slice_u)
        .arg(&k_u)
        .arg(&s[0])
        .arg(&s[1])
        .arg(&s[2])
        .arg(&s[3])
        .arg(&d[0])
        .arg(&d[1])
        .arg(&d[2])
        .arg(&d[3])
        .arg(&zero);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: scatter_nd launch failed");
    }
    if rlx_ir::env::flag("RLX_CUDA_SCATTER_ND_TRACE") {
        eprintln!(
            "[scatter_nd_gpu] ok num_updates={num_updates} slice={slice} k={k} data_len={}",
            *data_len
        );
    }
    true
}
