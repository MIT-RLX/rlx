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
//! Host-side GGUF K-quant `Op::DequantMatMul` for ROCm device arenas.

use crate::device::RocmContext;
use crate::hip::HipBuffer;
use rlx_ir::quant::QuantScheme;
use std::sync::Arc;

pub fn gguf_scheme_id(scheme: QuantScheme) -> u32 {
    match scheme {
        QuantScheme::GgufQ4K => 0,
        QuantScheme::GgufQ5K => 1,
        QuantScheme::GgufQ6K => 2,
        QuantScheme::GgufQ8K => 3,
        other => panic!("rlx-rocm gguf_host: unsupported scheme {other:?}"),
    }
}

pub fn scheme_from_id(scheme_id: u32) -> QuantScheme {
    match scheme_id {
        0 => QuantScheme::GgufQ4K,
        1 => QuantScheme::GgufQ5K,
        2 => QuantScheme::GgufQ6K,
        3 => QuantScheme::GgufQ8K,
        _ => panic!("rlx-rocm gguf_host: bad scheme_id {scheme_id}"),
    }
}

fn dtoh_bytes(rt: &Arc<crate::hip::HipRuntime>, ptr: u64, byte_off: usize, len: usize) -> Vec<u8> {
    let start_f32 = byte_off / 4;
    let end_byte = byte_off + len;
    let end_f32 = end_byte.div_ceil(4);
    let mut words = vec![0f32; end_f32 - start_f32];
    let src = ptr + (start_f32 as u64) * 4;
    unsafe {
        let _ = (rt.hip_memcpy_dtoh)(words.as_mut_ptr() as *mut _, src, words.len() * 4);
    }
    let mut raw = vec![0u8; words.len() * 4];
    for (i, w) in words.iter().enumerate() {
        raw[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    raw[byte_off % 4..byte_off % 4 + len].to_vec()
}

fn htod_bytes(rt: &Arc<crate::hip::HipRuntime>, ptr: u64, byte_off: usize, data: &[u8]) {
    let start_f32 = byte_off / 4;
    let end_byte = byte_off + data.len();
    let end_f32 = end_byte.div_ceil(4);
    let mut words = vec![0f32; end_f32 - start_f32];
    let src = ptr + (start_f32 as u64) * 4;
    unsafe {
        let _ = (rt.hip_memcpy_dtoh)(words.as_mut_ptr() as *mut _, src, words.len() * 4);
    }
    let mut raw = vec![0u8; words.len() * 4];
    for (i, w) in words.iter().enumerate() {
        raw[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    raw[byte_off % 4..byte_off % 4 + data.len()].copy_from_slice(data);
    for (i, chunk) in raw.chunks_exact(4).enumerate() {
        words[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    let dst = ptr + (start_f32 as u64) * 4;
    unsafe {
        let _ = (rt.hip_memcpy_htod)(dst, words.as_ptr() as *const _, words.len() * 4);
    }
}

pub fn run_dequant_matmul_gguf(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    m: usize,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) {
    let scheme = scheme_from_id(scheme_id);
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let block_elems = scheme.gguf_block_size() as usize;
    let total_bytes = (k * n) / block_elems * block_bytes;
    let rt = &ctx.runtime;

    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
    }

    let x_f32_off = x_byte_off / 4;
    let mut x_host = vec![0f32; m * k];
    unsafe {
        let _ = (rt.hip_memcpy_dtoh)(
            x_host.as_mut_ptr() as *mut _,
            buffer.ptr + (x_f32_off as u64) * 4,
            x_host.len() * 4,
        );
    }

    let w_host = dtoh_bytes(rt, buffer.ptr, w_byte_off, total_bytes);
    let mut out_host = vec![0f32; m * n];
    rlx_cpu::gguf_matmul::gguf_matmul_bt(&x_host, &w_host, &mut out_host, m, k, n, scheme);

    let out_f32_off = out_byte_off / 4;
    unsafe {
        let _ = (rt.hip_memcpy_htod)(
            buffer.ptr + (out_f32_off as u64) * 4,
            out_host.as_ptr() as *const _,
            out_host.len() * 4,
        );
    }
}

/// Fused GGUF dequant grouped matmul on the host (MoE expert stacks).
pub fn run_dequant_grouped_matmul_gguf(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    idx_byte_off: usize,
    out_byte_off: usize,
) {
    let scheme = scheme_from_id(scheme_id);
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let block_elems = scheme.gguf_block_size() as usize;
    let slab_bytes = (k * n) / block_elems * block_bytes;
    let total_bytes = num_experts * slab_bytes;
    let rt = &ctx.runtime;

    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
    }

    let x_f32_off = x_byte_off / 4;
    let mut x_host = vec![0f32; m * k];
    unsafe {
        let _ = (rt.hip_memcpy_dtoh)(
            x_host.as_mut_ptr() as *mut _,
            buffer.ptr + (x_f32_off as u64) * 4,
            x_host.len() * 4,
        );
    }

    let w_host = dtoh_bytes(rt, buffer.ptr, w_byte_off, total_bytes);

    let idx_f32_off = idx_byte_off / 4;
    let mut idx_host = vec![0f32; m];
    unsafe {
        let _ = (rt.hip_memcpy_dtoh)(
            idx_host.as_mut_ptr() as *mut _,
            buffer.ptr + (idx_f32_off as u64) * 4,
            idx_host.len() * 4,
        );
    }

    let mut out_host = vec![0f32; m * n];
    rlx_cpu::gguf_matmul::gguf_grouped_matmul_bt(
        &x_host,
        &w_host,
        &idx_host,
        &mut out_host,
        m,
        k,
        n,
        num_experts,
        scheme,
    );

    let out_f32_off = out_byte_off / 4;
    unsafe {
        let _ = (rt.hip_memcpy_htod)(
            buffer.ptr + (out_f32_off as u64) * 4,
            out_host.as_ptr() as *const _,
            out_host.len() * 4,
        );
    }
}

pub fn upload_param_bytes(
    ctx: &RocmContext,
    buffer: &mut HipBuffer<f32>,
    byte_off: usize,
    data: &[u8],
) {
    htod_bytes(&ctx.runtime, buffer.ptr, byte_off, data);
}
