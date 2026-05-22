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
//! Host-side GGUF K-quant `Op::DequantMatMul` for CUDA device arenas.
//!
//! CUDA's f32 arena stores packed U8 weights inline (first `N` bytes of
//! each param slot). This module D2H → `rlx_cpu::gguf_matmul` → H2D.

use cudarc::driver::{CudaSlice, CudaStream};
use rlx_ir::quant::QuantScheme;
use std::sync::Arc;

pub fn gguf_scheme_id(scheme: QuantScheme) -> u32 {
    match scheme {
        QuantScheme::GgufQ4K => 0,
        QuantScheme::GgufQ5K => 1,
        QuantScheme::GgufQ6K => 2,
        QuantScheme::GgufQ8K => 3,
        QuantScheme::GgufQ2K => 4,
        QuantScheme::GgufQ3K => 5,
        other => panic!("rlx-cuda gguf_host: unsupported scheme {other:?}"),
    }
}

pub fn scheme_from_id(scheme_id: u32) -> QuantScheme {
    match scheme_id {
        0 => QuantScheme::GgufQ4K,
        1 => QuantScheme::GgufQ5K,
        2 => QuantScheme::GgufQ6K,
        3 => QuantScheme::GgufQ8K,
        4 => QuantScheme::GgufQ2K,
        5 => QuantScheme::GgufQ3K,
        _ => panic!("rlx-cuda gguf_host: bad scheme_id {scheme_id}"),
    }
}

fn dtoh_bytes(
    stream: &Arc<CudaStream>,
    buffer: &CudaSlice<f32>,
    byte_off: usize,
    len: usize,
) -> Vec<u8> {
    let start_f32 = byte_off / 4;
    let end_byte = byte_off + len;
    let end_f32 = end_byte.div_ceil(4);
    let mut words = vec![0f32; end_f32 - start_f32];
    stream
        .memcpy_dtoh(
            &buffer.slice(start_f32..end_f32),
            &mut words,
        )
        .expect("rlx-cuda: gguf dtoh failed");
    let mut raw = vec![0u8; words.len() * 4];
    for (i, w) in words.iter().enumerate() {
        raw[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    raw[byte_off % 4..byte_off % 4 + len].to_vec()
}

fn htod_bytes(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    byte_off: usize,
    data: &[u8],
) {
    let start_f32 = byte_off / 4;
    let end_byte = byte_off + data.len();
    let end_f32 = end_byte.div_ceil(4);
    let mut words = vec![0f32; end_f32 - start_f32];
    stream
        .memcpy_dtoh(
            &buffer.slice(start_f32..end_f32),
            &mut words,
        )
        .expect("rlx-cuda: gguf htod staging dtoh failed");
    let mut raw = vec![0u8; words.len() * 4];
    for (i, w) in words.iter().enumerate() {
        raw[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    raw[byte_off % 4..byte_off % 4 + data.len()].copy_from_slice(data);
    for (i, chunk) in raw.chunks_exact(4).enumerate() {
        words[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    stream
        .memcpy_htod(&words, &mut buffer.slice_mut(start_f32..end_f32))
        .expect("rlx-cuda: gguf htod failed");
}

/// Fused GGUF dequant matmul on the host; syncs the stream around D2H/H2D.
pub fn run_dequant_matmul_gguf(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
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

    stream.synchronize().expect("rlx-cuda: gguf pre-sync failed");

    let x_f32_off = x_byte_off / 4;
    let mut x_host = vec![0f32; m * k];
    stream
        .memcpy_dtoh(
            &buffer.slice(x_f32_off..x_f32_off + m * k),
            &mut x_host,
        )
        .expect("rlx-cuda: gguf x dtoh failed");

    let w_host = dtoh_bytes(stream, buffer, w_byte_off, total_bytes);

    let mut out_host = vec![0f32; m * n];
    rlx_cpu::gguf_matmul::gguf_matmul_bt(
        &x_host,
        &w_host,
        &mut out_host,
        m,
        k,
        n,
        scheme,
    );

    let out_f32_off = out_byte_off / 4;
    stream
        .memcpy_htod(
            &out_host,
            &mut buffer.slice_mut(out_f32_off..out_f32_off + m * n),
        )
        .expect("rlx-cuda: gguf out htod failed");
}

/// Fused GGUF dequant grouped matmul on the host (MoE expert stacks).
pub fn run_dequant_grouped_matmul_gguf(
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
    out_byte_off: usize,
) {
    let scheme = scheme_from_id(scheme_id);
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let block_elems = scheme.gguf_block_size() as usize;
    let slab_bytes = (k * n) / block_elems * block_bytes;
    let total_bytes = num_experts * slab_bytes;

    stream.synchronize().expect("rlx-cuda: grouped gguf pre-sync failed");

    let x_f32_off = x_byte_off / 4;
    let mut x_host = vec![0f32; m * k];
    stream
        .memcpy_dtoh(
            &buffer.slice(x_f32_off..x_f32_off + m * k),
            &mut x_host,
        )
        .expect("rlx-cuda: grouped gguf x dtoh failed");

    let w_host = dtoh_bytes(stream, buffer, w_byte_off, total_bytes);

    let idx_f32_off = idx_byte_off / 4;
    let mut idx_host = vec![0f32; m];
    stream
        .memcpy_dtoh(
            &buffer.slice(idx_f32_off..idx_f32_off + m),
            &mut idx_host,
        )
        .expect("rlx-cuda: grouped gguf idx dtoh failed");

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
    stream
        .memcpy_htod(
            &out_host,
            &mut buffer.slice_mut(out_f32_off..out_f32_off + m * n),
        )
        .expect("rlx-cuda: grouped gguf out htod failed");
}

/// Upload raw U8 param bytes into the f32 arena slot at `byte_off`.
pub fn upload_param_bytes(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    byte_off: usize,
    data: &[u8],
) {
    htod_bytes(stream, buffer, byte_off, data);
}
