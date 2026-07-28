// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side GGUF K-quant `Op::DequantMatMul` for f32-uniform GPU arenas.
//!
//! Packed U8 weights live inline in the arena (first `N` bytes of each param
//! slot). When the GPU `dequant_gguf` path is unavailable this module does
//! D2H → `rlx_cpu::gguf_matmul` → H2D.
//!
//! Scheme ids are shared with Metal/CUDA/ROCm/WGPU — see [`gguf_scheme_id`] and
//! `docs/gguf-backend-paths.md`.

use crate::DeviceArena;
use rlx_ir::quant::QuantScheme;

/// Maps [`QuantScheme`] to the shared GPU `dequant_gguf` kernel scheme id.
pub fn gguf_scheme_id(scheme: QuantScheme) -> u32 {
    scheme
        .gpu_dequant_scheme_id()
        .unwrap_or_else(|| panic!("rlx-gpu-host gguf: unsupported scheme {scheme:?}"))
}

/// Inverse of [`gguf_scheme_id`].
pub fn scheme_from_id(scheme_id: u32) -> QuantScheme {
    QuantScheme::from_gpu_dequant_scheme_id(scheme_id)
        .unwrap_or_else(|| panic!("rlx-gpu-host gguf: bad scheme_id {scheme_id}"))
}

/// Read packed bytes that may start mid-f32-word in the arena.
fn dtoh_packed_bytes<A: DeviceArena>(a: &mut A, byte_off: usize, len: usize) -> Vec<u8> {
    let start_f32 = byte_off / 4;
    let end_f32 = (byte_off + len).div_ceil(4);
    let mut words = vec![0u8; (end_f32 - start_f32) * 4];
    a.dtoh(start_f32 * 4, &mut words);
    words[byte_off % 4..byte_off % 4 + len].to_vec()
}

/// Write packed bytes; aligned uploads skip the RMW round-trip.
fn htod_packed_bytes<A: DeviceArena>(a: &mut A, byte_off: usize, data: &[u8]) {
    if byte_off.is_multiple_of(4) && data.len().is_multiple_of(4) && !data.is_empty() {
        a.htod(byte_off, data);
        return;
    }
    let start_f32 = byte_off / 4;
    let end_f32 = (byte_off + data.len()).div_ceil(4);
    let mut words = vec![0u8; (end_f32 - start_f32) * 4];
    a.dtoh(start_f32 * 4, &mut words);
    words[byte_off % 4..byte_off % 4 + data.len()].copy_from_slice(data);
    a.htod(start_f32 * 4, &words);
}

/// Upload raw U8 param bytes into the f32 arena slot at `byte_off`.
pub fn upload_param_bytes<A: DeviceArena>(a: &mut A, byte_off: usize, data: &[u8]) {
    htod_packed_bytes(a, byte_off, data);
}

/// Fused GGUF dequant matmul on the host; syncs around D2H/H2D.
pub fn run_dequant_matmul_gguf<A: DeviceArena>(
    a: &mut A,
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

    a.sync();

    let mut x_bytes = vec![0u8; m * k * 4];
    a.dtoh(x_byte_off, &mut x_bytes);
    let x_host: &[f32] = bytemuck::cast_slice(&x_bytes);

    let w_host = dtoh_packed_bytes(a, w_byte_off, total_bytes);

    let mut out_host = vec![0f32; m * n];
    rlx_cpu::gguf_matmul::gguf_matmul_bt(x_host, &w_host, &mut out_host, m, k, n, scheme);

    a.htod(out_byte_off, bytemuck::cast_slice(&out_host));
}

/// Fused GGUF dequant grouped matmul on the host (MoE expert stacks).
pub fn run_dequant_grouped_matmul_gguf<A: DeviceArena>(
    a: &mut A,
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

    a.sync();

    let mut x_bytes = vec![0u8; m * k * 4];
    a.dtoh(x_byte_off, &mut x_bytes);
    let x_host: &[f32] = bytemuck::cast_slice(&x_bytes);

    let w_host = dtoh_packed_bytes(a, w_byte_off, total_bytes);

    let mut idx_bytes = vec![0u8; m * 4];
    a.dtoh(idx_byte_off, &mut idx_bytes);
    let idx_host: &[f32] = bytemuck::cast_slice(&idx_bytes);

    let mut out_host = vec![0f32; m * n];
    rlx_cpu::gguf_matmul::gguf_grouped_matmul_bt(
        x_host,
        &w_host,
        idx_host,
        &mut out_host,
        m,
        k,
        n,
        num_experts,
        scheme,
    );

    a.htod(out_byte_off, bytemuck::cast_slice(&out_host));
}
