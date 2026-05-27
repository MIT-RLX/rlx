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
//! Host-side GGUF K-quant `Op::DequantMatMul` for wgpu arenas.

use crate::buffer::Arena;
use rlx_ir::quant::QuantScheme;

pub fn gguf_scheme_id(scheme: QuantScheme) -> u32 {
    match scheme {
        QuantScheme::GgufQ4K => 0,
        QuantScheme::GgufQ5K => 1,
        QuantScheme::GgufQ6K => 2,
        QuantScheme::GgufQ8K => 3,
        other => panic!("rlx-wgpu gguf_host: unsupported scheme {other:?}"),
    }
}

pub fn scheme_from_id(scheme_id: u32) -> QuantScheme {
    match scheme_id {
        0 => QuantScheme::GgufQ4K,
        1 => QuantScheme::GgufQ5K,
        2 => QuantScheme::GgufQ6K,
        3 => QuantScheme::GgufQ8K,
        _ => panic!("rlx-wgpu gguf_host: bad scheme_id {scheme_id}"),
    }
}

pub fn run_dequant_matmul_gguf(
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
    let scheme = scheme_from_id(scheme_id);
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let block_elems = scheme.gguf_block_size() as usize;
    let total_bytes = (k * n) / block_elems * block_bytes;

    let x_bytes = arena.read_bytes_range(device, queue, x_byte_off, m * k * 4);
    let x_host: Vec<f32> = x_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let w_host = arena.read_bytes_range(device, queue, w_byte_off, total_bytes);

    let mut out_host = vec![0f32; m * n];
    rlx_cpu::gguf_matmul::gguf_matmul_bt(&x_host, &w_host, &mut out_host, m, k, n, scheme);

    let out_bytes: Vec<u8> = out_host.iter().flat_map(|v| v.to_le_bytes()).collect();
    arena.write_bytes_range(queue, out_byte_off, &out_bytes);
}

/// Fused GGUF dequant grouped matmul on the host (MoE expert stacks).
pub fn run_dequant_grouped_matmul_gguf(
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
    out_byte_off: usize,
) {
    let scheme = scheme_from_id(scheme_id);
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let block_elems = scheme.gguf_block_size() as usize;
    let slab_bytes = (k * n) / block_elems * block_bytes;
    let total_bytes = num_experts * slab_bytes;

    let x_bytes = arena.read_bytes_range(device, queue, x_byte_off, m * k * 4);
    let x_host: Vec<f32> = x_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let w_host = arena.read_bytes_range(device, queue, w_byte_off, total_bytes);

    let idx_bytes = arena.read_bytes_range(device, queue, idx_byte_off, m * 4);
    let idx_host: Vec<f32> = idx_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

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

    let out_bytes: Vec<u8> = out_host.iter().flat_map(|v| v.to_le_bytes()).collect();
    arena.write_bytes_range(queue, out_byte_off, &out_bytes);
}
