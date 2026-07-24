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

//! Host-side GGUF `Op::DequantMatMul` for wgpu arenas (CPU dequant fallback).
//!
//! Thin adapter over [`rlx_gpu_host`]. Prefer [`crate::gguf_gpu`] when arena
//! planning reserves dequant scratch. See
//! [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md).

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

pub use rlx_gpu_host::{gguf_scheme_id, scheme_from_id};

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
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_dequant_matmul_gguf(
        &mut a,
        m,
        k,
        n,
        scheme_id,
        x_byte_off,
        w_byte_off,
        out_byte_off,
    );
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
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_dequant_grouped_matmul_gguf(
        &mut a,
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
}

#[allow(clippy::too_many_arguments)]
pub fn run_dequant_matmul_mlx(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    m: usize,
    k: usize,
    n: usize,
    scheme: rlx_ir::quant::QuantScheme,
    x_byte_off: usize,
    w_byte_off: usize,
    scale_byte_off: usize,
    zp_byte_off: usize,
    out_byte_off: usize,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_dequant_matmul_mlx(
        &mut a,
        m,
        k,
        n,
        scheme,
        x_byte_off,
        w_byte_off,
        scale_byte_off,
        zp_byte_off,
        out_byte_off,
    );
}
