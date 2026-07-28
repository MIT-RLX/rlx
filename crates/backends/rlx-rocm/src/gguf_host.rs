// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side GGUF `Op::DequantMatMul` for ROCm device arenas (CPU fallback).
//!
//! Thin adapter over [`rlx_gpu_host`]. GPU path uses the same `dequant_gguf`
//! kernel and [`gguf_scheme_id`] as CUDA. See
//! [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md).

use crate::device::RocmContext;
use crate::hip::HipBuffer;
use crate::host_stage::RocmArena;

pub use rlx_gpu_host::{gguf_scheme_id, scheme_from_id};

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
    let mut arena = RocmArena {
        ctx,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_dequant_matmul_gguf(
        &mut arena,
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
    let mut arena = RocmArena {
        ctx,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_dequant_grouped_matmul_gguf(
        &mut arena,
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

pub fn upload_param_bytes(
    ctx: &RocmContext,
    buffer: &mut HipBuffer<f32>,
    byte_off: usize,
    data: &[u8],
) {
    let mut arena = RocmArena {
        ctx,
        buffer: &*buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::upload_param_bytes(&mut arena, byte_off, data);
}

#[allow(clippy::too_many_arguments)]
pub fn run_dequant_matmul_mlx(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
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
    let mut arena = RocmArena {
        ctx,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_dequant_matmul_mlx(
        &mut arena,
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
