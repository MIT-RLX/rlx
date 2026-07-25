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
//! Thin adapter over [`rlx_gpu_host`]. Scheme ids for the GPU kernel are shared
//! with Metal/ROCm/WGPU — see [`gguf_scheme_id`] and
//! [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md).

use crate::host_stage::CudaArena;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

pub use rlx_gpu_host::{gguf_scheme_id, scheme_from_id};

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
    let mut arena = CudaArena {
        stream,
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
    let mut arena = CudaArena {
        stream,
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

/// MLX-affine MoE grouped matmul on the host (packed expert stacks).
#[allow(clippy::too_many_arguments)]
pub fn run_dequant_grouped_matmul_mlx(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme: rlx_ir::quant::QuantScheme,
    x_byte_off: usize,
    w_byte_off: usize,
    scale_byte_off: usize,
    zp_byte_off: usize,
    idx_byte_off: usize,
    out_byte_off: usize,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_dequant_grouped_matmul_mlx(
        &mut arena,
        m,
        k,
        n,
        num_experts,
        scheme,
        x_byte_off,
        w_byte_off,
        scale_byte_off,
        zp_byte_off,
        idx_byte_off,
        out_byte_off,
    );
}

/// Upload raw U8 param bytes into the f32 arena slot at `byte_off`.
pub fn upload_param_bytes(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    byte_off: usize,
    data: &[u8],
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::upload_param_bytes(&mut arena, byte_off, data);
}

/// Host MLX affine/mxfp DequantMatMul.
#[allow(clippy::too_many_arguments)]
pub fn run_dequant_matmul_mlx(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
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
    let mut arena = CudaArena {
        stream,
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
