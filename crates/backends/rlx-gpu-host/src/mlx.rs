// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Host-side MLX affine / mxfp `Op::DequantMatMul` for f32-uniform GPU arenas.
//!
//! D2H → [`rlx_mlx_io`] dequant-matmul → H2D. Used when GPU kernels are
//! disabled (`RLX_MLX_DEQUANT_GPU_DISABLE`) or as a reference / deferred-host
//! fallback.

use crate::DeviceArena;
use rlx_ir::quant::QuantScheme;
use rlx_mlx_io::{
    dequant_matmul_affine, dequant_mxfp4_f32, dequant_mxfp8_f32, pack_factor,
    validate_dequant_matmul_dims,
};

/// When set, backends run MLX `DequantMatMul` on the host instead of GPU kernels.
///
/// Also honors Metal's broader `RLX_METAL_DEQUANT_GPU_DISABLE` so one flag can
/// A/B all Metal dequant paths.
pub fn mlx_dequant_gpu_disabled() -> bool {
    rlx_ir::env::flag("RLX_MLX_DEQUANT_GPU_DISABLE")
        || rlx_ir::env::flag("RLX_METAL_DEQUANT_GPU_DISABLE")
}

fn dtoh_packed_bytes<A: DeviceArena>(a: &mut A, byte_off: usize, len: usize) -> Vec<u8> {
    let start_f32 = byte_off / 4;
    let end_f32 = (byte_off + len).div_ceil(4);
    let mut words = vec![0u8; (end_f32 - start_f32) * 4];
    a.dtoh(start_f32 * 4, &mut words);
    words[byte_off % 4..byte_off % 4 + len].to_vec()
}

fn bpp_for_bits(bits: u8) -> usize {
    match bits {
        3 | 6 => 3,
        5 => 5,
        _ => 1,
    }
}

fn mlx_weight_bytes(scheme: QuantScheme, k: usize, n: usize) -> Result<usize, String> {
    validate_dequant_matmul_dims(scheme, k, n, None).map_err(|e| e.to_string())?;
    match scheme {
        QuantScheme::MlxAffine { bits, group_size } => {
            let gs = group_size as usize;
            let n_groups = k / gs;
            let pf = pack_factor(bits as u32).map_err(|e| e.to_string())? as usize;
            let packs = gs / pf;
            Ok(n * n_groups * packs * bpp_for_bits(bits))
        }
        QuantScheme::MlxMxfp4 { .. } => Ok(n * k / 2),
        QuantScheme::MlxMxfp8 { .. } => Ok(n * k),
        other => Err(format!("rlx-gpu-host mlx: unexpected scheme {other:?}")),
    }
}

/// Fused MLX dequant matmul on the host; syncs around D2H/H2D.
#[allow(clippy::too_many_arguments)]
pub fn run_dequant_matmul_mlx<A: DeviceArena>(
    a: &mut A,
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    x_byte_off: usize,
    w_byte_off: usize,
    scale_byte_off: usize,
    zp_byte_off: usize,
    out_byte_off: usize,
) {
    a.sync();

    let mut x_bytes = vec![0u8; m * k * 4];
    a.dtoh(x_byte_off, &mut x_bytes);
    let x_host: &[f32] = bytemuck::cast_slice(&x_bytes);

    let w_len = match mlx_weight_bytes(scheme, k, n) {
        Ok(n) => n,
        Err(e) => panic!("rlx-gpu-host mlx: {e}"),
    };
    let w_host = dtoh_packed_bytes(a, w_byte_off, w_len);
    if let Err(e) = validate_dequant_matmul_dims(scheme, k, n, Some(w_host.len())) {
        panic!("rlx-gpu-host mlx: {e}");
    }

    let out = match scheme {
        QuantScheme::MlxAffine { bits, group_size } => {
            let gs = group_size as usize;
            let n_groups = k / gs;
            let mut scale_bytes = vec![0u8; n * n_groups * 4];
            a.dtoh(scale_byte_off, &mut scale_bytes);
            let scales: &[f32] = bytemuck::cast_slice(&scale_bytes);
            let mut bias_bytes = vec![0u8; n * n_groups * 4];
            a.dtoh(zp_byte_off, &mut bias_bytes);
            let biases: &[f32] = bytemuck::cast_slice(&bias_bytes);
            match dequant_matmul_affine(
                x_host,
                &w_host,
                scales,
                biases,
                bits as u32,
                group_size,
                m,
                k,
                n,
            ) {
                Ok(v) => v,
                Err(e) => panic!("rlx-gpu-host mlx affine: {e}"),
            }
        }
        QuantScheme::MlxMxfp4 { group_size } => {
            let gs = group_size as usize;
            let n_groups = k / gs;
            let scales_u8 = dtoh_packed_bytes(a, scale_byte_off, n * n_groups);
            let w_f = match dequant_mxfp4_f32(&w_host, &scales_u8, group_size, n, n_groups) {
                Ok(v) => v,
                Err(e) => panic!("rlx-gpu-host mlx mxfp4: {e}"),
            };
            matmul_nk(x_host, &w_f, m, k, n)
        }
        QuantScheme::MlxMxfp8 { group_size } => {
            let gs = group_size as usize;
            let n_groups = k / gs;
            let scales_u8 = dtoh_packed_bytes(a, scale_byte_off, n * n_groups);
            let w_f = match dequant_mxfp8_f32(&w_host, &scales_u8, group_size, n, n_groups) {
                Ok(v) => v,
                Err(e) => panic!("rlx-gpu-host mlx mxfp8: {e}"),
            };
            matmul_nk(x_host, &w_f, m, k, n)
        }
        other => panic!("rlx-gpu-host mlx: unexpected scheme {other:?}"),
    };

    a.htod(out_byte_off, bytemuck::cast_slice(&out));
}

/// Host MLX-affine MoE grouped matmul (per-row expert dequant). Mirrors
/// [`run_dequant_matmul_mlx`] but the packed weight/scales/biases are
/// `num_experts` contiguous slabs and `expert_idx` (`[m]` f32) selects the
/// slab per row. Reused by wgpu / vulkan / cuda host-delegate paths.
#[allow(clippy::too_many_arguments)]
pub fn run_dequant_grouped_matmul_mlx<A: DeviceArena>(
    a: &mut A,
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme: QuantScheme,
    x_byte_off: usize,
    w_byte_off: usize,
    scale_byte_off: usize,
    zp_byte_off: usize,
    idx_byte_off: usize,
    out_byte_off: usize,
) {
    let (bits, group_size) = match scheme {
        QuantScheme::MlxAffine { bits, group_size } => (bits, group_size as usize),
        other => panic!("rlx-gpu-host grouped mlx: expected MlxAffine, got {other:?}"),
    };
    let n_groups = k / group_size.max(1);
    let per_expert_w = match mlx_weight_bytes(scheme, k, n) {
        Ok(v) => v,
        Err(e) => panic!("rlx-gpu-host grouped mlx: {e}"),
    };
    let sb = num_experts * n * n_groups; // f32 scales/biases across all experts

    a.sync();

    let mut x_bytes = vec![0u8; m * k * 4];
    a.dtoh(x_byte_off, &mut x_bytes);
    let x_host: &[f32] = bytemuck::cast_slice(&x_bytes);

    let w_host = dtoh_packed_bytes(a, w_byte_off, num_experts * per_expert_w);

    let mut scale_bytes = vec![0u8; sb * 4];
    a.dtoh(scale_byte_off, &mut scale_bytes);
    let scales: &[f32] = bytemuck::cast_slice(&scale_bytes);

    let mut bias_bytes = vec![0u8; sb * 4];
    a.dtoh(zp_byte_off, &mut bias_bytes);
    let biases: &[f32] = bytemuck::cast_slice(&bias_bytes);

    let mut idx_bytes = vec![0u8; m * 4];
    a.dtoh(idx_byte_off, &mut idx_bytes);
    let idx_host: &[f32] = bytemuck::cast_slice(&idx_bytes);

    let mut out_host = vec![0f32; m * n];
    rlx_cpu::thunk::dequant_grouped_matmul_affine_bt(
        x_host,
        &w_host,
        scales,
        biases,
        idx_host,
        &mut out_host,
        m,
        k,
        n,
        num_experts,
        bits as u32,
        group_size,
    );

    a.htod(out_byte_off, bytemuck::cast_slice(&out_host));
}

fn matmul_nk(x: &[f32], w_nk: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += x[i * k + p] * w_nk[j * k + p];
            }
            out[i * n + j] = acc;
        }
    }
    out
}
