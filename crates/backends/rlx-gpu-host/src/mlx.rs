// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side MLX affine / mxfp `Op::DequantMatMul` for f32-uniform GPU arenas.
//!
//! D2H → [`rlx_mlx_io`] dequant-matmul → H2D. Used when GPU kernels are
//! disabled (`RLX_MLX_DEQUANT_GPU_DISABLE`) or as a reference / deferred-host
//! fallback.

use crate::DeviceArena;
use rlx_ir::quant::QuantScheme;
use rlx_mlx_io::{
    dequant_matvec_affine, dequant_mxfp4_f32, dequant_mxfp8_f32, pack_factor,
    validate_dequant_matmul_dims,
};

// Host-delegate phase profiling (RLX_HD_PROFILE=1): accumulate ns spent in each
// phase across ALL DequantMatMul/grouped calls so we can see whether an
// oversubscribed Vulkan/CUDA MoE stage is bound by GPU sync, dtoh, CPU compute,
// or htod. Dumped (and reset) each call so the last line of a forward is the total.
use std::sync::atomic::{AtomicU64, Ordering};
static HD_SYNC_NS: AtomicU64 = AtomicU64::new(0);
static HD_DTOH_NS: AtomicU64 = AtomicU64::new(0);
static HD_COMPUTE_NS: AtomicU64 = AtomicU64::new(0);
static HD_HTOD_NS: AtomicU64 = AtomicU64::new(0);
static HD_CALLS: AtomicU64 = AtomicU64::new(0);
fn hd_prof() -> bool {
    rlx_ir::env::flag("RLX_HD_PROFILE")
}
fn hd_dump() {
    let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
    eprintln!(
        "[HD-PROFILE] calls={} sync={:.0}ms dtoh={:.0}ms compute={:.0}ms htod={:.0}ms",
        HD_CALLS.load(Ordering::Relaxed),
        ms(&HD_SYNC_NS),
        ms(&HD_DTOH_NS),
        ms(&HD_COMPUTE_NS),
        ms(&HD_HTOD_NS),
    );
}

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
    let prof = hd_prof();
    let tt = std::time::Instant::now();
    a.sync();
    if prof {
        HD_SYNC_NS.fetch_add(tt.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let tt = std::time::Instant::now();
    let mut x_bytes = vec![0u8; m * k * 4];
    a.dtoh(x_byte_off, &mut x_bytes);
    let x_host: &[f32] = bytemuck::cast_slice(&x_bytes);

    let w_len = match mlx_weight_bytes(scheme, k, n) {
        Ok(n) => n,
        Err(e) => panic!("rlx-gpu-host mlx: {e}"),
    };
    let w_host = dtoh_packed_bytes(a, w_byte_off, w_len);
    if prof {
        HD_DTOH_NS.fetch_add(tt.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if let Err(e) = validate_dequant_matmul_dims(scheme, k, n, Some(w_host.len())) {
        panic!("rlx-gpu-host mlx: {e}");
    }

    let out = match scheme {
        QuantScheme::MlxAffine { bits, group_size } => {
            let gs = group_size as usize;
            let n_groups = k / gs;
            let tt = std::time::Instant::now();
            let mut scale_bytes = vec![0u8; n * n_groups * 4];
            a.dtoh(scale_byte_off, &mut scale_bytes);
            let scales: &[f32] = bytemuck::cast_slice(&scale_bytes);
            let mut bias_bytes = vec![0u8; n * n_groups * 4];
            a.dtoh(zp_byte_off, &mut bias_bytes);
            let biases: &[f32] = bytemuck::cast_slice(&bias_bytes);
            if prof {
                HD_DTOH_NS.fetch_add(tt.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            // Fused per-row matvec (parallel over n, no n×k f32 materialization) —
            // this is the dense attention/shared-expert host-delegate matmul; m is
            // the token count (small), so loop rows and let each matvec use the
            // cores. The old materialize-then-naive-matmul was single-threaded and
            // pure memory traffic → the dominant cost of a Vulkan/cuda MoE stage.
            let tt = std::time::Instant::now();
            let mut v = vec![0f32; m * n];
            for (i, out_row) in v.chunks_mut(n).enumerate() {
                let xi = &x_host[i * k..(i + 1) * k];
                match dequant_matvec_affine(
                    xi,
                    &w_host,
                    scales,
                    biases,
                    bits as u32,
                    group_size,
                    k,
                    n,
                ) {
                    Ok(o) => out_row.copy_from_slice(&o),
                    Err(e) => panic!("rlx-gpu-host mlx affine matvec: {e}"),
                }
            }
            if prof {
                HD_COMPUTE_NS.fetch_add(tt.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            v
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

    let tt = std::time::Instant::now();
    a.htod(out_byte_off, bytemuck::cast_slice(&out));
    if prof {
        HD_HTOD_NS.fetch_add(tt.elapsed().as_nanos() as u64, Ordering::Relaxed);
        HD_CALLS.fetch_add(1, Ordering::Relaxed);
        hd_dump();
    }
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
    scale_bf16: bool,
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
    let sb = num_experts * n * n_groups; // scales/biases across all experts
    // Scales/biases may be stored BF16 on the arena (halves the resident bytes);
    // widen to f32 for the affine kernel. Otherwise they are already f32.
    let scale_elt = if scale_bf16 { 2 } else { 4 };
    let widen = |bytes: &[u8]| -> Vec<f32> {
        if scale_bf16 {
            bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        } else {
            bytemuck::cast_slice::<u8, f32>(bytes).to_vec()
        }
    };

    let prof = hd_prof();
    let tt = std::time::Instant::now();
    a.sync();
    if prof {
        HD_SYNC_NS.fetch_add(tt.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let tt = std::time::Instant::now();
    let mut x_bytes = vec![0u8; m * k * 4];
    a.dtoh(x_byte_off, &mut x_bytes);
    let x_host: &[f32] = bytemuck::cast_slice(&x_bytes);

    // Learn which experts are actually routed (top-k of `num_experts`) by copying
    // back the tiny index first, then copy back ONLY those experts' code/scale/bias
    // slabs. The old path copied ALL `num_experts` slabs every forward (~9.7 GB
    // single-threaded for amd's 12-layer stage), even though only ~top_k*seq are
    // used — the dominant cost of a host-delegate MoE. `_` = unused `sb`.
    let _ = sb;
    let mut idx_bytes = vec![0u8; m * 4];
    a.dtoh(idx_byte_off, &mut idx_bytes);
    let idx_host: &[f32] = bytemuck::cast_slice(&idx_bytes);
    let row_experts: Vec<usize> = idx_host
        .iter()
        .map(|&x| (x as usize).min(num_experts.saturating_sub(1)))
        .collect();
    let mut distinct = row_experts.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let ne = distinct.len().max(1);
    let mut remap = vec![0u32; num_experts.max(1)];
    for (ci, &e) in distinct.iter().enumerate() {
        remap[e] = ci as u32;
    }
    let pe_sb = n * n_groups; // scale/bias elems per expert
    let mut w_host = vec![0u8; ne * per_expert_w];
    let mut scales_v = vec![0f32; ne * pe_sb];
    let mut biases_v = vec![0f32; ne * pe_sb];
    let se = pe_sb * scale_elt; // scale/bias bytes per expert
    for (ci, &e) in distinct.iter().enumerate() {
        // `dtoh_packed_bytes` reads word-aligned then slices exact bytes — safe for
        // arbitrary (possibly unaligned, e.g. bf16) per-expert offsets.
        let wslab = dtoh_packed_bytes(a, w_byte_off + e * per_expert_w, per_expert_w);
        w_host[ci * per_expert_w..(ci + 1) * per_expert_w].copy_from_slice(&wslab);
        let sslab = dtoh_packed_bytes(a, scale_byte_off + e * se, se);
        scales_v[ci * pe_sb..(ci + 1) * pe_sb].copy_from_slice(&widen(&sslab));
        let bslab = dtoh_packed_bytes(a, zp_byte_off + e * se, se);
        biases_v[ci * pe_sb..(ci + 1) * pe_sb].copy_from_slice(&widen(&bslab));
    }
    let scales: &[f32] = &scales_v;
    let biases: &[f32] = &biases_v;
    // Rewrite each row's expert id to its compact index (0..ne).
    let idx_compact: Vec<f32> = row_experts.iter().map(|&e| remap[e] as f32).collect();
    if prof {
        HD_DTOH_NS.fetch_add(tt.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let tt = std::time::Instant::now();
    let mut out_host = vec![0f32; m * n];
    rlx_cpu::thunk::dequant_grouped_matmul_affine_bt(
        x_host,
        &w_host,
        scales,
        biases,
        &idx_compact,
        &mut out_host,
        m,
        k,
        n,
        ne,
        bits as u32,
        group_size,
    );
    if prof {
        HD_COMPUTE_NS.fetch_add(tt.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let tt = std::time::Instant::now();
    a.htod(out_byte_off, bytemuck::cast_slice(&out_host));
    if prof {
        HD_HTOD_NS.fetch_add(tt.elapsed().as_nanos() as u64, Ordering::Relaxed);
        HD_CALLS.fetch_add(1, Ordering::Relaxed);
        hd_dump();
    }
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
