// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fused GGUF K-quant dequant + matmul without materializing full F32
//! weights (Tier C.11).
//!
//! Computes `C[m,n] = A[m,k] @ B^T` where `B` is `[n,k]` row-major in
//! packed GGUF layout. For Q4_K decode GEMVs (`m==1`, `k` multiple of 256)
//! activations are quantized once to Q8_K and dotted with int8 NEON kernels
//! (llama.cpp-style); other schemes still fold one 256-element super-block
//! at a time into stack storage.

use rlx_gguf::QK_K;
use rlx_ir::quant::QuantScheme;

pub(crate) fn dequant_block(scheme: QuantScheme, block: &[u8], out: &mut [f32; QK_K]) {
    match scheme {
        QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k_block(block, out),
        QuantScheme::GgufQ5K => rlx_gguf::dequant_q5_k_block(block, out),
        QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k_block(block, out),
        QuantScheme::GgufQ8K => rlx_gguf::dequant_q8_k_block(block, out),
        QuantScheme::GgufQ2K => rlx_gguf::dequant_q2_k_block(block, out),
        QuantScheme::GgufQ3K => rlx_gguf::dequant_q3_k_block(block, out),
        QuantScheme::GgufQ4_0 => rlx_gguf::dequant_q4_0_block(block, &mut out[..rlx_gguf::QK4_0]),
        QuantScheme::GgufQ4_1 => {
            rlx_gguf::dequant_q4_1_block(block, &mut out[..32]);
        }
        QuantScheme::GgufQ5_0 => {
            rlx_gguf::dequant_q5_0_block(block, &mut out[..32]);
        }
        QuantScheme::GgufQ5_1 => {
            rlx_gguf::dequant_q5_1_block(block, &mut out[..32]);
        }
        QuantScheme::GgufQ8_0 => rlx_gguf::dequant_q8_0_block(block, &mut out[..rlx_gguf::QK8_0]),
        // Block-level fast paths for the new schemes that share QK_K.
        QuantScheme::GgufTQ1_0 => rlx_gguf::tq_dequant::dequant_tq1_0_block(block, out),
        QuantScheme::GgufTQ2_0 => rlx_gguf::tq_dequant::dequant_tq2_0_block(block, out),
        // 128-element 1-bit block (PrismML Bonsai-27B); caller slices `out`.
        QuantScheme::GgufQ1_0 => rlx_gguf::q1_dequant::dequant_q1_0_block(
            block,
            (&mut out[..rlx_gguf::q1_dequant::QK1_0])
                .try_into()
                .unwrap(),
        ),
        QuantScheme::GgufQ2_0 => rlx_gguf::q2_dequant::dequant_q2_0_block(
            block,
            (&mut out[..rlx_gguf::q2_dequant::QK2_0])
                .try_into()
                .unwrap(),
        ),
        // 32-element blocks: caller slices `out` to the correct length.
        QuantScheme::GgufMXFP4 => rlx_gguf::mx_dequant::dequant_mxfp4_block(
            block,
            (&mut out[..rlx_gguf::mx_dequant::QK_MXFP4])
                .try_into()
                .unwrap(),
        ),
        QuantScheme::GgufNVFP4 => rlx_gguf::mx_dequant::dequant_nvfp4_block(
            block,
            (&mut out[..rlx_gguf::mx_dequant::QK_NVFP4])
                .try_into()
                .unwrap(),
        ),
        // IQ-family: no dedicated block-level helper, but the
        // whole-tensor dequant works fine on a single QK_K block —
        // pass the 256-elem slice straight through. ~Same cost as
        // dedicated block functions, just slightly less inlinable.
        QuantScheme::GgufIQ4XS => {
            let v = rlx_gguf::iq_dequant::dequant_iq4_xs(block, QK_K).expect("IQ4_XS block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ2XXS => {
            let v = rlx_gguf::iq_dequant::dequant_iq2_xxs(block, QK_K).expect("IQ2_XXS block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ2XS => {
            let v = rlx_gguf::iq_dequant::dequant_iq2_xs(block, QK_K).expect("IQ2_XS block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ2S => {
            let v = rlx_gguf::iq_dequant::dequant_iq2_s(block, QK_K).expect("IQ2_S block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ3XXS => {
            let v = rlx_gguf::iq_dequant::dequant_iq3_xxs(block, QK_K).expect("IQ3_XXS block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ3S => {
            let v = rlx_gguf::iq_dequant::dequant_iq3_s(block, QK_K).expect("IQ3_S block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ1S => {
            let v = rlx_gguf::iq_dequant::dequant_iq1_s(block, QK_K).expect("IQ1_S block");
            out.copy_from_slice(&v);
        }
        QuantScheme::GgufIQ1M => {
            let v = rlx_gguf::iq_dequant::dequant_iq1_m(block, QK_K).expect("IQ1_M block");
            out.copy_from_slice(&v);
        }
        // 32-element block schemes that go through dequant_block need
        // the caller to slice `out`; IQ4_NL is the only one not handled
        // above. Mirrors the Q4_0/Q8_0 idiom.
        QuantScheme::GgufIQ4NL => {
            rlx_gguf::iq_dequant::dequant_iq4_nl(block, rlx_gguf::iq_dequant::QK4_NL)
                .map(|v| out[..rlx_gguf::iq_dequant::QK4_NL].copy_from_slice(&v))
                .expect("IQ4_NL block")
        }
        other => panic!(
            "gguf_matmul: scheme {other:?} has no block-level dequant — use load-time dequant_cache"
        ),
    }
}

/// Fused dequant + `sgemm_bt` — `out` is zeroed then accumulated.
///
/// Block-fused reference kernel (no full-weight materialization). Opt in via
/// `RLX_GGUF_MATMUL_LEGACY=1`; default dispatch uses [`gguf_matmul_bt_dispatch`].
pub fn gguf_matmul_bt(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    gguf_matmul_bt_ex(
        x, w_bytes, out, m, k, n, scheme, /*allow_parallel=*/ true,
    );
}

/// Like [`gguf_matmul_bt`] but never spawns Rayon inside the `m==1` kernel.
///
/// Use this from an outer Rayon parallel region (e.g. MoE experts) so block-level
/// parallelism does not nest and oversubscribe the pool.
pub fn gguf_matmul_bt_serial(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    gguf_matmul_bt_ex(
        x, w_bytes, out, m, k, n, scheme, /*allow_parallel=*/ false,
    );
}

fn gguf_matmul_bt_ex(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    allow_parallel: bool,
) {
    assert_eq!(x.len(), m * k);
    assert_eq!(out.len(), m * n);
    out.fill(0.0);

    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let total_elems = k * n;
    debug_assert!(
        total_elems.is_multiple_of(block_elems),
        "k*n={total_elems} not aligned to GGUF block {block_elems}"
    );
    // Some backends (notably rlx-mlx, see lower.rs:1554) recover (m, k, n)
    // from inferred MLX shapes that can round one dim up by a single block
    // when an intermediate node carries padding; the caller's k*n then
    // implies one more Q4K block than the GGUF actually stored. Clamp to
    // the bytes we actually have rather than panic in release mode.
    let blocks_in_bytes = w_bytes.len() / block_bytes;
    let num_blocks_computed = total_elems / block_elems;
    if num_blocks_computed != blocks_in_bytes {
        debug_assert_eq!(
            w_bytes.len(),
            num_blocks_computed * block_bytes,
            "Q4K matmul: caller (k={k}, n={n}) implies {num_blocks_computed} blocks but w_bytes holds {blocks_in_bytes}"
        );
    }
    let num_blocks = num_blocks_computed.min(blocks_in_bytes);

    let mut block_f32 = [0f32; QK_K];

    if m == 1 {
        let x_row = x;
        if allow_parallel && num_blocks >= 32 && crate::pool::num_threads() > 1 {
            gguf_matmul_bt_m1_parallel(
                x_row,
                w_bytes,
                out,
                k,
                n,
                scheme,
                num_blocks,
                block_bytes,
                block_elems,
            );
        } else {
            gguf_matmul_bt_m1_sequential(
                x_row,
                w_bytes,
                out,
                scheme,
                num_blocks,
                block_bytes,
                block_elems,
                k,
            );
        }
        return;
    }

    for bi in 0..num_blocks {
        let off = bi * block_bytes;
        dequant_block(scheme, &w_bytes[off..off + block_bytes], &mut block_f32);
        let idx0 = bi * block_elems;
        for t in 0..block_elems {
            let idx = idx0 + t;
            let j = idx / k;
            let p = idx % k;
            let w_val = block_f32[t];
            for mi in 0..m {
                out[mi * n + j] += x[mi * k + p] * w_val;
            }
        }
    }
}

/// `true` when `RLX_GGUF_MATMUL_LEGACY=1` — force block-fused [`gguf_matmul_bt`].
#[inline]
pub fn gguf_matmul_use_legacy() -> bool {
    matches!(
        rlx_ir::env::var("RLX_GGUF_MATMUL_LEGACY").as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    )
}

/// Minimum `k*n` for cached dequant + BLAS (tiny tiles stay on fused path).
const CACHED_BLAS_MIN_WEIGHT_ELEMS: usize = 32 * 32;

#[inline]
fn prefer_cached_blas(k: usize, n: usize, m: usize) -> bool {
    !gguf_matmul_use_legacy() && (m > 1 || k.saturating_mul(n) >= CACHED_BLAS_MIN_WEIGHT_ELEMS)
}

/// Dequant once (cached by weight bytes) + Accelerate/OpenBLAS `sgemm_bt`.
///
/// `C[m,n] = A[m,k] @ B^T` with GGUF `B` stored `[n,k]` row-major. Mirrors the MLX
/// dequant-cache path; repeated decode matmuls on the same static param reuse f32 weights.
#[cfg(rlx_cpu_blas)]
pub fn gguf_matmul_bt_cached(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    w_off: usize,
) {
    assert_eq!(x.len(), m * k);
    assert_eq!(out.len(), m * n);
    let w_f32 = crate::dequant_cache::gguf_weight_f32(w_off, w_bytes, k, n, scheme);
    if m == 1 {
        out.fill(0.0);
        crate::blas::sgemv_nn(w_f32.as_ref(), x, out, n, k, 1.0, 0.0);
    } else {
        crate::blas::sgemm_bt(x, w_f32.as_ref(), out, m, k, n, 1.0);
    }
}

#[cfg(not(rlx_cpu_blas))]
pub fn gguf_matmul_bt_cached(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    _w_off: usize,
) {
    gguf_matmul_bt(x, w_bytes, out, m, k, n, scheme);
}

/// Default GGUF matmul entry: cached BLAS when available, else legacy fused blocks.
pub fn gguf_matmul_bt_dispatch(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    gguf_matmul_bt_dispatch_at(x, w_bytes, out, m, k, n, scheme, 0);
}

/// Like [`gguf_matmul_bt_dispatch`] but keys the dequant cache by arena `w_off`.
pub fn gguf_matmul_bt_dispatch_at(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    w_off: usize,
) {
    if prefer_cached_blas(k, n, m) {
        gguf_matmul_bt_cached(x, w_bytes, out, m, k, n, scheme, w_off);
    } else {
        gguf_matmul_bt(x, w_bytes, out, m, k, n, scheme);
    }
}

/// Fused GGUF dequant + grouped matmul for MoE expert stacks.
///
/// `w_bytes` holds `num_experts` contiguous packed slabs; expert `e` occupies
/// `[e * slab_bytes .. (e+1) * slab_bytes)` with the same GGML layout as a
/// standalone 2-D K-quant matrix of shape `[n, k]`.
pub fn gguf_grouped_matmul_bt(
    x: &[f32],
    w_bytes: &[u8],
    expert_idx: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme: QuantScheme,
) {
    gguf_grouped_matmul_bt_ex(
        x,
        w_bytes,
        expert_idx,
        out,
        m,
        k,
        n,
        num_experts,
        scheme,
        /*cache=*/ true,
    );
}

/// Like [`gguf_grouped_matmul_bt`] but never materializes expert slabs into the
/// process-wide F32 dequant cache (required for large MoE packs).
///
/// Unique expert groups run in parallel; each group uses the serial fused
/// kernel so Rayon is not nested. Tokens that share an expert are batched
/// (`m = count`) inside that group.
pub fn gguf_grouped_matmul_bt_fused(
    x: &[f32],
    w_bytes: &[u8],
    expert_idx: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme: QuantScheme,
) {
    gguf_grouped_matmul_bt_ex(
        x,
        w_bytes,
        expert_idx,
        out,
        m,
        k,
        n,
        num_experts,
        scheme,
        /*cache=*/ false,
    );
}

fn gguf_grouped_matmul_bt_ex(
    x: &[f32],
    w_bytes: &[u8],
    expert_idx: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme: QuantScheme,
    cache: bool,
) {
    assert_eq!(x.len(), m * k);
    assert_eq!(expert_idx.len(), m);
    assert_eq!(out.len(), m * n);

    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let slab_bytes = (k * n) / block_elems * block_bytes;
    assert_eq!(w_bytes.len(), num_experts * slab_bytes);

    let (packed_in, original_pos, offsets) =
        grouped_moe_sort_plan(x, expert_idx, m, k, num_experts);

    let jobs: Vec<(usize, usize, usize)> = (0..num_experts)
        .filter_map(|e| {
            let count = offsets[e + 1] - offsets[e];
            (count > 0).then_some((e, offsets[e], count))
        })
        .collect();

    let mut packed_out = vec![0f32; m * n];

    if cache {
        for &(e, start, count) in &jobs {
            let in_slice = &packed_in[start * k..(start + count) * k];
            let w_slice = &w_bytes[e * slab_bytes..(e + 1) * slab_bytes];
            let out_slice = &mut packed_out[start * n..(start + count) * n];
            gguf_matmul_bt_dispatch(in_slice, w_slice, out_slice, count, k, n, scheme);
        }
    } else if jobs.len() > 1 {
        // Parallelize over unique experts; serial fused GEMM per group so Rayon
        // is not nested. Batches tokens that share an expert (`count > 1`).
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            let chunks: Vec<(usize, Vec<f32>)> = jobs
                .par_iter()
                .map(|&(e, start, count)| {
                    let in_slice = &packed_in[start * k..(start + count) * k];
                    let w_slice = &w_bytes[e * slab_bytes..(e + 1) * slab_bytes];
                    let mut local = vec![0f32; count * n];
                    gguf_matmul_bt_serial(in_slice, w_slice, &mut local, count, k, n, scheme);
                    (start, local)
                })
                .collect();
            for (start, local) in chunks {
                let count = local.len() / n;
                packed_out[start * n..(start + count) * n].copy_from_slice(&local);
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            for &(e, start, count) in &jobs {
                let in_slice = &packed_in[start * k..(start + count) * k];
                let w_slice = &w_bytes[e * slab_bytes..(e + 1) * slab_bytes];
                let out_slice = &mut packed_out[start * n..(start + count) * n];
                gguf_matmul_bt_serial(in_slice, w_slice, out_slice, count, k, n, scheme);
            }
        }
    } else {
        // Single expert group: full-width parallel fused GEMM.
        for &(e, start, count) in &jobs {
            let in_slice = &packed_in[start * k..(start + count) * k];
            let w_slice = &w_bytes[e * slab_bytes..(e + 1) * slab_bytes];
            let out_slice = &mut packed_out[start * n..(start + count) * n];
            gguf_matmul_bt(in_slice, w_slice, out_slice, count, k, n, scheme);
        }
    }

    grouped_moe_unpermute_out(&packed_out, &original_pos, out, m, n);
}

/// Dequant an MoE expert stack `[E, K, N]` into GroupedMatMul layout (row-major
/// `[k, n]` slabs per expert). Used by `Op::DequantMoEWeights` and autodiff.
pub fn dequant_moe_weights_to_grouped_f32(
    packed: &[u8],
    out: &mut [f32],
    num_experts: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let slab_bytes = (k * n) / block_elems * block_bytes;
    assert_eq!(packed.len(), num_experts * slab_bytes);
    assert_eq!(out.len(), num_experts * k * n);
    for e in 0..num_experts {
        let slab = &packed[e * slab_bytes..(e + 1) * slab_bytes];
        let deq = match scheme {
            QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k(slab, k * n),
            QuantScheme::GgufQ5K => rlx_gguf::dequant_q5_k(slab, k * n),
            QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k(slab, k * n),
            QuantScheme::GgufQ8K => rlx_gguf::dequant_q8_k(slab, k * n),
            QuantScheme::GgufQ2K => rlx_gguf::dequant_q2_k(slab, k * n),
            QuantScheme::GgufQ3K => rlx_gguf::dequant_q3_k(slab, k * n),
            QuantScheme::GgufQ4_0 => rlx_gguf::dequant_q4_0(slab, k * n),
            QuantScheme::GgufQ4_1 => rlx_gguf::dequant_q4_1(slab, k * n),
            QuantScheme::GgufQ5_0 => rlx_gguf::dequant_q5_0(slab, k * n),
            QuantScheme::GgufQ5_1 => rlx_gguf::dequant_q5_1(slab, k * n),
            QuantScheme::GgufQ8_0 => rlx_gguf::dequant_q8_0(slab, k * n),
            QuantScheme::GgufIQ4NL => rlx_gguf::iq_dequant::dequant_iq4_nl(slab, k * n),
            QuantScheme::GgufIQ4XS => rlx_gguf::iq_dequant::dequant_iq4_xs(slab, k * n),
            QuantScheme::GgufIQ2XXS => rlx_gguf::iq_dequant::dequant_iq2_xxs(slab, k * n),
            QuantScheme::GgufIQ2XS => rlx_gguf::iq_dequant::dequant_iq2_xs(slab, k * n),
            QuantScheme::GgufIQ2S => rlx_gguf::iq_dequant::dequant_iq2_s(slab, k * n),
            QuantScheme::GgufIQ3XXS => rlx_gguf::iq_dequant::dequant_iq3_xxs(slab, k * n),
            QuantScheme::GgufIQ3S => rlx_gguf::iq_dequant::dequant_iq3_s(slab, k * n),
            QuantScheme::GgufIQ1S => rlx_gguf::iq_dequant::dequant_iq1_s(slab, k * n),
            QuantScheme::GgufIQ1M => rlx_gguf::iq_dequant::dequant_iq1_m(slab, k * n),
            QuantScheme::GgufTQ1_0 => rlx_gguf::tq_dequant::dequant_tq1_0(slab, k * n),
            QuantScheme::GgufTQ2_0 => rlx_gguf::tq_dequant::dequant_tq2_0(slab, k * n),
            QuantScheme::GgufMXFP4 => rlx_gguf::mx_dequant::dequant_mxfp4(slab, k * n),
            QuantScheme::GgufNVFP4 => rlx_gguf::mx_dequant::dequant_nvfp4(slab, k * n),
            QuantScheme::GgufQ1_0 => rlx_gguf::q1_dequant::dequant_q1_0(slab, k * n),
            QuantScheme::GgufQ2_0 => rlx_gguf::q2_dequant::dequant_q2_0(slab, k * n),
            other => panic!("dequant_moe_weights: unsupported scheme {other:?}"),
        }
        .expect("dequant_moe_weights: slab dequant failed");
        let base = e * k * n;
        for i in 0..k {
            for j in 0..n {
                out[base + i * n + j] = deq[j * k + i];
            }
        }
    }
}

/// Counting-sort tokens by expert (shared by host and GPU prep paths).
pub fn grouped_moe_sort_plan(
    x: &[f32],
    expert_idx: &[f32],
    m: usize,
    k: usize,
    num_experts: usize,
) -> (Vec<f32>, Vec<usize>, Vec<usize>) {
    let mut counts = vec![0usize; num_experts];
    for i in 0..m {
        let e = expert_idx[i] as usize;
        debug_assert!(e < num_experts);
        counts[e] += 1;
    }
    let mut offsets = vec![0usize; num_experts + 1];
    for e in 0..num_experts {
        offsets[e + 1] = offsets[e] + counts[e];
    }
    let mut packed_in = vec![0f32; m * k];
    let mut original_pos = vec![0usize; m];
    let mut write_idx = vec![0usize; num_experts];
    for i in 0..m {
        let e = expert_idx[i] as usize;
        let dst_row = offsets[e] + write_idx[e];
        packed_in[dst_row * k..(dst_row + 1) * k].copy_from_slice(&x[i * k..(i + 1) * k]);
        original_pos[dst_row] = i;
        write_idx[e] += 1;
    }
    (packed_in, original_pos, offsets)
}

pub fn grouped_moe_unpermute_out(
    packed_out: &[f32],
    original_pos: &[usize],
    out: &mut [f32],
    m: usize,
    n: usize,
) {
    for packed_idx in 0..m {
        let i = original_pos[packed_idx];
        out[i * n..(i + 1) * n].copy_from_slice(&packed_out[packed_idx * n..(packed_idx + 1) * n]);
    }
}

/// Parallel fused matmul — delegates to [`gguf_matmul_bt_dispatch`].
pub fn gguf_matmul_bt_parallel(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    gguf_matmul_bt_dispatch(x, w_bytes, out, m, k, n, scheme);
}

/// Decode GEMV (`m == 1`): single-threaded block fold.
fn gguf_matmul_bt_m1_sequential(
    x_row: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    scheme: QuantScheme,
    num_blocks: usize,
    block_bytes: usize,
    block_elems: usize,
    k: usize,
) {
    if scheme == QuantScheme::GgufQ4K && k.is_multiple_of(QK_K) && block_elems == QK_K {
        q4k_gemv_rowmajor(x_row, w_bytes, out, k);
        return;
    }
    let mut block_f32 = [0f32; QK_K];
    for bi in 0..num_blocks {
        let off = bi * block_bytes;
        dequant_block(scheme, &w_bytes[off..off + block_bytes], &mut block_f32);
        let idx0 = bi * block_elems;
        for t in 0..block_elems {
            let idx = idx0 + t;
            let j = idx / k;
            let p = idx % k;
            out[j] += x_row[p] * block_f32[t];
        }
    }
}

/// Decode GEMV (`m == 1`): fold/reduce over GGUF super-blocks across Rayon workers.
fn gguf_matmul_bt_m1_parallel(
    x_row: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    k: usize,
    n: usize,
    scheme: QuantScheme,
    num_blocks: usize,
    block_bytes: usize,
    block_elems: usize,
) {
    if scheme == QuantScheme::GgufQ4K && k.is_multiple_of(QK_K) && block_elems == QK_K {
        // Independent output rows — no n-wide partial reduction.
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            let blocks_per_row = k / QK_K;
            let row_bytes = blocks_per_row * rlx_gguf::Q4K_BLOCK_BYTES;
            debug_assert_eq!(n * row_bytes, w_bytes.len().min(n * row_bytes));
            let mut x_q8 = vec![0u8; blocks_per_row * rlx_gguf::Q8K_BLOCK_BYTES];
            rlx_gguf::quantize_q8_k_row(&x_row[..k], &mut x_q8);
            out.par_iter_mut().enumerate().for_each(|(j, slot)| {
                let row = &w_bytes[j * row_bytes..(j + 1) * row_bytes];
                *slot = q4k_dot_row_q8(row, &x_q8, blocks_per_row);
            });
            return;
        }
        #[cfg(target_arch = "wasm32")]
        {
            q4k_gemv_rowmajor(x_row, w_bytes, out, k);
            return;
        }
    }
    // wasm: single-threaded serial accumulate (no Rayon thread pool).
    #[cfg(target_arch = "wasm32")]
    {
        let _ = n;
        for v in out.iter_mut() {
            *v = 0.0;
        }
        let mut block_f32 = [0f32; QK_K];
        for bi in 0..num_blocks {
            let off = bi * block_bytes;
            dequant_block(scheme, &w_bytes[off..off + block_bytes], &mut block_f32);
            let idx0 = bi * block_elems;
            for t in 0..block_elems {
                let idx = idx0 + t;
                let j = idx / k;
                let p = idx % k;
                out[j] += x_row[p] * block_f32[t];
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;

        let partial = (0..num_blocks)
            .into_par_iter()
            .fold(
                || vec![0f32; n],
                |mut local, bi| {
                    let off = bi * block_bytes;
                    let mut block_f32 = [0f32; QK_K];
                    dequant_block(scheme, &w_bytes[off..off + block_bytes], &mut block_f32);
                    let idx0 = bi * block_elems;
                    for t in 0..block_elems {
                        let idx = idx0 + t;
                        let j = idx / k;
                        let p = idx % k;
                        local[j] += x_row[p] * block_f32[t];
                    }
                    local
                },
            )
            .reduce(
                || vec![0f32; n],
                |mut acc, chunk| {
                    for (a, b) in acc.iter_mut().zip(chunk) {
                        *a += b;
                    }
                    acc
                },
            );
        out.copy_from_slice(&partial);
    }
}

/// `C[n] = x[k] @ W[n,k]^T` for Q4_K with `k` a multiple of [`QK_K`].
///
/// Activations are quantized once to Q8_K; each output row then uses the
/// int8 Q4_K×Q8_K vec-dot (llama.cpp-style). This is the hot path for MoE
/// expert GEMVs.
fn q4k_gemv_rowmajor(x: &[f32], w_bytes: &[u8], out: &mut [f32], k: usize) {
    let n = out.len();
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * rlx_gguf::Q4K_BLOCK_BYTES;
    debug_assert!(x.len() >= k);
    debug_assert!(w_bytes.len() >= n * row_bytes);

    let mut x_q8 = vec![0u8; blocks_per_row * rlx_gguf::Q8K_BLOCK_BYTES];
    rlx_gguf::quantize_q8_k_row(&x[..k], &mut x_q8);

    for j in 0..n {
        let row = &w_bytes[j * row_bytes..(j + 1) * row_bytes];
        out[j] = q4k_dot_row_q8(row, &x_q8, blocks_per_row);
    }
}

/// Like [`q4k_gemv_rowmajor`] but `x` is already packed Q8_K (for parallel rows).
#[allow(dead_code)] // kept for callers that pre-quantize activations
fn q4k_gemv_rowmajor_q8(x_q8: &[u8], w_bytes: &[u8], out: &mut [f32], k: usize) {
    let n = out.len();
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * rlx_gguf::Q4K_BLOCK_BYTES;
    debug_assert_eq!(x_q8.len(), blocks_per_row * rlx_gguf::Q8K_BLOCK_BYTES);
    for j in 0..n {
        let row = &w_bytes[j * row_bytes..(j + 1) * row_bytes];
        out[j] = q4k_dot_row_q8(row, x_q8, blocks_per_row);
    }
}

#[inline]
fn q4k_dot_row_q8(row_bytes: &[u8], x_q8: &[u8], blocks_per_row: usize) -> f32 {
    let bb = rlx_gguf::Q4K_BLOCK_BYTES;
    let qb = rlx_gguf::Q8K_BLOCK_BYTES;
    let mut acc = 0.0f32;
    for b in 0..blocks_per_row {
        let q4 = &row_bytes[b * bb..(b + 1) * bb];
        let q8 = &x_q8[b * qb..(b + 1) * qb];
        acc += q4k_dot_q8_block(q4, q8);
    }
    acc
}

#[inline]
fn q4k_dot_q8_block(q4: &[u8], q8: &[u8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        q4k_dot_q8_block_neon(q4, q8)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        rlx_gguf::q4_k_dot_q8_k(q4, q8)
    }
}

/// NEON-friendly Q4_K × Q8_K block dot (aarch64).
///
/// Uses int16/int32 accumulators (no `dotprod` requirement). Falls back to the
/// scalar reference for the scale/min bookkeeping structure.
#[cfg(target_arch = "aarch64")]
#[inline]
fn q4k_dot_q8_block_neon(q4: &[u8], q8: &[u8]) -> f32 {
    // SAFETY: full Q4_K / Q8_K slabs; NEON baseline on our aarch64 targets.
    unsafe {
        let d = half::f16::from_le_bytes([q4[0], q4[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([q4[2], q4[3]]).to_f32();
        let scales = &q4[4..16];
        let qs = q4.as_ptr().add(16);
        let yd = f32::from_le_bytes([q8[0], q8[1], q8[2], q8[3]]);
        let q8s = q8.as_ptr().add(4) as *const i8;
        let bsums = q8.as_ptr().add(4 + QK_K);

        let mut sumi_min = 0i32;
        for j in 0..16 {
            let (_, m) = q4k_scale_min(j / 2, scales);
            let bs = i16::from_le_bytes([*bsums.add(j * 2), *bsums.add(j * 2 + 1)]) as i32;
            sumi_min += m as i32 * bs;
        }

        let mut sumi = 0i32;
        let mut is = 0usize;
        let mut yi = 0usize;
        for j in (0..8).step_by(2) {
            let (sc0, _) = q4k_scale_min(j, scales);
            let (sc1, _) = q4k_scale_min(j + 1, scales);
            sumi += sc0 as i32 * q4k_group32_q8_dot_neon(qs.add(is), q8s.add(yi), false);
            yi += 32;
            sumi += sc1 as i32 * q4k_group32_q8_dot_neon(qs.add(is), q8s.add(yi), true);
            yi += 32;
            is += 32;
        }
        d * yd * sumi as f32 - dmin * yd * sumi_min as f32
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn q4k_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Dot 32 Q4 nibbles (lo or hi) with 32 Q8 activations → i32 sum of products.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn q4k_group32_q8_dot_neon(qs: *const u8, q8: *const i8, high_nibble: bool) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = vdupq_n_s32(0);
        for i in (0..32).step_by(16) {
            let qbytes = vld1q_u8(qs.add(i));
            let nibble = if high_nibble {
                vshrq_n_u8(qbytes, 4)
            } else {
                vandq_u8(qbytes, vdupq_n_u8(0x0F))
            };
            // u8 nibbles 0..15 → i8 (still non-negative)
            let q4 = vreinterpretq_s8_u8(nibble);
            let y = vld1q_s8(q8.add(i));
            // 8×8 → 16 in low/high halves, then widen-add into i32.
            let lo = vmull_s8(vget_low_s8(q4), vget_low_s8(y));
            let hi = vmull_s8(vget_high_s8(q4), vget_high_s8(y));
            acc = vpadalq_s16(acc, lo);
            acc = vpadalq_s16(acc, hi);
        }
        vaddvq_s32(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_blas_matches_fused_q4k_decode() {
        use crate::dequant_cache::clear_dequant_cache;
        clear_dequant_cache();
        let k = 256;
        let n = 64;
        let m = 1;
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.001).sin()).collect();
        let packed = rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q4K).unwrap();
        let x: Vec<f32> = (0..k).map(|i| 0.02 * i as f32).collect();
        let mut legacy = vec![0f32; m * n];
        let mut cached = vec![0f32; m * n];
        gguf_matmul_bt(&x, &packed, &mut legacy, m, k, n, QuantScheme::GgufQ4K);
        gguf_matmul_bt_cached(&x, &packed, &mut cached, m, k, n, QuantScheme::GgufQ4K, 0);
        for i in 0..legacy.len() {
            // Q4_K fused decode quantizes activations to Q8_K (llama.cpp-style);
            // allow small relative error vs full-f32 cached BLAS.
            let scale = cached[i].abs().max(1.0);
            assert!(
                (legacy[i] - cached[i]).abs() < 1e-2 * scale,
                "i={i}: legacy={} cached={}",
                legacy[i],
                cached[i]
            );
        }
    }

    #[test]
    fn q4k_rowmajor_serial_matches_generic_laguna_shape() {
        // Laguna-XS expert gate/up: n=512, k=2048 (both Q4_K aligned).
        let k = 2048;
        let n = 512;
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let packed = rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q4K).unwrap();
        let x: Vec<f32> = (0..k).map(|i| 0.001 * i as f32 - 0.5).collect();
        let mut fast = vec![0f32; n];
        let mut slow = vec![0f32; n];
        gguf_matmul_bt_serial(&x, &packed, &mut fast, 1, k, n, QuantScheme::GgufQ4K);
        // Reference: same Q8_K activations + scalar Q4×Q8 block dots.
        let bpr = k / QK_K;
        let mut x_q8 = vec![0u8; bpr * rlx_gguf::Q8K_BLOCK_BYTES];
        rlx_gguf::quantize_q8_k_row(&x, &mut x_q8);
        let bb = rlx_gguf::Q4K_BLOCK_BYTES;
        let qb = rlx_gguf::Q8K_BLOCK_BYTES;
        for j in 0..n {
            let mut acc = 0.0f32;
            for b in 0..bpr {
                let q4 = &packed[(j * bpr + b) * bb..(j * bpr + b + 1) * bb];
                let q8 = &x_q8[b * qb..(b + 1) * qb];
                acc += rlx_gguf::q4_k_dot_q8_k(q4, q8);
            }
            slow[j] = acc;
        }
        let mut max_err = 0.0f32;
        for i in 0..n {
            max_err = max_err.max((fast[i] - slow[i]).abs());
        }
        assert!(max_err < 1e-3, "max_err={max_err}");
    }

    #[test]
    #[ignore = "manual kernel timing"]
    fn q4k_gemv_microbench() {
        let k = 2048usize;
        let n = 512usize;
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let packed = rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q4K).unwrap();
        let x: Vec<f32> = (0..k).map(|i| 0.001 * i as f32 - 0.5).collect();
        let mut out = vec![0f32; n];
        for _ in 0..5 {
            gguf_matmul_bt_serial(&x, &packed, &mut out, 1, k, n, QuantScheme::GgufQ4K);
        }
        let iters = 40;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            gguf_matmul_bt_serial(&x, &packed, &mut out, 1, k, n, QuantScheme::GgufQ4K);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            gguf_matmul_bt(&x, &packed, &mut out, 1, k, n, QuantScheme::GgufQ4K);
        }
        let ms_par = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        eprintln!("q4k gemv n={n} k={k}: serial={ms:.3}ms  parallel={ms_par:.3}ms");
        assert!(ms < 50.0, "serial unexpectedly slow: {ms}ms");
    }

    #[test]
    fn dispatch_matches_legacy_q8k_prefill() {
        use crate::dequant_cache::clear_dequant_cache;
        clear_dequant_cache();
        let k = 256;
        let n = 4;
        let m = 2;
        let scale = 0.5f32;
        let mut packed = Vec::new();
        for _ in 0..n {
            packed.extend_from_slice(&scale.to_le_bytes());
            for i in 0..QK_K {
                let q = (i as i32 - 128).clamp(-128, 127) as i8;
                packed.push(q as u8);
            }
            for _ in 0..(QK_K / 16) {
                packed.extend_from_slice(&0i16.to_le_bytes());
            }
        }
        let x: Vec<f32> = (0..m * k).map(|i| 0.01 * i as f32).collect();
        let mut legacy = vec![0f32; m * n];
        let mut dispatched = vec![0f32; m * n];
        gguf_matmul_bt(&x, &packed, &mut legacy, m, k, n, QuantScheme::GgufQ8K);
        gguf_matmul_bt_dispatch(&x, &packed, &mut dispatched, m, k, n, QuantScheme::GgufQ8K);
        for i in 0..legacy.len() {
            assert!(
                (legacy[i] - dispatched[i]).abs() < 0.05,
                "i={i}: {} vs {}",
                legacy[i],
                dispatched[i]
            );
        }
    }

    #[test]
    fn fused_q1_0_matches_full_dequant() {
        // Custom 1-bit Q1_0 (PrismML Bonsai-27B): 128-element blocks,
        // f16 group scale + 128 sign bits. Exercises both the m=1 and
        // the m>1 accumulation paths of gguf_matmul_bt.
        use rlx_gguf::q1_dequant::QK1_0;
        let k = 256usize; // 2 blocks per row
        let n = 8usize;
        let blocks_per_row = k / QK1_0;

        let mut packed = Vec::new();
        let mut w_ref = vec![0f32; n * k]; // row-major [n, k]
        for row in 0..n {
            for b in 0..blocks_per_row {
                // f16-roundtrip the scale so w_ref == packed-dequant exactly.
                let d =
                    half::f16::from_f32(0.1 + 0.05 * (row * blocks_per_row + b) as f32).to_f32();
                packed.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                let mut bits = [0u8; QK1_0 / 8];
                for j in 0..QK1_0 {
                    let elem = b * QK1_0 + j;
                    let positive = !(row + elem).is_multiple_of(3);
                    if positive {
                        bits[j / 8] |= 1 << (j % 8);
                    }
                    w_ref[row * k + elem] = if positive { d } else { -d };
                }
                packed.extend_from_slice(&bits);
            }
        }
        // Whole-tensor dequant agrees with the hand-built reference.
        let deq = rlx_gguf::q1_dequant::dequant_q1_0(&packed, n * k).unwrap();
        assert_eq!(deq, w_ref);

        for m in [1usize, 3] {
            let x: Vec<f32> = (0..m * k).map(|i| 0.01 * i as f32 - 0.3).collect();
            let mut reference = vec![0f32; m * n];
            for mi in 0..m {
                for row in 0..n {
                    let mut acc = 0f32;
                    for p in 0..k {
                        acc += x[mi * k + p] * w_ref[row * k + p];
                    }
                    reference[mi * n + row] = acc;
                }
            }
            let mut fused = vec![0f32; m * n];
            gguf_matmul_bt(&x, &packed, &mut fused, m, k, n, QuantScheme::GgufQ1_0);
            for i in 0..reference.len() {
                assert!(
                    (reference[i] - fused[i]).abs() < 1e-3,
                    "m={m} i={i}: ref={} fused={}",
                    reference[i],
                    fused[i]
                );
            }
        }
    }

    #[test]
    fn fused_q8k_matches_full_dequant() {
        let k = 256;
        let n = 4;
        let m = 2;
        let scale = 0.5f32;
        let mut packed = Vec::new();
        for _ in 0..n {
            packed.extend_from_slice(&scale.to_le_bytes());
            for i in 0..QK_K {
                let q = (i as i32 - 128).clamp(-128, 127) as i8;
                packed.push(q as u8);
            }
            for _ in 0..(QK_K / 16) {
                packed.extend_from_slice(&0i16.to_le_bytes());
            }
        }
        let w_ref = rlx_gguf::dequant_q8_k(&packed, k * n).unwrap();
        let x: Vec<f32> = (0..m * k).map(|i| 0.01 * i as f32).collect();
        let mut fused = vec![0f32; m * n];
        gguf_matmul_bt(&x, &packed, &mut fused, m, k, n, QuantScheme::GgufQ8K);
        let mut expected = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += x[r * k + i] * w_ref[c * k + i];
                }
                expected[r * n + c] = acc;
            }
        }
        for i in 0..fused.len() {
            assert!(
                (fused[i] - expected[i]).abs() < 1e-4,
                "i={i}: {} vs {}",
                fused[i],
                expected[i]
            );
        }
    }

    #[test]
    fn parallel_m1_matches_sequential_q8k() {
        let k = 512;
        let n = 128;
        let scale = 0.5f32;
        let mut packed = Vec::new();
        for _ in 0..n {
            for _ in 0..(k / QK_K) {
                packed.extend_from_slice(&scale.to_le_bytes());
                for i in 0..QK_K {
                    let q = (i as i32 - 128).clamp(-128, 127) as i8;
                    packed.push(q as u8);
                }
                for _ in 0..(QK_K / 16) {
                    packed.extend_from_slice(&0i16.to_le_bytes());
                }
            }
        }
        let x: Vec<f32> = (0..k).map(|i| 0.01 * i as f32).collect();
        let scheme = QuantScheme::GgufQ8K;
        let block_elems = scheme.gguf_block_size() as usize;
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let num_blocks = (k * n / block_elems).min(packed.len() / block_bytes);
        let mut seq = vec![0f32; n];
        let mut par = vec![0f32; n];
        gguf_matmul_bt_m1_sequential(
            &x,
            &packed,
            &mut seq,
            scheme,
            num_blocks,
            block_bytes,
            block_elems,
            k,
        );
        if num_blocks >= 32 && crate::pool::num_threads() > 1 {
            gguf_matmul_bt_m1_parallel(
                &x,
                &packed,
                &mut par,
                k,
                n,
                scheme,
                num_blocks,
                block_bytes,
                block_elems,
            );
        } else {
            par.copy_from_slice(&seq);
        }
        for i in 0..n {
            let tol = seq[i].abs().max(1.0) * 1e-5 + 1e-3;
            assert!(
                (seq[i] - par[i]).abs() <= tol,
                "parallel mismatch at {i}: {} vs {} (tol {tol})",
                seq[i],
                par[i]
            );
        }
    }

    #[test]
    fn grouped_q8k_matches_per_expert_reference() {
        let k = 256;
        let n = 4;
        let m = 3;
        let num_experts = 2;
        let scale = 0.5f32;
        let mut packed = Vec::new();
        for _ in 0..(num_experts * n) {
            packed.extend_from_slice(&scale.to_le_bytes());
            for i in 0..QK_K {
                let q = (i as i32 - 128).clamp(-128, 127) as i8;
                packed.push(q as u8);
            }
            for _ in 0..(QK_K / 16) {
                packed.extend_from_slice(&0i16.to_le_bytes());
            }
        }
        let x: Vec<f32> = (0..m * k).map(|i| 0.01 * i as f32).collect();
        let expert_idx = vec![0f32, 1.0, 0.0];
        let mut grouped = vec![0f32; m * n];
        gguf_grouped_matmul_bt(
            &x,
            &packed,
            &expert_idx,
            &mut grouped,
            m,
            k,
            n,
            num_experts,
            QuantScheme::GgufQ8K,
        );
        let slab = (k * n) / QK_K * QuantScheme::GgufQ8K.gguf_block_bytes() as usize;
        let mut expected = vec![0f32; m * n];
        for row in 0..m {
            let e = expert_idx[row] as usize;
            let w_ref = rlx_gguf::dequant_q8_k(&packed[e * slab..(e + 1) * slab], k * n).unwrap();
            for col in 0..n {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += x[row * k + i] * w_ref[col * k + i];
                }
                expected[row * n + col] = acc;
            }
        }
        for i in 0..grouped.len() {
            // f32 GEMM (OpenBLAS/Accelerate) block-accumulates in a different
            // order than the naive f32 reference loop, so the absolute error
            // scales with the output magnitude — in the thousands for these
            // inputs. Compare with a relative tolerance; the fixed 1e-2 was
            // too tight and only passed on backends whose accumulation order
            // happened to stay under it (SIMD, Accelerate — not OpenBLAS).
            let tol = 1e-2 + 1e-4 * expected[i].abs();
            assert!(
                (grouped[i] - expected[i]).abs() <= tol,
                "i={i}: {} vs {} (tol {tol})",
                grouped[i],
                expected[i]
            );
        }
    }
}
