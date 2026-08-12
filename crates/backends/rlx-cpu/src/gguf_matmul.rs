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
use rlx_gguf::q2_dequant::{Q2_0_BLOCK_BYTES, Q8_0_G128_BYTES, QK2_0};
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
        // Fermion five-value ternary (Neutrino): 256-element blocks == QK_K,
        // so `out` maps 1:1. FV5 = transformer linears, FV5B = int8 embed/lm_head.
        QuantScheme::GgufFV5 => rlx_gguf::fv5_dequant::dequant_fv5_block(block, out),
        QuantScheme::GgufFV5B => rlx_gguf::fv5_dequant::dequant_fv5b_block(block, out),
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

    #[cfg(not(target_arch = "wasm32"))]
    if allow_parallel
        && k.is_multiple_of(block_elems)
        && block_elems <= QK_K
        && crate::pool::num_threads() > 1
    {
        gguf_matmul_bt_rows_generic_gemm(
            x,
            w_bytes,
            out,
            m,
            k,
            n,
            scheme,
            block_bytes,
            block_elems,
        );
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

/// Crossover `n` above which the parallel int8 Q4_K decode GEMV beats
/// cached-f32-BLAS (Accelerate/AMX). Measured on Apple: BLAS ahead at n≈1k,
/// tie at n≈3k, fused ~3× ahead at n≈128k → `n≥4096` captures the wide matmuls
/// (LM head, large FFN) for a ~15–25% CPU decode speedup.
///
/// **Opt-in** (default disabled): the fused path quantizes the *activation* to
/// Q8_K (llama.cpp-style), which is not bit-identical to the f32 cached-BLAS
/// path and flips occasional near-tie greedy tokens — rlx keeps decode on f32
/// for fidelity (and decode↔prefill parity) by default. Set
/// `RLX_Q4K_FUSED_MIN_N=4096` (or another crossover) to trade that for speed.
fn q4k_fused_decode_min_n() -> usize {
    rlx_ir::env::var("RLX_Q4K_FUSED_MIN_N")
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX)
}

#[inline]
fn prefer_cached_blas(k: usize, n: usize, m: usize) -> bool {
    if gguf_matmul_use_legacy() {
        return false;
    }
    // GEMV (decode) against a THRASHING dequant cache is the worst case: the
    // slab is evicted before it is reused, so every token re-dequantizes the
    // whole model AND churns allocations, while the fused int8 kernel reads the
    // packed bytes in place. Measured on Muse-Glimmer-30B UD-Q4_K_XL (f32 form
    // ~111 GB vs a 15 GB budget): 43.1 s/token cached vs 0.76 s fused — 57x.
    //
    // Only `m == 1` is diverted. For `m > 1` the cached path amortizes ONE
    // dequant across all rows and hands the work to BLAS, which still wins even
    // while thrashing (5-token prefill: 61 s cached vs 111 s fused). A model
    // whose f32 form fits the budget never evicts, so `cache_thrashing()` stays
    // false and this is inert.
    // Extended to m > 1 as well: with the row-parallel generic GEMM below, the
    // fused path no longer loses to BLAS once "cached" means re-dequantizing the
    // whole model on every call plus allocator churn. Both do the same total
    // dequant work; only the fused one skips the cache bookkeeping.
    if crate::dequant_cache::cache_thrashing() {
        return false;
    }
    m > 1 || k.saturating_mul(n) >= CACHED_BLAS_MIN_WEIGHT_ELEMS
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
    // Q2_0 decode GEMV: int8 VNNI/NEON dot (llama.cpp-style) beats
    // dequant→f32→BLAS and stays on the int path even for large `k*n`, so it
    // bypasses `prefer_cached_blas`. Default-on, runtime-detected.
    if m == 1 && scheme == QuantScheme::GgufQ2_0 && k.is_multiple_of(QK2_0) {
        q2_0_gemv_parallel(x, w_bytes, out, k);
        return;
    }
    // Large Q4_K decode GEMV: the parallel int8 (Q8_K-dot) kernel beats
    // cached-f32-BLAS (single-thread AMX) once `n` is big — measured ~3× at
    // n≈128k, ~tie at n≈3k, BLAS ahead below. So route wide matmuls (LM head,
    // large FFN gate/up) to the fused path; narrow ones stay on AMX. Tune the
    // crossover with `RLX_Q4K_FUSED_MIN_N` (set huge to disable).
    if m == 1
        && scheme == QuantScheme::GgufQ4K
        && k.is_multiple_of(QK_K)
        && n >= q4k_fused_decode_min_n()
        && crate::pool::num_threads() > 1
    {
        gguf_matmul_bt(x, w_bytes, out, m, k, n, scheme);
        return;
    }
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

// ---------------------------------------------------------------------------
// Per-precision GEMV / GEMM kernels, generated.
//
// The dynamic form calls `dequant_block(scheme, ..)` once per 256-element block,
// which is a runtime `match` on the scheme in the innermost loop: the compiler
// cannot inline the actual dequant, so it cannot fuse or vectorize across the
// dequant→dot boundary. Generating one monomorphic kernel per precision makes
// the dequant a direct call, and lets the dot use four independent accumulators
// (a single `acc` serializes on FP latency, ~4 cycles per element).
//
// Adding a precision is one line in `gguf_qk_kernels!`. Every scheme here uses
// the 256-element super-block layout and the `(&[u8], &mut [f32; QK_K])` block
// signature, so the macro body is genuinely shared rather than copy-pasted.
// ---------------------------------------------------------------------------

/// 256-element dot with 4 independent accumulators.
///
/// Both callers are the rayon-parallel `gguf_qk_kernels!` GEMV/GEMM, which are
/// themselves `cfg(not(wasm32))` — so on wasm this is dead code. Gate it the
/// same way rather than let it warn there.
#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn dot256(xs: &[f32], blk: &[f32; QK_K]) -> f32 {
    let (mut a0, mut a1, mut a2, mut a3) = (0f32, 0f32, 0f32, 0f32);
    let mut t = 0;
    while t < QK_K {
        a0 += xs[t] * blk[t];
        a1 += xs[t + 1] * blk[t + 1];
        a2 += xs[t + 2] * blk[t + 2];
        a3 += xs[t + 3] * blk[t + 3];
        t += 4;
    }
    (a0 + a1) + (a2 + a3)
}

macro_rules! gguf_qk_kernels {
    ($( $scheme:path => ($gemv:ident, $gemm:ident, $dq:path) ),+ $(,)?) => {
        $(
            /// Row-parallel GEMV (`m == 1`) specialized to one precision.
            #[cfg(not(target_arch = "wasm32"))]
            fn $gemv(x_row: &[f32], w_bytes: &[u8], out: &mut [f32], k: usize, block_bytes: usize) {
                use rayon::prelude::*;
                let blocks_per_row = k / QK_K;
                let row_bytes = blocks_per_row * block_bytes;
                out.par_iter_mut().enumerate().for_each(|(j, slot)| {
                    let start = j * row_bytes;
                    let Some(row) = w_bytes.get(start..start + row_bytes) else {
                        *slot = 0.0;
                        return;
                    };
                    let mut blk = [0f32; QK_K];
                    let mut acc = 0f32;
                    for b in 0..blocks_per_row {
                        $dq(&row[b * block_bytes..(b + 1) * block_bytes], &mut blk);
                        acc += dot256(&x_row[b * QK_K..(b + 1) * QK_K], &blk);
                    }
                    *slot = acc;
                });
            }

            /// Row-parallel GEMM (`m > 1`) specialized to one precision. Each block
            /// is dequantized ONCE and reused across all `m` rows.
            #[cfg(not(target_arch = "wasm32"))]
            #[allow(clippy::too_many_arguments)]
            fn $gemm(
                x: &[f32], w_bytes: &[u8], out: &mut [f32],
                m: usize, k: usize, n: usize, block_bytes: usize,
            ) {
                use rayon::prelude::*;
                let blocks_per_row = k / QK_K;
                let row_bytes = blocks_per_row * block_bytes;
                let cols: Vec<Vec<f32>> = (0..n).into_par_iter().map(|j| {
                    let mut acc = vec![0f32; m];
                    let start = j * row_bytes;
                    let Some(row) = w_bytes.get(start..start + row_bytes) else { return acc };
                    let mut blk = [0f32; QK_K];
                    for b in 0..blocks_per_row {
                        $dq(&row[b * block_bytes..(b + 1) * block_bytes], &mut blk);
                        let off = b * QK_K;
                        for (mi, a) in acc.iter_mut().enumerate() {
                            *a += dot256(&x[mi * k + off..mi * k + off + QK_K], &blk);
                        }
                    }
                    acc
                }).collect();
                for (j, acc) in cols.iter().enumerate() {
                    for (mi, a) in acc.iter().enumerate() {
                        out[mi * n + j] = *a;
                    }
                }
            }
        )+

        /// Route to a specialized GEMV when one exists. `false` ⇒ caller falls back.
        #[cfg(not(target_arch = "wasm32"))]
        fn gguf_gemv_specialized(
            scheme: QuantScheme, x_row: &[f32], w_bytes: &[u8], out: &mut [f32],
            k: usize, block_bytes: usize,
        ) -> bool {
            match scheme {
                $( $scheme => { $gemv(x_row, w_bytes, out, k, block_bytes); true } )+
                _ => false,
            }
        }

        /// Route to a specialized GEMM when one exists. `false` ⇒ caller falls back.
        #[cfg(not(target_arch = "wasm32"))]
        #[allow(clippy::too_many_arguments)]
        fn gguf_gemm_specialized(
            scheme: QuantScheme, x: &[f32], w_bytes: &[u8], out: &mut [f32],
            m: usize, k: usize, n: usize, block_bytes: usize,
        ) -> bool {
            match scheme {
                $( $scheme => { $gemm(x, w_bytes, out, m, k, n, block_bytes); true } )+
                _ => false,
            }
        }
    };
}

// NOTE: `GgufQ8K` is deliberately absent — it is the ACTIVATION quantization
// format, never a weight dtype in these checkpoints, so a weight-GEMV
// specialization for it would add reassociation risk for no gain.
gguf_qk_kernels! {
    QuantScheme::GgufQ2K => (gemv_q2k, gemm_q2k, rlx_gguf::dequant_q2_k_block),
    QuantScheme::GgufQ3K => (gemv_q3k, gemm_q3k, rlx_gguf::dequant_q3_k_block),
    QuantScheme::GgufQ4K => (gemv_q4k, gemm_q4k, rlx_gguf::dequant_q4_k_block),
    QuantScheme::GgufQ5K => (gemv_q5k, gemm_q5k, rlx_gguf::dequant_q5_k_block),
    QuantScheme::GgufQ6K => (gemv_q6k, gemm_q6k, rlx_gguf::dequant_q6_k_block),
}

/// Generic decode GEMV (`m == 1`) for ANY GGUF scheme with a row-aligned `k`.
///
/// The old generic path was doubly pathological: it walked `k*n` elements
/// SERIALLY and recovered the output row / input position with `idx / k` and
/// `idx % k` — two integer divisions PER ELEMENT. Only `Q4_K` (and `Q2_0`) had a
/// fast row-dot, so every other scheme took it, including `Q5_K`, which is what
/// unsloth ships the Muse-Glimmer LM head as: `[6656, 202048]` = 1.345B params
/// on a serial scalar loop.
///
/// When `k` is a whole number of blocks, block `b` of row `j` covers input
/// positions `[b*block_elems, (b+1)*block_elems)` — the row index is structural,
/// so no division is needed at all. Rows are independent, so this parallelizes
/// over `n` exactly like the Q4_K path, and the inner loop is a contiguous
/// f32 dot the autovectorizer can handle.
#[cfg(not(target_arch = "wasm32"))]
fn gguf_matmul_bt_m1_rows_generic(
    x_row: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    k: usize,
    scheme: QuantScheme,
    block_bytes: usize,
    block_elems: usize,
) {
    use rayon::prelude::*;
    if block_elems == QK_K && gguf_gemv_specialized(scheme, x_row, w_bytes, out, k, block_bytes) {
        return;
    }
    let blocks_per_row = k / block_elems;
    let row_bytes = blocks_per_row * block_bytes;
    out.par_iter_mut().enumerate().for_each(|(j, slot)| {
        let start = j * row_bytes;
        let Some(row) = w_bytes.get(start..start + row_bytes) else {
            *slot = 0.0;
            return;
        };
        let mut blk = [0f32; QK_K];
        let mut acc = 0f32;
        for b in 0..blocks_per_row {
            dequant_block(
                scheme,
                &row[b * block_bytes..(b + 1) * block_bytes],
                &mut blk,
            );
            let xs = &x_row[b * block_elems..(b + 1) * block_elems];
            for t in 0..block_elems {
                acc += xs[t] * blk[t];
            }
        }
        *slot = acc;
    });
}

/// Generic GEMM (`m > 1`) for any GGUF scheme with a row-aligned `k`.
///
/// The stock `m > 1` fallback walks `k*n` elements serially and recovers indices
/// with `idx / k` + `idx % k` — two integer divisions per element — which is why
/// fully-fused prefill measured 111 s against 61 s for dequant+BLAS. This does
/// the same arithmetic without either problem: parallel over output columns,
/// each block dequantized ONCE and reused across all `m` rows (so total dequant
/// work matches the cached path), and contiguous inner dots.
///
/// The point is to beat cached BLAS *when the dequant cache is thrashing*, where
/// "cached" degenerates into re-dequantizing the whole model per call plus
/// allocator churn. With a healthy cache, BLAS still wins and callers keep it.
#[cfg(not(target_arch = "wasm32"))]
fn gguf_matmul_bt_rows_generic_gemm(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
    block_bytes: usize,
    block_elems: usize,
) {
    use rayon::prelude::*;
    if block_elems == QK_K && gguf_gemm_specialized(scheme, x, w_bytes, out, m, k, n, block_bytes) {
        return;
    }
    let blocks_per_row = k / block_elems;
    let row_bytes = blocks_per_row * block_bytes;
    // One task per output column; each writes `out[mi * n + j]` for every mi, so
    // scatter into a per-column buffer then transpose in.
    let cols: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|j| {
            let mut acc = vec![0f32; m];
            let start = j * row_bytes;
            let Some(row) = w_bytes.get(start..start + row_bytes) else {
                return acc;
            };
            let mut blk = [0f32; QK_K];
            for b in 0..blocks_per_row {
                dequant_block(
                    scheme,
                    &row[b * block_bytes..(b + 1) * block_bytes],
                    &mut blk,
                );
                let off = b * block_elems;
                for (mi, a) in acc.iter_mut().enumerate() {
                    let xs = &x[mi * k + off..mi * k + off + block_elems];
                    let mut s = 0f32;
                    for t in 0..block_elems {
                        s += xs[t] * blk[t];
                    }
                    *a += s;
                }
            }
            acc
        })
        .collect();
    for (j, acc) in cols.iter().enumerate() {
        for (mi, a) in acc.iter().enumerate() {
            out[mi * n + j] = *a;
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
    // Every other scheme (Q5_K / Q6_K / Q2_K / Q3_K / Q8_0 / IQ*): row-parallel,
    // division-free generic GEMV whenever `k` is a whole number of blocks.
    #[cfg(not(target_arch = "wasm32"))]
    if k.is_multiple_of(block_elems) && block_elems <= QK_K && out.len() == n {
        gguf_matmul_bt_m1_rows_generic(x_row, w_bytes, out, k, scheme, block_bytes, block_elems);
        return;
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
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            // SAFETY: gated on runtime AVX2 detection; slabs are full blocks.
            return unsafe { q4k_dot_q8_block_avx2(q4, q8) };
        }
        rlx_gguf::q4_k_dot_q8_k(q4, q8)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        rlx_gguf::q4_k_dot_q8_k(q4, q8)
    }
}

// ---------------------------------------------------------------------------
// Q4_K x Q8_K block dot, AVX2 (x86-64).
//
// aarch64 had a hand-written NEON dot while x86 fell through to the pure-scalar
// `rlx_gguf::q4_k_dot_q8_k` — a nibble-at-a-time loop over every weight. Since
// Q4_K is ~95% of a K-quant checkpoint's parameters, that scalar loop WAS the
// x86 decode cost. This mirrors the NEON structure exactly (same scale/min
// bookkeeping, same group order), so results are bit-comparable.
//
// The inner shape maps cleanly onto AVX2: Q4 nibbles are unsigned 0..15 and Q8
// activations are signed i8, which is precisely `_mm256_maddubs_epi16`
// (u8 x i8 -> i16 pairwise sums), then `_mm256_madd_epi16` against ones widens
// to i32 without overflow (max |sum| per pair = 15*127*2 << i16::MAX).
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn q4k_scale_min_x86(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Horizontal sum of 8 x i32 in a `__m256i`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_i32_avx2(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        let s = _mm_add_epi32(lo, hi);
        let sh = _mm_shuffle_epi32(s, 0b01_00_11_10);
        let s = _mm_add_epi32(s, sh);
        let sh2 = _mm_shuffle_epi32(s, 0b10_11_00_01);
        _mm_cvtsi128_si32(_mm_add_epi32(s, sh2))
    }
}

/// BOTH nibble halves of one 32-byte Q4 group, from a SINGLE load.
///
/// The obvious form calls a per-half helper twice, which loads the same 32 `qs`
/// bytes and rebuilds the mask constants each time. Q4_K stores the low nibbles
/// of group `j` and the high nibbles of group `j+1` in the same bytes, and their
/// two Q8 segments are adjacent, so one load feeds both dots. Returns
/// `(lo_sum, hi_sum)` for the caller to scale by `sc0` / `sc1`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4k_group32_q8_dot_avx2_both(qs: *const u8, q8_lo: *const i8) -> (i32, i32) {
    use std::arch::x86_64::*;
    unsafe {
        let qbytes = _mm256_loadu_si256(qs as *const __m256i);
        let mask = _mm256_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);
        let nib_lo = _mm256_and_si256(qbytes, mask);
        let nib_hi = _mm256_and_si256(_mm256_srli_epi16(qbytes, 4), mask);
        // The two activation groups are contiguous: lo uses [0,32), hi [32,64).
        let y_lo = _mm256_loadu_si256(q8_lo as *const __m256i);
        let y_hi = _mm256_loadu_si256(q8_lo.add(32) as *const __m256i);
        let a = _mm256_madd_epi16(_mm256_maddubs_epi16(nib_lo, y_lo), ones);
        let b = _mm256_madd_epi16(_mm256_maddubs_epi16(nib_hi, y_hi), ones);
        (hsum_i32_avx2(a), hsum_i32_avx2(b))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4k_dot_q8_block_avx2(q4: &[u8], q8: &[u8]) -> f32 {
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
            let (_, m) = q4k_scale_min_x86(j / 2, scales);
            let bs = i16::from_le_bytes([*bsums.add(j * 2), *bsums.add(j * 2 + 1)]) as i32;
            sumi_min += m as i32 * bs;
        }

        let mut sumi = 0i32;
        let mut is = 0usize;
        let mut yi = 0usize;
        for j in (0..8).step_by(2) {
            let (sc0, _) = q4k_scale_min_x86(j, scales);
            let (sc1, _) = q4k_scale_min_x86(j + 1, scales);
            let (p0, p1) = q4k_group32_q8_dot_avx2_both(qs.add(is), q8s.add(yi));
            sumi += sc0 as i32 * p0 + sc1 as i32 * p1;
            yi += 64;
            is += 32;
        }
        d * yd * sumi as f32 - dmin * yd * sumi_min as f32
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| std::is_x86_feature_detected!("avx2"))
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
/// Cached `dotprod` (ARMv8.2-A) availability — Cortex-A76 (Raspberry Pi 5) /
/// Apple Silicon / Graviton2+ / most ARM servers have it; Cortex-A72 (Pi 4)
/// does not. Detected once; the result is a hot-loop branch predictor's dream.
#[cfg(target_arch = "aarch64")]
#[inline]
fn aarch64_has_dotprod() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    // `RLX_Q4K_NO_DOTPROD=1` forces the baseline path (A/B benchmarking, or a
    // hedge against a mis-detected feature on an exotic core).
    *OK.get_or_init(|| {
        std::env::var_os("RLX_Q4K_NO_DOTPROD").is_none()
            && std::arch::is_aarch64_feature_detected!("dotprod")
    })
}

/// Q4_K 32-nibble × int8 dot. Dispatches to the `vdotq_s32` path on ARMv8.2-A
/// (one dot instruction per 16 lanes vs a widen-multiply + pairwise-accumulate
/// chain) or the universal baseline. Both compute the identical *integer* dot
/// (no rounding), so the choice is purely a throughput detail.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn q4k_group32_q8_dot_neon(qs: *const u8, q8: *const i8, high_nibble: bool) -> i32 {
    if aarch64_has_dotprod() {
        // SAFETY: guarded by the runtime `dotprod` detection above.
        unsafe { q4k_group32_q8_dot_dotprod(qs, q8, high_nibble) }
    } else {
        // SAFETY: baseline NEON, present on every aarch64 target.
        unsafe { q4k_group32_q8_dot_vmull(qs, q8, high_nibble) }
    }
}

/// Baseline-NEON (ARMv8.0) Q4_K group dot: `vmull_s8` widen-multiply →
/// `vpadalq_s16` pairwise-accumulate.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn q4k_group32_q8_dot_vmull(qs: *const u8, q8: *const i8, high_nibble: bool) -> i32 {
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

/// `dotprod`-accelerated Q4_K group dot: one `SDOT` accumulates 16 int8
/// products per 128-bit register, collapsing the widen-multiply chain above.
/// The `vdotq_s32` intrinsic is still unstable (`stdarch_neon_dotprod`) on the
/// pinned stable toolchain, so we emit `SDOT` directly — gated by the
/// `dotprod` target feature (verified at runtime by [`aarch64_has_dotprod`]).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn q4k_group32_q8_dot_dotprod(qs: *const u8, q8: *const i8, high_nibble: bool) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = vdupq_n_s32(0);
        let mask = vdupq_n_u8(0x0F);
        for i in (0..32).step_by(16) {
            let qbytes = vld1q_u8(qs.add(i));
            let nibble = if high_nibble {
                vshrq_n_u8(qbytes, 4)
            } else {
                vandq_u8(qbytes, mask)
            };
            // u8 nibbles 0..15 → i8 (still non-negative)
            let q4 = vreinterpretq_s8_u8(nibble);
            let y = vld1q_s8(q8.add(i));
            // acc.4s += dot4(q4.16b, y.16b)
            std::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc,
                a = in(vreg) q4,
                b = in(vreg) y,
                options(nomem, nostack, preserves_flags),
            );
        }
        vaddvq_s32(acc)
    }
}

// ---------------------------------------------------------------------------
// Q2_0 int8 dot GEMV (llama.cpp-style): activations quantized once to int8 per
// 128-group, then dotted directly against the packed 2-bit codes — VNNI on
// x86-64, NEON on aarch64, scalar elsewhere. See `intrinsics::vnni` and
// `rlx_gguf::q2_dequant`.
// ---------------------------------------------------------------------------

/// One `Q2_0` weight block (34 B) dotted with one packed int8 activation block.
#[cfg(target_arch = "x86_64")]
#[inline]
fn q2_0_dot_q8_block(w: &[u8], a: &[u8]) -> f32 {
    crate::intrinsics::vnni::q2_0_dot_q8_g128(w, a)
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn q2_0_dot_q8_block(w: &[u8], a: &[u8]) -> f32 {
    q2_0_dot_q8_block_neon(w, a)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn q2_0_dot_q8_block(w: &[u8], a: &[u8]) -> f32 {
    rlx_gguf::q2_dequant::q2_0_dot_q8_g128(w, a)
}

#[inline]
fn q2_0_dot_row_q8(row: &[u8], x_q8: &[u8], blocks_per_row: usize) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..blocks_per_row {
        let w = &row[b * Q2_0_BLOCK_BYTES..(b + 1) * Q2_0_BLOCK_BYTES];
        let a = &x_q8[b * Q8_0_G128_BYTES..(b + 1) * Q8_0_G128_BYTES];
        acc += q2_0_dot_q8_block(w, a);
    }
    acc
}

/// `C[n] = x[k] @ W[n,k]^T` for `Q2_0` with `k` a multiple of [`QK2_0`].
///
/// Activations are quantized once to int8 per 128-group; each output row then
/// uses the int8 dot (VNNI/NEON/scalar). Rows are independent — no n-wide
/// partial reduction.
fn q2_0_gemv_parallel(x: &[f32], w_bytes: &[u8], out: &mut [f32], k: usize) {
    let n = out.len();
    let blocks_per_row = k / QK2_0;
    let row_bytes = blocks_per_row * Q2_0_BLOCK_BYTES;
    debug_assert!(x.len() >= k);
    debug_assert!(w_bytes.len() >= n * row_bytes);

    let mut x_q8 = vec![0u8; blocks_per_row * Q8_0_G128_BYTES];
    rlx_gguf::q2_dequant::quantize_q8_0_g128_row(&x[..k], &mut x_q8);

    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(j, slot)| {
            let row = &w_bytes[j * row_bytes..(j + 1) * row_bytes];
            *slot = q2_0_dot_row_q8(row, &x_q8, blocks_per_row);
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (j, slot) in out.iter_mut().enumerate() {
            let row = &w_bytes[j * row_bytes..(j + 1) * row_bytes];
            *slot = q2_0_dot_row_q8(row, &x_q8, blocks_per_row);
        }
    }
}

/// NEON `Q2_0 × int8` block dot (aarch64). Uses signed weights `(q−1)` so no
/// `xsum` term; widening `vmull`/`vpadal` (no `dotprod` requirement).
#[cfg(target_arch = "aarch64")]
#[inline]
fn q2_0_dot_q8_block_neon(w: &[u8], a: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    // SAFETY: fixed-size Q2_0 (34 B) / Q8_0_G128 blocks; NEON baseline on aarch64.
    unsafe {
        let d = half::f16::from_le_bytes([w[0], w[1]]).to_f32();
        let dx = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
        let qs = w.as_ptr().add(2);
        let acts = a.as_ptr().add(4) as *const i8;
        let mask = vdupq_n_u8(0x03);
        let one = vdupq_n_s8(1);
        let mut acc = vdupq_n_s32(0);
        // 128 codes = 32 packed bytes; 16 packed bytes (64 codes) per iter.
        for blk in 0..2 {
            let pk = vld1q_u8(qs.add(blk * 16)); // 16 bytes → 64 codes
            // Code (q−1) at natural position 4i+r for r = 0..3.
            let w0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pk, mask)), one);
            let w1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pk, 2), mask)), one);
            let w2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pk, 4), mask)), one);
            let w3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(pk, 6), mask)), one);
            // vld4 deinterleaves: av.i[j] = acts[4j+i] — matches wi ordering.
            let av = vld4q_s8(acts.add(blk * 64));
            for (wr, ar) in [(w0, av.0), (w1, av.1), (w2, av.2), (w3, av.3)] {
                let lo = vmull_s8(vget_low_s8(wr), vget_low_s8(ar));
                let hi = vmull_s8(vget_high_s8(wr), vget_high_s8(ar));
                acc = vpadalq_s16(acc, lo);
                acc = vpadalq_s16(acc, hi);
            }
        }
        d * dx * vaddvq_s32(acc) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q2_0_gemv_matches_f32_reference() {
        let k = 256; // 2 groups of 128
        let n = 48;
        let w: Vec<f32> = (0..k * n)
            .map(|i| (((i * 3) % 3) as i32 - 1) as f32 * ((i % 5) as f32 * 0.1 + 0.1))
            .collect();
        let packed = rlx_gguf::q2_dequant::quantize_q2_0(&w).unwrap();
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.017).sin() * 2.0).collect();

        // int8 dot GEMV (dispatch entry).
        let mut got = vec![0f32; n];
        gguf_matmul_bt_dispatch(&x, &packed, &mut got, 1, k, n, QuantScheme::GgufQ2_0);

        // Reference: dequant weight to f32, plain dot.
        let wd = rlx_gguf::q2_dequant::dequant_q2_0(&packed, k * n).unwrap();
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += x[p] * wd[j * k + p];
            }
            // int8 activation quant → small relative tolerance.
            assert!(
                (got[j] - acc).abs() < 2e-2 * (1.0 + acc.abs()),
                "row {j}: {} vs {acc}",
                got[j]
            );
        }
    }

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

    /// The decisive decode question: does the int8 Q4_K GEMV (the SDOT-
    /// optimizable path) actually BEAT cached-f32-BLAS (Accelerate = AMX), which
    /// is what `prefer_cached_blas` routes real LLM Q4_K matmuls to? If int8
    /// already wins, an SDOT speedup is a real decode lever; if Accelerate wins,
    /// the framework's routing is already optimal and SDOT is moot.
    ///   cargo test -p rlx-cpu q4k_int8_vs_cached_blas -- --ignored --nocapture
    #[test]
    #[ignore = "manual kernel timing"]
    fn q4k_int8_vs_cached_blas() {
        use crate::dequant_cache::clear_dequant_cache;
        use std::time::Instant;
        let time = |label: &str, f: &mut dyn FnMut()| {
            for _ in 0..3 {
                f();
            }
            let it = 30;
            let t = Instant::now();
            for _ in 0..it {
                f();
            }
            eprintln!(
                "    {label}: {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3 / it as f64
            );
        };
        for (k, n) in [(1024usize, 1024usize), (1024, 3072), (1024, 151936)] {
            let w: Vec<f32> = (0..k * n)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
                .collect();
            let packed = rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q4K).unwrap();
            let x: Vec<f32> = (0..k).map(|i| 0.001 * i as f32 - 0.5).collect();
            let mut out = vec![0f32; n];
            eprintln!("decode GEMV k={k} n={n}:");
            // int8 Q4_K GEMV (serial int8 Q8_K dots — the path SDOT would speed up).
            time("int8 q4k gemv (serial)", &mut || {
                super::q4k_gemv_rowmajor(&x, &packed, &mut out, k)
            });
            // int8 Q4_K GEMV (parallel, rayon over rows) — the actual decode path
            // when routed to the fused kernel; the real competitor to single-AMX.
            time("int8 q4k gemv (parallel)", &mut || {
                gguf_matmul_bt(&x, &packed, &mut out, 1, k, n, QuantScheme::GgufQ4K)
            });
            // cached-f32-BLAS = what the model actually runs (Accelerate/AMX).
            clear_dequant_cache();
            time("cached f32 BLAS (Accel)", &mut || {
                gguf_matmul_bt_cached(&x, &packed, &mut out, 1, k, n, QuantScheme::GgufQ4K, 0)
            });
        }
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

    /// The `dotprod` Q4_K group dot must be BYTE-IDENTICAL to the baseline
    /// `vmull` path — it's a pure integer dot, so there's no rounding slack.
    /// Guards the runtime dispatch used by the fused int8 decode GEMV.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q4k_group32_dotprod_matches_vmull() {
        if !super::aarch64_has_dotprod() {
            return; // ARMv8.0 (e.g. Pi 4) — only the vmull path exists.
        }
        let mut s: u32 = 0x1234_5678;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for _ in 0..256 {
            let qs: Vec<u8> = (0..32).map(|_| (rng() & 0xFF) as u8).collect();
            let q8: Vec<i8> = (0..32).map(|_| rng() as i8).collect();
            for high in [false, true] {
                let a = unsafe { super::q4k_group32_q8_dot_vmull(qs.as_ptr(), q8.as_ptr(), high) };
                let b =
                    unsafe { super::q4k_group32_q8_dot_dotprod(qs.as_ptr(), q8.as_ptr(), high) };
                assert_eq!(a, b, "dotprod != vmull (high_nibble={high})");
            }
        }
    }
}
