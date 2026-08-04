// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Single-pass online-softmax SDPA (FlashAttention v1 inner-row form).
// One thread per (batch, head, q_row); each thread walks the K dimension
// exactly once, maintaining a running (m, l, O[D]) tuple:
//
//   m_new = max(m, s)
//   e_old = exp(m - m_new); e_cur = exp(s - m_new)
//   l     = e_old · l + e_cur
//   O[d]  = e_old · O[d] + e_cur · V[s][d]
//   m     = m_new
//
// At end: out[d] = O[d] / l. This does ≈ seq_k · 2·head_dim FMAs per row
// vs. the 3-pass form's ≈ seq_k · 4·head_dim — measured ~2.6× faster on
// MiniLM6/bge-* shapes — and the running max/scale rebase is also more
// numerically stable than the 3-pass "max then sum_exp then weighted sum".
//
// Inputs all live in the arena as [B, H, S, D] f32 tensors.
// Mask kinds:
//   2 = binary key-padding mask (Custom): element < 0.5 → score = -1e9.
//   4 = additive bias mask (Bias): score += mask element (e.g. a
//       block-diagonal window bias carrying 0 / large-negative).
// Both index the mask via the per-axis mask strides below. Caller is
// responsible for normalizing other shapes upstream.
//
// `O` is held in a per-thread private array<f32, MAX_HEAD_DIM>. Gemma 3
// decode uses head_dim=256; keep headroom for Llama-class 128 and future
// 512-dim heads without spilling on Apple GPUs (private limit ~10 KiB/thread).

const MAX_HEAD_DIM: u32 = 512u;

struct Params {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,

    out_off: u32,
    mask_off: u32,
    mask_kind: u32,    // 0=None, 1=Causal, 2=Custom(binary), 3=SlidingWindow, 4=Bias(additive)
    scale_bits: u32,   // bitcast<f32>(1/sqrt(D))
    window: u32,       // SlidingWindow width (only used when mask_kind == 3)
    // MASK address strides. The kernel computes:
    //   mask_addr = mask_off
    //             + b  * mask_batch_stride
    //             + h  * mask_head_stride
    //             + qi * seq_q_stride
    //             + s  * seq_k_stride
    // Setting head/q strides to 0 lets the kernel read a [B, S]
    // padding mask directly without materializing the [B, H, S_q, S_k]
    // broadcast (saves the Expand pre-pass per attention block).
    seq_q_stride: u32,
    seq_k_stride: u32,
    mask_batch_stride: u32,
    mask_head_stride: u32,
    kv_heads: u32,     // GQA/MQA: #KV heads query heads share (heads=MHA, 0=unset→MHA)
    // Asymmetric SDPA (DeepSeek/Kimi MLA): V rows + output are `v_head_dim`
    // wide while Q/K scores still use `head_dim`. Equals `head_dim` for
    // ordinary symmetric attention; 0 (zero-init) falls back to `head_dim`.
    v_head_dim: u32,
    _pad_mask_2: u32,

    // Per-tensor strides (in f32 elements). Q/K/V/out can be in either
    // [B, H, S, D] or [B, S, H, D] layout — caller sets the strides.
    q_batch_stride: u32, q_head_stride: u32, q_seq_stride: u32, _pad_q: u32,
    k_batch_stride: u32, k_head_stride: u32, k_seq_stride: u32, _pad_k: u32,
    v_batch_stride: u32, v_head_stride: u32, v_seq_stride: u32, _pad_v: u32,
    o_batch_stride: u32, o_head_stride: u32, o_seq_stride: u32, _pad_o: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn attention(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let total = params.batch * params.heads * params.seq_q;
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= total) { return; }
    let qi = row % params.seq_q;
    let q1 = row / params.seq_q;
    let h  = q1 % params.heads;
    let b  = q1 / params.heads;
    let scale = bitcast<f32>(params.scale_bits);
    // Absolute query position for causal/window masking. In a decode step
    // seq_q=1 but the query sits at position seq_k-1 (past KV preceded it), so
    // causality must compare against `qi + (seq_k - seq_q)`, not the local
    // `qi`. Prefill (seq_q == seq_k) leaves this as `qi`.
    let q_abs = qi + select(0u, params.seq_k - params.seq_q, params.seq_k >= params.seq_q);

    // Mask address uses generic per-axis strides. Each axis is folded
    // independently; setting head/q strides to 0 lets us read a
    // broadcast mask without materializing it. The s-dependent part
    // is left to the inner loop.
    let mask_partial = params.mask_off
        + b  * params.mask_batch_stride
        + h  * params.mask_head_stride
        + qi * params.seq_q_stride;

    // Q, K, V, output base addresses use explicit per-axis strides so the
    // kernel works with [B, H, S, D] OR [B, S, H, D] layout uniformly.
    let q_base = params.q_off
        + b * params.q_batch_stride
        + h * params.q_head_stride
        + qi * params.q_seq_stride;
    // GQA / MQA: several query heads share one KV head. Map query head `h` to
    // its KV head. kv_heads is always set (== heads for plain MHA → group 1);
    // the max() guards protect against a stray 0.
    let kv_denom = max(params.kv_heads, 1u);
    let kv_h = h / max(params.heads / kv_denom, 1u);
    let k_bh   = params.k_off + b * params.k_batch_stride + kv_h * params.k_head_stride;
    let v_bh   = params.v_off + b * params.v_batch_stride + kv_h * params.v_head_stride;
    let o_base = params.out_off
        + b * params.o_batch_stride
        + h * params.o_head_stride
        + qi * params.o_seq_stride;

    let hd = params.head_dim;
    // V/output per-head width. For symmetric SDPA this equals `head_dim`;
    // for MLA the V rows and output are `v_head_dim` wide while the Q·K
    // score still contracts over `head_dim`. A zero-init params value
    // (older callers / defaulted uniforms) falls back to `head_dim`.
    let hd_v = select(params.v_head_dim, hd, params.v_head_dim == 0u);
    if (hd > MAX_HEAD_DIM || hd_v > MAX_HEAD_DIM) { return; }

    // Cache Q[qi, :] in registers — read seq_k times by the dot product.
    var q_reg: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        q_reg[d] = arena[q_base + d];
    }

    // Online softmax accumulators. `o` spans the V-width `hd_v`.
    var m: f32 = -3.4e38;
    var l: f32 = 0.0;
    var o: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < hd_v; d = d + 1u) { o[d] = 0.0; }

    for (var s: u32 = 0u; s < params.seq_k; s = s + 1u) {
        // Score: scale * Q · K[s] + mask
        let k_base = k_bh + s * params.k_seq_stride;
        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            score = score + q_reg[d] * arena[k_base + d];
        }
        score = score * scale;
        if (params.mask_kind == 1u) {
            if (s > q_abs) { score = -3.4e38; }
        } else if (params.mask_kind == 2u) {
            // BERT-style binary multiplicative mask (1=valid, 0=padding).
            // Matches CPU/Metal: a position with mask < 0.5 → score driven
            // to -1e9. Hardcoded 0.5 keeps parity across backends.
            if (arena[mask_partial + s * params.seq_k_stride] < 0.5) { score = -1e9; }
        } else if (params.mask_kind == 3u) {
            if (s > q_abs) { score = -3.4e38; }
            else if (q_abs - s > params.window) { score = -3.4e38; }
        } else if (params.mask_kind == 4u) {
            // Additive bias mask (e.g. block-diagonal window bias): the mask
            // carries additive values (0 to attend, large-negative to drop)
            // added to the score pre-softmax — NOT a 0/1 indicator.
            score = score + arena[mask_partial + s * params.seq_k_stride];
        }

        // Online softmax update.
        let m_new = max(m, score);
        let e_old = exp(m - m_new);
        let e_cur = exp(score - m_new);
        l = e_old * l + e_cur;
        // V rows are `hd_v` wide (v_seq_stride already reflects that width).
        let v_base = v_bh + s * params.v_seq_stride;
        for (var d: u32 = 0u; d < hd_v; d = d + 1u) {
            o[d] = e_old * o[d] + e_cur * arena[v_base + d];
        }
        m = m_new;
    }

    // Normalize and emit `hd_v`-wide output rows. l is guaranteed > 0
    // (at least one finite score).
    let inv_l = 1.0 / l;
    for (var d: u32 = 0u; d < hd_v; d = d + 1u) {
        arena[o_base + d] = o[d] * inv_l;
    }
}
