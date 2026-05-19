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
// Mask layout (when mask_kind == 2):
//   [B, H, S_q, S_k] additive — added to scores pre-softmax.
// Caller is responsible for normalizing other shapes upstream.
//
// `O` is held in a per-thread private array<f32, MAX_HEAD_DIM>. BERT-class
// models all use head_dim ≤ 128, so this stays well within Apple-Metal's
// per-thread private storage budget without spilling.

const MAX_HEAD_DIM: u32 = 128u;

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
    mask_kind: u32,    // 0=None, 1=Causal, 2=Array(custom), 3=SlidingWindow
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
    _pad_mask_0: u32,
    _pad_mask_1: u32,
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
    let k_bh   = params.k_off + b * params.k_batch_stride + h * params.k_head_stride;
    let v_bh   = params.v_off + b * params.v_batch_stride + h * params.v_head_stride;
    let o_base = params.out_off
        + b * params.o_batch_stride
        + h * params.o_head_stride
        + qi * params.o_seq_stride;

    let hd = params.head_dim;

    // Cache Q[qi, :] in registers — read seq_k times by the dot product.
    var q_reg: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        q_reg[d] = arena[q_base + d];
    }

    // Online softmax accumulators.
    var m: f32 = -3.4e38;
    var l: f32 = 0.0;
    var o: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) { o[d] = 0.0; }

    for (var s: u32 = 0u; s < params.seq_k; s = s + 1u) {
        // Score: scale * Q · K[s] + mask
        let k_base = k_bh + s * params.k_seq_stride;
        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            score = score + q_reg[d] * arena[k_base + d];
        }
        score = score * scale;
        if (params.mask_kind == 1u) {
            if (s > qi) { score = -3.4e38; }
        } else if (params.mask_kind == 2u) {
            // BERT-style binary multiplicative mask (1=valid, 0=padding).
            // Matches CPU/Metal: a position with mask < 0.5 → score driven
            // to -1e9. Hardcoded 0.5 keeps parity across backends.
            if (arena[mask_partial + s * params.seq_k_stride] < 0.5) { score = -1e9; }
        } else if (params.mask_kind == 3u) {
            if (s > qi) { score = -3.4e38; }
            else if (qi - s > params.window) { score = -3.4e38; }
        }

        // Online softmax update.
        let m_new = max(m, score);
        let e_old = exp(m - m_new);
        let e_cur = exp(score - m_new);
        l = e_old * l + e_cur;
        let v_base = v_bh + s * params.v_seq_stride;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            o[d] = e_old * o[d] + e_cur * arena[v_base + d];
        }
        m = m_new;
    }

    // Normalize and emit. l is guaranteed > 0 (at least one finite score).
    let inv_l = 1.0 / l;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        arena[o_base + d] = o[d] * inv_l;
    }
}
