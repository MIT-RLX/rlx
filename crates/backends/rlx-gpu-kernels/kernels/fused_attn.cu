// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Native fused-attention core for `Op::FusedAttentionBlock`.
//
// Reads the PACKED QKV projection `[B, S, 3*inner]` (per token:
// `[Q(inner) | K(inner) | V(inner)]`, heads interleaved within each
// section), applies optional NeoX RoPE to Q/K inline, runs softmax SDPA
// one (batch, head) per block with the score matrix resident in shared
// memory, and writes the attention output `[B, S, inner]`. The QKV and
// output projections stay as separate GEMMs (cuBLAS-class), so this
// kernel collapses the decompose chain's narrow×3 + transpose×3 + rope×2
// + attention into a single launch with no arena round-trips between
// stages.
//
// Grid:  one block per (batch * head); blockIdx.x = bi*heads + hi.
// Block: any 1-D size (threads stride over score / output elements).
// Shared memory: seq*seq floats (the score matrix) — caller sizes it.
//
// mask_kind: 0=None, 1=Causal (ki>qi dropped), 2=Custom (binary [B,S];
//            mask[bi*seq+ki] < 0.5 ⇒ dropped). Matches the CPU thunk.

extern "C" __global__ void fused_attn_block(
    float* arena,
    unsigned int qkv_off,    // f32 element offset of QKV [B, S, 3*inner]
    unsigned int mask_off,   // f32 element offset of mask [B, S] (custom)
    unsigned int cos_off,    // f32 element offset of cos [S, head_dim/2]
    unsigned int sin_off,    // f32 element offset of sin [S, head_dim/2]
    unsigned int out_off,    // f32 element offset of attn out [B, S, inner]
    unsigned int batch,
    unsigned int seq,
    unsigned int heads,
    unsigned int head_dim,
    unsigned int mask_kind,
    unsigned int scale_bits, // 1/sqrt(head_dim) as f32 bits
    unsigned int has_rope
) {
    unsigned int inner = heads * head_dim;
    unsigned int bh = blockIdx.x;
    unsigned int bi = bh / heads;
    unsigned int hi = bh % heads;
    if (bi >= batch) return;

    float scale = __int_as_float((int)scale_bits);
    unsigned int half = head_dim / 2;
    unsigned int tok_stride = 3u * inner;           // per-token stride in QKV
    unsigned int q_sec = 0u;
    unsigned int k_sec = inner;
    unsigned int v_sec = 2u * inner;

    extern __shared__ float scores[];               // [seq * seq]

    unsigned int tid = threadIdx.x;
    unsigned int tsize = blockDim.x;

    // 1. scores[qi][ki] = (RoPE(Q_qi) . RoPE(K_ki)) * scale, masked.
    unsigned int total = seq * seq;
    for (unsigned int idx = tid; idx < total; idx += tsize) {
        unsigned int qi = idx / seq;
        unsigned int ki = idx % seq;
        unsigned int qb = qkv_off + (bi * seq + qi) * tok_stride + q_sec + hi * head_dim;
        unsigned int kb = qkv_off + (bi * seq + ki) * tok_stride + k_sec + hi * head_dim;
        float dot = 0.0f;
        if (has_rope) {
            unsigned int qcos = qi * half;
            unsigned int kcos = ki * half;
            for (unsigned int i = 0; i < half; ++i) {
                float q1 = arena[qb + i],        q2 = arena[qb + half + i];
                float k1 = arena[kb + i],        k2 = arena[kb + half + i];
                float cq = arena[cos_off + qcos + i], sq = arena[sin_off + qcos + i];
                float ck = arena[cos_off + kcos + i], sk = arena[sin_off + kcos + i];
                float qr1 = q1 * cq - q2 * sq, qr2 = q2 * cq + q1 * sq;
                float kr1 = k1 * ck - k2 * sk, kr2 = k2 * ck + k1 * sk;
                dot += qr1 * kr1 + qr2 * kr2;
            }
        } else {
            for (unsigned int d = 0; d < head_dim; ++d) {
                dot += arena[qb + d] * arena[kb + d];
            }
        }
        float s = dot * scale;
        if (mask_kind == 1u) {
            if (ki > qi) s = -1e9f;
        } else if (mask_kind == 2u) {
            if (arena[mask_off + bi * seq + ki] < 0.5f) s = -1e9f;
        }
        scores[qi * seq + ki] = s;
    }
    __syncthreads();

    // 2. Row-softmax — one thread per query row.
    for (unsigned int qi = tid; qi < seq; qi += tsize) {
        float mx = -1e30f;
        for (unsigned int ki = 0; ki < seq; ++ki) mx = fmaxf(mx, scores[qi * seq + ki]);
        float sum = 0.0f;
        for (unsigned int ki = 0; ki < seq; ++ki) {
            float e = expf(scores[qi * seq + ki] - mx);
            scores[qi * seq + ki] = e;
            sum += e;
        }
        float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (unsigned int ki = 0; ki < seq; ++ki) scores[qi * seq + ki] *= inv;
    }
    __syncthreads();

    // 3. out[qi][d] = sum_ki softmax[qi][ki] * V[ki][d] → [B, S, inner].
    unsigned int otot = seq * head_dim;
    for (unsigned int idx = tid; idx < otot; idx += tsize) {
        unsigned int qi = idx / head_dim;
        unsigned int d = idx % head_dim;
        float acc = 0.0f;
        for (unsigned int ki = 0; ki < seq; ++ki) {
            unsigned int vb = qkv_off + (bi * seq + ki) * tok_stride + v_sec + hi * head_dim;
            acc += scores[qi * seq + ki] * arena[vb + d];
        }
        arena[out_off + (bi * seq + qi) * inner + hi * head_dim + d] = acc;
    }
}
