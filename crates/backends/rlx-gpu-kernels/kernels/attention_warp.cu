// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Warp-per-row online-softmax SDPA — drop-in replacement for `attention_row`
// (identical parameter list, identical math, same mask kinds and strides).
//
// `attention_row` assigns ONE THREAD per (batch, head, q_row). At decode
// seq_q == 1 and batch == 1, so the whole kernel runs with `heads` live
// threads — one partially-filled warp on one SM. It also declares
// `q_reg[MAX_HEAD_DIM]` / `o_acc[MAX_HEAD_DIM]` indexed by the runtime
// `head_dim`, which cannot be promoted to registers and spills to local
// memory (4096 B/thread at MAX_HEAD_DIM 512).
//
// This kernel assigns one 32-lane GROUP per (batch, head, q_row): lane `t`
// owns head-dim elements `t, t+32, t+64, ...`, so the per-row accumulators
// are `ceil(head_dim/32)` registers per lane (local memory = 0) and the
// Q·K dot product becomes a shuffle tree reduction. Row count is unchanged,
// but each row now has 32x the lanes, and the reduction is pairwise rather
// than a serial sum (slightly MORE accurate — cf. the tree-reduction work in
// the wgpu norms).
//
// Wave portability: AMD wavefronts are 64 lanes on gfx9 and 32 on gfx10+.
// The reduction is therefore an explicit width-32 sub-group shuffle and rows
// are indexed by 32-lane group, so a 64-wide wavefront simply carries two
// independent rows. Do not replace the `32` width argument with `warpSize`.
//
// Mask kinds: 0=None 1=Causal 2=Custom (binary) 3=SlidingWindow 4=Bias (additive)

#define RLX_ATTN_WARP_MAX_HEAD_DIM 512
#define RLX_ATTN_LANES 32

// Cross-lane butterfly step, restricted to a 32-lane sub-group on both stacks.
// CUDA's variant takes a participation mask; HIP's classic `__shfl_xor` exists
// on every ROCm version and takes the same trailing `width`. The explicit width
// is what makes this correct on 64-lane wavefronts — see the header note.
//
// NOTE: this is the only shared kernel that uses shuffles. `reduce.cu` avoids
// them deliberately so it stays valid under HIP-CPU, whose wavefront semantics
// differ; `attention_row.cu` (and therefore this drop-in) is not part of the
// HIP-CPU validation TU in `rlx-cuda/cpp/cpu_dispatch.cpp`, so the shuffle is
// safe here. If this kernel is ever added to that TU, replace the butterfly
// with a shared-memory tree first.
#if defined(__HIP_DEVICE_COMPILE__)
#define RLX_ATTN_SHFL_XOR(v, m) __shfl_xor((v), (m), RLX_ATTN_LANES)
#else
#define RLX_ATTN_SHFL_XOR(v, m)                                                \
    __shfl_xor_sync(0xffffffffu, (v), (m), RLX_ATTN_LANES)
#endif

// `EPL` (elements per lane) is a compile-time constant so `q_reg`/`o_acc`
// unroll into registers. The launcher passes a runtime `head_dim`; the entry
// point below switches once (grid-uniform, so no divergence) into the
// specialisation that covers it.
template <int EPL>
__device__ __forceinline__ void rlx_attention_warp_partial(
    float* arena,
    unsigned int heads,
    unsigned int seq_q,
    unsigned int seq_k,
    unsigned int head_dim,
    unsigned int q_off,
    unsigned int k_off,
    unsigned int v_off,
    unsigned int out_off,
    unsigned int mask_off,
    unsigned int mask_kind,
    float scale,
    unsigned int window,
    unsigned int seq_q_stride,
    unsigned int seq_k_stride,
    unsigned int mask_batch_stride,
    unsigned int mask_head_stride,
    unsigned int q_batch_stride,
    unsigned int q_head_stride,
    unsigned int q_seq_stride,
    unsigned int k_batch_stride,
    unsigned int k_head_stride,
    unsigned int k_seq_stride,
    unsigned int v_batch_stride,
    unsigned int v_head_stride,
    unsigned int v_seq_stride,
    unsigned int o_batch_stride,
    unsigned int o_head_stride,
    unsigned int o_seq_stride,
    float softcap,
    unsigned int row,
    unsigned int lane,
    unsigned int s_begin,
    unsigned int s_end,
    float (&o_acc)[EPL],
    float& m_out,
    float& l_out,
    unsigned int& o_base_out
) {
    unsigned int qi = row % seq_q;
    unsigned int q1 = row / seq_q;
    unsigned int h = q1 % heads;
    unsigned int b = q1 / heads;

    // Absolute query position for causal / sliding-window masking. During
    // incremental decode seq_q < seq_k (the query sits after the cached KV),
    // so causality must compare against qi + (seq_k - seq_q), not the local qi.
    unsigned int q_pos = (seq_k >= seq_q) ? (qi + (seq_k - seq_q)) : qi;

    unsigned int mask_partial = mask_off
        + b * mask_batch_stride
        + h * mask_head_stride
        + qi * seq_q_stride;

    unsigned int q_base = q_off
        + b * q_batch_stride
        + h * q_head_stride
        + qi * q_seq_stride;
    unsigned int k_bh = k_off + b * k_batch_stride + h * k_head_stride;
    unsigned int v_bh = v_off + b * v_batch_stride + h * v_head_stride;
    unsigned int o_base = out_off
        + b * o_batch_stride
        + h * o_head_stride
        + qi * o_seq_stride;

    // Lane-strided ownership of the head-dim axis. `head_dim` need not be a
    // multiple of 32; out-of-range slots hold 0 and are never stored back.
    float q_reg[EPL];
    #pragma unroll
    for (int e = 0; e < EPL; ++e) {
        unsigned int d = (unsigned int)e * RLX_ATTN_LANES + lane;
        q_reg[e] = (d < head_dim) ? arena[q_base + d] : 0.0f;
        o_acc[e] = 0.0f;
    }

    float m = -3.4e38f;
    float l = 0.0f;

    for (unsigned int s = s_begin; s < s_end; ++s) {
        unsigned int k_base = k_bh + s * k_seq_stride;
        float part = 0.0f;
        #pragma unroll
        for (int e = 0; e < EPL; ++e) {
            unsigned int d = (unsigned int)e * RLX_ATTN_LANES + lane;
            if (d < head_dim) { part += q_reg[e] * arena[k_base + d]; }
        }
        // Width-32 butterfly: every lane ends up with the full dot product.
        #pragma unroll
        for (int off = RLX_ATTN_LANES / 2; off > 0; off >>= 1) {
            part += RLX_ATTN_SHFL_XOR(part, off);
        }

        float score = part * scale;
        // Gemma 2 attention logit soft-cap (pre-mask so the -inf sentinel survives).
        if (softcap > 0.0f) { score = softcap * tanhf(score / softcap); }
        if (mask_kind == 1u) {
            if (s > q_pos) score = -3.4e38f;
        } else if (mask_kind == 2u) {
            if (arena[mask_partial + s * seq_k_stride] < 0.5f) score = -1e9f;
        } else if (mask_kind == 3u) {
            if (s > q_pos) score = -3.4e38f;
            else if (q_pos - s > window) score = -3.4e38f;
        } else if (mask_kind == 4u) {
            // Additive bias mask (e.g. ALiBi / block-diagonal window bias):
            // the mask carries additive values added to the score pre-softmax
            // — NOT a 0/1 indicator.
            score += arena[mask_partial + s * seq_k_stride];
        }

        float m_new = fmaxf(m, score);
        float e_old = (m <= -1e30f) ? 0.0f : expf(m - m_new);
        float e_cur = (score <= -1e30f) ? 0.0f : expf(score - m_new);
        l = e_old * l + e_cur;
        unsigned int v_base = v_bh + s * v_seq_stride;
        #pragma unroll
        for (int e = 0; e < EPL; ++e) {
            unsigned int d = (unsigned int)e * RLX_ATTN_LANES + lane;
            float vv = (d < head_dim) ? arena[v_base + d] : 0.0f;
            o_acc[e] = e_old * o_acc[e] + e_cur * vv;
        }
        m = m_new;
    }

    m_out = m;
    l_out = l;
    o_base_out = o_base;
}

// The whole key range for one row, normalised and stored.
template <int EPL>
__device__ __forceinline__ void rlx_attention_warp_row(
    float* arena, unsigned int heads, unsigned int seq_q, unsigned int seq_k,
    unsigned int head_dim, unsigned int q_off, unsigned int k_off,
    unsigned int v_off, unsigned int out_off, unsigned int mask_off,
    unsigned int mask_kind, float scale, unsigned int window,
    unsigned int seq_q_stride, unsigned int seq_k_stride,
    unsigned int mask_batch_stride, unsigned int mask_head_stride,
    unsigned int q_batch_stride, unsigned int q_head_stride,
    unsigned int q_seq_stride, unsigned int k_batch_stride,
    unsigned int k_head_stride, unsigned int k_seq_stride,
    unsigned int v_batch_stride, unsigned int v_head_stride,
    unsigned int v_seq_stride, unsigned int o_batch_stride,
    unsigned int o_head_stride, unsigned int o_seq_stride, float softcap,
    unsigned int row, unsigned int lane
) {
    float o_acc[EPL];
    float m, l;
    unsigned int o_base;
    rlx_attention_warp_partial<EPL>(
        arena, heads, seq_q, seq_k, head_dim, q_off, k_off, v_off, out_off,
        mask_off, mask_kind, scale, window, seq_q_stride, seq_k_stride,
        mask_batch_stride, mask_head_stride, q_batch_stride, q_head_stride,
        q_seq_stride, k_batch_stride, k_head_stride, k_seq_stride,
        v_batch_stride, v_head_stride, v_seq_stride, o_batch_stride,
        o_head_stride, o_seq_stride, softcap, row, lane, 0u, seq_k, o_acc, m,
        l, o_base);
    float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    #pragma unroll
    for (int e = 0; e < EPL; ++e) {
        unsigned int d = (unsigned int)e * RLX_ATTN_LANES + lane;
        if (d < head_dim) { arena[o_base + d] = o_acc[e] * inv_l; }
    }
}

// Split-KV variant for decode. With seq_q == 1 the row count is only
// `batch * heads`, so one group per row leaves most lanes idle while each
// group walks the whole cache serially. Here one BLOCK serves one row: its
// `blockDim.x / 32` groups each take a contiguous slice of seq_k, then the
// partial (max, sum, acc) states are merged in shared memory with the usual
// log-sum-exp rescale. Single launch, no arena scratch, so it is safe inside
// a captured graph. Same parameter list as `attention_warp`.
//
// Dynamic shared memory: `groups * (2 + 32 * EPL)` floats, where EPL is the
// specialisation picked for head_dim (the launcher sizes it with the same
// rule as the switch below).
template <int EPL>
__device__ __forceinline__ void rlx_attention_warp_split_row(
    float* arena, unsigned int heads, unsigned int seq_q, unsigned int seq_k,
    unsigned int head_dim, unsigned int q_off, unsigned int k_off,
    unsigned int v_off, unsigned int out_off, unsigned int mask_off,
    unsigned int mask_kind, float scale, unsigned int window,
    unsigned int seq_q_stride, unsigned int seq_k_stride,
    unsigned int mask_batch_stride, unsigned int mask_head_stride,
    unsigned int q_batch_stride, unsigned int q_head_stride,
    unsigned int q_seq_stride, unsigned int k_batch_stride,
    unsigned int k_head_stride, unsigned int k_seq_stride,
    unsigned int v_batch_stride, unsigned int v_head_stride,
    unsigned int v_seq_stride, unsigned int o_batch_stride,
    unsigned int o_head_stride, unsigned int o_seq_stride, float softcap,
    unsigned int row, unsigned int group, unsigned int lane,
    unsigned int groups, float* smem
) {
    const unsigned int width = (unsigned int)EPL * RLX_ATTN_LANES;
    unsigned int chunk = (seq_k + groups - 1) / groups;
    unsigned int s_begin = min(group * chunk, seq_k);
    unsigned int s_end = min(s_begin + chunk, seq_k);

    float o_acc[EPL];
    float m, l;
    unsigned int o_base;
    rlx_attention_warp_partial<EPL>(
        arena, heads, seq_q, seq_k, head_dim, q_off, k_off, v_off, out_off,
        mask_off, mask_kind, scale, window, seq_q_stride, seq_k_stride,
        mask_batch_stride, mask_head_stride, q_batch_stride, q_head_stride,
        q_seq_stride, k_batch_stride, k_head_stride, k_seq_stride,
        v_batch_stride, v_head_stride, v_seq_stride, o_batch_stride,
        o_head_stride, o_seq_stride, softcap, row, lane, s_begin, s_end,
        o_acc, m, l, o_base);

    float* s_m = smem;
    float* s_l = smem + groups;
    float* s_o = smem + 2 * groups;
    if (lane == 0) { s_m[group] = m; s_l[group] = l; }
    #pragma unroll
    for (int e = 0; e < EPL; ++e) {
        s_o[group * width + (unsigned int)e * RLX_ATTN_LANES + lane] = o_acc[e];
    }
    __syncthreads();

    // Every thread computes the shared max and denominator (cheap: `groups`
    // terms), then the block covers head_dim with a block-strided loop.
    float m_all = -3.4e38f;
    for (unsigned int g = 0; g < groups; ++g) { m_all = fmaxf(m_all, s_m[g]); }
    float l_all = 0.0f;
    for (unsigned int g = 0; g < groups; ++g) {
        float w = (s_m[g] <= -1e30f) ? 0.0f : expf(s_m[g] - m_all);
        l_all += w * s_l[g];
    }
    float inv_l = (l_all > 0.0f) ? 1.0f / l_all : 0.0f;
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int g = 0; g < groups; ++g) {
            float w = (s_m[g] <= -1e30f) ? 0.0f : expf(s_m[g] - m_all);
            acc += w * s_o[g * width + d];
        }
        arena[o_base + d] = acc * inv_l;
    }
}

// Parameter list is byte-for-byte identical to `attention_row` so the two are
// interchangeable at the launch site.
extern "C" __global__ void attention_warp(
    float* arena,
    unsigned int batch,
    unsigned int heads,
    unsigned int seq_q,
    unsigned int seq_k,
    unsigned int head_dim,
    unsigned int q_off,
    unsigned int k_off,
    unsigned int v_off,
    unsigned int out_off,
    unsigned int mask_off,
    unsigned int mask_kind,
    unsigned int scale_bits,
    unsigned int window,
    unsigned int seq_q_stride,
    unsigned int seq_k_stride,
    unsigned int mask_batch_stride,
    unsigned int mask_head_stride,
    unsigned int q_batch_stride,
    unsigned int q_head_stride,
    unsigned int q_seq_stride,
    unsigned int k_batch_stride,
    unsigned int k_head_stride,
    unsigned int k_seq_stride,
    unsigned int v_batch_stride,
    unsigned int v_head_stride,
    unsigned int v_seq_stride,
    unsigned int o_batch_stride,
    unsigned int o_head_stride,
    unsigned int o_seq_stride,
    unsigned int softcap_bits
) {
    if (head_dim > RLX_ATTN_WARP_MAX_HEAD_DIM) return;
    float scale = __int_as_float((int)scale_bits);
    float softcap = __int_as_float((int)softcap_bits);

    // One 32-lane group per query row. Groups are carved from threadIdx.x, not
    // from the hardware wavefront, so this is identical on wave32 and wave64.
    unsigned int group = threadIdx.x / RLX_ATTN_LANES;
    unsigned int lane = threadIdx.x % RLX_ATTN_LANES;
    unsigned int groups_per_block = blockDim.x / RLX_ATTN_LANES;
    unsigned int row = blockIdx.x * groups_per_block + group;
    unsigned int total = batch * heads * seq_q;
    // Warp-uniform: every lane of a group takes the same branch, so the
    // shuffle reduction below always has its full 32-lane cohort.
    if (row >= total) return;

#define RLX_ATTN_WARP_CALL(EPL_)                                               \
    rlx_attention_warp_row<EPL_>(                                              \
        arena, heads, seq_q, seq_k, head_dim, q_off, k_off, v_off, out_off,    \
        mask_off, mask_kind, scale, window, seq_q_stride, seq_k_stride,        \
        mask_batch_stride, mask_head_stride, q_batch_stride, q_head_stride,    \
        q_seq_stride, k_batch_stride, k_head_stride, k_seq_stride,             \
        v_batch_stride, v_head_stride, v_seq_stride, o_batch_stride,           \
        o_head_stride, o_seq_stride, softcap, row, lane)

    if (head_dim <= 32u)       { RLX_ATTN_WARP_CALL(1); }
    else if (head_dim <= 64u)  { RLX_ATTN_WARP_CALL(2); }
    else if (head_dim <= 128u) { RLX_ATTN_WARP_CALL(4); }
    else if (head_dim <= 256u) { RLX_ATTN_WARP_CALL(8); }
    else                       { RLX_ATTN_WARP_CALL(16); }

#undef RLX_ATTN_WARP_CALL
}

// 16 groups max: 1024-thread blocks run out of registers at EPL >= 4.
extern "C" __global__ void __launch_bounds__(512) attention_warp_split(
    float* arena,
    unsigned int batch,
    unsigned int heads,
    unsigned int seq_q,
    unsigned int seq_k,
    unsigned int head_dim,
    unsigned int q_off,
    unsigned int k_off,
    unsigned int v_off,
    unsigned int out_off,
    unsigned int mask_off,
    unsigned int mask_kind,
    unsigned int scale_bits,
    unsigned int window,
    unsigned int seq_q_stride,
    unsigned int seq_k_stride,
    unsigned int mask_batch_stride,
    unsigned int mask_head_stride,
    unsigned int q_batch_stride,
    unsigned int q_head_stride,
    unsigned int q_seq_stride,
    unsigned int k_batch_stride,
    unsigned int k_head_stride,
    unsigned int k_seq_stride,
    unsigned int v_batch_stride,
    unsigned int v_head_stride,
    unsigned int v_seq_stride,
    unsigned int o_batch_stride,
    unsigned int o_head_stride,
    unsigned int o_seq_stride,
    unsigned int softcap_bits
) {
    extern __shared__ float rlx_attn_split_smem[];
    if (head_dim > RLX_ATTN_WARP_MAX_HEAD_DIM) return;
    float scale = __int_as_float((int)scale_bits);
    float softcap = __int_as_float((int)softcap_bits);

    // One block per query row; block-uniform, so __syncthreads is safe.
    unsigned int row = blockIdx.x;
    if (row >= batch * heads * seq_q) return;
    unsigned int group = threadIdx.x / RLX_ATTN_LANES;
    unsigned int lane = threadIdx.x % RLX_ATTN_LANES;
    unsigned int groups = blockDim.x / RLX_ATTN_LANES;

#define RLX_ATTN_SPLIT_CALL(EPL_)                                              \
    rlx_attention_warp_split_row<EPL_>(                                        \
        arena, heads, seq_q, seq_k, head_dim, q_off, k_off, v_off, out_off,    \
        mask_off, mask_kind, scale, window, seq_q_stride, seq_k_stride,        \
        mask_batch_stride, mask_head_stride, q_batch_stride, q_head_stride,    \
        q_seq_stride, k_batch_stride, k_head_stride, k_seq_stride,             \
        v_batch_stride, v_head_stride, v_seq_stride, o_batch_stride,           \
        o_head_stride, o_seq_stride, softcap, row, group, lane, groups,        \
        rlx_attn_split_smem)

    if (head_dim <= 32u)       { RLX_ATTN_SPLIT_CALL(1); }
    else if (head_dim <= 64u)  { RLX_ATTN_SPLIT_CALL(2); }
    else if (head_dim <= 128u) { RLX_ATTN_SPLIT_CALL(4); }
    else if (head_dim <= 256u) { RLX_ATTN_SPLIT_CALL(8); }
    else                       { RLX_ATTN_SPLIT_CALL(16); }

#undef RLX_ATTN_SPLIT_CALL
}
