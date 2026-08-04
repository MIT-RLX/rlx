// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Tensor-Core FlashAttention (fp16 WMMA QK^T + P@V, f32 accumulate). Two
// entry points share one templated body:
//   attention_wmma       — 4 warps, 64-query block tile, head_dim <= 64
//   attention_wmma_d128  — 2 warps, 32-query block tile, head_dim <= 128
// The host dispatches whichever fits (and, in `auto` mode, only when the
// workload is big enough to amortize — see backend/run.rs). Both have the SAME
// signature as the scalar `attention` kernel (drop-in), fall back to scalar
// otherwise, and keep the static shared-memory working set under the 48 KB
// default budget. CUDA-only (`nvcuda::wmma`, `#include <mma.h>`), so rlx-rocm
// never hipRTC-compiles this file.
//
// Each warp owns a 16-query WMMA tile; warps share the K/V smem tiles. Q/K/V
// are cast f32->half on the shared-memory load (arena stays f32). The online-
// softmax O accumulator lives in shared memory (f32) because a WMMA accumulator
// *fragment* has no portable element->row mapping for the per-row rescale. KV
// tile BC=32 (two WMMA tiles) amortizes the per-iteration softmax/addressing
// overhead — that overhead, not the matmul, was the bottleneck at small
// head_dim (see the datapath analysis in tools/kernel-inspect).

#include <mma.h>
using namespace nvcuda;

#define WM 16
#define WN 16
#define WK 16
#define AW_BC 32               // key/value rows per KV tile (2 WMMA tiles)
#define AW_BC_T (AW_BC / WN)   // 2

// WARPS warps -> BR = WARPS*16 query rows / block. DPAD = head_dim padded to a
// multiple of 16 (>= head_dim, <= this cap).
template <int WARPS, int DPAD>
__device__ __forceinline__ void attn_wmma_impl(
    float* arena, unsigned int batch, unsigned int heads, unsigned int seq_q,
    unsigned int seq_k, unsigned int head_dim, unsigned int q_off,
    unsigned int k_off, unsigned int v_off, unsigned int out_off,
    unsigned int mask_off, unsigned int mask_kind, unsigned int scale_bits,
    unsigned int window, unsigned int seq_q_stride, unsigned int seq_k_stride,
    unsigned int mask_batch_stride, unsigned int mask_head_stride,
    unsigned int q_batch_stride, unsigned int q_head_stride,
    unsigned int q_seq_stride, unsigned int k_batch_stride,
    unsigned int k_head_stride, unsigned int k_seq_stride,
    unsigned int v_batch_stride, unsigned int v_head_stride,
    unsigned int v_seq_stride, unsigned int o_batch_stride,
    unsigned int o_head_stride, unsigned int o_seq_stride,
    unsigned int softcap_bits) {
    const int BR = WARPS * WM;
    const int THREADS = WARPS * 32;
    (void)window;
    (void)mask_off;
    (void)mask_batch_stride;
    (void)mask_head_stride;
    (void)softcap_bits;
    if (head_dim > (unsigned)DPAD) return;
    float scale = __int_as_float((int)scale_bits);

    unsigned int q_block = blockIdx.x;
    unsigned int bh = blockIdx.y;
    if (bh >= batch * heads) return;
    unsigned int qi0 = q_block * BR;
    if (qi0 >= seq_q) return;

    unsigned int tid = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31u;
    unsigned int wr0 = warp * WM;

    unsigned int h_idx = bh % heads;
    unsigned int b_idx = bh / heads;
    unsigned int q_base = q_off + b_idx * q_batch_stride + h_idx * q_head_stride;
    unsigned int k_bh = k_off + b_idx * k_batch_stride + h_idx * k_head_stride;
    unsigned int v_bh = v_off + b_idx * v_batch_stride + h_idx * v_head_stride;
    unsigned int o_base = out_off + b_idx * o_batch_stride + h_idx * o_head_stride;

    unsigned int n_d = (head_dim + WK - 1u) / WK;
    unsigned int d_pad = n_d * WK;

    __shared__ __half q_h[WARPS * WM][DPAD];
    __shared__ __half k_h[AW_BC][DPAD];
    __shared__ __half v_h[AW_BC][DPAD];
    __shared__ float  sc[WARPS * WM][AW_BC];   // scores; reused as P@V scratch
    __shared__ __half p_h[WARPS * WM][AW_BC];
    __shared__ float  o_f[WARPS * WM][DPAD];
    __shared__ float  m_i[WARPS * WM];
    __shared__ float  l_i[WARPS * WM];

    for (unsigned int idx = tid; idx < (unsigned)BR * d_pad; idx += THREADS)
        o_f[idx / d_pad][idx % d_pad] = 0.0f;
    if (tid < (unsigned)BR) { m_i[tid] = -3.4e38f; l_i[tid] = 0.0f; }

    for (unsigned int idx = tid; idx < (unsigned)BR * d_pad; idx += THREADS) {
        unsigned int r = idx / d_pad, d = idx % d_pad;
        unsigned int qi = qi0 + r;
        float v = (qi < seq_q && d < head_dim)
                      ? arena[q_base + qi * q_seq_stride + d] : 0.0f;
        q_h[r][d] = __float2half(v);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> kt_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> v_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> s_acc, pv_acc;

    unsigned int n_kv = (seq_k + AW_BC - 1u) / AW_BC;
    for (unsigned int kt = 0; kt < n_kv; ++kt) {
        unsigned int kc0 = kt * AW_BC;

        for (unsigned int idx = tid; idx < (unsigned)AW_BC * d_pad; idx += THREADS) {
            unsigned int r = idx / d_pad, d = idx % d_pad;
            unsigned int s = kc0 + r;
            bool ok = (s < seq_k && d < head_dim);
            k_h[r][d] = __float2half(ok ? arena[k_bh + s * k_seq_stride + d] : 0.0f);
            v_h[r][d] = __float2half(ok ? arena[v_bh + s * v_seq_stride + d] : 0.0f);
        }
        __syncthreads();

        // S_w = Q_w @ K^T (col_major b gives K^T). BC=32 -> 2 WMMA N-tiles.
        for (unsigned int nt = 0; nt < AW_BC_T; ++nt) {
            wmma::fill_fragment(s_acc, 0.0f);
            for (unsigned int dt = 0; dt < n_d; ++dt) {
                wmma::load_matrix_sync(a_frag, &q_h[wr0][dt * WK], DPAD);
                wmma::load_matrix_sync(kt_frag, &k_h[nt * WN][dt * WK], DPAD);
                wmma::mma_sync(s_acc, a_frag, kt_frag, s_acc);
            }
            wmma::store_matrix_sync(&sc[wr0][nt * WN], s_acc, AW_BC, wmma::mem_row_major);
        }
        __syncwarp();

        if (lane < WM) {
            unsigned int r = wr0 + lane;
            unsigned int qi = qi0 + r;
            unsigned int q_pos = (seq_k >= seq_q) ? (qi + (seq_k - seq_q)) : qi;
            float row_m = m_i[r];
            for (unsigned int j = 0; j < AW_BC; ++j) {
                unsigned int s = kc0 + j;
                float d = sc[r][j] * scale;
                bool valid = (s < seq_k) && (qi < seq_q);
                if (mask_kind == 1 && s > q_pos) valid = false;  // causal
                d = valid ? d : -3.4e38f;
                sc[r][j] = d;
                row_m = fmaxf(row_m, d);
            }
            float old_m = m_i[r];
            float alpha = (old_m <= -1e30f) ? 0.0f : __expf(old_m - row_m);
            float sum = l_i[r] * alpha;
            for (unsigned int j = 0; j < AW_BC; ++j) {
                float p = (sc[r][j] <= -1e30f) ? 0.0f : __expf(sc[r][j] - row_m);
                p_h[r][j] = __float2half(p);
                sum += p;
            }
            m_i[r] = row_m;
            l_i[r] = sum;
            for (unsigned int d = 0; d < d_pad; ++d) o_f[r][d] *= alpha;
        }
        __syncwarp();

        // O_w += P_w @ V. BC=32 -> 2 WMMA K-tiles; `sc` reused as P@V scratch.
        for (unsigned int ot = 0; ot < n_d; ++ot) {
            wmma::fill_fragment(pv_acc, 0.0f);
            for (unsigned int kt2 = 0; kt2 < AW_BC_T; ++kt2) {
                wmma::load_matrix_sync(a_frag, &p_h[wr0][kt2 * WK], AW_BC);
                wmma::load_matrix_sync(v_frag, &v_h[kt2 * WK][ot * WK], DPAD);
                wmma::mma_sync(pv_acc, a_frag, v_frag, pv_acc);
            }
            wmma::store_matrix_sync(&sc[wr0][0], pv_acc, AW_BC, wmma::mem_row_major);
            __syncwarp();
            if (lane < WM) {
                unsigned int r = wr0 + lane;
                for (unsigned int j = 0; j < WN; ++j)
                    o_f[r][ot * WN + j] += sc[r][j];
            }
            __syncwarp();
        }
        __syncthreads();
    }

    if (lane < WM) {
        unsigned int r = wr0 + lane;
        unsigned int qi = qi0 + r;
        if (qi < seq_q) {
            float inv = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;
            for (unsigned int d = 0; d < head_dim; ++d)
                arena[o_base + qi * o_seq_stride + d] = o_f[r][d] * inv;
        }
    }
}

#define AW_ARGS \
    arena, batch, heads, seq_q, seq_k, head_dim, q_off, k_off, v_off, out_off, \
    mask_off, mask_kind, scale_bits, window, seq_q_stride, seq_k_stride, \
    mask_batch_stride, mask_head_stride, q_batch_stride, q_head_stride, \
    q_seq_stride, k_batch_stride, k_head_stride, k_seq_stride, v_batch_stride, \
    v_head_stride, v_seq_stride, o_batch_stride, o_head_stride, o_seq_stride, \
    softcap_bits

#define AW_PARAMS \
    float* arena, unsigned int batch, unsigned int heads, unsigned int seq_q, \
    unsigned int seq_k, unsigned int head_dim, unsigned int q_off, \
    unsigned int k_off, unsigned int v_off, unsigned int out_off, \
    unsigned int mask_off, unsigned int mask_kind, unsigned int scale_bits, \
    unsigned int window, unsigned int seq_q_stride, unsigned int seq_k_stride, \
    unsigned int mask_batch_stride, unsigned int mask_head_stride, \
    unsigned int q_batch_stride, unsigned int q_head_stride, \
    unsigned int q_seq_stride, unsigned int k_batch_stride, \
    unsigned int k_head_stride, unsigned int k_seq_stride, \
    unsigned int v_batch_stride, unsigned int v_head_stride, \
    unsigned int v_seq_stride, unsigned int o_batch_stride, \
    unsigned int o_head_stride, unsigned int o_seq_stride, \
    unsigned int softcap_bits

// 4 warps, 64-query tile, head_dim <= 64.
extern "C" __global__ void __launch_bounds__(128) attention_wmma(AW_PARAMS) {
    attn_wmma_impl<4, 64>(AW_ARGS);
}

// 2 warps, 32-query tile, head_dim <= 128.
extern "C" __global__ void __launch_bounds__(64) attention_wmma_d128(AW_PARAMS) {
    attn_wmma_impl<2, 128>(AW_ARGS);
}
