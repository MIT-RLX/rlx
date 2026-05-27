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

// FlashAttention-1 SDPA. One block per (batch, head, q-tile); each block
// processes BR query rows against the full K/V sequence in BC-row tiles.
// K/V tiles are loaded into shared memory once per tile, reused for the
// QK and PV passes. Online softmax across KV tiles maintains
// row_max/row_sum and rescales the running V accumulator on each tile.
//
// Q/K/V layout: [B, H, S, D] f32 in arena.
// Mask kinds: 0=None 1=Causal 2=Custom (binary) 3=SlidingWindow
//
// Block geometry:
//   BR  = 16 query rows / block
//   BC  = 32 key/value rows / KV tile
//   D   ≤ MAX_HEAD_DIM = 128
//   threads/block = BR * WARPS_PER_Q = 16 * 8 = 128
//
// Shared memory (head_dim ≤ 128, +1 padding to break bank conflicts):
//   q_shared[BR][D+1], k_tile[BC][D+1], v_tile[BC][D+1],
//   scores[BR][BC], row_max[BR], row_sum[BR], rescale[BR]
//   ≈ 43.5 KB total — fits in the default 48 KB SM shared budget.
//
// Per-thread output accumulator lives in registers — head_dim/WARPS_PER_Q
// floats (≤ 16 for D=128).

#define MAX_HEAD_DIM 128
#define BR 16
#define BC 32
#define WARPS_PER_Q 8
#define THREADS (BR * WARPS_PER_Q)
#define D_PAD (MAX_HEAD_DIM + 1)
#define MAX_D_PER_THREAD (MAX_HEAD_DIM / WARPS_PER_Q)

extern "C" __global__ void attention(
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
    unsigned int window
) {
    if (head_dim > MAX_HEAD_DIM) return;
    float scale = __int_as_float((int)scale_bits);

    // 2-D production launch: grid=(q_blocks, batch*heads, 1).
    // 1-D HIP-CPU validation launch: grid=(q_blocks*batch*heads, 1, 1).
    // We support both by re-decoding when gridDim.y == 1.
    unsigned int q_block;
    unsigned int bh;
    if (gridDim.y == 1) {
        unsigned int q_blocks = (seq_q + BR - 1) / BR;
        q_block = blockIdx.x % q_blocks;
        bh      = blockIdx.x / q_blocks;
    } else {
        q_block = blockIdx.x;
        bh      = blockIdx.y;
    }
    if (bh >= batch * heads) return;

    unsigned int qi0 = q_block * BR;
    if (qi0 >= seq_q) return;

    unsigned int tid = threadIdx.x;
    unsigned int q_lane = tid / WARPS_PER_Q;   // 0..15
    unsigned int d_lane = tid % WARPS_PER_Q;   // 0..7

    unsigned int qi = qi0 + q_lane;
    bool q_valid = qi < seq_q;

    unsigned int q_base = q_off + (bh * seq_q + qi) * head_dim;
    unsigned int k_base_g = k_off + bh * seq_k * head_dim;
    unsigned int v_base_g = v_off + bh * seq_k * head_dim;
    unsigned int o_base = out_off + (bh * seq_q + qi) * head_dim;

    __shared__ float q_shared[BR][D_PAD];
    __shared__ float k_tile[BC][D_PAD];
    __shared__ float v_tile[BC][D_PAD];
    __shared__ float scores[BR][BC];
    __shared__ float row_max[BR];
    __shared__ float row_sum[BR];
    __shared__ float rescale[BR];

    // Per-thread register accumulator: covers head_dim positions
    // strided by WARPS_PER_Q (one slot per d_step).
    float acc[MAX_D_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_D_PER_THREAD; ++i) acc[i] = 0.0f;

    // Init Q tile, row_max, row_sum.
    if (d_lane == 0) {
        row_max[q_lane] = -3.4e38f;
        row_sum[q_lane] = 0.0f;
    }
    for (unsigned int d = d_lane; d < head_dim; d += WARPS_PER_Q) {
        q_shared[q_lane][d] = q_valid ? arena[q_base + d] : 0.0f;
    }
    __syncthreads();

    unsigned int n_kv = (seq_k + BC - 1) / BC;
    for (unsigned int kt = 0; kt < n_kv; ++kt) {
        unsigned int kc0 = kt * BC;

        // Cooperative K/V tile load: BC rows × head_dim cols.
        // (BC / BR = 2) row chunks per q_lane.
        for (unsigned int r_step = 0; r_step < BC / BR; ++r_step) {
            unsigned int r = r_step * BR + q_lane;
            unsigned int s = kc0 + r;
            for (unsigned int d = d_lane; d < head_dim; d += WARPS_PER_Q) {
                if (s < seq_k) {
                    k_tile[r][d] = arena[k_base_g + s * head_dim + d];
                    v_tile[r][d] = arena[v_base_g + s * head_dim + d];
                } else {
                    k_tile[r][d] = 0.0f;
                    v_tile[r][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Score compute: each thread covers BC/WARPS_PER_Q kc's for its
        // q_lane.
        for (unsigned int kc_step = 0; kc_step < BC / WARPS_PER_Q; ++kc_step) {
            unsigned int kc = kc_step * WARPS_PER_Q + d_lane;
            unsigned int s = kc0 + kc;
            float dot = 0.0f;
            if (q_valid && s < seq_k) {
                for (unsigned int d = 0; d < head_dim; ++d) {
                    dot += q_shared[q_lane][d] * k_tile[kc][d];
                }
                dot *= scale;
                if (mask_kind == 1) {
                    if (s > qi) dot = -3.4e38f;
                } else if (mask_kind == 2) {
                    unsigned int m_idx = (bh * seq_q + qi) * seq_k + s;
                    if (arena[mask_off + m_idx] < 0.5f) dot = -1e9f;
                } else if (mask_kind == 3) {
                    if (s > qi) dot = -3.4e38f;
                    else if (qi - s > window) dot = -3.4e38f;
                }
            } else {
                dot = -3.4e38f;
            }
            scores[q_lane][kc] = dot;
        }
        __syncthreads();

        // Online softmax — one thread per q-row.
        if (d_lane == 0 && q_valid) {
            float new_max = row_max[q_lane];
            for (unsigned int kc = 0; kc < BC; ++kc) {
                new_max = fmaxf(new_max, scores[q_lane][kc]);
            }
            float old_max = row_max[q_lane];
            float rs = (old_max <= -1e30f) ? 0.0f : expf(old_max - new_max);
            float new_sum = row_sum[q_lane] * rs;
            for (unsigned int kc = 0; kc < BC; ++kc) {
                float p = (scores[q_lane][kc] <= -1e30f) ? 0.0f
                        : expf(scores[q_lane][kc] - new_max);
                scores[q_lane][kc] = p;
                new_sum += p;
            }
            row_max[q_lane] = new_max;
            row_sum[q_lane] = new_sum;
            rescale[q_lane] = rs;
        }
        __syncthreads();

        // V accumulation: rescale running acc, then add p @ V.
        if (q_valid) {
            float rs = rescale[q_lane];
            #pragma unroll
            for (int d_step = 0; d_step < MAX_D_PER_THREAD; ++d_step) {
                unsigned int d = d_step * WARPS_PER_Q + d_lane;
                if (d >= head_dim) break;
                float a = acc[d_step] * rs;
                for (unsigned int kc = 0; kc < BC; ++kc) {
                    a += scores[q_lane][kc] * v_tile[kc][d];
                }
                acc[d_step] = a;
            }
        }
        __syncthreads();
    }

    // Final normalize + write.
    if (q_valid) {
        float inv_sum = (row_sum[q_lane] > 0.0f) ? 1.0f / row_sum[q_lane] : 0.0f;
        #pragma unroll
        for (int d_step = 0; d_step < MAX_D_PER_THREAD; ++d_step) {
            unsigned int d = d_step * WARPS_PER_Q + d_lane;
            if (d >= head_dim) break;
            arena[o_base + d] = acc[d_step] * inv_sum;
        }
    }
}
