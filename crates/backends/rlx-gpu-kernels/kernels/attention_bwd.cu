// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// Scaled dot-product attention backward for [B, H, S, D] f32 layout.
// Recomputes per-row scores + softmax, then emits dQ, dK, or dV
// selected by `wrt` (0=Query, 1=Key, 2=Value).
//
// Mask kinds (match forward `attention.cu`): 0=None 1=Causal 2=Custom
// 3=SlidingWindow 4=Bias (additive [B,H,S_q,S_k] tensor at mask_off).
//
// PERF: dQ is thread-per-query, so each query's softmax row is built once —
// O(S_q·S_k·D). The dK/dV outputs are indexed by key, so a naive thread-per-key
// kernel rebuilds the WHOLE S_q×S_k softmax matrix once per key = O(S_q·S_k²·D),
// i.e. ~S_k× slower (this dominated training: 91% of GPU time, 136ms vs 0.8ms
// forward). When one block spans all keys of a (batch,head) — the common case,
// seq_k ≤ blockDim.x — the key-threads instead build each query's softmax row
// ONCE cooperatively in shared memory, restoring O(S_q·S_k·D). The original
// per-key loops are kept as a correctness fallback for seq_k > blockDim.x.

#define MAX_HEAD_DIM 128
#define MAX_ATTN_SEQ 512

__device__ inline float mask_score(
    float dot,
    unsigned qi,
    unsigned ki,
    unsigned bh,
    unsigned seq_q,
    unsigned seq_k,
    unsigned mask_kind,
    unsigned mask_off,
    unsigned window,
    float* arena
) {
    if (mask_kind == 1u) {
        if (ki > qi) return -3.4e38f;
    } else if (mask_kind == 2u) {
        unsigned int m_idx = (bh * seq_q + qi) * seq_k + ki;
        if (arena[mask_off + m_idx] < 0.5f) return -1e9f;
    } else if (mask_kind == 3u) {
        if (ki > qi) return -3.4e38f;
        else if (qi - ki > window) return -3.4e38f;
    } else if (mask_kind == 4u) {
        dot += arena[mask_off + (bh * seq_q + qi) * seq_k + ki];
    }
    return dot;
}

__device__ inline void softmax_row(float* scores, unsigned seq_k) {
    float m = -3.4e38f;
    for (unsigned int s = 0; s < seq_k; ++s) {
        m = fmaxf(m, scores[s]);
    }
    float sum = 0.0f;
    for (unsigned int s = 0; s < seq_k; ++s) {
        float e = (scores[s] <= -1e30f) ? 0.0f : expf(scores[s] - m);
        scores[s] = e;
        sum += e;
    }
    float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    for (unsigned int s = 0; s < seq_k; ++s) {
        scores[s] *= inv;
    }
}

extern "C" __global__ void attention_bwd(
    float* arena,
    unsigned int batch,
    unsigned int heads,
    unsigned int seq_q,
    unsigned int seq_k,
    unsigned int head_dim,
    unsigned int q_off,
    unsigned int k_off,
    unsigned int v_off,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int mask_off,
    unsigned int mask_kind,
    unsigned int scale_bits,
    unsigned int window,
    unsigned int wrt
) {
    if (head_dim > MAX_HEAD_DIM || seq_k > MAX_ATTN_SEQ || seq_q > MAX_ATTN_SEQ) return;
    float scale = __int_as_float((int)scale_bits);

    unsigned int bh = blockIdx.x;
    if (bh >= batch * heads) return;

    unsigned int q_base_g = q_off + bh * seq_q * head_dim;
    unsigned int k_base_g = k_off + bh * seq_k * head_dim;
    unsigned int v_base_g = v_off + bh * seq_k * head_dim;
    unsigned int dy_base_g = dy_off + bh * seq_q * head_dim;

    float scores[MAX_ATTN_SEQ];
    float dp[MAX_ATTN_SEQ];

    if (wrt == 0u) {
        unsigned int qi = blockIdx.y * blockDim.x + threadIdx.x;
        if (qi >= seq_q) return;
        unsigned int q_base = q_base_g + qi * head_dim;
        unsigned int dy_base = dy_base_g + qi * head_dim;
        unsigned int o_base = out_off + (bh * seq_q + qi) * head_dim;

        for (unsigned int ki = 0; ki < seq_k; ++ki) {
            float dot = 0.0f;
            unsigned int k_base = k_base_g + ki * head_dim;
            for (unsigned int d = 0; d < head_dim; ++d) {
                dot += arena[q_base + d] * arena[k_base + d];
            }
            dot *= scale;
            scores[ki] = mask_score(dot, qi, ki, bh, seq_q, seq_k, mask_kind, mask_off, window, arena);
        }
        softmax_row(scores, seq_k);

        for (unsigned int ki = 0; ki < seq_k; ++ki) {
            float acc = 0.0f;
            unsigned int v_base = v_base_g + ki * head_dim;
            for (unsigned int d = 0; d < head_dim; ++d) {
                acc += arena[dy_base + d] * arena[v_base + d];
            }
            dp[ki] = acc;
        }
        float row_sum = 0.0f;
        for (unsigned int ki = 0; ki < seq_k; ++ki) {
            row_sum += scores[ki] * dp[ki];
        }
        for (unsigned int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (unsigned int ki = 0; ki < seq_k; ++ki) {
                float ds = scores[ki] * (dp[ki] - row_sum) * scale;
                acc += ds * arena[k_base_g + ki * head_dim + d];
            }
            arena[o_base + d] = acc;
        }
        return;
    }

    // wrt == 1 (dK) or wrt == 2 (dV).
    if (seq_k <= blockDim.x) {
        // ---- Fast path: one block spans all keys; softmax row built ONCE ----
        __shared__ float sh_p[MAX_ATTN_SEQ];   // holds row scores, then P[qi,:]
        __shared__ float sh_dp[MAX_ATTN_SEQ];  // dV-projection dp[qi,:] (dK only)
        unsigned int ki = threadIdx.x;
        bool active = (ki < seq_k);
        unsigned int k_base = k_base_g + ki * head_dim;
        unsigned int v_base = v_base_g + ki * head_dim;
        unsigned int o_base = out_off + (bh * seq_k + ki) * head_dim;

        float acc[MAX_HEAD_DIM];
        for (unsigned int d = 0; d < head_dim; ++d) acc[d] = 0.0f;

        for (unsigned int qi = 0; qi < seq_q; ++qi) {
            unsigned int q_base = q_base_g + qi * head_dim;
            unsigned int dy_base = dy_base_g + qi * head_dim;

            // (1) each key-thread computes its own score s = scale·Q[qi]·K[ki].
            float s = -3.4e38f;
            if (active) {
                float dot = 0.0f;
                for (unsigned int d = 0; d < head_dim; ++d) {
                    dot += arena[q_base + d] * arena[k_base + d];
                }
                dot *= scale;
                s = mask_score(dot, qi, ki, bh, seq_q, seq_k, mask_kind, mask_off, window, arena);
                sh_p[ki] = s;
            }
            __syncthreads();

            // (2) each thread reduces the shared row to (max, sum) → P[qi,ki].
            float m = -3.4e38f;
            for (unsigned int kk = 0; kk < seq_k; ++kk) m = fmaxf(m, sh_p[kk]);
            float Z = 0.0f;
            for (unsigned int kk = 0; kk < seq_k; ++kk) {
                Z += (sh_p[kk] <= -1e30f) ? 0.0f : expf(sh_p[kk] - m);
            }
            float invZ = (Z > 0.0f) ? 1.0f / Z : 0.0f;
            float p = active ? (((s <= -1e30f) ? 0.0f : expf(s - m)) * invZ) : 0.0f;

            if (wrt == 2u) {
                // dV[ki,d] += P[qi,ki] · dY[qi,d]
                if (active) {
                    for (unsigned int d = 0; d < head_dim; ++d) {
                        acc[d] += p * arena[dy_base + d];
                    }
                }
                __syncthreads(); // sh_p reused by next qi
            } else {
                // dK: dp[qi,ki] = dY[qi]·V[ki]; delta = Σ_k P[qi,k]·dp[qi,k]
                __syncthreads(); // all threads done reading s before overwrite
                float dpk = 0.0f;
                if (active) {
                    for (unsigned int d = 0; d < head_dim; ++d) {
                        dpk += arena[dy_base + d] * arena[v_base + d];
                    }
                    sh_dp[ki] = dpk;
                    sh_p[ki] = p; // overwrite row scores with P for the delta reduction
                }
                __syncthreads();
                float delta = 0.0f;
                for (unsigned int kk = 0; kk < seq_k; ++kk) delta += sh_p[kk] * sh_dp[kk];
                float dscore = active ? (p * (dpk - delta) * scale) : 0.0f;
                if (active) {
                    for (unsigned int d = 0; d < head_dim; ++d) {
                        acc[d] += dscore * arena[q_base + d];
                    }
                }
                __syncthreads(); // sh_p/sh_dp reused by next qi
            }
        }
        if (active) {
            for (unsigned int d = 0; d < head_dim; ++d) arena[o_base + d] = acc[d];
        }
    } else if (wrt == 2u) {
        // ---- Fallback (seq_k > blockDim.x): original thread-per-key dV ----
        unsigned int ki = blockIdx.y * blockDim.x + threadIdx.x;
        if (ki >= seq_k) return;
        unsigned int v_base = v_base_g + ki * head_dim;
        unsigned int o_base = out_off + (bh * seq_k + ki) * head_dim;
        (void)v_base;

        for (unsigned int d = 0; d < head_dim; ++d) {
            arena[o_base + d] = 0.0f;
        }

        for (unsigned int qi = 0; qi < seq_q; ++qi) {
            unsigned int q_base = q_base_g + qi * head_dim;
            unsigned int dy_base = dy_base_g + qi * head_dim;
            for (unsigned int kj = 0; kj < seq_k; ++kj) {
                float dot = 0.0f;
                unsigned int kb = k_base_g + kj * head_dim;
                for (unsigned int d = 0; d < head_dim; ++d) {
                    dot += arena[q_base + d] * arena[kb + d];
                }
                dot *= scale;
                scores[kj] = mask_score(dot, qi, kj, bh, seq_q, seq_k, mask_kind, mask_off, window, arena);
            }
            softmax_row(scores, seq_k);
            for (unsigned int d = 0; d < head_dim; ++d) {
                arena[o_base + d] += scores[ki] * arena[dy_base + d];
            }
        }
    } else {
        // ---- Fallback (seq_k > blockDim.x): original thread-per-key dK ----
        unsigned int ki = blockIdx.y * blockDim.x + threadIdx.x;
        if (ki >= seq_k) return;
        unsigned int o_base = out_off + (bh * seq_k + ki) * head_dim;
        for (unsigned int d = 0; d < head_dim; ++d) {
            arena[o_base + d] = 0.0f;
        }

        for (unsigned int qi = 0; qi < seq_q; ++qi) {
            unsigned int q_base = q_base_g + qi * head_dim;
            unsigned int dy_base = dy_base_g + qi * head_dim;
            for (unsigned int kj = 0; kj < seq_k; ++kj) {
                float dot = 0.0f;
                unsigned int kb = k_base_g + kj * head_dim;
                for (unsigned int d = 0; d < head_dim; ++d) {
                    dot += arena[q_base + d] * arena[kb + d];
                }
                dot *= scale;
                scores[kj] = mask_score(dot, qi, kj, bh, seq_q, seq_k, mask_kind, mask_off, window, arena);
            }
            softmax_row(scores, seq_k);

            for (unsigned int kj = 0; kj < seq_k; ++kj) {
                float acc = 0.0f;
                unsigned int vb = v_base_g + kj * head_dim;
                for (unsigned int d = 0; d < head_dim; ++d) {
                    acc += arena[dy_base + d] * arena[vb + d];
                }
                dp[kj] = acc;
            }
            float row_sum = 0.0f;
            for (unsigned int kj = 0; kj < seq_k; ++kj) {
                row_sum += scores[kj] * dp[kj];
            }
            float ds_ki = scores[ki] * (dp[ki] - row_sum) * scale;
            for (unsigned int d = 0; d < head_dim; ++d) {
                arena[o_base + d] += ds_ki * arena[q_base + d];
            }
        }
    }
}
