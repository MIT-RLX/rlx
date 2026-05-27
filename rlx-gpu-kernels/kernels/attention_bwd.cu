// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Scaled dot-product attention backward for [B, H, S, D] f32 layout.
// Recomputes per-row scores + softmax, then emits dQ, dK, or dV
// selected by `wrt` (0=Query, 1=Key, 2=Value).
//
// Mask kinds (match forward `attention.cu`): 0=None 1=Causal 2=Custom
// 3=SlidingWindow 4=Bias (additive [B,H,S_q,S_k] tensor at mask_off).

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
    } else if (wrt == 2u) {
        unsigned int ki = blockIdx.y * blockDim.x + threadIdx.x;
        if (ki >= seq_k) return;
        unsigned int k_base = k_base_g + ki * head_dim;
        unsigned int v_base = v_base_g + ki * head_dim;
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
            for (unsigned int d = 0; d < head_dim; ++d) {
                arena[o_base + d] += scores[ki] * arena[dy_base + d];
            }
        }
    } else if (wrt == 1u) {
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
