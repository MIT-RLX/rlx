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

// RLX — versatile ML compiler + runtime.
// Row-wise online-softmax SDPA (matches rlx-wgpu attention.wgsl).
// One thread per (batch, head, q_row); supports arbitrary Q/K/V strides.

#define MAX_HEAD_DIM 128

extern "C" __global__ void attention_row(
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
    unsigned int o_seq_stride
) {
    if (head_dim > MAX_HEAD_DIM) return;
    float scale = __int_as_float((int)scale_bits);

    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * heads * seq_q;
    if (row >= total) return;

    unsigned int qi = row % seq_q;
    unsigned int q1 = row / seq_q;
    unsigned int h = q1 % heads;
    unsigned int b = q1 / heads;

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

    float q_reg[MAX_HEAD_DIM];
    for (unsigned int d = 0; d < head_dim; ++d) {
        q_reg[d] = arena[q_base + d];
    }

    float m = -3.4e38f;
    float l = 0.0f;
    float o_acc[MAX_HEAD_DIM];
    for (unsigned int d = 0; d < head_dim; ++d) {
        o_acc[d] = 0.0f;
    }

    for (unsigned int s = 0; s < seq_k; ++s) {
        unsigned int k_base = k_bh + s * k_seq_stride;
        float score = 0.0f;
        for (unsigned int d = 0; d < head_dim; ++d) {
            score += q_reg[d] * arena[k_base + d];
        }
        score *= scale;
        if (mask_kind == 1) {
            if (s > qi) score = -3.4e38f;
        } else if (mask_kind == 2) {
            if (arena[mask_partial + s * seq_k_stride] < 0.5f) score = -1e9f;
        } else if (mask_kind == 3) {
            if (s > qi) score = -3.4e38f;
            else if (qi - s > window) score = -3.4e38f;
        }

        float m_new = fmaxf(m, score);
        float e_old = (m <= -1e30f) ? 0.0f : expf(m - m_new);
        float e_cur = (score <= -1e30f) ? 0.0f : expf(score - m_new);
        l = e_old * l + e_cur;
        unsigned int v_base = v_bh + s * v_seq_stride;
        for (unsigned int d = 0; d < head_dim; ++d) {
            o_acc[d] = e_old * o_acc[d] + e_cur * arena[v_base + d];
        }
        m = m_new;
    }

    float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (unsigned int d = 0; d < head_dim; ++d) {
        arena[o_base + d] = o_acc[d] * inv_l;
    }
}
