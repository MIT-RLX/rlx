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

// Packed AdaLayerNorm backward: out = [dx ∥ dscale ∥ dshift] (1-D floats).
// Launch: grid=(mod_rows,1,1), block=(256,1,1).
// Each block owns one modulation row and loops `seq_per_mod` feature-rows
// that share it (DiT [B,S,D] / [B,1,D] → mod_rows=B, seq_per_mod=S).

#define ADA_BWD_BLOCK 256

__device__ __forceinline__ float ada_bwd_block_sum(float v, float* s,
        unsigned int tid, unsigned int bsz) {
    s[tid] = v;
    __syncthreads();
    for (unsigned int s_off = bsz / 2; s_off > 0; s_off >>= 1) {
        if (tid < s_off) s[tid] += s[tid + s_off];
        __syncthreads();
    }
    float r = s[0];
    __syncthreads();
    return r;
}

extern "C" __global__ void ada_layer_norm_backward(
    float* arena,
    unsigned int mod_rows,
    unsigned int seq_per_mod,
    unsigned int inner,
    unsigned int x_off,
    unsigned int scale_off,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int eps_bits,
    unsigned int layer_norm
) {
    unsigned int m = blockIdx.x;
    if (m >= mod_rows) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int nx = mod_rows * seq_per_mod * inner;
    unsigned int mod_len = mod_rows * inner;
    unsigned int mod_base = m * inner;
    float* dx = arena + out_off;
    float* dscale = arena + out_off + nx;
    float* dshift = arena + out_off + nx + mod_len;
    float n_inv = 1.0f / (float)inner;
    float eps = __int_as_float((int)eps_bits);

    __shared__ float s[ADA_BWD_BLOCK];

    for (unsigned int i = tid; i < inner; i += bsz) {
        dscale[mod_base + i] = 0.0f;
        dshift[mod_base + i] = 0.0f;
    }
    __syncthreads();

    for (unsigned int seq = 0; seq < seq_per_mod; seq++) {
        unsigned int row = m * seq_per_mod + seq;
        unsigned int x_base = x_off + row * inner;
        unsigned int dy_base = dy_off + row * inner;
        unsigned int dx_base = row * inner;

        float local_sum = 0.0f;
        float local_sumsq = 0.0f;
        for (unsigned int i = tid; i < inner; i += bsz) {
            float v = arena[x_base + i];
            local_sum += v;
            local_sumsq += v * v;
        }
        float sum_x = ada_bwd_block_sum(local_sum, s, tid, bsz);
        float sum_x2 = ada_bwd_block_sum(local_sumsq, s, tid, bsz);

        float mean = 0.0f;
        float inv;
        if (layer_norm != 0u) {
            mean = sum_x * n_inv;
            float var = fmaxf(sum_x2 * n_inv - mean * mean, 0.0f);
            inv = rsqrtf(var + eps);
        } else {
            inv = rsqrtf(sum_x2 * n_inv + eps);
        }

        float local_sy = 0.0f;
        float local_sxh = 0.0f;
        for (unsigned int i = tid; i < inner; i += bsz) {
            float n = (arena[x_base + i] - mean) * inv;
            float d = arena[dy_base + i];
            float sc = arena[scale_off + mod_base + i];
            float sy = d * (1.0f + sc);
            dscale[mod_base + i] += d * n;
            dshift[mod_base + i] += d;
            local_sy += sy;
            local_sxh += sy * n;
        }
        float sum_sy = ada_bwd_block_sum(local_sy, s, tid, bsz);
        float sum_sxh = ada_bwd_block_sum(local_sxh, s, tid, bsz);
        float m_sy = sum_sy * n_inv;
        float m_sxh = sum_sxh * n_inv;

        for (unsigned int i = tid; i < inner; i += bsz) {
            float n = (arena[x_base + i] - mean) * inv;
            float d = arena[dy_base + i];
            float sc = arena[scale_off + mod_base + i];
            float sy = d * (1.0f + sc);
            if (layer_norm != 0u) {
                dx[dx_base + i] = inv * (sy - m_sy - n * m_sxh);
            } else {
                float n_rms = arena[x_base + i] * inv;
                dx[dx_base + i] = inv * (sy - n_rms * m_sxh);
            }
        }
        __syncthreads();
    }
}
