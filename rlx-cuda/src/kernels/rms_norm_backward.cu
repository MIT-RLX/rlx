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
// RMSNorm backward (row = last-axis slice). wrt: 0=dx, 1=dgamma (atomic), 2=dbeta (atomic).
// y = x * inv_r * gamma + beta, inv_r = rsqrt(mean(x^2)+eps)

#define RNB_BLOCK 256

__device__ __forceinline__ float rnb_block_sum(float v, float* s,
        unsigned int tid, unsigned int bsz) {
    s[tid] = v;
    __syncthreads();
    for (unsigned int off = bsz / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    float r = s[0];
    __syncthreads();
    return r;
}

extern "C" __global__ void rlx_rms_norm_bwd(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int x_off,
    unsigned int gamma_off,
    unsigned int beta_off,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int eps_bits,
    unsigned int wrt
) {
    unsigned int row = blockIdx.x;
    if (row >= outer) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int x_base = x_off + row * inner;
    unsigned int dy_base = dy_off + row * inner;
    float n_inv = 1.0f / (float)inner;
    float eps = __int_as_float((int)eps_bits);

    __shared__ float s[RNB_BLOCK];

    float local_dot = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        float xv = arena[x_base + i];
        float gv = arena[gamma_off + i];
        float dyv = arena[dy_base + i];
        local_dot += dyv * gv * xv;
    }
    float dot = rnb_block_sum(local_dot, s, tid, bsz) * n_inv;

    float local_ss = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        float xv = arena[x_base + i];
        local_ss += xv * xv;
    }
    float ss = rnb_block_sum(local_ss, s, tid, bsz);
    float inv_r = rsqrtf(ss * n_inv + eps);
    float inv_r3 = inv_r * inv_r * inv_r;

    if (wrt == 0u) {
        unsigned int out_base = out_off + row * inner;
        for (unsigned int i = tid; i < inner; i += bsz) {
            float xv = arena[x_base + i];
            float gv = arena[gamma_off + i];
            float dyv = arena[dy_base + i];
            float term = gv * dyv - xv * dot * inv_r3;
            arena[out_base + i] = term * inv_r;
        }
    } else if (wrt == 1u) {
        for (unsigned int i = tid; i < inner; i += bsz) {
            float xv = arena[x_base + i];
            float dyv = arena[dy_base + i];
            atomicAdd(&arena[out_off + i], dyv * xv * inv_r);
        }
    } else {
        for (unsigned int i = tid; i < inner; i += bsz) {
            float dyv = arena[dy_base + i];
            atomicAdd(&arena[out_off + i], dyv);
        }
    }
}

extern "C" __global__ void rlx_zero_f32(
    float* arena,
    unsigned int off,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arena[off + i] = 0.0f;
    }
}
