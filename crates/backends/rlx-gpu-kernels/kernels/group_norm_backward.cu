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

// GroupNorm (NCHW) backward. Matches CPU `training_bwd::group_norm_backward_*`
// and Metal / wgpu group_norm_bwd_*. One block per (batch, group) for dx;
// single-thread serial for dgamma / dbeta.

#define GNB_BLOCK 256

extern "C" __global__ void group_norm_bwd_input(
    float* arena,
    unsigned int x_off,
    unsigned int gamma_off,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int n,
    unsigned int c,
    unsigned int h,
    unsigned int w,
    unsigned int num_groups,
    unsigned int eps_bits
) {
    unsigned int ng = blockIdx.x;
    if (ng >= n * num_groups) return;
    unsigned int bn = ng / num_groups;
    unsigned int g = ng % num_groups;
    unsigned int cpg = c / num_groups;
    unsigned int c0 = g * cpg;
    unsigned int spatial = h * w;
    unsigned int count = cpg * spatial;
    float n_inv = 1.0f / (float)count;
    unsigned int plane = c * spatial;
    unsigned int b_base = bn * plane;
    unsigned int tid = threadIdx.x;
    float eps = __int_as_float((int)eps_bits);

    __shared__ float partial_a[GNB_BLOCK];
    __shared__ float partial_b[GNB_BLOCK];

    float local_sum = 0.0f;
    for (unsigned int i = tid; i < count; i += GNB_BLOCK) {
        unsigned int c_off = i / spatial;
        unsigned int s = i % spatial;
        local_sum += arena[x_off + b_base + (c0 + c_off) * spatial + s];
    }
    partial_a[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = GNB_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial_a[tid] += partial_a[tid + stride];
        __syncthreads();
    }
    float mean = partial_a[0] * n_inv;
    __syncthreads();

    float local_var = 0.0f;
    for (unsigned int i = tid; i < count; i += GNB_BLOCK) {
        unsigned int c_off = i / spatial;
        unsigned int s = i % spatial;
        float d = arena[x_off + b_base + (c0 + c_off) * spatial + s] - mean;
        local_var += d * d;
    }
    partial_a[tid] = local_var;
    __syncthreads();
    for (unsigned int stride = GNB_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial_a[tid] += partial_a[tid + stride];
        __syncthreads();
    }
    float inv_std = rsqrtf(partial_a[0] * n_inv + eps);
    __syncthreads();

    float local_sy = 0.0f;
    float local_sxh = 0.0f;
    for (unsigned int i = tid; i < count; i += GNB_BLOCK) {
        unsigned int c_off = i / spatial;
        unsigned int s = i % spatial;
        unsigned int gi = c0 + c_off;
        float xh = (arena[x_off + b_base + gi * spatial + s] - mean) * inv_std;
        float sy = arena[dy_off + b_base + gi * spatial + s] * arena[gamma_off + gi];
        local_sy += sy;
        local_sxh += sy * xh;
    }
    partial_a[tid] = local_sy;
    partial_b[tid] = local_sxh;
    __syncthreads();
    for (unsigned int stride = GNB_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_a[tid] += partial_a[tid + stride];
            partial_b[tid] += partial_b[tid + stride];
        }
        __syncthreads();
    }
    float m_sy = partial_a[0] * n_inv;
    float m_sxh = partial_b[0] * n_inv;

    for (unsigned int i = tid; i < count; i += GNB_BLOCK) {
        unsigned int c_off = i / spatial;
        unsigned int s = i % spatial;
        unsigned int gi = c0 + c_off;
        float xh = (arena[x_off + b_base + gi * spatial + s] - mean) * inv_std;
        float sy = arena[dy_off + b_base + gi * spatial + s] * arena[gamma_off + gi];
        arena[out_off + b_base + gi * spatial + s] = inv_std * (sy - m_sy - xh * m_sxh);
    }
}

extern "C" __global__ void group_norm_bwd_gamma(
    float* arena,
    unsigned int x_off,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int n,
    unsigned int c,
    unsigned int h,
    unsigned int w,
    unsigned int num_groups,
    unsigned int eps_bits
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned int spatial = h * w;
    unsigned int plane = c * spatial;
    unsigned int cpg = c / num_groups;
    float n_inv = 1.0f / (float)(cpg * spatial);
    float eps = __int_as_float((int)eps_bits);

    for (unsigned int ch = 0; ch < c; ++ch) {
        arena[out_off + ch] = 0.0f;
    }

    for (unsigned int bn = 0; bn < n; ++bn) {
        unsigned int b_base = bn * plane;
        for (unsigned int g = 0; g < num_groups; ++g) {
            unsigned int c0 = g * cpg;
            float mean = 0.0f;
            for (unsigned int ci = 0; ci < cpg; ++ci) {
                unsigned int base = x_off + b_base + (c0 + ci) * spatial;
                for (unsigned int s = 0; s < spatial; ++s) mean += arena[base + s];
            }
            mean *= n_inv;
            float var = 0.0f;
            for (unsigned int ci = 0; ci < cpg; ++ci) {
                unsigned int base = x_off + b_base + (c0 + ci) * spatial;
                for (unsigned int s = 0; s < spatial; ++s) {
                    float d = arena[base + s] - mean;
                    var += d * d;
                }
            }
            float inv_std = rsqrtf(var * n_inv + eps);
            for (unsigned int ci = 0; ci < cpg; ++ci) {
                unsigned int gi = c0 + ci;
                unsigned int x_base = x_off + b_base + gi * spatial;
                unsigned int dy_base = dy_off + b_base + gi * spatial;
                float acc = arena[out_off + gi];
                for (unsigned int s = 0; s < spatial; ++s) {
                    float xh = (arena[x_base + s] - mean) * inv_std;
                    acc += arena[dy_base + s] * xh;
                }
                arena[out_off + gi] = acc;
            }
        }
    }
}

extern "C" __global__ void group_norm_bwd_beta(
    float* arena,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int n,
    unsigned int c,
    unsigned int h,
    unsigned int w
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned int spatial = h * w;
    unsigned int plane = c * spatial;
    for (unsigned int ch = 0; ch < c; ++ch) {
        arena[out_off + ch] = 0.0f;
    }
    for (unsigned int bn = 0; bn < n; ++bn) {
        unsigned int b_base = bn * plane;
        for (unsigned int ch = 0; ch < c; ++ch) {
            unsigned int dy_base = dy_off + b_base + ch * spatial;
            float acc = arena[out_off + ch];
            for (unsigned int s = 0; s < spatial; ++s) {
                acc += arena[dy_base + s];
            }
            arena[out_off + ch] = acc;
        }
    }
}
