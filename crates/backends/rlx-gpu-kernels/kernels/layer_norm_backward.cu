// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// LayerNorm backward (last-axis rows). Matches CPU `LayerNormBackward*` /
// Metal `layer_norm_bwd` / wgpu `layer_norm_bwd_*`.
//   Input: one block per row, shared-memory tree reductions.
//   Gamma: serial single-thread (same shape as `group_norm_bwd_gamma`);
//   `layer_norm_bwd_gamma_cols` is the parallel version CUDA launches.

#define LNB_BLOCK 256

__device__ __forceinline__ float lnb_block_sum(float v, float* s,
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

extern "C" __global__ void layer_norm_bwd_input(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int x_off,
    unsigned int gamma_off,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int eps_bits
) {
    unsigned int row = blockIdx.x;
    if (row >= outer || inner == 0u) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int x_base = x_off + row * inner;
    unsigned int dy_base = dy_off + row * inner;
    unsigned int out_base = out_off + row * inner;
    float n_inv = 1.0f / (float)inner;
    float eps = __int_as_float((int)eps_bits);

    __shared__ float s_a[LNB_BLOCK];
    __shared__ float s_b[LNB_BLOCK];

    float local_sum = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        local_sum += arena[x_base + i];
    }
    float mean = lnb_block_sum(local_sum, s_a, tid, bsz) * n_inv;

    float local_var = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        float d = arena[x_base + i] - mean;
        local_var += d * d;
    }
    // Precise 1/sqrt — matches CPU `1.0 / (var*n_inv + eps).sqrt()`.
    float inv_std = 1.0f / sqrtf(lnb_block_sum(local_var, s_a, tid, bsz) * n_inv + eps);

    float local_sy = 0.0f;
    float local_sxh = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        float xh = (arena[x_base + i] - mean) * inv_std;
        float sy = arena[dy_base + i] * arena[gamma_off + i];
        local_sy += sy;
        local_sxh += sy * xh;
    }
    s_a[tid] = local_sy;
    s_b[tid] = local_sxh;
    __syncthreads();
    for (unsigned int off = bsz / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_a[tid] += s_a[tid + off];
            s_b[tid] += s_b[tid + off];
        }
        __syncthreads();
    }
    float m_sy = s_a[0] * n_inv;
    float m_sxh = s_b[0] * n_inv;

    for (unsigned int i = tid; i < inner; i += bsz) {
        float xh = (arena[x_base + i] - mean) * inv_std;
        float sy = arena[dy_base + i] * arena[gamma_off + i];
        arena[out_base + i] = inv_std * (sy - m_sy - xh * m_sxh);
    }
}

extern "C" __global__ void layer_norm_bwd_gamma(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int x_off,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int eps_bits
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (inner == 0u) return;
    float n_inv = 1.0f / (float)inner;
    float eps = __int_as_float((int)eps_bits);

    for (unsigned int i = 0; i < inner; ++i) {
        arena[out_off + i] = 0.0f;
    }

    for (unsigned int row = 0; row < outer; ++row) {
        unsigned int x_base = x_off + row * inner;
        unsigned int dy_base = dy_off + row * inner;
        float sum = 0.0f;
        for (unsigned int i = 0; i < inner; ++i) sum += arena[x_base + i];
        float mean = sum * n_inv;
        float var = 0.0f;
        for (unsigned int i = 0; i < inner; ++i) {
            float d = arena[x_base + i] - mean;
            var += d * d;
        }
        float inv_std = 1.0f / sqrtf(var * n_inv + eps);
        for (unsigned int i = 0; i < inner; ++i) {
            float xh = (arena[x_base + i] - mean) * inv_std;
            arena[out_off + i] += arena[dy_base + i] * xh;
        }
    }
}

// Parallel gamma gradient: launch grid (ceil(inner/32), 1, 1), block (1024, 1, 1).
// Each block owns 32 columns; its 32 warps take rows w, w+32, ... A warp computes
// the row's mean / inv_std with shuffles (two-pass, like the serial kernel),
// then lane c accumulates dy * xh for column tile*32 + c. The 32 warp partials
// are summed in a fixed order, so the result is deterministic.
#define LNG_WARPS 32

// Width-32 sub-group butterfly, portable to HIP (wave32/64) like
// `attention_warp.cu`; ROCm compiles this file too (it launches only the
// serial kernels, but the whole TU must build).
#if defined(__HIP_DEVICE_COMPILE__)
#define LNG_SHFL_XOR(v, m) __shfl_xor((v), (m), 32)
#else
#define LNG_SHFL_XOR(v, m) __shfl_xor_sync(0xffffffffu, (v), (m), 32)
#endif

__device__ __forceinline__ float lng_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += LNG_SHFL_XOR(v, off);
    return v;
}

extern "C" __global__ void layer_norm_bwd_gamma_cols(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int x_off,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int eps_bits
) {
    if (inner == 0u) return;
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int col = blockIdx.x * 32u + lane;
    float n_inv = 1.0f / (float)inner;
    float eps = __int_as_float((int)eps_bits);

    float acc = 0.0f;
    for (unsigned int row = warp; row < outer; row += LNG_WARPS) {
        unsigned int x_base = x_off + row * inner;
        float s = 0.0f;
        for (unsigned int i = lane; i < inner; i += 32u) s += arena[x_base + i];
        float mean = lng_warp_sum(s) * n_inv;
        float v = 0.0f;
        for (unsigned int i = lane; i < inner; i += 32u) {
            float d = arena[x_base + i] - mean;
            v += d * d;
        }
        float inv_std = 1.0f / sqrtf(lng_warp_sum(v) * n_inv + eps);
        if (col < inner) {
            float xh = (arena[x_base + col] - mean) * inv_std;
            acc += arena[dy_off + row * inner + col] * xh;
        }
    }

    __shared__ float part[LNG_WARPS][32];
    part[warp][lane] = acc;
    __syncthreads();
    if (warp == 0 && col < inner) {
        float t = 0.0f;
        for (unsigned int w = 0; w < LNG_WARPS; ++w) t += part[w][lane];
        arena[out_off + col] = t;
    }
}
