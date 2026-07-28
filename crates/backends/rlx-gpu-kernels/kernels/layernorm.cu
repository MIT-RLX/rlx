// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// LayerNorm and RmsNorm fused into one kernel via op flag.
// Block-per-row with shared-memory tree reductions.
//   LayerNorm (op=0): y = (x - mean) / sqrt(var + eps) * gamma + beta
//   RmsNorm   (op=1): y = x / sqrt(mean(x^2) + eps) * gamma
//
// Launch shape: grid=(outer,1,1), block=(256,1,1).

#define LN_BLOCK 256

__device__ __forceinline__ float ln_block_sum(float v, float* s,
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

// Renamed from `norm` to `rlx_norm` to avoid a collision with CUDA's
// built-in `norm()` overload set, which lives at file scope under
// `extern "C"` linkage on CUDA 13+. NVRTC rejected the original name
// with: `more than one instance of overloaded function "norm" has "C"
// linkage`. Mirror the rename in `kernels/mod.rs` and the HIP-CPU
// launcher in `cpp/cpu_dispatch.cpp`.
extern "C" __global__ void rlx_norm(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int in_off,
    unsigned int out_off,
    unsigned int gamma_off,
    unsigned int beta_off,
    unsigned int eps_bits,
    unsigned int op
) {
    unsigned int row = blockIdx.x;
    if (row >= outer) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int in_base  = in_off  + row * inner;
    unsigned int out_base = out_off + row * inner;
    float n_inv = 1.0f / (float)inner;
    float eps = __int_as_float((int)eps_bits);

    __shared__ float s[LN_BLOCK];

    if (op == 0) {
        // LayerNorm: var = max(E[x²] − E[x]², 0) — matches CPU / wgpu / PyTorch
        // nn.LayerNorm (one read pass for moments, not two-pass (x−μ)²).
        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;
        for (unsigned int i = tid; i < inner; i += bsz) {
            float v = arena[in_base + i];
            local_sum += v;
            local_sum_sq += v * v;
        }
        float sum_x = ln_block_sum(local_sum, s, tid, bsz);
        __syncthreads();
        float sum_x2 = ln_block_sum(local_sum_sq, s, tid, bsz);
        float mean = sum_x * n_inv;
        float var = fmaxf(sum_x2 * n_inv - mean * mean, 0.0f);
        // Precise 1/sqrt — matches CPU `1.0/(var+eps).sqrt()` (not fast rsqrtf).
        float inv_std = 1.0f / sqrtf(var + eps);

        for (unsigned int i = tid; i < inner; i += bsz) {
            float g = arena[gamma_off + i];
            float b = arena[beta_off + i];
            arena[out_base + i] = (arena[in_base + i] - mean) * inv_std * g + b;
        }
    } else {
        // RmsNorm.
        float local_ss = 0.0f;
        for (unsigned int i = tid; i < inner; i += bsz) {
            float v = arena[in_base + i];
            local_ss += v * v;
        }
        float ss = ln_block_sum(local_ss, s, tid, bsz);
        float inv_rms = 1.0f / sqrtf(ss * n_inv + eps);

        for (unsigned int i = tid; i < inner; i += bsz) {
            float g = arena[gamma_off + i];
            arena[out_base + i] = arena[in_base + i] * inv_rms * g;
        }
    }
}
