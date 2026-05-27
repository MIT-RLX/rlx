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
        // Phase 1: mean.
        float local_sum = 0.0f;
        for (unsigned int i = tid; i < inner; i += bsz) {
            local_sum += arena[in_base + i];
        }
        float mean = ln_block_sum(local_sum, s, tid, bsz) * n_inv;

        // Phase 2: variance.
        float local_var = 0.0f;
        for (unsigned int i = tid; i < inner; i += bsz) {
            float d = arena[in_base + i] - mean;
            local_var += d * d;
        }
        float var = ln_block_sum(local_var, s, tid, bsz);
        float inv_std = rsqrtf(var * n_inv + eps);

        // Phase 3: normalize.
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
        float inv_rms = rsqrtf(ss * n_inv + eps);

        for (unsigned int i = tid; i < inner; i += bsz) {
            float g = arena[gamma_off + i];
            arena[out_base + i] = arena[in_base + i] * inv_rms * g;
        }
    }
}
