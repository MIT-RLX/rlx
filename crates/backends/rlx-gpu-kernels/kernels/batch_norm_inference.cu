// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// BatchNormInference (channels-last). Matches CPU `batch_norm_inference*`
// and MLX last-axis lowering: frozen μ/σ²,
//   y = γ · x̂ + β,  x̂ = (x − μ) / √(σ² + ε)
//   dx = dy · γ · inv_std
//   dγ / dβ reduce over all non-channel axes.
// Layout: idx = row * channels + c  (row ∈ [0, count)).

// One thread per element.
extern "C" __global__ void batch_norm_inference(
    float* arena,
    unsigned int src_off,
    unsigned int g_off,
    unsigned int b_off,
    unsigned int mean_off,
    unsigned int var_off,
    unsigned int dst_off,
    unsigned int count,
    unsigned int channels,
    unsigned int eps_bits
) {
    unsigned int n = count * channels;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || channels == 0u) return;
    unsigned int c = i % channels;
    float eps = __int_as_float((int)eps_bits);
    float inv = 1.0f / sqrtf(arena[var_off + c] + eps);
    float xhat = (arena[src_off + i] - arena[mean_off + c]) * inv;
    arena[dst_off + i] = arena[g_off + c] * xhat + arena[b_off + c];
}

// One thread per element. Mean / x unused (frozen stats).
extern "C" __global__ void batch_norm_inference_bwd_input(
    float* arena,
    unsigned int gamma_off,
    unsigned int var_off,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int count,
    unsigned int channels,
    unsigned int eps_bits
) {
    unsigned int n = count * channels;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || channels == 0u) return;
    unsigned int c = i % channels;
    float eps = __int_as_float((int)eps_bits);
    float inv = 1.0f / sqrtf(arena[var_off + c] + eps);
    arena[out_off + i] = arena[dy_off + i] * arena[gamma_off + c] * inv;
}

// One thread per channel: dγ_c = Σ dy · x̂ over rows.
extern "C" __global__ void batch_norm_inference_bwd_gamma(
    float* arena,
    unsigned int x_off,
    unsigned int mean_off,
    unsigned int var_off,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int count,
    unsigned int channels,
    unsigned int eps_bits
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float eps = __int_as_float((int)eps_bits);
    float inv = 1.0f / sqrtf(arena[var_off + c] + eps);
    float mean = arena[mean_off + c];
    float acc = 0.0f;
    for (unsigned int row = 0u; row < count; ++row) {
        unsigned int idx = row * channels + c;
        float xhat = (arena[x_off + idx] - mean) * inv;
        acc += arena[dy_off + idx] * xhat;
    }
    arena[out_off + c] = acc;
}

// One thread per channel: dβ_c = Σ dy over rows.
extern "C" __global__ void batch_norm_inference_bwd_beta(
    float* arena,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int count,
    unsigned int channels
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float acc = 0.0f;
    for (unsigned int row = 0u; row < count; ++row) {
        acc += arena[dy_off + row * channels + c];
    }
    arena[out_off + c] = acc;
}
