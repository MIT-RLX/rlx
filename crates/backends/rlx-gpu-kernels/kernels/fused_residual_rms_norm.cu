// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Fused (residual add + optional bias add) + RMSNorm.
//   y = rms_norm(x + residual + [bias], gamma, beta)
//
// Launch shape: grid=(outer,1,1), block=(256,1,1).

#define FRRN_BLOCK 256

__device__ __forceinline__ float frrn_block_sum(float v, float* s,
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

extern "C" __global__ void fused_residual_rms_norm(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int in_off,
    unsigned int residual_off,
    unsigned int bias_off,
    unsigned int gamma_off,
    unsigned int beta_off,
    unsigned int out_off,
    unsigned int eps_bits,
    unsigned int has_bias
) {
    unsigned int row = blockIdx.x;
    if (row >= outer) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int in_base  = in_off       + row * inner;
    unsigned int res_base = residual_off + row * inner;
    unsigned int out_base = out_off      + row * inner;
    float n_inv = 1.0f / (float)inner;
    float eps = __int_as_float((int)eps_bits);
    bool with_bias = has_bias != 0;

    __shared__ float s[FRRN_BLOCK];

    float local_ss = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        float v = arena[in_base + i] + arena[res_base + i];
        if (with_bias) v += arena[bias_off + i];
        arena[out_base + i] = v;
        local_ss += v * v;
    }
    float ss = frrn_block_sum(local_ss, s, tid, bsz);
    float inv_rms = 1.0f / sqrtf(ss * n_inv + eps);

    for (unsigned int i = tid; i < inner; i += bsz) {
        float g = arena[gamma_off + i];
        float b = arena[beta_off + i];
        arena[out_base + i] = arena[out_base + i] * inv_rms * g + b;
    }
}
