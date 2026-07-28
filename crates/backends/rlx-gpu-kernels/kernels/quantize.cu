// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// INT8 Quantize / Dequantize matching `rlx_cpu::thunk::ops::quant::{exec_quantize,
// exec_dequantize}` (asymmetric; symmetric is zp=0).
// Channel layout: c = (i / inner) % chan_dim (chan_dim==1 → c=0).
// Rounding: Rust `f32::round` (half away from zero).
//
// Affine table `affine[2*c + 0]` = scale bits (f32), `affine[2*c + 1]` = zp as i32 bits.
// I8 codes live at byte offsets into the f32 arena (1 byte/elem slots).

__device__ __forceinline__ float round_half_away(float x) {
    float sgn = (x > 0.0f) - (x < 0.0f);
    return sgn * floorf(fabsf(x) + 0.5f);
}

__device__ __forceinline__ unsigned int q_channel_of(
        unsigned int i, unsigned int chan_dim, unsigned int inner) {
    if (chan_dim <= 1u) return 0u;
    return (i / inner) % chan_dim;
}

// f32 → packed i8: q = clamp(round(x/scale) + zp, -128, 127)
extern "C" __global__ void quantize_i8(
    float* arena,
    unsigned int n,
    unsigned int chan_dim,
    unsigned int inner,
    unsigned int in_off,
    unsigned int q_byte_off,
    const unsigned int* affine
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int c = q_channel_of(i, chan_dim, inner);
    float s = __uint_as_float(affine[2u * c]);
    int zp = (int)affine[2u * c + 1u];
    float inv = 1.0f / s;
    float scaled = arena[in_off + i] * inv;
    int v = (int)round_half_away(scaled) + zp;
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    signed char* q = reinterpret_cast<signed char*>(arena) + q_byte_off;
    q[i] = (signed char)v;
}

// packed i8 → f32: out = (q - zp) * scale
extern "C" __global__ void dequantize_i8(
    float* arena,
    unsigned int n,
    unsigned int chan_dim,
    unsigned int inner,
    unsigned int q_byte_off,
    unsigned int out_off,
    const unsigned int* affine
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int c = q_channel_of(i, chan_dim, inner);
    float s = __uint_as_float(affine[2u * c]);
    int zp = (int)affine[2u * c + 1u];
    const signed char* q = reinterpret_cast<const signed char*>(arena) + q_byte_off;
    int qv = (int)q[i];
    arena[out_off + i] = (float)(qv - zp) * s;
}
