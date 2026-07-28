// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Real INT8 `Op::QMatMul` matching `rlx_cpu::thunk::ops::quant::exec_q_mat_mul`.
//
//   x[M,K] i8, w[K,N] i8, bias[N] i32 → out[M,N] i8
//   out = clamp(round((bias + Σ (x−x_zp)(w−w_zp)) · mult) + out_zp, -128, 127)
//
// Arena layout (f32 buffer base):
//   - x / w / out: packed i8 at byte offsets (same as QuantizeI8 / DequantizeI8)
//   - bias: f32-lane I32 convention (value stored as float, cast to int) —
//     matches CUDA/ROCm Constant + set_param widening for DType::I32

__device__ __forceinline__ float round_half_away(float x) {
    float sgn = (x > 0.0f) - (x < 0.0f);
    return sgn * floorf(fabsf(x) + 0.5f);
}

extern "C" __global__ void q_matmul(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int x_byte_off,
    unsigned int w_byte_off,
    unsigned int bias_off,
    unsigned int out_byte_off,
    int x_zp,
    int w_zp,
    int out_zp,
    unsigned int mult_bits
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = m * n;
    if (idx >= total) return;

    unsigned int mi = idx / n;
    unsigned int ni = idx - mi * n;

    const signed char* x = reinterpret_cast<const signed char*>(arena) + x_byte_off;
    const signed char* w = reinterpret_cast<const signed char*>(arena) + w_byte_off;
    signed char* out = reinterpret_cast<signed char*>(arena) + out_byte_off;

    int acc = (int)truncf(arena[bias_off + ni]);
    for (unsigned int ki = 0u; ki < k; ++ki) {
        int xv = (int)x[mi * k + ki] - x_zp;
        int wv = (int)w[ki * n + ni] - w_zp;
        acc += xv * wv;
    }
    float mult = __uint_as_float(mult_bits);
    int r = (int)round_half_away((float)acc * mult) + out_zp;
    if (r < -128) r = -128;
    if (r > 127) r = 127;
    out[mi * n + ni] = (signed char)r;
}
