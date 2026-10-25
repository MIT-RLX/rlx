// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Cast a contiguous f32 buffer to a u16 half-precision buffer
// (f16 / IEEE-754 binary16, or bf16 / Brain float). Used by the
// mixed-precision matmul path to convert activations on-the-fly when
// the matching weight is stored in the half-arena side-buffer.
//
// `dtype`:
//   0 = F16
//   1 = BF16

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void cast_f32_to_half(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    unsigned int n,
    unsigned int dtype
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    unsigned short out;
    if (dtype == 0u) {
        __half h = __float2half_rn(v);
        out = *reinterpret_cast<const unsigned short*>(&h);
    } else {
        __nv_bfloat16 h = __float2bfloat16_rn(v);
        out = *reinterpret_cast<const unsigned short*>(&h);
    }
    dst[i] = out;
}

// The reverse, for a weight kept at 2 B: widen `n` u16 lanes at `src` into
// f32 at `dst` (both raw device addresses inside the arena). Exact for both
// formats, so the GEMM that follows is the f32 matmul over the stored weight.
extern "C" __global__ void widen_half_to_f32(
    unsigned long long src_addr,
    unsigned long long dst_addr,
    unsigned long long n,
    unsigned int dtype
) {
    const unsigned short* src = reinterpret_cast<const unsigned short*>(src_addr);
    float* dst = reinterpret_cast<float*>(dst_addr);
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        unsigned short b = src[i];
        if (dtype == 0u) {
            __half h = *reinterpret_cast<const __half*>(&b);
            dst[i] = __half2float(h);
        } else {
            dst[i] = __uint_as_float(((unsigned int)b) << 16);
        }
    }
}
