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

// Compiled by NVRTC (CUDA) and by hipRTC (ROCm), which has no `cuda_*.h`.
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

// f32 -> bf16, round to nearest even (NaN stays a quiet NaN).
__device__ __forceinline__ unsigned short rlx_f32_to_bf16_rn(float v) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
    unsigned int u = __float_as_uint(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (unsigned short)((u >> 16) | 0x0040u);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
#else
    __nv_bfloat16 h = __float2bfloat16_rn(v);
    return *reinterpret_cast<const unsigned short*>(&h);
#endif
}

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
        out = rlx_f32_to_bf16_rn(v);
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
