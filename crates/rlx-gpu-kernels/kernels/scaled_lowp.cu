// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// Native low-precision (FP8) quantize producers for Op::ScaledMatMul.
//
// These feed the native tensor-core FP8 GEMM (cublasLt / hipBLASLt): activations
// are dynamically quantized to FP8 E4M3/E5M2 codes + a per-tensor f32 scale; the
// GEMM then consumes the codes directly with f32 accumulation and applies the
// scales via its A/B scale pointers. Per-tensor only — what cublasLt 12.3 /
// CDNA3 hipBLASLt support without block-scaled descriptors.
//
// Arena convention (matches dequant_matmul.cu): the single f32 arena buffer is
// passed as `float* arena`; f32 tensors are addressed by f32-element offset,
// u8 code tensors by byte offset via a `reinterpret_cast<unsigned char*>`. Each
// quantize thread writes exactly one distinct output byte, so byte stores into
// shared f32 words are race-free. Compiled at runtime by NVRTC / hipRTC.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_fp8.h>
#else
#include <cuda_fp8.h>
#endif

// Per-tensor amax → scale = amax / max_finite. One block, grid-stride load then
// a shared-memory tree reduction; thread 0 writes the single f32 scale.
extern "C" __global__ void scaled_quant_scale_per_tensor(
    float* __restrict__ arena,
    unsigned int x_off_f32,
    unsigned int scale_off_f32,
    unsigned int n,
    float max_finite)
{
    __shared__ float sdata[256];
    float local = 0.0f;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        local = fmaxf(local, fabsf(arena[x_off_f32 + i]));
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float amax = sdata[0];
        arena[scale_off_f32] = amax > 0.0f ? amax / max_finite : 1.0f;
    }
}

// Encode arena[x] / scale to FP8 codes. fmt: 0 = E4M3, 1 = E5M2. Round-to-
// nearest, saturating (the hardware default), matching the rlx-cpu oracle.
extern "C" __global__ void scaled_quantize_fp8_per_tensor(
    float* __restrict__ arena,
    unsigned int x_off_f32,
    unsigned int scale_off_f32,
    unsigned int out_byte_off,
    unsigned int n,
    unsigned int fmt)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = arena[scale_off_f32];
    float v = (s != 0.0f) ? (arena[x_off_f32 + i] / s) : 0.0f;
    unsigned char* out = reinterpret_cast<unsigned char*>(arena) + out_byte_off;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    // CDNA3 hipBLASLt FP8 is "FNUZ"; CDNA4 supports OCP. Use FNUZ here.
    __hip_fp8_storage_t q =
        __hip_cvt_float_to_fp8(v, __HIP_SATFINITE,
                               fmt == 0u ? __HIP_E4M3_FNUZ : __HIP_E5M2_FNUZ);
    out[i] = (unsigned char)q;
#else
    if (fmt == 0u) {
        __nv_fp8_e4m3 q(v);
        out[i] = *reinterpret_cast<const unsigned char*>(&q);
    } else {
        __nv_fp8_e5m2 q(v);
        out[i] = *reinterpret_cast<const unsigned char*>(&q);
    }
#endif
}
