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

// f32 → FP8 encode done in closed form (no <cuda_fp8.h> / <hip/hip_fp8.h> — NVRTC
// and hipRTC have no include search path for the toolkit headers, so pulling
// them in fails the whole translation unit). This mirrors rlx-ir/src/lowp_codec.rs
// bit-for-bit, so the codes fed to the tensor-core GEMM match the CPU oracle
// exactly. ids: 0 e4m3 (OCP), 1 e5m2 (OCP), 2 e4m3 FNUZ, 3 e5m2 FNUZ.
__device__ __forceinline__ float rlx_fp8_decode(unsigned int fmt, unsigned int code) {
    unsigned int e_bits, m_bits;
    int bias;
    unsigned int fnuz = 0u, has_inf = 0u, e4m3ocp = 0u;
    switch (fmt) {
        case 0u: e_bits = 4u; m_bits = 3u; bias = 7;  e4m3ocp = 1u; break;
        case 1u: e_bits = 5u; m_bits = 2u; bias = 15; has_inf = 1u; break;
        case 2u: e_bits = 4u; m_bits = 3u; bias = 8;  fnuz = 1u; break;
        default: e_bits = 5u; m_bits = 2u; bias = 16; fnuz = 1u; break; // 3 e5m2fnuz
    }
    unsigned int width = e_bits + m_bits;
    unsigned int sign_bit = (code >> width) & 1u;
    unsigned int exp = (code >> m_bits) & ((1u << e_bits) - 1u);
    unsigned int mant = code & ((1u << m_bits) - 1u);
    float sign = sign_bit ? -1.0f : 1.0f;
    unsigned int max_exp = (1u << e_bits) - 1u;
    if (fnuz) {
        if (sign_bit && exp == 0u && mant == 0u) return nanf("");
    } else if (has_inf) {
        if (exp == max_exp) return mant == 0u ? sign * __int_as_float(0x7f800000) : nanf("");
    } else if (e4m3ocp) {
        if (exp == max_exp && mant == ((1u << m_bits) - 1u)) return nanf("");
    }
    float m_div = (float)(1u << m_bits);
    float val = (exp == 0u)
        ? ((float)mant / m_div) * exp2f((float)(1 - bias))
        : (1.0f + (float)mant / m_div) * exp2f((float)((int)exp - bias));
    return sign * val;
}

// Nearest-representable encode (round-half-to-even, saturating, NaN→0, ±inf →
// ±max_finite) by exhaustive search of the 256-code space — matches the oracle.
__device__ __forceinline__ unsigned char rlx_fp8_encode(unsigned int fmt, float x) {
    if (isnan(x)) return 0u;
    if (isinf(x)) {
        float mf = 0.0f;
        for (unsigned int c = 0u; c < 256u; ++c) {
            float v = fabsf(rlx_fp8_decode(fmt, c));
            if (isfinite(v)) mf = fmaxf(mf, v);
        }
        x = (x > 0.0f ? mf : -mf);
    }
    unsigned char best = 0u;
    double best_err = 1.0e300;
    unsigned char best_lsb = 1u;
    for (unsigned int c = 0u; c < 256u; ++c) {
        float v = rlx_fp8_decode(fmt, c);
        if (!isfinite(v)) continue;
        double err = fabs((double)v - (double)x);
        unsigned char lsb = (unsigned char)(c & 1u);
        if (err < best_err || (err == best_err && lsb < best_lsb)) {
            best_err = err;
            best = (unsigned char)c;
            best_lsb = lsb;
        }
    }
    return best;
}

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
    unsigned int id = (fmt == 0u) ? 2u : 3u;
#else
    unsigned int id = (fmt == 0u) ? 0u : 1u; // OCP e4m3 / e5m2
#endif
    out[i] = rlx_fp8_encode(id, v);
}
