// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// General (all-format, all-scale-layout) low-precision quantize + GEMM for
// Op::ScaledMatMul on CUDA / ROCm. This is the **decode-and-accumulate
// reference on GPU cores** — NOT tensor-core native. It's the path for formats
// the FP8 tensor-core GEMM can't do on the current toolkit (block-scaled MX,
// FP4 NVFP4/MXFP4, FP6), so those graphs still run on-device instead of
// erroring. Per-tensor FP8 keeps using the native cublasLt/hipBLASLt path.
//
// All decode/encode logic mirrors rlx-ir/src/lowp_codec.rs bit-for-bit (the
// CPU oracle every backend is checked against). Arena convention matches
// dequant_matmul.cu: f32 arena base + f32-element offsets for f32 tensors,
// byte offsets (via reinterpret_cast<unsigned char*>) for U8 code/scale tensors.

// Format ids: 0 e4m3, 1 e5m2, 2 e4m3fnuz, 3 e5m2fnuz, 4 e2m3, 5 e3m2, 6 e2m1.
// Scale modes: 0 per-tensor (f32), 1 block E8M0 (u8), 2 NVFP4 E4M3 (u8).

__device__ __forceinline__ float rlx_decode_lowp(unsigned int fmt, unsigned int code) {
    if (fmt == 6u) { // FP4 E2M1 LUT
        const float lut[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                               -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
        return lut[code & 0xFu];
    }
    unsigned int e_bits, m_bits;
    int bias;
    unsigned int fnuz = 0u, has_inf = 0u, e4m3ocp = 0u;
    switch (fmt) {
        case 0u: e_bits = 4u; m_bits = 3u; bias = 7;  e4m3ocp = 1u; break;
        case 1u: e_bits = 5u; m_bits = 2u; bias = 15; has_inf = 1u; break;
        case 2u: e_bits = 4u; m_bits = 3u; bias = 8;  fnuz = 1u; break;
        case 3u: e_bits = 5u; m_bits = 2u; bias = 16; fnuz = 1u; break;
        case 4u: e_bits = 2u; m_bits = 3u; bias = 1;  break; // e2m3 (finite)
        case 5u: e_bits = 3u; m_bits = 2u; bias = 3;  break; // e3m2 (finite)
        default: return 0.0f;
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
        if (exp == max_exp) return mant == 0u ? sign * INFINITY : nanf("");
    } else if (e4m3ocp) {
        if (exp == max_exp && mant == ((1u << m_bits) - 1u)) return nanf("");
    }
    float m_div = (float)(1u << m_bits);
    float val;
    if (exp == 0u) {
        val = ((float)mant / m_div) * exp2f((float)(1 - bias));
    } else {
        val = (1.0f + (float)mant / m_div) * exp2f((float)((int)exp - bias));
    }
    return sign * val;
}

// Nearest-representable encode by exhaustive search of the code space (≤256) —
// simple and exact, round-half-to-even, saturating, NaN→0 (matches the oracle).
__device__ __forceinline__ unsigned char rlx_encode_lowp(unsigned int fmt, float x) {
    if (isnan(x)) return 0u;
    unsigned int width = (fmt == 6u) ? 4u : ((fmt == 4u || fmt == 5u) ? 6u : 8u);
    unsigned int n_codes = 1u << width;
    unsigned char best = 0u;
    double best_err = 1.0e300;
    unsigned char best_lsb = 1u;
    for (unsigned int c = 0u; c < n_codes; ++c) {
        float v = rlx_decode_lowp(fmt, (unsigned char)c);
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

__device__ __forceinline__ float rlx_e8m0(unsigned char b) {
    return b == 0xFFu ? nanf("") : exp2f((float)((int)b - 127));
}

__device__ __forceinline__ unsigned char rlx_f32_to_e8m0(float s) {
    if (!(s > 0.0f) || !isfinite(s)) return 0u;
    int e = (int)ceilf(log2f(s)) + 127;
    if (e < 0) e = 0;
    if (e > 254) e = 254;
    return (unsigned char)e;
}

// Largest finite magnitude of a format (for amax→scale).
__device__ __forceinline__ float rlx_max_finite(unsigned int fmt) {
    switch (fmt) {
        case 0u: return 448.0f;
        case 1u: return 57344.0f;
        case 2u: return 240.0f;
        case 3u: return 57344.0f;
        case 4u: return 7.5f;
        case 5u: return 28.0f;
        default: return 6.0f; // e2m1
    }
}

// Per-row block (or per-tensor) amax → scale; stores f32 (per-tensor) or u8
// (E8M0 / NVFP4-E4M3) snapped scale. One thread per scale element.
extern "C" __global__ void scaled_quant_scale_general(
    float* __restrict__ arena,
    unsigned int x_off_f32,
    unsigned int scale_byte_off,
    unsigned int rows,
    unsigned int cols,
    unsigned int fmt,
    unsigned int scale_mode,
    unsigned int block)
{
    unsigned int nblk = (scale_mode == 0u) ? 1u : ((cols + block - 1u) / block);
    unsigned int total = (scale_mode == 0u) ? 1u : rows * nblk;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float maxf = rlx_max_finite(fmt);

    float amax = 0.0f;
    if (scale_mode == 0u) {
        for (unsigned int i = 0u; i < rows * cols; ++i) {
            amax = fmaxf(amax, fabsf(arena[x_off_f32 + i]));
        }
    } else {
        unsigned int r = idx / nblk, b = idx % nblk;
        unsigned int lo = b * block, hi = min(lo + block, cols);
        for (unsigned int c = lo; c < hi; ++c) {
            amax = fmaxf(amax, fabsf(arena[x_off_f32 + r * cols + c]));
        }
    }
    float s = amax > 0.0f ? amax / maxf : 1.0f;
    if (scale_mode == 0u) {
        arena[scale_byte_off / 4u] = s; // per-tensor f32
    } else {
        unsigned char* out = reinterpret_cast<unsigned char*>(arena) + scale_byte_off;
        out[idx] = (scale_mode == 1u) ? rlx_f32_to_e8m0(s)
                                      : rlx_encode_lowp(0u, s); // NVFP4 E4M3 scale
    }
}

// Quantize x / scale(block) → codes for any format / scale layout.
extern "C" __global__ void scaled_quantize_general(
    float* __restrict__ arena,
    unsigned int x_off_f32,
    unsigned int scale_byte_off,
    unsigned int out_byte_off,
    unsigned int rows,
    unsigned int cols,
    unsigned int fmt,
    unsigned int scale_mode,
    unsigned int block)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int r = i / cols, c = i % cols;
    unsigned int nblk = (scale_mode == 0u) ? 1u : ((cols + block - 1u) / block);
    float s;
    if (scale_mode == 0u) {
        s = arena[scale_byte_off / 4u];
    } else {
        const unsigned char* sb = reinterpret_cast<const unsigned char*>(arena) + scale_byte_off;
        unsigned int si = r * nblk + c / block;
        s = (scale_mode == 1u) ? rlx_e8m0(sb[si]) : rlx_decode_lowp(0u, sb[si]);
    }
    float v = (s != 0.0f) ? (arena[x_off_f32 + i] / s) : 0.0f;
    unsigned char* out = reinterpret_cast<unsigned char*>(arena) + out_byte_off;
    out[i] = rlx_encode_lowp(fmt, v);
}

// Dequantize: codes → f32 via decode(code) * scale(block). The exact inverse of
// scaled_quantize_general; one thread per element. Used by the ScaledMatMul
// backward (straight-through QAT) to rebuild f32 operands, and as a standalone
// dequantizer.
extern "C" __global__ void scaled_dequantize_general(
    float* __restrict__ arena,
    unsigned int codes_byte_off,
    unsigned int scale_byte_off,
    unsigned int out_off_f32,
    unsigned int rows,
    unsigned int cols,
    unsigned int fmt,
    unsigned int scale_mode,
    unsigned int block)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int r = i / cols, c = i % cols;
    unsigned int nblk = (scale_mode == 0u) ? 1u : ((cols + block - 1u) / block);
    float s;
    if (scale_mode == 0u) {
        s = arena[scale_byte_off / 4u];
    } else {
        const unsigned char* sb = reinterpret_cast<const unsigned char*>(arena) + scale_byte_off;
        unsigned int si = r * nblk + c / block;
        s = (scale_mode == 1u) ? rlx_e8m0(sb[si]) : rlx_decode_lowp(0u, sb[si]);
    }
    const unsigned char* codes = reinterpret_cast<const unsigned char*>(arena) + codes_byte_off;
    arena[out_off_f32 + i] = rlx_decode_lowp(fmt, codes[i]) * s;
}

// Decode-and-accumulate GEMM (TN: lhs[m,k]·rhs[n,k]ᵀ → out[m,n]), one thread
// per output element. The non-tensor-core fallback for formats cublasLt can't do.
extern "C" __global__ void scaled_matmul_decode(
    float* __restrict__ arena,
    unsigned int lhs_byte_off,
    unsigned int rhs_byte_off,
    unsigned int lhs_scale_byte_off,
    unsigned int rhs_scale_byte_off,
    unsigned int out_off_f32,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int lhs_fmt,
    unsigned int rhs_fmt,
    unsigned int scale_mode,
    unsigned int block,
    unsigned int has_bias,
    unsigned int bias_off_f32)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;
    const unsigned char* lhs = reinterpret_cast<const unsigned char*>(arena) + lhs_byte_off;
    const unsigned char* rhs = reinterpret_cast<const unsigned char*>(arena) + rhs_byte_off;
    const unsigned char* lsb = reinterpret_cast<const unsigned char*>(arena) + lhs_scale_byte_off;
    const unsigned char* rsb = reinterpret_cast<const unsigned char*>(arena) + rhs_scale_byte_off;
    unsigned int nblk = (scale_mode == 0u) ? 1u : ((k + block - 1u) / block);
    float ls0 = arena[lhs_scale_byte_off / 4u];
    float rs0 = arena[rhs_scale_byte_off / 4u];

    float acc = 0.0f;
    for (unsigned int p = 0u; p < k; ++p) {
        float ls, rs;
        if (scale_mode == 0u) {
            ls = ls0;
            rs = rs0;
        } else {
            unsigned int li = i * nblk + p / block, ri = j * nblk + p / block;
            if (scale_mode == 1u) { ls = rlx_e8m0(lsb[li]); rs = rlx_e8m0(rsb[ri]); }
            else { ls = rlx_decode_lowp(0u, lsb[li]); rs = rlx_decode_lowp(0u, rsb[ri]); }
        }
        float a = rlx_decode_lowp(lhs_fmt, lhs[i * k + p]) * ls;
        float b = rlx_decode_lowp(rhs_fmt, rhs[j * k + p]) * rs;
        acc += a * b;
    }
    if (has_bias) acc += arena[bias_off_f32 + j];
    arena[out_off_f32 + i * n + j] = acc;
}
