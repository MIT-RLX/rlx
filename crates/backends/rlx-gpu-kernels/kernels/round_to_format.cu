// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Op::RoundToFormat: round to a binary float format emulated on f32, with an
// optional power-of-two scale. Bit-identical to
// `rlx_ir::FloatFormat::round_slice_scaled`. Exponent, NaN/inf and subnormal
// decisions read the f32 bits (independent of -ftz / fast-math); ldexpf and
// rintf (ties to even) are exact here. Shared by CUDA (NVRTC) and ROCm (HIP).
// One thread per element, or per block when mode == 2. A block's scale follows
// `FloatFormat::block_scale`: the squared errors are summed in index order with
// `__fmul_rn` / `__fadd_rn`, which are never fused into an FMA, so the choice
// matches the reference.

__device__ __forceinline__ int rtf_exp(float a) {
    return (int)((__float_as_uint(a) >> 23) & 0xffu) - 127;
}

__device__ __forceinline__ float rtf_flush(float y) {
    unsigned int b = __float_as_uint(y);
    return (b & 0x7fffffffu) < 0x00800000u ? __uint_as_float(b & 0x80000000u) : y;
}

__device__ __forceinline__ float rtf_round(float x, int man_bits, int emin, unsigned int max_bits) {
    unsigned int bits = __float_as_uint(x);
    unsigned int ab = bits & 0x7fffffffu;
    unsigned int sgn = bits & 0x80000000u;
    if (ab == 0u || ab > 0x7f800000u) return x;
    if (ab < 0x00800000u) return __uint_as_float(sgn);
    if (ab >= max_bits) return __uint_as_float(sgn | max_bits);
    float ax = __uint_as_float(ab);
    int e = rtf_exp(ax);
    int shift = man_bits - (e > emin ? e : emin);
    unsigned int qb = __float_as_uint(ldexpf(rintf(ldexpf(ax, shift)), -shift));
    if (qb > max_bits) qb = max_bits;
    return __uint_as_float(sgn | (qb < 0x00800000u ? 0u : qb));
}

// Bits of |x|, or 0 for NaN, so NaN never wins a maximum.
__device__ __forceinline__ unsigned int rtf_abs_bits(float x) {
    unsigned int b = __float_as_uint(x) & 0x7fffffffu;
    return b > 0x7f800000u ? 0u : b;
}

__device__ __forceinline__ int rtf_scale_exp(unsigned int amax_bits, int emax) {
    if (amax_bits == 0u || amax_bits >= 0x7f800000u) return 0;
    int e = (int)(amax_bits >> 23) - 127 - emax;
    return e < -126 ? -126 : (e > 127 ? 127 : e);
}

__device__ __forceinline__ float rtf_scaled(float x, int e, int man_bits, int emin, unsigned int max_bits) {
    return rtf_flush(ldexpf(rtf_round(ldexpf(x, -e), man_bits, emin, max_bits), e));
}

__device__ __forceinline__ float rtf_sq_err(const float* x, unsigned int start, unsigned int end, int e,
                                            int man_bits, int emin, unsigned int max_bits) {
    float acc = 0.0f;
    for (unsigned int i = start; i < end; i++) {
        float d = __fsub_rn(rtf_scaled(x[i], e, man_bits, emin, max_bits), x[i]);
        acc = __fadd_rn(acc, __fmul_rn(d, d));
    }
    return acc;
}

// `FloatFormat::fine_scale`: the smallest (1 + m/8)·2^k with s·max ≥ amax. Every
// product is exact, so this is the same on every backend.
__device__ __forceinline__ float rtf_fine_scale(float amax, int emax, float max_normal) {
    int k0 = rtf_exp(amax) - emax - 1;
    k0 = k0 < -126 ? -126 : (k0 > 125 ? 125 : k0);
    for (int k = k0; k <= k0 + 2; k++) {
        float p = __uint_as_float((unsigned int)(k + 127) << 23);
        for (int m = 0; m < 8; m++) {
            float s = __fmul_rn(1.0f + 0.125f * (float)m, p);
            if (__fmul_rn(s, max_normal) >= amax) return s;
        }
    }
    return __uint_as_float(0x7f700000u);
}

// `fine_reciprocal`: 1/s from the 8-entry table (`FINE_RECIPROCALS`) and the exponent.
__device__ __forceinline__ float rtf_fine_rcp(float s) {
    const unsigned int tab[8] = {0x3f800000u, 0x3f638e39u, 0x3f4ccccdu, 0x3f3a2e8cu,
                                 0x3f2aaaabu, 0x3f1d89d9u, 0x3f124925u, 0x3f088889u};
    unsigned int b = __float_as_uint(s);
    int ne = 127 - (int)((b >> 23) & 0xffu);
    if (ne < -126) return 0.0f;
    if (ne > 127) ne = 127;
    return __fmul_rn(__uint_as_float(tab[(b >> 20) & 7u]), __uint_as_float((unsigned int)(ne + 127) << 23));
}

extern "C" __global__ void round_to_format(
    float* arena,
    unsigned int n,
    unsigned int last_dim,
    unsigned int block,
    unsigned int mode,
    int man_bits,
    int emin,
    int emax,
    unsigned int max_bits,
    unsigned int in_off,
    unsigned int amax_off,
    unsigned int out_off
) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    const float* x = arena + in_off;
    float* out = arena + out_off;
    if (mode >= 2u) {
        unsigned int per_row = (last_dim + block - 1u) / block;
        unsigned int row = t / per_row;
        unsigned int start = row * last_dim + (t % per_row) * block;
        if (start >= n) return;
        unsigned int end = start + block;
        unsigned int row_end = row * last_dim + last_dim;
        if (end > row_end) end = row_end;
        unsigned int m = 0u;
        for (unsigned int i = start; i < end; i++) {
            unsigned int b = rtf_abs_bits(x[i]);
            if (b > m) m = b;
        }
        if (mode == 3u) {
            float s = 1.0f, r = 1.0f;
            if (m != 0u && m < 0x7f800000u) {
                s = rtf_fine_scale(__uint_as_float(m), emax, __uint_as_float(max_bits));
                r = rtf_fine_rcp(s);
            }
            for (unsigned int i = start; i < end; i++) {
                out[i] = rtf_flush(__fmul_rn(rtf_round(__fmul_rn(x[i], r), man_bits, emin, max_bits), s));
            }
            return;
        }
        int e = rtf_scale_exp(m, emax);
        // One binade up when the MX scale clips the block max and that rounds
        // the block with a smaller squared error (ties keep the MX scale).
        if (m != 0u && m < 0x7f800000u && e < 127
            && __float_as_uint(ldexpf(__uint_as_float(m), -e)) > max_bits
            && rtf_sq_err(x, start, end, e + 1, man_bits, emin, max_bits)
                   < rtf_sq_err(x, start, end, e, man_bits, emin, max_bits)) {
            e += 1;
        }
        for (unsigned int i = start; i < end; i++) {
            out[i] = rtf_scaled(x[i], e, man_bits, emin, max_bits);
        }
        return;
    }
    if (t >= n) return;
    int e = mode == 1u ? rtf_scale_exp(rtf_abs_bits(arena[amax_off]), emax) : 0;
    out[t] = rtf_flush(ldexpf(rtf_round(ldexpf(x[t], -e), man_bits, emin, max_bits), e));
}
