// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// RLX — versatile ML compiler + runtime.
// Port of gpu-fft butterfly kernels for RLX 2N real-block layout:
// each row is [re[0..n) | im[0..n)] in the arena (f32 elements).
// Use ____cosf/____sinf so NVRTC does not need a host math.h include path.

__device__ inline float fft_re(float* a, unsigned base, unsigned k, unsigned n) {
    return a[base + k];
}
__device__ inline float fft_im(float* a, unsigned base, unsigned k, unsigned n) {
    return a[base + n + k];
}
__device__ inline void fft_set_re(float* a, unsigned base, unsigned k, unsigned n, float v) {
    a[base + k] = v;
}
__device__ inline void fft_set_im(float* a, unsigned base, unsigned k, unsigned n, float v) {
    a[base + n + k] = v;
}

// Bit-reverse permute one row (2N block). One thread per element.
extern "C" __global__ void fft_bit_reverse(
    float* arena,
    unsigned off,
    unsigned n,
    unsigned log2n,
    unsigned outer
) {
    unsigned row = blockIdx.y + blockIdx.z * gridDim.y;
    if (row >= outer) return;
    unsigned base = off + row * 2u * n;
    unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    unsigned rev = __brev(k) >> (32u - log2n);
    if (k >= rev) return;
    float tr = fft_re(arena, base, k, n);
    float ti = fft_im(arena, base, k, n);
    fft_set_re(arena, base, k, n, fft_re(arena, base, rev, n));
    fft_set_im(arena, base, k, n, fft_im(arena, base, rev, n));
    fft_set_re(arena, base, rev, n, tr);
    fft_set_im(arena, base, rev, n, ti);
}

// Fused inner stages in shared memory (tile = min(n, 1024)).
extern "C" __global__ void fft_inner(
    float* arena,
    unsigned off,
    unsigned n,
    unsigned tile,
    unsigned stages,
    unsigned inverse,
    float norm_scale,
    unsigned outer
) {
    extern __shared__ float smem[];
    float* sre = smem;
    float* sim = smem + tile;
    unsigned half_tile = tile / 2u;

    unsigned row = blockIdx.y + blockIdx.z * gridDim.y;
    if (row >= outer) return;
    unsigned row_base = off + row * 2u * n;

    unsigned tile_id = blockIdx.x;
    unsigned local = threadIdx.x;
    if (local >= half_tile) return;
    unsigned num_tiles = (n + tile - 1u) / tile;
    if (tile_id >= num_tiles) return;
    unsigned tile_base = tile_id * tile;

    if (local + half_tile < tile && tile_base + local + half_tile < n) {
        sre[local] = fft_re(arena, row_base, tile_base + local, n);
        sre[local + half_tile] = fft_re(arena, row_base, tile_base + local + half_tile, n);
        sim[local] = fft_im(arena, row_base, tile_base + local, n);
        sim[local + half_tile] = fft_im(arena, row_base, tile_base + local + half_tile, n);
    } else {
        if (tile_base + local < n) {
            sre[local] = fft_re(arena, row_base, tile_base + local, n);
            sim[local] = fft_im(arena, row_base, tile_base + local, n);
        }
        if (tile_base + local + half_tile < n) {
            sre[local + half_tile] = fft_re(arena, row_base, tile_base + local + half_tile, n);
            sim[local + half_tile] = fft_im(arena, row_base, tile_base + local + half_tile, n);
        }
    }
    __syncthreads();

    float sign = inverse ? 1.0f : -1.0f;
    for (unsigned s = 0; s < stages; ++s) {
        unsigned hs = 1u << s;
        unsigned k = local % hs;
        unsigned i = (local / hs) * (hs * 2u) + k;
        unsigned j = i + hs;
        float angle = sign * 3.14159265358979323846f * (float)k / (float)hs;
        float cos_a, sin_a;
        __sincosf(angle, &sin_a, &cos_a);
        float ur = sre[i], ui = sim[i];
        float vr = cos_a * sre[j] - sin_a * sim[j];
        float vi = sin_a * sre[j] + cos_a * sim[j];
        sre[i] = ur + vr; sim[i] = ui + vi;
        sre[j] = ur - vr; sim[j] = ui - vi;
        __syncthreads();
    }

    if (local + half_tile < tile && tile_base + local + half_tile < n) {
        float sr = sre[local] * norm_scale;
        float si = sim[local] * norm_scale;
        float sr2 = sre[local + half_tile] * norm_scale;
        float si2 = sim[local + half_tile] * norm_scale;
        fft_set_re(arena, row_base, tile_base + local, n, sr);
        fft_set_im(arena, row_base, tile_base + local, n, si);
        fft_set_re(arena, row_base, tile_base + local + half_tile, n, sr2);
        fft_set_im(arena, row_base, tile_base + local + half_tile, n, si2);
    } else {
        if (tile_base + local < n) {
            fft_set_re(arena, row_base, tile_base + local, n, sre[local] * norm_scale);
            fft_set_im(arena, row_base, tile_base + local, n, sim[local] * norm_scale);
        }
        if (tile_base + local + half_tile < n) {
            fft_set_re(arena, row_base, tile_base + local + half_tile, n,
                       sre[local + half_tile] * norm_scale);
            fft_set_im(arena, row_base, tile_base + local + half_tile, n,
                       sim[local + half_tile] * norm_scale);
        }
    }
}

// Radix-4 outer stage (one row per blockIdx.y).
extern "C" __global__ void fft_outer_r4(
    float* arena,
    unsigned off,
    unsigned n,
    unsigned q,
    unsigned inverse,
    float norm_scale,
    unsigned outer
) {
    unsigned row = blockIdx.y + blockIdx.z * gridDim.y;
    if (row >= outer) return;
    unsigned base = off + row * 2u * n;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 4u) return;

    unsigned k = tid % q;
    unsigned group = tid / q;
    unsigned p = group * (q * 4u) + k;

    float ar = fft_re(arena, base, p, n);
    float ai = fft_im(arena, base, p, n);
    float br = fft_re(arena, base, p + q, n);
    float bi = fft_im(arena, base, p + q, n);
    float cr = fft_re(arena, base, p + q * 2u, n);
    float ci = fft_im(arena, base, p + q * 2u, n);
    float dr = fft_re(arena, base, p + q * 3u, n);
    float di = fft_im(arena, base, p + q * 3u, n);

    float sign = inverse ? 1.0f : -1.0f;
    float neg_sign = inverse ? -1.0f : 1.0f;
    float angle1 = sign * 3.14159265358979323846f * (float)k / (float)q;
    float cos1, sin1;
    __sincosf(angle1, &sin1, &cos1);
    float w1b_r = cos1 * br - sin1 * bi;
    float w1b_i = sin1 * br + cos1 * bi;
    float w1d_r = cos1 * dr - sin1 * di;
    float w1d_i = sin1 * dr + cos1 * di;

    float u0r = ar + w1b_r, u0i = ai + w1b_i;
    float u1r = ar - w1b_r, u1i = ai - w1b_i;
    float u2r = cr + w1d_r, u2i = ci + w1d_i;
    float u3r = cr - w1d_r, u3i = ci - w1d_i;

    float angle2a = sign * 3.14159265358979323846f * (float)k / (float)(q * 2u);
    float cos2a, sin2a;
    __sincosf(angle2a, &sin2a, &cos2a);
    float cos2b = neg_sign * sin2a;
    float sin2b = sign * cos2a;

    float w2a_u2r = cos2a * u2r - sin2a * u2i;
    float w2a_u2i = sin2a * u2r + cos2a * u2i;
    float w2b_u3r = cos2b * u3r - sin2b * u3i;
    float w2b_u3i = sin2b * u3r + cos2b * u3i;

    fft_set_re(arena, base, p, n, (u0r + w2a_u2r) * norm_scale);
    fft_set_im(arena, base, p, n, (u0i + w2a_u2i) * norm_scale);
    fft_set_re(arena, base, p + q * 2u, n, (u0r - w2a_u2r) * norm_scale);
    fft_set_im(arena, base, p + q * 2u, n, (u0i - w2a_u2i) * norm_scale);
    fft_set_re(arena, base, p + q, n, (u1r + w2b_u3r) * norm_scale);
    fft_set_im(arena, base, p + q, n, (u1i + w2b_u3i) * norm_scale);
    fft_set_re(arena, base, p + q * 3u, n, (u1r - w2b_u3r) * norm_scale);
    fft_set_im(arena, base, p + q * 3u, n, (u1i - w2b_u3i) * norm_scale);
}

// Trailing radix-2 outer stage.
extern "C" __global__ void fft_outer_r2(
    float* arena,
    unsigned off,
    unsigned n,
    unsigned half_stride,
    unsigned inverse,
    float norm_scale,
    unsigned outer
) {
    unsigned row = blockIdx.y + blockIdx.z * gridDim.y;
    if (row >= outer) return;
    unsigned base = off + row * 2u * n;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2u) return;

    unsigned k = tid % half_stride;
    unsigned i = (tid / half_stride) * (half_stride * 2u) + k;
    unsigned j = i + half_stride;

    float sign = inverse ? 1.0f : -1.0f;
    float angle = sign * 3.14159265358979323846f * (float)k / (float)half_stride;
    float cos_a, sin_a;
    __sincosf(angle, &sin_a, &cos_a);

    float ur = fft_re(arena, base, i, n);
    float ui = fft_im(arena, base, i, n);
    float vr = cos_a * fft_re(arena, base, j, n) - sin_a * fft_im(arena, base, j, n);
    float vi = sin_a * fft_re(arena, base, j, n) + cos_a * fft_im(arena, base, j, n);

    fft_set_re(arena, base, i, n, (ur + vr) * norm_scale);
    fft_set_im(arena, base, i, n, (ui + vi) * norm_scale);
    fft_set_re(arena, base, j, n, (ur - vr) * norm_scale);
    fft_set_im(arena, base, j, n, (ui - vi) * norm_scale);
}

// Single-kernel radix-2 for n <= 1024 (bit-reverse load + all stages).
extern "C" __global__ void fft_radix2_full(
    float* arena,
    unsigned src_off,
    unsigned dst_off,
    unsigned n,
    unsigned log2n,
    unsigned inverse,
    float norm_scale,
    unsigned outer
) {
    extern __shared__ float smem[];
    float* sre = smem;
    float* sim = smem + 1024u;

    unsigned row = blockIdx.y + blockIdx.z * gridDim.y;
    if (row >= outer) return;
    unsigned src_base = src_off + row * 2u * n;
    unsigned dst_base = dst_off + row * 2u * n;
    unsigned tid = threadIdx.x;
    unsigned tg = blockDim.x;

    for (unsigned k = tid; k < n; k += tg) {
        unsigned rev = __brev(k) >> (32u - log2n);
        sre[rev] = fft_re(arena, src_base, k, n);
        sim[rev] = fft_im(arena, src_base, k, n);
    }
    __syncthreads();

    float sign = inverse ? 1.0f : -1.0f;
    for (unsigned len = 2u; len <= n; len <<= 1u) {
        unsigned h2 = len >> 1u;
        float theta_base = sign * 6.28318530717958647692f / (float)len;
        for (unsigned b = tid; b < n / 2u; b += tg) {
            unsigned group = b / h2;
            unsigned k_in = b % h2;
            unsigned i_lo = group * len + k_in;
            unsigned i_hi = i_lo + h2;
            float theta = theta_base * (float)k_in;
            float wre, wim;
            __sincosf(theta, &wim, &wre);
            float t_re = wre * sre[i_hi] - wim * sim[i_hi];
            float t_im = wre * sim[i_hi] + wim * sre[i_hi];
            float u_re = sre[i_lo];
            float u_im = sim[i_lo];
            sre[i_lo] = u_re + t_re;
            sim[i_lo] = u_im + t_im;
            sre[i_hi] = u_re - t_re;
            sim[i_hi] = u_im - t_im;
        }
        __syncthreads();
    }

    for (unsigned k = tid; k < n; k += tg) {
        fft_set_re(arena, dst_base, k, n, sre[k] * norm_scale);
        fft_set_im(arena, dst_base, k, n, sim[k] * norm_scale);
    }
}

// ── cuFFT bridge: RLX 2N planar [re|im] block ⇄ interleaved float2 ──
// cuFFT operates on interleaved cufftComplex; the RLX arena stores each row as
// [re[0..n) | im[0..n)]. These two kernels convert in/out of a scratch buffer
// (laid out as 2 f32 per complex element, i.e. cufftComplex). Used only when
// the `cufft` feature is on.

// Planar arena → interleaved scratch (one f32 pair per element).
extern "C" __global__ void fft_pack_interleave(
    const float* arena, float* scratch, unsigned off, unsigned n, unsigned outer
) {
    unsigned row = blockIdx.y + blockIdx.z * gridDim.y;
    if (row >= outer) return;
    unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    unsigned base = off + row * 2u * n;
    unsigned d = (row * n + k) * 2u;
    scratch[d]      = arena[base + k];        // re
    scratch[d + 1u] = arena[base + n + k];    // im
}

// Interleaved scratch → planar arena, applying the FFT norm scale.
extern "C" __global__ void fft_unpack_planar(
    float* arena, const float* scratch, unsigned off, unsigned n, unsigned outer,
    float norm_scale
) {
    unsigned row = blockIdx.y + blockIdx.z * gridDim.y;
    if (row >= outer) return;
    unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    unsigned base = off + row * 2u * n;
    unsigned d = (row * n + k) * 2u;
    arena[base + k]     = scratch[d]      * norm_scale;   // re
    arena[base + n + k] = scratch[d + 1u] * norm_scale;   // im
}

// ── native-cuda-fft: register/shared Stockham, planar I/O ──────────────
// Single-block-per-FFT, mixed-radix Cooley-Tukey that minimizes DRAM passes.
// A Stockham *autosort* FFT needs no bit-reversal: it ping-pongs two shared
// buffers, gathering naturally each stage and emitting natural-order output.
// We read the RLX 2N planar block
// [re|im] straight into interleaved shared float2 and write it back planar —
// the layout conversion is folded into the single load/store, so unlike the
// `cufft` bridge there is NO extra conversion pass. One block per row
// (blockIdx.x over `outer`). Dynamic shared = 2·n float2 (16·n bytes; the
// caller opt-ins >48 KB via cuFuncSetAttribute).
//
// `inverse` flips the twiddle sign (and, for radix-4, the ±i core rotation via
// `rs`); `norm_scale` is the FFT normalization applied once on store —
// numerically matching the native butterfly and cuFFT (~1e-7 rel err, f32).

// Radix-4 (pow-4 sizes: n = 4,16,64,256,1024,4096). threads = n/4.
extern "C" __global__ void fft_stockham_r4(
    float* arena,
    unsigned src_off,
    unsigned dst_off,
    unsigned n,
    unsigned inverse,
    float norm_scale,
    unsigned outer
) {
    extern __shared__ float2 sh4[];
    float2* a = sh4;
    float2* b = sh4 + n;
    unsigned row = blockIdx.x;
    if (row >= outer) return;
    unsigned sb = src_off + row * 2u * n;
    unsigned db = dst_off + row * 2u * n;
    unsigned tid = threadIdx.x;
    unsigned q = n >> 2;

    a[tid]        = make_float2(arena[sb + tid],          arena[sb + n + tid]);
    a[tid + q]    = make_float2(arena[sb + tid + q],      arena[sb + n + tid + q]);
    a[tid + 2u*q] = make_float2(arena[sb + tid + 2u*q],   arena[sb + n + tid + 2u*q]);
    a[tid + 3u*q] = make_float2(arena[sb + tid + 3u*q],   arena[sb + n + tid + 3u*q]);
    __syncthreads();

    float sgn = inverse ? 1.0f : -1.0f;   // forward: e^{-i...}
    float rs  = inverse ? -1.0f : 1.0f;   // ±i core rotation
    for (unsigned p = 1u; p < n; p <<= 2) {
        unsigned k = tid & (p - 1u);
        float2 u0 = a[tid], u1 = a[tid + q], u2 = a[tid + 2u*q], u3 = a[tid + 3u*q];
        float base = sgn * 3.14159265358979323846f * (float)k / (2.0f * (float)p);
        float s1, c1, s2, c2, s3, c3;
        __sincosf(base, &s1, &c1);
        __sincosf(2.0f * base, &s2, &c2);
        __sincosf(3.0f * base, &s3, &c3);
        float2 w1 = {u1.x*c1 - u1.y*s1, u1.x*s1 + u1.y*c1};
        float2 w2 = {u2.x*c2 - u2.y*s2, u2.x*s2 + u2.y*c2};
        float2 w3 = {u3.x*c3 - u3.y*s3, u3.x*s3 + u3.y*c3};
        float2 t0 = {u0.x + w2.x, u0.y + w2.y}, t1 = {u0.x - w2.x, u0.y - w2.y};
        float2 t2 = {w1.x + w3.x, w1.y + w3.y}, t3 = {w1.x - w3.x, w1.y - w3.y};
        float2 y0 = {t0.x + t2.x, t0.y + t2.y};
        float2 y1 = {t1.x + rs*t3.y, t1.y - rs*t3.x};
        float2 y2 = {t0.x - t2.x, t0.y - t2.y};
        float2 y3 = {t1.x - rs*t3.y, t1.y + rs*t3.x};
        unsigned j = ((tid - k) << 2) + k;
        b[j] = y0; b[j + p] = y1; b[j + 2u*p] = y2; b[j + 3u*p] = y3;
        __syncthreads();
        float2* tmp = a; a = b; b = tmp;
    }

    arena[db + tid]          = a[tid].x        * norm_scale;
    arena[db + n + tid]      = a[tid].y        * norm_scale;
    arena[db + tid + q]      = a[tid + q].x    * norm_scale;
    arena[db + n + tid + q]  = a[tid + q].y    * norm_scale;
    arena[db + tid + 2u*q]   = a[tid + 2u*q].x * norm_scale;
    arena[db + n + tid + 2u*q] = a[tid + 2u*q].y * norm_scale;
    arena[db + tid + 3u*q]   = a[tid + 3u*q].x * norm_scale;
    arena[db + n + tid + 3u*q] = a[tid + 3u*q].y * norm_scale;
}

// Radix-2 (pow-2 non-pow-4 sizes: n = 2,8,32,128,512,2048). threads = n/2.
extern "C" __global__ void fft_stockham_r2(
    float* arena,
    unsigned src_off,
    unsigned dst_off,
    unsigned n,
    unsigned inverse,
    float norm_scale,
    unsigned outer
) {
    extern __shared__ float2 sh2[];
    float2* a = sh2;
    float2* b = sh2 + n;
    unsigned row = blockIdx.x;
    if (row >= outer) return;
    unsigned sb = src_off + row * 2u * n;
    unsigned db = dst_off + row * 2u * n;
    unsigned tid = threadIdx.x;
    unsigned half = n >> 1;

    a[tid]        = make_float2(arena[sb + tid],        arena[sb + n + tid]);
    a[tid + half] = make_float2(arena[sb + tid + half], arena[sb + n + tid + half]);
    __syncthreads();

    float sgn = inverse ? 1.0f : -1.0f;
    for (unsigned p = 1u; p < n; p <<= 1) {
        unsigned k = tid & (p - 1u);
        float2 u0 = a[tid], u1 = a[tid + half];
        float s, c;
        __sincosf(sgn * 3.14159265358979323846f * (float)k / (float)p, &s, &c);
        float2 t = {u1.x*c - u1.y*s, u1.x*s + u1.y*c};
        unsigned j = (tid << 1) - k;
        b[j].x     = u0.x + t.x; b[j].y     = u0.y + t.y;
        b[j + p].x = u0.x - t.x; b[j + p].y = u0.y - t.y;
        __syncthreads();
        float2* tmp = a; a = b; b = tmp;
    }

    arena[db + tid]         = a[tid].x        * norm_scale;
    arena[db + n + tid]     = a[tid].y        * norm_scale;
    arena[db + tid + half]  = a[tid + half].x * norm_scale;
    arena[db + n + tid + half] = a[tid + half].y * norm_scale;
}

// Complex multiply.
__device__ __forceinline__ float2 fft_cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// In-register 4-point DFT (the radix-4 butterfly core). `rs` = +1 forward,
// -1 inverse (flips the ±i rotation). Higher radices compose from this.
__device__ __forceinline__ void fft_dft4(
    float2 x0, float2 x1, float2 x2, float2 x3, float rs,
    float2& X0, float2& X1, float2& X2, float2& X3
) {
    float2 t0 = {x0.x + x2.x, x0.y + x2.y}, t1 = {x0.x - x2.x, x0.y - x2.y};
    float2 t2 = {x1.x + x3.x, x1.y + x3.y}, t3 = {x1.x - x3.x, x1.y - x3.y};
    X0 = make_float2(t0.x + t2.x, t0.y + t2.y);
    X2 = make_float2(t0.x - t2.x, t0.y - t2.y);
    X1 = make_float2(t1.x + rs * t3.y, t1.y - rs * t3.x);
    X3 = make_float2(t1.x - rs * t3.y, t1.y + rs * t3.x);
}

// 16th root of unity W16^e = exp(sgn·2π·e/16), trivial powers baked in as
// immediates (the values cuFFT folds into FFMAs); the rest never occur in the
// 4×4 decomposition but fall back to __sincosf for safety.
__device__ __forceinline__ float2 fft_w16(int e, float sgn) {
    const float C1 = 0.92387953251128676f;  // cos(pi/8)
    const float S1 = 0.38268343236508977f;  // sin(pi/8)
    const float R2 = 0.70710678118654752f;  // 1/sqrt2
    switch (e & 15) {
        case 0: return make_float2(1.0f, 0.0f);
        case 1: return make_float2(C1, sgn * S1);
        case 2: return make_float2(R2, sgn * R2);
        case 3: return make_float2(S1, sgn * C1);
        case 4: return make_float2(0.0f, sgn);
        case 6: return make_float2(-R2, sgn * R2);
        case 9: return make_float2(-C1, -sgn * S1);
        default: {
            float s, c;
            __sincosf(sgn * 6.28318530717958647692f * (float)e / 16.0f, &s, &c);
            return make_float2(c, s);
        }
    }
}

// Radix-8 Stockham (pow-8 sizes: n = 8,64,512,4096). threads = n/8.
// 8-point DFT via two radix-4 cores + a radix-2 combine (8 = 2×4).
extern "C" __global__ void fft_stockham_r8(
    float* arena,
    unsigned src_off,
    unsigned dst_off,
    unsigned n,
    unsigned inverse,
    float norm_scale,
    unsigned outer
) {
    extern __shared__ float2 sh8[];
    float2* a = sh8;
    float2* b = sh8 + n;
    unsigned row = blockIdx.x;
    if (row >= outer) return;
    unsigned sb = src_off + row * 2u * n;
    unsigned db = dst_off + row * 2u * n;
    unsigned tid = threadIdx.x;
    unsigned e = n >> 3;  // n/8

    #pragma unroll
    for (unsigned i = 0u; i < 8u; ++i)
        a[tid + i * e] = make_float2(arena[sb + tid + i * e], arena[sb + n + tid + i * e]);
    __syncthreads();

    float sgn = inverse ? 1.0f : -1.0f;
    float rs = inverse ? -1.0f : 1.0f;
    const float R2 = 0.70710678118654752f;
    for (unsigned p = 1u; p < n; p <<= 3) {
        unsigned k = tid & (p - 1u);
        float2 u[8];
        #pragma unroll
        for (unsigned i = 0u; i < 8u; ++i) u[i] = a[tid + i * e];
        float base = sgn * 6.28318530717958647692f * (float)k / (8.0f * (float)p);
        #pragma unroll
        for (unsigned i = 1u; i < 8u; ++i) {
            float s, c;
            __sincosf((float)i * base, &s, &c);
            u[i] = make_float2(u[i].x * c - u[i].y * s, u[i].x * s + u[i].y * c);
        }
        float2 E0, E1, E2, E3, O0, O1, O2, O3;
        fft_dft4(u[0], u[2], u[4], u[6], rs, E0, E1, E2, E3);
        fft_dft4(u[1], u[3], u[5], u[7], rs, O0, O1, O2, O3);
        float2 w1 = {R2, sgn * R2}, w2 = {0.0f, sgn}, w3 = {-R2, sgn * R2};
        float2 c0 = O0, c1 = fft_cmul(w1, O1), c2 = fft_cmul(w2, O2), c3 = fft_cmul(w3, O3);
        float2 Y[8];
        Y[0] = make_float2(E0.x + c0.x, E0.y + c0.y); Y[4] = make_float2(E0.x - c0.x, E0.y - c0.y);
        Y[1] = make_float2(E1.x + c1.x, E1.y + c1.y); Y[5] = make_float2(E1.x - c1.x, E1.y - c1.y);
        Y[2] = make_float2(E2.x + c2.x, E2.y + c2.y); Y[6] = make_float2(E2.x - c2.x, E2.y - c2.y);
        Y[3] = make_float2(E3.x + c3.x, E3.y + c3.y); Y[7] = make_float2(E3.x - c3.x, E3.y - c3.y);
        unsigned j = ((tid - k) << 3) + k;
        #pragma unroll
        for (unsigned m = 0u; m < 8u; ++m) b[j + m * p] = Y[m];
        __syncthreads();
        float2* tmp = a; a = b; b = tmp;
    }

    #pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) {
        arena[db + tid + i * e]     = a[tid + i * e].x * norm_scale;
        arena[db + n + tid + i * e] = a[tid + i * e].y * norm_scale;
    }
}

// Radix-16 Stockham (pow-16 sizes: n = 16,256,4096). threads = n/16, EPT=16
// (matches cuFFT's vector_fft<.., EPT<16>>). 16-point DFT via 4×4 Cooley-Tukey:
// 4 inner radix-4 cores → W16 twiddles → 4 outer radix-4 cores.
extern "C" __global__ void fft_stockham_r16(
    float* arena,
    unsigned src_off,
    unsigned dst_off,
    unsigned n,
    unsigned inverse,
    float norm_scale,
    unsigned outer
) {
    extern __shared__ float2 sh16[];
    float2* a = sh16;
    float2* b = sh16 + n;
    unsigned row = blockIdx.x;
    if (row >= outer) return;
    unsigned sb = src_off + row * 2u * n;
    unsigned db = dst_off + row * 2u * n;
    unsigned tid = threadIdx.x;
    unsigned e = n >> 4;  // n/16

    #pragma unroll
    for (unsigned i = 0u; i < 16u; ++i)
        a[tid + i * e] = make_float2(arena[sb + tid + i * e], arena[sb + n + tid + i * e]);
    __syncthreads();

    float sgn = inverse ? 1.0f : -1.0f;
    float rs = inverse ? -1.0f : 1.0f;
    for (unsigned p = 1u; p < n; p <<= 4) {
        unsigned k = tid & (p - 1u);
        float2 u[16];
        #pragma unroll
        for (unsigned i = 0u; i < 16u; ++i) u[i] = a[tid + i * e];
        float base = sgn * 6.28318530717958647692f * (float)k / (16.0f * (float)p);
        #pragma unroll
        for (unsigned i = 1u; i < 16u; ++i) {
            float s, c;
            __sincosf((float)i * base, &s, &c);
            u[i] = make_float2(u[i].x * c - u[i].y * s, u[i].x * s + u[i].y * c);
        }
        // 4 inner radix-4 over stride-4 groups + W16 twiddle.
        float2 A[16];
        #pragma unroll
        for (int n1 = 0; n1 < 4; ++n1) {
            float2 X0, X1, X2, X3;
            fft_dft4(u[n1], u[n1 + 4], u[n1 + 8], u[n1 + 12], rs, X0, X1, X2, X3);
            A[n1 * 4 + 0] = X0;
            A[n1 * 4 + 1] = fft_cmul(X1, fft_w16(n1 * 1, sgn));
            A[n1 * 4 + 2] = fft_cmul(X2, fft_w16(n1 * 2, sgn));
            A[n1 * 4 + 3] = fft_cmul(X3, fft_w16(n1 * 3, sgn));
        }
        // 4 outer radix-4 → natural-order Y[4*k1 + k2].
        float2 Y[16];
        #pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {
            float2 X0, X1, X2, X3;
            fft_dft4(A[0 * 4 + k2], A[1 * 4 + k2], A[2 * 4 + k2], A[3 * 4 + k2], rs, X0, X1, X2, X3);
            Y[0 * 4 + k2] = X0; Y[1 * 4 + k2] = X1; Y[2 * 4 + k2] = X2; Y[3 * 4 + k2] = X3;
        }
        unsigned j = ((tid - k) << 4) + k;
        #pragma unroll
        for (unsigned m = 0u; m < 16u; ++m) b[j + m * p] = Y[m];
        __syncthreads();
        float2* tmp = a; a = b; b = tmp;
    }

    #pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) {
        arena[db + tid + i * e]     = a[tid + i * e].x * norm_scale;
        arena[db + n + tid + i * e] = a[tid + i * e].y * norm_scale;
    }
}

// Mixed-radix Stockham: a different radix (2/4/8) per stage, so pow-2 sizes that
// aren't a pure power of one radix avoid the slow many-stage radix-2 path (e.g.
// 2048 = 8×8×8×4, 4 stages, vs 11 radix-2 stages). `packed` holds log2(radix)
// for each stage in 4-bit fields (stage 0 = low bits); the schedule is computed
// host-side as `[8]·⌊m/3⌋ + [2^(m%3)]`. Capped at radix-8 (u[8]/Y[8]) to keep
// register pressure low — generic runtime-radix code can't be register-blocked
// as tightly as the fully-unrolled dedicated kernels, and FFT is occupancy/
// memory-bound, so fewer registers beats fewer stages here. radix-16 sizes use
// the dedicated `fft_stockham_r16` kernel. Strided loops make the kernel
// agnostic to the per-stage radix and block size.
extern "C" __global__ void fft_stockham_mixed(
    float* arena,
    unsigned src_off,
    unsigned dst_off,
    unsigned n,
    unsigned inverse,
    float norm_scale,
    unsigned outer,
    unsigned packed,
    unsigned num_stages
) {
    extern __shared__ float2 shm[];
    float2* a = shm;
    float2* b = shm + n;
    unsigned row = blockIdx.x;
    if (row >= outer) return;
    unsigned sb = src_off + row * 2u * n;
    unsigned db = dst_off + row * 2u * n;
    unsigned T = blockDim.x;

    for (unsigned i = threadIdx.x; i < n; i += T)
        a[i] = make_float2(arena[sb + i], arena[sb + n + i]);
    __syncthreads();

    float sgn = inverse ? 1.0f : -1.0f;
    float rs = inverse ? -1.0f : 1.0f;
    const float R2 = 0.70710678118654752f;
    unsigned p = 1u;
    for (unsigned st = 0u; st < num_stages; ++st) {
        unsigned R = 1u << ((packed >> (4u * st)) & 0xFu);
        unsigned mm = n / R;  // butterflies this stage
        for (unsigned j = threadIdx.x; j < mm; j += T) {
            unsigned k = j & (p - 1u);
            float2 u[8];
            #pragma unroll
            for (unsigned i = 0u; i < 8u; ++i)
                if (i < R) u[i] = a[j + i * mm];
            float baseang = sgn * 6.28318530717958647692f * (float)k / (float)(R * p);
            #pragma unroll
            for (unsigned i = 1u; i < 8u; ++i)
                if (i < R) {
                    float s, c;
                    __sincosf((float)i * baseang, &s, &c);
                    u[i] = make_float2(u[i].x * c - u[i].y * s, u[i].x * s + u[i].y * c);
                }

            float2 Y[8];
            if (R == 2u) {
                Y[0] = make_float2(u[0].x + u[1].x, u[0].y + u[1].y);
                Y[1] = make_float2(u[0].x - u[1].x, u[0].y - u[1].y);
            } else if (R == 4u) {
                fft_dft4(u[0], u[1], u[2], u[3], rs, Y[0], Y[1], Y[2], Y[3]);
            } else {  // R == 8
                float2 E0, E1, E2, E3, O0, O1, O2, O3;
                fft_dft4(u[0], u[2], u[4], u[6], rs, E0, E1, E2, E3);
                fft_dft4(u[1], u[3], u[5], u[7], rs, O0, O1, O2, O3);
                float2 w1 = {R2, sgn * R2}, w2 = {0.0f, sgn}, w3 = {-R2, sgn * R2};
                float2 c0 = O0, c1 = fft_cmul(w1, O1), c2 = fft_cmul(w2, O2), c3 = fft_cmul(w3, O3);
                Y[0] = make_float2(E0.x + c0.x, E0.y + c0.y); Y[4] = make_float2(E0.x - c0.x, E0.y - c0.y);
                Y[1] = make_float2(E1.x + c1.x, E1.y + c1.y); Y[5] = make_float2(E1.x - c1.x, E1.y - c1.y);
                Y[2] = make_float2(E2.x + c2.x, E2.y + c2.y); Y[6] = make_float2(E2.x - c2.x, E2.y - c2.y);
                Y[3] = make_float2(E3.x + c3.x, E3.y + c3.y); Y[7] = make_float2(E3.x - c3.x, E3.y - c3.y);
            }

            unsigned ob = (j - k) * R + k;  // (j/p)·p·R + k
            #pragma unroll
            for (unsigned i = 0u; i < 8u; ++i)
                if (i < R) b[ob + i * p] = Y[i];
        }
        __syncthreads();
        float2* tmp = a; a = b; b = tmp;
        p *= R;
    }

    for (unsigned i = threadIdx.x; i < n; i += T) {
        arena[db + i]     = a[i].x * norm_scale;
        arena[db + n + i] = a[i].y * norm_scale;
    }
}
