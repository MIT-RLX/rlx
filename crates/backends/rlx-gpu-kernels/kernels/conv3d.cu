// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// 3D conv NCDHW. Weight: [c_out, c_in/groups, kd, kh, kw].
extern "C" __global__ void conv3d(
    float* arena,
    unsigned int n, unsigned int c_in, unsigned int c_out,
    unsigned int d, unsigned int h, unsigned int w,
    unsigned int d_out, unsigned int h_out, unsigned int w_out,
    unsigned int kd, unsigned int kh, unsigned int kw,
    unsigned int sd, unsigned int sh, unsigned int sw,
    unsigned int pd, unsigned int ph, unsigned int pw,
    unsigned int dd, unsigned int dh, unsigned int dw,
    unsigned int groups,
    unsigned int in_off,
    unsigned int w_off,
    unsigned int out_off
) {
    unsigned int total = n * c_out * d_out * h_out * w_out;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int wo = i % w_out;
    unsigned int q1 = i / w_out;
    unsigned int ho = q1 % h_out;
    unsigned int q2 = q1 / h_out;
    unsigned int do_ = q2 % d_out;
    unsigned int q3 = q2 / d_out;
    unsigned int co = q3 % c_out;
    unsigned int nn = q3 / c_out;
    unsigned int c_in_per_g = c_in / groups;
    unsigned int c_out_per_g = c_out / groups;
    unsigned int g = co / c_out_per_g;
    unsigned int ci_start = g * c_in_per_g;
    float acc = 0.0f;
    for (unsigned int ci_off = 0; ci_off < c_in_per_g; ++ci_off) {
        unsigned int ci = ci_start + ci_off;
        for (unsigned int ki = 0; ki < kd; ++ki)
        for (unsigned int kj = 0; kj < kh; ++kj)
        for (unsigned int kk = 0; kk < kw; ++kk) {
            int id = (int)(do_ * sd + ki * dd) - (int)pd;
            int ih = (int)(ho  * sh + kj * dh) - (int)ph;
            int iw = (int)(wo  * sw + kk * dw) - (int)pw;
            if (id < 0 || ih < 0 || iw < 0
                || id >= (int)d || ih >= (int)h || iw >= (int)w) continue;
            float xv = arena[in_off + (((nn * c_in + ci) * d + id) * h + ih) * w + iw];
            float wv = arena[w_off + ((((co * c_in_per_g + ci_off) * kd + ki) * kh + kj) * kw + kk)];
            acc += xv * wv;
        }
    }
    arena[out_off + i] = acc;
}

// ── Register-blocked conv3d (groups = 1, kh*kw <= 27) ──
// `conv3d` reads every input voxel once per output channel. Here a thread
// computes CO_T = 8 output channels for its output position, so each input
// read serves eight; a block (256 positions x 8 channels) stages the
// filter for its channels, CI_T input channels at a time, in shared memory.
// Positions run along the fastest axis, so input loads coalesce.
//
// `TIn` is the input and filter storage: float (`conv3d_cot`, arena
// offsets) or f16 bits (`conv3d_cot_h`, separate half buffers, for a graph
// run in f16: half the bytes read). Accumulation and the output are f32.

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif

#define CONV3D_CO_T 8u
#define CONV3D_CI_T 4u
#define CONV3D_TAPS_MAX 27u

__device__ __forceinline__ float conv3d_load(const float* p, unsigned int i) { return p[i]; }
__device__ __forceinline__ float conv3d_load(const unsigned short* p, unsigned int i) {
    unsigned short b = p[i];
    return __half2float(*reinterpret_cast<const __half*>(&b));
}

template <typename TIn>
__device__ __forceinline__ void conv3d_cot_impl(
    const TIn* x, const TIn* wt, float* y,
    unsigned int n, unsigned int c_in, unsigned int c_out,
    unsigned int d, unsigned int h, unsigned int w,
    unsigned int d_out, unsigned int h_out, unsigned int w_out,
    unsigned int kd, unsigned int kh, unsigned int kw,
    unsigned int sd, unsigned int sh, unsigned int sw,
    unsigned int pd, unsigned int ph, unsigned int pw,
    unsigned int dd, unsigned int dh, unsigned int dw
) {
    __shared__ float ws[CONV3D_CI_T * CONV3D_CO_T * CONV3D_TAPS_MAX];
    unsigned int taps = kd * kh * kw;
    unsigned int sp_out = d_out * h_out * w_out;
    unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
    bool m_ok = m < n * sp_out;
    unsigned int co0 = blockIdx.y * CONV3D_CO_T;
    unsigned int wo = 0, ho = 0, do_ = 0, nn = 0;
    if (m_ok) {
        unsigned int t = m;
        wo = t % w_out; t /= w_out;
        ho = t % h_out; t /= h_out;
        do_ = t % d_out; nn = t / d_out;
    }
    float acc[CONV3D_CO_T];
    for (unsigned int j = 0; j < CONV3D_CO_T; ++j) acc[j] = 0.0f;
    unsigned int dhw = d * h * w;
    // A filter past CONV3D_TAPS_MAX taps (5^3) is staged kdc depth planes at
    // a time; up to CONV3D_TAPS_MAX it is one chunk, as before. The host
    // checks kh * kw <= CONV3D_TAPS_MAX.
    unsigned int plane = kh * kw;
    unsigned int kdc = max(1u, min(kd, CONV3D_TAPS_MAX / plane));
    for (unsigned int c0 = 0; c0 < c_in; c0 += CONV3D_CI_T)
    for (unsigned int k0 = 0; k0 < kd; k0 += kdc) {
        unsigned int cn = min(CONV3D_CI_T, c_in - c0);
        unsigned int tn = min(kdc, kd - k0) * plane;
        // ws[(ci * CO_T + j) * tn + t] = filter[co0 + j][c0 + ci][k0 * plane + t]
        for (unsigned int i = threadIdx.x; i < cn * CONV3D_CO_T * tn; i += blockDim.x) {
            unsigned int t = i % tn, r = i / tn;
            unsigned int j = r % CONV3D_CO_T, ci = r / CONV3D_CO_T;
            unsigned int co = co0 + j;
            ws[i] = co < c_out ? conv3d_load(wt, (co * c_in + c0 + ci) * taps + k0 * plane + t) : 0.0f;
        }
        __syncthreads();
        if (m_ok) {
            for (unsigned int ci = 0; ci < cn; ++ci) {
                unsigned int xb = (nn * c_in + c0 + ci) * dhw;
                unsigned int t = 0;
                for (unsigned int ki = k0; ki < k0 + tn / plane; ++ki) {
                    int id = (int)(do_ * sd + ki * dd) - (int)pd;
                    for (unsigned int kj = 0; kj < kh; ++kj) {
                        int ih = (int)(ho * sh + kj * dh) - (int)ph;
                        for (unsigned int kk = 0; kk < kw; ++kk, ++t) {
                            int iw = (int)(wo * sw + kk * dw) - (int)pw;
                            if (id < 0 || ih < 0 || iw < 0 || id >= (int)d || ih >= (int)h || iw >= (int)w) continue;
                            float xv = conv3d_load(x, xb + ((unsigned int)id * h + (unsigned int)ih) * w + (unsigned int)iw);
                            const float* wr = &ws[ci * CONV3D_CO_T * tn + t];
                            for (unsigned int j = 0; j < CONV3D_CO_T; ++j) acc[j] += xv * wr[j * tn];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    if (!m_ok) return;
    unsigned int sp = m % sp_out;
    for (unsigned int j = 0; j < CONV3D_CO_T; ++j) {
        unsigned int co = co0 + j;
        if (co < c_out) y[(nn * c_out + co) * sp_out + sp] = acc[j];
    }
}

extern "C" __global__ void conv3d_cot(
    float* arena,
    unsigned int n, unsigned int c_in, unsigned int c_out,
    unsigned int d, unsigned int h, unsigned int w,
    unsigned int d_out, unsigned int h_out, unsigned int w_out,
    unsigned int kd, unsigned int kh, unsigned int kw,
    unsigned int sd, unsigned int sh, unsigned int sw,
    unsigned int pd, unsigned int ph, unsigned int pw,
    unsigned int dd, unsigned int dh, unsigned int dw,
    unsigned int in_off, unsigned int w_off, unsigned int out_off
) {
    conv3d_cot_impl<float>(arena + in_off, arena + w_off, arena + out_off, n, c_in, c_out, d, h, w,
                           d_out, h_out, w_out, kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw);
}

extern "C" __global__ void conv3d_cot_h(
    const unsigned short* xh, const unsigned short* wh, float* arena,
    unsigned int n, unsigned int c_in, unsigned int c_out,
    unsigned int d, unsigned int h, unsigned int w,
    unsigned int d_out, unsigned int h_out, unsigned int w_out,
    unsigned int kd, unsigned int kh, unsigned int kw,
    unsigned int sd, unsigned int sh, unsigned int sw,
    unsigned int pd, unsigned int ph, unsigned int pw,
    unsigned int dd, unsigned int dh, unsigned int dw,
    unsigned int out_off
) {
    conv3d_cot_impl<unsigned short>(xh, wh, arena + out_off, n, c_in, c_out, d, h, w,
                                    d_out, h_out, w_out, kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw);
}
