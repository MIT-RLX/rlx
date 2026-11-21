// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// 3-D convolution (NCDHW, groups = 1) on RDNA3's matrix cores: an implicit
// GEMM with M = output positions, N = output channels, K = input channels x
// taps, as 16x16x16 f16 WMMA products accumulated in f32. HIP only, gfx11
// only (the host checks the arch; elsewhere the body is empty).
//
// Input and filter come from the f32 arena, rounded to f16 as the tiles are
// staged — f16 arithmetic, ~1e-3 relative, as `AlwaysF16` asks — so no
// conversion pass runs; the output is f32.
//
// K tiles are whole input channels, as Metal's `conv3d_gemm_c`: taps are
// padded to `tp`, the next power of two (27 -> 32, 8 -> 8, 1 -> 1), and a
// 32-slot tile holds 32 / tp channels. A slot's tap is then the same in
// every tile, so its spatial offset and bounds test are computed once and a
// tile only adds a channel offset — no division in the loop.
//
// Block: 128 threads = 4 wave32s, tile 64 positions x BN channels (32 or
// 64); wave w owns positions 16w..16w+15 and BN/16 accumulators. WMMA layout
// (gfx11, wave32): lane l holds A row l % 16 and B column l % 16, 16 halves
// each, lanes 16-31 duplicating 0-15; the accumulator's 8 floats in lane l
// are rows 2i + l / 16 of column l % 16.

typedef _Float16 rlx_half16 __attribute__((ext_vector_type(16)));
typedef _Float16 rlx_half8 __attribute__((ext_vector_type(8)));
typedef float rlx_float8 __attribute__((ext_vector_type(8)));

#define WM_BM 64u
#define WM_BK 32u
#define WM_SLOTS 16u   // A slots a thread stages: 64 x 32 / 128
// Shared rows of 40 halves (80 bytes): every 8-half run an operand reads is
// 16-byte aligned, so it is one vector load.
#define WM_LD 40u

template <unsigned int BN>
__device__ __forceinline__ void conv3d_wmma_impl(
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
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) \
    || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
    constexpr unsigned int NACC = BN / 16u;
    // As[position][slot]; Bs[channel][slot] — B stored transposed, so both
    // operands are contiguous rows.
    __shared__ __attribute__((aligned(16))) _Float16 As[WM_BM][WM_LD];
    __shared__ __attribute__((aligned(16))) _Float16 Bs[BN][WM_LD];
    const float* x = arena + in_off;
    const float* wt = arena + w_off;
    float* y = arena + out_off;
    unsigned int taps = kd * kh * kw;
    unsigned int tp = 1;
    while (tp < taps) tp <<= 1;
    unsigned int cpt = WM_BK / tp;   // channels a K tile
    unsigned int K = c_in * taps;
    unsigned int sp_out = d_out * h_out * w_out;
    unsigned int M = n * sp_out;
    unsigned int dhw = d * h * w;
    unsigned int m0 = blockIdx.x * WM_BM, n0 = blockIdx.y * BN;
    unsigned int tid = threadIdx.x, wave = tid / 32u, lane = tid % 32u;

    // This thread stages A row `tid % 64` at slots s = tid / 64 + 2j: the
    // channel within the tile is s / tp, the tap s % tp — fixed for the
    // whole loop, so the spatial offset is too (-1: padding or out of bounds).
    unsigned int arow = tid % WM_BM, akk = tid / WM_BM;
    unsigned int am = m0 + arow;
    int aoff[WM_SLOTS];
    unsigned int abase = 0;
    {
        int bz = 0, by = 0, bx = 0;
        if (am < M) {
            unsigned int t = am;
            unsigned int wo = t % w_out; t /= w_out;
            unsigned int ho = t % h_out; t /= h_out;
            unsigned int dz = t % d_out; unsigned int nn = t / d_out;
            bz = (int)(dz * sd) - (int)pd;
            by = (int)(ho * sh) - (int)ph;
            bx = (int)(wo * sw) - (int)pw;
            abase = nn * c_in * dhw;
        }
        for (unsigned int j = 0; j < WM_SLOTS; ++j) {
            unsigned int s = akk + 2u * j, t = s % tp, cl = s / tp;
            aoff[j] = -1;
            if (am >= M || t >= taps) continue;
            unsigned int kx = t % kw, r = t / kw, ky = r % kh, kz = r / kh;
            int iz = bz + (int)(kz * dd), iy = by + (int)(ky * dh), ix = bx + (int)(kx * dw);
            if (iz < 0 || iy < 0 || ix < 0 || iz >= (int)d || iy >= (int)h || ix >= (int)w) continue;
            aoff[j] = (int)(cl * dhw + ((unsigned int)iz * h + (unsigned int)iy) * w + (unsigned int)ix);
        }
    }

    rlx_float8 acc[NACC];
    for (unsigned int q = 0; q < NACC; ++q) acc[q] = rlx_float8{};
    for (unsigned int c0 = 0; c0 < c_in; c0 += cpt) {
        unsigned int cn = min(cpt, c_in - c0);
        const float* xc = x + abase + c0 * dhw;
        for (unsigned int j = 0; j < WM_SLOTS; ++j) {
            unsigned int s = akk + 2u * j;
            float v = (aoff[j] >= 0 && s / tp < cn) ? xc[aoff[j]] : 0.0f;
            As[arow][s] = (_Float16)v;
        }
        // B[s][col] = filter[n0 + col][c0 + s / tp][s % tp]: consecutive
        // threads walk consecutive slots of one filter row.
        for (unsigned int i = tid; i < WM_BK * BN; i += 128u) {
            unsigned int s = i % WM_BK, col = i / WM_BK;
            unsigned int t = s % tp, cl = s / tp, co = n0 + col;
            float v = (t < taps && cl < cn && co < c_out) ? wt[co * K + (c0 + cl) * taps + t] : 0.0f;
            Bs[col][s] = (_Float16)v;
        }
        __syncthreads();
        unsigned int r = wave * 16u + lane % 16u, cc = lane % 16u;
        for (unsigned int kb = 0; kb < WM_BK; kb += 16u) {
            rlx_half8 a0 = *reinterpret_cast<const rlx_half8*>(&As[r][kb]);
            rlx_half8 a1 = *reinterpret_cast<const rlx_half8*>(&As[r][kb + 8u]);
            rlx_half16 a = __builtin_shufflevector(a0, a1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            for (unsigned int q = 0; q < NACC; ++q) {
                const _Float16* br = &Bs[q * 16u + cc][kb];
                rlx_half8 b0 = *reinterpret_cast<const rlx_half8*>(br);
                rlx_half8 b1 = *reinterpret_cast<const rlx_half8*>(br + 8);
                rlx_half16 b = __builtin_shufflevector(b0, b1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                acc[q] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[q]);
            }
        }
        __syncthreads();
    }

    for (unsigned int e = 0; e < 8u; ++e) {
        unsigned int m = m0 + wave * 16u + 2u * e + lane / 16u;
        if (m >= M) continue;
        unsigned int nn = m / sp_out, sp = m % sp_out;
        for (unsigned int q = 0; q < NACC; ++q) {
            unsigned int co = n0 + q * 16u + lane % 16u;
            if (co < c_out) y[(nn * c_out + co) * sp_out + sp] = acc[q][e];
        }
    }
#endif
}

#define RLX_CONV3D_WMMA_ARGS \
    float* arena, \
    unsigned int n, unsigned int c_in, unsigned int c_out, \
    unsigned int d, unsigned int h, unsigned int w, \
    unsigned int d_out, unsigned int h_out, unsigned int w_out, \
    unsigned int kd, unsigned int kh, unsigned int kw, \
    unsigned int sd, unsigned int sh, unsigned int sw, \
    unsigned int pd, unsigned int ph, unsigned int pw, \
    unsigned int dd, unsigned int dh, unsigned int dw, \
    unsigned int in_off, unsigned int w_off, unsigned int out_off
#define RLX_CONV3D_WMMA_PASS \
    arena, n, c_in, c_out, d, h, w, d_out, h_out, w_out, kd, kh, kw, sd, sh, sw, \
    pd, ph, pw, dd, dh, dw, in_off, w_off, out_off

// 32 output channels a block (c_out < 64), and 64.
extern "C" __global__ void __launch_bounds__(128) conv3d_wmma(RLX_CONV3D_WMMA_ARGS) {
    conv3d_wmma_impl<32u>(RLX_CONV3D_WMMA_PASS);
}
extern "C" __global__ void __launch_bounds__(128) conv3d_wmma_n64(RLX_CONV3D_WMMA_ARGS) {
    conv3d_wmma_impl<64u>(RLX_CONV3D_WMMA_PASS);
}

// ── Patch-staged variant: stride 1, dilation 1, filters up to 3x3x3 ──
// `conv3d_wmma` gathers every A element from global memory: a 3^3 filter
// reads each input voxel 27 times, and that gather — not the matrix cores —
// is what bounds it. Here a block computes a 4 x 16 box of output positions
// (one output row of 16 a wave, so a lane's A row is position x = lane % 16
// of its wave's row) and stages, per input channel, the input patch the box
// reads — kd x (4 + kh - 1) x (16 + kw - 1) values, zero outside the volume —
// into shared memory once, coalesced along x. A fragments are then read from
// the patch through a per-slot offset table, with no index arithmetic in the
// loop. K tiles are whole channels as above: 32 / tp channels a tile.

#define WP_TY 4u
#define WP_TX 16u
#define WP_PATCH_MAX 2048u   // halves: 32 channels of a 1^3 box, or one 3^3 patch

template <unsigned int BN>
__device__ __forceinline__ void conv3d_wmma_patch_impl(
    float* arena,
    unsigned int n, unsigned int c_in, unsigned int c_out,
    unsigned int d, unsigned int h, unsigned int w,
    unsigned int d_out, unsigned int h_out, unsigned int w_out,
    unsigned int kd, unsigned int kh, unsigned int kw,
    unsigned int pd, unsigned int ph, unsigned int pw,
    unsigned int in_off, unsigned int w_off, unsigned int out_off
) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) \
    || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
    constexpr unsigned int NACC = BN / 16u;
    __shared__ __attribute__((aligned(16))) _Float16 patch[WP_PATCH_MAX];
    __shared__ __attribute__((aligned(16))) _Float16 Bs[BN][WM_LD];
    __shared__ int toff[WM_BK];   // patch offset of each slot's tap, -1 if padding
    const float* x = arena + in_off;
    const float* wt = arena + w_off;
    float* y = arena + out_off;
    unsigned int taps = kd * kh * kw;
    unsigned int tp = 1;
    while (tp < taps) tp <<= 1;
    unsigned int cpt = WM_BK / tp;
    unsigned int K = c_in * taps;
    unsigned int py = WP_TY + kh - 1, px = WP_TX + kw - 1;
    unsigned int psz = kd * py * px;   // one channel's patch
    unsigned int hw = h * w, dhw = d * hw;
    unsigned int sp_out = d_out * h_out * w_out;

    // Block -> (batch, z, y tile, x tile).
    unsigned int tiles_x = (w_out + WP_TX - 1) / WP_TX, tiles_y = (h_out + WP_TY - 1) / WP_TY;
    unsigned int b = blockIdx.x;
    unsigned int bx = b % tiles_x; b /= tiles_x;
    unsigned int by = b % tiles_y; b /= tiles_y;
    unsigned int z = b % d_out, nn = b / d_out;
    int z0 = (int)z - (int)pd, y0 = (int)(by * WP_TY) - (int)ph, x0 = (int)(bx * WP_TX) - (int)pw;
    unsigned int n0 = blockIdx.y * BN;
    unsigned int tid = threadIdx.x, wave = tid / 32u, lane = tid % 32u;

    if (tid < WM_BK) {
        unsigned int t = tid % tp, cl = tid / tp;
        int o = -1;
        if (t < taps) {
            unsigned int kx = t % kw, r = t / kw, ky = r % kh, kz = r / kh;
            o = (int)(cl * psz + (kz * py + ky) * px + kx);
        }
        toff[tid] = o;
    }
    // A row of this lane: output (y = by*4 + wave, x = bx*16 + lane % 16).
    unsigned int lbase = wave * px + lane % 16u;

    rlx_float8 acc[NACC];
    for (unsigned int q = 0; q < NACC; ++q) acc[q] = rlx_float8{};
    const float* xn = x + nn * c_in * dhw;
    for (unsigned int c0 = 0; c0 < c_in; c0 += cpt) {
        unsigned int cn = min(cpt, c_in - c0);
        __syncthreads();
        for (unsigned int i = tid; i < cn * psz; i += 128u) {
            unsigned int cl = i / psz, r = i % psz;
            unsigned int ix = r % px, r2 = r / px, iy = r2 % py, iz = r2 / py;
            int gz = z0 + (int)iz, gy = y0 + (int)iy, gx = x0 + (int)ix;
            float v = 0.0f;
            if (gz >= 0 && gy >= 0 && gx >= 0 && gz < (int)d && gy < (int)h && gx < (int)w)
                v = xn[(c0 + cl) * dhw + (unsigned int)gz * hw + (unsigned int)gy * w + (unsigned int)gx];
            patch[i] = (_Float16)v;
        }
        for (unsigned int i = tid; i < WM_BK * BN; i += 128u) {
            unsigned int s = i % WM_BK, col = i / WM_BK;
            unsigned int t = s % tp, cl = s / tp, co = n0 + col;
            float v = (t < taps && cl < cn && co < c_out) ? wt[co * K + (c0 + cl) * taps + t] : 0.0f;
            Bs[col][s] = (_Float16)v;
        }
        __syncthreads();
        unsigned int cc = lane % 16u;
        for (unsigned int kb = 0; kb < WM_BK; kb += 16u) {
            rlx_half16 a;
            for (unsigned int e = 0; e < 16u; ++e) {
                int o = toff[kb + e];
                a[e] = (o >= 0 && (kb + e) / tp < cn) ? patch[o + lbase] : (_Float16)0.0f;
            }
            for (unsigned int q = 0; q < NACC; ++q) {
                const _Float16* br = &Bs[q * 16u + cc][kb];
                rlx_half8 b0 = *reinterpret_cast<const rlx_half8*>(br);
                rlx_half8 b1 = *reinterpret_cast<const rlx_half8*>(br + 8);
                rlx_half16 bv = __builtin_shufflevector(b0, b1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                acc[q] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bv, acc[q]);
            }
        }
    }

    // Accumulator row 2e + lane / 16 of this wave is output x = bx*16 + that.
    unsigned int oy = by * WP_TY + wave;
    if (oy >= h_out) return;
    for (unsigned int e = 0; e < 8u; ++e) {
        unsigned int ox = bx * WP_TX + 2u * e + lane / 16u;
        if (ox >= w_out) continue;
        unsigned int sp = (z * h_out + oy) * w_out + ox;
        for (unsigned int q = 0; q < NACC; ++q) {
            unsigned int co = n0 + q * 16u + lane % 16u;
            if (co < c_out) y[(nn * c_out + co) * sp_out + sp] = acc[q][e];
        }
    }
#endif
}

#define RLX_CONV3D_WMMA_PATCH_ARGS \
    float* arena, \
    unsigned int n, unsigned int c_in, unsigned int c_out, \
    unsigned int d, unsigned int h, unsigned int w, \
    unsigned int d_out, unsigned int h_out, unsigned int w_out, \
    unsigned int kd, unsigned int kh, unsigned int kw, \
    unsigned int pd, unsigned int ph, unsigned int pw, \
    unsigned int in_off, unsigned int w_off, unsigned int out_off
#define RLX_CONV3D_WMMA_PATCH_PASS \
    arena, n, c_in, c_out, d, h, w, d_out, h_out, w_out, kd, kh, kw, pd, ph, pw, in_off, w_off, out_off

extern "C" __global__ void __launch_bounds__(128) conv3d_wmma_patch(RLX_CONV3D_WMMA_PATCH_ARGS) {
    conv3d_wmma_patch_impl<32u>(RLX_CONV3D_WMMA_PATCH_PASS);
}
extern "C" __global__ void __launch_bounds__(128) conv3d_wmma_patch_n64(RLX_CONV3D_WMMA_PATCH_ARGS) {
    conv3d_wmma_patch_impl<64u>(RLX_CONV3D_WMMA_PATCH_PASS);
}
