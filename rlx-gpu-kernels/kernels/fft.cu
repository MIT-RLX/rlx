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
    unsigned row = blockIdx.y;
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

    unsigned row = blockIdx.y;
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
        float cos_a = __cosf(angle);
        float sin_a = __sinf(angle);
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
    unsigned row = blockIdx.y;
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
    float cos1 = __cosf(angle1);
    float sin1 = __sinf(angle1);
    float w1b_r = cos1 * br - sin1 * bi;
    float w1b_i = sin1 * br + cos1 * bi;
    float w1d_r = cos1 * dr - sin1 * di;
    float w1d_i = sin1 * dr + cos1 * di;

    float u0r = ar + w1b_r, u0i = ai + w1b_i;
    float u1r = ar - w1b_r, u1i = ai - w1b_i;
    float u2r = cr + w1d_r, u2i = ci + w1d_i;
    float u3r = cr - w1d_r, u3i = ci - w1d_i;

    float angle2a = sign * 3.14159265358979323846f * (float)k / (float)(q * 2u);
    float cos2a = __cosf(angle2a);
    float sin2a = __sinf(angle2a);
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
    unsigned row = blockIdx.y;
    if (row >= outer) return;
    unsigned base = off + row * 2u * n;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2u) return;

    unsigned k = tid % half_stride;
    unsigned i = (tid / half_stride) * (half_stride * 2u) + k;
    unsigned j = i + half_stride;

    float sign = inverse ? 1.0f : -1.0f;
    float angle = sign * 3.14159265358979323846f * (float)k / (float)half_stride;
    float cos_a = __cosf(angle);
    float sin_a = __sinf(angle);

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

    unsigned row = blockIdx.y;
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
            float wre = __cosf(theta);
            float wim = __sinf(theta);
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
