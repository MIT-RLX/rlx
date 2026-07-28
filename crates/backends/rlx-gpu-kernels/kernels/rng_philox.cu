// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// On-device Philox4×32-10 stream, bit-matched to `rlx_ir::Philox4x32`.
// Each normal sample consumes two u32s (Box–Muller); each Philox counter
// yields four u32s → two normals. Sample `i` reads lanes of counter `i/2`.

__device__ void philox_round(unsigned* s, unsigned k0, unsigned k1) {
    const unsigned long long M0 = 0xD2561A75ull;
    const unsigned long long M1 = 0xCD9E8D57ull;
    unsigned long long p0 = (unsigned long long)s[0] * M0;
    unsigned long long p1 = (unsigned long long)s[2] * M1;
    unsigned hi0 = (unsigned)(p0 >> 32);
    unsigned lo0 = (unsigned)p0;
    unsigned hi1 = (unsigned)(p1 >> 32);
    unsigned lo1 = (unsigned)p1;
    s[0] = hi1 ^ s[1] ^ k0;
    s[1] = lo1;
    s[2] = hi0 ^ s[3] ^ k1;
    s[3] = lo0;
}

__device__ void philox_10(unsigned c0, unsigned c1, unsigned c2, unsigned c3,
                         unsigned seed_lo, unsigned seed_hi, unsigned* out) {
    unsigned s[4] = {c0, c1, c2, c3};
    unsigned k0 = seed_lo;
    unsigned k1 = seed_hi;
    for (int i = 0; i < 10; ++i) {
        philox_round(s, k0, k1);
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    out[0] = s[0]; out[1] = s[1]; out[2] = s[2]; out[3] = s[3];
}

__device__ float u32_to_unit(unsigned bits) {
    return (float)(bits >> 8) / (float)(1u << 24);
}

// counter for Philox block `blk` (matches sequential counter increments).
__device__ void counter_from_block(unsigned long long blk,
                                  unsigned* c0, unsigned* c1,
                                  unsigned* c2, unsigned* c3) {
    *c0 = (unsigned)blk;
    *c1 = (unsigned)(blk >> 32);
    *c2 = 0u;
    *c3 = 0u;
}

extern "C" __global__ void rng_normal_philox(
    float* arena,
    unsigned dst_off,
    unsigned len,
    float mean,
    float scale,
    unsigned seed_lo,
    unsigned seed_hi
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    unsigned long long blk = (unsigned long long)i / 2ull;
    unsigned lane0 = (i & 1u) ? 2u : 0u;
    unsigned c0, c1, c2, c3;
    counter_from_block(blk, &c0, &c1, &c2, &c3);
    unsigned buf[4];
    philox_10(c0, c1, c2, c3, seed_lo, seed_hi, buf);
    float u1 = u32_to_unit(buf[lane0]);
    float u2 = u32_to_unit(buf[lane0 + 1u]);
    if (u1 < 1.17549435e-38f) u1 = 1.17549435e-38f; // f32::MIN_POSITIVE
    float r = sqrtf(-2.f * logf(u1));
    float theta = 6.283185307179586f * u2; // 2π
    arena[dst_off + i] = mean + scale * (r * cosf(theta));
}

extern "C" __global__ void rng_uniform_philox(
    float* arena,
    unsigned dst_off,
    unsigned len,
    float low,
    float high,
    unsigned seed_lo,
    unsigned seed_hi
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    // One u32 per uniform sample → counter block i/4, lane i%4.
    unsigned long long blk = (unsigned long long)i / 4ull;
    unsigned lane = i & 3u;
    unsigned c0, c1, c2, c3;
    counter_from_block(blk, &c0, &c1, &c2, &c3);
    unsigned buf[4];
    philox_10(c0, c1, c2, c3, seed_lo, seed_hi, buf);
    float u = u32_to_unit(buf[lane]);
    arena[dst_off + i] = low + u * (high - low);
}

extern "C" __global__ void rng_fill_zero(
    float* arena,
    unsigned dst_off,
    unsigned len
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    arena[dst_off + i] = 0.f;
}
