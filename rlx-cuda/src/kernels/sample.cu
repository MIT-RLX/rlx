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

// Multinomial sampling kernel with optional top-k / top-p / temperature.
// Threefry-2×32-20 RNG + JAX-style key derivation + Gumbel-max categorical
// sampling. Mirrors rlx-wgpu's sample.wgsl exactly so picks are
// algorithmically equivalent across backends.
//
// One thread per batch row; serial inside each row.

__device__ __forceinline__ unsigned int rotl32(unsigned int x, unsigned int n) {
    return (x << n) | (x >> (32u - n));
}

__device__ void threefry2x32_20(
    unsigned int c0_in, unsigned int c1_in,
    unsigned int k0_in, unsigned int k1_in,
    unsigned int* out0, unsigned int* out1
) {
    const unsigned int KS_PARITY = 0x1BD11BDAu;
    unsigned int ks0 = k0_in;
    unsigned int ks1 = k1_in;
    unsigned int ks2 = ks0 ^ ks1 ^ KS_PARITY;
    unsigned int x0 = c0_in + ks0;
    unsigned int x1 = c1_in + ks1;

    // Rounds 1-4
    x0 += x1; x1 = rotl32(x1, 13); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1, 15); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1, 26); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1,  6); x1 ^= x0;
    x0 += ks1; x1 += ks2; x1 += 1;

    // Rounds 5-8
    x0 += x1; x1 = rotl32(x1, 17); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1, 29); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1, 16); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1, 24); x1 ^= x0;
    x0 += ks2; x1 += ks0; x1 += 2;

    // Rounds 9-12
    x0 += x1; x1 = rotl32(x1, 13); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1, 15); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1, 26); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1,  6); x1 ^= x0;
    x0 += ks0; x1 += ks1; x1 += 3;

    // Rounds 13-16
    x0 += x1; x1 = rotl32(x1, 17); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1, 29); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1, 16); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1, 24); x1 ^= x0;
    x0 += ks1; x1 += ks2; x1 += 4;

    // Rounds 17-20
    x0 += x1; x1 = rotl32(x1, 13); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1, 15); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1, 26); x1 ^= x0;
    x0 += x1; x1 = rotl32(x1,  6); x1 ^= x0;
    x0 += ks2; x1 += ks0; x1 += 5;

    *out0 = x0;
    *out1 = x1;
}

extern "C" __global__ void sample(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int in_off,
    unsigned int out_off,
    unsigned int top_k,
    unsigned int top_p_bits,
    unsigned int temp_bits,
    unsigned int seed_lo,
    unsigned int seed_hi
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    unsigned int base = in_off + row * inner;
    float temp = __int_as_float((int)temp_bits);
    float top_p = __int_as_float((int)top_p_bits);
    float inv_temp = 1.0f / fmaxf(temp, 1e-6f);

    // Apply temperature; track row max.
    float m = -3.4e38f;
    for (unsigned int i = 0; i < inner; ++i) {
        float v = arena[base + i] * inv_temp;
        arena[base + i] = v;
        m = fmaxf(m, v);
    }
    // exp / sum → probabilities.
    float sum_e = 0.0f;
    for (unsigned int i = 0; i < inner; ++i) {
        float e = expf(arena[base + i] - m);
        arena[base + i] = e;
        sum_e += e;
    }
    float inv_sum = 1.0f / sum_e;
    for (unsigned int i = 0; i < inner; ++i) {
        arena[base + i] *= inv_sum;
    }

    // Top-K / Top-P selection (negate-mark sentinel).
    bool need_filter = (top_k > 0) || (top_p < 1.0f && top_p > 0.0f);
    if (need_filter) {
        const float SENTINEL_EPS = 1e-30f;
        float cum = 0.0f;
        unsigned int picked_count = 0;
        unsigned int k_limit = (top_k > 0) ? top_k : inner;
        while (picked_count < k_limit) {
            float best_v = -1.0f;
            unsigned int best_i = 0;
            bool found = false;
            for (unsigned int i = 0; i < inner; ++i) {
                float v = arena[base + i];
                if (v >= 0.0f && v > best_v) {
                    best_v = v; best_i = i; found = true;
                }
            }
            if (!found) break;
            arena[base + best_i] = -best_v - SENTINEL_EPS;
            cum += best_v;
            picked_count++;
            if (top_p < 1.0f && cum >= top_p) break;
        }
        float new_sum = 0.0f;
        for (unsigned int i = 0; i < inner; ++i) {
            float v = arena[base + i];
            if (v < 0.0f) {
                float restored = -v - SENTINEL_EPS;
                arena[base + i] = restored;
                new_sum += restored;
            } else {
                arena[base + i] = 0.0f;
            }
        }
        float inv_new = 1.0f / fmaxf(new_sum, 1e-12f);
        for (unsigned int i = 0; i < inner; ++i) {
            arena[base + i] *= inv_new;
        }
    }

    // JAX-style seed → key derivation (one Threefry round on the seed).
    unsigned int kx, ky;
    threefry2x32_20(0, 0, seed_lo, seed_hi, &kx, &ky);

    // Gumbel-max sampling: argmax(log p + g) where g = -log(-log(u)).
    float best_score = -3.4e38f;
    unsigned int picked = 0;
    for (unsigned int i = 0; i < inner; ++i) {
        float p = arena[base + i];
        if (p <= 0.0f) continue;
        unsigned int rx, ry;
        threefry2x32_20(row, i, kx, ky, &rx, &ry);
        float u = (float)(rx >> 8) / 16777216.0f;
        float u_safe = fmaxf(u, 1e-30f);
        float g = -logf(-logf(u_safe));
        float score = logf(p) + g;
        if (score > best_score) {
            best_score = score;
            picked = i;
        }
    }
    arena[out_off + row] = (float)picked;
}
