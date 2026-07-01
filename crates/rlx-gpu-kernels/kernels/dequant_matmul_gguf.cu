// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// GPL-3.0-only. Fused GGUF K-quant dequant + GEMV (decode, m == 1).
// Mirrors rlx-vulkan/shaders/dequant_matmul.comp and rlx-cpu gguf_matmul_bt.

__device__ __forceinline__ unsigned int rd_byte(const float* arena, unsigned int w_word, unsigned int rel) {
    unsigned int w = __float_as_uint(arena[w_word + (rel >> 2u)]);
    return (w >> ((rel & 3u) * 8u)) & 0xFFu;
}

__device__ __forceinline__ float rd_f16(const float* arena, unsigned int w_word, unsigned int rel) {
    unsigned int w = __float_as_uint(arena[w_word + (rel >> 2u)]);
    unsigned int h = (w >> ((rel & 3u) * 8u)) & 0xFFFFu;
    unsigned int sign = (h >> 15u) & 1u;
    unsigned int exp = (h >> 10u) & 0x1Fu;
    unsigned int mant = (unsigned int)h & 0x3FFu;
    float v;
    if (exp == 0u) {
        v = (float)mant / 1024.0f * exp2f(-14.0f);
    } else if (exp == 31u) {
        v = (mant == 0u) ? __int_as_float(0x7f800000) : 0.0f;
    } else {
        v = (1.0f + (float)mant / 1024.0f) * exp2f((float)((int)exp - 15));
    }
    return (sign != 0u) ? -v : v;
}

__device__ __forceinline__ int sx8(unsigned int b) {
    return (int)b - 256 * (int)((b >> 7u) & 1u);
}

__device__ __forceinline__ void scale_min_k4(
    const float* arena,
    unsigned int w_word,
    unsigned int j,
    unsigned int sc_base,
    unsigned int& sc,
    unsigned int& mn
) {
    if (j < 4u) {
        sc = rd_byte(arena, w_word, sc_base + j) & 63u;
        mn = rd_byte(arena, w_word, sc_base + j + 4u) & 63u;
    } else {
        unsigned int a = rd_byte(arena, w_word, sc_base + j + 4u);
        sc = (a & 0x0Fu) | ((rd_byte(arena, w_word, sc_base + j - 4u) >> 6u) << 4u);
        mn = (a >> 4u) | ((rd_byte(arena, w_word, sc_base + j) >> 6u) << 4u);
    }
}

extern "C" __global__ void dequant_matmul_gguf(
    float* arena,
    unsigned int n,
    unsigned int k,
    unsigned int x_off,
    unsigned int w_byte_off,
    unsigned int out_off,
    unsigned int scheme_id
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    unsigned int w_word = w_byte_off / 4u;
    unsigned int blocks_per_row = k / 256u;
    unsigned int block_bytes = (scheme_id == 0u) ? 144u : 210u;

    float blk[256];
    float acc = 0.0f;

    for (unsigned int r = 0u; r < blocks_per_row; r++) {
        unsigned int gbi = j * blocks_per_row + r;
        unsigned int base = gbi * block_bytes;

        if (scheme_id == 0u) {
            float d = rd_f16(arena, w_word, base);
            float dmin = rd_f16(arena, w_word, base + 2u);
            unsigned int sc_base = base + 4u;
            unsigned int qs_base = base + 16u;
            unsigned int out_i = 0u;
            unsigned int is = 0u;
            for (unsigned int jj = 0u; jj < 8u; jj += 2u) {
                unsigned int sc0, m0, sc1, m1;
                scale_min_k4(arena, w_word, jj, sc_base, sc0, m0);
                scale_min_k4(arena, w_word, jj + 1u, sc_base, sc1, m1);
                float d0 = d * (float)sc0;
                float m0f = dmin * (float)m0;
                float d1 = d * (float)sc1;
                float m1f = dmin * (float)m1;
                for (unsigned int l = 0u; l < 32u; l++) {
                    unsigned int q = rd_byte(arena, w_word, qs_base + is + l);
                    blk[out_i++] = d0 * (float)(q & 0x0Fu) - m0f;
                }
                for (unsigned int l = 0u; l < 32u; l++) {
                    unsigned int q = rd_byte(arena, w_word, qs_base + is + l);
                    blk[out_i++] = d1 * (float)(q >> 4u) - m1f;
                }
                is += 32u;
            }
        } else {
            unsigned int ql_base = base;
            unsigned int qh_base = base + 128u;
            unsigned int sc_base = base + 192u;
            float d = rd_f16(arena, w_word, base + 208u);
            for (unsigned int h = 0u; h < 2u; h++) {
                unsigned int dst_base = h * 128u;
                unsigned int ql_off = h * 64u;
                unsigned int qh_off = h * 32u;
                unsigned int sc_off = h * 8u;
                for (unsigned int l = 0u; l < 32u; l++) {
                    unsigned int isb = l / 16u;
                    unsigned int qhb = rd_byte(arena, w_word, qh_base + qh_off + l);
                    unsigned int lo0 = rd_byte(arena, w_word, ql_base + ql_off + l);
                    unsigned int lo1 = rd_byte(arena, w_word, ql_base + ql_off + l + 32u);
                    int q1 = (int)((lo0 & 0x0Fu) | ((qhb & 3u) << 4u)) - 32;
                    int q2 = (int)((lo1 & 0x0Fu) | (((qhb >> 2u) & 3u) << 4u)) - 32;
                    int q3 = (int)((lo0 >> 4u) | (((qhb >> 4u) & 3u) << 4u)) - 32;
                    int q4 = (int)((lo1 >> 4u) | (((qhb >> 6u) & 3u) << 4u)) - 32;
                    blk[dst_base + l] =
                        d * (float)sx8(rd_byte(arena, w_word, sc_base + sc_off + isb)) * (float)q1;
                    blk[dst_base + l + 32u] =
                        d * (float)sx8(rd_byte(arena, w_word, sc_base + sc_off + isb + 2u)) * (float)q2;
                    blk[dst_base + l + 64u] =
                        d * (float)sx8(rd_byte(arena, w_word, sc_base + sc_off + isb + 4u)) * (float)q3;
                    blk[dst_base + l + 96u] =
                        d * (float)sx8(rd_byte(arena, w_word, sc_base + sc_off + isb + 6u)) * (float)q4;
                }
            }
        }

        unsigned int xb = x_off + r * 256u;
        for (unsigned int t = 0u; t < 256u; t++) {
            acc += arena[xb + t] * blk[t];
        }
    }

    arena[out_off + j] = acc;
}
