// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// GPL-3.0-only. Fused GGUF K-quant dequant + GEMV (decode, m == 1).
// Mirrors rlx-vulkan/shaders/dequant_matmul.comp and rlx-cpu gguf_matmul_bt.
//
// Schemes: 0 = Q4_K (256 elems / 144 B), 2 = Q6_K (256 / 210 B),
// 24 = Q1_0 (128 elems / 18 B) — Bonsai-27B decode path.

// 64-bit weight-base index: packed 27B arenas exceed 4 GiB, so a u32
// `w_word + (rel>>2)` truncates and the GEMV reads the wrong bytes.
__device__ __forceinline__ unsigned int rd_byte(
    const float* arena,
    unsigned long long w_word,
    unsigned long long rel
) {
    unsigned int w = __float_as_uint(arena[w_word + (rel >> 2ull)]);
    return (w >> ((rel & 3ull) * 8ull)) & 0xFFu;
}

__device__ __forceinline__ float rd_f16(
    const float* arena,
    unsigned long long w_word,
    unsigned long long rel
) {
    unsigned int w = __float_as_uint(arena[w_word + (rel >> 2ull)]);
    unsigned int h = (w >> ((unsigned int)(rel & 3ull) * 8u)) & 0xFFFFu;
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

// Neumaier compensated sum — same scheme as matmul_bt.cu `precise`.
// Plain f32 reduction on Q1_0 (coarse ±d weights) can flip near-tie argmaxes
// vs dequant+matmul / Metal / CPU.
__device__ __forceinline__ void neumaier_add(float& acc, float& comp, float x) {
    float s = acc + x;
    float err = (fabsf(acc) >= fabsf(x)) ? ((acc - s) + x) : ((x - s) + acc);
    acc = s;
    comp += err;
}

__device__ __forceinline__ int sx8(unsigned int b) {
    return (int)b - 256 * (int)((b >> 7u) & 1u);
}

__device__ __forceinline__ void scale_min_k4(
    const float* arena,
    unsigned long long w_word,
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

    unsigned long long w_word = (unsigned long long)(w_byte_off / 4u);
    float acc = 0.0f;
    float comp = 0.0f;

    // Q1_0: 128 elems / 18 bytes — fuse dequant into the dot (no local slab).
    if (scheme_id == 24u) {
        unsigned int blocks_per_row = k / 128u;
        for (unsigned int r = 0u; r < blocks_per_row; r++) {
            unsigned long long base =
                ((unsigned long long)j * (unsigned long long)blocks_per_row + (unsigned long long)r)
                * 18ull;
            float d = rd_f16(arena, w_word, base);
            float neg_d = -d;
            unsigned int xb = x_off + r * 128u;
            #pragma unroll 4
            for (unsigned int byte = 0u; byte < 16u; byte++) {
                unsigned int qs = rd_byte(arena, w_word, base + 2ull + (unsigned long long)byte);
                #pragma unroll
                for (unsigned int bit = 0u; bit < 8u; bit++) {
                    float w = ((qs >> bit) & 1u) ? d : neg_d;
                    float p = arena[xb + byte * 8u + bit] * w;
                    float e = __fmaf_rn(arena[xb + byte * 8u + bit], w, -p);
                    neumaier_add(acc, comp, p);
                    comp += e;
                }
            }
        }
        arena[out_off + j] = acc + comp;
        return;
    }

    unsigned int blocks_per_row = k / 256u;
    unsigned int block_bytes = (scheme_id == 0u) ? 144u : 210u;
    float blk[256];

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


// Cooperative Q1_0 GEMV: one block per output row, threads split weight blocks.
// Uses 64-bit offsets so packed arenas >4 GiB address correctly.
// Compensated (Neumaier) accumulation matches matmul_bt precise / Metal
// dequant+matmul so decode argmaxes stay on the Metal token path.
extern "C" __global__ void dequant_matmul_gguf_q1_gemv(
    float* arena,
    unsigned long long n,
    unsigned long long k,
    unsigned long long x_off,
    unsigned long long w_byte_off,
    unsigned long long out_off
) {
    unsigned long long j = blockIdx.x;
    if (j >= n) return;

    unsigned long long w_word = w_byte_off / 4ull;
    unsigned long long blocks_per_row = k / 128ull;
    float acc = 0.0f;
    float comp = 0.0f;

    for (unsigned long long r = threadIdx.x; r < blocks_per_row; r += blockDim.x) {
        unsigned long long base = (j * blocks_per_row + r) * 18ull;
        float d = rd_f16(arena, w_word, base);
        float neg_d = -d;
        unsigned long long xb = x_off + r * 128ull;
        #pragma unroll 4
        for (unsigned int byte = 0u; byte < 16u; byte++) {
            unsigned int qs = rd_byte(arena, w_word, base + 2ull + (unsigned long long)byte);
            #pragma unroll
            for (unsigned int bit = 0u; bit < 8u; bit++) {
                float w = ((qs >> bit) & 1u) ? d : neg_d;
                float xv = arena[xb + (unsigned long long)(byte * 8u + bit)];
                float p = xv * w;
                float e = __fmaf_rn(xv, w, -p);
                neumaier_add(acc, comp, p);
                comp += e;
            }
        }
    }

    // Tree-reduce (acc, comp) pairs — fold compensation after each add.
    __shared__ float smem_acc[256];
    __shared__ float smem_comp[256];
    unsigned int tid = threadIdx.x;
    smem_acc[tid] = acc;
    smem_comp[tid] = comp;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) {
            float a = smem_acc[tid];
            float c = smem_comp[tid];
            float a2 = smem_acc[tid + s];
            float c2 = smem_comp[tid + s];
            float ssum = a + a2;
            float err = (fabsf(a) >= fabsf(a2)) ? ((a - ssum) + a2) : ((a2 - ssum) + a);
            smem_acc[tid] = ssum;
            smem_comp[tid] = c + c2 + err;
        }
        __syncthreads();
    }
    if (tid == 0u) {
        arena[out_off + j] = smem_acc[0] + smem_comp[0];
    }
}
