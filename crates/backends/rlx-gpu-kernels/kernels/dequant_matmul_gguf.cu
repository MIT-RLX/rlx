// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0
// Fused GGUF K-quant dequant + GEMV (decode, m == 1).
// Mirrors rlx-vulkan/shaders/dequant_matmul.comp and rlx-cpu gguf_matmul_bt.
//
// Schemes: 0 = Q4_K (256 elems / 144 B), 2 = Q6_K (256 / 210 B), 20 = Q8_0,
// 24 = Q1_0 (128 elems / 18 B) — Bonsai-27B decode path.

// 64-bit weight-base index: packed 27B arenas exceed 4 GiB, so a u32
// `w_word + (rel>>2)` truncates and the GEMV reads the wrong bytes.
// Portable full-lane shuffle mask. HIP static_asserts that the mask is a
// 64-BIT integer (AMD wavefronts are 64 lanes), so the CUDA-idiomatic
// `0xFFFFFFFFu` fails to compile under hipRTC with
// "The mask must be a 64-bit integer" — which kept this whole translation
// unit, and therefore the fused GEMV, off ROCm entirely.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_DEVICE_COMPILE__)
#define RLX_SHFL_ALL_LANES 0xFFFFFFFFFFFFFFFFull
#else
#define RLX_SHFL_ALL_LANES 0xFFFFFFFFu
#endif

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

// Cooperative Q2_0 GEMV (PrismML Ternary-Bonsai / Doses AI Pestle factors).
// One block per output row; threads split the k/128 blocks. Fusing dequant
// into the dot is what makes this worth having: the scratch path materializes
// the whole [n, k] weight as f32 first, which for Pestle's 248320-row lm_head
// is a 5.1 GiB slab that will not fit beside the packed weights on a 16 GiB
// card. Block layout: f16 scale + 32 bytes of 2-bit codes, value = (q-1)*d.
// Neumaier-compensated accumulation + tree reduce, matching the Q1_0 path.
extern "C" __global__ void dequant_matmul_gguf_q2_gemv(
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
        unsigned long long base = (j * blocks_per_row + r) * 34ull;
        float d = rd_f16(arena, w_word, base);
        unsigned long long xb = x_off + r * 128ull;
        #pragma unroll 4
        for (unsigned int byte = 0u; byte < 32u; byte++) {
            unsigned int qs = rd_byte(arena, w_word, base + 2ull + (unsigned long long)byte);
            #pragma unroll
            for (unsigned int c = 0u; c < 4u; c++) {
                float w = (float)((int)((qs >> (2u * c)) & 3u) - 1) * d;
                float xv = arena[xb + (unsigned long long)(byte * 4u + c)];
                float p = xv * w;
                float e = __fmaf_rn(xv, w, -p);
                neumaier_add(acc, comp, p);
                comp += e;
            }
        }
    }

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

// Cooperative G8_0 GEMV (Doses AI Pestle `token_embd` / lm_head). 32 elements
// per block / 16 bytes: four **bf16** scales, one per group of 8, then 8 bytes
// of 2-bit codes; value = (q-1)*bf16(d[j/8]). The per-group-of-8 scale is the
// whole point of the format — a single block-wide scale reads as garbage.
extern "C" __global__ void dequant_matmul_gguf_g8_gemv(
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
    unsigned long long blocks_per_row = k / 32ull;
    float acc = 0.0f;
    float comp = 0.0f;

    for (unsigned long long r = threadIdx.x; r < blocks_per_row; r += blockDim.x) {
        unsigned long long base = (j * blocks_per_row + r) * 16ull;
        unsigned long long xb = x_off + r * 32ull;
        #pragma unroll
        for (unsigned int g = 0u; g < 4u; g++) {
            // bf16 = the top 16 bits of an f32.
            unsigned int lo = rd_byte(arena, w_word, base + (unsigned long long)(g * 2u));
            unsigned int hi = rd_byte(arena, w_word, base + (unsigned long long)(g * 2u + 1u));
            float d = __uint_as_float((lo | (hi << 8u)) << 16u);
            #pragma unroll
            for (unsigned int t = 0u; t < 8u; t++) {
                unsigned int e = g * 8u + t;
                unsigned int qs = rd_byte(arena, w_word, base + 8ull + (unsigned long long)(e >> 2u));
                float w = (float)((int)((qs >> (2u * (e & 3u))) & 3u) - 1) * d;
                float xv = arena[xb + (unsigned long long)e];
                float p = xv * w;
                float er = __fmaf_rn(xv, w, -p);
                neumaier_add(acc, comp, p);
                comp += er;
            }
        }
    }

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

// Cooperative Q4_K GEMV: one block per output row, threads split the k/256
// super-blocks (contiguous in the row-major weight → coalesced loads, unlike
// the one-thread-per-row `dequant_matmul_gguf` whose 32 lanes each stride a
// whole row). Dequant is fused into the dot on-the-fly (no `float blk[256]`
// local slab → no local-memory spill). Neumaier-compensated accumulation +
// tree reduce, matching the Q1_0 path — tighter than the plain-f32 `acc +=`
// of the scalar kernel, so decode argmaxes stay on the reference token path.
// 64-bit offsets for packed arenas > 4 GiB. Block size ≤ 256 (smem sizing).
extern "C" __global__ void dequant_matmul_gguf_q4k_gemv(
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
    unsigned long long blocks_per_row = k / 256ull;
    float acc = 0.0f;
    float comp = 0.0f;

    for (unsigned long long r = threadIdx.x; r < blocks_per_row; r += blockDim.x) {
        unsigned long long base = (j * blocks_per_row + r) * 144ull;
        float d = rd_f16(arena, w_word, base);
        float dmin = rd_f16(arena, w_word, base + 2ull);
        unsigned int sc_base = (unsigned int)(base + 4ull); // within-tensor byte off
        unsigned long long qs_base = base + 16ull;
        unsigned long long xb = x_off + r * 256ull;
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
                unsigned int q = rd_byte(arena, w_word, qs_base + (unsigned long long)(is + l));
                float w = d0 * (float)(q & 0x0Fu) - m0f;
                float xv = arena[xb + (unsigned long long)out_i];
                float p = xv * w;
                float e = __fmaf_rn(xv, w, -p);
                neumaier_add(acc, comp, p);
                comp += e;
                out_i++;
            }
            for (unsigned int l = 0u; l < 32u; l++) {
                unsigned int q = rd_byte(arena, w_word, qs_base + (unsigned long long)(is + l));
                float w = d1 * (float)(q >> 4u) - m1f;
                float xv = arena[xb + (unsigned long long)out_i];
                float p = xv * w;
                float e = __fmaf_rn(xv, w, -p);
                neumaier_add(acc, comp, p);
                comp += e;
                out_i++;
            }
            is += 32u;
        }
    }

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

// Warp-per-row Q4_K GEMV: one WARP (32 lanes) per output row; the 32 lanes
// split the 256 elements of each super-block (strided -> coalesced x + qs loads),
// so occupancy stays full at small k (k=1024 -> 4 super-blocks/row would leave
// the block-per-row `..._q4k_gemv` kernel with only 4 active lanes of 128).
// Same fused dequant math + per-lane Neumaier accumulation; warp-shuffle reduce.
// One thread block holds `blockDim.x / 32` output rows.
extern "C" __global__ void dequant_matmul_gguf_q4k_gemv_warp(
    float* arena,
    unsigned long long n,
    unsigned long long k,
    unsigned long long x_off,
    unsigned long long w_byte_off,
    unsigned long long out_off
) {
    const unsigned int WARP = 32u;
    unsigned int warps_per_block = blockDim.x / WARP;
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp_id = threadIdx.x >> 5u;
    unsigned long long j = (unsigned long long)blockIdx.x * warps_per_block + warp_id;
    if (j >= n) return;

    unsigned long long w_word = w_byte_off / 4ull;
    unsigned long long blocks_per_row = k / 256ull;
    float acc = 0.0f;
    float comp = 0.0f;

    for (unsigned long long r = 0ull; r < blocks_per_row; r++) {
        unsigned long long base = (j * blocks_per_row + r) * 144ull;
        float d = rd_f16(arena, w_word, base);
        float dmin = rd_f16(arena, w_word, base + 2ull);
        unsigned int sc_base = (unsigned int)(base + 4ull);
        unsigned long long qs_base = base + 16ull;
        unsigned long long xb = x_off + r * 256ull;
        // lane strides the 256 elements: oi = lane, lane+32, ... (8 per lane).
        for (unsigned int oi = lane; oi < 256u; oi += WARP) {
            unsigned int pair = oi >> 6u;      // 0..3 (which 64-elem low/high pair)
            unsigned int wp = oi & 63u;        // 0..63 within the pair
            unsigned int hi = wp >> 5u;        // 0 = low nibble, 1 = high nibble
            unsigned int l = wp & 31u;         // 0..31 byte index within sub-block
            unsigned int sub = pair * 2u + hi; // 0..7 sub-block (scale/min index)
            unsigned int sc, mn;
            scale_min_k4(arena, w_word, sub, sc_base, sc, mn);
            unsigned int q = rd_byte(arena, w_word, qs_base + (unsigned long long)(pair * 32u + l));
            unsigned int nib = hi ? (q >> 4u) : (q & 0x0Fu);
            float w = d * (float)sc * (float)nib - dmin * (float)mn;
            float xv = arena[xb + (unsigned long long)oi];
            float p = xv * w;
            float e = __fmaf_rn(xv, w, -p);
            neumaier_add(acc, comp, p);
            comp += e;
        }
    }

    // Warp-shuffle reduce acc and comp separately, then combine (mirrors the
    // block kernel's smem_acc[0] + smem_comp[0]).
    for (unsigned int off = WARP >> 1u; off > 0u; off >>= 1u) {
        acc += __shfl_down_sync(RLX_SHFL_ALL_LANES, acc, off);
        comp += __shfl_down_sync(RLX_SHFL_ALL_LANES, comp, off);
    }
    if (lane == 0u) {
        arena[out_off + j] = acc + comp;
    }
}

// 8 bytes from tensor byte `rel` at any alignment: three word loads (not
// eight `rd_byte`s), returned as two little-endian words.
__device__ __forceinline__ void rd8(
    const float* arena,
    unsigned long long w_word,
    unsigned long long rel,
    unsigned int& lo,
    unsigned int& hi
) {
    unsigned long long w = w_word + (rel >> 2ull);
    unsigned int a = __float_as_uint(arena[w]);
    unsigned int b = __float_as_uint(arena[w + 1ull]);
    unsigned int sh = (unsigned int)(rel & 3ull) * 8u;
    if (sh == 0u) {
        lo = a;
        hi = b;
        return;
    }
    unsigned int c = __float_as_uint(arena[w + 2ull]);
    lo = (a >> sh) | (b << (32u - sh));
    hi = (b >> sh) | (c << (32u - sh));
}

// f16 at tensor byte `rel`, by bits: normals and zero exactly, the rest
// through `rd_f16`. No `exp2f` on the common path.
__device__ __forceinline__ float rd_f16_bits(
    const float* arena,
    unsigned long long w_word,
    unsigned long long rel
) {
    unsigned int w = __float_as_uint(arena[w_word + (rel >> 2ull)]);
    unsigned int h = (w >> ((unsigned int)(rel & 3ull) * 8u)) & 0xFFFFu;
    unsigned int exp = (h >> 10u) & 0x1Fu;
    if (exp == 0u || exp == 31u) {
        return ((h & 0x7FFFu) == 0u) ? __uint_as_float((h & 0x8000u) << 16u)
                                      : rd_f16(arena, w_word, rel);
    }
    return __uint_as_float(((h & 0x8000u) << 16u) | ((exp + 112u) << 23u) | ((h & 0x3FFu) << 13u));
}

// Accumulate `x · w` over 8 consecutive weights of a GGUF row from element
// `e0` (a multiple of 8, so all 8 share one scale group): the block's scales
// decode once per chunk, not per element. `row_base` = the row's first byte.
// Q8_0 (20): 32 elems / 34 B, f16 d then 32 i8. Q4_K (0): 256 / 144 B.
// Q5_K (1): 256 / 176 B, Q4_K's layout plus a 32-byte plane of fifth bits.
// Q6_K (2): 256 / 210 B. Same decode as the scalar `dequant_matmul_gguf`
// above and rlx-gguf.
// The 8 weights at elements `e0..e0+8` of the row at `row_base`, decoded.
__device__ __forceinline__ void gguf_load8(
    const float* arena,
    unsigned long long w_word,
    unsigned long long row_base,
    unsigned int scheme_id,
    unsigned long long e0,
    float w[8]
) {
    if (scheme_id == 20u) {
        unsigned long long base = row_base + (e0 >> 5ull) * 34ull;
        float d = rd_f16_bits(arena, w_word, base);
        unsigned int q0, q1;
        rd8(arena, w_word, base + 2ull + (e0 & 31ull), q0, q1);
        #pragma unroll
        for (unsigned int t = 0u; t < 8u; t++) {
            unsigned int b = ((t < 4u) ? (q0 >> (t * 8u)) : (q1 >> ((t - 4u) * 8u))) & 0xFFu;
            w[t] = d * (float)sx8(b);
        }
    } else if (scheme_id <= 1u) {
        unsigned int q5 = scheme_id;
        unsigned int oi = (unsigned int)(e0 & 255ull);
        unsigned long long base = row_base + (e0 >> 8ull) * (q5 ? 176ull : 144ull);
        unsigned int pair = oi >> 6u;
        unsigned int hi = (oi >> 5u) & 1u;
        unsigned int l0 = oi & 31u;
        unsigned int sc, mn;
        scale_min_k4(arena, w_word, pair * 2u + hi, (unsigned int)(base + 4ull), sc, mn);
        float dl = rd_f16_bits(arena, w_word, base) * (float)sc;
        float ml = rd_f16_bits(arena, w_word, base + 2ull) * (float)mn;
        unsigned int q0, q1, h0 = 0u, h1 = 0u;
        rd8(arena, w_word, base + (q5 ? 48ull : 16ull) + (unsigned long long)(pair * 32u + l0), q0, q1);
        if (q5) {
            rd8(arena, w_word, base + 16ull + (unsigned long long)l0, h0, h1);
        }
        unsigned int bit = pair * 2u + hi;
        #pragma unroll
        for (unsigned int t = 0u; t < 8u; t++) {
            unsigned int sh = (t & 3u) * 8u;
            unsigned int q = ((t < 4u) ? (q0 >> sh) : (q1 >> sh)) & 0xFFu;
            unsigned int nib = hi ? (q >> 4u) : (q & 0x0Fu);
            if (q5) {
                unsigned int qh = ((t < 4u) ? (h0 >> sh) : (h1 >> sh)) & 0xFFu;
                nib += ((qh >> bit) & 1u) << 4u;
            }
            w[t] = dl * (float)nib - ml;
        }
    } else {
        // Q6_K: two halves of 128; in each, four runs of 32 (`quad`).
        unsigned int oi = (unsigned int)(e0 & 255ull);
        unsigned long long base = row_base + (e0 >> 8ull) * 210ull;
        unsigned int h = oi >> 7u;
        unsigned int quad = (oi >> 5u) & 3u;
        unsigned int l0 = oi & 31u;
        float dl = rd_f16_bits(arena, w_word, base + 208ull)
            * (float)sx8(rd_byte(arena, w_word, base + 192ull + (unsigned long long)(h * 8u + (l0 >> 4u) + quad * 2u)));
        unsigned int l0w, l1w, h0w, h1w;
        rd8(arena, w_word, base + (unsigned long long)(h * 64u + l0 + (quad & 1u) * 32u), l0w, l1w);
        rd8(arena, w_word, base + 128ull + (unsigned long long)(h * 32u + l0), h0w, h1w);
        unsigned int lo_shift = (quad >> 1u) * 4u;
        unsigned int qh_shift = quad * 2u;
        #pragma unroll
        for (unsigned int t = 0u; t < 8u; t++) {
            unsigned int sh = (t & 3u) * 8u;
            unsigned int lo = ((t < 4u) ? (l0w >> sh) : (l1w >> sh)) & 0xFFu;
            unsigned int qhb = ((t < 4u) ? (h0w >> sh) : (h1w >> sh)) & 0xFFu;
            int q = (int)(((lo >> lo_shift) & 0x0Fu) | (((qhb >> qh_shift) & 3u) << 4u)) - 32;
            w[t] = dl * (float)q;
        }
    }
}

// `x[e0..e0+8] · w`, compensated into `acc` / `comp`. Where `x` is 16-byte
// aligned (`x4`) the 8 values come in two vector loads rather than eight:
// the same values, so the same sums.
__device__ __forceinline__ void gguf_acc8(
    const float* w,
    const float* x,
    unsigned long long e0,
    float& acc,
    float& comp,
    bool x4 = false
) {
    float xs[8];
    if (x4) {
        float4 a = *reinterpret_cast<const float4*>(x + e0);
        float4 b = *reinterpret_cast<const float4*>(x + e0 + 4ull);
        xs[0] = a.x; xs[1] = a.y; xs[2] = a.z; xs[3] = a.w;
        xs[4] = b.x; xs[5] = b.y; xs[6] = b.z; xs[7] = b.w;
    } else {
        #pragma unroll
        for (unsigned int t = 0u; t < 8u; t++) {
            xs[t] = x[e0 + t];
        }
    }
    #pragma unroll
    for (unsigned int t = 0u; t < 8u; t++) {
        float xv = xs[t];
        float p = xv * w[t];
        float er = __fmaf_rn(xv, w[t], -p);
        neumaier_add(acc, comp, p);
        comp += er;
    }
}

__device__ __forceinline__ void gguf_dot8(
    const float* arena,
    unsigned long long w_word,
    unsigned long long row_base,
    unsigned int scheme_id,
    unsigned long long e0,
    const float* x,
    float& acc,
    float& comp
) {
    float w[8];
    gguf_load8(arena, w_word, row_base, scheme_id, e0, w);
    gguf_acc8(w, x, e0, acc, comp);
}

// The compensated sum of one output row's 32 lanes (an aligned group of 32
// threads), at its first lane: the pairwise tree lane i += lane i + s for
// s = 16 … 1, each pair merged Neumaier-style. Register shuffles (width 32,
// so a 64-lane wavefront holds two rows) instead of shared memory: the old
// form paid five block-wide barriers per row, a second set for each further
// activation row, and that — not the weight bytes — set a 3584×1024 Q8_0
// GEMV at ~30 GB/s on a 780M. Same pairs in the same order, so the same bits.
__device__ __forceinline__ float row32_sum(float acc, float comp) {
    #pragma unroll
    for (unsigned int s = 16u; s > 0u; s >>= 1u) {
        float a2 = __shfl_down_sync(RLX_SHFL_ALL_LANES, acc, s, 32);
        float c2 = __shfl_down_sync(RLX_SHFL_ALL_LANES, comp, s, 32);
        float ssum = acc + a2;
        float err = (fabsf(acc) >= fabsf(a2)) ? ((acc - ssum) + a2) : ((a2 - ssum) + acc);
        acc = ssum;
        comp = comp + c2 + err;
    }
    return acc + comp;
}

// Decode GEMV (m == 1) for Q8_0 / Q4_K / Q5_K / Q6_K: 32 threads per output
// row, `blockDim.x / 32` rows per block, each thread taking chunks of 8
// consecutive elements — every thread works at any k (the block-strided
// kernels leave all but k/256 of 128 threads idle at k = 1024), and one
// block-wide reduction serves several rows. Neumaier accumulation and tree
// reduce as in the kernels above. Needs k % 8 == 0; block size ≤ 256 and a
// multiple of 32 (smem sizing). Launch ceil(n / (blockDim.x / 32)) blocks.
extern "C" __global__ void dequant_matmul_gguf_gemv_elem(
    float* arena,
    unsigned long long n,
    unsigned long long k,
    unsigned long long x_off,
    unsigned long long w_byte_off,
    unsigned long long out_off,
    unsigned int scheme_id
) {
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31u;
    unsigned long long j = (unsigned long long)blockIdx.x * (blockDim.x >> 5u) + (tid >> 5u);

    unsigned long long w_word = w_byte_off / 4ull;
    unsigned long long row_bytes = (scheme_id == 20u) ? (k / 32ull) * 34ull
        : (k / 256ull) * ((scheme_id == 0u) ? 144ull : (scheme_id == 1u) ? 176ull : 210ull);
    float acc = 0.0f;
    float comp = 0.0f;
    // No early return: every thread reaches the barriers below.
    if (j < n) {
        unsigned long long row_base = j * row_bytes;
        bool x4 = (x_off & 3ull) == 0ull;
        for (unsigned long long e0 = (unsigned long long)lane * 8ull; e0 < k; e0 += 256ull) {
            float w[8];
            gguf_load8(arena, w_word, row_base, scheme_id, e0, w);
            gguf_acc8(w, arena + x_off, e0, acc, comp, x4);
        }
    }
    // Every lane of a row shuffles (a row is all in or all out of range).
    float sum = row32_sum(acc, comp);
    if (lane == 0u && j < n) {
        arena[out_off + j] = sum;
    }
}

// `dequant_matmul_gguf_gemv_elem` for `m` (≤ 4) activation rows at once, a
// speculative verify step's: each weight chunk is decoded once and dotted with
// every row. A row's sums run in the m = 1 kernel's order, so each output row
// is that kernel's bit for bit.
#define GGUF_GEMV_MAX_ROWS 4u
extern "C" __global__ void dequant_matmul_gguf_gemv_rows(
    float* arena,
    unsigned long long m,
    unsigned long long n,
    unsigned long long k,
    unsigned long long x_off,
    unsigned long long w_byte_off,
    unsigned long long out_off,
    unsigned int scheme_id
) {
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31u;
    unsigned long long j = (unsigned long long)blockIdx.x * (blockDim.x >> 5u) + (tid >> 5u);

    unsigned long long w_word = w_byte_off / 4ull;
    unsigned long long row_bytes = (scheme_id == 20u) ? (k / 32ull) * 34ull
        : (k / 256ull) * ((scheme_id == 0u) ? 144ull : (scheme_id == 1u) ? 176ull : 210ull);
    float acc[GGUF_GEMV_MAX_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    float comp[GGUF_GEMV_MAX_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    // No early return: every thread reaches the barriers below.
    if (j < n) {
        unsigned long long row_base = j * row_bytes;
        // Row r's x starts r·k floats on: aligned for every row when k % 4 == 0.
        bool x4 = (x_off & 3ull) == 0ull && (k & 3ull) == 0ull;
        for (unsigned long long e0 = (unsigned long long)lane * 8ull; e0 < k; e0 += 256ull) {
            float w[8];
            gguf_load8(arena, w_word, row_base, scheme_id, e0, w);
            #pragma unroll
            for (unsigned int r = 0u; r < GGUF_GEMV_MAX_ROWS; r++) {
                if (r < m) {
                    gguf_acc8(w, arena + x_off + r * k, e0, acc[r], comp[r], x4);
                }
            }
        }
    }

    #pragma unroll
    for (unsigned int r = 0u; r < GGUF_GEMV_MAX_ROWS; r++) {
        if (r < m) {
            float sum = row32_sum(acc[r], comp[r]);
            if (lane == 0u && j < n) {
                arena[out_off + r * n + j] = sum;
            }
        }
    }
}
