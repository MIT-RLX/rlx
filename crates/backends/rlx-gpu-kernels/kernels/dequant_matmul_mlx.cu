// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Dequant-on-the-fly matmul for MLX weight packs (mlx-lm Linear layout).
//
// Weight storage: [n, k] packed along K (row j → output column j).
//   kind 0 = MlxAffine  — unsigned codes, f32 scale+bias per (n, group)
//   kind 1 = MlxMxfp4   — E2M1 nibbles × u8 group scale
//   kind 2 = MlxMxfp8   — E4M3 bytes × u8 group scale
//
// Scale u8 decode: group_size==16 → FP8 E4M3; else E8M0 (bf16 s<<7).
// Affine: out = scale * code + bias (MLX affine_dequantize).
//
// Arena: packed bytes via bitcast of f32 words; x/out/affine scales are f32.

__device__ __forceinline__ unsigned int mlx_rd_byte(
    const float* arena,
    unsigned long long byte_off
) {
    unsigned int w = __float_as_uint(arena[byte_off >> 2ull]);
    return (w >> ((unsigned int)(byte_off & 3ull) * 8u)) & 0xffu;
}

/// 32-bit word at a 4-byte-aligned arena byte offset (8 packed E2M1 nibbles).
__device__ __forceinline__ unsigned int mlx_rd_word(
    const float* arena,
    unsigned long long byte_off
) {
    return __float_as_uint(arena[byte_off >> 2ull]);
}

__device__ __forceinline__ float mlx_fp4_lut(unsigned int nib) {
    static const float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return lut[nib & 0xfu];
}

__device__ __forceinline__ float mlx_decode_e4m3(unsigned int byte) {
    unsigned int sign = (byte >> 7) & 1u;
    int exp = (int)((byte >> 3) & 0xfu);
    unsigned int mant = byte & 0x7u;
    if (exp == 0x0f && mant == 0x7u) {
        // Match host `dequant_scale_fp8_e4m3` — return NaN (not 0).
        return nanf("");
    }
    if (exp == 0) {
        if (mant == 0u) {
            return sign ? -0.0f : 0.0f;
        }
        unsigned int m = mant;
        int e = -6;
        while ((m & 0x8u) == 0u) {
            m <<= 1;
            e -= 1;
        }
        m &= 0x7u;
        unsigned int bits = (sign << 31) | ((unsigned int)(e + 127) << 23) | (m << 20);
        return __uint_as_float(bits);
    }
    unsigned int bits = (sign << 31) | ((unsigned int)(exp - 7 + 127) << 23) | (mant << 20);
    return __uint_as_float(bits);
}

// MLX E8M0 group scale: s==0 → bf16(0x40); else bf16(s<<7) → 2^(s-127).
__device__ __forceinline__ float mlx_decode_e8m0(unsigned int s) {
    if (s == 0u) {
        return __uint_as_float(0x0040u << 16);
    }
    return __uint_as_float(s << 23);
}

__device__ __forceinline__ float mlx_group_scale(unsigned int s, unsigned int gs) {
    return (gs == 16u) ? mlx_decode_e4m3(s) : mlx_decode_e8m0(s);
}

__device__ __forceinline__ unsigned int mlx_pack_factor(unsigned int bits) {
    if (bits == 2u || bits == 4u || bits == 8u) return 8u / bits;
    if (bits == 3u || bits == 5u) return 8u;
    if (bits == 6u) return 4u;
    return 1u;
}

__device__ __forceinline__ unsigned int mlx_bytes_per_pack(unsigned int bits) {
    if (bits == 2u || bits == 4u || bits == 8u) return 1u;
    if (bits == 3u || bits == 6u) return 3u;
    if (bits == 5u) return 5u;
    return 1u;
}

__device__ __forceinline__ void mlx_extract_bits_3(unsigned int b0, unsigned int b1, unsigned int b2, unsigned int* out) {
    out[0] = b0 & 0x7u;
    out[1] = (b0 & 0x38u) >> 3;
    out[2] = ((b0 & 0xc0u) >> 6) + ((b1 & 0x1u) << 2);
    out[3] = (b1 & 0xeu) >> 1;
    out[4] = (b1 & 0x70u) >> 4;
    out[5] = ((b1 & 0x80u) >> 7) + ((b2 & 0x3u) << 1);
    out[6] = (b2 & 0x1cu) >> 2;
    out[7] = (b2 & 0xe0u) >> 5;
}

__device__ __forceinline__ void mlx_extract_bits_5(
    unsigned int b0, unsigned int b1, unsigned int b2, unsigned int b3, unsigned int b4,
    unsigned int* out
) {
    out[0] = b0 & 0x1fu;
    out[1] = ((b0 & 0xe0u) >> 5) + ((b1 & 0x3u) << 3);
    out[2] = (b1 & 0x7cu) >> 2;
    out[3] = ((b1 & 0x80u) >> 7) + ((b2 & 0xfu) << 1);
    out[4] = ((b2 & 0xf0u) >> 4) + ((b3 & 0x1u) << 4);
    out[5] = (b3 & 0x3eu) >> 1;
    out[6] = ((b3 & 0xc0u) >> 6) + ((b4 & 0x7u) << 2);
    out[7] = (b4 & 0xf8u) >> 3;
}

__device__ __forceinline__ void mlx_extract_bits_6(unsigned int b0, unsigned int b1, unsigned int b2, unsigned int* out) {
    out[0] = b0 & 0x3fu;
    out[1] = ((b0 >> 6) & 0x03u) + ((b1 & 0x0fu) << 2);
    out[2] = ((b1 >> 4) & 0x0fu) + ((b2 & 0x03u) << 4);
    out[3] = (b2 >> 2) & 0x3fu;
}

__device__ __forceinline__ float mlx_affine_w(
    const float* arena,
    unsigned long long w_byte_off,
    unsigned int bits,
    unsigned int gs,
    unsigned int n_groups,
    unsigned int j,
    unsigned int p
) {
    unsigned int pf = mlx_pack_factor(bits);
    unsigned int bpp = mlx_bytes_per_pack(bits);
    unsigned int packs_in_group = gs / pf;
    unsigned int g = p / gs;
    unsigned int local = p % gs;
    unsigned long long row_base =
        (unsigned long long)j * (unsigned long long)n_groups * (unsigned long long)packs_in_group * (unsigned long long)bpp
        + (unsigned long long)g * (unsigned long long)packs_in_group * (unsigned long long)bpp;

    unsigned int code = 0u;
    if (bits == 2u || bits == 4u || bits == 8u) {
        unsigned int pack_idx = local / pf;
        unsigned int in_pack = local % pf;
        unsigned int byte = mlx_rd_byte(arena, w_byte_off + row_base + pack_idx);
        unsigned int mask = (1u << bits) - 1u;
        code = (byte >> (in_pack * bits)) & mask;
    } else if (bits == 3u) {
        unsigned int pack_idx = local / 8u;
        unsigned int in_pack = local % 8u;
        unsigned long long bo = w_byte_off + row_base + (unsigned long long)pack_idx * 3ull;
        unsigned int codes[8];
        mlx_extract_bits_3(
            mlx_rd_byte(arena, bo),
            mlx_rd_byte(arena, bo + 1ull),
            mlx_rd_byte(arena, bo + 2ull),
            codes);
        code = codes[in_pack];
    } else if (bits == 5u) {
        unsigned int pack_idx = local / 8u;
        unsigned int in_pack = local % 8u;
        unsigned long long bo = w_byte_off + row_base + (unsigned long long)pack_idx * 5ull;
        unsigned int codes[8];
        mlx_extract_bits_5(
            mlx_rd_byte(arena, bo),
            mlx_rd_byte(arena, bo + 1ull),
            mlx_rd_byte(arena, bo + 2ull),
            mlx_rd_byte(arena, bo + 3ull),
            mlx_rd_byte(arena, bo + 4ull),
            codes);
        code = codes[in_pack];
    } else {
        // bits == 6
        unsigned int pack_idx = local / 4u;
        unsigned int in_pack = local % 4u;
        unsigned long long bo = w_byte_off + row_base + (unsigned long long)pack_idx * 3ull;
        unsigned int codes[4];
        mlx_extract_bits_6(
            mlx_rd_byte(arena, bo),
            mlx_rd_byte(arena, bo + 1ull),
            mlx_rd_byte(arena, bo + 2ull),
            codes);
        code = codes[in_pack];
    }
    return (float)code;
}

extern "C" __global__ void dequant_matmul_mlx(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int kind,
    unsigned int bits,
    unsigned int group_size,
    unsigned long long x_byte_off,
    unsigned long long w_byte_off,
    unsigned long long scale_byte_off,
    unsigned long long zp_byte_off,
    unsigned long long out_byte_off
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    unsigned int gs = group_size;
    unsigned int n_groups = k / gs;
    unsigned long long x_off = x_byte_off / 4ull;
    unsigned long long out_off = out_byte_off / 4ull;
    unsigned long long scale_f_off = scale_byte_off / 4ull;
    unsigned long long zp_f_off = zp_byte_off / 4ull;

    float acc = 0.0f;
    for (unsigned int p = 0u; p < k; ++p) {
        unsigned int g = p / gs;
        float w_dq;
        if (kind == 0u) {
            float code = mlx_affine_w(arena, (unsigned long long)w_byte_off, bits, gs, n_groups, col, p);
            float scale = arena[scale_f_off + col * n_groups + g];
            float bias = arena[zp_f_off + col * n_groups + g];
            w_dq = scale * code + bias;
        } else if (kind == 1u) {
            // mxfp4: 2 nibbles / byte along K
            unsigned long long bidx =
                (unsigned long long)col * (unsigned long long)(k / 2u) + (unsigned long long)(p / 2u);
            unsigned int byte = mlx_rd_byte(arena, (unsigned long long)w_byte_off + bidx);
            unsigned int nib = ((p & 1u) == 0u) ? (byte & 0x0fu) : (byte >> 4);
            unsigned int sb = mlx_rd_byte(
                arena,
                (unsigned long long)scale_byte_off + (unsigned long long)col * (unsigned long long)n_groups
                    + (unsigned long long)g);
            w_dq = mlx_fp4_lut(nib) * mlx_group_scale(sb, gs);
        } else {
            // mxfp8: one E4M3 byte per weight
            unsigned long long bidx =
                (unsigned long long)col * (unsigned long long)k + (unsigned long long)p;
            unsigned int wb = mlx_rd_byte(arena, (unsigned long long)w_byte_off + bidx);
            unsigned int sb = mlx_rd_byte(
                arena,
                (unsigned long long)scale_byte_off + (unsigned long long)col * (unsigned long long)n_groups
                    + (unsigned long long)g);
            w_dq = mlx_decode_e4m3(wb) * mlx_group_scale(sb, gs);
        }
        acc += arena[x_off + row * k + p] * w_dq;
    }
    arena[out_off + row * n + col] = acc;
}

// Decode GEMV (m == 1): one block per output column, threads split K.
extern "C" __global__ void dequant_matmul_mlx_gemv(
    float* arena,
    unsigned int k,
    unsigned int n,
    unsigned int kind,
    unsigned int bits,
    unsigned int group_size,
    unsigned long long x_byte_off,
    unsigned long long w_byte_off,
    unsigned long long scale_byte_off,
    unsigned long long zp_byte_off,
    unsigned long long out_byte_off
) {
    unsigned int col = blockIdx.x;
    if (col >= n) return;

    unsigned int gs = group_size;
    unsigned int n_groups = k / gs;
    unsigned long long x_off = x_byte_off / 4ull;
    unsigned long long out_off = out_byte_off / 4ull;
    unsigned long long scale_f_off = scale_byte_off / 4ull;
    unsigned long long zp_f_off = zp_byte_off / 4ull;

    float acc = 0.0f;
    for (unsigned int p = threadIdx.x; p < k; p += blockDim.x) {
        unsigned int g = p / gs;
        float w_dq;
        if (kind == 0u) {
            float code = mlx_affine_w(arena, (unsigned long long)w_byte_off, bits, gs, n_groups, col, p);
            float scale = arena[scale_f_off + col * n_groups + g];
            float bias = arena[zp_f_off + col * n_groups + g];
            w_dq = scale * code + bias;
        } else if (kind == 1u) {
            unsigned long long bidx =
                (unsigned long long)col * (unsigned long long)(k / 2u) + (unsigned long long)(p / 2u);
            unsigned int byte = mlx_rd_byte(arena, (unsigned long long)w_byte_off + bidx);
            unsigned int nib = ((p & 1u) == 0u) ? (byte & 0x0fu) : (byte >> 4);
            unsigned int sb = mlx_rd_byte(
                arena,
                (unsigned long long)scale_byte_off + (unsigned long long)col * (unsigned long long)n_groups
                    + (unsigned long long)g);
            w_dq = mlx_fp4_lut(nib) * mlx_group_scale(sb, gs);
        } else {
            unsigned long long bidx =
                (unsigned long long)col * (unsigned long long)k + (unsigned long long)p;
            unsigned int wb = mlx_rd_byte(arena, (unsigned long long)w_byte_off + bidx);
            unsigned int sb = mlx_rd_byte(
                arena,
                (unsigned long long)scale_byte_off + (unsigned long long)col * (unsigned long long)n_groups
                    + (unsigned long long)g);
            w_dq = mlx_decode_e4m3(wb) * mlx_group_scale(sb, gs);
        }
        acc += arena[x_off + p] * w_dq;
    }

    __shared__ float smem[256];
    unsigned int tid = threadIdx.x;
    smem[tid] = acc;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0u) {
        arena[out_off + col] = smem[0];
    }
}

// Prefill GEMM: one threadgroup per (col, row_tile); threads split K and
// stage an X tile in shared memory (TM rows × blockDim.x K-slice).
#define MLX_TM 8u

extern "C" __global__ void dequant_matmul_mlx_gemm(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int kind,
    unsigned int bits,
    unsigned int group_size,
    unsigned long long x_byte_off,
    unsigned long long w_byte_off,
    unsigned long long scale_byte_off,
    unsigned long long zp_byte_off,
    unsigned long long out_byte_off
) {
    unsigned int col = blockIdx.x;
    unsigned int row0 = blockIdx.y * MLX_TM;
    if (col >= n) return;

    unsigned int tid = threadIdx.x;
    unsigned int tpg = blockDim.x;
    unsigned int gs = group_size;
    unsigned int n_groups = k / gs;
    unsigned long long x_off = x_byte_off / 4ull;
    unsigned long long out_off = out_byte_off / 4ull;
    unsigned long long scale_f_off = scale_byte_off / 4ull;
    unsigned long long zp_f_off = zp_byte_off / 4ull;

    // Only the final cross-thread reduction needs shared memory. X used to be
    // staged here too (`xs[t*tpg+tid]`), but each thread wrote and read back its
    // OWN slot — a no-op round-trip that cost 8 KB of occupancy-limiting shared
    // memory and a `__syncthreads()` per K-chunk. It lives in registers now.
    __shared__ float smem[MLX_TM * 256];

    float acc[MLX_TM];
    #pragma unroll
    for (unsigned int t = 0u; t < MLX_TM; ++t) acc[t] = 0.0f;

    // WHERE THIS KERNEL ACTUALLY LOSES: the weight column is re-read once per
    // ROW-TILE. `MLX_TM = 8`, so an m=64 GEMM launches 8 blocks in y for the same
    // `col`, and each independently streams that column's whole packed row. Ling's
    // `lm_head` (m=64, k=1536, n=157184) therefore moves 8 x 120 MB = 960 MB for a
    // 120 MB weight — measured 58.8 ms, i.e. ~525 GFLOP/s but only ~2.1 GB/s of
    // USEFUL weight traffic on a card that does ~400 GB/s. The fix is to cover
    // more rows per block (raise MLX_TM, or stage the decoded column in shared
    // memory and loop rows over it) so the weight is read once, NOT to change how
    // each thread walks K.
    //
    // MEASURED AND REVERTED — do not re-try the word-wise trick here. Loading
    // one 32-bit word per EIGHT nibbles (instead of `mlx_rd_byte`'s one word per
    // nibble) is what took the GROUPED kernel 10.33 ms -> 2.11 ms, but it LOSES
    // in this one: 3080 Ti, Ling `lm_head` m=64 k=1536 n=157184, 57.7 -> 69.0 ms;
    // `attn_proj` 0.73 -> 0.89 ms (checksums identical, so it was correct).
    //
    // The two kernels get their parallelism from opposite places. The grouped
    // kernel parallelizes over m*n and each thread already owned a whole K row,
    // so folding 8 elements into one thread cost nothing. Here the threads SPLIT
    // K (`p = p0 + tid`), so 8 elements per thread means 8x fewer active
    // threads — at k=1536 that is 192 of 256 doing all the work, with less
    // latency to hide the loads. Fixing this kernel means more parallelism
    // (split n across more blocks), not fewer/fatter threads.

    for (unsigned int p0 = 0u; p0 < k; p0 += tpg) {
        unsigned int p = p0 + tid;
        float xv[MLX_TM];
        #pragma unroll
        for (unsigned int t = 0u; t < MLX_TM; ++t) {
            unsigned int row = row0 + t;
            xv[t] = (row < m && p < k) ? arena[x_off + row * k + p] : 0.0f;
        }

        if (p < k) {
            unsigned int g = p / gs;
            float w_dq;
            if (kind == 0u) {
                float code = mlx_affine_w(arena, (unsigned long long)w_byte_off, bits, gs, n_groups, col, p);
                w_dq = arena[scale_f_off + col * n_groups + g] * code
                     + arena[zp_f_off + col * n_groups + g];
            } else if (kind == 1u) {
                unsigned long long bidx =
                    (unsigned long long)col * (unsigned long long)(k / 2u) + (unsigned long long)(p / 2u);
                unsigned int byte = mlx_rd_byte(arena, (unsigned long long)w_byte_off + bidx);
                unsigned int nib = ((p & 1u) == 0u) ? (byte & 0x0fu) : (byte >> 4);
                unsigned int sb = mlx_rd_byte(
                    arena,
                    (unsigned long long)scale_byte_off + (unsigned long long)col * (unsigned long long)n_groups
                        + (unsigned long long)g);
                w_dq = mlx_fp4_lut(nib) * mlx_group_scale(sb, gs);
            } else {
                unsigned long long bidx =
                    (unsigned long long)col * (unsigned long long)k + (unsigned long long)p;
                unsigned int wb = mlx_rd_byte(arena, (unsigned long long)w_byte_off + bidx);
                unsigned int sb = mlx_rd_byte(
                    arena,
                    (unsigned long long)scale_byte_off + (unsigned long long)col * (unsigned long long)n_groups
                        + (unsigned long long)g);
                w_dq = mlx_decode_e4m3(wb) * mlx_group_scale(sb, gs);
            }
            #pragma unroll
            for (unsigned int t = 0u; t < MLX_TM; ++t) {
                acc[t] += xv[t] * w_dq;
            }
        }
    }

    #pragma unroll
    for (unsigned int t = 0u; t < MLX_TM; ++t) {
        smem[t * tpg + tid] = acc[t];
    }
    __syncthreads();
    for (unsigned int s = tpg >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) {
            #pragma unroll
            for (unsigned int t = 0u; t < MLX_TM; ++t) {
                smem[t * tpg + tid] += smem[t * tpg + tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
        #pragma unroll
        for (unsigned int t = 0u; t < MLX_TM; ++t) {
            unsigned int row = row0 + t;
            if (row < m) {
                arena[out_off + row * n + col] = smem[t * tpg];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Native GROUPED (MoE) MXFP4 decode-GEMM — the on-device replacement for the
// `DequantGroupedMatMulMlx` host-delegate. Each output row r picks its expert
// e = (uint)idx[r]; W_e is a [n, k/2] packed-e2m1 slab at w_byte_off + e*n*(k/2).
// UNLIKE the single-expert kernels above, the grouped op's group scales are already
// e8m0→f32 in the arena (the loader decodes them and the backend widens bf16→f32),
// so we read them as f32 [E, n, n_groups] and do NOT re-run mlx_group_scale.
// Accumulation order (x·lut·scale) matches rlx_mlx_io::grouped_matmul_mxfp4_bt.
//
// V1 — one thread per output element (r, j). Optimal for MoE decode where each row
// routes to a distinct expert (no cross-row weight reuse to amortize). 16×16 block,
// grid (ceil(n/16), ceil(m/16)). Fully on-device: no host round-trip, no f32 weight.
extern "C" __global__ void dequant_grouped_matmul_mlx_mxfp4(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int num_experts,
    unsigned int group_size,
    unsigned long long x_byte_off,
    unsigned long long w_byte_off,
    unsigned long long scale_byte_off,
    unsigned long long idx_byte_off,
    unsigned long long out_byte_off
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    unsigned int gs = group_size;
    unsigned int n_groups = k / gs;
    unsigned long long x_off = x_byte_off / 4ull;
    unsigned long long out_off = out_byte_off / 4ull;
    unsigned long long idx_off = idx_byte_off / 4ull;
    unsigned long long scale_f_off = scale_byte_off / 4ull;

    unsigned int e = (unsigned int)arena[idx_off + (unsigned long long)row];
    if (e >= num_experts) e = num_experts - 1u;

    unsigned long long half_k = (unsigned long long)(k / 2u);
    unsigned long long ecol = (unsigned long long)e * (unsigned long long)n + (unsigned long long)col;
    unsigned long long wrow_bytes = w_byte_off + ecol * half_k;                 // packed e2m1
    unsigned long long srow_f = scale_f_off + ecol * (unsigned long long)n_groups; // f32 scales
    unsigned long long xrow_f = x_off + (unsigned long long)row * (unsigned long long)k;

    float acc = 0.0f;
    for (unsigned int p = 0u; p < k; ++p) {
        unsigned int g = p / gs;
        unsigned int byte = mlx_rd_byte(arena, wrow_bytes + (unsigned long long)(p >> 1));
        unsigned int nib = ((p & 1u) == 0u) ? (byte & 0x0fu) : (byte >> 4);
        float scale = arena[srow_f + (unsigned long long)g];
        acc += arena[xrow_f + (unsigned long long)p] * mlx_fp4_lut(nib) * scale;
    }
    arena[out_off + (unsigned long long)row * (unsigned long long)n + (unsigned long long)col] = acc;
}

// V3 — V1's thread mapping, WORD-WISE inner loop. This is the default whenever
// `k % 8 == 0` (every real MoE width is).
//
// V1 called `mlx_rd_byte` per element, and that helper loads a whole 32-bit word
// and shifts a byte out of it — so it issued ONE 32-bit load per NIBBLE, i.e. 8x
// more load instructions than the data requires, plus a per-element scale load.
// V3 loads each word once and unrolls the 8 nibbles it carries, and hoists the
// group scale out of the group's inner loop (48 scale loads per row instead of
// 1536). Same thread mapping, so occupancy and the (uncoalesced) access pattern
// are unchanged — this is purely about issuing 8x fewer memory instructions.
//
// Accumulation is now `sum(x*lut) * scale` per group rather than `sum(x*lut*scale)`;
// the scale is constant within a group so this is the same value up to f32
// rounding, and strictly fewer roundings.
extern "C" __global__ void dequant_grouped_matmul_mlx_mxfp4_v3(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int num_experts,
    unsigned int group_size,
    unsigned long long x_byte_off,
    unsigned long long w_byte_off,
    unsigned long long scale_byte_off,
    unsigned long long idx_byte_off,
    unsigned long long out_byte_off
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    unsigned int gs = group_size;
    unsigned int n_groups = k / gs;
    unsigned long long x_off = x_byte_off / 4ull;
    unsigned long long out_off = out_byte_off / 4ull;
    unsigned long long idx_off = idx_byte_off / 4ull;
    unsigned long long scale_f_off = scale_byte_off / 4ull;

    unsigned int e = (unsigned int)arena[idx_off + (unsigned long long)row];
    if (e >= num_experts) e = num_experts - 1u;

    unsigned long long half_k = (unsigned long long)(k / 2u);
    unsigned long long ecol = (unsigned long long)e * (unsigned long long)n + (unsigned long long)col;
    // Word index: `w_byte_off` is a param offset in an f32-indexed arena (4-byte
    // aligned) and `half_k` is a multiple of 4 whenever k % 8 == 0, so every
    // expert-column row starts word-aligned.
    unsigned long long wrow_w = (w_byte_off + ecol * half_k) >> 2;
    unsigned long long srow_f = scale_f_off + ecol * (unsigned long long)n_groups;
    unsigned long long xrow_f = x_off + (unsigned long long)row * (unsigned long long)k;

    unsigned int wpg = gs >> 3; // 32-bit words per group (8 nibbles each)
    float acc = 0.0f;
    for (unsigned int g = 0u; g < n_groups; ++g) {
        float gacc = 0.0f;
        unsigned long long wbase = wrow_w + (unsigned long long)g * (unsigned long long)wpg;
        unsigned long long xbase = xrow_f + (unsigned long long)g * (unsigned long long)gs;
        for (unsigned int wi = 0u; wi < wpg; ++wi) {
            unsigned int word = __float_as_uint(arena[wbase + (unsigned long long)wi]);
            unsigned long long xb = xbase + (unsigned long long)wi * 8ull;
            #pragma unroll
            for (unsigned int j = 0u; j < 8u; ++j) {
                gacc = fmaf(arena[xb + (unsigned long long)j],
                            mlx_fp4_lut((word >> (4u * j)) & 0xfu),
                            gacc);
            }
        }
        acc = fmaf(gacc, arena[srow_f + (unsigned long long)g], acc);
    }
    arena[out_off + (unsigned long long)row * (unsigned long long)n + (unsigned long long)col] = acc;
}

// V4 — SPLIT-K, one WARP per output element. The coalescing fix, and the default
// for `k % 8 == 0` on m > 1.
//
// V1/V3 give one THREAD per output (r, j), so the 32 lanes of a warp hold 32
// different columns j — and column j's packed weight row lives at stride
// `k/2` bytes. Every weight load in the warp therefore lands in a different
// 32-byte sector: fully uncoalesced, ~32x wasted DRAM traffic. That, not the
// arithmetic, is why V3 tops out around 24 GB/s on a card that does ~400.
//
// Here a whole warp owns one (row, col) and the lanes split K: lane t takes
// words t, t+32, t+64, ... of that single packed row, so the warp's 32 loads
// cover 128 CONTIGUOUS bytes — one coalesced transaction. Each word still
// carries 8 nibbles (V3's other win, kept). The per-lane partials are folded
// with a warp shuffle reduction.
//
// Block = 256 threads = 8 warps = 8 output columns; grid = (ceil(n/8), m).
extern "C" __global__ void dequant_grouped_matmul_mlx_mxfp4_splitk(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int num_experts,
    unsigned int group_size,
    unsigned long long x_byte_off,
    unsigned long long w_byte_off,
    unsigned long long scale_byte_off,
    unsigned long long idx_byte_off,
    unsigned long long out_byte_off
) {
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int col = blockIdx.x * (blockDim.x >> 5) + warp;
    unsigned int row = blockIdx.y;
    if (row >= m || col >= n) return;

    unsigned int gs = group_size;
    unsigned int n_groups = k / gs;
    unsigned long long x_off = x_byte_off / 4ull;
    unsigned long long out_off = out_byte_off / 4ull;
    unsigned long long idx_off = idx_byte_off / 4ull;
    unsigned long long scale_f_off = scale_byte_off / 4ull;

    unsigned int e = (unsigned int)arena[idx_off + (unsigned long long)row];
    if (e >= num_experts) e = num_experts - 1u;

    unsigned long long half_k = (unsigned long long)(k / 2u);
    unsigned long long ecol = (unsigned long long)e * (unsigned long long)n + (unsigned long long)col;
    unsigned long long wrow_w = (w_byte_off + ecol * half_k) >> 2;
    unsigned long long srow_f = scale_f_off + ecol * (unsigned long long)n_groups;
    unsigned long long xrow_f = x_off + (unsigned long long)row * (unsigned long long)k;

    unsigned int words = k >> 3;      // 32-bit words in the packed row (8 nibbles each)
    unsigned int wpg = gs >> 3;       // words per scale group

    float acc = 0.0f;
    for (unsigned int wi = lane; wi < words; wi += 32u) {
        unsigned int word = __float_as_uint(arena[wrow_w + (unsigned long long)wi]);
        unsigned long long xb = xrow_f + (unsigned long long)wi * 8ull;
        float part = 0.0f;
        #pragma unroll
        for (unsigned int j = 0u; j < 8u; ++j) {
            part = fmaf(arena[xb + (unsigned long long)j],
                        mlx_fp4_lut((word >> (4u * j)) & 0xfu),
                        part);
        }
        acc = fmaf(part, arena[srow_f + (unsigned long long)(wi / wpg)], acc);
    }
    #pragma unroll
    for (unsigned int off = 16u; off > 0u; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0u) {
        arena[out_off + (unsigned long long)row * (unsigned long long)n + (unsigned long long)col] = acc;
    }
}

// V2 — m>1 AMORTIZATION (prefill). Same signature as V1, but one thread per output
// COLUMN j processing ALL m rows, grouped by expert IN-THREAD: each DISTINCT expert's
// W_e[j, :] is streamed + nibble-decoded ONCE and multiplied into every row routing to
// that expert (accumulators held in registers). So when several tokens hit the same
// expert (common in prefill) the weight read + e2m1 decode are shared across them — the
// reuse the per-element kernel cannot get. No host row-sort, no scratch: the grouping
// is discovered in-thread from idx[0..m]. Requires m ≤ MLX_AMORT_MAXR (the launch uses
// V1 otherwise). acc[]/ex[] live in registers. grid = (ceil(n/BX), 1); block = (BX,1).
#define MLX_AMORT_MAXR 16u
extern "C" __global__ void dequant_grouped_matmul_mlx_mxfp4_amort(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int num_experts,
    unsigned int group_size,
    unsigned long long x_byte_off,
    unsigned long long w_byte_off,
    unsigned long long scale_byte_off,
    unsigned long long idx_byte_off,
    unsigned long long out_byte_off
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    unsigned int gs = group_size;
    unsigned int n_groups = k / gs;
    unsigned long long x_off = x_byte_off / 4ull;
    unsigned long long out_off = out_byte_off / 4ull;
    unsigned long long idx_off = idx_byte_off / 4ull;
    unsigned long long scale_f_off = scale_byte_off / 4ull;
    unsigned long long half_k = (unsigned long long)(k / 2u);

    unsigned int mm = (m < MLX_AMORT_MAXR) ? m : MLX_AMORT_MAXR;
    unsigned int ex[MLX_AMORT_MAXR];
    float acc[MLX_AMORT_MAXR];
    for (unsigned int r = 0u; r < mm; ++r) {
        unsigned int e = (unsigned int)arena[idx_off + (unsigned long long)r];
        if (e >= num_experts) e = num_experts - 1u;
        ex[r] = e;
        acc[r] = 0.0f;
    }

    // Iterate distinct experts (first occurrence); decode W_e[col,:] once per expert.
    for (unsigned int r0 = 0u; r0 < mm; ++r0) {
        unsigned int e = ex[r0];
        bool seen = false;
        for (unsigned int q = 0u; q < r0; ++q) {
            if (ex[q] == e) { seen = true; break; }
        }
        if (seen) continue;
        unsigned long long ecol = (unsigned long long)e * (unsigned long long)n + (unsigned long long)col;
        unsigned long long wrow_bytes = w_byte_off + ecol * half_k;
        unsigned long long srow_f = scale_f_off + ecol * (unsigned long long)n_groups;
        for (unsigned int p = 0u; p < k; ++p) {
            unsigned int g = p / gs;
            unsigned int byte = mlx_rd_byte(arena, wrow_bytes + (unsigned long long)(p >> 1));
            unsigned int nib = ((p & 1u) == 0u) ? (byte & 0x0fu) : (byte >> 4);
            float w_dq = mlx_fp4_lut(nib) * arena[srow_f + (unsigned long long)g]; // decode ONCE
            for (unsigned int r = r0; r < mm; ++r) {
                if (ex[r] == e) {
                    acc[r] += arena[x_off + (unsigned long long)r * (unsigned long long)k + (unsigned long long)p] * w_dq;
                }
            }
        }
    }
    for (unsigned int r = 0u; r < mm; ++r) {
        arena[out_off + (unsigned long long)r * (unsigned long long)n + (unsigned long long)col] = acc[r];
    }
}
