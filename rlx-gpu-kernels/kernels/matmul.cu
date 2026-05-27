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

// Tiled fp32 matmul with register blocking + optional float4 vector
// loads. Block tile 64×64 of C, inner-K tile 16. 16×16=256 threads per
// block; each thread computes a 4×4 micro-tile of C accumulated in
// registers. Same call shapes as v1 (2D × 2D, [B,M,K]×[K,N],
// [B,M,K]×[B,K,N]) and same epilogue (optional bias + activation).
//
// `vec_loads` (passed via the `act_id` upper bit at runtime is not
// portable; we instead branch on alignment: when k % 4 == 0 and
// n % 4 == 0 the tile loaders use float4 reads, halving the issue
// count for shared-memory fills). Falls back to scalar loads otherwise.
//
// Activation IDs match the unary kernel's table:
//   0=relu 1=sigmoid 2=tanh 5=sqrt 7=neg 8=abs 9=gelu 10=silu 11=gelu_approx

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define THREADS (BLOCK_DIM_X * BLOCK_DIM_Y)

__device__ __forceinline__ float apply_act(float v, unsigned int act_id) {
    if (act_id == 0xFFFFu) return v;
    switch (act_id) {
        case 0:  return fmaxf(v, 0.0f);
        case 1:  return 1.0f / (1.0f + expf(-fminf(fmaxf(v, -88.0f), 88.0f)));
        case 2:  return tanhf(fminf(fmaxf(v, -15.0f), 15.0f));
        case 5:  return sqrtf(v);
        case 7:  return -v;
        case 8:  return fabsf(v);
        case 9:
        case 11: {
            const float c = 0.7978845608028654f;
            float x3 = v * v * v;
            float inner = c * (v + 0.044715f * x3);
            inner = fminf(fmaxf(inner, -15.0f), 15.0f);
            return 0.5f * v * (1.0f + tanhf(inner));
        }
        case 10: {
            float nx = fminf(fmaxf(-v, -88.0f), 88.0f);
            return v / (1.0f + expf(nx));
        }
        default: return v;
    }
}

extern "C" __global__ void matmul(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int a_off,
    unsigned int b_off,
    unsigned int c_off,
    unsigned int batch,
    unsigned int a_batch_stride,
    unsigned int b_batch_stride,
    unsigned int c_batch_stride,
    unsigned int has_bias,
    unsigned int bias_off,
    unsigned int act_id
) {
    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];

    unsigned int bz = blockIdx.z;
    if (bz >= batch) return;

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int tid = ty * BLOCK_DIM_X + tx;

    unsigned int row0 = blockIdx.y * BM + ty * TM;
    unsigned int col0 = blockIdx.x * BN + tx * TN;

    unsigned int a_base = a_off + bz * a_batch_stride;
    unsigned int b_base = b_off + bz * b_batch_stride;
    unsigned int c_base = c_off + bz * c_batch_stride;

    // Block-level alignment check for float4 vector loads. These are
    // shape-only conditions; the arena pointer itself is f32-aligned so
    // float4-aligned slots only need the row stride to be a multiple of
    // 4. blockIdx.y * BM and blockIdx.x * BN are both multiples of 64,
    // so the leading element of each tile is also float4-aligned.
    bool vec_a = ((k & 3u) == 0u);
    bool vec_b = ((n & 3u) == 0u);
    bool full_block_a = (blockIdx.y * BM + BM <= m);
    bool full_block_b = (blockIdx.x * BN + BN <= n);

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    }

    unsigned int n_tiles = (k + BK - 1) / BK;
    const unsigned int A_PER_THREAD = (BM * BK) / THREADS;  // 4
    const unsigned int B_PER_THREAD = (BK * BN) / THREADS;  // 4

    for (unsigned int t = 0; t < n_tiles; ++t) {
        bool full_k_tile = (t * BK + BK <= k);

        // ── Load A tile [BM][BK] ────────────────────────────────────
        if (vec_a && full_block_a && full_k_tile) {
            // Each thread owns one float4 of the tile (BM*BK/4 = 256
            // float4 chunks for 256 threads).
            unsigned int idx4 = tid;  // 0..255
            unsigned int r = idx4 / (BK / 4);              // 0..63
            unsigned int c4 = idx4 % (BK / 4);              // 0..3
            unsigned int gr = blockIdx.y * BM + r;
            unsigned int gc = t * BK + c4 * 4;
            float4 v = *reinterpret_cast<const float4*>(&arena[a_base + gr * k + gc]);
            tile_a[r][c4 * 4 + 0] = v.x;
            tile_a[r][c4 * 4 + 1] = v.y;
            tile_a[r][c4 * 4 + 2] = v.z;
            tile_a[r][c4 * 4 + 3] = v.w;
        } else {
            #pragma unroll
            for (unsigned int li = 0; li < A_PER_THREAD; ++li) {
                unsigned int idx = tid + li * THREADS;
                unsigned int r = idx / BK;
                unsigned int c = idx % BK;
                unsigned int gr = blockIdx.y * BM + r;
                unsigned int gc = t * BK + c;
                tile_a[r][c] = (gr < m && gc < k) ? arena[a_base + gr * k + gc] : 0.0f;
            }
        }

        // ── Load B tile [BK][BN] ────────────────────────────────────
        if (vec_b && full_block_b && full_k_tile) {
            unsigned int idx4 = tid;  // 0..255
            unsigned int r = idx4 / (BN / 4);   // 0..15
            unsigned int c4 = idx4 % (BN / 4);   // 0..15
            unsigned int gr = t * BK + r;
            unsigned int gc = blockIdx.x * BN + c4 * 4;
            float4 v = *reinterpret_cast<const float4*>(&arena[b_base + gr * n + gc]);
            tile_b[r][c4 * 4 + 0] = v.x;
            tile_b[r][c4 * 4 + 1] = v.y;
            tile_b[r][c4 * 4 + 2] = v.z;
            tile_b[r][c4 * 4 + 3] = v.w;
        } else {
            #pragma unroll
            for (unsigned int li = 0; li < B_PER_THREAD; ++li) {
                unsigned int idx = tid + li * THREADS;
                unsigned int r = idx / BN;
                unsigned int c = idx % BN;
                unsigned int gr = t * BK + r;
                unsigned int gc = blockIdx.x * BN + c;
                tile_b[r][c] = (gr < k && gc < n) ? arena[b_base + gr * n + gc] : 0.0f;
            }
        }

        __syncthreads();

        // Compute TM×TN micro-tile via outer product over BK.
        #pragma unroll
        for (unsigned int kk = 0; kk < BK; ++kk) {
            float a_reg[TM];
            float b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = tile_a[ty * TM + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = tile_b[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

    // Epilogue: optional bias + activation, masked write.
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            unsigned int row = row0 + i;
            unsigned int col = col0 + j;
            if (row < m && col < n) {
                float v = acc[i][j];
                if (has_bias) v += arena[bias_off + col];
                v = apply_act(v, act_id);
                arena[c_base + row * n + col] = v;
            }
        }
    }
}
