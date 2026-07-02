// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// GPL-3.0-only. See LICENSE.
//
// C[m,n] = A[m,k] @ B^T where B is stored row-major as [n,k] (GGUF dequant layout).
// Mirrors rlx-wgpu `matmul.wgsl` `matmul_bt` and rlx-cpu `gguf_matmul_bt`.

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define RM 4
#define RN 4

extern "C" __global__ void matmul_bt(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int a_off,
    unsigned int b_off,
    unsigned int c_off
) {
    __shared__ float tile_a[TILE_M][TILE_K];
    __shared__ float tile_b[TILE_K][TILE_N];

    unsigned int lr = threadIdx.y;
    unsigned int lc = threadIdx.x;
    unsigned int wid_y = blockIdx.y;
    unsigned int wid_x = blockIdx.x;

    unsigned int row_base = wid_y * TILE_M + lr * RM;
    unsigned int col_base = wid_x * TILE_N + lc * RN;

    float acc[RM][RN];
#pragma unroll
    for (int i = 0; i < RM; ++i) {
#pragma unroll
        for (int j = 0; j < RN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    unsigned int n_tiles = (k + TILE_K - 1u) / TILE_K;

    for (unsigned int t = 0; t < n_tiles; ++t) {
#pragma unroll
        for (unsigned int i = 0; i < RM; ++i) {
            unsigned int m_local = lr * RM + i;
            unsigned int global_row = wid_y * TILE_M + m_local;
#pragma unroll
            for (unsigned int j = 0; j < 2u; ++j) {
                unsigned int k_local = lc * 2u + j;
                unsigned int global_k = t * TILE_K + k_local;
                if (global_row < m && global_k < k) {
                    tile_a[m_local][k_local] = arena[a_off + global_row * k + global_k];
                } else {
                    tile_a[m_local][k_local] = 0.0f;
                }
            }
        }
#pragma unroll
        for (unsigned int i = 0; i < 2u; ++i) {
            unsigned int k_local = lr * 2u + i;
            unsigned int global_k = t * TILE_K + k_local;
#pragma unroll
            for (unsigned int j = 0; j < RN; ++j) {
                unsigned int n_local = lc * RN + j;
                unsigned int global_col = wid_x * TILE_N + n_local;
                if (global_k < k && global_col < n) {
                    tile_b[k_local][n_local] = arena[b_off + global_col * k + global_k];
                } else {
                    tile_b[k_local][n_local] = 0.0f;
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (unsigned int kk = 0; kk < TILE_K; ++kk) {
            float a_reg[RM];
            float b_reg[RN];
#pragma unroll
            for (unsigned int i = 0; i < RM; ++i) {
                a_reg[i] = tile_a[lr * RM + i][kk];
            }
#pragma unroll
            for (unsigned int j = 0; j < RN; ++j) {
                b_reg[j] = tile_b[kk][lc * RN + j];
            }
#pragma unroll
            for (unsigned int i = 0; i < RM; ++i) {
#pragma unroll
                for (unsigned int j = 0; j < RN; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (unsigned int i = 0; i < RM; ++i) {
        unsigned int global_row = row_base + i;
        if (global_row >= m) {
            continue;
        }
#pragma unroll
        for (unsigned int j = 0; j < RN; ++j) {
            unsigned int global_col = col_base + j;
            if (global_col >= n) {
                continue;
            }
            arena[c_off + global_row * n + global_col] = acc[i][j];
        }
    }
}
