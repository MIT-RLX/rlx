// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Hopper (sm_90) TMA-staged fp32 NT GEMM: C[M,N] = A[M,K] @ W[N,K]^T, all
// row-major in the shared f32 arena. This is the transposed-B twin of
// `matmul_tma.cu`, targeting the GGUF prefill path (`run_matmul_bt`) once the
// packed weight has been dequantized into an f32 [N,K] scratch slab.
//
// Both operands are row-major with K as the contiguous (fastest) dimension, so
// both tensor-maps use dim0=K. A stages to [BM][BK]; W stages to [BN][BK]; the
// FMA reads W transposed (W[n][k]) against A (A[m][k]) to form C[m][n].
//
// Same status/caveats as matmul_tma.cu: single-buffered, FMA-not-wgmma,
// `compute_90a`-only, `#if`-guarded, bias/act via the epilogue kernel. NOT
// hardware-validated (no sm_90 reachable) — the packed-weight dequant is still
// a separate global roundtrip; fusing dequant into the TMA-staged tiles is the
// real perf follow-on. This kernel only stages the GEMM half.

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// Byte-identical to cudarc's `CUtensorMap` (16 u64 = 128 bytes, 64B aligned).
struct __align__(64) TmaDesc {
    unsigned long long opaque[16];
};

extern "C" __global__ void matmul_bt_tma(
    const __grid_constant__ TmaDesc a_map, // A [M,K] row-major, box [BM,BK]
    const __grid_constant__ TmaDesc w_map, // W [N,K] row-major, box [BN,BK]
    float* __restrict__ arena,
    unsigned int M,
    unsigned int K,
    unsigned int N,
    unsigned long long c_off // 64-bit: the GGUF f32 arena can exceed 4 GB
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    __shared__ __align__(128) float tile_a[BM][BK]; // A[m][k], k contiguous
    __shared__ __align__(128) float tile_w[BN][BK]; // W[n][k], k contiguous
    __shared__ __align__(8) unsigned long long mbar;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tid = ty * BLOCK_DIM_X + tx;
    const unsigned int row0 = blockIdx.y * BM + ty * TM; // first C row (m)
    const unsigned int col0 = blockIdx.x * BN + tx * TN; // first C col (n)

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    const unsigned int a_smem = (unsigned int)__cvta_generic_to_shared(&tile_a[0][0]);
    const unsigned int w_smem = (unsigned int)__cvta_generic_to_shared(&tile_w[0][0]);
    const unsigned int bar_smem = (unsigned int)__cvta_generic_to_shared(&mbar);
    const unsigned long long a_desc = reinterpret_cast<unsigned long long>(&a_map);
    const unsigned long long w_desc = reinterpret_cast<unsigned long long>(&w_map);

    const unsigned int tx_bytes = (BM * BK + BN * BK) * (unsigned int)sizeof(float);
    const bool leader = (tid == 0);

    if (leader) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(bar_smem));
    }
    __syncthreads();

    unsigned int phase = 0;
    for (unsigned int k0 = 0; k0 < K; k0 += BK) {
        if (leader) {
            unsigned long long state;
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
                : "=l"(state)
                : "r"(bar_smem), "r"(tx_bytes));
            // A tile at coords {k0, blockIdx.y*BM}  (dim0=K, dim1=M).
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::
                    "r"(a_smem),
                "l"(a_desc), "r"(k0), "r"(blockIdx.y * BM), "r"(bar_smem)
                : "memory");
            // W tile at coords {k0, blockIdx.x*BN}  (dim0=K, dim1=N).
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::
                    "r"(w_smem),
                "l"(w_desc), "r"(k0), "r"(blockIdx.x * BN), "r"(bar_smem)
                : "memory");
        }
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "L_wait_%=:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
            "@!p bra L_wait_%=;\n"
            "}\n" ::"r"(bar_smem),
            "r"(phase)
            : "memory");
        __syncthreads();

        // C[m,n] = Σ_k A[m,k] · W[n,k] — W read transposed from its [n][k] tile.
#pragma unroll
        for (unsigned int kk = 0; kk < BK; ++kk) {
            float a_reg[TM], w_reg[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = tile_a[ty * TM + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) w_reg[j] = tile_w[tx * TN + j][kk];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = fmaf(a_reg[i], w_reg[j], acc[i][j]);
        }
        __syncthreads();
        phase ^= 1u;
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        unsigned int r = row0 + i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            unsigned int c = col0 + j;
            if (c >= N) continue;
            arena[c_off + (unsigned long long)r * N + c] = acc[i][j];
        }
    }
#else
    (void)a_map;
    (void)w_map;
    (void)arena;
    (void)M;
    (void)K;
    (void)N;
    (void)c_off;
    __trap();
#endif
}
