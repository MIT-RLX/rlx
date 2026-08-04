// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Hopper (sm_90) TMA-staged fp32 GEMM: C[M,N] = A[M,K] @ B[K,N], all
// row-major in the shared f32 arena.
//
// A and B tiles are bulk-copied global->shared by the Tensor Memory
// Accelerator (`cp.async.bulk.tensor.2d`) with an mbarrier byte-count
// completion handshake. The MMA itself is a plain register-blocked FMA
// loop — the point of this kernel is to exercise the TMA load pipeline
// end-to-end (host CUtensorMap -> device bulk copy -> mbarrier -> compute);
// swapping the FMA inner loop for `wgmma` is the perf follow-on, as is
// double-buffering the shared tiles. Single-buffered here: correctness-first.
//
// Compiled ONLY under `--gpu-architecture=compute_90a` (see
// `helpers::tma_arch`, gated by `RLX_CUDA_TMA`). The body is `#if`-guarded on
// __CUDA_ARCH__ so a stray non-Hopper compile traps rather than emitting
// invalid SASS. Bias/activation are NOT fused — the caller runs the shared
// `matmul_epilogue` kernel afterward, exactly like the WMMA path.
//
// STATUS: structurally complete, NOT yet validated on H100 silicon (no
// sm_90 hardware reachable from this project — see the FP8 / Vulkan-native
// precedent). The inline-PTX operands most in need of on-hardware
// verification: the grid-constant tensor-map address (`%1` below), the
// mbarrier phase-parity handshake, and the shared-address `cvta`.

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// Byte-identical to cudarc's `CUtensorMap` (16 u64 = 128 bytes, 64B aligned)
// so the host-encoded descriptor lands correctly in the grid-constant bank.
struct __align__(64) TmaDesc {
    unsigned long long opaque[16];
};

extern "C" __global__ void matmul_tma(
    const __grid_constant__ TmaDesc a_map, // A [M,K] row-major, box [BM,BK]
    const __grid_constant__ TmaDesc b_map, // B [K,N] row-major, box [BK,BN]
    float* __restrict__ arena,
    unsigned int M,
    unsigned int K,
    unsigned int N,
    unsigned int c_off
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    __shared__ __align__(128) float tile_a[BM][BK]; // fastest dim = K (contiguous)
    __shared__ __align__(128) float tile_b[BK][BN]; // fastest dim = N (contiguous)
    __shared__ __align__(8) unsigned long long mbar;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tid = ty * BLOCK_DIM_X + tx;
    const unsigned int row0 = blockIdx.y * BM + ty * TM; // first C row this thread owns
    const unsigned int col0 = blockIdx.x * BN + tx * TN; // first C col

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    // Shared-window (32-bit) addresses for the PTX operands.
    const unsigned int a_smem = (unsigned int)__cvta_generic_to_shared(&tile_a[0][0]);
    const unsigned int b_smem = (unsigned int)__cvta_generic_to_shared(&tile_b[0][0]);
    const unsigned int bar_smem = (unsigned int)__cvta_generic_to_shared(&mbar);
    // Generic address of each grid-constant descriptor.
    const unsigned long long a_desc = reinterpret_cast<unsigned long long>(&a_map);
    const unsigned long long b_desc = reinterpret_cast<unsigned long long>(&b_map);

    // Bytes the TMA delivers per K-step: both tiles.
    const unsigned int tx_bytes = (BM * BK + BK * BN) * (unsigned int)sizeof(float);
    const bool leader = (tid == 0);

    // One mbarrier, reused across K-steps with a toggling phase parity.
    if (leader) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(bar_smem));
    }
    __syncthreads();

    unsigned int phase = 0;
    for (unsigned int k0 = 0; k0 < K; k0 += BK) {
        if (leader) {
            // Arrive (count=1) and expect `tx_bytes` of async transactions.
            unsigned long long state;
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
                : "=l"(state)
                : "r"(bar_smem), "r"(tx_bytes));
            // A tile at tensor coords {k0, blockIdx.y*BM}  (dim0=K, dim1=M).
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::
                    "r"(a_smem),
                "l"(a_desc), "r"(k0), "r"(blockIdx.y * BM), "r"(bar_smem)
                : "memory");
            // B tile at tensor coords {blockIdx.x*BN, k0}  (dim0=N, dim1=K).
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::
                    "r"(b_smem),
                "l"(b_desc), "r"(blockIdx.x * BN), "r"(k0), "r"(bar_smem)
                : "memory");
        }
        // All threads spin until this phase's tiles have fully landed.
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

        // Register-blocked FMA over the staged tiles.
#pragma unroll
        for (unsigned int kk = 0; kk < BK; ++kk) {
            float a_reg[TM], b_reg[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = tile_a[ty * TM + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = tile_b[kk][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
        }
        __syncthreads();
        phase ^= 1u;
    }

    // Epilogue: bounds-guarded store (partial M/N tiles; OOB K was zero-filled
    // by TMA so it contributed nothing). Bias/activation run separately.
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        unsigned int r = row0 + i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            unsigned int c = col0 + j;
            if (c >= N) continue;
            arena[c_off + r * N + c] = acc[i][j];
        }
    }
#else
    (void)a_map;
    (void)b_map;
    (void)arena;
    (void)M;
    (void)K;
    (void)N;
    (void)c_off;
    __trap();
#endif
}
