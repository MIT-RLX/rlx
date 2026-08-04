// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Skinny-m split-K GEMV: out[m,n] = A[m,k] @ B[k,n] for SMALL m (decode). The
// 64-row tiled GEMM under-occupies the GPU when m << 64 — its grid is only
// (n/64, 1, batch), a few dozen blocks that can't saturate HBM on a 100+-CU part.
// This splits the K dimension across grid.y so many blocks run in parallel and the
// weight stream saturates bandwidth. Each thread owns one output COLUMN and reads
// B row-major [k,n] — consecutive threads read consecutive columns = coalesced.
// Each block accumulates its K-slice and atomicAdds the partial into C (which the
// caller pre-zeros with rlx_zero_f32); bias/activation run through the shared
// matmul_epilogue kernel afterward. gfx908+ have native global float atomicAdd.
//
//   Grid:  (ceil(n / GEMV_BN), k_splits, batch)      Block: GEMV_BN threads.
//   Caller MUST gate on m <= GEMV_MAX_M (register accumulator is that wide).
//
// MEASURED NON-WIN (gfx908 MI100, qwen3-0.6B): correct/bit-exact, but neutral at
// seq=8 and a ~31% regression at seq=16 vs the tiled GEMM — the forward is launch-
// bound, not matmul-bandwidth-bound, so the extra occupancy never pays for the
// acc[16] register pressure + atomicAdd contention + 3 launches/matmul. The rocm
// backend keeps this behind `RLX_ROCM_GEMV=1` as a documented experiment; the
// default fallback is the tiled kernel. Left in tree for larger-part revisiting.

#define GEMV_BN 64
#define GEMV_MAX_M 16

extern "C" __global__ void gemv_splitk(
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
    unsigned int k_splits
) {
    unsigned int bz = blockIdx.z;
    if (bz >= batch) return;
    unsigned int col = blockIdx.x * GEMV_BN + threadIdx.x;
    if (col >= n) return;

    unsigned int kper = (k + k_splits - 1u) / k_splits;
    unsigned int k0 = blockIdx.y * kper;
    unsigned int k1 = k0 + kper;
    if (k1 > k) k1 = k;
    if (k0 >= k1) return;

    unsigned int ab = a_off + bz * a_batch_stride;
    unsigned int bb = b_off + bz * b_batch_stride;
    unsigned int cb = c_off + bz * c_batch_stride;

    float acc[GEMV_MAX_M];
    #pragma unroll
    for (unsigned int i = 0; i < GEMV_MAX_M; ++i) acc[i] = 0.0f;

    for (unsigned int kk = k0; kk < k1; ++kk) {
        float b = arena[bb + kk * n + col];        // coalesced across threads
        #pragma unroll
        for (unsigned int i = 0; i < GEMV_MAX_M; ++i) {
            if (i < m) acc[i] += arena[ab + i * k + kk] * b;   // A[i,kk] broadcast
        }
    }
    #pragma unroll
    for (unsigned int i = 0; i < GEMV_MAX_M; ++i) {
        if (i < m) atomicAdd(&arena[cb + i * n + col], acc[i]);
    }
}
