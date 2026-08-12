// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Grouped (MoE) matmul. Each thread computes one C[m, n] output;
// per-token expert id picks which weight matrix to multiply against.

extern "C" __global__ void grouped_matmul(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int num_experts,
    unsigned int in_off,
    unsigned int w_off,
    unsigned int idx_off,
    unsigned int out_off
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    unsigned int e = (unsigned int)arena[idx_off + row];
    if (e >= num_experts) return;
    unsigned int wb = w_off + e * k * n;
    unsigned int ib = in_off + row * k;
    float acc = 0.0f;
    for (unsigned int kk = 0; kk < k; ++kk) {
        acc += arena[ib + kk] * arena[wb + kk * n + col];
    }
    arena[out_off + row * n + col] = acc;
}


// MoE GEMV with K-splitting — decode counterpart of `grouped_matmul`.
// `grouped_matmul` maps one thread per output element, so at decode (m == 1) an
// 8x8 block leaves 8 useful threads of 64 and the grid is only n/8 blocks: far
// too little parallelism to saturate memory. Here KSPLIT threads cooperate per
// output column, each striding `k` and summing a k/KSPLIT slice, then reducing
// through shared memory. `arena[wb + kk*n + col .. +31]` stays fully coalesced.
//
// Split-K reassociates the k reduction (~1 ulp vs the sequential kernel) but is
// deterministic: partials are summed in fixed order, not atomically.
extern "C" __global__ void grouped_gemv_splitk(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int num_experts,
    unsigned int in_off,
    unsigned int w_off,
    unsigned int idx_off,
    unsigned int out_off
) {
    const unsigned int KSPLIT = 32u;
    __shared__ float partial[32][32];
    unsigned int col = blockIdx.x * 32u + threadIdx.x;
    unsigned int row = blockIdx.y;
    unsigned int ks  = threadIdx.y;
    unsigned int e = (row < m) ? (unsigned int)arena[idx_off + row] : num_experts;
    bool live = (col < n) && (row < m) && (e < num_experts);
    float acc = 0.0f;
    if (live) {
        unsigned int wb = w_off + e * k * n;
        unsigned int ib = in_off + row * k;
        for (unsigned int kk = ks; kk < k; kk += KSPLIT) {
            acc += arena[ib + kk] * arena[wb + kk * n + col];
        }
    }
    partial[threadIdx.x][ks] = acc;
    __syncthreads();
    if (ks == 0u && live) {
        float s = 0.0f;
        for (unsigned int j = 0u; j < KSPLIT; ++j) s += partial[threadIdx.x][j];
        arena[out_off + row * n + col] = s;
    }
}
