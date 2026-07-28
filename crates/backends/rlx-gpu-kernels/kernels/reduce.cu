// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Reduce along the last axis. Block-per-row + shared-memory tree
// reduction. op:
//   0=sum 1=mean 2=max 3=min 4=prod
//
// Launch shape: grid=(outer,1,1), block=(256,1,1). Each block reduces
// one row of `inner` elements via strided loads + log2(256) shared-mem
// tree reduce. Replaces the v1 one-thread-per-row pattern that left
// the GPU 95%+ idle for typical hidden=768/1024+ shapes.
//
// Tree reduction (vs. warp shuffles) keeps the kernel portable to
// HIP-CPU's 64-lane wavefront in the dev validation path.

#define REDUCE_BLOCK 256

// Accumulate sum/mean/prod in f64. A block reduces up to `inner` elements; an
// f32 running sum drifts from the CPU/PyTorch result (which accumulate in higher
// precision), and over many training steps that ~1e-6/step gap compounds and
// breaks cross-backend / cross-framework training reproduction (the BN γ/β
// gradient reduction was the last non-bit-reproducible op on CUDA). Reducing in
// double closes it to the float floor. max/min are precision-invariant; the tiny
// f64 register/smem cost is negligible for a memory-bound reduction.
__device__ __forceinline__ double combine_op_d(unsigned int op, double a, double b) {
    switch (op) {
        case 0: case 1: return a + b;
        case 2: return fmax(a, b);
        case 3: return fmin(a, b);
        case 4: return a * b;
        default: return a;
    }
}

extern "C" __global__ void reduce(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int in_off,
    unsigned int out_off,
    unsigned int op
) {
    unsigned int row = blockIdx.x;
    if (row >= outer) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int base = in_off + row * inner;

    __shared__ double s[REDUCE_BLOCK];

    double ident = (op == 2) ? -3.4e38
                 : (op == 3) ?  3.4e38
                 : (op == 4) ? 1.0
                 : 0.0;

    double acc = ident;
    for (unsigned int i = tid; i < inner; i += bsz) {
        acc = combine_op_d(op, acc, (double)arena[base + i]);
    }

    s[tid] = acc;
    __syncthreads();

    for (unsigned int s_off = bsz / 2; s_off > 0; s_off >>= 1) {
        if (tid < s_off) s[tid] = combine_op_d(op, s[tid], s[tid + s_off]);
        __syncthreads();
    }

    if (tid == 0) {
        double final_v = s[0];
        if (op == 1) final_v /= (double)inner;
        arena[out_off + row] = (float)final_v;
    }
}
