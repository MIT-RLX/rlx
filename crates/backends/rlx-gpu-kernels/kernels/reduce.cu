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

// `reduce` for short rows (`inner` <= 32, the hosts check): one thread a row,
// the same f64 accumulation and order. A block a row of two elements left
// 254 of its 256 threads idle and launched a block per output: SynthSeg's
// 2x2x2 max pool (rows of 2 after the axis lowering) at 160^3 took 50 ms a
// reduce on an RTX 3080 Ti, 600 of the network's 760 profiled ms.
// Launch shape: grid=(ceil(outer/256),1,1), block=(256,1,1).
extern "C" __global__ void reduce_rows(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int in_off,
    unsigned int out_off,
    unsigned int op
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    unsigned int base = in_off + row * inner;
    double acc = (op == 2) ? -3.4e38
               : (op == 3) ?  3.4e38
               : (op == 4) ? 1.0
               : 0.0;
    for (unsigned int i = 0; i < inner; ++i) {
        acc = combine_op_d(op, acc, (double)arena[base + i]);
    }
    if (op == 1) acc /= (double)inner;
    arena[out_off + row] = (float)acc;
}

// ArgMax / ArgMin over the middle axis of [outer, reduced, inner], writing
// f32-encoded indices to [outer, inner] — the semantics of the host
// `execute_argreduce_f32` it replaces: the first index of the strict best,
// NaN never better (a NaN in position 0 keeps index 0). 64-bit element
// offsets (arenas past 4 GiB). Host-staged, SynthSeg robust's two label
// argmaxes over 160^3 voxels took ~50 ms each on an RTX 3080 Ti.

// A thread an output; neighbouring threads read neighbouring `inner`
// elements, so loads coalesce whenever inner > 1.
// Launch shape: grid=(ceil(outer*inner/256),1,1), block=(256,1,1).
extern "C" __global__ void argreduce(
    float* arena, unsigned int outer, unsigned int reduced, unsigned int inner,
    unsigned long long in_off, unsigned long long out_off, unsigned int is_max) {
    unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)outer * inner;
    if (t >= total) return;
    unsigned long long o = t / inner, i = t % inner;
    unsigned long long base = in_off + o * (unsigned long long)reduced * inner + i;
    float best = arena[base];
    unsigned int best_idx = 0;
    for (unsigned int r = 1; r < reduced; ++r) {
        float v = arena[base + (unsigned long long)r * inner];
        if (is_max ? v > best : v < best) { best = v; best_idx = r; }
    }
    arena[out_off + t] = (float)best_idx;
}

// A block a row, for inner == 1 and long rows (vocabulary argmax), where a
// thread a row would serialise the whole reduction.
// Launch shape: grid=(outer,1,1), block=(256,1,1).
extern "C" __global__ void argreduce_row(
    float* arena, unsigned int outer, unsigned int reduced, unsigned int inner,
    unsigned long long in_off, unsigned long long out_off, unsigned int is_max) {
    (void)inner;
    unsigned int row = blockIdx.x;
    if (row >= outer) return;
    unsigned long long base = in_off + (unsigned long long)row * reduced;
    __shared__ float sv[256];
    __shared__ unsigned int si[256];
    // Each thread's first-of-best over its strided indices; ties and the
    // cross-thread merge both keep the smaller index, so the result is the
    // first strict best as in a left-to-right scan. NaN is never better
    // (comparisons with it are false).
    float best = is_max ? -3.402823466e38f : 3.402823466e38f;
    unsigned int idx = 0xffffffffu;
    for (unsigned int r = threadIdx.x; r < reduced; r += blockDim.x) {
        float v = arena[base + r];
        bool better = is_max ? v > best : v < best;
        if (better || (v == best && r < idx)) { best = v; idx = r; }
    }
    sv[threadIdx.x] = best;
    si[threadIdx.x] = idx;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float v = sv[threadIdx.x + s];
            unsigned int j = si[threadIdx.x + s];
            float b = sv[threadIdx.x];
            unsigned int k = si[threadIdx.x];
            bool better = is_max ? v > b : v < b;
            if (j != 0xffffffffu && (k == 0xffffffffu || better || (v == b && j < k))) {
                sv[threadIdx.x] = v;
                si[threadIdx.x] = j;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // A NaN first element is never beaten in the scan; neither is a row
        // of NaNs (no candidate at all).
        float x0 = arena[base];
        unsigned int r = (x0 != x0 || si[0] == 0xffffffffu) ? 0u : si[0];
        arena[out_off + row] = (float)r;
    }
}
