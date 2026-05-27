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

__device__ __forceinline__ float combine_op(unsigned int op, float a, float b) {
    switch (op) {
        case 0: case 1: return a + b;
        case 2: return fmaxf(a, b);
        case 3: return fminf(a, b);
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

    __shared__ float s[REDUCE_BLOCK];

    float ident = (op == 2) ? -3.4e38f
                : (op == 3) ?  3.4e38f
                : (op == 4) ? 1.0f
                : 0.0f;

    float acc = ident;
    for (unsigned int i = tid; i < inner; i += bsz) {
        acc = combine_op(op, acc, arena[base + i]);
    }

    s[tid] = acc;
    __syncthreads();

    for (unsigned int s_off = bsz / 2; s_off > 0; s_off >>= 1) {
        if (tid < s_off) s[tid] = combine_op(op, s[tid], s[tid + s_off]);
        __syncthreads();
    }

    if (tid == 0) {
        float final_v = s[0];
        if (op == 1) final_v /= (float)inner;
        arena[out_off + row] = final_v;
    }
}
