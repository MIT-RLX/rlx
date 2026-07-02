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

// Softmax along the last axis. Block-per-row with shared-memory tree
// reductions for max and sum_exp.
//
// Launch shape: grid=(outer,1,1), block=(256,1,1). Each block computes
// row max (tree reduce), stashes exp(x - max) into output, then sums
// (tree reduce) and normalizes.

#define SM_BLOCK 256

extern "C" __global__ void softmax(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int in_off,
    unsigned int out_off
) {
    unsigned int row = blockIdx.x;
    if (row >= outer) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int in_base  = in_off  + row * inner;
    unsigned int out_base = out_off + row * inner;

    __shared__ float s[SM_BLOCK];

    // Phase 1: row max.
    float local_max = -3.4e38f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        local_max = fmaxf(local_max, arena[in_base + i]);
    }
    s[tid] = local_max;
    __syncthreads();
    for (unsigned int s_off = bsz / 2; s_off > 0; s_off >>= 1) {
        if (tid < s_off) s[tid] = fmaxf(s[tid], s[tid + s_off]);
        __syncthreads();
    }
    float row_max = s[0];
    __syncthreads();

    // Phase 2: stash exp(x - max), accumulate sum.
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        float e = expf(arena[in_base + i] - row_max);
        arena[out_base + i] = e;
        local_sum += e;
    }
    s[tid] = local_sum;
    __syncthreads();
    for (unsigned int s_off = bsz / 2; s_off > 0; s_off >>= 1) {
        if (tid < s_off) s[tid] += s[tid + s_off];
        __syncthreads();
    }
    float inv_sum = 1.0f / s[0];
    __syncthreads();

    // Phase 3: normalize.
    for (unsigned int i = tid; i < inner; i += bsz) {
        arena[out_base + i] = arena[out_base + i] * inv_sum;
    }
}
