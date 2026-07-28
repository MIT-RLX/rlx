// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// On-GPU strided slice: out[..,j,..] = in[.., start + j*step, ..] along `axis`
// (`step` may be negative). One thread per OUTPUT element. f32 arena; non-f32
// dtypes take the host-staged `SliceHost` fallback.
//
// meta = [ out_dims[rank], in_strides[rank] ] (u32).
extern "C" __global__ void slice(
    float* arena,
    unsigned int n,
    unsigned int src_off,
    unsigned int dst_off,
    unsigned int axis,
    int start,
    int step,
    unsigned int rank,
    const unsigned int* meta
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const unsigned int* odims = meta;
    const unsigned int* istr = meta + rank;

    unsigned int rem = i;
    size_t iidx = 0;
    for (int d = (int)rank - 1; d >= 0; --d) {
        unsigned int od = odims[d];
        int oc = (int)(rem % od);
        rem /= od;
        int ic = ((unsigned int)d == axis) ? (start + oc * step) : oc;
        iidx += (size_t)ic * istr[d];
    }

    arena[dst_off + i] = arena[src_off + iidx];
}
