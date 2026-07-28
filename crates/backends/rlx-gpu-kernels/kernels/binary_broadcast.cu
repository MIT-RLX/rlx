// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Stride-aware element-wise binary op — the broadcasting form of `binary`.
// Output is contiguous [n]; each operand is read through per-axis strides so a
// broadcast operand (stride 0 on a size-1 axis) is never materialized. Lets the
// CUDA compiler drop the `Op::Expand` that `LegalizeBroadcast` would otherwise
// insert (e.g. squeeze-excite `[1,C,1,1] * [1,C,H,W]`).
//
// meta = [ out_dims[rank], a_strides[rank], b_strides[rank] ] (u32).
// Selector `op`: 0=add 1=sub 2=mul 3=div 4=max 5=min 6=pow.
extern "C" __global__ void binary_broadcast(
    float* arena,
    unsigned int n,
    unsigned int a_off,
    unsigned int b_off,
    unsigned int c_off,
    unsigned int op,
    unsigned int rank,
    const unsigned int* meta
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const unsigned int* odims = meta;
    const unsigned int* astr = meta + rank;
    const unsigned int* bstr = meta + 2u * rank;

    unsigned int rem = i;
    size_t aidx = 0;
    size_t bidx = 0;
    for (int d = (int)rank - 1; d >= 0; --d) {
        unsigned int od = odims[d];
        unsigned int coord = rem % od;
        rem /= od;
        aidx += (size_t)coord * astr[d];
        bidx += (size_t)coord * bstr[d];
    }

    float a = arena[a_off + aidx];
    float b = arena[b_off + bidx];
    float c;
    switch (op) {
        case 0: c = a + b; break;
        case 1: c = a - b; break;
        case 2: c = a * b; break;
        case 3: c = a / b; break;
        case 4: c = fmaxf(a, b); break;
        case 5: c = fminf(a, b); break;
        case 6: c = powf(a, b); break;
        default: c = 0.0f;
    }
    arena[c_off + i] = c;
}
