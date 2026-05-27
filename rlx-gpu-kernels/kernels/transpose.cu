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

// N-D transpose with arbitrary perm. The kernel reads dims+strides
// metadata from a side buffer to keep the uniform layout small.
//
// `meta` layout (4 * 2 * rank u32 words, packed by the caller):
//   [out_dims[0..rank], in_strides_for_out[0..rank], perm_unused..]
// Caller pre-computes `in_strides_for_out[i] = in_stride[perm[i]]`.

extern "C" __global__ void transpose(
    float* arena,
    unsigned int rank,
    unsigned int out_total,
    unsigned int in_off,
    unsigned int out_off,
    const unsigned int* meta
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_total) return;
    unsigned int idx = i;
    unsigned int src = 0;
    for (int axis = (int)rank - 1; axis >= 0; --axis) {
        unsigned int d = meta[axis];
        unsigned int s = meta[rank + axis];
        unsigned int coord = idx % d;
        idx /= d;
        src += coord * s;
    }
    arena[out_off + i] = arena[in_off + src];
}
