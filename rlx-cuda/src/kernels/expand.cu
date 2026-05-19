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

// Broadcast (expand) to a target shape. `meta` packs:
//   [out_dims[0..rank], strides_for_out[0..rank]]
// where strides_for_out[i] = 0 for broadcast axes, else
// in_stride[i].

extern "C" __global__ void expand(
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
