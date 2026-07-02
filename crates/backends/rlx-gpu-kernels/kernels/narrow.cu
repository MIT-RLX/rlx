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

// Slice along an axis: take `axis_out_size` contiguous elements
// starting at `start` in the chosen axis.

extern "C" __global__ void narrow(
    float* arena,
    unsigned int total,
    unsigned int outer,
    unsigned int inner,
    unsigned int axis_in_size,
    unsigned int axis_out_size,
    unsigned int start,
    unsigned int in_off,
    unsigned int out_off
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    // total = outer * axis_out_size * inner
    unsigned int in_idx_innermost = i % inner;
    unsigned int q1 = i / inner;
    unsigned int axis_idx = q1 % axis_out_size;
    unsigned int outer_idx = q1 / axis_out_size;
    unsigned int src_axis = start + axis_idx;
    unsigned int src = (outer_idx * axis_in_size + src_axis) * inner + in_idx_innermost;
    arena[out_off + i] = arena[in_off + src];
}
