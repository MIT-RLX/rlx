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
// Gather-axis backward: scatter-add dy into table. Zero dst before launch.

extern "C" __global__ void rlx_gather_axis_bwd(
    float* arena,
    unsigned int outer,
    unsigned int axis_dim,
    unsigned int num_idx,
    unsigned int trailing,
    unsigned int dy_off,
    unsigned int idx_off,
    unsigned int dst_off
) {
    unsigned int o = blockIdx.x;
    if (o >= outer) return;
    unsigned int t = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int total = num_idx * trailing;
    if (t >= total) return;
    unsigned int k = t / trailing;
    unsigned int j = t % trailing;
    unsigned int row = (unsigned int)arena[idx_off + k];
    if (row >= axis_dim) return;
    float v = arena[dy_off + (o * num_idx + k) * trailing + j];
    atomicAdd(&arena[dst_off + (o * axis_dim + row) * trailing + j], v);
}
