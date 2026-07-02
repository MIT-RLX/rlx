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

// Embedding-style gather along the leading axis.
//   input  [vocab, dim], indices [n_idx]  →  output [n_idx, dim]

extern "C" __global__ void gather(
    float* arena,
    unsigned int n_out,
    unsigned int n_idx,
    unsigned int dim,
    unsigned int vocab,
    unsigned int in_off,
    unsigned int idx_off,
    unsigned int out_off
) {
    unsigned int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    unsigned int d = o % dim;
    unsigned int i = o / dim;
    float idx_f = arena[idx_off + i];
    unsigned int idx_u = (unsigned int)fmaxf(idx_f, 0.0f);
    if (idx_u >= vocab) idx_u = vocab - 1;
    arena[out_off + o] = arena[in_off + idx_u * dim + d];
}
