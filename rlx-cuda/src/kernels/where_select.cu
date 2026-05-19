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

// 3-input select: y[i] = cond[i] ? x[i] : y[i]
// `cond` is the f32-encoded Bool (≠0 → true).
extern "C" __global__ void where_select(
    float* arena,
    unsigned int n,
    unsigned int cond_off,
    unsigned int x_off,
    unsigned int y_off,
    unsigned int out_off
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c = arena[cond_off + i];
    arena[out_off + i] = (c != 0.0f) ? arena[x_off + i] : arena[y_off + i];
}
