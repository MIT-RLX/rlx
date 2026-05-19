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

// Argmax along the last axis. Output is f32-encoded index per row.

extern "C" __global__ void argmax(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int in_off,
    unsigned int out_off
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer || inner == 0) return;
    unsigned int base = in_off + row * inner;
    float best_v = arena[base];
    unsigned int best_i = 0;
    for (unsigned int i = 1; i < inner; ++i) {
        float v = arena[base + i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    arena[out_off + row] = (float)best_i;
}
