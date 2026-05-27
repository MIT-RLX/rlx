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

// Cumulative sum along the last axis. One thread per outer row,
// sequential inner. `exclusive=1` shifts so out[0] = 0.

extern "C" __global__ void cumsum(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int in_off,
    unsigned int out_off,
    unsigned int exclusive
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    unsigned int in_base  = in_off  + row * inner;
    unsigned int out_base = out_off + row * inner;
    float acc = 0.0f;
    for (unsigned int i = 0; i < inner; ++i) {
        if (exclusive != 0) {
            arena[out_base + i] = acc;
            acc += arena[in_base + i];
        } else {
            acc += arena[in_base + i];
            arena[out_base + i] = acc;
        }
    }
}
