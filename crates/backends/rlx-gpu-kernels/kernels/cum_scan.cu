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

// Cumulative product / maximum along the last axis. One thread per outer row,
// sequential inner. `is_max=1` runs a running max (identity -inf), else a
// running product (identity 1). `exclusive=1` shifts so out[0] = identity.
// Mirrors cumsum.cu — the native O(L) scan behind Op::{CumProd, CumMax}.

extern "C" __global__ void cum_scan(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int in_off,
    unsigned int out_off,
    unsigned int exclusive,
    unsigned int is_max
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    unsigned int in_base  = in_off  + row * inner;
    unsigned int out_base = out_off + row * inner;
    // NVRTC doesn't define the <math.h> INFINITY macro — build -inf from bits.
    float acc = (is_max != 0) ? __uint_as_float(0xff800000u) : 1.0f;
    for (unsigned int i = 0; i < inner; ++i) {
        float v = arena[in_base + i];
        if (exclusive != 0) {
            arena[out_base + i] = acc;
            acc = (is_max != 0) ? fmaxf(acc, v) : (acc * v);
        } else {
            acc = (is_max != 0) ? fmaxf(acc, v) : (acc * v);
            arena[out_base + i] = acc;
        }
    }
}
