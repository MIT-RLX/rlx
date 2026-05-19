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

// Top-K indices along the last axis. One thread per outer row;
// k passes of serial argmax masking previously-picked entries.
// Output: f32-encoded indices, shape [..., k].

extern "C" __global__ void topk(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int k,
    unsigned int in_off,
    unsigned int out_off
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    unsigned int in_base  = in_off  + row * inner;
    unsigned int out_base = out_off + row * k;

    for (unsigned int pass = 0; pass < k; ++pass) {
        float best_v = -3.4e38f;
        unsigned int best_i = 0;
        for (unsigned int j = 0; j < inner; ++j) {
            bool taken = false;
            for (unsigned int p = 0; p < pass; ++p) {
                if ((unsigned int)arena[out_base + p] == j) { taken = true; break; }
            }
            if (taken) continue;
            float v = arena[in_base + j];
            if (v > best_v || (v == best_v && j < best_i)) {
                best_v = v; best_i = j;
            }
        }
        arena[out_base + pass] = (float)best_i;
    }
}
