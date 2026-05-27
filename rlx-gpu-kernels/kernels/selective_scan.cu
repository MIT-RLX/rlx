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

// Mamba-style selective state-space scan. One thread per (batch,
// channel); each thread walks seq sequentially carrying its own state
// vector. Static cap of 256 covers every practical Mamba config.

#define MAX_STATE 256

extern "C" __global__ void selective_scan(
    float* arena,
    unsigned int batch,
    unsigned int seq,
    unsigned int hidden,
    unsigned int state_size,
    unsigned int x_off,
    unsigned int delta_off,
    unsigned int a_off,
    unsigned int b_off,
    unsigned int c_off,
    unsigned int out_off
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * hidden;
    if (id >= total || state_size > MAX_STATE) return;
    unsigned int bi = id / hidden;
    unsigned int ci = id % hidden;

    float state[MAX_STATE];
    for (unsigned int i = 0; i < state_size; ++i) state[i] = 0.0f;

    unsigned int a_base = ci * state_size;
    for (unsigned int si = 0; si < seq; ++si) {
        unsigned int x_idx = (bi * seq + si) * hidden + ci;
        float xv = arena[x_off + x_idx];
        float d  = arena[delta_off + x_idx];
        unsigned int bc_base = (bi * seq + si) * state_size;
        float acc = 0.0f;
        for (unsigned int ni = 0; ni < state_size; ++ni) {
            float da = expf(d * arena[a_off + a_base + ni]);
            state[ni] = da * state[ni] + d * arena[b_off + bc_base + ni] * xv;
            acc += arena[c_off + bc_base + ni] * state[ni];
        }
        arena[out_off + x_idx] = acc;
    }
}
