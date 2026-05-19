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

// Rotary position embeddings, Llama-style split (first half / second
// half), with per-head rotation. last_dim may be head_dim (single
// head per row) or n*head_dim (n heads packed).

extern "C" __global__ void rope(
    float* arena,
    unsigned int n_total,
    unsigned int seq,
    unsigned int head_dim,
    unsigned int half,
    unsigned int in_off,
    unsigned int cos_off,
    unsigned int sin_off,
    unsigned int out_off,
    unsigned int last_dim
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    unsigned int d = i % last_dim;
    unsigned int q1 = i / last_dim;
    unsigned int pos = q1 % seq;
    unsigned int d_in_head = d % head_dim;
    unsigned int head_base = i - d_in_head;

    if (d_in_head < half) {
        float xf = arena[in_off + i];
        float xs = arena[in_off + head_base + d_in_head + half];
        float c  = arena[cos_off + pos * half + d_in_head];
        float s  = arena[sin_off + pos * half + d_in_head];
        arena[out_off + i] = xf * c - xs * s;
    } else {
        unsigned int dl = d_in_head - half;
        float xs = arena[in_off + i];
        float xf = arena[in_off + head_base + dl];
        float c  = arena[cos_off + pos * half + dl];
        float s  = arena[sin_off + pos * half + dl];
        arena[out_off + i] = xs * c + xf * s;
    }
}
