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

// Rotary position embeddings, with per-head rotation. last_dim may be
// head_dim (single head per row) or n*head_dim (n heads packed).
//
// `interleaved` selects the pairing convention (rlx_ir::op::RopeStyle):
//   0 = NeoX  (HF rotate-half): dim i pairs with i + head_dim/2.
//   1 = GptJ  (llama.cpp NORM): adjacent pairs (2f, 2f+1), cos/sin by freq f.
// cos/sin tables are identical for both; only the pairing differs. GGUF Llama
// weights are permuted by the HF→GGUF converter for the GptJ flavor.

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
    unsigned int last_dim,
    unsigned int interleaved
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    unsigned int d = i % last_dim;
    unsigned int q1 = i / last_dim;
    unsigned int pos = q1 % seq;
    unsigned int d_in_head = d % head_dim;
    unsigned int head_base = i - d_in_head;

    if (interleaved != 0u) {
        // GptJ / llama.cpp NORM: rotated pairs are adjacent (2f, 2f+1); cos/sin
        // indexed by freq f. Each thread owns one element and reads its partner.
        unsigned int f = d_in_head >> 1;
        float c = arena[cos_off + pos * half + f];
        float s = arena[sin_off + pos * half + f];
        if ((d_in_head & 1u) == 0u) {
            // even element = x1: out[2f] = x1*c - x2*s
            float x1 = arena[in_off + i];
            float x2 = arena[in_off + i + 1];
            arena[out_off + i] = x1 * c - x2 * s;
        } else {
            // odd element = x2: out[2f+1] = x2*c + x1*s
            float x2 = arena[in_off + i];
            float x1 = arena[in_off + i - 1];
            arena[out_off + i] = x2 * c + x1 * s;
        }
        return;
    }

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
