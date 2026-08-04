// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// GRU (gate order r, z, n; linear_before_reset=1; separate b_ih/b_hh). One block
// per batch item; thread `k` owns hidden unit `k`; the recurrence loops inside.
// Dispatched once per (layer, direction) — the caller (`run_gru`) supplies:
//   • x_off/x_is_scratch, dst_off/dst_is_scratch — the layer's input/output live
//     in the arena (layer 0 in, final out) or a ping-pong scratch (intermediate
//     layers); WEIGHTS (wih/whh/bih/bhh) + h0 are always arena tensors.
//   • in_l = input width (input_size for layer 0, else dirs·hidden).
//   • out_width = dirs·hidden, dir_off = dir·hidden — this direction owns the
//     `[dir_off, dir_off+hidden)` feature slice of a `[batch, seq, out_width]`
//     output; `reverse` walks the sequence backwards (dir==1).
//   • h0_off = 0 → h0 = 0; else seed from `arena[h0_off + bi·hidden + k]`.
// Single-layer / unidirectional / no-carry reduces to the original kernel
// (x_is_scratch=dst_is_scratch=0, out_width=hidden, dir_off=0, reverse=0).
// Bit-for-bit mirror of `execute_gru_f32`. hidden ≤ GRU_MAX_H.

#define GRU_MAX_H 1024u

extern "C" __global__ void gru(
    float* arena,
    float* scratch,
    unsigned int x_off,
    unsigned int x_is_scratch,
    unsigned int wih_off,
    unsigned int whh_off,
    unsigned int bih_off,
    unsigned int bhh_off,
    unsigned int dst_off,
    unsigned int dst_is_scratch,
    unsigned int batch,
    unsigned int seq,
    unsigned int in_l,
    unsigned int hidden,
    unsigned int h0_off,
    unsigned int out_width,
    unsigned int dir_off,
    unsigned int reverse
) {
    __shared__ float h_sh[GRU_MAX_H];

    unsigned int bi = blockIdx.x;
    unsigned int k = threadIdx.x;
    if (hidden > GRU_MAX_H || bi >= batch) return;

    const float* xin = (x_is_scratch != 0u) ? scratch : arena;
    float* out = (dst_is_scratch != 0u) ? scratch : arena;

    if (k < hidden) {
        h_sh[k] = (h0_off != 0u) ? arena[h0_off + bi * hidden + k] : 0.0f;
    }
    __syncthreads();

    for (unsigned int step = 0; step < seq; ++step) {
        unsigned int t = (reverse != 0u) ? (seq - 1u - step) : step;
        float h_k = 0.0f;
        if (k < hidden) {
            unsigned int x_base = x_off + (bi * seq + t) * in_l;
            float xi[3], hi[3];
            for (unsigned int g = 0u; g < 3u; ++g) {
                unsigned int r = g * hidden + k;
                float ax = arena[bih_off + r];
                unsigned int wih_row = wih_off + r * in_l;
                for (unsigned int j = 0u; j < in_l; ++j) {
                    ax += arena[wih_row + j] * xin[x_base + j];
                }
                float ah = arena[bhh_off + r];
                unsigned int whh_row = whh_off + r * hidden;
                for (unsigned int j = 0u; j < hidden; ++j) {
                    ah += arena[whh_row + j] * h_sh[j];
                }
                xi[g] = ax;
                hi[g] = ah;
            }
            float rg = 1.0f / (1.0f + expf(-(xi[0] + hi[0])));
            float zg = 1.0f / (1.0f + expf(-(xi[1] + hi[1])));
            float ng = tanhf(xi[2] + rg * hi[2]);
            h_k = (1.0f - zg) * ng + zg * h_sh[k];
        }
        // All threads finished reading the old h_sh.
        __syncthreads();
        if (k < hidden) {
            h_sh[k] = h_k;
            out[dst_off + (bi * seq + t) * out_width + dir_off + k] = h_k;
        }
        __syncthreads();
    }
}
