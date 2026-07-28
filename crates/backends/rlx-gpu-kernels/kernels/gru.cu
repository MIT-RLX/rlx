// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Single-layer unidirectional GRU (gate order r, z, n; linear_before_reset=1;
// separate b_ih/b_hh; h0 = 0). One block per batch item; thread `k` owns
// hidden unit `k`. Matches `execute_gru_f32` / Metal `gru` / wgpu `gru.wgsl`.
// hidden ≤ GRU_MAX_H (multi-layer / bidir / carry take the host path).

#define GRU_MAX_H 1024u

extern "C" __global__ void gru(
    float* arena,
    unsigned int x_off,
    unsigned int wih_off,
    unsigned int whh_off,
    unsigned int bih_off,
    unsigned int bhh_off,
    unsigned int dst_off,
    unsigned int batch,
    unsigned int seq,
    unsigned int input_size,
    unsigned int hidden
) {
    __shared__ float h_sh[GRU_MAX_H];

    unsigned int bi = blockIdx.x;
    unsigned int k = threadIdx.x;
    if (hidden > GRU_MAX_H || bi >= batch) return;

    if (k < hidden) {
        h_sh[k] = 0.0f;
    }
    __syncthreads();

    for (unsigned int t = 0; t < seq; ++t) {
        float h_k = 0.0f;
        if (k < hidden) {
            unsigned int x_base = x_off + (bi * seq + t) * input_size;
            float xi[3], hi[3];
            for (unsigned int g = 0u; g < 3u; ++g) {
                unsigned int r = g * hidden + k;
                float ax = arena[bih_off + r];
                unsigned int wih_row = wih_off + r * input_size;
                for (unsigned int j = 0u; j < input_size; ++j) {
                    ax += arena[wih_row + j] * arena[x_base + j];
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
            arena[dst_off + (bi * seq + t) * hidden + k] = h_k;
        }
        __syncthreads();
    }
}
