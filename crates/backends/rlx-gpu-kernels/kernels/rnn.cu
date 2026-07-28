// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Single-layer unidirectional Elman RNN (`relu_flag` ? relu : tanh; h0 = 0).
// One block per batch item; thread `k` owns hidden unit `k`. Matches
// `execute_rnn_f32` / Metal `rnn` / wgpu `rnn.wgsl`. hidden ≤ RNN_MAX_H
// (multi-layer / bidir / carry take the host path).

#define RNN_MAX_H 1024u

extern "C" __global__ void rnn(
    float* arena,
    unsigned int x_off,
    unsigned int wih_off,
    unsigned int whh_off,
    unsigned int bias_off,
    unsigned int dst_off,
    unsigned int batch,
    unsigned int seq,
    unsigned int input_size,
    unsigned int hidden,
    unsigned int relu_flag
) {
    __shared__ float h_sh[RNN_MAX_H];

    unsigned int bi = blockIdx.x;
    unsigned int k = threadIdx.x;
    if (hidden > RNN_MAX_H || bi >= batch) return;

    if (k < hidden) {
        h_sh[k] = 0.0f;
    }
    __syncthreads();

    for (unsigned int t = 0; t < seq; ++t) {
        float h_k = 0.0f;
        if (k < hidden) {
            unsigned int x_base = x_off + (bi * seq + t) * input_size;
            float acc = arena[bias_off + k];
            unsigned int wih_row = wih_off + k * input_size;
            for (unsigned int j = 0u; j < input_size; ++j) {
                acc += arena[wih_row + j] * arena[x_base + j];
            }
            unsigned int whh_row = whh_off + k * hidden;
            for (unsigned int j = 0u; j < hidden; ++j) {
                acc += arena[whh_row + j] * h_sh[j];
            }
            h_k = relu_flag != 0u ? fmaxf(acc, 0.0f) : tanhf(acc);
        }
        __syncthreads();
        if (k < hidden) {
            h_sh[k] = h_k;
            arena[dst_off + (bi * seq + t) * hidden + k] = h_k;
        }
        __syncthreads();
    }
}
