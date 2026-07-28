// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
/* Standalone validation harness for rlx_qnn_shim.c.
 *
 *   shim_test <path-to-libQnnCpu.so> [M K N]
 *
 * Fills deterministic inputs, computes a reference matmul in C, runs the shim
 * against the real QNN backend, and checks parity. Not shipped in the Rust
 * build — it's the C-level analog of the codegen path's verify.py. */

#include <stdio.h>
#include <stdlib.h>

#include "rlx_qnn_shim.h"

static float ref_at(const float *in0, const float *in1,
                    uint32_t i, uint32_t j, uint32_t K, uint32_t N) {
    float acc = 0.0f;
    for (uint32_t k = 0; k < K; ++k) acc += in0[i * K + k] * in1[k * N + j];
    return acc;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <backend_lib> [M K N]\n", argv[0]);
        return 2;
    }
    const char *backend = argv[1];
    uint32_t M = argc > 4 ? (uint32_t)atoi(argv[2]) : 8;
    uint32_t K = argc > 4 ? (uint32_t)atoi(argv[3]) : 16;
    uint32_t N = argc > 4 ? (uint32_t)atoi(argv[4]) : 4;

    float *in0 = malloc(sizeof(float) * M * K);
    float *in1 = malloc(sizeof(float) * K * N);
    float *out = calloc((size_t)M * N, sizeof(float));
    for (uint32_t i = 0; i < M * K; ++i) in0[i] = (float)((int)(i % 7) - 3) * 0.5f;
    for (uint32_t i = 0; i < K * N; ++i) in1[i] = (float)((int)(i % 5) - 2) * 0.25f;

    uint64_t err = 0;
    int rc = rlx_qnn_matmul_f32(backend, M, K, N, in0, in1, out, &err);
    if (rc != 0) {
        fprintf(stderr, "shim failed: step=%d qnn_err=0x%llx\n", -rc,
                (unsigned long long)err);
        return 1;
    }

    float max_diff = 0.0f;
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t j = 0; j < N; ++j) {
            float d = out[i * N + j] - ref_at(in0, in1, i, j, K, N);
            if (d < 0) d = -d;
            if (d > max_diff) max_diff = d;
        }

    free(in0); free(in1); free(out);
    if (max_diff < 1e-3f) {
        printf("SUCCESS! %ux%ux%u matmul on QNN backend, max_diff=%.2e\n", M, K, N, max_diff);
        return 0;
    }
    printf("FAIL: max_diff=%.4e exceeds 1e-3\n", max_diff);
    return 1;
}
