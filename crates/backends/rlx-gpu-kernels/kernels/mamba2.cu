// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Mamba-2 / SSD scalar-decay SSM scan.
//   dA = exp(dt * a);  S = dA * S + (dt * x) ⊗ b;  y = Σ_n S[:,n] * c[n]
//
// One thread per (batch, head, head_dim_pos). Each thread carries its own
// N-state vector in private storage and walks the seq dimension. Static cap
// of 256 covers every practical config (typical n=16). Matches
// `execute_mamba2_f32` / Metal `mamba2` / wgpu `mamba2.wgsl`.
// Inputs (f32): x [B,S,H,P], dt [B,S,H], a [H], b/c [B,S,H,N];
// output y [B,S,H,P]. Host path when state_size > MAMBA2_MAX_N.

#define MAMBA2_MAX_N 256u

extern "C" __global__ void mamba2(
    float* arena,
    unsigned int x_off,
    unsigned int dt_off,
    unsigned int a_off,
    unsigned int b_off,
    unsigned int c_off,
    unsigned int dst_off,
    unsigned int batch,
    unsigned int seq,
    unsigned int heads,
    unsigned int head_dim,
    unsigned int state_size
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int p = head_dim;
    unsigned int n = state_size;
    unsigned int total = batch * heads * p;
    if (id >= total || n > MAMBA2_MAX_N) return;

    unsigned int pi = id % p;
    unsigned int hi = (id / p) % heads;
    unsigned int bi = id / (p * heads);

    float state[256];
    for (unsigned int i = 0u; i < n; ++i) {
        state[i] = 0.0f;
    }
    float ah = arena[a_off + hi];

    for (unsigned int t = 0u; t < seq; ++t) {
        unsigned int bsh = (bi * seq + t) * heads + hi;
        float dt_t = arena[dt_off + bsh];
        float da = expf(dt_t * ah);
        float dtx = dt_t * arena[x_off + bsh * p + pi];
        unsigned int bc = bsh * n;
        float acc = 0.0f;
        for (unsigned int ni = 0u; ni < n; ++ni) {
            float st = da * state[ni] + dtx * arena[b_off + bc + ni];
            state[ni] = st;
            acc += st * arena[c_off + bc + ni];
        }
        arena[dst_off + bsh * p + pi] = acc;
    }
}
