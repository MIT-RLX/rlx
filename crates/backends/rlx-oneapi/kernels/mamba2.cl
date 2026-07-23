// Mamba-2 / SSD scalar-decay SSM scan.
//   dA = exp(dt * a);  S = dA * S + (dt * x) ⊗ b;  y = Σ_n S[:,n] * c[n]
// One work-item per (batch, head, head_dim_pos). Private N-state ≤ 256.
// Mirrors wgpu `mamba2.wgsl` / CUDA `mamba2.cu`. Larger state → host fallback.

#define MAMBA2_MAX_N 256u

__kernel void mamba2(__global float* arena,
                     uint batch, uint seq, uint heads, uint head_dim,
                     uint state_size,
                     uint x_off, uint dt_off, uint a_off,
                     uint b_off, uint c_off, uint out_off,
                     uint seq_stride) {
    uint id = get_global_id(0);
    uint p = head_dim;
    uint n = state_size;
    uint total = batch * heads * p;
    if (id >= total || n > MAMBA2_MAX_N) return;

    uint pi = id % p;
    uint hi = (id / p) % heads;
    uint bi = id / (p * heads);

    float state[256];
    for (uint i = 0u; i < n; i++) state[i] = 0.0f;
    float ah = arena[a_off + hi];

    for (uint si = 0u; si < seq; si++) {
        uint bsh = (bi * seq_stride + si) * heads + hi;
        float dt_t = arena[dt_off + bsh];
        float da = exp(dt_t * ah);
        float dtx = dt_t * arena[x_off + bsh * p + pi];
        uint bc = bsh * n;
        float acc = 0.0f;
        for (uint ni = 0u; ni < n; ni++) {
            float st = da * state[ni] + dtx * arena[b_off + bc + ni];
            state[ni] = st;
            acc += st * arena[c_off + bc + ni];
        }
        arena[out_off + bsh * p + pi] = acc;
    }
}
