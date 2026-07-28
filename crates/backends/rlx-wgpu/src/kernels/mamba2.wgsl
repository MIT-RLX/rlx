// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Mamba-2 / SSD scalar-decay SSM scan.
//   dA = exp(dt * a);  S = dA * S + (dt * x) ⊗ b;  y = Σ_n S[:,n] * c[n]
//
// One thread per (batch, head, head_dim_pos). Each thread carries its own
// N-state vector in private storage and walks the seq dimension. Static cap
// of 256 covers every practical config (typical n=16). Inputs (f32):
//   x [B,S,H,P], dt [B,S,H], a [H], b/c [B,S,H,N]; output y [B,S,H,P].

struct Params {
    batch: u32,
    seq: u32,
    heads: u32,
    head_dim: u32,
    state_size: u32,
    x_off: u32,
    dt_off: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    out_off: u32,
    // Full-extent seq stride for per-(batch,seq) offset math (stays at
    // compile-time seq even when params.seq is scaled at runtime).
    seq_stride: u32,
    _p1: u32, _p2: u32, _p3: u32, _p4: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const MAX_STATE: u32 = 256u;

@compute @workgroup_size(64)
fn mamba2(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let id = gid.x + gid.y * ngs.x * 64u;
    let p = params.head_dim;
    let n = params.state_size;
    let total = params.batch * params.heads * p;
    if (id >= total) { return; }
    if (n > MAX_STATE) { return; }

    let pi = id % p;
    let hi = (id / p) % params.heads;
    let bi = id / (p * params.heads);

    var state: array<f32, 256>;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        state[i] = 0.0;
    }
    let ah = arena[params.a_off + hi];

    for (var si: u32 = 0u; si < params.seq; si = si + 1u) {
        let bsh = (bi * params.seq_stride + si) * params.heads + hi;
        let dt_t = arena[params.dt_off + bsh];
        let da = exp(dt_t * ah);
        let dtx = dt_t * arena[params.x_off + bsh * p + pi];
        let bc = bsh * n;
        var acc: f32 = 0.0;
        for (var ni: u32 = 0u; ni < n; ni = ni + 1u) {
            let st = da * state[ni] + dtx * arena[params.b_off + bc + ni];
            state[ni] = st;
            acc = acc + st * arena[params.c_off + bc + ni];
        }
        arena[params.out_off + bsh * p + pi] = acc;
    }
}
