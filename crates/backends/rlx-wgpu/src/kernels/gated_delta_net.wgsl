// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Gated-DeltaNet scan (f32). One workgroup per (batch, head); workgroup
// size is GDN_MAX_N=128, thread j owning column j of the head's [n, n] state.
// Matches Metal `gated_delta_net` / CPU `execute_gated_delta_net_f32`.
//
// Per step, with S the state, k/q/v the head's rows, g the decay, beta the
// write strength:
//   S[i,j] *= exp(g[i] or g)       (decay)
//   d_j     = (v_j - Σ_i S[i,j]·k_i) · beta
//   S[i,j] += k_i · d_j            (delta-rule write)
//   out_j   = Σ_i S[i,j]·q_i / sqrt(n)
// Every term of column j depends only on column j, so each thread runs its
// column on its own: decay fused into the first pass, write into the second,
// no barriers. (Thread 0 used to decay all n² entries alone while the rest
// waited — 16384 serial read-modify-writes per step for n = 128.)

struct Params {
    batch: u32,
    seq: u32,
    heads: u32,
    state_size: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    g_off: u32,
    beta_off: u32,
    state_off: u32,
    out_off: u32,
    use_carry: u32,
    // PLAN L1 — full-extent seq stride for per-batch offset math.
    seq_stride: u32,
    // 1 = per-channel gate (g is [b,s,h,n], Kimi-K3 KDA); 0 = per-head scalar.
    gate_per_channel: u32,
    _p2: u32,
    _p3: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const GDN_MAX_N: u32 = 128u;

@compute @workgroup_size(128)
fn gated_delta_net(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let gid = wid.x;
    let j = lid.x;
    let b = params.batch;
    let h = params.heads;
    let n = params.state_size;
    if (n > GDN_MAX_N || gid >= b * h || j >= n) { return; }

    let bi = gid / h;
    let hi = gid % h;
    let scale = inverseSqrt(f32(n));
    let s_base = params.state_off + (bi * h + hi) * n * n;
    let hs_n = h * n;

    // Zero the state when not carrying it: this thread's column.
    if (params.use_carry == 0u) {
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            arena[s_base + i * n + j] = 0.0;
        }
    }

    for (var ti: u32 = 0u; ti < params.seq; ti = ti + 1u) {
        let qkv_step = bi * params.seq_stride * hs_n + ti * hs_n + hi * n;
        let gb_step = bi * params.seq_stride * h + ti * h + hi;
        let q_row = params.q_off + qkv_step;
        let k_row = params.k_off + qkv_step;
        let v_row = params.v_off + qkv_step;
        let g_exp = exp(arena[params.g_off + gb_step]);
        let beta_t = arena[params.beta_off + gb_step];
        let per_channel = params.gate_per_channel != 0u;
        // Per-channel decay rows: g is [b,s,h,n] (Kimi-K3 KDA).
        let g_pc_base = params.g_off + qkv_step;

        // Pass 1: decay column j and accumulate Σ_i S[i,j]·k_i.
        var acc: f32 = 0.0;
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let a = select(g_exp, exp(arena[g_pc_base + i]), per_channel);
            let sv = arena[s_base + i * n + j] * a;
            arena[s_base + i * n + j] = sv;
            acc = acc + sv * arena[k_row + i];
        }
        let delta = (arena[v_row + j] - acc) * beta_t;

        // Pass 2: delta-rule write and the output dot in one sweep.
        var out_acc: f32 = 0.0;
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let sv = arena[s_base + i * n + j] + arena[k_row + i] * delta;
            arena[s_base + i * n + j] = sv;
            out_acc = out_acc + sv * arena[q_row + i];
        }
        arena[params.out_off + qkv_step + j] = out_acc * scale;
    }
}
