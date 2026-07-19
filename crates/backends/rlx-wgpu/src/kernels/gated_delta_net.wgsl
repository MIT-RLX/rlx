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

// Gated-DeltaNet scan (f32). One workgroup per (batch, head); workgroup
// size is GDN_MAX_N=128 with early-out for tid >= n. Matches Metal
// `gated_delta_net` / CPU `execute_gated_delta_net_f32`.
//
// Barriers sit in uniform control flow (outside the active-lane guard),
// same discipline as `gru.wgsl`.

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
    _p1: u32,
    _p2: u32,
    _p3: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const GDN_MAX_N: u32 = 128u;
var<workgroup> sk_sh: array<f32, 128>;

@compute @workgroup_size(128)
fn gated_delta_net(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let gid = wid.x;
    let tid = lid.x;
    let b = params.batch;
    let s = params.seq;
    let h = params.heads;
    let n = params.state_size;
    let lane_on = (n <= GDN_MAX_N) && (gid < b * h) && (tid < n);

    let bi = gid / h;
    let hi = gid % h;
    let j = tid;
    let scale = inverseSqrt(f32(n));

    let s_base = params.state_off + (bi * h + hi) * n * n;

    // Zero ephemeral state when not carrying (tid 0 only).
    if (lane_on && params.use_carry == 0u && tid == 0u) {
        for (var i: u32 = 0u; i < n * n; i = i + 1u) {
            arena[s_base + i] = 0.0;
        }
    }
    workgroupBarrier();

    let hs_n = h * n;

    for (var ti: u32 = 0u; ti < s; ti = ti + 1u) {
        var q_row: u32 = 0u;
        var k_row: u32 = 0u;
        var v_row: u32 = 0u;
        var g_exp: f32 = 1.0;
        var beta_t: f32 = 0.0;
        if (lane_on) {
            let qkv_step = bi * params.seq_stride * hs_n + ti * hs_n + hi * n;
            let gb_step = bi * params.seq_stride * h + ti * h + hi;
            q_row = params.q_off + qkv_step;
            k_row = params.k_off + qkv_step;
            v_row = params.v_off + qkv_step;
            g_exp = exp(arena[params.g_off + gb_step]);
            beta_t = arena[params.beta_off + gb_step];
        }

        if (lane_on && tid == 0u) {
            for (var idx: u32 = 0u; idx < n * n; idx = idx + 1u) {
                arena[s_base + idx] = arena[s_base + idx] * g_exp;
            }
        }
        workgroupBarrier();

        var acc: f32 = 0.0;
        if (lane_on) {
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                acc = acc + arena[s_base + i * n + j] * arena[k_row + i];
            }
            sk_sh[j] = acc;
        }
        workgroupBarrier();

        if (lane_on) {
            sk_sh[j] = (arena[v_row + j] - sk_sh[j]) * beta_t;
        }
        workgroupBarrier();

        if (lane_on) {
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                let ki = arena[k_row + i];
                arena[s_base + i * n + j] = arena[s_base + i * n + j] + ki * sk_sh[j];
            }
        }
        workgroupBarrier();

        if (lane_on) {
            let out_row = params.out_off + bi * params.seq_stride * hs_n + ti * hs_n + hi * n;
            acc = 0.0;
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                acc = acc + arena[s_base + i * n + j] * arena[q_row + i];
            }
            arena[out_row + j] = acc * scale;
        }
        workgroupBarrier();
    }
}
