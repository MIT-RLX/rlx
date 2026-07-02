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

// Mamba-style selective state-space scan.
//   h[t] = exp(Δ[t] * A) * h[t-1] + Δ[t] * B[t] * x[t]
//   y[t] = C[t] * h[t]
//
// Parallelization: one thread per (batch, channel) pair. Each thread
// walks the seq dimension sequentially, carrying its own state vector
// of length `state_size` in private storage. Static cap of 256 covers
// every practical Mamba config (typical n=16); larger configs error
// out at compile time on the Rust side.

struct Params {
    batch: u32,
    seq: u32,
    hidden: u32,
    state_size: u32,
    x_off: u32,
    delta_off: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    out_off: u32,
    // PLAN L1 — full-extent seq stride for per-batch offset math.
    seq_stride: u32,
    _p1: u32, _p2: u32, _p3: u32, _p4: u32, _p5: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const MAX_STATE: u32 = 256u;

@compute @workgroup_size(64)
fn selective_scan(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let id = gid.x + gid.y * ngs.x * 64u;
    let total = params.batch * params.hidden;
    if (id >= total) { return; }
    if (params.state_size > MAX_STATE) { return; }
    let bi = id / params.hidden;
    let ci = id % params.hidden;

    var state: array<f32, 256>;
    for (var i: u32 = 0u; i < params.state_size; i = i + 1u) {
        state[i] = 0.0;
    }

    let a_base = ci * params.state_size;

    for (var si: u32 = 0u; si < params.seq; si = si + 1u) {
        // Per-(bi, si) offset uses full-extent seq_stride so per-batch
        // strides stay correct when params.seq is scaled at runtime.
        let x_idx = (bi * params.seq_stride + si) * params.hidden + ci;
        let xv = arena[params.x_off + x_idx];
        let d  = arena[params.delta_off + x_idx];
        let bc_base = (bi * params.seq_stride + si) * params.state_size;

        var acc: f32 = 0.0;
        for (var ni: u32 = 0u; ni < params.state_size; ni = ni + 1u) {
            let da = exp(d * arena[params.a_off + a_base + ni]);
            state[ni] = da * state[ni] + d * arena[params.b_off + bc_base + ni] * xv;
            acc = acc + arena[params.c_off + bc_base + ni] * state[ni];
        }
        arena[params.out_off + x_idx] = acc;
    }
}
