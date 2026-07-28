// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Packed GatedResidual backward: out = [dx ∥ dy ∥ dgate] (1-D floats).
// Launch: grid=(mod_rows,1,1), block=(256,1,1).

struct Params {
    mod_rows: u32,
    seq_per_mod: u32,
    inner: u32,
    y_off: u32,
    gate_off: u32,
    dy_off: u32,
    out_off: u32,
    _p0: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(256)
fn gated_residual_backward(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m = wid.x;
    if (m >= params.mod_rows || params.inner == 0u) { return; }
    let tid = lid.x;
    let bsz = 256u;
    let inner = params.inner;
    let nx = params.mod_rows * params.seq_per_mod * inner;
    let gate_base = m * inner;

    var i = tid;
    loop {
        if (i >= inner) { break; }
        var acc: f32 = 0.0;
        let g = arena[params.gate_off + gate_base + i];
        for (var seq: u32 = 0u; seq < params.seq_per_mod; seq = seq + 1u) {
            let row = m * params.seq_per_mod + seq;
            let idx = row * inner + i;
            let d = arena[params.dy_off + idx];
            arena[params.out_off + idx] = d;
            arena[params.out_off + nx + idx] = d * g;
            acc = acc + d * arena[params.y_off + idx];
        }
        arena[params.out_off + 2u * nx + gate_base + i] = acc;
        i += bsz;
    }
}
