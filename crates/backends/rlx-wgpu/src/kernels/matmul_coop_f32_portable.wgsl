// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// Portable cooperative-matrix matmul for Vulkan / DX12 (8×8 f32 tiles).
// Matches the wgpu `cooperative_matrix/shader.wgsl` load/store pattern:
// row-major `coopLoad`, dispatch (M/8, N/8).

enable wgpu_cooperative_matrix;

struct Params {
    m: u32,
    k: u32,
    n: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    batch: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    c_batch_stride: u32,
    has_bias: u32,
    bias_off: u32,
    act_id: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

const TILE: u32 = 8u;

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;

fn gelu_erf(x: f32) -> f32 {
    let arg = x * 0.70710678118654752;
    let s = select(-1.0, 1.0, arg >= 0.0);
    let xa = abs(arg);
    let t = 1.0 / (1.0 + 0.3275911 * xa);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))));
    let e = s * (1.0 - poly * exp(-xa * xa));
    return 0.5 * x * (1.0 + e);
}

fn apply_act(v_in: f32) -> f32 {
    var v = v_in;
    if (params.act_id == 0xFFFFu) { return v; }
    switch (params.act_id) {
        case 0u: { v = max(v, 0.0); }
        case 1u: { v = 1.0 / (1.0 + exp(-clamp(v, -88.0, 88.0))); }
        case 2u: { v = tanh(clamp(v, -15.0, 15.0)); }
        case 5u: { v = sqrt(v); }
        case 7u: { v = -v; }
        case 8u: { v = abs(v); }
        case 9u: { v = gelu_erf(v); }
        case 11u: {
            let c = 0.7978845608028654;
            let x3 = v * v * v;
            let inner = clamp(c * (v + 0.044715 * x3), -15.0, 15.0);
            v = 0.5 * v * (1.0 + tanh(inner));
        }
        case 10u: {
            let nx = clamp(-v, -88.0, 88.0);
            v = v / (1.0 + exp(nx));
        }
        default: {}
    }
    return v;
}

@compute @workgroup_size(8, 8, 1)
fn matmul_coop_f32_portable(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bz = wid.z;
    if (bz >= params.batch) { return; }

    let tile_row = wid.x * TILE;
    let tile_col = wid.y * TILE;

    let a_base = params.a_off + bz * params.a_batch_stride;
    let b_base = params.b_off + bz * params.b_batch_stride;
    let c_base = params.c_off + bz * params.c_batch_stride;

    var acc: coop_mat8x8<f32, C> = coop_mat8x8<f32, C>();

    let n_tiles = (params.k + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        let k_off = t * TILE;
        let a_ptr = a_base + tile_row * params.k + k_off;
        let b_ptr = b_base + k_off * params.n + tile_col;
        let a_tile: coop_mat8x8<f32, A> = coopLoad<coop_mat8x8<f32, A>>(&arena[a_ptr], params.k);
        let b_tile: coop_mat8x8<f32, B> = coopLoad<coop_mat8x8<f32, B>>(&arena[b_ptr], params.n);
        acc = coopMultiplyAdd(a_tile, b_tile, acc);
    }

    let c_ptr = c_base + tile_row * params.n + tile_col;
    coopStore(acc, &arena[c_ptr], params.n);

    let gr = tile_row + lid.y;
    let gc = tile_col + lid.x;
    if (gr >= params.m || gc >= params.n) { return; }
    let out_idx = c_base + gr * params.n + gc;
    var v = arena[out_idx];
    if (params.has_bias != 0u) {
        v = v + arena[params.bias_off + gc];
    }
    v = apply_act(v);
    arena[out_idx] = v;
}
