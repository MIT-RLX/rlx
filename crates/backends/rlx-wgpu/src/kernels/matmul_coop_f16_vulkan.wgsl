// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// Vulkan/DX12 16×16 f16 cooperative matmul with K-slab f32 reduction.
// RTX row-major GEMM: coopLoad on A, coopLoadT on B (N <= 768).

enable f16;
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
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

const TILE: u32 = 16u;
const K_SLAB: u32 = 16u;

@group(0) @binding(0) var<storage, read>         arena_f16: array<f16>;
@group(0) @binding(1) var<storage, read_write>   arena: array<f32>;
@group(0) @binding(2) var<uniform>               params: Params;

var<workgroup> f32_tile: array<f32, 256>;
var<workgroup> h16_tile: array<f16, 256>;

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
        case 9u, 11u: {
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

@compute @workgroup_size(64, 1, 1)
fn matmul_coop_f16_vulkan(
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

    let lane = lid.x;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        f32_tile[lane * 4u + i] = 0.0;
    }
    workgroupBarrier();

    var k_slab: u32 = 0u;
    while (k_slab < params.k) {
        let slab_end = min(k_slab + K_SLAB, params.k);
        var acc: coop_mat16x16<f16, C> = coop_mat16x16<f16, C>();

        var k_off: u32 = k_slab;
        while (k_off < slab_end) {
            let a_ptr = a_base + tile_row * params.k + k_off;
            let b_ptr = b_base + k_off * params.n + tile_col;
            let a_tile: coop_mat16x16<f16, A> =
                coopLoad<coop_mat16x16<f16, A>>(&arena_f16[a_ptr], params.k);
            let b_tile: coop_mat16x16<f16, B> =
                coopLoadT<coop_mat16x16<f16, B>>(&arena_f16[b_ptr], params.n);
            acc = coopMultiplyAdd(a_tile, b_tile, acc);
            k_off = k_off + TILE;
        }

        coopStore(acc, &h16_tile[0], TILE);
        workgroupBarrier();

        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let idx = lane * 4u + i;
            f32_tile[idx] = f32_tile[idx] + f32(h16_tile[idx]);
        }
        workgroupBarrier();

        k_slab = slab_end;
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let idx = lane * 4u + i;
        let r = idx / TILE;
        let c = idx % TILE;
        let gr = tile_row + r;
        let gc = tile_col + c;
        if (gr >= params.m || gc >= params.n) { continue; }
        var v = f32_tile[idx];
        if (params.has_bias != 0u) {
            v = v + arena[params.bias_off + gc];
        }
        v = apply_act(v);
        arena[c_base + gr * params.n + gc] = v;
    }
}
