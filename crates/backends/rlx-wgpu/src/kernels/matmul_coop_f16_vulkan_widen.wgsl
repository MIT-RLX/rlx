// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// RLX — wide-N variant, now identical to matmul_coop_f16_vulkan.wgsl.

enable f16;

//
// ── COOPERATIVE-MATRIX LAYOUT CONVENTION (read before editing) ──────────────
//
// naga sets `row_major = function_name.ends_with("T")`: on SPIR-V `coopLoadT` /
// `coopStoreT` are `RowMajorKHR` and the plain forms `ColumnMajorKHR`. A, B and
// the output tile are all row-major here (`a_ptr + r*k`, `b_ptr + r*n`,
// `tile[r*TILE + c]`), so every load and store is the `T` form — the same
// spelling the Metal kernels (`matmul_coop_f32.wgsl`, `matmul_coop16.wgsl`) use.
//
// Measured on an RTX 3080 Ti (Vulkan, VK_KHR_cooperative_matrix) with the
// non-commuting probe in `src/coop_probe.rs`, over all eight load/store
// combinations: only `coopLoadT` on A and B with `coopStoreT` computes `a·b`.
// The spelling this family shipped with (`coopLoad` on A, `coopStore` on the
// tile) failed it in every variant, so the runtime probe had been refusing the
// whole path on that device.
//
// The `_widen` siblings read B with `coopLoad` and were picked for `n > 768`.
// A layout does not depend on N, and that form computes `a·bᵀ`. They now match
// their base kernels token for token (save the entry point) and stay only
// because the size-based selection still names them.
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
    a_off_f16: u32,  // A in the f16 window (`a_off` is the f32 arena's)
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

// Exact GELU — A&S 7.1.26 erf, the same as `matmul.wgsl` / `matmul_coop_f32.wgsl`
// and rlx-cpu's `scalar_gelu`. Opcode 9 (`Activation::Gelu`) is the EXACT form;
// only opcode 11 (`GeluApprox`) is the tanh form. This kernel used to map both
// to tanh, so a fused matmul+bias+GELU here drifted ~3e-4 from CPU at |x|≈2–3.
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

@compute @workgroup_size(64, 1, 1)
fn matmul_coop_f16_vulkan_widen(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bz = wid.z;
    if (bz >= params.batch) { return; }

    let tile_row = wid.x * TILE;
    let tile_col = wid.y * TILE;

    let a_base = params.a_off_f16 + bz * params.a_batch_stride;
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
                coopLoadT<coop_mat16x16<f16, A>>(&arena_f16[a_ptr], params.k);
            let b_tile: coop_mat16x16<f16, B> =
                coopLoadT<coop_mat16x16<f16, B>>(&arena_f16[b_ptr], params.n);
            acc = coopMultiplyAdd(a_tile, b_tile, acc);
            k_off = k_off + TILE;
        }

        coopStoreT(acc, &h16_tile[0], TILE);
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
