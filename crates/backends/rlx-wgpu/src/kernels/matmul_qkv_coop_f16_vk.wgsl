// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// RLX — Split-write QKV variant of `matmul_coop_f16_vulkan`.

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
    q_off: u32,
    k_off: u32,
    v_off: u32,
    head_width: u32,
    has_bias: u32,
    bias_off: u32,
    a_off_f16: u32,  // A in the f16 window (`a_off` is the f32 arena's)
    _p1: u32,
    _p2: u32,
    _p3: u32,
    _p4: u32,
};

const TILE: u32 = 16u;
const K_SLAB: u32 = 16u;

@group(0) @binding(0) var<storage, read>       arena_f16: array<f16>;
@group(0) @binding(1) var<storage, read_write> arena: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

var<workgroup> f32_tile: array<f32, 256>;
var<workgroup> h16_tile: array<f16, 256>;

@compute @workgroup_size(64, 1, 1)
fn matmul_qkv_coop_f16_vk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tile_row = wid.x * TILE;
    let tile_col = wid.y * TILE;
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
            let a_ptr = params.a_off_f16 + tile_row * params.k + k_off;
            let b_ptr = params.b_off + k_off * params.n + tile_col;
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

    let hw = params.head_width;
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

        var sink_off: u32;
        var col_in_sink: u32;
        if (gc < hw) {
            sink_off = params.q_off;
            col_in_sink = gc;
        } else if (gc < 2u * hw) {
            sink_off = params.k_off;
            col_in_sink = gc - hw;
        } else {
            sink_off = params.v_off;
            col_in_sink = gc - 2u * hw;
        }
        arena[sink_off + gr * hw + col_in_sink] = v;
    }
}
