// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// RLX — Split-write QKV variant of `matmul_coop_f16_vulkan`.

enable f16;

//
// ── COOPERATIVE-MATRIX LAYOUT CONVENTION (read before editing) ──────────────
//
// naga's shared WGSL frontend sets `row_major = function_name.ends_with("T")`,
// so `coopLoad`/`coopLoadT` differ by one bool. That bool reaches the two
// backends as DIFFERENT THINGS: the MSL backend passes it to Metal's
// `simdgroup_load(..., transpose_matrix)`, the SPIR-V backend turns it into
// `RowMajorKHR`/`ColumnMajorKHR` on `OpCooperativeMatrixLoadKHR`. Identical
// WGSL therefore does NOT mean identical behaviour on Metal and Vulkan.
//
// This is measured, not inferred. The Metal kernels
// (`matmul_coop_f32.wgsl`, `matmul_qkv_coop_f32.wgsl`, `matmul_coop16.wgsl`)
// needed `coopLoadT` + `coopStoreT` on EVERY operand; with the plain forms they
// computed `b·a` instead of `a·b` (near-orthogonal for unrelated matrices —
// the cos 0.016 "orthogonal garbage" that kept Metal CoopF32 opt-in for a long
// time). The Vulkan convention in this file — `coopLoad` on A, `coopLoadT` on B
// — was then tried on Metal and is WRONG there: it leaves only the first column
// of each 8x8 tile correct and zeroes the rest. So the two conventions are
// genuinely different and must NOT be unified. These Vulkan kernels were
// deliberately left alone.
//
// This file holds the REFERENCE Vulkan convention (`coopLoad` on A,
// `coopLoadT` on B), established on RTX hardware. Note that its `_widen`
// sibling — selected purely by `n > 768` — uses `coopLoad` on B against the
// same pointer and stride, which contradicts this file. See the ⚠ block in
// `matmul_coop_f16_vulkan_widen.wgsl`; one of the two is wrong.
//
// HOW THIS IS HANDLED NOW: `src/coop_probe.rs` runs a non-commuting GEMM probe
// through this kernel on the actual device, once per process, and
// `derive_matmul_compute` / `pick_coop_f16_vk_matmul` believe the result. If
// this shader computes `a·b` here it is used; if it does not, it is reported
// ineligible and the caller falls back to the portable tiled matmul — slower,
// correct. So the open question above can no longer produce wrong numbers on
// any device, including ones no developer machine can reach. It is still worth
// settling on real hardware, because the answer decides whether the fast path
// is available at all; `tests/coop_self_check.rs` prints the verdict per kernel.
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
    _p0: u32,
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
            let a_ptr = params.a_off + tile_row * params.k + k_off;
            let b_ptr = params.b_off + k_off * params.n + tile_col;
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
