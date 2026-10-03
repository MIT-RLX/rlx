// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

enable wgpu_cooperative_matrix;

// Split-write QKV variant of `matmul_coop_f32`. Same 32×32 / 16-hw-GEMM
// tile structure (`simdgroup_float8x8` on Apple, `coop_mat<f32>` in
// portable WGSL), but the epilogue routes each output column to one of
// three sinks (Q/K/V) based on `global_col` against `head_width = H·D`.
//
// Replaces (FusedMatMulBiasAct(qkv) → Narrow×3) with one dispatch on
// the CoopF32 path — same trick as `matmul_qkv.wgsl` but for the
// hardware-GEMM kernel that fires on aligned shapes (BERT-base ≥ b=32,
// NomicVision at every batch). Without this the CoopF32 path defaults
// to (CoopF32 → Narrow×3): CoopF32 writes the fused QKV, then 3 narrow
// dispatches each copy ~M·H·D values into the Q/K/V sink buffers.
// On NomicVision (12 layers × 3 narrows) that's 36 dispatches and
// ~3·M·H·D extra memory traffic per forward.

struct Params {
    m: u32,
    k: u32,
    n: u32,            // = 3 · head_width
    a_off: u32,
    b_off: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    head_width: u32,
    has_bias: u32,
    bias_off: u32,
    _p0: u32, _p1: u32, _p2: u32, _p3: u32, _p4: u32,
};

const TILE_M: u32 = 32u;
const TILE_N: u32 = 32u;
const TILE_K: u32 = 8u;

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;

var<workgroup> acc_scratch: array<f32, 1024>;
var<workgroup> a_stage:     array<f32, 256>;     // 32 × 8

@compute @workgroup_size(32)
fn matmul_qkv_coop_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let row_base = wid.y * TILE_M;
    let col_base = wid.x * TILE_N;

    // Zero acc_scratch (1024 f32; 32 elements per thread).
    for (var s: u32 = 0u; s < 32u; s = s + 1u) {
        acc_scratch[lid + s * 32u] = 0.0;
    }
    workgroupBarrier();

    var acc_00: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_01: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_02: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_03: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_10: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_11: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_12: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_13: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_20: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_21: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_22: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_23: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_30: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_31: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_32: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_33: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);

    let n_tiles = (params.k + TILE_K - 1u) / TILE_K;
    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        let k_off = t * TILE_K;
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let idx = lid + s * 32u;
            let r = idx / 8u;
            let c = idx % 8u;
            a_stage[idx] = arena[params.a_off + (row_base + r) * params.k + k_off + c];
        }
        workgroupBarrier();

        // ROW-MAJOR LOADS/STORES ARE LOAD-BEARING — the `T` suffixes are not
        // decoration. `coopLoad`/`coopStore` (no T) and `coopLoadT`/`coopStoreT`
        // differ only by naga's `row_major = function_name.ends_with("T")` in the
        // shared WGSL frontend, but that one bool reaches the two backends as
        // DIFFERENT things: the MSL backend feeds it to Metal's
        // `simdgroup_load(..., transpose_matrix)`, while the SPIR-V backend turns
        // it into `RowMajorKHR`/`ColumnMajorKHR` on
        // `OpCooperativeMatrixLoadKHR`. So the same WGSL does NOT mean the same
        // thing on Metal and Vulkan, and a convention validated on one is not
        // transferable to the other. Measured both ways (see below).
        //
        // With the non-T forms this kernel computed `b·a` instead of `a·b`: for
        // unrelated matrices those are near-orthogonal, which is the cos≈0.016
        // "orthogonal garbage" that kept Metal CoopF32 opt-in. Probe, at
        // M=N=32/K=8 with A a column vector and B a row vector — `a·b` is a dense
        // rank-1 outer product, `b·a` collapses to one value per 8x8 fragment:
        // out[0][0] was 0.01*SUM (c+1)^2 = 2.04 and out[0][8] was
        // 0.01*SUM (c+1)(c+9) = 4.92, both exactly `b·a`.
        //
        // The Vulkan siblings use a DIFFERENT and equally deliberate convention
        // (`coopLoad` on A, `coopLoadT` on B — see the header of
        // `matmul_coop_f16_vulkan.wgsl`, established on RTX). That convention was
        // measured here and is WRONG on Metal: it leaves only the first column of
        // each 8x8 tile correct and zeroes the rest. Do not unify them.
        let a_0: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[0u  ], 8u);
        let a_1: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[64u ], 8u);
        let a_2: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[128u], 8u);
        let a_3: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[192u], 8u);
        let b_row = params.b_off + k_off * params.n + col_base;
        let b_0: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row + 0u],  params.n);
        let b_1: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row + 8u],  params.n);
        let b_2: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row + 16u], params.n);
        let b_3: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row + 24u], params.n);

        acc_00 = coopMultiplyAdd(a_0, b_0, acc_00);
        acc_01 = coopMultiplyAdd(a_0, b_1, acc_01);
        acc_02 = coopMultiplyAdd(a_0, b_2, acc_02);
        acc_03 = coopMultiplyAdd(a_0, b_3, acc_03);
        acc_10 = coopMultiplyAdd(a_1, b_0, acc_10);
        acc_11 = coopMultiplyAdd(a_1, b_1, acc_11);
        acc_12 = coopMultiplyAdd(a_1, b_2, acc_12);
        acc_13 = coopMultiplyAdd(a_1, b_3, acc_13);
        acc_20 = coopMultiplyAdd(a_2, b_0, acc_20);
        acc_21 = coopMultiplyAdd(a_2, b_1, acc_21);
        acc_22 = coopMultiplyAdd(a_2, b_2, acc_22);
        acc_23 = coopMultiplyAdd(a_2, b_3, acc_23);
        acc_30 = coopMultiplyAdd(a_3, b_0, acc_30);
        acc_31 = coopMultiplyAdd(a_3, b_1, acc_31);
        acc_32 = coopMultiplyAdd(a_3, b_2, acc_32);
        acc_33 = coopMultiplyAdd(a_3, b_3, acc_33);
        workgroupBarrier();
    }

    coopStoreT(acc_00, &acc_scratch[0u   * 32u + 0u ], 32u);
    coopStoreT(acc_01, &acc_scratch[0u   * 32u + 8u ], 32u);
    coopStoreT(acc_02, &acc_scratch[0u   * 32u + 16u], 32u);
    coopStoreT(acc_03, &acc_scratch[0u   * 32u + 24u], 32u);
    coopStoreT(acc_10, &acc_scratch[8u   * 32u + 0u ], 32u);
    coopStoreT(acc_11, &acc_scratch[8u   * 32u + 8u ], 32u);
    coopStoreT(acc_12, &acc_scratch[8u   * 32u + 16u], 32u);
    coopStoreT(acc_13, &acc_scratch[8u   * 32u + 24u], 32u);
    coopStoreT(acc_20, &acc_scratch[16u  * 32u + 0u ], 32u);
    coopStoreT(acc_21, &acc_scratch[16u  * 32u + 8u ], 32u);
    coopStoreT(acc_22, &acc_scratch[16u  * 32u + 16u], 32u);
    coopStoreT(acc_23, &acc_scratch[16u  * 32u + 24u], 32u);
    coopStoreT(acc_30, &acc_scratch[24u  * 32u + 0u ], 32u);
    coopStoreT(acc_31, &acc_scratch[24u  * 32u + 8u ], 32u);
    coopStoreT(acc_32, &acc_scratch[24u  * 32u + 16u], 32u);
    coopStoreT(acc_33, &acc_scratch[24u  * 32u + 24u], 32u);
    workgroupBarrier();

    // Split-write epilogue. Identical layout decision to `matmul_qkv.wgsl`:
    // each output column is routed to Q (col < hw), K (hw ≤ col < 2·hw),
    // or V (2·hw ≤ col < 3·hw). Sink stride is `head_width`; bias
    // remains a [3·head_width] tensor read at the matmul column.
    let hw = params.head_width;
    for (var s: u32 = 0u; s < 32u; s = s + 1u) {
        let idx = lid + s * 32u;
        let r = idx / 32u;
        let c = idx % 32u;
        let global_row = row_base + r;
        let global_col = col_base + c;
        if (global_row >= params.m || global_col >= params.n) { continue; }
        var v: f32 = acc_scratch[idx];
        if (params.has_bias != 0u) {
            v = v + arena[params.bias_off + global_col];
        }

        var sink_off: u32 = 0u;
        var col_in_sink: u32 = 0u;
        if (global_col < hw) {
            sink_off = params.q_off;
            col_in_sink = global_col;
        } else if (global_col < 2u * hw) {
            sink_off = params.k_off;
            col_in_sink = global_col - hw;
        } else {
            sink_off = params.v_off;
            col_in_sink = global_col - 2u * hw;
        }
        arena[sink_off + global_row * hw + col_in_sink] = v;
    }
}
