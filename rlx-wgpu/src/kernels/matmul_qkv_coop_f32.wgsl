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

    var acc_00: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_01: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_02: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_03: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_10: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_11: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_12: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_13: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_20: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_21: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_22: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_23: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_30: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_31: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_32: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_33: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);

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

        let a_0: coop_mat8x8<f32, A> = coopLoad<coop_mat8x8<f32, A>>(&a_stage[0u  ], 8u);
        let a_1: coop_mat8x8<f32, A> = coopLoad<coop_mat8x8<f32, A>>(&a_stage[64u ], 8u);
        let a_2: coop_mat8x8<f32, A> = coopLoad<coop_mat8x8<f32, A>>(&a_stage[128u], 8u);
        let a_3: coop_mat8x8<f32, A> = coopLoad<coop_mat8x8<f32, A>>(&a_stage[192u], 8u);
        let b_row = params.b_off + k_off * params.n + col_base;
        let b_0: coop_mat8x8<f32, B> = coopLoad<coop_mat8x8<f32, B>>(&arena[b_row + 0u],  params.n);
        let b_1: coop_mat8x8<f32, B> = coopLoad<coop_mat8x8<f32, B>>(&arena[b_row + 8u],  params.n);
        let b_2: coop_mat8x8<f32, B> = coopLoad<coop_mat8x8<f32, B>>(&arena[b_row + 16u], params.n);
        let b_3: coop_mat8x8<f32, B> = coopLoad<coop_mat8x8<f32, B>>(&arena[b_row + 24u], params.n);

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

    coopStore(acc_00, &acc_scratch[0u   * 32u + 0u ], 32u);
    coopStore(acc_01, &acc_scratch[0u   * 32u + 8u ], 32u);
    coopStore(acc_02, &acc_scratch[0u   * 32u + 16u], 32u);
    coopStore(acc_03, &acc_scratch[0u   * 32u + 24u], 32u);
    coopStore(acc_10, &acc_scratch[8u   * 32u + 0u ], 32u);
    coopStore(acc_11, &acc_scratch[8u   * 32u + 8u ], 32u);
    coopStore(acc_12, &acc_scratch[8u   * 32u + 16u], 32u);
    coopStore(acc_13, &acc_scratch[8u   * 32u + 24u], 32u);
    coopStore(acc_20, &acc_scratch[16u  * 32u + 0u ], 32u);
    coopStore(acc_21, &acc_scratch[16u  * 32u + 8u ], 32u);
    coopStore(acc_22, &acc_scratch[16u  * 32u + 16u], 32u);
    coopStore(acc_23, &acc_scratch[16u  * 32u + 24u], 32u);
    coopStore(acc_30, &acc_scratch[24u  * 32u + 0u ], 32u);
    coopStore(acc_31, &acc_scratch[24u  * 32u + 8u ], 32u);
    coopStore(acc_32, &acc_scratch[24u  * 32u + 16u], 32u);
    coopStore(acc_33, &acc_scratch[24u  * 32u + 24u], 32u);
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
        if (global_row >= params.m) { continue; }
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
