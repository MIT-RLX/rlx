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

// Pure-f32 cooperative-matrix matmul. Same structural design as
// `matmul_coop16.wgsl` (32×32 output tile = 4×4 sub-tiles of 8×8 hw
// GEMM, 32-thread workgroup) but every coop_mat is f32. On Apple this
// lowers to `simdgroup_float8x8` — same hardware GEMM unit as
// `simdgroup_half8x8` but with f32 operands and f32 accumulator, so
// no overflow risk in deep BERT FFN sums (3072-element accumulations
// of ~5-magnitude activations).
//
// Why this exists: opportunistically enabling matmul_coop16 for f32
// IR tags broke BERT cosine (0.4-0.6 vs 1.0) — the f16 accumulator
// overflows on FFN sums. Naga 29 can't compile the mixed
// `coop_mat<f32, C>` + `coop_mat<f16, A/B>` form, so the fix is to
// stay all-f32: same 7-13× speedup over `matmul_wide` (if Apple ALU
// is similar for float8x8 vs half8x8) without the precision cliff.
//
// Bind group: 0=arena (f32 rw), 1=params (uniform). Reads A and B
// directly from arena (B is a Param node — its f32 data is already
// in-arena from `set_param`), no f16 shadow buffer needed.
//
// REQUIREMENTS:
//   - Device feature: EXPERIMENTAL_COOPERATIVE_MATRIX
//   - M and N multiples of 32; K multiple of 8
//   - workgroup_size(32) — Apple subgroup width

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

const TILE_M: u32 = 32u;
const TILE_N: u32 = 32u;
const TILE_K: u32 = 8u;

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;

var<workgroup> acc_scratch: array<f32, 1024>;
var<workgroup> a_stage:     array<f32, 256>;     // 32 × 8

// Exact GELU — A&S 7.1.26 erf, matches rlx-cpu's scalar_gelu.
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

@compute @workgroup_size(32)
fn matmul_coop_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let bz = wid.z;
    let row_base = wid.y * TILE_M;
    let col_base = wid.x * TILE_N;

    let a_base = params.a_off + bz * params.a_batch_stride;
    let b_base = params.b_off + bz * params.b_batch_stride;
    let c_base = params.c_off + bz * params.c_batch_stride;

    // Zero acc_scratch (1024 f32 = 32 elements per thread).
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
            a_stage[idx] = arena[a_base + (row_base + r) * params.k + k_off + c];
        }
        workgroupBarrier();

        let a_0: coop_mat8x8<f32, A> = coopLoad<coop_mat8x8<f32, A>>(&a_stage[0u  ], 8u);
        let a_1: coop_mat8x8<f32, A> = coopLoad<coop_mat8x8<f32, A>>(&a_stage[64u ], 8u);
        let a_2: coop_mat8x8<f32, A> = coopLoad<coop_mat8x8<f32, A>>(&a_stage[128u], 8u);
        let a_3: coop_mat8x8<f32, A> = coopLoad<coop_mat8x8<f32, A>>(&a_stage[192u], 8u);
        let b_row = b_base + k_off * params.n + col_base;
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

    for (var s: u32 = 0u; s < 32u; s = s + 1u) {
        let idx = lid + s * 32u;
        let r = idx / 32u;
        let c = idx % 32u;
        let global_row = row_base + r;
        let global_col = col_base + c;
        // Bounds-check on M: when params.m isn't a multiple of 32, the
        // dispatcher rounds up the y-dim and the kernel computes garbage
        // for the padded rows (it reads OOB in A and accumulates garbage
        // into acc). Skipping the write keeps the padded compute harmless
        // — the output rows past `params.m` are left untouched.
        if (global_row >= params.m) { continue; }
        var v: f32 = acc_scratch[idx];
        if (params.has_bias != 0u) {
            v = v + arena[params.bias_off + global_col];
        }
        v = apply_act(v);
        arena[c_base + global_row * params.n + global_col] = v;
    }
}
