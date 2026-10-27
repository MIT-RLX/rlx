// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

enable wgpu_cooperative_matrix;

// Edge-tolerant twin of `matmul_coop_f32.wgsl` for shapes that kernel cannot
// take: K not a multiple of 8 and/or N not a multiple of 32 (e.g. attention
// with head_dim 25 — scores `[S,25]·[25,S]`, context `[S,S]·[S,25]` — and
// d_model 200 projections). Same 32×32 tile, same `simdgroup_float8x8`
// accumulation, same Metal row-major (`...T`) fragment convention.
//
// Difference: A is staged with zero padding past M/K; B is staged (zero-padded
// past K/N) only on a ragged tile, so a ragged tile neither reads past the
// operand nor multiplies neighbouring data into a valid output (0 × NaN would
// poison it, hence B is zeroed too, not just A). Full B tiles load directly.
// B may be any arena tensor (activation or Param). The aligned kernel stays
// the fast path; this one is only selected when `k % 8 != 0 || n % 32 != 0`
// (see `matmul_coop_f32_kernel_for`).

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
var<workgroup> b_stage:     array<f32, 256>;     // 8 × 32
var<workgroup> b_full_wg:   u32;

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
fn matmul_coop_f32_edge(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let bz = wid.z;
    let row_base = wid.y * TILE_M;
    let col_base = wid.x * TILE_N;

    let a_base = params.a_off + bz * params.a_batch_stride;
    let b_base = params.b_off + bz * params.b_batch_stride;
    let c_base = params.c_off + bz * params.c_batch_stride;

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
        // A tile 32×8, zero outside [M,K] (always staged: it is also the M guard).
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let idx = lid + s * 32u;
            let ar = row_base + idx / 8u;
            let ac = k_off + idx % 8u;
            var av: f32 = 0.0;
            if (ar < params.m && ac < params.k) {
                av = arena[a_base + ar * params.k + ac];
            }
            a_stage[idx] = av;
        }
        // B: a FULL 8×32 tile (inside K and N) is read straight from the arena,
        // as the aligned kernel does; only the ragged column tile / K tail goes
        // through zero-padded staging. Workgroup-uniform (wid.x, t), as coop
        // ops require.
        // Built from uniform values only, but naga's uniformity analysis does not
        // accept a coop load under a plain `if` on it; `workgroupUniformLoad`
        // is the WGSL-sanctioned way to hand it a provably uniform value.
        if (lid == 0u) {
            b_full_wg = select(0u, 1u, (k_off + TILE_K <= params.k) && (col_base + TILE_N <= params.n));
        }
        let b_full = workgroupUniformLoad(&b_full_wg) == 1u;
        if (!b_full) {
            for (var s: u32 = 0u; s < 8u; s = s + 1u) {
                let idx = lid + s * 32u;
                let br = k_off + idx / 32u;
                let bc = col_base + idx % 32u;
                var bv: f32 = 0.0;
                if (br < params.k && bc < params.n) {
                    bv = arena[b_base + br * params.n + bc];
                }
                b_stage[idx] = bv;
            }
        }
        workgroupBarrier();

        // Row-major (`T`) loads — see the convention note in matmul_coop_f32.wgsl.
        let a_0: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[0u], 8u);
        let a_1: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[64u], 8u);
        let a_2: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[128u], 8u);
        let a_3: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[192u], 8u);
        if (b_full) {
            let b_row = b_base + k_off * params.n + col_base;
            let b_0: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row + 0u], params.n);
            let b_1: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row + 8u], params.n);
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
        } else {
            let b_0: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&b_stage[0u], 32u);
            let b_1: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&b_stage[8u], 32u);
            let b_2: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&b_stage[16u], 32u);
            let b_3: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&b_stage[24u], 32u);
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
        }
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
        if (global_row >= params.m || global_col >= params.n) { continue; }
        var v: f32 = acc_scratch[idx];
        if (params.has_bias != 0u) {
            v = v + arena[params.bias_off + global_col];
        }
        v = apply_act(v);
        arena[c_base + global_row * params.n + global_col] = v;
    }
}
