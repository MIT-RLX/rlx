// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

enable wgpu_cooperative_matrix;

// Batched C[z] = A[z] · B[z] with A and B read through STRIDED views, so a
// multi-head attention matmul reads Q / Kᵀ / V straight out of the fused qkv
// projection instead of from narrow + transpose copies (the graph rewrite is
// `crate::unfuse::fold_strided_bmm`). Batch index z splits as (z / b2, z % b2)
// — e.g. (batch, head) — and element (z, r, c) of an operand lives at
//   off + (z / b2)·s0 + (z % b2)·s1 + r·sr + c·sc     (f32 elements).
// C is written through the same kind of view (`c_s0, c_s1, c_sr, c_sc`), so
// the attention output can land directly in `[B, S, H·Dh]`. Same 32×32 tile and `simdgroup_float8x8`
// K order as `matmul_coop_f32_edge.wgsl`; both operands go through zero-padded
// workgroup staging, so ragged M / K / N and any strides are safe.

struct Params {
    m: u32, k: u32, n: u32, batch: u32,
    b2: u32, c_off: u32, c_s0: u32, c_s1: u32,
    a_off: u32, a_s0: u32, a_s1: u32, a_sr: u32,
    a_sc: u32, b_off: u32, b_s0: u32, b_s1: u32,
    b_sr: u32, b_sc: u32, c_sr: u32, c_sc: u32,
};

const TILE_M: u32 = 32u;
const TILE_N: u32 = 32u;
const TILE_K: u32 = 8u;

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;

var<workgroup> acc_scratch: array<f32, 1024>;
var<workgroup> a_stage:     array<f32, 256>;     // 32 × 8
var<workgroup> b_stage:     array<f32, 256>;     // 8 × 32

@compute @workgroup_size(32)
fn matmul_coop_f32_strided(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let bz = wid.z;
    let row_base = wid.y * TILE_M;
    let col_base = wid.x * TILE_N;
    let z1 = bz / params.b2;
    let z2 = bz % params.b2;
    let a_base = params.a_off + z1 * params.a_s0 + z2 * params.a_s1;
    let b_base = params.b_off + z1 * params.b_s0 + z2 * params.b_s1;
    let c_base = params.c_off + z1 * params.c_s0 + z2 * params.c_s1;

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
            let ar = row_base + idx / 8u;
            let ac = k_off + idx % 8u;
            var av: f32 = 0.0;
            if (ar < params.m && ac < params.k) {
                av = arena[a_base + ar * params.a_sr + ac * params.a_sc];
            }
            a_stage[idx] = av;
            let br = k_off + idx / 32u;
            let bc = col_base + idx % 32u;
            var bv: f32 = 0.0;
            if (br < params.k && bc < params.n) {
                bv = arena[b_base + br * params.b_sr + bc * params.b_sc];
            }
            b_stage[idx] = bv;
        }
        workgroupBarrier();

        // Row-major (`T`) loads — see the convention note in matmul_coop_f32.wgsl.
        let a_0: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[0u], 8u);
        let a_1: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[64u], 8u);
        let a_2: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[128u], 8u);
        let a_3: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[192u], 8u);
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
        let r = row_base + idx / 32u;
        let c = col_base + idx % 32u;
        if (r < params.m && c < params.n) {
            arena[c_base + r * params.c_sr + c * params.c_sc] = acc_scratch[idx];
        }
    }
}
