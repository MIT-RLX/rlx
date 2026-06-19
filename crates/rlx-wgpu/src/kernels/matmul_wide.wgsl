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

// Wide-tile matmul variant: 32×64 output tile, 4×8 register block.
//
// Same 8×8 = 64-thread workgroup as `matmul.wgsl`, but each thread holds
// a 4×8 (32-float) accumulator block instead of 4×4 (16-float). Per
// K-step inner loop: 4 a-loads + 8 b-loads → 32 FMAs (2.67 FMA / register
// load, vs the small variant's 2.0). Each thread holds 32 acc + 12 load
// registers ≈ 44 f32 — within Apple M-series per-thread budget. The
// 8×8 / 64-acc variant (TILE_M=64, TILE_N=64, RM=8, RN=8) was tested
// and spilled hard, measuring 2-3× slower than this variant.
//
// Use criterion: when M ≥ 32 AND N ≥ 64, dispatch this kernel for
// ~7-13 % speedup on BERT FFN/QKV matmuls at batch ≥ 64. For smaller
// shapes the partial-tile waste in N (1 wg covers 64 cols) outweighs
// the AI win, so fall back to the 4×4-block kernel.

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
const TILE_N: u32 = 64u;
const TILE_K: u32 = 16u;
const RM: u32 = 4u;
const RN: u32 = 8u;
const WG_M: u32 = 8u;     // TILE_M / RM
const WG_N: u32 = 8u;     // TILE_N / RN

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

var<workgroup> tile_a: array<array<f32, 16>, 32>;  // [TILE_M][TILE_K]
var<workgroup> tile_b: array<array<f32, 64>, 16>;  // [TILE_K][TILE_N]

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

@compute @workgroup_size(8, 8)
fn matmul_wide(
    @builtin(global_invocation_id)   gid: vec3<u32>,
    @builtin(local_invocation_id)    lid: vec3<u32>,
    @builtin(workgroup_id)           wid: vec3<u32>,
) {
    let bz = gid.z;
    if (bz >= params.batch) { return; }

    let lr = lid.y;
    let lc = lid.x;
    let row_base = wid.y * TILE_M + lr * RM;
    let col_base = wid.x * TILE_N + lc * RN;

    let a_base = params.a_off + bz * params.a_batch_stride;
    let b_base = params.b_off + bz * params.b_batch_stride;
    let c_base = params.c_off + bz * params.c_batch_stride;

    var acc: array<array<f32, 8>, 4>;
    for (var i: u32 = 0u; i < RM; i = i + 1u) {
        for (var j: u32 = 0u; j < RN; j = j + 1u) {
            acc[i][j] = 0.0;
        }
    }

    let n_tiles = (params.k + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        // tile_a (32×16): each thread writes 4 rows × 2 cols = 8 elem.
        for (var i: u32 = 0u; i < RM; i = i + 1u) {
            let m_local = lr * RM + i;
            let global_row = wid.y * TILE_M + m_local;
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                let k_local = lc * 2u + j;
                let global_k = t * TILE_K + k_local;
                if (global_row < params.m && global_k < params.k) {
                    tile_a[m_local][k_local] = arena[a_base + global_row * params.k + global_k];
                } else {
                    tile_a[m_local][k_local] = 0.0;
                }
            }
        }
        // tile_b (16×64): each thread writes 2 rows × 8 cols = 16 elem.
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let k_local = lr * 2u + i;
            let global_k = t * TILE_K + k_local;
            for (var j: u32 = 0u; j < RN; j = j + 1u) {
                let n_local = lc * RN + j;
                let global_col = wid.x * TILE_N + n_local;
                if (global_k < params.k && global_col < params.n) {
                    tile_b[k_local][n_local] = arena[b_base + global_k * params.n + global_col];
                } else {
                    tile_b[k_local][n_local] = 0.0;
                }
            }
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
            var a_reg: array<f32, 4>;
            var b_reg: array<f32, 8>;
            for (var i: u32 = 0u; i < RM; i = i + 1u) {
                a_reg[i] = tile_a[lr * RM + i][k];
            }
            for (var j: u32 = 0u; j < RN; j = j + 1u) {
                b_reg[j] = tile_b[k][lc * RN + j];
            }
            for (var i: u32 = 0u; i < RM; i = i + 1u) {
                for (var j: u32 = 0u; j < RN; j = j + 1u) {
                    acc[i][j] = acc[i][j] + a_reg[i] * b_reg[j];
                }
            }
        }

        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < RM; i = i + 1u) {
        let global_row = row_base + i;
        if (global_row >= params.m) { continue; }
        for (var j: u32 = 0u; j < RN; j = j + 1u) {
            let global_col = col_base + j;
            if (global_col >= params.n) { continue; }
            var v = acc[i][j];
            if (params.has_bias != 0u) {
                v = v + arena[params.bias_off + global_col];
            }
            v = apply_act(v);
            arena[c_base + global_row * params.n + global_col] = v;
        }
    }
}
