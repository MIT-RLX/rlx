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

// Tiled fp32 matmul that splits its output across 3 separate Q/K/V
// buffers. Identical compute to `matmul.wgsl` (32×32 tile, 4×4 register
// blocking, cooperative shared-memory loads); the only change is the
// epilogue, which routes each output column to one of three sinks based
// on `global_col` against `head_width = H · D`.
//
// Pattern this kernel collapses (one dispatch instead of four):
//   FusedMatMulBiasAct: input @ qkv_w (+ bias)  →  [B·S, 3·H·D]
//   Narrow Q (axis=last, start=0,        len=H·D)
//   Narrow K (axis=last, start=H·D,      len=H·D)
//   Narrow V (axis=last, start=2·H·D,    len=H·D)
//
// Detected at lowering time in `backend.rs` (`detect_split_qkv_pattern`).
// The 3 narrows' arena slots become the kernel's 3 output bases — the
// fused matmul output buffer is never written.
//
// Why split-write rather than strided reads from a single QKV buffer:
// stepping by 3·H·D between K rows defeats Apple's hardware prefetcher
// (~7-15× regression on M-series). Split-write keeps each Q/K/V buffer
// internally contiguous, so attention reads stay sequential.

struct Params {
    m: u32,
    k: u32,
    n: u32,            // = 3 · head_width  (full QKV column count)
    a_off: u32,
    b_off: u32,
    q_off: u32,        // Q output base (per-row stride = head_width)
    k_off: u32,        // K output base
    v_off: u32,        // V output base
    head_width: u32,   // = H · D
    has_bias: u32,
    bias_off: u32,     // bias is [3·H·D]; read bias[global_col]
    _p0: u32, _p1: u32, _p2: u32, _p3: u32, _p4: u32,
};

const TILE_M: u32 = 32u;
const TILE_N: u32 = 32u;
const TILE_K: u32 = 16u;
const RM: u32 = 4u;
const RN: u32 = 4u;

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

var<workgroup> tile_a: array<array<f32, 16>, 32>;
var<workgroup> tile_b: array<array<f32, 32>, 16>;

@compute @workgroup_size(8, 8)
fn matmul_qkv(
    @builtin(local_invocation_id)    lid: vec3<u32>,
    @builtin(workgroup_id)           wid: vec3<u32>,
) {
    let lr = lid.y;
    let lc = lid.x;
    let row_base = wid.y * TILE_M + lr * RM;
    let col_base = wid.x * TILE_N + lc * RN;

    var acc: array<array<f32, 4>, 4>;
    for (var i: u32 = 0u; i < RM; i = i + 1u) {
        for (var j: u32 = 0u; j < RN; j = j + 1u) {
            acc[i][j] = 0.0;
        }
    }

    let n_tiles = (params.k + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        for (var i: u32 = 0u; i < RM; i = i + 1u) {
            let m_local = lr * RM + i;
            let global_row = wid.y * TILE_M + m_local;
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                let k_local = lc * 2u + j;
                let global_k = t * TILE_K + k_local;
                if (global_row < params.m && global_k < params.k) {
                    tile_a[m_local][k_local] = arena[params.a_off + global_row * params.k + global_k];
                } else {
                    tile_a[m_local][k_local] = 0.0;
                }
            }
        }
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let k_local = lr * 2u + i;
            let global_k = t * TILE_K + k_local;
            for (var j: u32 = 0u; j < RN; j = j + 1u) {
                let n_local = lc * RN + j;
                let global_col = wid.x * TILE_N + n_local;
                if (global_k < params.k && global_col < params.n) {
                    tile_b[k_local][n_local] = arena[params.b_off + global_k * params.n + global_col];
                } else {
                    tile_b[k_local][n_local] = 0.0;
                }
            }
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
            var a_reg: array<f32, 4>;
            var b_reg: array<f32, 4>;
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

    // Split-write epilogue. For column j in [0, H·D)        → Q
    //                              [H·D, 2·H·D)  → K
    //                              [2·H·D, 3·H·D) → V
    // Each sink buffer's per-row stride is `head_width` (Q/K/V are dense
    // [B·S, H·D] tensors), independent of the matmul's per-row stride n.
    let hw = params.head_width;
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
}
