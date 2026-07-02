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

// RLX — wide-N QKV variant: coopLoad on B (N > 768).

enable f16;
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
fn matmul_qkv_coop_f16_vk_widen(
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
                coopLoad<coop_mat16x16<f16, B>>(&arena_f16[b_ptr], params.n);
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
