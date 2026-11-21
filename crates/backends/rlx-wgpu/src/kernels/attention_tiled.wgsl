// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// `rlx_div_rn`, `rlx_exp`, …: rlxsl's correctly-rounded helpers, appended by
// `kernel_module` (`with_numeric_helpers`).

// Tiled SDPA for many query rows (prefill, encoders, long cross-attention).
//
// `attention` (attention.wgsl) runs a thread per query row, each walking every
// key and value straight from the arena with `q` / `O` in 512-wide private
// arrays: no reuse of a K/V row between the rows that read it, and private
// arrays that size spill out of registers. At 3680 queries × 59k keys that is
// 2.5 s where Metal takes 0.1 s.
//
// Here a 64-thread workgroup owns 64 consecutive query rows of one (batch,
// head). It walks the keys in tiles of `BK`: the workgroup loads a K tile and a
// V tile into workgroup memory once, and every thread scores its row against
// the whole tile from there (a broadcast read). The online softmax is rebased
// once per tile (FlashAttention v1/v2 form):
//
//   m_new = max(m, max_j s_j);  p_j = exp(s_j − m_new);  e = exp(m − m_new)
//   l = e·l + Σ_j p_j;          O[d] = e·O[d] + Σ_j p_j·V[j][d]
//
// `HD` (head_dim == V width) is substituted per build, so `q` and `O` are
// register arrays of exactly that size; `BK · HD = 1024` keeps the two tiles
// at 8 KiB of the 16 KiB WebGPU guarantees.
//
// Same `Params`, masks, strides, GQA mapping and soft-cap as `attention`, so it
// runs from the same bind group.

const HD: u32 = {{HD}}u;
const BK: u32 = {{BK}}u;
const BQ: u32 = 64u;

struct Params {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,

    out_off: u32,
    mask_off: u32,
    mask_kind: u32,
    scale_bits: u32,
    window: u32,
    seq_q_stride: u32,
    seq_k_stride: u32,
    mask_batch_stride: u32,
    mask_head_stride: u32,
    kv_heads: u32,
    v_head_dim: u32,
    softcap_bits: u32,

    q_batch_stride: u32, q_head_stride: u32, q_seq_stride: u32, _pad_q: u32,
    k_batch_stride: u32, k_head_stride: u32, k_seq_stride: u32, _pad_k: u32,
    v_batch_stride: u32, v_head_stride: u32, v_seq_stride: u32, _pad_v: u32,
    o_batch_stride: u32, o_head_stride: u32, o_seq_stride: u32, _pad_o: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// K and V tiles, `BK` rows of `HD / 4` vec4s each (8 KiB together).
var<workgroup> k_sh: array<vec4<f32>, 256>;
var<workgroup> v_sh: array<vec4<f32>, 256>;

fn softcap(score: f32) -> f32 {
    let c = bitcast<f32>(params.softcap_bits);
    if (c > 0.0) {
        return c * tanh(clamp(score / c, -15.0, 15.0));
    }
    return score;
}

// Workgroup (x = query block, y·nwg.x… = batch·head); one thread per row.
// Workgroup (x = query block, y·nwg.x… = batch·head); one thread per row.
//
// Every per-thread loop has a constant bound (`HD4`, `BK`), so `q`, `o` and
// `s` stay in registers (naga/Metal promote constant-bounded local arrays;
// a bound read at run time sends them to thread memory), and every tile
// access is a `vec4` — what took `matmul_wide_vec4` from 122 to 346 GF/s on
// the same path. Keys past `seq_k` in the last tile are scored −inf, which
// gives them weight 0.
const HD4: u32 = HD / 4u;

@compute @workgroup_size(64)
fn attention_tiled(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let bh = wid.y + wid.z * nwg.y;
    // Uniform across the workgroup: an early return before the barriers is fine.
    if (bh >= params.batch * params.heads) { return; }
    let h = bh % params.heads;
    let b = bh / params.heads;
    let qi = wid.x * BQ + t;
    let live = qi < params.seq_q;
    let scale = bitcast<f32>(params.scale_bits);
    let q_abs = qi + select(0u, params.seq_k - params.seq_q, params.seq_k >= params.seq_q);

    let kv_denom = max(params.kv_heads, 1u);
    let kv_h = h / max(params.heads / kv_denom, 1u);
    let k_bh = params.k_off + b * params.k_batch_stride + kv_h * params.k_head_stride;
    let v_bh = params.v_off + b * params.v_batch_stride + kv_h * params.v_head_stride;
    let mask_partial = params.mask_off
        + b  * params.mask_batch_stride
        + h  * params.mask_head_stride
        + qi * params.seq_q_stride;

    var q: array<vec4<f32>, HD4>;
    var o: array<vec4<f32>, HD4>;
    if (live) {
        let q_base = params.q_off + b * params.q_batch_stride + h * params.q_head_stride
            + qi * params.q_seq_stride;
        for (var d: u32 = 0u; d < HD4; d = d + 1u) {
            let a = q_base + 4u * d;
            q[d] = vec4<f32>(arena[a], arena[a + 1u], arena[a + 2u], arena[a + 3u]);
        }
    }
    for (var d: u32 = 0u; d < HD4; d = d + 1u) { o[d] = vec4<f32>(0.0); }
    var m: f32 = -3.4e38;
    var l: f32 = 0.0;
    var s: array<f32, BK>;

    let n_tiles = (params.seq_k + BK - 1u) / BK;
    for (var tile: u32 = 0u; tile < n_tiles; tile = tile + 1u) {
        let k0 = tile * BK;
        // Cooperative tile load: BK·HD4 vec4s each of K and V over 64 threads.
        // Rows past seq_k load zeros (their scores are masked below).
        for (var e: u32 = t; e < BK * HD4; e = e + 64u) {
            let j = e / HD4;
            let d = 4u * (e % HD4);
            var kv = vec4<f32>(0.0);
            var vv = vec4<f32>(0.0);
            if (k0 + j < params.seq_k) {
                let ka = k_bh + (k0 + j) * params.k_seq_stride + d;
                let va = v_bh + (k0 + j) * params.v_seq_stride + d;
                kv = vec4<f32>(arena[ka], arena[ka + 1u], arena[ka + 2u], arena[ka + 3u]);
                vv = vec4<f32>(arena[va], arena[va + 1u], arena[va + 2u], arena[va + 3u]);
            }
            k_sh[e] = kv;
            v_sh[e] = vv;
        }
        workgroupBarrier();

        if (live) {
            var tile_max: f32 = -3.4e38;
            for (var j: u32 = 0u; j < BK; j = j + 1u) {
                // Sequential over the head dim, as the CPU and the per-row
                // kernel sum it (vec4 loads, scalar accumulation).
                var acc: f32 = 0.0;
                for (var d: u32 = 0u; d < HD4; d = d + 1u) {
                    let kv = k_sh[j * HD4 + d];
                    acc = acc + q[d].x * kv.x;
                    acc = acc + q[d].y * kv.y;
                    acc = acc + q[d].z * kv.z;
                    acc = acc + q[d].w * kv.w;
                }
                var score = softcap(acc * scale);
                let sk = k0 + j;
                if (sk >= params.seq_k) {
                    score = -3.4e38;
                } else if (params.mask_kind == 1u) {
                    if (sk > q_abs) { score = -3.4e38; }
                } else if (params.mask_kind == 2u) {
                    if (arena[mask_partial + sk * params.seq_k_stride] < 0.5) { score = -1e9; }
                } else if (params.mask_kind == 3u) {
                    if (sk > q_abs) { score = -3.4e38; }
                    else if (q_abs - sk > params.window) { score = -3.4e38; }
                } else if (params.mask_kind == 4u) {
                    score = score + arena[mask_partial + sk * params.seq_k_stride];
                }
                s[j] = score;
                tile_max = max(tile_max, score);
            }
            let m_new = max(m, tile_max);
            let e_old = rlx_exp(m - m_new);
            var p_sum: f32 = 0.0;
            for (var j: u32 = 0u; j < BK; j = j + 1u) {
                // Past seq_k: exactly 0, even while every score so far is −inf.
                let p = select(rlx_exp(s[j] - m_new), 0.0, k0 + j >= params.seq_k);
                s[j] = p;
                p_sum = p_sum + p;
            }
            l = e_old * l + p_sum;
            for (var d: u32 = 0u; d < HD4; d = d + 1u) {
                o[d] = o[d] * e_old;
            }
            for (var j: u32 = 0u; j < BK; j = j + 1u) {
                let p = s[j];
                for (var d: u32 = 0u; d < HD4; d = d + 1u) {
                    o[d] = fma(vec4<f32>(p), v_sh[j * HD4 + d], o[d]);
                }
            }
            m = m_new;
        }
        // The next tile overwrites k_sh / v_sh.
        workgroupBarrier();
    }

    if (live) {
        let o_base = params.out_off + b * params.o_batch_stride + h * params.o_head_stride
            + qi * params.o_seq_stride;
        let inv_l = rlx_div_rn(1.0, l);
        for (var d: u32 = 0u; d < HD4; d = d + 1u) {
            let v = o[d] * inv_l;
            let a = o_base + 4u * d;
            arena[a] = v.x;
            arena[a + 1u] = v.y;
            arena[a + 2u] = v.z;
            arena[a + 3u] = v.w;
        }
    }
}
