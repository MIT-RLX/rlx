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

// Rotary position embeddings. Llama-style split (first half / second
// half), per-head rotation. Input last-dim may be either `head_dim`
// (one head per row, the simple case) or `n * head_dim` (n heads
// packed per row, the QKV-direct case).
//
// Inputs (offsets in f32 elements):
//   in_off:  [..., seq, last_dim]  where last_dim % head_dim == 0
//   cos_off: [max_seq, half]
//   sin_off: [max_seq, half]
// Output:
//   out_off: same shape as input
//
// One thread per output element.

struct Params {
    n_total: u32,    // RUNTIME-scaled iteration bound (= batch * seq * last_dim)
    seq: u32,        // RUNTIME-scaled seq (loop bound, NOT stride)
    head_dim: u32,   // rotation width (per-head)
    half: u32,       // head_dim / 2
    in_off: u32,
    cos_off: u32,
    sin_off: u32,
    out_off: u32,
    last_dim: u32,   // input last dim (== head_dim for single-head; > for QKV-direct)
    // PLAN L1 — full-extent fields for offset math, set at compile time.
    batch: u32,
    seq_stride: u32, // full seq, used for per-batch buffer offset.
    _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn rope(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n_total) { return; }
    // Iteration index `i` covers active positions: (bi, si, d) for
    // bi 0..batch, si 0..seq, d 0..last_dim. Derive (bi, si, d) from i:
    let d  = i % params.last_dim;
    let q1 = i / params.last_dim;             // 0..(batch * seq)
    let bi = q1 / params.seq;                 // batch index
    let si = q1 % params.seq;                 // active position within seq
    let pos = si;
    let half = params.half;
    let d_in_head = d % params.head_dim;
    // Map to underlying full-extent buffer offset using seq_stride.
    let buf_q1 = bi * params.seq_stride + si;
    let buf_idx = buf_q1 * params.last_dim + d;
    let head_base = buf_idx - d_in_head;

    if (d_in_head < half) {
        let xf = arena[params.in_off + buf_idx];
        let xs = arena[params.in_off + head_base + d_in_head + half];
        let c  = arena[params.cos_off + pos * half + d_in_head];
        let s  = arena[params.sin_off + pos * half + d_in_head];
        arena[params.out_off + buf_idx] = xf * c - xs * s;
    } else {
        let dl = d_in_head - half;
        let xs = arena[params.in_off + buf_idx];
        let xf = arena[params.in_off + head_base + dl];
        let c  = arena[params.cos_off + pos * half + dl];
        let s  = arena[params.sin_off + pos * half + dl];
        arena[params.out_off + buf_idx] = xs * c + xf * s;
    }
}
