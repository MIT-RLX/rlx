// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Rotary position embeddings. Llama-style split (first half / second
// half), per-head rotation. Input last-dim may be either `head_dim`
// (one head per row, the simple case) or `n * head_dim` (n heads
// packed per row, the QKV-direct case).
//
// Inputs (offsets in f32 elements):
//   in_off:  [..., seq, last_dim]  where last_dim % head_dim == 0
//   cos_off: [max_seq, rot_half]
//   sin_off: [max_seq, rot_half]
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
    style: u32,      // 0 = NeoX rotate-half, 1 = GPT-J interleaved (2i, 2i+1)
    rot_half: u32,   // n_rot/2 (rotated width). dims >= n_rot are copied. == half for full rotation.
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
    let rot_half = params.rot_half;
    let d_in_head = d % params.head_dim;
    // Map to underlying full-extent buffer offset using seq_stride.
    let buf_q1 = bi * params.seq_stride + si;
    let buf_idx = buf_q1 * params.last_dim + d;
    let head_base = buf_idx - d_in_head;

    // Partial rotary (Gemma 4 global layers use n_rot < head_dim): only the
    // first n_rot = 2*rot_half dims rotate; the trailing dims pass through.
    let n_rot = rot_half * 2u;
    if (d_in_head >= n_rot) {
        arena[params.out_off + buf_idx] = arena[params.in_off + buf_idx];
        return;
    }

    if (params.style == 1u) {
        // GPT-J / llama.cpp-NORM: adjacent pairs (2i, 2i+1) rotate by angle i.
        // cos/sin row index is the freq i = d_in_head / 2 (0..rot_half). One thread
        // per output element: even lane writes the first of its pair, odd lane
        // the second, mirroring the CPU reference exactly.
        let i = d_in_head / 2u;
        let c = arena[params.cos_off + pos * rot_half + i];
        let s = arena[params.sin_off + pos * rot_half + i];
        if ((d_in_head & 1u) == 0u) {
            let x1 = arena[params.in_off + buf_idx];        // x[2i]
            let x2 = arena[params.in_off + buf_idx + 1u];   // x[2i+1]
            arena[params.out_off + buf_idx] = x1 * c - x2 * s;
        } else {
            let x2 = arena[params.in_off + buf_idx];        // x[2i+1]
            let x1 = arena[params.in_off + buf_idx - 1u];   // x[2i]
            arena[params.out_off + buf_idx] = x2 * c + x1 * s;
        }
        return;
    }

    // NeoX rotate-half: pair (i, i+rot_half). The cos/sin row stride is
    // `rot_half` (n_rot/2), matching the CPU reference: the table stores exactly
    // the rotation angles, not head_dim/2 of them. Striding by head_dim/2 under
    // PARTIAL rope reads into the next token's angles from position 1 onward.
    if (d_in_head < rot_half) {
        let xf = arena[params.in_off + buf_idx];
        let xs = arena[params.in_off + head_base + d_in_head + rot_half];
        let c  = arena[params.cos_off + pos * rot_half + d_in_head];
        let s  = arena[params.sin_off + pos * rot_half + d_in_head];
        arena[params.out_off + buf_idx] = xf * c - xs * s;
    } else {
        let dl = d_in_head - rot_half;
        let xs = arena[params.in_off + buf_idx];
        let xf = arena[params.in_off + head_base + dl];
        let c  = arena[params.cos_off + pos * rot_half + dl];
        let s  = arena[params.sin_off + pos * rot_half + dl];
        arena[params.out_off + buf_idx] = xs * c + xf * s;
    }
}
