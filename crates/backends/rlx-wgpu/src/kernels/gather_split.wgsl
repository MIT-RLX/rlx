// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Embedding-style gather along the leading axis — SPLIT bindings.
//
// Identical math to gather.wgsl, but the embedding table, the index array,
// and the output each get their OWN binding. This lets the backend bind a
// separate ≤4 GiB window for the (multi-GiB) table and another for the small
// index/output region when those slots are more than one binding window
// apart in a >4 GiB arena (the single-binding gather.wgsl cannot reach an
// output that lies outside the table's window). The output is a dedicated
// buffer; the caller copies it back into the arena.
//
// Input  table shape [vocab, dim]
// Indices flat shape [n_idx]  (f32-encoded; rounded to u32)
// Output shape [n_idx, dim]

struct Params {
    n_out: u32,        // total output elements (n_idx * dim)
    n_idx: u32,        // number of indices
    dim: u32,          // inner axis size
    vocab: u32,        // input axis size (for clamping out-of-range indices)
    in_off: u32,       // f32 offset of the table within its bound window
    idx_off: u32,      // f32 offset of the indices within their bound window
    out_off: u32,      // f32 offset within the dedicated output buffer (0)
    _p0: u32,
};

@group(0) @binding(0) var<storage, read>       table:  array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;
@group(0) @binding(2) var<storage, read>       idxbuf: array<f32>;
@group(0) @binding(3) var<storage, read_write> outbuf: array<f32>;

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let o = gid.x + gid.y * ngs.x * 64u;
    if (o >= params.n_out) { return; }
    let d = o % params.dim;
    let i = o / params.dim;
    let idx_f = idxbuf[params.idx_off + i];
    var idx_u: u32 = u32(max(idx_f, 0.0));
    if (idx_u >= params.vocab) { idx_u = params.vocab - 1u; }
    outbuf[params.out_off + o] = table[params.in_off + idx_u * params.dim + d];
}
