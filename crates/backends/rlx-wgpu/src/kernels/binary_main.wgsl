// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Plumbing half of the standalone `binary` kernel. The per-op scalar math
// (`rlx_binary_apply`) is @generated once from the shared rlxsl manifest and
// prepended to this file by build.rs — so the op set and the negative-base
// `pow` fix live in a single source shared with every other backend.

struct Params {
    n: u32,         // total elements
    a_off: u32,     // f32-element offset
    b_off: u32,
    c_off: u32,
    op: u32,        // BinaryOp opcode (see rlx_ir::opcodes)
    // Per-operand broadcast: element `i` of the output reads operand index
    // `(i / rep) % len`. `len == 0` means the operand is dense and indexed by
    // `i` directly. A scalar is `(1, 1)`; a per-channel `[1,C,1,1,1]` against
    // `[N,C,D,H,W]` is `(D*H*W, C)`.
    //
    // The divisor matters: plain `i % len` walks the channel with the trailing
    // axes and silently corrupts every per-channel bias.
    a_rep: u32,
    a_len: u32,
    b_rep: u32,
    b_len: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn binary(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    var ai = i;
    if (params.a_len != 0u) { ai = (i / params.a_rep) % params.a_len; }
    var bi = i;
    if (params.b_len != 0u) { bi = (i / params.b_rep) % params.b_len; }
    let a = arena[params.a_off + ai];
    let b = arena[params.b_off + bi];
    arena[params.c_off + i] = rlx_binary_apply(params.op, a, b);
}
