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
    let a = arena[params.a_off + i];
    let b = arena[params.b_off + i];
    arena[params.c_off + i] = rlx_binary_apply(params.op, a, b);
}
