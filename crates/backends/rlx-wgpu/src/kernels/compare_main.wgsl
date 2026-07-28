// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Plumbing half of the standalone `compare` kernel. The per-op comparison
// (`rlx_compare_apply`, producing a 1.0/0.0 f32 mask) is @generated once from
// the shared rlxsl manifest and prepended to this file by build.rs. Bool is
// stored as an f32 lane for arena uniformity; Bool consumers (Where) treat any
// nonzero as true.

struct Params {
    n: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    op: u32,        // CmpOp opcode (see rlx_ir::opcodes)
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn compare(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    let a = arena[params.a_off + i];
    let b = arena[params.b_off + i];
    arena[params.c_off + i] = rlx_compare_apply(params.op, a, b);
}
