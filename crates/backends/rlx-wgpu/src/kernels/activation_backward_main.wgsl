// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Element-wise activation backward: dx = act'(x) · dy.
//
// The derivative dispatch `rlx_activation_backward(op, x, dy)` (op 0..17,
// relu-first ids) is @generated from the shared rlxsl manifest — the derivative
// is auto-differentiated from the forward `activation_expr` — and prepended to
// this file at build time (see build.rs). Op ids follow
// `rlx_ir::opcodes::Activation::opcode_relu_first`.

struct Params {
    n: u32,
    x_off: u32,
    dy_off: u32,
    dx_off: u32,
    op: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn activation_backward(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    let x = arena[params.x_off + i];
    let dy = arena[params.dy_off + i];
    arena[params.dx_off + i] = rlx_activation_backward(params.op, x, dy);
}
