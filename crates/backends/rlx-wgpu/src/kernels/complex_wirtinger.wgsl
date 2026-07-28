// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// C64 Wirtinger surface on the f32-uniform arena (interleaved [re, im] pairs).
// Formulas mirror rlx-cpu `exec_complex_norm_sq{,_backward}_f32` /
// `exec_conjugate_c64` and CUDA `complex_wirtinger.cu`. Dispatched over the
// complex-element index `k ∈ [0, n)`. Offsets are f32-ELEMENT offsets.
//
//   ComplexNormSq:          out[k] = re² + im²           (C64 → F32)
//                           a_off=src, c_off=dst (b_off unused)
//   ComplexNormSqBackward: dz = g · z  (Wirtinger)       (C64, F32 → C64)
//                           a_off=z, b_off=g, c_off=dz
//   Conjugate:              out = (re, -im)              (C64 → C64)
//                           a_off=src, c_off=dst (b_off unused)

struct Params {
    n: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn complex_norm_sq(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let k = gid.x + gid.y * ngs.x * 64u;
    if (k >= params.n) { return; }
    let re = arena[params.a_off + 2u * k];
    let im = arena[params.a_off + 2u * k + 1u];
    arena[params.c_off + k] = re * re + im * im;
}

@compute @workgroup_size(64)
fn complex_norm_sq_backward(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let k = gid.x + gid.y * ngs.x * 64u;
    if (k >= params.n) { return; }
    let re = arena[params.a_off + 2u * k];
    let im = arena[params.a_off + 2u * k + 1u];
    let gv = arena[params.b_off + k];
    arena[params.c_off + 2u * k]      = gv * re;
    arena[params.c_off + 2u * k + 1u] = gv * im;
}

@compute @workgroup_size(64)
fn conjugate_c64(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let k = gid.x + gid.y * ngs.x * 64u;
    if (k >= params.n) { return; }
    arena[params.c_off + 2u * k]      =  arena[params.a_off + 2u * k];
    arena[params.c_off + 2u * k + 1u] = -arena[params.a_off + 2u * k + 1u];
}
