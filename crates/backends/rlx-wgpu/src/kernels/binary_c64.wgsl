// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Element-wise C64 binary op, dispatched over the complex-element index
// `k ∈ [0, n)`. C64 = 2 f32 lanes `[re, im]` (8 B/elem), so each thread reads
// BOTH lanes of its operands — the reason this cannot ride the fused
// `elementwise_region` scalar-per-thread path (that model can't reach the
// partner `im` lane). Formulas mirror rlx-cpu `exec_binary_full_c64`:
//   Add: (ar+br, ai+bi)   Sub: (ar-br, ai-bi)
//   Mul: (ar*br - ai*bi, ar*bi + ai*br)
//   Div: d = br*br + bi*bi; ((ar*br + ai*bi)/d, (ai*br - ar*bi)/d)
// Max/Min/Pow are rejected at lowering (undefined for complex).
//
// Broadcast: `n_a` / `n_b` are the operands' complex-element counts. Indexing
// uses `k % n_a` / `k % n_b` (complex-element units), matching the CPU modulo
// fallback — a scalar operand (count 1) reads element 0 for every k. Offsets
// are f32-element offsets; lane j of complex element m is `off + 2*m + j`.

struct Params {
    n: u32,      // output complex-element count
    a_off: u32,  // f32-element offset of lhs
    b_off: u32,  // f32-element offset of rhs
    c_off: u32,  // f32-element offset of output
    op: u32,     // 0=add, 1=sub, 2=mul, 3=div
    n_a: u32,    // lhs complex-element count (for broadcast)
    n_b: u32,    // rhs complex-element count (for broadcast)
    _p0: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn binary_c64_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let k = gid.x + gid.y * ngs.x * 64u;
    if (k >= params.n) { return; }
    let ka = k % params.n_a;
    let kb = k % params.n_b;
    let ar = arena[params.a_off + 2u * ka];
    let ai = arena[params.a_off + 2u * ka + 1u];
    let br = arena[params.b_off + 2u * kb];
    let bi = arena[params.b_off + 2u * kb + 1u];
    var cr: f32 = 0.0;
    var ci: f32 = 0.0;
    switch (params.op) {
        case 0u: { cr = ar + br; ci = ai + bi; }
        case 1u: { cr = ar - br; ci = ai - bi; }
        case 2u: { cr = ar * br - ai * bi; ci = ar * bi + ai * br; }
        case 3u: {
            let d = br * br + bi * bi;
            cr = (ar * br + ai * bi) / d;
            ci = (ai * br - ar * bi) / d;
        }
        default: {}
    }
    arena[params.c_off + 2u * k]      = cr;
    arena[params.c_off + 2u * k + 1u] = ci;
}
