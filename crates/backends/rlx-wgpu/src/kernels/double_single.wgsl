// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Op::DoubleSingle: f64 emulated as (hi, lo) f32 pairs, interleaved in the
// arena. Indexing follows `rlx_ir::double_single::DfPlan`; the arithmetic is
// the fast-math-guarded `rlxsl::dw` prelude prepended at build time. kind: 0 elementwise
// (op = `DfOp::code`), 1 matmul, 2 reduction (op 0 sum, 1 max).

struct Params {
    len: u32, rank: u32, op: u32, k: u32,
    m: u32, n: u32, a_batch: u32, b_batch: u32,
    red_len: u32, kept_rank: u32, red_rank: u32, kind: u32,
    a_off: u32, b_off: u32, out_off: u32, _pad: u32,
    d0: array<vec4<u32>, 2>,
    s0: array<vec4<u32>, 2>,
    s1: array<vec4<u32>, 2>,
    d1: array<vec4<u32>, 2>,
    s2: array<vec4<u32>, 2>,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// `_pad` is always 0; read at run time it is the zero the fast-math guard
// (`rlxsl::dw::WGSL_FAST_MATH_GUARDED`) XORs every intermediate with.
fn dw_zero() -> u32 { return params._pad; }

fn at(v: array<vec4<u32>, 2>, i: u32) -> u32 { return v[i >> 2u][i & 3u]; }

fn off_of(i0: u32, rank: u32, dims: array<vec4<u32>, 2>, strides: array<vec4<u32>, 2>) -> u32 {
    var i = i0;
    var off = 0u;
    var d = i32(rank) - 1;
    loop {
        if (d < 0) { break; }
        let n = at(dims, u32(d));
        off = off + (i % n) * at(strides, u32(d));
        i = i / n;
        d = d - 1;
    }
    return off;
}

fn ld(base: u32, i: u32) -> DwF32 { return DwF32(arena[base + 2u * i], arena[base + 2u * i + 1u]); }
fn st(i: u32, v: DwF32) {
    arena[params.out_off + 2u * i] = v.hi;
    arena[params.out_off + 2u * i + 1u] = v.lo;
}

@compute @workgroup_size(64)
fn double_single(@builtin(global_invocation_id) gid3: vec3<u32>,
                 @builtin(num_workgroups) nwg: vec3<u32>) {
    let gid = gid3.x + gid3.y * nwg.x * 64u;
    if (gid >= params.len) { return; }
    if (params.kind == 0u) {
        let x = ld(params.a_off, off_of(gid, params.rank, params.d0, params.s0));
        let y = ld(params.b_off, off_of(gid, params.rank, params.d0, params.s1));
        var r: DwF32;
        switch (params.op) {
            case 0u: { r = dw_add(x, y); }
            case 1u: { r = dw_sub(x, y); }
            case 2u: { r = dw_mul(x, y); }
            case 4u: { r = dw_sqrt(x); }
            case 5u: { r = dw_exp(x); }
            case 6u: { r = x; if (y.hi > x.hi || (y.hi == x.hi && y.lo > x.lo)) { r = y; } }
            default: { r = dw_div(x, y); }
        }
        st(gid, r);
    } else if (params.kind == 1u) {
        let j = gid % params.n;
        let row = gid / params.n;
        let i = row % params.m;
        let bi = row / params.m;
        let a0 = bi * params.a_batch + i * params.k;
        let b0 = bi * params.b_batch + j;
        var acc = DwF32(0.0, 0.0);
        for (var kk = 0u; kk < params.k; kk = kk + 1u) {
            acc = dw_add(acc, dw_mul(ld(params.a_off, a0 + kk), ld(params.b_off, b0 + kk * params.n)));
        }
        st(gid, acc);
    } else {
        // op 0: sum; op 1: max (by hi, then lo).
        let base = off_of(gid, params.kept_rank, params.d0, params.s0);
        let is_max = params.op == 1u;
        var acc = DwF32(0.0, 0.0);
        if (is_max) { acc = ld(params.a_off, base); }
        for (var r = select(0u, 1u, is_max); r < params.red_len; r = r + 1u) {
            let x = ld(params.a_off, base + off_of(r, params.red_rank, params.d1, params.s2));
            if (is_max) {
                if (x.hi > acc.hi || (x.hi == acc.hi && x.lo > acc.lo)) { acc = x; }
            } else {
                acc = dw_add(acc, x);
            }
        }
        st(gid, acc);
    }
}
