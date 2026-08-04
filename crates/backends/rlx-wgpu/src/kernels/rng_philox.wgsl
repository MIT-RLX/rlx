// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// On-device Philox4x32-10 RNG, bit-matched to `rlx_ir::Philox4x32` / the shared
// `rng_philox.cu`. WGSL has no 64-bit integers, so the Philox 32x32->64 multiply
// is emulated with 16-bit halves (`umul_wide`). normal sample i reads Philox
// block i/2 lanes {0,1}|{2,3}; uniform i reads block i/4 lane i%4.
// The arena is bound at a 256B-aligned window; `out_off` is relative to it.

struct Params {
    n: u32,
    out_off: u32,
    a: f32,        // mean (normal) | low (uniform)
    b: f32,        // scale (normal) | high (uniform)
    seed_lo: u32,
    seed_hi: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// (hi, lo) = a * b, exact 64-bit product via 16-bit partial products.
fn umul_wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu; let a1 = a >> 16u;
    let b0 = b & 0xFFFFu; let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    let lo = (p00 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(hi, lo);
}

fn philox_10(blk: u32, seed_lo: u32, seed_hi: u32) -> array<u32, 4> {
    // Counter for block `blk` (blk < 2^31 for u32 len → high words zero).
    var s0 = blk; var s1 = 0u; var s2 = 0u; var s3 = 0u;
    var k0 = seed_lo; var k1 = seed_hi;
    for (var i = 0; i < 10; i = i + 1) {
        let p0 = umul_wide(s0, 0xD2561A75u);
        let p1 = umul_wide(s2, 0xCD9E8D57u);
        let n0 = p1.x ^ s1 ^ k0;
        let n1 = p1.y;
        let n2 = p0.x ^ s3 ^ k1;
        let n3 = p0.y;
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
        k0 = k0 + 0x9E3779B9u;
        k1 = k1 + 0xBB67AE85u;
    }
    return array<u32, 4>(s0, s1, s2, s3);
}

fn u32_to_unit(bits: u32) -> f32 {
    return f32(bits >> 8u) / 16777216.0; // (bits>>8) / 2^24
}

@compute @workgroup_size(64)
fn rng_normal_philox(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    var buf = philox_10(i / 2u, params.seed_lo, params.seed_hi);
    let lane0 = select(0u, 2u, (i & 1u) != 0u);
    var u1 = u32_to_unit(buf[lane0]);
    let u2 = u32_to_unit(buf[lane0 + 1u]);
    if (u1 < 1.17549435e-38) { u1 = 1.17549435e-38; } // f32::MIN_POSITIVE
    let r = sqrt(-2.0 * log(u1));
    let theta = 6.283185307179586 * u2;
    arena[params.out_off + i] = params.a + params.b * (r * cos(theta));
}

@compute @workgroup_size(64)
fn rng_uniform_philox(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    var buf = philox_10(i / 4u, params.seed_lo, params.seed_hi);
    let u = u32_to_unit(buf[i & 3u]);
    arena[params.out_off + i] = params.a + u * (params.b - params.a);
}

@compute @workgroup_size(64)
fn rng_fill_zero(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    arena[params.out_off + i] = 0.0;
}
