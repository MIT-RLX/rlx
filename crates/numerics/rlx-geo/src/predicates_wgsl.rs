// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact geometric predicates in WGSL — the foundation of a GPU Delaunay flip.
//!
//! `orient2d` fits `i32` for coordinate spans up to ~23 000 (the cross product
//! stays under 2^31). `in_circle` needs the degree-4 determinant, which overflows
//! `i32`, so it is evaluated in **emulated signed 64-bit** (two `u32` limbs):
//! a 32×32→64 multiply plus 64-bit add/negate/sign. That covers the fast-path
//! span (≤ 29 609); wider spans would need 128-bit limbs (the `rlxsl` integer
//! prelude direction).
//!
//! Both follow the standard `WgpuGpuKernel` binding convention (`arena:
//! array<f32>` @0, `params` @1); integer test data rides in the arena via
//! `bitcast`. These are validated on-device against the CPU predicates by
//! `examples/gpu_validate.rs`.

/// `orient2d` over a batch: input `[6N] I32` (a.x,a.y,b.x,b.y,c.x,c.y per case),
/// output `[N] I32` sign in {-1,0,1}.
pub const ORIENT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<storage, read>       params: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let out_off = params[0];
    let out_len = params[1];
    if (i >= out_len) { return; }
    let base = params[4] + 6u * i;
    let ax = bitcast<i32>(arena[base]);      let ay = bitcast<i32>(arena[base + 1u]);
    let bx = bitcast<i32>(arena[base + 2u]); let by = bitcast<i32>(arena[base + 3u]);
    let cx = bitcast<i32>(arena[base + 4u]); let cy = bitcast<i32>(arena[base + 5u]);
    let d = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    var s: i32 = 0;
    if (d > 0) { s = 1; } else if (d < 0) { s = -1; }
    arena[out_off + i] = bitcast<f32>(s);
}
"#;

/// `in_circle` over a batch: input `[8N] I32` (a,b,c,d points), output `[N] I32`
/// sign in {-1,0,1}; > 0 iff `d` is strictly inside the circumcircle of CCW
/// triangle a,b,c.
pub const INCIRCLE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<storage, read>       params: array<u32>;

// --- emulated signed 64-bit (vec2<u32> = (lo, hi), two's complement) ---
fn mul_u32(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xffffu; let ah = a >> 16u;
    let bl = b & 0xffffu; let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;
    let cross = lh + hl;
    let cross_carry = select(0u, 1u, cross < lh);
    let lo = ll + (cross << 16u);
    let lo_carry = select(0u, 1u, lo < ll);
    let hi = hh + (cross >> 16u) + (cross_carry << 16u) + lo_carry;
    return vec2<u32>(lo, hi);
}
fn neg_i64(x: vec2<u32>) -> vec2<u32> {
    let lo = ~x.x + 1u;
    let carry = select(0u, 1u, lo == 0u);
    return vec2<u32>(lo, ~x.y + carry);
}
fn mul_i32(a: i32, b: i32) -> vec2<u32> {
    let neg = (a < 0) != (b < 0);
    let r = mul_u32(u32(abs(a)), u32(abs(b)));
    if (neg) { return neg_i64(r); }
    return r;
}
fn add_i64(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> {
    let lo = x.x + y.x;
    let carry = select(0u, 1u, lo < x.x);
    return vec2<u32>(lo, x.y + y.y + carry);
}
fn sign_i64(x: vec2<u32>) -> i32 {
    let hi = bitcast<i32>(x.y);
    if (hi < 0) { return -1; }
    if (hi > 0) { return 1; }
    if (x.x != 0u) { return 1; }
    return 0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let out_off = params[0];
    let out_len = params[1];
    if (i >= out_len) { return; }
    let base = params[4] + 8u * i;
    let dx = bitcast<i32>(arena[base + 6u]);
    let dy = bitcast<i32>(arena[base + 7u]);
    let ax = bitcast<i32>(arena[base])       - dx; let ay = bitcast<i32>(arena[base + 1u]) - dy;
    let bx = bitcast<i32>(arena[base + 2u])  - dx; let by = bitcast<i32>(arena[base + 3u]) - dy;
    let cx = bitcast<i32>(arena[base + 4u])  - dx; let cy = bitcast<i32>(arena[base + 5u]) - dy;

    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let m1 = bx * cy - cx * by;
    let m2 = ax * cy - cx * ay;
    let m3 = ax * by - bx * ay;

    // det = a2*m1 - b2*m2 + c2*m3, exact in 64-bit.
    var det = mul_i32(a2, m1);
    det = add_i64(det, neg_i64(mul_i32(b2, m2)));
    det = add_i64(det, mul_i32(c2, m3));
    arena[out_off + i] = bitcast<f32>(sign_i64(det));
}
"#;
