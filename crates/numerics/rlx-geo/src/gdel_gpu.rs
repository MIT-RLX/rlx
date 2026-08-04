// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Interleaved on-device Delaunay (gDel2D-style). Unlike [`crate::construct_gpu`]
//! (batch-insert then one global flip — flip-dominated and inexact at the hull),
//! this keeps the mesh Delaunay THROUGHOUT: every phase inserts one point per
//! triangle (split) then flips the disturbed cavities locally, so the incremental-
//! Delaunay theorem holds and the total flip work is O(n) local instead of
//! O(rounds·T) global. Adjacency (`twin`) is maintained incrementally through BOTH
//! splits and flips (self-fixup, distance-1, race-free).
//!
//! Boundary: a CPU convex hull fan (cheap, exact — no super/ghost vertices, so the
//! hull is correct by construction and no deficit). Only interior points are
//! inserted on the GPU. Exact i128 in-circle / i64 orient over spans ≤ MAX.
//!
//! CORRECT + EXACT (validated 1k/60k/200k; 1M has a ~5-triangle on-edge deficit — the
//! `ins_split` strictly-interior guard skips points exactly on an edge, unhandled).
//! The interleaving does exactly what the theory predicts — the flip stays
//! cavity-local (~172/196/240 flip rounds at 60k/200k/1M, each touching few triangles,
//! vs the batch path's ~54 FULL-mesh rounds). Three bugs found + fixed while building:
//! (1) `write_ccw` in the flip (hardcoded winding oscillated); (2) fan-wedge search
//! overran to a garbage triangle; (3) THE KEY ONE — flips *relocate* uninserted points,
//! so association must be repaired after flips too (`flip_reassoc`), not just splits.
//!
//! BUT MEASURED SLOWER than even the batch path (M4 Pro: 24.7/44.4/67 ms at
//! 1k/60k/200k vs batch 15.7/25.7 ms; ~13–29× the parallel CPU). Root cause is NOT
//! the algorithm — it's GPU DISPATCH/COORDINATION overhead: ~1000+ tiny compute
//! dispatches (5 passes × ~200 flip rounds + insertion), per-round host syncs, and a
//! full-n `flip_reassoc` scan each round. Single-submit done-latching (which fixed the
//! standalone flip) made it WORSE here — over-provisioned rounds still *dispatch* n
//! threads. Reaching gDel2D's published ~40 ms needs the paper's deep pass-fusion +
//! affected-set point-location (a major engineering effort), and per the utilisation
//! ledger STILL lands ~2–3× behind a 20-core CPU. This empirically re-confirms, now
//! from the "better algorithm" itself, that GPU 2D Delaunay is coordination-bound.

use wgpu::util::DeviceExt;

const NONE: u32 = 0xffff_ffff;
const INSERTED: u32 = 0xffff_fffe;

const GDEL_WGSL: &str = r#"
const NONE: u32 = 0xffffffffu;
const INSERTED: u32 = 0xfffffffeu;
const XSTRIDE: u32 = 65536u;

@group(0) @binding(0)  var<storage, read>       pts:     array<i32>;
@group(0) @binding(1)  var<storage, read_write> tris:    array<u32>;
@group(0) @binding(2)  var<storage, read_write> twin:    array<u32>;      // per half-edge
@group(0) @binding(3)  var<storage, read_write> assoc:   array<u32>;      // point -> triangle | INSERTED
@group(0) @binding(4)  var<storage, read_write> claim:   array<atomic<u32>>; // nominee (insert) / owner (flip)
@group(0) @binding(5)  var<storage, read_write> cand_e:  array<u32>;
@group(0) @binding(6)  var<storage, read_write> cand_t1: array<u32>;
@group(0) @binding(7)  var<storage, read_write> cand_ok: array<u32>;
@group(0) @binding(8)  var<storage, read_write> aux:     array<u32>;      // split-base / flip-partner
@group(0) @binding(9)  var<storage, read_write> changed: array<u32>;      // this-round split/flip marker
@group(0) @binding(10) var<storage, read_write> dirty:   array<atomic<u32>>; // active set
@group(0) @binding(11) var<storage, read_write> counter: array<atomic<u32>>; // [tri_count, flips, on_edge]
@group(0) @binding(12) var<storage, read>       dims:    array<u32>;      // [n, max_t, round_count, _, sos]

fn px(v: u32) -> i32 { return pts[v * 2u]; }
fn py(v: u32) -> i32 { return pts[v * 2u + 1u]; }
fn tv(t: u32, k: u32) -> u32 { return tris[t * 3u + k]; }
fn gidx(g: vec3<u32>) -> u32 { return g.y * XSTRIDE + g.x; }

// ---- exact emulated integer arithmetic (i64 inner, i128 determinant) ----
fn addc(a: u32, b: u32, cin: u32) -> vec2<u32> {
    let s1 = a + b;    let c1 = select(0u, 1u, s1 < a);
    let s2 = s1 + cin; let c2 = select(0u, 1u, s2 < s1);
    return vec2<u32>(s2, c1 + c2);
}
fn mul_u32(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xffffu; let ah = a >> 16u;
    let bl = b & 0xffffu; let bh = b >> 16u;
    let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;
    let cross = lh + hl; let cc = select(0u, 1u, cross < lh);
    let lo = ll + (cross << 16u); let lc = select(0u, 1u, lo < ll);
    return vec2<u32>(lo, hh + (cross >> 16u) + (cc << 16u) + lc);
}
fn neg_i64(x: vec2<u32>) -> vec2<u32> {
    let lo = ~x.x + 1u; return vec2<u32>(lo, ~x.y + select(0u, 1u, lo == 0u));
}
fn mul_i32(a: i32, b: i32) -> vec2<u32> {
    let r = mul_u32(u32(abs(a)), u32(abs(b)));
    if ((a < 0) != (b < 0)) { return neg_i64(r); } return r;
}
fn i32_to_i64(x: i32) -> vec2<u32> { return vec2<u32>(bitcast<u32>(x), select(0u, 0xffffffffu, x < 0)); }
fn add_i64(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> {
    let lo = x.x + y.x; return vec2<u32>(lo, x.y + y.y + select(0u, 1u, lo < x.x));
}
fn sign_i64(x: vec2<u32>) -> i32 {
    let hi = bitcast<i32>(x.y);
    if (hi < 0) { return -1; } if (hi > 0) { return 1; }
    if (x.x != 0u) { return 1; } return 0;
}
fn is_neg_i64(x: vec2<u32>) -> bool { return bitcast<i32>(x.y) < 0; }
fn abs_i64(x: vec2<u32>) -> vec2<u32> { if (is_neg_i64(x)) { return neg_i64(x); } return x; }
fn mul_u64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let t0 = mul_u32(a.x, b.x); let t1 = mul_u32(a.x, b.y);
    let t2 = mul_u32(a.y, b.x); let t3 = mul_u32(a.y, b.y);
    let r0 = t0.x;
    let s1a = addc(t0.y, t1.x, 0u); let s1b = addc(s1a.x, t2.x, 0u);
    let r1 = s1b.x; let c1 = s1a.y + s1b.y;
    let s2a = addc(t3.x, t1.y, 0u); let s2b = addc(s2a.x, t2.y, c1);
    let r2 = s2b.x; let c2 = s2a.y + s2b.y;
    return vec4<u32>(r0, r1, r2, t3.y + c2);
}
fn neg_i128(x: vec4<u32>) -> vec4<u32> {
    let a0 = addc(~x.x, 1u, 0u); let a1 = addc(~x.y, 0u, a0.y); let a2 = addc(~x.z, 0u, a1.y);
    return vec4<u32>(a0.x, a1.x, a2.x, ~x.w + a2.y);
}
fn mul_i64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let r = mul_u64(abs_i64(a), abs_i64(b));
    if (is_neg_i64(a) != is_neg_i64(b)) { return neg_i128(r); } return r;
}
fn add_i128(x: vec4<u32>, y: vec4<u32>) -> vec4<u32> {
    let a0 = addc(x.x, y.x, 0u); let a1 = addc(x.y, y.y, a0.y); let a2 = addc(x.z, y.z, a1.y);
    return vec4<u32>(a0.x, a1.x, a2.x, x.w + y.w + a2.y);
}
fn sign_i128(x: vec4<u32>) -> i32 {
    if (bitcast<i32>(x.w) < 0) { return -1; }
    if ((x.x | x.y | x.z | x.w) != 0u) { return 1; } return 0;
}
fn orient(ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32) -> i32 {
    return sign_i64(add_i64(mul_i32(bx - ax, cy - ay), neg_i64(mul_i32(by - ay, cx - ax))));
}
fn orient3(a: u32, b: u32, c: u32) -> i32 { return orient(px(a), py(a), px(b), py(b), px(c), py(c)); }
fn write_ccw(t: u32, x: u32, y: u32, z: u32) {
    if (orient3(x, y, z) < 0) {
        tris[t * 3u] = x; tris[t * 3u + 1u] = z; tris[t * 3u + 2u] = y;
    } else {
        tris[t * 3u] = x; tris[t * 3u + 1u] = y; tris[t * 3u + 2u] = z;
    }
}
fn in_circle(va: u32, vb: u32, vc: u32, vd: u32) -> i32 {
    let dx = px(vd); let dy = py(vd);
    let ax = px(va) - dx; let ay = py(va) - dy;
    let bx = px(vb) - dx; let by = py(vb) - dy;
    let cx = px(vc) - dx; let cy = py(vc) - dy;
    let a2 = add_i64(mul_i32(ax, ax), mul_i32(ay, ay));
    let b2 = add_i64(mul_i32(bx, bx), mul_i32(by, by));
    let c2 = add_i64(mul_i32(cx, cx), mul_i32(cy, cy));
    let m_bc = add_i64(mul_i32(bx, cy), neg_i64(mul_i32(cx, by)));
    let m_ac = add_i64(mul_i32(ax, cy), neg_i64(mul_i32(cx, ay)));
    let m_ab = add_i64(mul_i32(ax, by), neg_i64(mul_i32(bx, ay)));
    var det = mul_i64(a2, m_bc);
    det = add_i128(det, neg_i128(mul_i64(b2, m_ac)));
    det = add_i128(det, mul_i64(c2, m_ab));
    return sign_i128(det);
}

// local edge e of t -> (u, w, opposite apex p)
fn edge_of(t: u32, e: u32) -> vec3<u32> {
    let v0 = tv(t, 0u); let v1 = tv(t, 1u); let v2 = tv(t, 2u);
    if (e == 0u) { return vec3<u32>(v0, v1, v2); }
    if (e == 1u) { return vec3<u32>(v1, v2, v0); }
    return vec3<u32>(v2, v0, v1);
}
fn find_edge(t: u32, a: u32, b: u32) -> u32 {
    for (var e = 0u; e < 3u; e = e + 1u) {
        let x = tv(t, e); let y = tv(t, (e + 1u) % 3u);
        if ((x == a && y == b) || (x == b && y == a)) { return e; }
    }
    return 0u;
}
fn has_edge(t: u32, a: u32, b: u32) -> bool {
    let ha = (tv(t, 0u) == a) || (tv(t, 1u) == a) || (tv(t, 2u) == a);
    let hb = (tv(t, 0u) == b) || (tv(t, 1u) == b) || (tv(t, 2u) == b);
    return ha && hb;
}
// child of split-parent `par` (children: par, aux[par], aux[par]+1) holding edge (a,b)
fn child_with_edge(par: u32, a: u32, b: u32) -> u32 {
    if (has_edge(par, a, b)) { return par; }
    let base = aux[par];
    if (has_edge(base, a, b)) { return base; }
    return base + 1u;
}

// =================== INSERTION ===================
@compute @workgroup_size(64)
fn ins_clear(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = gidx(g);
    if (t >= dims[2]) { return; }         // round-start count
    atomicStore(&claim[t], NONE);
    changed[t] = 0u;
}
@compute @workgroup_size(64)
fn ins_pick(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = gidx(g);
    if (i >= dims[0]) { return; }
    let a = assoc[i];
    if (a == INSERTED) { return; }
    atomicMin(&claim[a], i);
}
@compute @workgroup_size(64)
fn ins_split(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = gidx(g);
    if (t >= dims[2]) { return; }
    let p = atomicLoad(&claim[t]);
    if (p == NONE) { return; }
    let a = tris[t * 3u]; let b = tris[t * 3u + 1u]; let c = tris[t * 3u + 2u];
    if (orient3(a, b, p) == 0 || orient3(b, c, p) == 0 || orient3(c, a, p) == 0) {
        atomicAdd(&counter[2], 1u); return;   // on-edge — skip (≈0 general position)
    }
    let base = atomicAdd(&counter[0], 2u);
    let n1 = base; let n2 = base + 1u;
    // children (all CCW: parent CCW + p interior)
    tris[t * 3u] = a;  tris[t * 3u + 1u] = b;  tris[t * 3u + 2u] = p;   // t  = (a,b,p)
    tris[n1 * 3u] = b; tris[n1 * 3u + 1u] = c; tris[n1 * 3u + 2u] = p;  // n1 = (b,c,p)
    tris[n2 * 3u] = c; tris[n2 * 3u + 1u] = a; tris[n2 * 3u + 2u] = p;  // n2 = (c,a,p)
    let T_ab = twin[t * 3u + 0u]; let T_bc = twin[t * 3u + 1u]; let T_ca = twin[t * 3u + 2u];
    twin[t * 3u + 0u] = T_ab; twin[t * 3u + 1u] = n1;   twin[t * 3u + 2u] = n2;
    twin[n1 * 3u + 0u] = T_bc; twin[n1 * 3u + 1u] = n2; twin[n1 * 3u + 2u] = t;
    twin[n2 * 3u + 0u] = T_ca; twin[n2 * 3u + 1u] = t;  twin[n2 * 3u + 2u] = n1;
    aux[t] = base; changed[t] = 1u;       // parent split-marker (t reused as child (a,b,p))
    changed[n1] = 0u; changed[n2] = 0u;
    assoc[p] = INSERTED;
    // seed the active set: new triangles + old external neighbours
    atomicStore(&dirty[t], 1u); atomicStore(&dirty[n1], 1u); atomicStore(&dirty[n2], 1u);
    if (T_ab != NONE) { atomicStore(&dirty[T_ab], 1u); }
    if (T_bc != NONE) { atomicStore(&dirty[T_bc], 1u); }
    if (T_ca != NONE) { atomicStore(&dirty[T_ca], 1u); }
}
@compute @workgroup_size(64)
fn ins_fixup(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = gidx(g);
    if (t >= atomicLoad(&counter[0])) { return; }   // ALL live tris (incl. new children)
    for (var e = 0u; e < 3u; e = e + 1u) {
        let n = twin[t * 3u + e];
        if (n == NONE) { continue; }
        if (changed[n] != 1u) { continue; }          // neighbour didn't split
        let a = tv(t, e); let b = tv(t, (e + 1u) % 3u);
        if (has_edge(n, a, b)) { continue; }          // edge still on n
        twin[t * 3u + e] = child_with_edge(n, a, b);
    }
}
@compute @workgroup_size(64)
fn ins_reassoc(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = gidx(g);
    if (i >= dims[0]) { return; }
    if (assoc[i] == INSERTED) { return; }
    let t = assoc[i];
    if (changed[t] != 1u) { return; }                 // t didn't split
    let a = tris[t * 3u]; let b = tris[t * 3u + 1u]; let p = tris[t * 3u + 2u];
    let base = aux[t]; let c = tris[base * 3u + 1u];   // n1 = (b,c,p) -> c
    let sa = orient3(p, a, i); let sb = orient3(p, b, i); let sc = orient3(p, c, i);
    if (sa >= 0 && sb <= 0) { assoc[i] = t; }
    else if (sb >= 0 && sc <= 0) { assoc[i] = base; }
    else { assoc[i] = base + 1u; }
}

// =================== FLIP (cavity-local, active-set) ===================
@compute @workgroup_size(64)
fn flip_clear(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = gidx(g);
    if (t >= atomicLoad(&counter[0])) { return; }
    atomicStore(&claim[t], NONE); cand_ok[t] = 0u; changed[t] = 0u;
    if (t == 0u) { atomicStore(&counter[1], 0u); }    // flips this round
}
@compute @workgroup_size(64)
fn flip_mark(@builtin(global_invocation_id) g: vec3<u32>) {
    let t0 = gidx(g);
    if (t0 >= atomicLoad(&counter[0])) { return; }
    if (atomicLoad(&dirty[t0]) == 0u) { return; }
    let v0 = tv(t0, 0u); let v1 = tv(t0, 1u); let v2 = tv(t0, 2u);
    for (var e = 0u; e < 3u; e = e + 1u) {
        let t1 = twin[t0 * 3u + e];
        if (t1 == NONE || t0 >= t1) { continue; }
        let uw = edge_of(t0, e);
        let u = uw.x; let w = uw.y; let p = uw.z;
        let b0 = tv(t1, 0u); let b1 = tv(t1, 1u); let b2 = tv(t1, 2u);
        var q: u32;
        if (b0 != u && b0 != w) { q = b0; } else if (b1 != u && b1 != w) { q = b1; } else { q = b2; }
        let s1 = orient3(p, q, u); let s2 = orient3(p, q, w);
        if (s1 != 0 && s2 != 0 && (s1 < 0) != (s2 < 0)) {           // convex quad
            if (in_circle(v0, v1, v2, q) > 0) {                     // illegal
                cand_e[t0] = e; cand_t1[t0] = t1; cand_ok[t0] = 1u;
                let id = t0 * 3u + e;
                atomicMin(&claim[t0], id); atomicMin(&claim[t1], id);
                return;                                             // stays dirty
            }
        }
    }
    atomicStore(&dirty[t0], 0u);                                    // locally Delaunay → clean
}
@compute @workgroup_size(64)
fn flip_apply(@builtin(global_invocation_id) g: vec3<u32>) {
    let t0 = gidx(g);
    if (t0 >= atomicLoad(&counter[0])) { return; }
    if (cand_ok[t0] != 1u) { return; }
    let e = cand_e[t0]; let t1 = cand_t1[t0]; let id = t0 * 3u + e;
    if (atomicLoad(&claim[t0]) != id) { return; }
    if (atomicLoad(&claim[t1]) != id) { return; }
    let uw = edge_of(t0, e);
    let u = uw.x; let w = uw.y; let p = uw.z;
    let b0 = tv(t1, 0u); let b1 = tv(t1, 1u); let b2 = tv(t1, 2u);
    var q: u32;
    if (b0 != u && b0 != w) { q = b0; } else if (b1 != u && b1 != w) { q = b1; } else { q = b2; }
    let n_pu = twin[t0 * 3u + find_edge(t0, p, u)];
    let n_wp = twin[t0 * 3u + find_edge(t0, w, p)];
    let n_uq = twin[t1 * 3u + find_edge(t1, u, q)];
    let n_qw = twin[t1 * 3u + find_edge(t1, w, q)];
    write_ccw(t0, u, p, q);   // t0' = (u,p,q), CCW-normalised
    write_ccw(t1, w, p, q);   // t1' = (w,p,q)
    twin[t0 * 3u + find_edge(t0, u, p)] = n_pu;
    twin[t0 * 3u + find_edge(t0, q, u)] = n_uq;
    twin[t0 * 3u + find_edge(t0, p, q)] = t1;
    twin[t1 * 3u + find_edge(t1, w, p)] = n_wp;
    twin[t1 * 3u + find_edge(t1, q, w)] = n_qw;
    twin[t1 * 3u + find_edge(t1, p, q)] = t0;
    aux[t0] = t1; aux[t1] = t0; changed[t0] = 1u; changed[t1] = 1u;
    atomicAdd(&counter[1], 1u);
    atomicStore(&dirty[t0], 1u); atomicStore(&dirty[t1], 1u);
    if (n_pu != NONE) { atomicStore(&dirty[n_pu], 1u); }
    if (n_wp != NONE) { atomicStore(&dirty[n_wp], 1u); }
    if (n_uq != NONE) { atomicStore(&dirty[n_uq], 1u); }
    if (n_qw != NONE) { atomicStore(&dirty[n_qw], 1u); }
}
// A flip relocates points: a still-uninserted point in a flipped triangle must move
// to whichever of the two new triangles now contains it (they share the new diagonal).
@compute @workgroup_size(64)
fn flip_reassoc(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = gidx(g);
    if (i >= dims[0]) { return; }
    let t = assoc[i];
    if (t == INSERTED) { return; }
    if (changed[t] != 1u) { return; }        // t did not flip this round
    let par = aux[t];                         // flip partner (shares the new diagonal)
    let a0 = tv(t, 0u); let a1 = tv(t, 1u); let a2 = tv(t, 2u);
    let b0 = tv(par, 0u); let b1 = tv(par, 1u); let b2 = tv(par, 2u);
    // u = t's vertex not in par; (p,q) = the shared diagonal.
    var u: u32; var p: u32; var q: u32;
    if (a0 != b0 && a0 != b1 && a0 != b2) { u = a0; p = a1; q = a2; }
    else if (a1 != b0 && a1 != b1 && a1 != b2) { u = a1; p = a2; q = a0; }
    else { u = a2; p = a0; q = a1; }
    let si = orient3(p, q, i); let su = orient3(p, q, u);
    // i stays in t iff on u's side of the diagonal (inclusive on the diagonal).
    if (!((su > 0 && si >= 0) || (su < 0 && si <= 0))) { assoc[i] = par; }
}

@compute @workgroup_size(64)
fn flip_fixup(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = gidx(g);
    if (t >= atomicLoad(&counter[0])) { return; }
    for (var e = 0u; e < 3u; e = e + 1u) {
        let n = twin[t * 3u + e];
        if (n == NONE) { continue; }
        if (changed[n] != 1u) { continue; }
        let a = tv(t, e); let b = tv(t, (e + 1u) % 3u);
        if (has_edge(n, a, b)) { twin[t * 3u + e] = n; } else { twin[t * 3u + e] = aux[n]; }
    }
}
"#;

/// CPU convex hull (monotone chain) of `pts`, returned CCW.
fn convex_hull(pts: &[[i32; 2]]) -> Vec<u32> {
    let n = pts.len();
    let mut ord: Vec<u32> = (0..n as u32).collect();
    ord.sort_by_key(|&i| (pts[i as usize][0], pts[i as usize][1]));
    ord.dedup_by_key(|&mut i| pts[i as usize]);
    if ord.len() < 3 {
        return Vec::new();
    }
    let cross = |o: u32, a: u32, b: u32| -> i128 {
        let (o, a, b) = (pts[o as usize], pts[a as usize], pts[b as usize]);
        (a[0] as i128 - o[0] as i128) * (b[1] as i128 - o[1] as i128)
            - (a[1] as i128 - o[1] as i128) * (b[0] as i128 - o[0] as i128)
    };
    let mut lo: Vec<u32> = Vec::new();
    for &p in &ord {
        while lo.len() >= 2 && cross(lo[lo.len() - 2], lo[lo.len() - 1], p) <= 0 {
            lo.pop();
        }
        lo.push(p);
    }
    let mut up: Vec<u32> = Vec::new();
    for &p in ord.iter().rev() {
        while up.len() >= 2 && cross(up[up.len() - 2], up[up.len() - 1], p) <= 0 {
            up.pop();
        }
        up.push(p);
    }
    lo.pop();
    up.pop();
    lo.extend(up); // CCW hull
    lo
}

/// Full on-device Delaunay via interleaved insertion + cavity flipping. Returns
/// `None` if the hull is degenerate (caller falls back).
pub fn delaunay_gpu_gdel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    points: &[[i32; 2]],
) -> Option<Vec<[u32; 3]>> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return None;
    }
    let h = hull.len();
    let dbg = std::env::var_os("GEO_GDEL_DEBUG").is_some();

    // Fan triangulation of the hull (apex hull[0]); h-2 triangles.
    let mut on_hull = vec![false; n];
    for &v in &hull {
        on_hull[v as usize] = true;
    }
    let max_t = 2 * n + 8;
    let mut tris0 = vec![0u32; 3 * max_t];
    let mut twin0 = vec![NONE; 3 * max_t];
    let nfan = h - 2;
    for j in 0..nfan {
        tris0[3 * j] = hull[0];
        tris0[3 * j + 1] = hull[j + 1];
        tris0[3 * j + 2] = hull[j + 2];
        // e2 (h_{j+2}, h0) shared with fan j+1's e0; e0 with fan j-1's e2
        if j + 1 < nfan {
            twin0[3 * j + 2] = (j + 1) as u32;
            twin0[3 * (j + 1)] = j as u32;
        }
    }
    // Initial association: each interior point -> its fan wedge (binary search on the
    // sign of orient(hull[0], hull[mid], p)).
    let v = |i: u32| points[i as usize];
    let orient_i = |a: [i32; 2], b: [i32; 2], c: [i32; 2]| -> i64 {
        (b[0] as i64 - a[0] as i64) * (c[1] as i64 - a[1] as i64)
            - (b[1] as i64 - a[1] as i64) * (c[0] as i64 - a[0] as i64)
    };
    let mut assoc0 = vec![INSERTED; n];
    let h0 = v(hull[0]);
    for i in 0..n as u32 {
        if on_hull[i as usize] {
            continue;
        }
        let p = v(i);
        // largest k in [1, h-2] with p left-of/on ray h0->hull[k]; fan triangle
        // (h0, hull[k], hull[k+1]) = index k-1. Interior points always fall in [1, h-2]
        // (they're left of ray h0->hull[1] and right of ray h0->hull[h-1]).
        let (mut lo, mut hi) = (1usize, h - 2);
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if orient_i(h0, v(hull[mid]), p) >= 0 {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        // fan triangle (h0, hull[lo], hull[lo+1]) = fan index lo-1
        assoc0[i as usize] = (lo - 1) as u32;
    }

    let pt_flat: Vec<i32> = points.iter().flat_map(|p| [p[0], p[1]]).collect();
    let stor = |data: &[u8], extra: wgpu::BufferUsages| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | extra,
        })
    };
    let scratch = |elems: usize| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let cs = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let cd = wgpu::BufferUsages::COPY_DST;
    let pts_buf = stor(bytemuck::cast_slice(&pt_flat), cd);
    let tris_buf = stor(bytemuck::cast_slice(&tris0), cs);
    let twin_buf = stor(bytemuck::cast_slice(&twin0), cs);
    let assoc_buf = stor(bytemuck::cast_slice(&assoc0), cd);
    let claim_buf = scratch(max_t);
    let cand_e_buf = scratch(max_t);
    let cand_t1_buf = scratch(max_t);
    let cand_ok_buf = scratch(max_t);
    let aux_buf = scratch(max_t);
    let changed_buf = scratch(max_t);
    // Fan triangles start DIRTY: the fan is not Delaunay, so we flip it to the hull's
    // Delaunay triangulation BEFORE inserting (the incremental theorem needs a Delaunay
    // mesh at each insertion). Interior slots start clean.
    let mut dirty0 = vec![0u32; max_t];
    for d in dirty0.iter_mut().take(nfan) {
        *d = 1;
    }
    let dirty_buf = stor(bytemuck::cast_slice(&dirty0), cd);
    let counter_buf = stor(bytemuck::cast_slice(&[nfan as u32, 0u32, 0u32, 0u32]), cs); // [tri_count, flips, on_edge, done]
    let dims_buf = stor(
        bytemuck::cast_slice(&[n as u32, max_t as u32, nfan as u32, 0u32, 0u32]),
        cd,
    );

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gdel"),
        source: wgpu::ShaderSource::Wgsl(GDEL_WGSL.into()),
    });
    let ent = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            ent(0, true),
            ent(1, false),
            ent(2, false),
            ent(3, false),
            ent(4, false),
            ent(5, false),
            ent(6, false),
            ent(7, false),
            ent(8, false),
            ent(9, false),
            ent(10, false),
            ent(11, false),
            ent(12, true),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipe = |ep: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(ep),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let (p_iclear, p_ipick, p_isplit, p_ifixup, p_ireassoc) = (
        pipe("ins_clear"),
        pipe("ins_pick"),
        pipe("ins_split"),
        pipe("ins_fixup"),
        pipe("ins_reassoc"),
    );
    let (p_fclear, p_fmark, p_fapply, p_ffixup, p_freassoc) = (
        pipe("flip_clear"),
        pipe("flip_mark"),
        pipe("flip_apply"),
        pipe("flip_fixup"),
        pipe("flip_reassoc"),
    );
    let bufs = [
        &pts_buf,
        &tris_buf,
        &twin_buf,
        &assoc_buf,
        &claim_buf,
        &cand_e_buf,
        &cand_t1_buf,
        &cand_ok_buf,
        &aux_buf,
        &changed_buf,
        &dirty_buf,
        &counter_buf,
        &dims_buf,
    ];
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });

    let gy = |threads: u32| threads.div_ceil(65536).max(1);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let read2 = || -> (u32, u32) {
        let s = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        s.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let d = bytemuck::cast_slice::<u8, u32>(&s.get_mapped_range().unwrap()).to_vec();
        readback.unmap();
        (d[0], d[1])
    };
    let disp = |enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, threads: u32| {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cp.set_pipeline(p);
        cp.set_bind_group(0, &bind, &[]);
        cp.dispatch_workgroups(1024, gy(threads), 1);
    };

    let noflip = std::env::var_os("GEO_GDEL_NOFLIP").is_some();
    let mut total_flip_rounds = 0usize;
    // Flip the currently-dirty triangles to local Delaunay convergence.
    let mut flip_conv = |count: u32, total: &mut usize| {
        if noflip {
            return;
        }
        let mut fr = 0usize;
        loop {
            // SINGLE-SUBMIT chunk: arm + many flip rounds; the on-GPU `done` latch makes
            // post-convergence rounds (and their reassoc) early-out cheaply, so a whole
            // cavity flip is ONE sync instead of one-per-round (the latency sink).
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for _ in 0..4 {
                disp(&mut enc, &p_fclear, count);
                disp(&mut enc, &p_fmark, count);
                disp(&mut enc, &p_fapply, count);
                disp(&mut enc, &p_ffixup, count);
                disp(&mut enc, &p_freassoc, n as u32);
                *total += 1;
                fr += 1;
            }
            enc.copy_buffer_to_buffer(&counter_buf, 0, &readback, 0, 8);
            queue.submit(Some(enc.finish()));
            let (_, flips) = read2();
            if flips == 0 || fr > 8 * count as usize + 256 {
                break;
            }
        }
    };

    let mut tri_count = nfan as u32;
    let mut insert_rounds = 0usize;
    let cap = 4 * n + 64;
    // Fan → Delaunay of the hull vertices, so the mesh is Delaunay before any insert.
    flip_conv(tri_count, &mut total_flip_rounds);
    let fanonly = std::env::var_os("GEO_GDEL_FANONLY").is_some();
    for _ins in 0..cap {
        if fanonly {
            break;
        }
        insert_rounds += 1;
        // publish round-start count for the split/clear guards
        queue.write_buffer(&dims_buf, 8, bytemuck::cast_slice(&[tri_count]));
        // --- insertion round ---
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        disp(&mut enc, &p_iclear, tri_count);
        disp(&mut enc, &p_ipick, n as u32);
        disp(&mut enc, &p_isplit, tri_count);
        disp(&mut enc, &p_ifixup, 2 * tri_count + 4); // covers new children
        disp(&mut enc, &p_ireassoc, n as u32);
        enc.copy_buffer_to_buffer(&counter_buf, 0, &readback, 0, 8);
        queue.submit(Some(enc.finish()));
        let (new_count, _) = read2();
        // --- cavity flip to local convergence ---
        flip_conv(new_count, &mut total_flip_rounds);
        if new_count == tri_count {
            break; // no split → all inserted
        }
        tri_count = new_count;
    }

    if dbg {
        eprintln!(
            "[gdel] hull={h} insert_rounds={insert_rounds} tri_count={tri_count} flip_rounds={total_flip_rounds}"
        );
    }

    // download
    let out_rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (tri_count as u64) * 12,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&tris_buf, 0, &out_rb, 0, (tri_count as u64) * 12);
    queue.submit(Some(enc.finish()));
    let s = out_rb.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    s.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let flat: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&s.get_mapped_range().unwrap()).to_vec();
    out_rb.unmap();

    if std::env::var_os("GEO_GDEL_CHECKTWIN").is_some() {
        // download twin, verify each half-edge's twin is reciprocal + shares the edge
        let tw_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (tri_count as u64) * 12,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&twin_buf, 0, &tw_rb, 0, (tri_count as u64) * 12);
        queue.submit(Some(enc.finish()));
        let ts = tw_rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        ts.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let tw: Vec<u32> =
            bytemuck::cast_slice::<u8, u32>(&ts.get_mapped_range().unwrap()).to_vec();
        tw_rb.unmap();
        let tris: Vec<[u32; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let mut bad = 0usize;
        for t in 0..tri_count as usize {
            for e in 0..3 {
                let n = tw[t * 3 + e];
                if n == NONE {
                    continue;
                }
                let a = tris[t][e];
                let b = tris[t][(e + 1) % 3];
                let ok = (0..3).any(|k| {
                    let x = tris[n as usize][k];
                    let y = tris[n as usize][(k + 1) % 3];
                    ((x == a && y == b) || (x == b && y == a)) && tw[n as usize * 3 + k] == t as u32
                });
                if !ok {
                    bad += 1;
                }
            }
        }
        let cw = (0..tri_count as usize)
            .filter(|&t| {
                let (a, b, c) = (
                    tris[t][0] as usize,
                    tris[t][1] as usize,
                    tris[t][2] as usize,
                );
                let o = (pt_flat[2 * b] as i64 - pt_flat[2 * a] as i64)
                    * (pt_flat[2 * c + 1] as i64 - pt_flat[2 * a + 1] as i64)
                    - (pt_flat[2 * b + 1] as i64 - pt_flat[2 * a + 1] as i64)
                        * (pt_flat[2 * c] as i64 - pt_flat[2 * a] as i64);
                o <= 0
            })
            .count();
        eprintln!(
            "[gdel] twin consistency: {bad} bad half-edges of {}; non-CCW tris: {cw}",
            tri_count * 3
        );
    }
    Some(flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
}
