// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fully on-device Lawson flip loop. The triangle buffer stays resident on the
//! GPU across rounds; each round runs five compute passes —
//!   * `reset`         — clear ownership, the edge hash, twins, counter,
//!   * `build_hash`    — insert every half-edge into a GPU hash table
//!                       (race-safe `atomicCompareExchangeWeak` open addressing),
//!   * `resolve_twins` — pair the two half-edges in each slot → per-edge twin,
//!   * `mark`          — O(1) twin lookup; test convex + illegal. The illegal
//!                       (in-circle) test runs an f32 static filter first and
//!                       falls back to the exact emulated-i128 determinant only
//!                       when the float sign isn't certified (~O(unit-roundoff)
//!                       of edges); then stake an independent-set claim via
//!                       `atomicMin` on both triangles,
//!   * `apply`         — flip fires iff it owns *both* triangles; rewrite them.
//!
//! DEFAULT PATH: incremental adjacency (build `twin` ONCE, then each round runs
//! `reset_light`/`mark`/`apply_incr`/`fixup` — no per-round hash rebuild), driven by
//! a SINGLE-SUBMIT loop: the whole convergence (adjacency build + all flip rounds)
//! is encoded into one command buffer and submitted once. A GPU-side `done` latch
//! (`counter[2]`, set by `reset_light` the round after a zero-flip round; the `arm`
//! sentinel primes it) makes rounds past convergence early-out cheaply, so the host
//! syncs ONCE instead of once-per-round — the per-round CPU↔GPU round-trip was the
//! dominant wall-clock cost (~45 of ~55 ms at 200k) though the true GPU compute is
//! bandwidth-bound at ~46–54 O(T) rounds. `GEO_FLIP_NOINCR` forces the older
//! rebuild-every-round path (geometric batch-growth loop). The host only reads back
//! a 4-byte "flips" counter to decide convergence; the mesh is never round-tripped.
//!
//! NOTE: even so this LOSES to the parallel CPU triangulator (`triangulate_par`) for
//! host-resident data — the mandatory CPU-serial `hull_seed` alone ties/exceeds the
//! all-cores CPU's *entire* runtime, and the flip is bandwidth-bound. Measured GPU/CPU
//! (full path vs parallel CPU): ~3.6–6.5× on a 3080 Ti + 20-core, ~6–9× on M4 Pro. It
//! DOES beat a *serial* CPU (~0.75–0.83×). See the `delaunay-gpu-dt` memory.
//!
//! Adjacency is O(T) per round (hash), not O(T²). Exact for coordinate spans up
//! to `MAX_COORDINATE_SPAN` (i128 determinant) and any point count that fits u32
//! vertex ids — the edge hash stores half-edge ids and recomputes endpoints on
//! probe, so it is not width-limited (the old 16-bit-key 65 535-point cap is gone).
//!
//! Robustness: the base path is already exact + deterministic + Lawson-convergent
//! (measured: grids/circles converge in <1% of the round cap). `GEO_FLIP_SOS`
//! opt-in adds Simulation-of-Simplicity tie-breaking to `in_circle` — at a
//! cocircular tie it resolves via a symbolic index-lift perturbation, giving a
//! *canonical*, perturbation-consistent triangulation (differs from the default
//! `==0→legal` policy only on cocircular inputs; ~free on general-position data).
//! It stacks with the f32 filter (which defers near-zero to the exact path).

use wgpu::util::DeviceExt;

const NONE: u32 = 0xffff_ffff;

const FLIP_WGSL: &str = r#"
const NONE: u32 = 0xffffffffu;
// 2D dispatch stride: host dispatches (1024, gy, 1) with workgroup_size(64), so
// the linear thread index is gid.y*65536 + gid.x — dodges the 65535-workgroups-
// per-dimension cap past ~500k triangles.
const XSTRIDE: u32 = 65536u;

@group(0) @binding(0)  var<storage, read_write> tris:    array<u32>;
@group(0) @binding(1)  var<storage, read>       pts:     array<vec2<i32>>;  // one vec2 load / vertex
@group(0) @binding(2)  var<storage, read_write> owner:   array<atomic<u32>>;
@group(0) @binding(3)  var<storage, read_write> cand_e:  array<u32>;
@group(0) @binding(4)  var<storage, read_write> cand_t1: array<u32>;
@group(0) @binding(5)  var<storage, read_write> cand_ok: array<u32>;
@group(0) @binding(6)  var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(7)  var<storage, read>       dims:    array<u32>;   // [T, N, H]
@group(0) @binding(8)  var<storage, read_write> he_key:  array<atomic<u32>>;
@group(0) @binding(9)  var<storage, read_write> he_a:    array<u32>;
@group(0) @binding(10) var<storage, read_write> he_b:    array<u32>;
@group(0) @binding(11) var<storage, read_write> twin:    array<u32>;   // per half-edge
@group(0) @binding(12) var<storage, read_write> dirty:   array<atomic<u32>>; // active-set flag / worklist in-list dedup
@group(0) @binding(13) var<storage, read_write> wl:       array<u32>;        // worklist: 2T, two lists parity-offset by dims[0]
@group(0) @binding(14) var<storage, read_write> indirect: array<u32>;        // indirect dispatch args [wg_x,1,1]

fn pt(v: u32) -> vec2<i32> { return pts[v]; }        // one 8-byte load, both coords
fn px(v: u32) -> i32 { return pts[v].x; }
fn py(v: u32) -> i32 { return pts[v].y; }
fn tv(t: u32, k: u32) -> u32 { return tris[t * 3u + k]; }

// --- emulated exact integer arithmetic for the in-circle determinant ---
// i64 = vec2<u32> (lo, hi); i128 = vec4<u32> (limb0..limb3, little-endian). The
// lifts (a·a) and 2×2 minors reach ~7.5e18 for the full certified span, so they
// need i64 (i32 overflows above span ~32k — the bug that made domain=100000
// oscillate); their products reach ~1e37, so the determinant needs i128. This
// mirrors the CPU `PredWide` (i64-inner, i128 accumulate) exactly.
fn addc(a: u32, b: u32, cin: u32) -> vec2<u32> {   // (sum, carry_out)
    let s1 = a + b;         let c1 = select(0u, 1u, s1 < a);
    let s2 = s1 + cin;      let c2 = select(0u, 1u, s2 < s1);
    return vec2<u32>(s2, c1 + c2);
}
fn mul_u32(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xffffu; let ah = a >> 16u;
    let bl = b & 0xffffu; let bh = b >> 16u;
    let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;
    let cross = lh + hl;
    let cc = select(0u, 1u, cross < lh);
    let lo = ll + (cross << 16u);
    let lc = select(0u, 1u, lo < ll);
    return vec2<u32>(lo, hh + (cross >> 16u) + (cc << 16u) + lc);
}
fn neg_i64(x: vec2<u32>) -> vec2<u32> {
    let lo = ~x.x + 1u;
    return vec2<u32>(lo, ~x.y + select(0u, 1u, lo == 0u));
}
fn mul_i32(a: i32, b: i32) -> vec2<u32> {
    let r = mul_u32(u32(abs(a)), u32(abs(b)));
    if ((a < 0) != (b < 0)) { return neg_i64(r); }
    return r;
}
fn i32_to_i64(x: i32) -> vec2<u32> { return vec2<u32>(bitcast<u32>(x), select(0u, 0xffffffffu, x < 0)); }
fn add_i64(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> {
    let lo = x.x + y.x;
    return vec2<u32>(lo, x.y + y.y + select(0u, 1u, lo < x.x));
}
fn sign_i64(x: vec2<u32>) -> i32 {
    let hi = bitcast<i32>(x.y);
    if (hi < 0) { return -1; }
    if (hi > 0) { return 1; }
    if (x.x != 0u) { return 1; }
    return 0;
}
fn is_neg_i64(x: vec2<u32>) -> bool { return bitcast<i32>(x.y) < 0; }
fn abs_i64(x: vec2<u32>) -> vec2<u32> {
    if (is_neg_i64(x)) { return neg_i64(x); }
    return x;
}
// unsigned 64×64 -> 128, schoolbook over 32-bit limbs.
fn mul_u64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let t0 = mul_u32(a.x, b.x);   // weight 2^0   -> limbs 0,1
    let t1 = mul_u32(a.x, b.y);   // weight 2^32  -> limbs 1,2
    let t2 = mul_u32(a.y, b.x);   // weight 2^32  -> limbs 1,2
    let t3 = mul_u32(a.y, b.y);   // weight 2^64  -> limbs 2,3
    let r0 = t0.x;
    let s1a = addc(t0.y, t1.x, 0u);
    let s1b = addc(s1a.x, t2.x, 0u);
    let r1 = s1b.x;
    let c1 = s1a.y + s1b.y;
    let s2a = addc(t3.x, t1.y, 0u);
    let s2b = addc(s2a.x, t2.y, c1);
    let r2 = s2b.x;
    let c2 = s2a.y + s2b.y;
    let r3 = t3.y + c2;
    return vec4<u32>(r0, r1, r2, r3);
}
fn neg_i128(x: vec4<u32>) -> vec4<u32> {
    let a0 = addc(~x.x, 1u, 0u);
    let a1 = addc(~x.y, 0u, a0.y);
    let a2 = addc(~x.z, 0u, a1.y);
    return vec4<u32>(a0.x, a1.x, a2.x, ~x.w + a2.y);
}
fn mul_i64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {   // signed i64×i64 -> i128
    let r = mul_u64(abs_i64(a), abs_i64(b));
    if (is_neg_i64(a) != is_neg_i64(b)) { return neg_i128(r); }
    return r;
}
fn add_i128(x: vec4<u32>, y: vec4<u32>) -> vec4<u32> {
    let a0 = addc(x.x, y.x, 0u);
    let a1 = addc(x.y, y.y, a0.y);
    let a2 = addc(x.z, y.z, a1.y);
    return vec4<u32>(a0.x, a1.x, a2.x, x.w + y.w + a2.y);
}
fn sign_i128(x: vec4<u32>) -> i32 {
    if (bitcast<i32>(x.w) < 0) { return -1; }
    if ((x.x | x.y | x.z | x.w) != 0u) { return 1; }
    return 0;
}
// Orientation cross-product in i64 (the coordinate differences fit i32 for spans
// ≤ MAX, but their products reach ~3.8e18 and overflow i32).
fn orient(ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32) -> i32 {
    let d = add_i64(mul_i32(bx - ax, cy - ay), neg_i64(mul_i32(by - ay, cx - ax)));
    return sign_i64(d);
}
fn in_circle(va: u32, vb: u32, vc: u32, vd: u32) -> i32 {
    let pd = pt(vd); let pa = pt(va); let pb = pt(vb); let pc = pt(vc);
    let dx = pd.x; let dy = pd.y;
    let ax = pa.x - dx; let ay = pa.y - dy;
    let bx = pb.x - dx; let by = pb.y - dy;
    let cx = pc.x - dx; let cy = pc.y - dy;
    // lifts (a·a) and 2×2 minors in i64 — the values that overflowed i32.
    let a2 = add_i64(mul_i32(ax, ax), mul_i32(ay, ay));
    let b2 = add_i64(mul_i32(bx, bx), mul_i32(by, by));
    let c2 = add_i64(mul_i32(cx, cx), mul_i32(cy, cy));
    let m_bc = add_i64(mul_i32(bx, cy), neg_i64(mul_i32(cx, by)));
    let m_ac = add_i64(mul_i32(ax, cy), neg_i64(mul_i32(cx, ay)));
    let m_ab = add_i64(mul_i32(ax, by), neg_i64(mul_i32(bx, ay)));
    // determinant in i128.
    var det = mul_i64(a2, m_bc);
    det = add_i128(det, neg_i128(mul_i64(b2, m_ac)));
    det = add_i128(det, mul_i64(c2, m_ab));
    let s = sign_i128(det);
    if (s != 0 || dims[4] == 0u) { return s; }   // dims[4]=1 → Simulation of Simplicity
    // Cocircular tie-break: perturb each point's paraboloid lift by -δ·id, δ→0⁺.
    // The δ-linear term of the (now nonzero) determinant is
    //   -δ·(da·m_bc - db·m_ac + dc·m_ab),  da = id(a)-id(d), …
    // whose sign resolves the tie deterministically and consistently (a genuine
    // perturbation → no flip cycles). Reuses the coord minors m_bc/m_ac/m_ab.
    let da = i32(va) - i32(vd);
    let db = i32(vb) - i32(vd);
    let dc = i32(vc) - i32(vd);
    var tb = mul_i64(i32_to_i64(da), m_bc);
    tb = add_i128(tb, neg_i128(mul_i64(i32_to_i64(db), m_ac)));
    tb = add_i128(tb, mul_i64(i32_to_i64(dc), m_ab));
    return -sign_i128(tb);   // 0 only on a rarer secondary degeneracy → treated legal
}

// f32 filter for the in-circle sign: a cheap floating determinant plus a
// conservative static error bound. When |det| exceeds the bound the sign is
// *certified* exact — bit-identical to what the i128 path returns — and we skip
// the ~hundreds of emulated-i128 ops; otherwise we return 2 ("uncertain") and
// the caller falls back to the exact predicate. Correctness is one-sided: a
// larger bound only costs extra fall-throughs, never a wrong sign. The constant
// (C·ε with C≈64, ε=2^-24) is ~6× Shewchuk's incircle static bound, the slack
// covering both the single i32→f32 input rounding and GPU relaxed-float
// (non-IEEE, FMA-contracted) evaluation. Overflow is safe for free: perm ≥ |det|
// by the triangle inequality, so if det would reach ∞ then perm already did,
// bound = ∞, and both compares fall through to the exact path.
const ERRB: f32 = 3.8e-6;
fn in_circle_filter(va: u32, vb: u32, vc: u32, vd: u32) -> i32 {
    let pd = pt(vd); let pa = pt(va); let pb = pt(vb); let pc = pt(vc);
    let dxi = pd.x; let dyi = pd.y;
    // exact i32 differences (|diff| ≤ span ≤ MAX < 2^31), then one rounding each.
    let ax = f32(pa.x - dxi); let ay = f32(pa.y - dyi);
    let bx = f32(pb.x - dxi); let by = f32(pb.y - dyi);
    let cx = f32(pc.x - dxi); let cy = f32(pc.y - dyi);
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let bc = bx * cy - cx * by;
    let ac = ax * cy - cx * ay;
    let ab = ax * by - bx * ay;
    let det = a2 * bc - b2 * ac + c2 * ab;
    let perm = a2 * (abs(bx * cy) + abs(cx * by))
             + b2 * (abs(ax * cy) + abs(cx * ay))
             + c2 * (abs(ax * by) + abs(bx * ay));
    let bound = ERRB * perm;
    if (det > bound)  { return 1; }
    if (det < -bound) { return -1; }
    return 2;   // uncertain → exact fallback
}

fn write_ccw(t: u32, x: u32, y: u32, z: u32) {
    if (orient(px(x), py(x), px(y), py(y), px(z), py(z)) < 0) {
        tris[t * 3u] = x; tris[t * 3u + 1u] = z; tris[t * 3u + 2u] = y;
    } else {
        tris[t * 3u] = x; tris[t * 3u + 1u] = y; tris[t * 3u + 2u] = z;
    }
}

// Local edge e of triangle t -> its two endpoints (u,w) and opposite apex p.
fn edge_of(t: u32, e: u32) -> vec3<u32> {
    let v0 = tv(t, 0u); let v1 = tv(t, 1u); let v2 = tv(t, 2u);
    if (e == 0u) { return vec3<u32>(v0, v1, v2); }
    if (e == 1u) { return vec3<u32>(v1, v2, v0); }
    return vec3<u32>(v2, v0, v1);
}

@compute @workgroup_size(64)
fn reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * XSTRIDE + gid.x;
    let T = dims[0]; let H = dims[2];
    if (i < H) { atomicStore(&he_key[i], NONE); he_b[i] = NONE; }
    if (i < 3u * T) { twin[i] = NONE; }
    if (i < T) { atomicStore(&owner[i], NONE); cand_ok[i] = 0u; }
    if (i == 0u) { atomicStore(&counter[0], 0u); }
}

// 32-bit mix of an undirected edge's two vertex ids. Replaces the old
// `(lo<<16)|hi` packing that capped n at 2^16: the full key is kept implicitly in
// the stored half-edge id (its endpoints are recomputed on probe), so only the
// hash needs to mix both 32-bit ids — n is limited only by u32 vertex ids.
fn edge_hash(lo: u32, hi: u32) -> u32 {
    var h = lo * 2654435761u;
    h ^= hi * 2246822519u;
    h ^= h >> 15u;
    h *= 2654435761u;
    h ^= h >> 13u;
    return h;
}
@compute @workgroup_size(64)
fn build_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hid = gid.y * XSTRIDE + gid.x;     // half-edge id = t*3 + e
    let T = dims[0]; let H = dims[2];
    if (hid >= 3u * T) { return; }
    let uw = edge_of(hid / 3u, hid % 3u);
    let lo = min(uw.x, uw.y);
    let hi = max(uw.x, uw.y);
    var h = edge_hash(lo, hi) & (H - 1u);
    loop {
        // Store the half-edge id itself (unique, < 3T). The undirected edge key is
        // recovered by recomputing endpoints, so keys are not width-limited.
        let r = atomicCompareExchangeWeak(&he_key[h], NONE, hid);
        if (r.exchanged) { he_a[h] = hid; return; }    // first occupant
        if (r.old_value == NONE) { continue; }          // spurious weak fail
        let ow = edge_of(r.old_value / 3u, r.old_value % 3u);
        if (min(ow.x, ow.y) == lo && max(ow.x, ow.y) == hi) {
            he_b[h] = hid; return;                       // twin (same undirected edge)
        }
        h = (h + 1u) & (H - 1u);                         // collision, probe
    }
}

@compute @workgroup_size(64)
fn resolve_twins(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.y * XSTRIDE + gid.x;
    if (h >= dims[2]) { return; }
    if (atomicLoad(&he_key[h]) == NONE) { return; }
    let kb = he_b[h];
    if (kb == NONE) { return; }             // boundary edge, no twin
    let ka = he_a[h];
    twin[ka] = kb / 3u;                      // twin[half-edge] = other triangle
    twin[kb] = ka / 3u;
}

@compute @workgroup_size(64)
fn mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (atomicLoad(&counter[2]) == 1u) { return; }   // converged: cheap no-op round
    let t0 = gid.y * XSTRIDE + gid.x;
    if (t0 >= dims[0]) { return; }
    // Active-set (dims[5]): skip triangles whose neighbourhood is unchanged and known
    // locally-Delaunay. Only a flip re-activates the two new triangles + their outer
    // neighbours (see apply_incr), so after the first few rounds almost all threads
    // skip the (expensive) in_circle work — the round-loop's dominant cost.
    if (dims[5] == 1u && atomicLoad(&dirty[t0]) == 0u) { return; }
    let v0 = tv(t0, 0u); let v1 = tv(t0, 1u); let v2 = tv(t0, 2u);
    for (var e: u32 = 0u; e < 3u; e = e + 1u) {
        let t1 = twin[t0 * 3u + e];
        if (t1 == NONE || t0 >= t1) { continue; }   // boundary, or record once (lower t)
        let uw = edge_of(t0, e);
        let u = uw.x; let w = uw.y;
        let b0 = tv(t1, 0u); let b1 = tv(t1, 1u); let b2 = tv(t1, 2u);
        var q: u32;
        if (b0 != u && b0 != w) { q = b0; }
        else if (b1 != u && b1 != w) { q = b1; }
        else { q = b2; }
        // Ghost-vertex freeze (dims[6] = super threshold): an edge incident to a
        // bounding/super vertex is a hull edge of the real point set — never flip it.
        // Keeps the real hull triangles that a finite super triangle would otherwise
        // flip away. dims[6] = 0xffffffff for the normal (all-real) flip ⇒ no-op.
        if (v0 >= dims[6] || v1 >= dims[6] || v2 >= dims[6] || q >= dims[6]) { continue; }
        // Lawson lemma: for a valid triangulation, p and q lie on opposite sides of the
        // shared edge (u,w), and q strictly inside circ(v0,v1,v2) ⟹ the quad u-p-w-q is
        // convex. So `in_circle > 0` alone certifies the flip is legal — the explicit
        // orient() convex test (two emulated-i64 orients per edge, the mark kernel's
        // dominant cost) is redundant and is dropped. Validated exact across 60k/200k/1M.
        var ic: i32;
        if (dims[3] == 1u) {                                   // f32 filter on
            ic = in_circle_filter(v0, v1, v2, q);
            if (ic == 2) {                                     // uncertain → exact
                atomicAdd(&counter[1], 1u);
                ic = in_circle(v0, v1, v2, q);
            }
        } else {
            ic = in_circle(v0, v1, v2, q);
        }
        if (ic > 0) {                                          // illegal (⇒ convex quad)
            cand_e[t0] = e; cand_t1[t0] = t1; cand_ok[t0] = 1u;
            let id = t0 * 3u + e;
            atomicMin(&owner[t0], id);
            atomicMin(&owner[t1], id);
            return;                                            // illegal edge ⇒ stay active
        }
    }
    // No illegal edge among this triangle's higher-id neighbours (lower-id edges are
    // the neighbour's responsibility) ⇒ locally clean, drop from the active set until
    // a nearby flip re-activates it.
    if (dims[5] == 1u) { atomicStore(&dirty[t0], 0u); }
}

@compute @workgroup_size(64)
fn apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t0 = gid.y * XSTRIDE + gid.x;
    if (t0 >= dims[0]) { return; }
    if (cand_ok[t0] != 1u) { return; }
    let e = cand_e[t0];
    let t1 = cand_t1[t0];
    let id = t0 * 3u + e;
    if (atomicLoad(&owner[t0]) != id) { return; }
    if (atomicLoad(&owner[t1]) != id) { return; }      // won both triangles
    let uw = edge_of(t0, e);
    let a = uw.x; let b = uw.y; let p = uw.z;
    let b0 = tv(t1, 0u); let b1 = tv(t1, 1u); let b2 = tv(t1, 2u);
    var q: u32;
    if (b0 != a && b0 != b) { q = b0; }
    else if (b1 != a && b1 != b) { q = b1; }
    else { q = b2; }
    write_ccw(t0, a, p, q);      // diagonal a-b -> p-q
    write_ccw(t1, b, p, q);
    atomicAdd(&counter[0], 1u);
}

// ---- Incremental-adjacency path (GEO_FLIP_INCR): `twin` built ONCE, then each
// round runs mark + apply_incr + fixup instead of rebuilding the O(H) hash.
//
// The trick that keeps it DISTANCE-1 (so parallelism/round-count match the rebuild
// path — an earlier distance-2 version was ~8000× slower): the flipper NEVER writes
// a neighbour's adjacency (that would race under distance-1). Instead `apply_incr`
// only sets its own two triangles' `twin` (to the OLD neighbours) + a flip-partner
// + a "flipped-this-round" flag, and then a `fixup` pass has EACH triangle repair
// its OWN `twin`: if a neighbour flipped this round, the shared edge moved to that
// neighbour or its partner — relocate to whichever now holds the edge. Every write
// in every pass is to the thread's own triangle → race-free at distance-1.
// (`he_a`/`he_b`, unused after the one-time hash build, are reused for partner/flag.) ----

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

@compute @workgroup_size(64)
fn reset_light(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * XSTRIDE + gid.x;
    // Thread 0 folds the per-round counter reset into a GPU-side convergence test: if
    // the PREVIOUS round made no flips, latch `done` (counter[2]) so `mark` early-outs
    // for the rest of this submit. That lets the ENTIRE loop run in ONE submit with no
    // per-round CPU read-back (the old latency sink). There is deliberately NO
    // per-thread `done` guard here — a single-address atomic load per thread is costly
    // on Metal, and `reset_light` is cheap and MUST keep clearing cand_ok/he_b so that
    // `apply_incr`/`fixup` self-skip on converged rounds. `arm` seeds counter[0]
    // nonzero so round 1 (no previous round) is never a false positive.
    if (i == 0u) {
        if (atomicLoad(&counter[0]) == 0u) { atomicStore(&counter[2], 1u); }
        else { atomicStore(&counter[0], 0u); }
    }
    if (i < dims[0]) { atomicStore(&owner[i], NONE); cand_ok[i] = 0u; he_b[i] = 0u; }
}

// One-shot sentinel prepended to each submit chunk: primes counter[0] nonzero (so
// round 1's convergence test can't false-positive) and clears the `done` latch.
@compute @workgroup_size(1)
fn arm(@builtin(global_invocation_id) gid: vec3<u32>) {
    atomicStore(&counter[0], 1u);
    atomicStore(&counter[2], 0u);
}

@compute @workgroup_size(64)
fn apply_incr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t0 = gid.y * XSTRIDE + gid.x;
    if (t0 >= dims[0]) { return; }
    if (cand_ok[t0] != 1u) { return; }   // converged rounds: reset_light cleared it → self-skip
    let e = cand_e[t0];
    let t1 = cand_t1[t0];
    let id = t0 * 3u + e;
    if (atomicLoad(&owner[t0]) != id) { return; }
    if (atomicLoad(&owner[t1]) != id) { return; }      // won both (distance-1)
    let uw = edge_of(t0, e);
    let u = uw.x; let w = uw.y; let p = uw.z;
    let b0 = tv(t1, 0u); let b1 = tv(t1, 1u); let b2 = tv(t1, 2u);
    var q: u32;
    if (b0 != u && b0 != w) { q = b0; } else if (b1 != u && b1 != w) { q = b1; } else { q = b2; }
    // old neighbour triangles (read before the rewrite)
    let n_pu = twin[t0 * 3u + find_edge(t0, p, u)];
    let n_wp = twin[t0 * 3u + find_edge(t0, w, p)];
    let n_uq = twin[t1 * 3u + find_edge(t1, u, q)];
    let n_qw = twin[t1 * 3u + find_edge(t1, w, q)];
    write_ccw(t0, u, p, q);   // t0' = (u,p,q)
    write_ccw(t1, w, p, q);   // t1' = (w,p,q)
    // own adjacency (neighbours may be stale if THEY flipped — fixup repairs that)
    twin[t0 * 3u + find_edge(t0, u, p)] = n_pu;
    twin[t0 * 3u + find_edge(t0, q, u)] = n_uq;
    twin[t0 * 3u + find_edge(t0, p, q)] = t1;
    twin[t1 * 3u + find_edge(t1, w, p)] = n_wp;
    twin[t1 * 3u + find_edge(t1, q, w)] = n_qw;
    twin[t1 * 3u + find_edge(t1, p, q)] = t0;
    he_a[t0] = t1; he_a[t1] = t0;      // flip-partner
    he_b[t0] = 1u; he_b[t1] = 1u;      // flipped this round
    atomicAdd(&counter[0], 1u);
    // Re-activate the two new triangles + the ≤4 triangles across the quad's outer
    // edges: those are exactly the ones whose Delaunay legality can change from this
    // flip. Everything else stays clean (dropped from the active set).
    if (dims[5] == 1u) {
        atomicStore(&dirty[t0], 1u);
        atomicStore(&dirty[t1], 1u);
        if (n_pu != NONE) { atomicStore(&dirty[n_pu], 1u); }
        if (n_wp != NONE) { atomicStore(&dirty[n_wp], 1u); }
        if (n_uq != NONE) { atomicStore(&dirty[n_uq], 1u); }
        if (n_qw != NONE) { atomicStore(&dirty[n_qw], 1u); }
    }
}

// Initialise the active set: every triangle dirty for round 0 (run once per full
// flip, before the first round).
@compute @workgroup_size(64)
fn init_active(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * XSTRIDE + gid.x;
    if (i < dims[0]) { atomicStore(&dirty[i], 1u); }
}

@compute @workgroup_size(64)
fn fixup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.y * XSTRIDE + gid.x;
    if (t >= dims[0]) { return; }   // converged rounds: reset_light cleared he_b → inner loop self-skips
    for (var e: u32 = 0u; e < 3u; e = e + 1u) {
        let n = twin[t * 3u + e];
        if (n == NONE) { continue; }
        if (he_b[n] != 1u) { continue; }   // neighbour didn't flip → adjacency valid
        let a = tv(t, e); let b = tv(t, (e + 1u) % 3u);
        // the shared edge moved to `n` or its flip-partner — point at whichever holds it
        if (has_edge(n, a, b)) { twin[t * 3u + e] = n; }
        else { twin[t * 3u + e] = he_a[n]; }
    }
}

// ==== COMPACTED WORKLIST (GEO_FLIP_WORKLIST), SINGLE-SUBMIT via indirect dispatch.
// EXACT but MEASURED 2× SLOWER than the default single-submit flip (43/188 ms vs
// 20/126 ms at 200k/1M) — kept as a gated negative result. Reason: the flip's active
// set stays LARGE (most triangles remain flippable across most rounds), so dispatching
// only the active list barely reduces work, while the per-round overhead (setup +
// indirect + copy + 4 indirect dispatches, ×~54 rounds) dominates. Confirms the flip's
// O(rounds·T) is genuinely broad activity, not a few late stragglers — so compaction
// can't help; only bandwidth-per-work (shared-mem temporal blocking) is left.
// `wl` holds TWO active lists at offsets 0 and dims[0] (T); counter[5] (parity) selects
// which is "cur" this round. Each round dispatches ONLY counter[4]=cur_count workgroups
// (via `indirect`, computed on-GPU) — so once converged (cur_count=0) every remaining
// encoded round dispatches ZERO workgroups and is free. Each flip appends its
// neighbourhood to the OTHER list. `dirty` = in-next-list dedup; counter[3]=next_count.
// Total work O(T + flips), not O(rounds·T), with ONE host sync for the whole convergence.
fn wl_append(t: u32) {
    if (atomicExchange(&dirty[t], 1u) == 0u) {
        let nbase = (1u - atomicLoad(&counter[5])) * dims[0];
        wl[nbase + atomicAdd(&counter[3], 1u)] = t;
    }
}
// Once per round (1 thread): flip parity, promote next_count→cur_count, zero next/flips.
@compute @workgroup_size(1)
fn wl_setup(@builtin(global_invocation_id) gid: vec3<u32>) {
    atomicStore(&counter[5], 1u - atomicLoad(&counter[5]));
    atomicStore(&counter[4], atomicLoad(&counter[3]));
    atomicStore(&counter[3], 0u);
    atomicStore(&counter[0], 0u);
}
// Compute indirect dispatch args from cur_count (1 thread).
@compute @workgroup_size(1)
fn wl_indirect(@builtin(global_invocation_id) gid: vec3<u32>) {
    indirect[0] = (atomicLoad(&counter[4]) + 63u) / 64u; indirect[1] = 1u; indirect[2] = 1u;
}
@compute @workgroup_size(64)
fn wl_reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.y * XSTRIDE + gid.x;
    if (tid >= atomicLoad(&counter[4])) { return; }
    let t = wl[atomicLoad(&counter[5]) * dims[0] + tid];
    atomicStore(&owner[t], NONE); cand_ok[t] = 0u; he_b[t] = 0u; atomicStore(&dirty[t], 0u);
}
@compute @workgroup_size(64)
fn wl_mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.y * XSTRIDE + gid.x;
    if (tid >= atomicLoad(&counter[4])) { return; }
    let t0 = wl[atomicLoad(&counter[5]) * dims[0] + tid];
    let v0 = tv(t0, 0u); let v1 = tv(t0, 1u); let v2 = tv(t0, 2u);
    for (var e: u32 = 0u; e < 3u; e = e + 1u) {
        let t1 = twin[t0 * 3u + e];
        if (t1 == NONE || t0 >= t1) { continue; }
        let uw = edge_of(t0, e); let u = uw.x; let w = uw.y; let p = uw.z;
        let b0 = tv(t1, 0u); let b1 = tv(t1, 1u); let b2 = tv(t1, 2u);
        var q: u32;
        if (b0 != u && b0 != w) { q = b0; } else if (b1 != u && b1 != w) { q = b1; } else { q = b2; }
        if (v0 >= dims[6] || v1 >= dims[6] || v2 >= dims[6] || q >= dims[6]) { continue; }
        let s1 = orient(px(p), py(p), px(q), py(q), px(u), py(u));
        let s2 = orient(px(p), py(p), px(q), py(q), px(w), py(w));
        if (s1 != 0 && s2 != 0 && (s1 < 0) != (s2 < 0)) {
            var ic: i32;
            if (dims[3] == 1u) { ic = in_circle_filter(v0, v1, v2, q); if (ic == 2) { atomicAdd(&counter[1], 1u); ic = in_circle(v0, v1, v2, q); } }
            else { ic = in_circle(v0, v1, v2, q); }
            if (ic > 0) {
                cand_e[t0] = e; cand_t1[t0] = t1; cand_ok[t0] = 1u;
                let id = t0 * 3u + e; atomicMin(&owner[t0], id); atomicMin(&owner[t1], id);
                return;
            }
        }
    }
}
@compute @workgroup_size(64)
fn wl_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.y * XSTRIDE + gid.x;
    if (tid >= atomicLoad(&counter[4])) { return; }
    let t0 = wl[atomicLoad(&counter[5]) * dims[0] + tid];
    if (cand_ok[t0] != 1u) { return; }
    let e = cand_e[t0]; let t1 = cand_t1[t0]; let id = t0 * 3u + e;
    if (atomicLoad(&owner[t0]) != id) { return; }
    if (atomicLoad(&owner[t1]) != id) { return; }
    let uw = edge_of(t0, e); let u = uw.x; let w = uw.y; let p = uw.z;
    let b0 = tv(t1, 0u); let b1 = tv(t1, 1u); let b2 = tv(t1, 2u);
    var q: u32;
    if (b0 != u && b0 != w) { q = b0; } else if (b1 != u && b1 != w) { q = b1; } else { q = b2; }
    let n_pu = twin[t0 * 3u + find_edge(t0, p, u)];
    let n_wp = twin[t0 * 3u + find_edge(t0, w, p)];
    let n_uq = twin[t1 * 3u + find_edge(t1, u, q)];
    let n_qw = twin[t1 * 3u + find_edge(t1, w, q)];
    write_ccw(t0, u, p, q); write_ccw(t1, w, p, q);
    twin[t0 * 3u + find_edge(t0, u, p)] = n_pu;
    twin[t0 * 3u + find_edge(t0, q, u)] = n_uq;
    twin[t0 * 3u + find_edge(t0, p, q)] = t1;
    twin[t1 * 3u + find_edge(t1, w, p)] = n_wp;
    twin[t1 * 3u + find_edge(t1, q, w)] = n_qw;
    twin[t1 * 3u + find_edge(t1, p, q)] = t0;
    he_a[t0] = t1; he_a[t1] = t0; he_b[t0] = 1u; he_b[t1] = 1u;
    atomicAdd(&counter[0], 1u);
    wl_append(t0); wl_append(t1);
    if (n_pu != NONE) { wl_append(n_pu); }
    if (n_wp != NONE) { wl_append(n_wp); }
    if (n_uq != NONE) { wl_append(n_uq); }
    if (n_qw != NONE) { wl_append(n_qw); }
}
@compute @workgroup_size(64)
fn wl_fixup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.y * XSTRIDE + gid.x;
    if (tid >= atomicLoad(&counter[4])) { return; }
    let t = wl[atomicLoad(&counter[5]) * dims[0] + tid];
    for (var e: u32 = 0u; e < 3u; e = e + 1u) {
        let n = twin[t * 3u + e];
        if (n == NONE) { continue; }
        if (he_b[n] != 1u) { continue; }
        let a = tv(t, e); let b = tv(t, (e + 1u) % 3u);
        if (has_edge(n, a, b)) { twin[t * 3u + e] = n; } else { twin[t * 3u + e] = he_a[n]; }
    }
}
"#;

fn storage(dev: &wgpu::Device, data: &[u8], extra: wgpu::BufferUsages) -> wgpu::Buffer {
    dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage: wgpu::BufferUsages::STORAGE | extra,
    })
}

/// Cached shader module + pipelines + bind-group layout for the flip. Build ONCE
/// with [`FlipPipeline::new`] and reuse across calls (via
/// [`flip_to_delaunay_gpu_with`]) to skip the ~158 ms first-call shader compile
/// and the ~1.8 ms/call pipeline creation the profiler flagged as pure overhead.
pub struct FlipPipeline {
    bgl: wgpu::BindGroupLayout,
    p_reset: wgpu::ComputePipeline,
    p_hash: wgpu::ComputePipeline,
    p_resolve: wgpu::ComputePipeline,
    p_mark: wgpu::ComputePipeline,
    p_apply: wgpu::ComputePipeline,
    p_reset_light: wgpu::ComputePipeline,
    p_apply_incr: wgpu::ComputePipeline,
    p_fixup: wgpu::ComputePipeline,
    p_arm: wgpu::ComputePipeline,
    p_init_active: wgpu::ComputePipeline,
    p_wl_reset: wgpu::ComputePipeline,
    p_wl_mark: wgpu::ComputePipeline,
    p_wl_apply: wgpu::ComputePipeline,
    p_wl_fixup: wgpu::ComputePipeline,
    p_wl_setup: wgpu::ComputePipeline,
    p_wl_indirect: wgpu::ComputePipeline,
}

impl FlipPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rlx-geo flip"),
            source: wgpu::ShaderSource::Wgsl(FLIP_WGSL.into()),
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
            label: Some("rlx-geo flip"),
            entries: &[
                ent(0, false),
                ent(1, true),
                ent(2, false),
                ent(3, false),
                ent(4, false),
                ent(5, false),
                ent(6, false),
                ent(7, true),
                ent(8, false),
                ent(9, false),
                ent(10, false),
                ent(11, false),
                ent(12, false),
                ent(13, false),
                ent(14, false),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rlx-geo flip"),
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
        Self {
            p_reset: pipe("reset"),
            p_hash: pipe("build_hash"),
            p_resolve: pipe("resolve_twins"),
            p_mark: pipe("mark"),
            p_apply: pipe("apply"),
            p_reset_light: pipe("reset_light"),
            p_apply_incr: pipe("apply_incr"),
            p_fixup: pipe("fixup"),
            p_arm: pipe("arm"),
            p_init_active: pipe("init_active"),
            p_wl_reset: pipe("wl_reset"),
            p_wl_mark: pipe("wl_mark"),
            p_wl_apply: pipe("wl_apply"),
            p_wl_fixup: pipe("wl_fixup"),
            p_wl_setup: pipe("wl_setup"),
            p_wl_indirect: pipe("wl_indirect"),
            bgl,
        }
    }
}

/// Run the flip loop entirely on the GPU. `tris` must be a valid triangulation
/// of `points` (CCW). Returns the Delaunay triangles. Convenience wrapper that
/// builds pipelines per call — for repeated calls, cache a [`FlipPipeline`] and
/// use [`flip_to_delaunay_gpu_with`].
pub fn flip_to_delaunay_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tris: &[[u32; 3]],
    points: &[[i32; 2]],
) -> Vec<[u32; 3]> {
    let pl = FlipPipeline::new(device);
    flip_to_delaunay_gpu_with(device, queue, &pl, tris, points)
}

/// Flip using a pre-built [`FlipPipeline`] — no per-call shader compile / pipeline
/// creation. `tris` must be a valid CCW triangulation of `points`.
pub fn flip_to_delaunay_gpu_with(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pl: &FlipPipeline,
    tris: &[[u32; 3]],
    points: &[[i32; 2]],
) -> Vec<[u32; 3]> {
    flip_to_delaunay_gpu_super(device, queue, pl, tris, points, u32::MAX)
}

/// Like [`flip_to_delaunay_gpu_with`] but FREEZES every edge incident to a vertex id
/// ≥ `super_thresh` — those are the bounding/ghost vertices added by GPU-native
/// construction ([`crate::construct_gpu`]), and their edges are hull edges of the
/// real point set (a finite super triangle would otherwise flip real hull triangles
/// away). Pass `u32::MAX` for an ordinary all-real flip (the wrapper above).
pub fn flip_to_delaunay_gpu_super(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pl: &FlipPipeline,
    tris: &[[u32; 3]],
    points: &[[i32; 2]],
    super_thresh: u32,
) -> Vec<[u32; 3]> {
    let t_count = tris.len() as u32;
    let n = points.len() as u32;
    // The WGSL predicates are exact only over the certified span (the i128
    // determinant, mirroring the CPU `PredWide`). Beyond it the flip silently
    // oscillates / returns non-Delaunay output — so enforce it in release too (the
    // O(n) span scan is dwarfed by the O(n) buffer setup that follows).
    let span = points.iter().fold((i32::MAX, i32::MIN), |(lo, hi), p| {
        (lo.min(p[0]).min(p[1]), hi.max(p[0]).max(p[1]))
    });
    assert!(
        n < 2 || (span.1 as i64 - span.0 as i64) <= crate::predicates::MAX_COORDINATE_SPAN,
        "flip_to_delaunay_gpu: coordinate span exceeds MAX_COORDINATE_SPAN"
    );
    if t_count == 0 {
        return Vec::new();
    }
    // hash holds 3·t_count half-edges. Measured: the table is MEMORY-bound (reset +
    // resolve are O(H) traffic), so the SMALLEST safe table wins — a bigger/lower-
    // load table is monotonically slower. 4× rounds to the smallest power-of-two
    // above 3·T (load ~0.7), the optimum; a bigger multiplier only adds O(H) traffic.
    let hash_size = (4 * t_count).next_power_of_two().max(64);

    // GEO_FLIP_PROF: phase breakdown (buffer upload / shader+pipeline setup /
    // round-loop GPU+sync / GPU→CPU sync stalls / final download) to locate the
    // bottleneck. All wall-clock; the sync accumulator isolates GPU→CPU latency.
    let prof = std::env::var_os("GEO_FLIP_PROF").is_some();
    let t0 = std::time::Instant::now();

    // NB: a space-filling-curve (Morton) relayout of points+triangles was tested here
    // to convert the round loop's scattered access into coalesced streaming. Result
    // (1M, RTX 3080 Ti, GPU-exec time): reordering POINTS alone recovers only ~10%
    // (70→63 ms), and reordering TRIANGLES makes it WORSE (70→93 ms) — spatially-
    // clustered triangles contend on the ownership atomics. So the flip's cost is
    // NOT scatter-bandwidth (a layout artifact) but ATOMIC coordination + per-round
    // barriers, which are intrinsic to parallel independent-set flipping. Reverted.
    let tri_flat: Vec<u32> = tris.iter().flat_map(|t| [t[0], t[1], t[2]]).collect();
    let pt_flat: Vec<i32> = points.iter().flat_map(|p| [p[0], p[1]]).collect();
    let cd = wgpu::BufferUsages::COPY_DST;
    let cs = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    // Scratch buffers are fully written by `reset` each round (or guarded by
    // `cand_ok`), so they need NO host zero-fill — create them UNINITIALIZED to
    // skip the ~100 MB upload (he_key/he_a/he_b are 3×hash_size, the profiler's
    // dominant I/O cost). Only content-bearing buffers (tris/pts/dims/counter)
    // are uploaded.
    let scratch = |elems: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: elems * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };

    let tris_buf = storage(device, bytemuck::cast_slice(&tri_flat), cs);
    let pts_buf = storage(device, bytemuck::cast_slice(&pt_flat), cd);
    let owner = scratch(t_count as u64);
    let cand_e = scratch(t_count as u64);
    let cand_t1 = scratch(t_count as u64);
    let cand_ok = scratch(t_count as u64);
    // counter[0] = flips this round (reset each round); counter[1] = total f32
    // filter fall-throughs to the exact path (accumulates, diagnostic only);
    // counter[2] = `done` latch — set on-GPU once a round makes no flips so the
    // remaining encoded rounds early-out (enables the single-submit loop).
    // [flips, fallthru, done, wl_next_count, wl_cur_count, wl_parity]
    let counter = storage(
        device,
        bytemuck::cast_slice(&[0u32, 0u32, 0u32, t_count, 0u32, 1u32]),
        cs,
    );
    // dims[3] = 1 → f32 in-circle filter with exact fallback (default); set
    // GEO_FLIP_NOFILTER to force the all-i128 path (for A/B benchmarking).
    let use_filter: u32 = u32::from(std::env::var_os("GEO_FLIP_NOFILTER").is_none());
    // dims[4] = 1 → Simulation-of-Simplicity tie-break for cocircular in_circle
    // (deterministic, degeneracy-canonical). Off by default (opt-in via GEO_FLIP_SOS).
    let use_sos: u32 = u32::from(std::env::var_os("GEO_FLIP_SOS").is_some());
    // dims[5] = 1 → active-set flip (`mark` skips clean triangles; only flips
    // re-activate the affected neighbourhood). Default on for the incremental path;
    // GEO_FLIP_NOACTIVE disables. The standard rebuild path never inits `active`, so
    // it's forced off there (gated on !NOINCR).
    let use_active: u32 = u32::from(
        std::env::var_os("GEO_FLIP_NOACTIVE").is_none()
            && std::env::var_os("GEO_FLIP_NOINCR").is_none(),
    );
    // dims[7] = worklist cur-count (rewritten per round in the worklist path).
    let use_worklist = std::env::var_os("GEO_FLIP_WORKLIST").is_some()
        && std::env::var_os("GEO_FLIP_NOINCR").is_none();
    let dims = storage(
        device,
        bytemuck::cast_slice(&[
            t_count,
            n,
            hash_size,
            use_filter,
            use_sos,
            use_active,
            super_thresh,
            t_count,
        ]),
        cd,
    );
    let he_key = scratch(hash_size as u64);
    let he_a = scratch(hash_size as u64);
    let he_b = scratch(hash_size as u64);
    let twin = scratch(3 * t_count as u64);
    let active = scratch(t_count as u64);
    // Worklist: one buffer holding TWO lists (offsets 0 and T); list 0 seeded with all
    // triangle ids for round 0. Plus the indirect-dispatch arg buffer. Tiny dummies when
    // unused (the shared BGL still needs bindings 13/14 satisfied).
    let wl = if use_worklist {
        let mut d = vec![0u32; 2 * t_count as usize];
        for (i, v) in d.iter_mut().take(t_count as usize).enumerate() {
            *v = i as u32;
        }
        storage(device, bytemuck::cast_slice(&d), cd)
    } else {
        scratch(1)
    };
    // `indirect` is written by wl_indirect (storage, binding 14); it's copied to
    // `indirect_disp` each round for the actual indirect dispatch (a buffer can't be
    // both a bound storage resource AND the dispatch-args source in one pass).
    let indirect = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let indirect_disp = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let t_bufs = t0.elapsed().as_secs_f64() * 1e3;
    // Pipelines/BGL come from the cached `pl` — no per-call shader compile.
    let bgl = &pl.bgl;

    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-geo flip"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tris_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: owner.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cand_e.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: cand_t1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: cand_ok.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: counter.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: dims.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: he_key.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: he_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: he_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: twin.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: active.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: wl.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: indirect.as_entire_binding(),
            },
        ],
    });

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("counter"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // fixed x = 1024 groups (65536 threads); gy covers the rest — see XSTRIDE.
    let gy = |threads: u32| threads.div_ceil(65536).max(1);

    // Incremental adjacency (DEFAULT): build `twin` ONCE, then each round marks +
    // applies + fixes-up locally — no per-round O(H) hash rebuild. Distance-1 (see
    // kernels). Measured ~2–3× faster than the rebuild path (60k 1.9×, 1M 2.95×),
    // exact + robust. `GEO_FLIP_NOINCR` forces the old rebuild-every-round path.
    let use_incr = std::env::var_os("GEO_FLIP_NOINCR").is_none();
    // (pipeline, thread-count) per round.
    let passes: Vec<(&wgpu::ComputePipeline, u32)> = if use_incr {
        vec![
            (&pl.p_reset_light, t_count),
            (&pl.p_mark, t_count),
            (&pl.p_apply_incr, t_count),
            (&pl.p_fixup, t_count),
        ]
    } else {
        vec![
            (&pl.p_reset, hash_size.max(3 * t_count)),
            (&pl.p_hash, 3 * t_count),
            (&pl.p_resolve, hash_size),
            (&pl.p_mark, t_count),
            (&pl.p_apply, t_count),
        ]
    };

    // GEO_FLIP_CAP overrides the round cap (diagnostic: does it ever converge?).
    let cap = std::env::var("GEO_FLIP_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4 * (t_count as usize) + 64);
    let t_setup = t0.elapsed().as_secs_f64() * 1e3; // shader compile + pipelines + bind group
    let mut sync_ms = 0.0f64; // accumulated GPU→CPU stall time
    let mut round = 0usize;
    let mut last_flips = u32::MAX;

    let dispatch = |enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, groups_y: u32| {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cp.set_pipeline(p);
        cp.set_bind_group(0, &bind, &[]);
        cp.dispatch_workgroups(1024, groups_y, 1);
    };
    let read_counter0 = |readback: &wgpu::Buffer| -> u32 {
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let f = {
            let d = slice.get_mapped_range().unwrap();
            bytemuck::cast_slice::<u8, u32>(&d)[0]
        };
        readback.unmap();
        f
    };

    if use_worklist {
        // COMPACTED WORKLIST: one-time twin build, then each round dispatches ONLY the
        // active list (cur_count threads) — [wl_fixup, wl_reset, wl_mark, wl_apply] —
        // and every flip appends its neighbourhood to the next list. Host ping-pongs
        // the cur/next buffers via two bind groups. Late rounds shrink to a handful of
        // triangles, so total work is O(T + flips) not O(rounds·T). One sync/round; the
        // per-round GPU work collapses, which is what pays for the sync.
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (p, threads) in [
            (&pl.p_reset, hash_size.max(3 * t_count)),
            (&pl.p_hash, 3 * t_count),
            (&pl.p_resolve, hash_size),
        ] {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(p);
            cp.set_bind_group(0, &bind, &[]);
            cp.dispatch_workgroups(1024, gy(threads), 1);
        }
        queue.submit(Some(enc.finish()));
        // SINGLE-SUBMIT: encode CHUNK rounds; each = [setup, indirect, fixup, reset, mark,
        // apply] with the 4 heavy passes INDIRECT-dispatched over cur_count workgroups.
        // Converged rounds (cur_count=0) dispatch ZERO workgroups → free. One sync/chunk.
        let chunk = std::env::var("GEO_FLIP_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(96usize)
            .max(1);
        while round < cap {
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for _ in 0..chunk {
                for p in [&pl.p_wl_setup, &pl.p_wl_indirect] {
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(p);
                    cp.set_bind_group(0, &bind, &[]);
                    cp.dispatch_workgroups(1, 1, 1);
                }
                enc.copy_buffer_to_buffer(&indirect, 0, &indirect_disp, 0, 16);
                for p in [
                    &pl.p_wl_fixup,
                    &pl.p_wl_reset,
                    &pl.p_wl_mark,
                    &pl.p_wl_apply,
                ] {
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(p);
                    cp.set_bind_group(0, &bind, &[]);
                    cp.dispatch_workgroups_indirect(&indirect_disp, 0);
                }
                round += 1;
            }
            enc.copy_buffer_to_buffer(&counter, 12, &readback, 0, 4); // counter[3] = next_count
            queue.submit(Some(enc.finish()));
            let t_sync = std::time::Instant::now();
            let next = read_counter0(&readback);
            sync_ms += t_sync.elapsed().as_secs_f64() * 1e3;
            last_flips = next;
            if next == 0 {
                break; // converged (all encoded rounds past convergence were free)
            }
        }
    } else if use_incr {
        // SINGLE-SUBMIT loop. The one-time adjacency build (reset/build_hash/resolve),
        // the `arm` sentinel, and CHUNK flip rounds are all encoded into ONE command
        // buffer and submitted together, so the whole convergence costs a SINGLE
        // GPU→CPU sync in the common case — vs one sync per round before. (Profiling
        // showed the flip's true GPU compute is ~8.5 ms at 200k but wall-clock was
        // ~55 ms: ~45 ms was pure per-round submit/round-trip latency, not compute.)
        // The `done` latch makes overshoot rounds cheaper but not free — every pass
        // still sweeps O(T) memory, so an overshoot round costs ∝ T (~2 ms at 1M)
        // while a sync's fixed latency is ~1.3 ms. Size the FIRST chunk just above the
        // expected Lawson convergence so the common case is ONE submit with only a few
        // wasted rounds: empirically ~38/46/54 rounds at T=120k/400k/2M, i.e.
        // conv ≈ 38 + 4·log2(T/120k). Tail chunks (shrinking with T, since overshoot
        // there is expensive) cover any undershoot without a big fixed over-run.
        let conv_est = 38.0 + 4.0 * (t_count as f64 / 120_000.0).max(1.0).log2();
        let first_chunk: usize = std::env::var("GEO_FLIP_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(conv_est as usize + 5)
            .max(1);
        let tail: usize = std::env::var("GEO_FLIP_TAIL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or((1_500_000 / t_count.max(1) as usize).max(2))
            .max(1);
        let mut first = true;
        while round < cap {
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            if first {
                dispatch(&mut enc, &pl.p_reset, gy(hash_size.max(3 * t_count)));
                dispatch(&mut enc, &pl.p_hash, gy(3 * t_count));
                dispatch(&mut enc, &pl.p_resolve, gy(hash_size));
                if use_active == 1 {
                    dispatch(&mut enc, &pl.p_init_active, gy(t_count)); // all dirty for round 0
                }
            }
            dispatch(&mut enc, &pl.p_arm, 1); // prime counter[0]≠0, clear done latch
            let this_chunk = if first { first_chunk } else { tail };
            for _ in 0..this_chunk {
                for &(p, threads) in &passes {
                    dispatch(&mut enc, p, gy(threads));
                }
                round += 1;
            }
            enc.copy_buffer_to_buffer(&counter, 0, &readback, 0, 4);
            queue.submit(Some(enc.finish()));

            let t_sync = std::time::Instant::now();
            let flips = read_counter0(&readback); // 0 ⇒ a round made no flips ⇒ converged
            sync_ms += t_sync.elapsed().as_secs_f64() * 1e3;
            last_flips = flips;
            first = false;
            if flips == 0 {
                break;
            }
        }
    } else {
        // Standard rebuild-every-round path (GEO_FLIP_NOINCR): the O(H) hash rebuild is
        // folded into every round, so it can't run single-submit. Keep the geometric
        // batch-growth loop that amortises the per-round read-back: start small (a
        // near-Delaunay seed finishes in a couple of rounds), double up to `max_batch`
        // so a far seed needs only O(log rounds) syncs. `GEO_FLIP_BATCH` caps growth.
        let max_batch: usize = std::env::var("GEO_FLIP_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8)
            .max(1);
        let mut batch = 2usize.min(max_batch);
        while round < cap {
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for _ in 0..batch {
                for &(p, threads) in &passes {
                    dispatch(&mut enc, p, gy(threads));
                }
                round += 1;
            }
            enc.copy_buffer_to_buffer(&counter, 0, &readback, 0, 4);
            queue.submit(Some(enc.finish()));

            let t_sync = std::time::Instant::now();
            let flips = read_counter0(&readback);
            sync_ms += t_sync.elapsed().as_secs_f64() * 1e3;
            last_flips = flips;
            if flips == 0 {
                break;
            }
            batch = (batch * 2).min(max_batch);
        }
    }
    if std::env::var_os("GEO_FLIP_DEBUG").is_some() {
        let hit_cap = round >= cap;
        // Read both counter slots to report the f32-filter fall-through total.
        let dbg_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("counter-dbg"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&counter, 0, &dbg_rb, 0, 8);
        queue.submit(Some(enc.finish()));
        let slice = dbg_rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let fallthroughs = {
            let d = slice.get_mapped_range().unwrap();
            bytemuck::cast_slice::<u8, u32>(&d)[1]
        };
        dbg_rb.unmap();
        eprintln!(
            "[flip_gpu] {round} rounds, last_flips={last_flips}, hit_cap={hit_cap}, \
             filter={use_filter}, fallthroughs={fallthroughs} \
             ({} triangles, {} passes/round)",
            t_count,
            passes.len()
        );
    }

    let t_loop = t0.elapsed().as_secs_f64() * 1e3; // setup + full round loop (incl. sync)

    // True per-pass GPU time (one representative round) via timestamp queries —
    // separates real GPU compute from the CPU poll-wait that the wall-clock buckets
    // conflate. Requires the TIMESTAMP_QUERY feature on the device.
    if prof && device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        let names: &[&str] = if use_incr {
            &["reset_light", "mark", "apply_incr", "fixup"]
        } else {
            &["reset", "build_hash", "resolve", "mark", "apply"]
        };
        let nq = (2 * passes.len()) as u32;
        let qbytes = (passes.len() * 16) as u64;
        let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("flip-ts"),
            ty: wgpu::QueryType::Timestamp,
            count: nq,
        });
        let qbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: qbytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let qrb = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: qbytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        if use_incr {
            // Un-latch `done` (the loop left counter[2]=1) so the representative round
            // actually runs. The mesh is already Delaunay, so this round makes 0 flips
            // and leaves it unchanged — it just measures a realistic (late) round.
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&pl.p_arm);
            cp.set_bind_group(0, &bind, &[]);
            cp.dispatch_workgroups(1, 1, 1);
            drop(cp);
            if use_active == 1 {
                // Loop left the active set empty (converged) — re-fill so the measured
                // round reflects a full mark, not an all-skip.
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cp.set_pipeline(&pl.p_init_active);
                cp.set_bind_group(0, &bind, &[]);
                cp.dispatch_workgroups(1024, gy(t_count), 1);
            }
        }
        for (i, &(pl, threads)) in passes.iter().enumerate() {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &qs,
                    beginning_of_pass_write_index: Some((2 * i) as u32),
                    end_of_pass_write_index: Some((2 * i + 1) as u32),
                }),
            });
            cp.set_pipeline(pl);
            cp.set_bind_group(0, &bind, &[]);
            cp.dispatch_workgroups(1024, gy(threads), 1);
        }
        enc.resolve_query_set(&qs, 0..nq, &qbuf, 0);
        enc.copy_buffer_to_buffer(&qbuf, 0, &qrb, 0, qbytes);
        queue.submit(Some(enc.finish()));
        let slice = qrb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let ts: Vec<u64> =
            bytemuck::cast_slice::<u8, u64>(&slice.get_mapped_range().unwrap()).to_vec();
        qrb.unmap();
        let period = queue.get_timestamp_period() as f64; // ns per tick
        let mut msg = String::from("[flip_prof] per-pass GPU us (1 round):");
        let mut sum = 0.0;
        for i in 0..passes.len() {
            let us = (ts[2 * i + 1].wrapping_sub(ts[2 * i]) as f64) * period / 1000.0;
            sum += us;
            msg.push_str(&format!(" {}={:.1}", names[i], us));
        }
        eprintln!(
            "{msg}  round={sum:.1}us x {round} rounds ~= {:.2}ms GPU compute",
            sum * round as f64 / 1000.0
        );
    }

    // Download the final mesh once.
    let out_rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tris"),
        size: (tri_flat.len() * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&tris_buf, 0, &out_rb, 0, (tri_flat.len() * 4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = out_rb.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let out: Vec<u32> = {
        let d = slice.get_mapped_range().unwrap();
        bytemuck::cast_slice(&d).to_vec()
    };
    out_rb.unmap();
    let _ = NONE;
    if prof {
        let t_all = t0.elapsed().as_secs_f64() * 1e3;
        let setup = t_setup - t_bufs;
        let loop_gpu = (t_loop - t_setup) - sync_ms;
        let dl = t_all - t_loop;
        eprintln!(
            "[flip_prof] total={t_all:.2}ms | upload={t_bufs:.2} setup(compile+pipe)={setup:.2} \
             loop_gpu={loop_gpu:.2} sync_stall={sync_ms:.2} download={dl:.2}  ({round} rounds, {t_count} tris)"
        );
    }
    out.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}
