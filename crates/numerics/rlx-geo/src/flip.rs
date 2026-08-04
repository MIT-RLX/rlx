// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lawson edge-flip Delaunay, structured as **parallel independent-set rounds** —
//! the CPU reference for the GPU flip pipeline.
//!
//! Given any valid triangulation of a point set, each round:
//!   1. builds edge → (triangle, apex) adjacency,
//!   2. marks every interior edge whose quad is convex **and** illegal (the
//!      opposite apex lies strictly inside the triangle's circumcircle — the
//!      exact `in_circle` predicate),
//!   3. selects an independent set (each triangle takes part in ≤ 1 flip),
//!   4. flips them all at once (diagonal `a-b` → `p-q`).
//!
//! Repeat until no edge is illegal — the result is Delaunay (Lawson's theorem).
//!
//! Adjacency is rebuilt each round, so a flip only rewrites the two triangles it
//! owns and there is no pointer surgery — which is exactly what makes the round a
//! race-free data-parallel kernel. `flip_all_convex_once` provides a cheap
//! non-Delaunay seed for testing (and mirrors the same round machinery).

use std::collections::HashMap;

// Exact i128 predicates over i64 coordinates.
#[inline]
fn orient(a: [i64; 2], b: [i64; 2], c: [i64; 2]) -> i32 {
    let v = (b[0] - a[0]) as i128 * (c[1] - a[1]) as i128
        - (b[1] - a[1]) as i128 * (c[0] - a[0]) as i128;
    (v > 0) as i32 - (v < 0) as i32
}

#[inline]
fn in_circle(a: [i64; 2], b: [i64; 2], c: [i64; 2], d: [i64; 2]) -> i32 {
    let ax = (a[0] - d[0]) as i128;
    let ay = (a[1] - d[1]) as i128;
    let bx = (b[0] - d[0]) as i128;
    let by = (b[1] - d[1]) as i128;
    let cx = (c[0] - d[0]) as i128;
    let cy = (c[1] - d[1]) as i128;
    let det = (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay);
    (det > 0) as i32 - (det < 0) as i32
}

fn coords(points: &[[i32; 2]]) -> Vec<[i64; 2]> {
    points.iter().map(|p| [p[0] as i64, p[1] as i64]).collect()
}

/// Order (a,b,c) counterclockwise.
fn ccw3(a: u32, b: u32, c: u32, v: &[[i64; 2]]) -> [u32; 3] {
    if orient(v[a as usize], v[b as usize], v[c as usize]) < 0 {
        [a, c, b]
    } else {
        [a, b, c]
    }
}

// One flip candidate: triangles t0,t1 sharing edge (a,b) with apexes p (in t0),
// q (in t1). After the flip the diagonal becomes p-q.
struct Flip {
    t0: usize,
    t1: usize,
    a: u32,
    b: u32,
    p: u32,
    q: u32,
}

/// Collect an independent set of flips. If `require_illegal`, only edges whose
/// opposite apex is inside the circumcircle are taken (Delaunay); otherwise every
/// convex interior edge is taken (used to scramble a mesh into a seed).
fn collect_flips(tris: &[[u32; 3]], v: &[[i64; 2]], require_illegal: bool) -> Vec<Flip> {
    let mut edges: HashMap<(u32, u32), Vec<(usize, u32)>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        let [x, y, z] = *t;
        for &(a, b, apex) in &[(x, y, z), (y, z, x), (z, x, y)] {
            let key = if a < b { (a, b) } else { (b, a) };
            edges.entry(key).or_default().push((ti, apex));
        }
    }

    let mut keys: Vec<(u32, u32)> = edges.keys().copied().collect();
    keys.sort_unstable(); // deterministic independent-set selection

    let mut claimed = vec![false; tris.len()];
    let mut flips = Vec::new();
    for key in keys {
        let e = &edges[&key];
        if e.len() != 2 {
            continue; // hull edge (1) — never flipped
        }
        let (t0, p) = e[0];
        let (t1, q) = e[1];
        let (a, b) = key;

        // Flip is valid iff p-q crosses a-b (the quad is convex).
        let s1 = orient(v[p as usize], v[q as usize], v[a as usize]);
        let s2 = orient(v[p as usize], v[q as usize], v[b as usize]);
        if s1 == 0 || s2 == 0 || (s1 < 0) == (s2 < 0) {
            continue;
        }

        if require_illegal {
            let t = tris[t0]; // CCW
            let inside = in_circle(
                v[t[0] as usize],
                v[t[1] as usize],
                v[t[2] as usize],
                v[q as usize],
            ) > 0;
            if !inside {
                continue;
            }
        }

        if !claimed[t0] && !claimed[t1] {
            claimed[t0] = true;
            claimed[t1] = true;
            flips.push(Flip { t0, t1, a, b, p, q });
        }
    }
    flips
}

fn apply_flips(tris: &mut [[u32; 3]], flips: &[Flip], v: &[[i64; 2]]) {
    for f in flips {
        // The four surrounding edges (a-p, a-q, b-p, b-q) are unchanged; only the
        // diagonal a-b becomes p-q. New triangles: (a,p,q) and (b,p,q).
        tris[f.t0] = ccw3(f.a, f.p, f.q, v);
        tris[f.t1] = ccw3(f.b, f.p, f.q, v);
    }
}

/// Drive any valid triangulation of `points` to Delaunay by parallel-round
/// Lawson flipping. Returns `(triangles, rounds)`.
pub fn flip_to_delaunay(mut tris: Vec<[u32; 3]>, points: &[[i32; 2]]) -> (Vec<[u32; 3]>, usize) {
    let v = coords(points);
    let mut rounds = 0usize;
    // Bound: each round makes ≥1 progress; O(n^2) edges is the hard ceiling.
    let cap = 4 * points.len() * points.len() + 64;
    loop {
        let flips = collect_flips(&tris, &v, true);
        if flips.is_empty() {
            break;
        }
        apply_flips(&mut tris, &flips, &v);
        rounds += 1;
        if rounds > cap {
            panic!("flip_to_delaunay: exceeded round cap (non-convergence?)");
        }
    }
    (tris, rounds)
}

/// One round flipping every convex interior edge (ignoring legality). Turns a
/// Delaunay mesh into a valid **non-Delaunay** seed. Returns `(tris, n_flips)`.
pub fn flip_all_convex_once(
    mut tris: Vec<[u32; 3]>,
    points: &[[i32; 2]],
) -> (Vec<[u32; 3]>, usize) {
    let v = coords(points);
    let flips = collect_flips(&tris, &v, false);
    let n = flips.len();
    apply_flips(&mut tris, &flips, &v);
    (tris, n)
}

/// A **complete** valid triangulation of `points` (covering the convex hull,
/// hull triangles included), built by an incremental convex-hull sweep: sort by
/// (x,y); each new point is outside the current hull, so connect it to every
/// hull edge it sees. Generally not Delaunay — feed it to [`flip_to_delaunay`]
/// (or the GPU loop) to perfect it. This is the seed completion that "fixes the
/// hull" the Voronoi dual can't reach.
pub fn hull_seed(points: &[[i32; 2]]) -> Vec<[u32; 3]> {
    let n = points.len();
    if n < 3 {
        return Vec::new();
    }
    let _sp = std::env::var_os("GEO_SEED_PROF").is_some();
    let _t = std::time::Instant::now();
    let v = coords(points);
    if _sp {
        eprintln!(
            "  [seed] coords: {:.2} ms",
            _t.elapsed().as_secs_f64() * 1e3
        );
    }
    let _t = std::time::Instant::now();
    // Sort ids by (x,y). The old comparison sort was O(n log n) with a random
    // gather per compare (~40 ms at 1M — most of the seed cost); an LSD radix on a
    // monotone-packed u64 key is O(n) and cache-linear. Pack x,y (each i32, biased
    // by 2^31 to stay unsigned-monotone) into one u64 so byte order == (x,y) order.
    let key = |i: u32| -> u64 {
        let p = points[i as usize];
        (((p[0] as i64 + (1 << 31)) as u64) << 32) | ((p[1] as i64 + (1 << 31)) as u64)
    };
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut keys: Vec<u64> = order.iter().map(|&i| key(i)).collect();
    {
        const RADIX: usize = 1 << 11;
        const MASK: u64 = (RADIX as u64) - 1;
        let maxkey = keys.iter().copied().max().unwrap_or(0);
        let mut order2 = vec![0u32; n];
        let mut keys2 = vec![0u64; n];
        let mut shift = 0u32;
        while shift < 64 && (maxkey >> shift) != 0 {
            let mut cnt = [0u32; RADIX + 1];
            for &k in &keys {
                cnt[((k >> shift) & MASK) as usize + 1] += 1;
            }
            for i in 0..RADIX {
                cnt[i + 1] += cnt[i];
            }
            for i in 0..n {
                let d = ((keys[i] >> shift) & MASK) as usize;
                let p = cnt[d] as usize;
                cnt[d] += 1;
                order2[p] = order[i];
                keys2[p] = keys[i];
            }
            std::mem::swap(&mut order, &mut order2);
            std::mem::swap(&mut keys, &mut keys2);
            shift += 11;
        }
    }
    // Duplicates are now adjacent (equal packed key ⇒ identical point) — dedup them.
    order.dedup_by(|a, b| v[*a as usize] == v[*b as usize]);
    if _sp {
        eprintln!(
            "  [seed] radix sort+dedup: {:.2} ms",
            _t.elapsed().as_secs_f64() * 1e3
        );
    }
    let _t = std::time::Instant::now();
    if order.len() < 3 {
        return Vec::new();
    }

    // First non-collinear prefix: fan the collinear run order[0..k] to order[k].
    let mut k = 2;
    while k < order.len()
        && orient(
            v[order[0] as usize],
            v[order[1] as usize],
            v[order[k] as usize],
        ) == 0
    {
        k += 1;
    }
    if k == order.len() {
        return Vec::new(); // all collinear
    }
    let apex = order[k];
    // A triangulation of m points has ≤ 2m triangles — pre-size to avoid ~2n reallocs
    // (each grow copies the whole [u32;3] buffer; ~15 ms at 1M otherwise).
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(2 * order.len());
    for t in 0..k - 1 {
        tris.push(ccw3(order[t], order[t + 1], apex, &v));
    }
    // Hull as a CCW doubly-linked ring over vertex ids (`next`/`prev` indexed by id).
    // The old version stored the hull as a Vec and, for every inserted point, did a
    // FULL O(m) visibility rescan + O(m) rebuild — O(n·m) i128 `orient`s (≈180 ms at
    // 1M, worse than the whole Delaunay). The ring updates the visible arc in O(1)
    // amortised: each point starts the search at the previous insertion (`last`, which
    // is almost always on the visible/right side since points arrive in x-order), then
    // the interior of the visible arc is spliced out — every vertex is linked once and
    // unlinked at most once, so total work is O(n) beyond the sort.
    let mut next = vec![0u32; n];
    let mut prev = vec![0u32; n];
    {
        let mut ring: Vec<u32> = Vec::with_capacity(k + 1);
        if orient(
            v[order[0] as usize],
            v[order[k - 1] as usize],
            v[apex as usize],
        ) > 0
        {
            ring.extend_from_slice(&order[0..k]); // 0..k-1 then apex
            ring.push(apex);
        } else {
            for t in (0..k).rev() {
                ring.push(order[t]);
            }
            ring.push(apex);
        }
        let h = ring.len();
        for i in 0..h {
            let a = ring[i] as usize;
            next[a] = ring[(i + 1) % h];
            prev[ring[(i + 1) % h] as usize] = ring[i];
        }
    }
    let mut last = apex; // last-inserted vertex; always a live hull vertex

    // Exact i64 cross product for the sweep. hull_seed's inputs satisfy span ≤
    // MAX_COORDINATE_SPAN (≈1.94e9), so |orient| ≤ 2·(1.94e9)² ≈ 7.5e18 < i64::MAX —
    // the full-width i128 predicate is unnecessary here and ~2× more expensive.
    let ori = |a: [i64; 2], b: [i64; 2], c: [i64; 2]| -> i64 {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    for &p in &order[k + 1..] {
        let pu = p as usize;
        // Locate one visible edge (`s`, next[`s`]): CCW edge `a→b` is visible from p
        // iff p is to its right, ori(a,b,p) < 0. Fast path: one of `last`'s two
        // incident edges. Fallback (degenerate, e.g. x-ties): scan the ring once.
        let nl = next[last as usize];
        let pl = prev[last as usize];
        let s;
        if ori(v[last as usize], v[nl as usize], v[pu]) < 0 {
            s = last;
        } else if ori(v[pl as usize], v[last as usize], v[pu]) < 0 {
            s = pl;
        } else {
            let mut x = nl;
            let mut found = None;
            while x != last {
                if ori(v[x as usize], v[next[x as usize] as usize], v[pu]) < 0 {
                    found = Some(x);
                    break;
                }
                x = next[x as usize];
            }
            match found {
                Some(x) => s = x,
                None => continue, // p sees no edge (not strictly outside) — skip
            }
        }
        // Expand the (contiguous) visible arc: r forward, l backward, to the tangents.
        let mut r = s;
        while ori(v[r as usize], v[next[r as usize] as usize], v[pu]) < 0 {
            r = next[r as usize];
        }
        let mut l = s;
        while ori(v[prev[l as usize] as usize], v[l as usize], v[pu]) < 0 {
            l = prev[l as usize];
        }
        // One triangle per visible edge (l → … → r), then splice p between the tangents
        // (interior visible vertices fall out of the ring). Winding is known: the edge
        // x→nx is visible (orient(x,nx,p) < 0), so (x, p, nx) is strictly CCW — emit it
        // directly instead of calling ccw3 (which would recompute that orient: ~2n
        // saved i128 predicates, ~half the sweep at 1M).
        let mut x = l;
        while x != r {
            let nx = next[x as usize];
            tris.push([x, p, nx]);
            x = nx;
        }
        next[l as usize] = p;
        prev[pu] = l;
        next[pu] = r;
        prev[r as usize] = p;
        last = p;
    }
    if _sp {
        eprintln!(
            "  [seed] ring sweep: {:.2} ms ({} tris)",
            _t.elapsed().as_secs_f64() * 1e3,
            tris.len()
        );
    }
    tris
}

/// For each interior edge, the four points `[t0.0, t0.1, t0.2, q]` (CCW triangle
/// plus the opposite apex) so a batch `in_circle` marks the illegal edges — the
/// exact quantity the parallel round evaluates, packaged for the GPU kernel.
pub fn interior_quads(tris: &[[u32; 3]], points: &[[i32; 2]]) -> Vec<[[i32; 2]; 4]> {
    let mut edges: HashMap<(u32, u32), Vec<(usize, u32)>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        let [x, y, z] = *t;
        for &(a, b, apex) in &[(x, y, z), (y, z, x), (z, x, y)] {
            let key = if a < b { (a, b) } else { (b, a) };
            edges.entry(key).or_default().push((ti, apex));
        }
    }
    let mut keys: Vec<(u32, u32)> = edges.keys().copied().collect();
    keys.sort_unstable();
    let mut out = Vec::new();
    for key in keys {
        let e = &edges[&key];
        if e.len() != 2 {
            continue;
        }
        let (t0, _p) = e[0];
        let (_t1, q) = e[1];
        let t = tris[t0];
        out.push([
            points[t[0] as usize],
            points[t[1] as usize],
            points[t[2] as usize],
            points[q as usize],
        ]);
    }
    out
}
