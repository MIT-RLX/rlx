// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Topology helpers over a triangulation: per-triangle neighbor adjacency (the
//! "halfedge" structure most downstream algorithms want) and the convex hull.
//! Both use exact `i128` arithmetic and take/return indices into the caller's
//! `points`/`triangles`, so they compose with [`crate::triangulate`] without
//! exposing the internal dart arena.

use std::collections::HashMap;

/// Sentinel for "no neighbor" (a hull edge) in [`triangle_adjacency`].
pub const NO_NEIGHBOR: u32 = u32::MAX;

#[inline]
fn orient(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i128 {
    (b[0] as i128 - a[0] as i128) * (c[1] as i128 - a[1] as i128)
        - (b[1] as i128 - a[1] as i128) * (c[0] as i128 - a[0] as i128)
}

/// Per-triangle neighbor triangles. For triangle `t` with vertices
/// `[v0, v1, v2]`, `adj[t][0]` is the triangle sharing edge `(v0,v1)`,
/// `adj[t][1]` shares `(v1,v2)`, and `adj[t][2]` shares `(v2,v0)` — or
/// [`NO_NEIGHBOR`] if that edge is on the convex hull.
///
/// O(triangles) via an undirected-edge hash map; assumes a manifold mesh (each
/// edge borders at most two triangles), which every rlx-geo triangulation is.
pub fn triangle_adjacency(triangles: &[[u32; 3]]) -> Vec<[u32; 3]> {
    let mut adj = vec![[NO_NEIGHBOR; 3]; triangles.len()];
    // key -> (triangle index, edge slot 0..3) of the first triangle seen.
    let mut seen: HashMap<u64, (u32, u8)> = HashMap::with_capacity(triangles.len() * 3);
    for (ti, t) in triangles.iter().enumerate() {
        let edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];
        for (slot, &(a, b)) in edges.iter().enumerate() {
            let key = if a < b {
                ((a as u64) << 32) | b as u64
            } else {
                ((b as u64) << 32) | a as u64
            };
            match seen.remove(&key) {
                Some((oti, oslot)) => {
                    adj[ti][slot] = oti;
                    adj[oti as usize][oslot as usize] = ti as u32;
                }
                None => {
                    seen.insert(key, (ti as u32, slot as u8));
                }
            }
        }
    }
    adj
}

/// Convex hull of `points` as vertex indices in counterclockwise order
/// (Andrew's monotone chain, exact `i128` orientation). Collinear points on a
/// hull edge are dropped; coincident points collapse to their lowest index.
/// Degenerate input (fewer than 3 distinct, non-collinear points) returns the
/// distinct extreme points (0, 1, or 2 indices).
pub fn convex_hull(points: &[[i32; 2]]) -> Vec<u32> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    // Sort by (x, y, index) then drop coincident coordinates (keep lowest index).
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.sort_by(|&a, &b| {
        let (pa, pb) = (points[a as usize], points[b as usize]);
        (pa[0], pa[1], a).cmp(&(pb[0], pb[1], b))
    });
    idx.dedup_by(|&mut a, &mut b| points[a as usize] == points[b as usize]);
    let m = idx.len();
    if m < 3 {
        return idx;
    }
    let p = |i: u32| points[i as usize];
    let mut hull: Vec<u32> = Vec::with_capacity(2 * m);
    // Lower chain.
    for &i in &idx {
        while hull.len() >= 2 && orient(p(hull[hull.len() - 2]), p(hull[hull.len() - 1]), p(i)) <= 0
        {
            hull.pop();
        }
        hull.push(i);
    }
    // Upper chain (skip the last point, which is already the lower chain's end).
    let lower = hull.len() + 1;
    for &i in idx.iter().rev().skip(1) {
        while hull.len() >= lower
            && orient(p(hull[hull.len() - 2]), p(hull[hull.len() - 1]), p(i)) <= 0
        {
            hull.pop();
        }
        hull.push(i);
    }
    hull.pop(); // first point is repeated at the end
    if hull.len() < 3 {
        // All input was collinear — return the two extremes.
        return vec![idx[0], idx[m - 1]];
    }
    hull
}
