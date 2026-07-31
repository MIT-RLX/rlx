// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Discrete (grid) Voronoi diagram over integer sites.
//!
//! `voronoi_grid_exact` is the CPU reference: for each cell `(x, y)` it finds the
//! nearest site by exact squared-distance argmin (ties broken by lowest site
//! index) — O(width * height * n). The wgpu kernel computes the same thing, one
//! dispatch, in parallel. (A Jump-Flooding variant trades exactness for
//! O(log(max_dim)) passes; see the crate README.)

/// Extract Delaunay triangles from a Voronoi label grid (the dual). At each
/// interior grid vertex the 2x2 block of cell labels is inspected: exactly three
/// distinct site labels ⇒ those sites form a Delaunay triangle (its circumcenter
/// is that Voronoi vertex). Deduplicated by canonical (sorted) triple.
///
/// This is resolution-limited: it only recovers triangles whose Voronoi vertex
/// (circumcenter) falls inside the grid, and skips degree-4 (cocircular) vertices.
/// Use a grid that comfortably covers the sites' circumcenters.
pub fn voronoi_dual(labels: &[u32], width: u32, height: u32) -> Vec<[u32; 3]> {
    use std::collections::HashSet;
    let (w, h) = (width as usize, height as usize);
    let at = |x: usize, y: usize| labels[y * w + x];
    let mut seen: HashSet<[u32; 3]> = HashSet::new();
    let mut out = Vec::new();
    for y in 0..h.saturating_sub(1) {
        for x in 0..w.saturating_sub(1) {
            let mut d = [at(x, y), at(x + 1, y), at(x, y + 1), at(x + 1, y + 1)];
            d.sort_unstable();
            let mut uniq = [u32::MAX; 4];
            let mut k = 0;
            for &l in &d {
                if l != u32::MAX && (k == 0 || uniq[k - 1] != l) {
                    uniq[k] = l;
                    k += 1;
                }
            }
            if k == 3 {
                let tri = [uniq[0], uniq[1], uniq[2]];
                if seen.insert(tri) {
                    out.push(tri);
                }
            }
        }
    }
    out
}

/// Nearest-site label for every cell of a `width x height` grid, row-major
/// (`labels[y * width + x]`). Cell centers are integer coordinates `(x, y)`.
/// Sites are `[x, y]`. Empty `sites` yields an all-`u32::MAX` grid.
pub fn voronoi_grid_exact(sites: &[[i32; 2]], width: u32, height: u32) -> Vec<u32> {
    let (w, h) = (width as usize, height as usize);
    let mut labels = vec![u32::MAX; w * h];
    if sites.is_empty() {
        return labels;
    }
    for y in 0..h {
        for x in 0..w {
            let mut best = u32::MAX;
            let mut best_d: i64 = i64::MAX;
            for (i, s) in sites.iter().enumerate() {
                let dx = s[0] as i64 - x as i64;
                let dy = s[1] as i64 - y as i64;
                let d = dx * dx + dy * dy;
                if d < best_d {
                    best_d = d;
                    best = i as u32;
                }
            }
            labels[y * w + x] = best;
        }
    }
    labels
}
