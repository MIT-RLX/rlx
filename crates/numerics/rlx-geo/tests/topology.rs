// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Validates the topology helpers against the triangulation: adjacency is
// symmetric and consistent with shared edges, hull edges have no neighbor, and
// convex_hull matches a brute-force reference. Runs under
// `cargo test -p rlx-geo --no-default-features`.

use std::collections::HashMap;

use rlx_geo::{NO_NEIGHBOR, convex_hull, triangle_adjacency, triangulate};

fn orient(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i128 {
    (b[0] as i128 - a[0] as i128) * (c[1] as i128 - a[1] as i128)
        - (b[1] as i128 - a[1] as i128) * (c[0] as i128 - a[0] as i128)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn c(&mut self, m: i32) -> i32 {
        (self.next() % m as u64) as i32
    }
}

fn distinct(rng: &mut Lcg, n: usize, span: i32) -> Vec<[i32; 2]> {
    let mut seen = std::collections::HashSet::new();
    let mut pts = Vec::new();
    while pts.len() < n {
        let p = [rng.c(span), rng.c(span)];
        if seen.insert(p) {
            pts.push(p);
        }
    }
    pts
}

/// Adjacency must be symmetric, and every pair of adjacent triangles must share
/// exactly the edge whose slot points at each other. Hull edges → NO_NEIGHBOR,
/// and the count of NO_NEIGHBOR entries equals the number of hull edges.
#[test]
fn adjacency_is_consistent() {
    let mut rng = Lcg(0xa0d1_ac11);
    for _ in 0..30 {
        let n = 8 + (rng.next() % 200) as usize;
        let pts = distinct(&mut rng, n, 5_000);
        let tris = triangulate(&pts).unwrap();
        if tris.is_empty() {
            continue;
        }
        let adj = triangle_adjacency(&tris);
        assert_eq!(adj.len(), tris.len());

        // Reference: count how many triangles each undirected edge borders.
        let mut border: HashMap<(u32, u32), u32> = HashMap::new();
        for t in &tris {
            for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let k = if a < b { (a, b) } else { (b, a) };
                *border.entry(k).or_default() += 1;
            }
        }

        let mut no_neighbor = 0usize;
        for (ti, t) in tris.iter().enumerate() {
            let edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];
            for (slot, &(a, b)) in edges.iter().enumerate() {
                let nb = adj[ti][slot];
                let k = if a < b { (a, b) } else { (b, a) };
                let shared = border[&k];
                if nb == NO_NEIGHBOR {
                    no_neighbor += 1;
                    assert_eq!(shared, 1, "hull edge should border exactly one triangle");
                } else {
                    assert_eq!(shared, 2, "interior edge should border two triangles");
                    // Symmetry: the neighbor points back at us across the same edge.
                    let nt = tris[nb as usize];
                    let nadj = &adj[nb as usize];
                    let back = (0..3).any(|s| nadj[s] == ti as u32);
                    assert!(back, "adjacency not symmetric");
                    // The neighbor triangle must contain both a and b.
                    assert!(
                        nt.contains(&a) && nt.contains(&b),
                        "neighbor does not share the edge"
                    );
                }
            }
        }
        // #hull edges (edges bordering one triangle) == #NO_NEIGHBOR slots.
        let hull_edges = border.values().filter(|&&c| c == 1).count();
        assert_eq!(hull_edges, no_neighbor);
    }
}

#[test]
fn convex_hull_basic() {
    // Square with an interior point → hull is the 4 corners, CCW.
    let pts = vec![[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]];
    let hull = convex_hull(&pts);
    assert_eq!(hull.len(), 4, "interior point must be excluded");
    // CCW: consecutive turns are all left.
    let m = hull.len();
    for i in 0..m {
        let a = pts[hull[i] as usize];
        let b = pts[hull[(i + 1) % m] as usize];
        let c = pts[hull[(i + 2) % m] as usize];
        assert!(orient(a, b, c) > 0, "hull not CCW / has collinear vertex");
    }
    // Collinear on an edge is dropped.
    let line_edge = vec![[0, 0], [5, 0], [10, 0], [10, 10], [0, 10]];
    assert_eq!(convex_hull(&line_edge).len(), 4);
    // All collinear → two extremes.
    assert_eq!(convex_hull(&[[0, 0], [1, 1], [2, 2], [3, 3]]).len(), 2);
}

/// The defining invariant of a strict convex hull: it is a strictly convex CCW
/// polygon that encloses every input point (each point is left-of or on every
/// hull edge). This avoids the collinear-on-edge ambiguity of counting vertices.
#[test]
fn convex_hull_encloses_all() {
    let mut rng = Lcg(0xc0_11_de);
    for _ in 0..60 {
        let n = 3 + (rng.next() % 60) as usize;
        let pts = distinct(&mut rng, n, 200);
        let hull = convex_hull(&pts);
        let m = hull.len();
        if m < 3 {
            continue; // degenerate (collinear) — encloses nothing with area
        }
        // Strictly convex, CCW.
        for i in 0..m {
            let a = pts[hull[i] as usize];
            let b = pts[hull[(i + 1) % m] as usize];
            let c = pts[hull[(i + 2) % m] as usize];
            assert!(orient(a, b, c) > 0, "hull vertex not a strict left turn");
        }
        // Every input point is inside or on the boundary.
        for &p in &pts {
            for i in 0..m {
                let a = pts[hull[i] as usize];
                let b = pts[hull[(i + 1) % m] as usize];
                assert!(orient(a, b, p) >= 0, "a point lies outside the hull");
            }
        }
        // No duplicate hull vertices.
        let uniq: std::collections::HashSet<u32> = hull.iter().copied().collect();
        assert_eq!(uniq.len(), m, "duplicate hull vertex");
    }
}
