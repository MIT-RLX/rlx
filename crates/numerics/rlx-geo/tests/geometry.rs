// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Pure-geometry validation (no rlx deps): empty-circumcircle + manifold checks
// for Delaunay, and a Voronoi grid sanity check. Runs under
// `cargo test -p rlx-geo --no-default-features`.

use std::collections::HashMap;

use rlx_geo::{triangulate, voronoi_grid_exact};

fn orient(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i128 {
    (b[0] as i128 - a[0] as i128) * (c[1] as i128 - a[1] as i128)
        - (b[1] as i128 - a[1] as i128) * (c[0] as i128 - a[0] as i128)
}

fn inside_circumcircle(a: [i32; 2], b: [i32; 2], c: [i32; 2], d: [i32; 2]) -> bool {
    let ax = a[0] as i128 - d[0] as i128;
    let ay = a[1] as i128 - d[1] as i128;
    let bx = b[0] as i128 - d[0] as i128;
    let by = b[1] as i128 - d[1] as i128;
    let cx = c[0] as i128 - d[0] as i128;
    let cy = c[1] as i128 - d[1] as i128;
    let det = (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay);
    let w = orient(a, b, c);
    if w > 0 { det > 0 } else { det < 0 }
}

fn edge_key(a: u32, b: u32) -> u64 {
    let (a, b) = if a < b { (a, b) } else { (b, a) };
    ((a as u64) << 32) | b as u64
}

fn validate(points: &[[i32; 2]], tris: &[[u32; 3]]) -> Result<(), String> {
    let mut edges: HashMap<u64, (u32, u32, u32, u32)> = HashMap::new();
    for t in tris {
        let (i0, i1, i2) = (t[0], t[1], t[2]);
        if i0 == i1 || i1 == i2 || i2 == i0 {
            return Err("repeated vertex".into());
        }
        if orient(
            points[i0 as usize],
            points[i1 as usize],
            points[i2 as usize],
        ) <= 0
        {
            return Err("triangle not strictly CCW".into());
        }
        for &(a, b, opp) in &[(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)] {
            let key = edge_key(a, b);
            match edges.get_mut(&key) {
                None => {
                    edges.insert(key, (a, b, opp, 1));
                }
                Some(rec) => {
                    if rec.3 != 1 {
                        return Err("non-manifold edge".into());
                    }
                    rec.3 = 2;
                    if inside_circumcircle(
                        points[rec.0 as usize],
                        points[rec.1 as usize],
                        points[rec.2 as usize],
                        points[opp as usize],
                    ) {
                        return Err("locally illegal edge (empty-circle violated)".into());
                    }
                }
            }
        }
    }
    Ok(())
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
    fn range(&mut self, n: i32) -> i32 {
        (self.next() % n as u64) as i32
    }
}

fn all_collinear(points: &[[i32; 2]]) -> bool {
    let mut u = points.to_vec();
    u.sort_unstable();
    u.dedup();
    if u.len() < 3 {
        return true;
    }
    u[2..].iter().all(|&p| orient(u[0], u[1], p) == 0)
}

fn check(points: &[[i32; 2]]) {
    let tris = triangulate(points).unwrap();
    if all_collinear(points) {
        assert!(tris.is_empty(), "collinear input should yield no triangles");
        return;
    }
    if let Err(e) = validate(points, &tris) {
        panic!(
            "invalid mesh ({} pts, {} tris): {e}",
            points.len(),
            tris.len()
        );
    }
}

#[test]
fn triangle_square_collinear_dupes() {
    check(&[[0, 0], [100, 0], [50, 80]]);
    check(&[[0, 0], [100, 0], [100, 100], [0, 100]]);
    check(&[[0, 0], [10, 10], [20, 20], [30, 30]]); // collinear
    check(&[[0, 0], [0, 0], [100, 0], [50, 90], [50, 90]]); // dupes
}

#[test]
fn grid_max_degeneracy() {
    let mut pts = Vec::new();
    for x in 0..16 {
        for y in 0..16 {
            pts.push([x * 7, y * 7]);
        }
    }
    check(&pts);
}

#[test]
fn random_fast_and_wide_paths() {
    let mut rng = Lcg(0x1234_5678);
    for _ in 0..150 {
        let n = 3 + (rng.next() % 80) as usize;
        let pts: Vec<[i32; 2]> = (0..n)
            .map(|_| [rng.range(5_000), rng.range(5_000)])
            .collect();
        check(&pts); // fast (i64) path
    }
    let mut rng = Lcg(0xdead_beef);
    for _ in 0..40 {
        let n = 50 + (rng.next() % 300) as usize;
        let pts: Vec<[i32; 2]> = (0..n)
            .map(|_| [rng.range(100_000), rng.range(100_000)])
            .collect();
        check(&pts); // wide (i128) path
    }
}

#[test]
fn voronoi_grid_nearest() {
    // Two sites: left half should label 0, right half label 1.
    let sites = [[1, 4], [8, 4]];
    let (w, h) = (10u32, 8u32);
    let labels = voronoi_grid_exact(&sites, w, h);
    assert_eq!(labels.len(), (w * h) as usize);
    // A cell at x=0 is nearest site 0; a cell at x=9 nearest site 1.
    assert_eq!(labels[4 * w as usize], 0);
    assert_eq!(labels[4 * w as usize + 9], 1);
    // Every label is a valid site index.
    assert!(labels.iter().all(|&l| l == 0 || l == 1));
}

#[test]
fn voronoi_empty() {
    let labels = voronoi_grid_exact(&[], 4, 4);
    assert!(labels.iter().all(|&l| l == u32::MAX));
}
