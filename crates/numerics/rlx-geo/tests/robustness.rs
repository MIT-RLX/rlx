// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Adversarial robustness: exact-Delaunay validation (CCW + manifold +
// empty-circumcircle, all in i128) on degenerate/hard configurations that stress
// the exact predicates — regular grids and circles (dense cocircular ties),
// collinear runs, tight clusters, duplicate floods, near-max spans, and tiny n.
// Both the serial and parallel paths must produce an exact, valid Delaunay of the
// same size on every one. Runs under `cargo test -p rlx-geo --no-default-features`.

use std::collections::HashMap;

use rlx_geo::{triangulate, triangulate_par};

fn orient(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i128 {
    (b[0] as i128 - a[0] as i128) * (c[1] as i128 - a[1] as i128)
        - (b[1] as i128 - a[1] as i128) * (c[0] as i128 - a[0] as i128)
}

// > 0 iff d is strictly inside the circumcircle of CCW triangle (a,b,c).
fn in_circle(a: [i32; 2], b: [i32; 2], c: [i32; 2], d: [i32; 2]) -> i128 {
    let ax = a[0] as i128 - d[0] as i128;
    let ay = a[1] as i128 - d[1] as i128;
    let bx = b[0] as i128 - d[0] as i128;
    let by = b[1] as i128 - d[1] as i128;
    let cx = c[0] as i128 - d[0] as i128;
    let cy = c[1] as i128 - d[1] as i128;
    (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay)
}

fn ekey(a: u32, b: u32) -> u64 {
    let (a, b) = if a < b { (a, b) } else { (b, a) };
    ((a as u64) << 32) | b as u64
}

/// Assert `tris` is an exact, valid Delaunay triangulation of `pts` (empty is
/// vacuously valid — collinear/degenerate input yields no triangles).
fn assert_valid_delaunay(name: &str, pts: &[[i32; 2]], tris: &[[u32; 3]]) {
    if tris.is_empty() {
        return;
    }
    let mut edges: HashMap<u64, Vec<(u32, u32, u32)>> = HashMap::with_capacity(tris.len() * 2);
    for t in tris {
        let (i0, i1, i2) = (t[0], t[1], t[2]);
        assert!(
            orient(pts[i0 as usize], pts[i1 as usize], pts[i2 as usize]) > 0,
            "{name}: triangle {t:?} not strictly CCW"
        );
        for &(a, b, opp) in &[(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)] {
            edges.entry(ekey(a, b)).or_default().push((a, b, opp));
        }
    }
    for recs in edges.values() {
        assert!(
            recs.len() <= 2,
            "{name}: non-manifold edge (shared by >2 tris)"
        );
        if recs.len() == 2 {
            let (a0, b0, p) = recs[0];
            let q = recs[1].2;
            let tri = if orient(pts[a0 as usize], pts[b0 as usize], pts[p as usize]) > 0 {
                [a0, b0, p]
            } else {
                [a0, p, b0]
            };
            let v = in_circle(
                pts[tri[0] as usize],
                pts[tri[1] as usize],
                pts[tri[2] as usize],
                pts[q as usize],
            );
            assert!(
                v <= 0,
                "{name}: illegal edge — apex strictly inside circumcircle"
            );
        }
    }
}

/// Validate both backends and check the triangle-count invariant between them.
fn check(name: &str, pts: &[[i32; 2]]) {
    let serial = triangulate(pts).unwrap();
    let parallel = triangulate_par(pts, 0).unwrap();
    assert_valid_delaunay(&format!("{name}/serial"), pts, &serial);
    assert_valid_delaunay(&format!("{name}/parallel"), pts, &parallel);
    assert_eq!(
        serial.len(),
        parallel.len(),
        "{name}: serial/parallel triangle count differ"
    );
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
}

/// `n` distinct points in `[0, span)²`. Panics (not hangs) if the grid can't hold
/// `n` distinct points — a bounded attempt budget guards against coupon-collector
/// stalls when a generator asks for more distinct points than the space contains.
fn distinct(rng: &mut Lcg, n: usize, span: i64) -> Vec<[i32; 2]> {
    assert!(
        (span * span) as usize >= n * 2,
        "span too small for {n} distinct"
    );
    let mut seen = std::collections::HashSet::new();
    let mut pts = Vec::with_capacity(n);
    // Use the HIGH bits (`>> 33`): an LCG's low bits have a short period, so a
    // power-of-2 modulus on them would starve the distinct set.
    let mut attempts = 0usize;
    while pts.len() < n {
        let p = [
            ((rng.next() >> 33) % span as u64) as i32,
            ((rng.next() >> 33) % span as u64) as i32,
        ];
        if seen.insert(p) {
            pts.push(p);
        }
        attempts += 1;
        assert!(attempts < n * 100, "distinct(): coupon-collector stall");
    }
    pts
}

#[test]
fn regular_grids() {
    // k×k integer grid: every unit cell is a cocircular square — maximal ties.
    for k in [2usize, 3, 5, 8, 16, 40] {
        let pts: Vec<[i32; 2]> = (0..k * k)
            .map(|i| [(i % k) as i32, (i / k) as i32])
            .collect();
        check(&format!("grid{k}x{k}"), &pts);
    }
}

#[test]
fn points_on_circle() {
    // Integer-ish points near a large circle: heavily cocircular.
    for m in [8usize, 32, 100, 500] {
        let r = 1_000_000.0f64;
        let pts: Vec<[i32; 2]> = (0..m)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / m as f64;
                [(r * a.cos()) as i32, (r * a.sin()) as i32]
            })
            .collect();
        // Dedup near-collisions the rounding may create.
        let mut seen = std::collections::HashSet::new();
        let pts: Vec<_> = pts.into_iter().filter(|p| seen.insert(*p)).collect();
        check(&format!("circle{m}"), &pts);
    }
}

#[test]
fn collinear_and_degenerate() {
    check("two", &[[0, 0], [10, 0]]); // < 3 → empty
    check("collinear", &[[0, 0], [1, 0], [2, 0], [3, 0], [100, 0]]);
    check("collinear_diag", &[[0, 0], [1, 1], [2, 2], [3, 3]]);
    check("tri", &[[0, 0], [10, 0], [5, 9]]);
    check(
        "collinear_plus_one",
        &[[0, 0], [1, 0], [2, 0], [3, 0], [1, 5]],
    );
    check("square", &[[0, 0], [10, 0], [10, 10], [0, 10]]);
    check(
        "square_center",
        &[[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]],
    );
}

#[test]
fn duplicate_flood() {
    // Many exact duplicates around a few distinct sites — dedup must collapse them.
    let base = [[0, 0], [1000, 0], [500, 800], [500, 300]];
    let mut pts = Vec::new();
    for _ in 0..200 {
        pts.extend_from_slice(&base);
    }
    check("dupflood", &pts);
}

#[test]
fn tight_cluster_plus_outliers() {
    // A dense small cluster plus a few far points — extreme span ratio within.
    let mut rng = Lcg(0x5eed_1234);
    // Dense-ish cluster in a tiny box, then three far outliers → extreme span ratio.
    let mut pts = distinct(&mut rng, 400, 256);
    pts.extend_from_slice(&[
        [-1_000_000, -1_000_000],
        [1_000_000, 1_000_000],
        [1_000_000, -1_000_000],
    ]);
    check("cluster_outliers", &pts);
}

#[test]
fn near_max_span_wide_path() {
    // Span near MAX_COORDINATE_SPAN exercises the i128 wide predicate at scale.
    // Span just under MAX_COORDINATE_SPAN (1.94e9) exercises the i128 wide predicate.
    let mut rng = Lcg(0xa11ce);
    let pts = distinct(&mut rng, 3000, 1_900_000_000);
    check("wide_span", &pts);
}

#[test]
fn tiny_n_exhaustive_ish() {
    // Small random distinct sets across n = 3..=40 and several seeds.
    for seed in 0..25u64 {
        let mut rng = Lcg(0xbeef ^ seed);
        for n in 3..=40usize {
            let mut seen = std::collections::HashSet::new();
            let mut pts = Vec::new();
            while pts.len() < n {
                let p = [(rng.next() % 50) as i32, (rng.next() % 50) as i32];
                if seen.insert(p) {
                    pts.push(p);
                }
            }
            check(&format!("tiny_n{n}_s{seed}"), &pts);
        }
    }
}
