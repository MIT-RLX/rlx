// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The flip loop must drive a scrambled (non-Delaunay) but valid triangulation
// back to Delaunay: seed = GS Delaunay, scrambled by one round of convex flips;
// then flip_to_delaunay must restore a valid Delaunay mesh (empty-circumcircle)
// with the same triangle count.

use std::collections::HashMap;

use rlx_geo::{flip_all_convex_once, flip_to_delaunay, hull_seed, triangulate, triangulate_par};

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

/// Valid Delaunay: manifold, CCW, empty-circumcircle everywhere.
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

/// Count locally-illegal interior edges (0 iff Delaunay).
fn illegal_count(points: &[[i32; 2]], tris: &[[u32; 3]]) -> usize {
    let mut edges: HashMap<u64, (u32, u32, u32, u32)> = HashMap::new();
    let mut bad = 0;
    for t in tris {
        for &(a, b, opp) in &[(t[0], t[1], t[2]), (t[1], t[2], t[0]), (t[2], t[0], t[1])] {
            let key = edge_key(a, b);
            match edges.get_mut(&key) {
                None => {
                    edges.insert(key, (a, b, opp, 1));
                }
                Some(rec) => {
                    if inside_circumcircle(
                        points[rec.0 as usize],
                        points[rec.1 as usize],
                        points[rec.2 as usize],
                        points[opp as usize],
                    ) {
                        bad += 1;
                    }
                }
            }
        }
    }
    bad
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

fn distinct_points(rng: &mut Lcg, n: usize, span: i32) -> Vec<[i32; 2]> {
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

fn run_case(pts: &[[i32; 2]]) {
    // Reference Delaunay via Guibas-Stolfi (indices 0..n since points are distinct).
    let delaunay = triangulate(pts).unwrap();
    assert!(!delaunay.is_empty());
    validate(pts, &delaunay).expect("reference GS mesh invalid");

    // Scramble into a valid, non-Delaunay seed.
    let (seed, scrambled) = flip_all_convex_once(delaunay.clone(), pts);
    assert_eq!(
        seed.len(),
        delaunay.len(),
        "scramble changed triangle count"
    );
    if scrambled == 0 {
        return; // no convex interior edges (tiny/degenerate) — nothing to test
    }
    let seed_bad = illegal_count(pts, &seed);
    assert!(seed_bad > 0, "seed should be non-Delaunay after scrambling");

    // Flip loop must restore Delaunay.
    let (out, rounds) = flip_to_delaunay(seed, pts);
    assert_eq!(out.len(), delaunay.len(), "flip changed triangle count");
    if let Err(e) = validate(pts, &out) {
        panic!("flipped mesh invalid after {rounds} rounds: {e}");
    }
    assert_eq!(
        illegal_count(pts, &out),
        0,
        "flip result still has illegal edges"
    );
}

#[test]
fn flip_restores_delaunay_small() {
    let mut rng = Lcg(0x1111_2222);
    for _ in 0..40 {
        let n = 8 + (rng.next() % 60) as usize;
        let pts = distinct_points(&mut rng, n, 4_000);
        run_case(&pts);
    }
}

#[test]
fn flip_restores_delaunay_wide_span() {
    let mut rng = Lcg(0x9999_abcd);
    for _ in 0..15 {
        let n = 40 + (rng.next() % 120) as usize;
        let pts = distinct_points(&mut rng, n, 100_000);
        run_case(&pts);
    }
}

#[test]
fn flip_bigger() {
    let mut rng = Lcg(0x5150_7ea1);
    let pts = distinct_points(&mut rng, 400, 50_000);
    run_case(&pts);
}

/// The parallel D&C must produce a valid Delaunay mesh with the same triangle
/// count as the serial path (above PARALLEL_MIN so it actually goes parallel).
#[test]
fn parallel_matches_serial() {
    let mut rng = Lcg(0x9a51_2340);
    let pts = distinct_points(&mut rng, 60_000, 29_000);
    let serial = triangulate(&pts).unwrap();
    for t in [2usize, 4, 8, 0] {
        let par = triangulate_par(&pts, t).unwrap();
        validate(&pts, &par).unwrap_or_else(|e| panic!("parallel (threads={t}) invalid: {e}"));
        assert_eq!(
            serial.len(),
            par.len(),
            "parallel (threads={t}) count differs"
        );
    }
}

fn used_all(points: &[[i32; 2]], tris: &[[u32; 3]]) -> bool {
    let mut used = vec![false; points.len()];
    for t in tris {
        for &i in t {
            used[i as usize] = true;
        }
    }
    used.iter().all(|&u| u)
}

/// The hull_seed -> flip pipeline must yield the *complete* Delaunay: valid,
/// empty-circumcircle, every point referenced (hull triangles included), and the
/// same triangle count as the reference. This is the fix for the dual's missing
/// hull triangles.
#[test]
fn hull_seed_completes_to_delaunay() {
    let mut rng = Lcg(0xc0ff_ee11);
    for _ in 0..30 {
        let n = 8 + (rng.next() % 120) as usize;
        let pts = distinct_points(&mut rng, n, 30_000);

        // The seed is a complete valid triangulation covering every point.
        let seed = hull_seed(&pts);
        assert!(used_all(&pts, &seed), "hull_seed dropped a point");
        // (seed is generally NOT Delaunay; validate manifold+CCW via edge counts)
        let reference = triangulate(&pts).unwrap();
        assert_eq!(
            seed.len(),
            reference.len(),
            "hull_seed wrong triangle count"
        );

        // Flip to Delaunay: complete, valid, all points present.
        let (out, _) = flip_to_delaunay(seed, &pts);
        validate(&pts, &out).expect("flipped hull_seed not Delaunay");
        assert_eq!(illegal_count(&pts, &out), 0);
        assert_eq!(out.len(), reference.len(), "incomplete: count != reference");
        assert!(used_all(&pts, &out), "incomplete: a point is missing");
    }
}
