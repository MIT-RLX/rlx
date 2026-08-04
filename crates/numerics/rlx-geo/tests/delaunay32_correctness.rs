// Validates the triangulation the same way the C++ reference validator does:
// manifold edges, no duplicate/degenerate triangles, every site referenced, and
// the empty-circumcircle (locally Delaunay) property on every shared edge.

use std::collections::HashMap;

use rlx_geo::delaunay32::{Point, Triangle, Triangulator};

// i128 exact predicates for validation (independent of the library's internals).
fn orient(a: Point, b: Point, c: Point) -> i128 {
    (b.x as i128 - a.x as i128) * (c.y as i128 - a.y as i128)
        - (b.y as i128 - a.y as i128) * (c.x as i128 - a.x as i128)
}

fn inside_circumcircle(a: Point, b: Point, c: Point, d: Point) -> bool {
    let ax = a.x as i128 - d.x as i128;
    let ay = a.y as i128 - d.y as i128;
    let bx = b.x as i128 - d.x as i128;
    let by = b.y as i128 - d.y as i128;
    let cx = c.x as i128 - d.x as i128;
    let cy = c.y as i128 - d.y as i128;
    let det = (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay);
    let w = orient(a, b, c);
    if w > 0 { det > 0 } else { det < 0 }
}

fn edge_key(a: u32, b: u32) -> u64 {
    let (a, b) = if a < b { (a, b) } else { (b, a) };
    ((a as u64) << 32) | b as u64
}

/// Returns Err(message) if the mesh is not a valid Delaunay triangulation.
fn validate(points: &[Point], tris: &[Triangle]) -> Result<(), String> {
    let mut used = vec![false; points.len()];
    // edge -> (a, b, opposite, count)
    let mut edges: HashMap<u64, (u32, u32, u32, u32)> = HashMap::new();

    for t in tris {
        let (i0, i1, i2) = (t.i0, t.i1, t.i2);
        for &i in &[i0, i1, i2] {
            if i as usize >= points.len() {
                return Err("index out of range".into());
            }
        }
        if i0 == i1 || i1 == i2 || i2 == i0 {
            return Err("triangle repeats a vertex".into());
        }
        if orient(
            points[i0 as usize],
            points[i1 as usize],
            points[i2 as usize],
        ) <= 0
        {
            return Err("triangle is not strictly CCW".into());
        }
        used[i0 as usize] = true;
        used[i1 as usize] = true;
        used[i2 as usize] = true;

        for &(a, b, opp) in &[(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)] {
            let key = edge_key(a, b);
            match edges.get_mut(&key) {
                None => {
                    edges.insert(key, (a, b, opp, 1));
                }
                Some(rec) => {
                    if rec.3 != 1 {
                        return Err("non-manifold edge (>2 triangles)".into());
                    }
                    rec.3 = 2;
                    // local Delaunay: neither opposite vertex inside the other's circle
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

    // Every input point that is not a duplicate must be referenced.
    // (Duplicates collapse, so only check that the *unique* set is covered.)
    let mut seen = std::collections::HashSet::new();
    for (i, p) in points.iter().enumerate() {
        if seen.insert((p.x, p.y)) && !used[i] {
            // Allow collinear inputs (no triangles) only when < 3 unique points
            // or all points collinear; the caller checks those separately.
            return Err(format!("unique point {i} not referenced"));
        }
    }
    Ok(())
}

// Small deterministic RNG so tests are reproducible without extra crates.
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

fn all_collinear(points: &[Point]) -> bool {
    let mut uniq: Vec<Point> = points.to_vec();
    uniq.sort_unstable_by_key(|p| (p.x, p.y));
    uniq.dedup();
    if uniq.len() < 3 {
        return true;
    }
    uniq[2..].iter().all(|&p| orient(uniq[0], uniq[1], p) == 0)
}

fn run_case(points: &[Point]) {
    let mut tri = Triangulator::new();
    let out = tri.triangulate(points);
    if all_collinear(points) {
        assert!(out.is_empty(), "collinear input should yield no triangles");
        return;
    }
    if let Err(e) = validate(points, &out) {
        panic!(
            "invalid mesh ({} pts, {} tris): {e}",
            points.len(),
            out.len()
        );
    }
}

#[test]
fn tiny_triangle() {
    run_case(&[Point::new(0, 0), Point::new(100, 0), Point::new(50, 80)]);
}

#[test]
fn square() {
    run_case(&[
        Point::new(0, 0),
        Point::new(100, 0),
        Point::new(100, 100),
        Point::new(0, 100),
    ]);
}

#[test]
fn collinear() {
    run_case(&[
        Point::new(0, 0),
        Point::new(10, 10),
        Point::new(20, 20),
        Point::new(30, 30),
    ]);
}

#[test]
fn duplicates() {
    run_case(&[
        Point::new(0, 0),
        Point::new(0, 0),
        Point::new(100, 0),
        Point::new(50, 90),
        Point::new(50, 90),
    ]);
}

#[test]
fn grid() {
    let mut pts = Vec::new();
    for x in 0..20 {
        for y in 0..20 {
            pts.push(Point::new(x * 7, y * 7));
        }
    }
    run_case(&pts);
}

#[test]
fn random_small_fast_path() {
    // span small -> exercises the i64 fast path
    let mut rng = Lcg(0x1234_5678);
    for _ in 0..200 {
        let n = 3 + (rng.next() % 60) as usize;
        let pts: Vec<Point> = (0..n)
            .map(|_| Point::new(rng.range(5000), rng.range(5000)))
            .collect();
        run_case(&pts);
    }
}

#[test]
fn random_medium_wide_path() {
    // span large -> exercises the i128 wide path
    let mut rng = Lcg(0xdead_beef);
    for _ in 0..40 {
        let n = 50 + (rng.next() % 400) as usize;
        let pts: Vec<Point> = (0..n)
            .map(|_| Point::new(rng.range(100_000), rng.range(100_000)))
            .collect();
        run_case(&pts);
    }
}

#[test]
fn random_large() {
    let mut rng = Lcg(0xa5a5_1111);
    let n = 20_000;
    let pts: Vec<Point> = (0..n)
        .map(|_| Point::new(rng.range(100_000), rng.range(100_000)))
        .collect();
    run_case(&pts);
}

// The parallel path (threads > 1) must produce a valid Delaunay mesh with the
// same triangle count as the serial path. (Exact triangle sets can legitimately
// differ on cocircular groups, where either diagonal is Delaunay-legal, so we
// check validity + count rather than set identity. A hole or merge bug would
// change the count.)
#[test]
fn parallel_matches_serial_and_validates() {
    let mut rng = Lcg(0x0dd_f00d);
    let n = 120_000; // > PARALLEL_MIN_POINTS (50k)
    let pts: Vec<Point> = (0..n)
        .map(|_| Point::new(rng.range(100_000), rng.range(100_000)))
        .collect();

    let serial = Triangulator::with_threads(1).triangulate(&pts);
    validate(&pts, &serial).expect("serial mesh invalid");
    // Try several worker counts, including odd/non-power-of-two.
    for t in [2usize, 3, 4, 8] {
        let par = Triangulator::with_threads(t).triangulate(&pts);
        if let Err(e) = validate(&pts, &par) {
            panic!("parallel mesh (threads={t}) invalid: {e}");
        }
        assert_eq!(
            serial.len(),
            par.len(),
            "parallel (threads={t}) triangle count differs from serial"
        );
    }
}

// Parallel path on the i64 fast path (small span) too — denser grid means more
// cocircular groups, a good stress test for the merge.
#[test]
fn parallel_fast_path() {
    let mut rng = Lcg(0x5eed_1234);
    let n = 80_000;
    let pts: Vec<Point> = (0..n)
        .map(|_| Point::new(rng.range(20_000), rng.range(20_000)))
        .collect();
    let serial = Triangulator::with_threads(1).triangulate(&pts);
    validate(&pts, &serial).expect("serial fast-path mesh invalid");
    let par = Triangulator::with_threads(6).triangulate(&pts);
    validate(&pts, &par).expect("parallel fast-path mesh invalid");
    assert_eq!(serial.len(), par.len());
}
