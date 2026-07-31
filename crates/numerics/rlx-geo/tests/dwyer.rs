// Validates the Dwyer (Morton alternating-cut) build against the x-cut reference:
// same triangle count and a valid Delaunay mesh (empty-circumcircle, manifold, CCW).
use rlx_geo::{triangulate, triangulate_dwyer};
use std::collections::HashMap;

fn orient(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i128 {
    (b[0] as i128 - a[0] as i128) * (c[1] as i128 - a[1] as i128)
        - (b[1] as i128 - a[1] as i128) * (c[0] as i128 - a[0] as i128)
}
fn inside(a: [i32; 2], b: [i32; 2], c: [i32; 2], d: [i32; 2]) -> bool {
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
fn validate(p: &[[i32; 2]], t: &[[u32; 3]]) -> Result<(), String> {
    let mut e: HashMap<u64, (u32, u32, u32, u32)> = HashMap::new();
    let k = |a: u32, b: u32| {
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        ((a as u64) << 32) | b as u64
    };
    for tr in t {
        if orient(p[tr[0] as usize], p[tr[1] as usize], p[tr[2] as usize]) <= 0 {
            return Err("not CCW".into());
        }
        for &(a, b, o) in &[
            (tr[0], tr[1], tr[2]),
            (tr[1], tr[2], tr[0]),
            (tr[2], tr[0], tr[1]),
        ] {
            match e.get_mut(&k(a, b)) {
                None => {
                    e.insert(k(a, b), (a, b, o, 1));
                }
                Some(r) => {
                    if r.3 != 1 {
                        return Err("non-manifold".into());
                    }
                    r.3 = 2;
                    if inside(
                        p[r.0 as usize],
                        p[r.1 as usize],
                        p[r.2 as usize],
                        p[o as usize],
                    ) {
                        return Err("illegal edge".into());
                    }
                }
            }
        }
    }
    Ok(())
}
struct Lcg(u64);
impl Lcg {
    fn n(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn c(&mut self, m: i32) -> i32 {
        (self.n() % m as u64) as i32
    }
}
fn pts(r: &mut Lcg, n: usize, span: i32) -> Vec<[i32; 2]> {
    let mut s = std::collections::HashSet::new();
    let mut v = Vec::new();
    while v.len() < n {
        let p = [r.c(span), r.c(span)];
        if s.insert(p) {
            v.push(p);
        }
    }
    v
}
#[test]
fn dwyer_matches_reference() {
    let mut r = Lcg(0xd1e_5a1e);
    for _ in 0..40 {
        let n = 4 + (r.n() % 400) as usize;
        let p = pts(&mut r, n, 29_000);
        let a = triangulate(&p);
        let d = triangulate_dwyer(&p);
        validate(&p, &d).unwrap_or_else(|e| panic!("dwyer invalid ({} pts): {e}", p.len()));
        assert_eq!(
            a.len(),
            d.len(),
            "count differs at {} pts: ref {} dwyer {}",
            p.len(),
            a.len(),
            d.len()
        );
    }
}
#[test]
fn dwyer_big() {
    let mut r = Lcg(0xb16_d1e);
    let p = pts(&mut r, 50_000, 29_000);
    let d = triangulate_dwyer(&p);
    validate(&p, &d).expect("dwyer 50k invalid");
    assert_eq!(triangulate(&p).len(), d.len());
}
