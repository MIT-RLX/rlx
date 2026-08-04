// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Independent, exact (i128) validation of the FULL triangulation at scale — the
// definition of Delaunay, checked directly on the 1M output rather than trusted:
//   1. every triangle strictly CCW,
//   2. manifold: each undirected edge shared by exactly 1 (hull) or 2 triangles,
//   3. empty-circumcircle: for each interior edge, the opposite apex is NOT
//      strictly inside the other triangle's circumcircle,
//   4. serial and parallel agree on the triangle SET (not just the count).
//   cargo run -p rlx-geo --example validate_scale --release -- <pointfile>

use std::collections::HashMap;
use std::io::Read;

use rlx_geo::{triangulate, triangulate_par};

fn read_points(path: &str) -> Vec<[i32; 2]> {
    let mut buf = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let count = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
    let mut pts = Vec::with_capacity(count);
    let mut o = 8;
    for _ in 0..count {
        let x = i32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
        let y = i32::from_le_bytes(buf[o + 4..o + 8].try_into().unwrap());
        pts.push([x, y]);
        o += 8;
    }
    pts
}

fn orient(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i128 {
    (b[0] as i128 - a[0] as i128) * (c[1] as i128 - a[1] as i128)
        - (b[1] as i128 - a[1] as i128) * (c[0] as i128 - a[0] as i128)
}

// > 0 iff d strictly inside circumcircle of CCW triangle (a,b,c).
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

/// Returns (ccw_ok, manifold_ok, empty_circ_ok, illegal_count, cocircular_count).
fn validate(pts: &[[i32; 2]], tris: &[[u32; 3]]) -> (bool, bool, bool, usize, usize) {
    let mut ccw = true;
    // edge -> (apex, up to 2 records)
    let mut edges: HashMap<u64, Vec<(u32, u32, u32)>> = HashMap::with_capacity(tris.len() * 2);
    for t in tris {
        let (i0, i1, i2) = (t[0], t[1], t[2]);
        if orient(pts[i0 as usize], pts[i1 as usize], pts[i2 as usize]) <= 0 {
            ccw = false;
        }
        for &(a, b, opp) in &[(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)] {
            edges.entry(ekey(a, b)).or_default().push((a, b, opp));
        }
    }
    let mut manifold = true;
    let mut illegal = 0usize;
    let mut cocircular = 0usize;
    for recs in edges.values() {
        if recs.len() > 2 {
            manifold = false;
            continue;
        }
        if recs.len() == 2 {
            // Each triangle's apex must not be strictly inside the other's circle.
            let (a0, b0, p) = recs[0];
            let q = recs[1].2;
            // Triangle (a0,b0,p) oriented CCW for the predicate.
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
            if v > 0 {
                illegal += 1;
            } else if v == 0 {
                cocircular += 1; // legal (on the circle) — a degenerate tie
            }
        }
    }
    (ccw, manifold, illegal == 0, illegal, cocircular)
}

fn canon(mut t: [u32; 3]) -> [u32; 3] {
    t.sort_unstable();
    t
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: validate_scale <pointfile>");
    let pts = read_points(path.as_str());
    println!("points: {}", pts.len());

    let serial = triangulate(&pts).unwrap();
    let parallel = triangulate_par(&pts, 0).unwrap();
    println!("serial tris:   {}", serial.len());
    println!("parallel tris: {}", parallel.len());

    for (name, tris) in [("serial", &serial), ("parallel", &parallel)] {
        let (ccw, manifold, empty, illegal, cocirc) = validate(&pts, tris);
        println!(
            "[{name}] CCW={ccw} manifold={manifold} empty_circumcircle={empty} \
             (illegal={illegal}, cocircular_ties={cocirc})"
        );
        assert!(
            ccw && manifold && empty,
            "{name}: NOT a valid Delaunay triangulation"
        );
    }

    // Exact set comparison serial vs parallel.
    let ss: std::collections::HashSet<[u32; 3]> = serial.iter().map(|&t| canon(t)).collect();
    let ps: std::collections::HashSet<[u32; 3]> = parallel.iter().map(|&t| canon(t)).collect();
    let only_serial = ss.difference(&ps).count();
    let only_par = ps.difference(&ss).count();
    println!(
        "serial vs parallel: identical={}  (only_serial={only_serial}, only_parallel={only_par})",
        ss == ps
    );
    println!("RESULT: both outputs are exact, valid Delaunay triangulations.");
}
