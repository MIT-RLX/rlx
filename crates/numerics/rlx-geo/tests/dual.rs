// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Measures how completely the Voronoi-dual extraction recovers the true Delaunay
// triangulation (precision/recall), and — when complete — that it forms a valid
// mesh usable as a flip seed.

use std::collections::HashSet;

use rlx_geo::{triangulate, voronoi_dual, voronoi_grid_exact};

fn canon(t: [u32; 3]) -> [u32; 3] {
    let mut v = t;
    v.sort_unstable();
    v
}
fn set(tris: &[[u32; 3]]) -> HashSet<[u32; 3]> {
    tris.iter().map(|&t| canon(t)).collect()
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
    fn c(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.n() % (hi - lo) as u64) as i32
    }
}

#[test]
fn dual_coverage() {
    // Well-separated points on a fine grid with margin so circumcenters stay in-grid.
    let (w, h) = (512u32, 512u32);
    let mut rng = Lcg(0xda1c_0de5u64);
    let mut seen = HashSet::new();
    let mut pts: Vec<[i32; 2]> = Vec::new();
    while pts.len() < 40 {
        let p = [rng.c(80, w as i32 - 80), rng.c(80, h as i32 - 80)];
        if seen.insert(p) {
            pts.push(p);
        }
    }

    let truth = set(&triangulate(&pts).unwrap());
    let labels = voronoi_grid_exact(&pts, w, h);
    let dual = voronoi_dual(&labels, w, h);
    let dset = set(&dual);

    let correct = dset.iter().filter(|t| truth.contains(*t)).count();
    let precision = correct as f64 / dset.len().max(1) as f64;
    let recall = correct as f64 / truth.len().max(1) as f64;

    println!(
        "dual coverage: {} extracted, {} true Delaunay, precision {:.3}, recall {:.3}",
        dset.len(),
        truth.len(),
        precision,
        recall
    );

    // Every extracted triangle must be a genuine Delaunay triangle.
    assert!(
        (precision - 1.0).abs() < 1e-9,
        "dual emitted a non-Delaunay triangle"
    );
    // Interior recall should be high (hull triangles have off-grid circumcenters).
    assert!(recall > 0.5, "dual recall unexpectedly low: {recall}");
}
