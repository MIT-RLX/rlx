// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// One place to run — and rank — every Delaunay/predicate variation that lives in rlx-geo,
// so you can see which is fastest and (via the module docs) why.
//   cargo run -p rlx-geo --example bench_variants --release -- [N] [K]
//   N = full-triangulation point count (default 200000); K = in-circle tile size (default 48)
//
// Full-Delaunay GPU paths + the on-chip leaf live in their own benches (flip_gpu_bench,
// leaf_bench) since they need a wgpu device; this covers the CPU triangulators + the
// portable in-circle predicate variations.

use rlx_geo::delaunay32::{Point, Triangulator};
use rlx_geo::incircle_gemm::incircle_signs;
use rlx_geo::{triangulate, triangulate_par};
use std::time::Instant;

fn gen_points(n: usize, span: i32) -> Vec<[i32; 2]> {
    let mut s = 0x1234_5678u64;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (s >> 33) as i32
    };
    let mut seen = std::collections::HashSet::new();
    let mut v = vec![];
    while v.len() < n {
        let p = [rnd().rem_euclid(span), rnd().rem_euclid(span)];
        if seen.insert(p) {
            v.push(p);
        }
    }
    v
}
fn best<F: FnMut() -> usize>(mut f: F, runs: usize) -> (f64, usize) {
    let out = f();
    let mut b = f64::INFINITY;
    for _ in 0..runs {
        let t = Instant::now();
        let o = std::hint::black_box(f());
        b = b.min(t.elapsed().as_secs_f64() * 1e3);
        debug_assert_eq!(o, out);
    }
    (b, out)
}
// exact i128 in-circle (+1 inside circumcircle of CCW (a,b,c))
fn orient(a: [i64; 2], b: [i64; 2], c: [i64; 2]) -> i64 {
    let d = (b[0] - a[0]) as i128 * (c[1] - a[1]) as i128
        - (b[1] - a[1]) as i128 * (c[0] - a[0]) as i128;
    (d > 0) as i64 - (d < 0) as i64
}
fn ic(a: [i64; 2], b: [i64; 2], c: [i64; 2], d: [i64; 2]) -> i8 {
    let (ax, ay) = ((a[0] - d[0]) as i128, (a[1] - d[1]) as i128);
    let (bx, by) = ((b[0] - d[0]) as i128, (b[1] - d[1]) as i128);
    let (cx, cy) = ((c[0] - d[0]) as i128, (c[1] - d[1]) as i128);
    let det = (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay);
    (det > 0) as i8 - (det < 0) as i8
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let k: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(48);
    let span = 1_000_000i32;
    let pts = gen_points(n, span);
    let dpts: Vec<Point> = pts.iter().map(|p| Point::new(p[0], p[1])).collect();

    println!("== Full Delaunay (CPU), n={n} points, best-of-5 ==");
    let mut rows: Vec<(&str, f64, usize)> = vec![];
    let (t, c) = best(|| triangulate_par(&pts, 0).unwrap().len(), 5);
    rows.push(("rlx-geo triangulate_par (Dwyer, all cores)", t, c));
    let (t, c) = best(|| triangulate(&pts).unwrap().len(), 5);
    rows.push(("rlx-geo triangulate (Dwyer, serial)", t, c));
    let (t, c) = best(|| Triangulator::with_threads(0).triangulate(&dpts).len(), 5);
    rows.push(("delaunay32 Triangulator (GS D&C, parallel)", t, c));
    let (t, c) = best(|| Triangulator::new().triangulate(&dpts).len(), 5);
    rows.push(("delaunay32 Triangulator (GS D&C, serial)", t, c));
    rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (i, (name, ms, tris)) in rows.iter().enumerate() {
        println!(
            "  {}{:<44} {ms:8.2} ms   ({tris} tris)",
            if i == 0 { "★ " } else { "  " },
            name
        );
    }

    println!("\n== In-circle predicate (brute-force tile), K={k} points ==");
    let tp: Vec<[i32; 2]> = pts[..k].to_vec();
    let ci: Vec<[i64; 2]> = tp.iter().map(|p| [p[0] as i64, p[1] as i64]).collect();
    let mut tris = vec![];
    for x in 0..k as u32 {
        for y in x + 1..k as u32 {
            for z in y + 1..k as u32 {
                tris.push([x, y, z]);
            }
        }
    }
    let all: Vec<u32> = (0..k as u32).collect();
    let ntests = k * tris.len();

    // scalar i128 tight loop
    let scalar = |_: ()| -> Vec<i8> {
        let mut out = vec![0i8; all.len() * tris.len()];
        for (qi, &q) in all.iter().enumerate() {
            for (ti, &[a, b, c]) in tris.iter().enumerate() {
                let (a, b, c) = (a as usize, b as usize, c as usize);
                let (a, b, c) = if orient(ci[a], ci[b], ci[c]) < 0 {
                    (a, c, b)
                } else {
                    (a, b, c)
                };
                out[qi * tris.len() + ti] = ic(ci[a], ci[b], ci[c], ci[q as usize]);
            }
        }
        out
    };
    let s_ref = scalar(());
    let g_ref = incircle_signs(&tp, &tris, &all);
    assert_eq!(s_ref, g_ref, "GEMM predicate must equal scalar exact");

    let mut best_ms = f64::INFINITY;
    for _ in 0..10 {
        let t = Instant::now();
        std::hint::black_box(scalar(()));
        best_ms = best_ms.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let scalar_ms = best_ms;
    let mut best_ms = f64::INFINITY;
    for _ in 0..10 {
        let t = Instant::now();
        std::hint::black_box(incircle_signs(&tp, &tris, &all));
        best_ms = best_ms.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let gemm_ms = best_ms;
    let mut prows = [
        (
            "incircle_gemm (paraboloid-lift GEMM, f32+i128 fallback)",
            gemm_ms,
        ),
        ("scalar i128 tight loop", scalar_ms),
    ];
    prows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (i, (name, ms)) in prows.iter().enumerate() {
        println!(
            "  {}{:<52} {ms:7.3} ms   {:8.0} M tests/s",
            if i == 0 { "★ " } else { "  " },
            name,
            ntests as f64 / ms / 1000.0
        );
    }
    println!("  (both exact & identical; {ntests} tests. GEMM wins by being matmul-shaped →");
    println!("   auto-vectorizes / hits AMX; see incircle_gemm docs for the cross-backend table.)");
}
