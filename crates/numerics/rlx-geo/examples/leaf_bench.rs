// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Validates the leaf cooperative shared-memory Delaunay solver: build a seed for a
// tile of ~128 points on the host, flip it to Delaunay entirely on-chip (one
// workgroup), and check the result is the exact Delaunay of those points.
//   cargo run -p rlx-geo --example leaf_bench --no-default-features --features gpu --release

use rlx_geo::leaf_gpu::{TILE, leaf_delaunay, leaf_throughput};
use rlx_geo::{hull_seed, triangulate};
use std::collections::HashMap;

fn orient(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i128 {
    (b[0] as i128 - a[0] as i128) * (c[1] as i128 - a[1] as i128)
        - (b[1] as i128 - a[1] as i128) * (c[0] as i128 - a[0] as i128)
}
fn in_circle(a: [i32; 2], b: [i32; 2], c: [i32; 2], d: [i32; 2]) -> i128 {
    let (ax, ay) = (a[0] as i128 - d[0] as i128, a[1] as i128 - d[1] as i128);
    let (bx, by) = (b[0] as i128 - d[0] as i128, b[1] as i128 - d[1] as i128);
    let (cx, cy) = (c[0] as i128 - d[0] as i128, c[1] as i128 - d[1] as i128);
    (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay)
}

fn build_twin(tris: &[[u32; 3]]) -> Vec<[u32; 3]> {
    let mut edge: HashMap<(u32, u32), Vec<(usize, usize)>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for e in 0..3 {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            let k = if a < b { (a, b) } else { (b, a) };
            edge.entry(k).or_default().push((ti, e));
        }
    }
    let mut tw = vec![[u32::MAX; 3]; tris.len()];
    for recs in edge.values() {
        if recs.len() == 2 {
            let (t0, e0) = recs[0];
            let (t1, e1) = recs[1];
            tw[t0][e0] = t1 as u32;
            tw[t1][e1] = t0 as u32;
        }
    }
    tw
}

fn validate(pts: &[[i32; 2]], tris: &[[u32; 3]], refc: usize) -> Result<(), String> {
    if tris.len() != refc {
        return Err(format!("count {} != ref {refc}", tris.len()));
    }
    let mut used = vec![false; pts.len()];
    let mut edges: HashMap<u64, Vec<(u32, u32, u32)>> = HashMap::new();
    for t in tris {
        for &i in t {
            used[i as usize] = true;
        }
        if orient(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]) <= 0 {
            return Err(format!("tri {t:?} not CCW"));
        }
        for &(a, b, o) in &[(t[0], t[1], t[2]), (t[1], t[2], t[0]), (t[2], t[0], t[1])] {
            let k = if a < b {
                ((a as u64) << 32) | b as u64
            } else {
                ((b as u64) << 32) | a as u64
            };
            edges.entry(k).or_default().push((a, b, o));
        }
    }
    if !used.iter().all(|&u| u) {
        return Err("missing points".into());
    }
    for recs in edges.values() {
        if recs.len() > 2 {
            return Err("non-manifold".into());
        }
        if recs.len() == 2 {
            let (a0, b0, p) = recs[0];
            let q = recs[1].2;
            let tri = if orient(pts[a0 as usize], pts[b0 as usize], pts[p as usize]) > 0 {
                [a0, b0, p]
            } else {
                [a0, p, b0]
            };
            if in_circle(
                pts[tri[0] as usize],
                pts[tri[1] as usize],
                pts[tri[2] as usize],
                pts[q as usize],
            ) > 0
            {
                return Err("illegal edge".into());
            }
        }
    }
    Ok(())
}

fn main() {
    // small-span point set (i64 predicate exact) of TILE points
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (s >> 33) as u32
    };
    let mut seen = std::collections::HashSet::new();
    let mut pts: Vec<[i32; 2]> = Vec::new();
    while pts.len() < TILE {
        let p = [(next() % 20000) as i32, (next() % 20000) as i32];
        if seen.insert(p) {
            pts.push(p);
        }
    }

    let instance = wgpu::Instance::default();
    let all = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    let adapter = match std::env::var("GEO_GPU_ADAPTER") {
        Ok(w) => all
            .into_iter()
            .find(|a| a.get_info().name.to_lowercase().contains(&w.to_lowercase()))
            .expect("no adapter"),
        Err(_) => {
            let pick = |t: wgpu::DeviceType| all.iter().position(|a| a.get_info().device_type == t);
            let idx = pick(wgpu::DeviceType::DiscreteGpu)
                .or_else(|| pick(wgpu::DeviceType::IntegratedGpu))
                .unwrap_or(0);
            all.into_iter().nth(idx).expect("no adapter")
        }
    };
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("no device");

    let seed = hull_seed(&pts);
    let seed_twin = build_twin(&seed);
    let refc = triangulate(&pts).unwrap().len();
    let out = leaf_delaunay(&device, &queue, &pts, &seed, &seed_twin);

    println!("device: {} ({:?})", info.name, info.backend);
    println!(
        "TILE={TILE} pts={} seed_tris={} ref_tris={refc} leaf_out={}",
        pts.len(),
        seed.len(),
        out.len()
    );
    match validate(&pts, &out, refc) {
        Ok(()) => println!("validate leaf: EXACT Delaunay OK (on-chip cooperative flip)"),
        Err(e) => {
            println!("validate leaf: FAIL: {e}");
            std::process::exit(1);
        }
    }

    // Throughput: process n_tiles·TILE points as parallel on-chip tiles.
    for &n_tiles in &[7813u32] {
        let total_pts = n_tiles as f64 * TILE as f64;
        let ms = leaf_throughput(&device, &queue, &pts, &seed, &seed_twin, n_tiles, 30);
        println!(
            "leaf phase: {n_tiles} tiles ({:.0}k pts) best {ms:.3} ms  → {:.1} M pts/s  (on-chip, no seam DRAM)",
            total_pts / 1000.0,
            total_pts / ms / 1000.0
        );
    }
}
