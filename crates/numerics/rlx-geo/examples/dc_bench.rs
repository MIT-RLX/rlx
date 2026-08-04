// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Full recursive-tile GPU D&C pipeline, one-merge-pass form: x-sort → strips → leaf
// Delaunay per strip → add ALL adjacent-strip gaps (valid: strips are x-disjoint) →
// ONE flip. Because each strip is already Delaunay, only the seams are non-Delaunay,
// so the flip is NEAR-Delaunay → converges in a handful of rounds instead of ~54.
// Measures the flip's round count + time vs the flat hull_seed flip.
//   cargo run -p rlx-geo --example dc_bench --no-default-features --features gpu --release -- <file>

use rlx_geo::flip_gpu::{FlipPipeline, flip_to_delaunay_gpu_with};
use rlx_geo::{hull_seed, triangulate, triangulate_par};
use std::collections::HashMap;
use std::io::Read;
use std::time::Instant;

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
fn hull(pts: &[[i32; 2]], ids: &[u32]) -> Vec<u32> {
    let mut o = ids.to_vec();
    o.sort_by_key(|&i| (pts[i as usize][0], pts[i as usize][1]));
    o.dedup_by_key(|&mut i| pts[i as usize]);
    if o.len() < 3 {
        return o;
    }
    let cr = |a: u32, b: u32, c: u32| orient(pts[a as usize], pts[b as usize], pts[c as usize]);
    let mut lo: Vec<u32> = vec![];
    for &p in &o {
        while lo.len() >= 2 && cr(lo[lo.len() - 2], lo[lo.len() - 1], p) <= 0 {
            lo.pop();
        }
        lo.push(p);
    }
    let mut up: Vec<u32> = vec![];
    for &p in o.iter().rev() {
        while up.len() >= 2 && cr(up[up.len() - 2], up[up.len() - 1], p) <= 0 {
            up.pop();
        }
        up.push(p);
    }
    lo.pop();
    up.pop();
    lo.extend(up);
    lo
}
// gap triangles between two x-disjoint hulls (L left of R)
fn gap_tris(pts: &[[i32; 2]], lh: &[u32], rh: &[u32]) -> Vec<[u32; 3]> {
    let (lm, rm) = (lh.len(), rh.len());
    if lm < 1 || rm < 1 {
        return vec![];
    }
    let allp: Vec<u32> = lh.iter().chain(rh.iter()).copied().collect();
    let find = |lower: bool| -> (usize, usize) {
        for i in 0..lm {
            for j in 0..rm {
                if allp.iter().all(|&p| {
                    let o = orient(pts[lh[i] as usize], pts[rh[j] as usize], pts[p as usize]);
                    if lower { o >= 0 } else { o <= 0 }
                }) {
                    return (i, j);
                }
            }
        }
        (0, 0)
    };
    let (lo_l, lo_r) = find(true);
    let (hi_l, hi_r) = find(false);
    let mut gap = vec![];
    let (mut cl, mut cr) = (lo_l, lo_r);
    let mut guard = 0;
    while (cl != hi_l || cr != hi_r) && guard < lm + rm + 4 {
        guard += 1;
        let nl = (cl + 1) % lm;
        let nr = (cr + rm - 1) % rm;
        let can_l = cl != hi_l;
        let can_r = cr != hi_r;
        let take_l = can_l && (!can_r || pts[lh[nl] as usize][1] <= pts[rh[nr] as usize][1]);
        if take_l {
            gap.push([lh[cl], lh[nl], rh[cr]]);
            cl = nl;
        } else {
            gap.push([lh[cl], rh[nr], rh[cr]]);
            cr = nr;
        }
    }
    gap
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pts = read_points(&args[1]);
    let n = pts.len();
    let m = 128usize; // strip size

    let instance = wgpu::Instance::default();
    let all = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    let adapter = match std::env::var("GEO_GPU_ADAPTER") {
        Ok(w) => all
            .into_iter()
            .find(|a| a.get_info().name.to_lowercase().contains(&w.to_lowercase()))
            .unwrap(),
        Err(_) => {
            let pick = |t: wgpu::DeviceType| all.iter().position(|a| a.get_info().device_type == t);
            let idx = pick(wgpu::DeviceType::DiscreteGpu)
                .or_else(|| pick(wgpu::DeviceType::IntegratedGpu))
                .unwrap_or(0);
            all.into_iter().nth(idx).unwrap()
        }
    };
    let info = adapter.get_info();
    let feats = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: feats,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
    let pl = FlipPipeline::new(&device);

    // --- build the D&C near-Delaunay seed (strips Delaunay + gaps) ---
    let t_seed = Instant::now();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&i| (pts[i as usize][0], pts[i as usize][1]));
    let nstrip = n.div_ceil(m);
    let mut seed: Vec<[u32; 3]> = Vec::with_capacity(2 * n);
    let mut hulls: Vec<Vec<u32>> = Vec::with_capacity(nstrip);
    for s in 0..nstrip {
        let ids = &order[s * m..((s + 1) * m).min(n)];
        let sub: Vec<[i32; 2]> = ids.iter().map(|&i| pts[i as usize]).collect();
        for t in triangulate(&sub).unwrap() {
            seed.push([ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize]]);
        }
        hulls.push(hull(&pts, ids));
    }
    for s in 0..nstrip - 1 {
        seed.extend(gap_tris(&pts, &hulls[s], &hulls[s + 1]));
    }
    for t in seed.iter_mut() {
        if orient(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]) < 0 {
            t.swap(1, 2);
        }
    }
    let seed_ms = t_seed.elapsed().as_secs_f64() * 1e3;

    // --- flip the near-Delaunay D&C seed vs the far hull_seed ---
    let refc = triangulate(&pts).unwrap().len();
    let hseed = hull_seed(&pts);

    let bench = |s: &[[u32; 3]], tag: &str| -> f64 {
        for _ in 0..3 {
            flip_to_delaunay_gpu_with(&device, &queue, &pl, s, &pts);
        }
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let t = Instant::now();
            let out =
                std::hint::black_box(flip_to_delaunay_gpu_with(&device, &queue, &pl, s, &pts));
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(out.len(), refc, "{tag} wrong count");
        }
        best
    };

    println!("device: {} ({:?})", info.name, info.backend);
    println!(
        "n={n}  strips={nstrip}(m={m})  seed_tris={}  ref={refc}",
        seed.len()
    );
    // round counts
    unsafe {
        std::env::set_var("GEO_FLIP_DEBUG", "1");
    }
    eprint!("D&C seed:   ");
    let _ = flip_to_delaunay_gpu_with(&device, &queue, &pl, &seed, &pts);
    eprint!("hull_seed:  ");
    let _ = flip_to_delaunay_gpu_with(&device, &queue, &pl, &hseed, &pts);
    unsafe {
        std::env::remove_var("GEO_FLIP_DEBUG");
    }

    let dc_flip = bench(&seed, "dc");
    let flat_flip = bench(&hseed, "flat");
    let cpu = {
        let mut c = f64::INFINITY;
        for _ in 0..15 {
            let t = Instant::now();
            std::hint::black_box(triangulate_par(&pts, 0).unwrap());
            c = c.min(t.elapsed().as_secs_f64() * 1e3);
        }
        c
    };
    println!(
        "D&C seed build (CPU): {seed_ms:.2} ms   |   D&C flip: {dc_flip:.3} ms   flat flip: {flat_flip:.3} ms  → flip {:.2}x faster",
        flat_flip / dc_flip
    );
    println!("CPU parallel: {cpu:.3} ms");

    // validate D&C exact
    let out = flip_to_delaunay_gpu_with(&device, &queue, &pl, &seed, &pts);
    let mut edges: HashMap<u64, Vec<(u32, u32, u32)>> = HashMap::new();
    let mut ok = out.len() == refc;
    for t in &out {
        if orient(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]) <= 0 {
            ok = false;
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
    for recs in edges.values() {
        if recs.len() > 2 {
            ok = false;
        }
    }
    println!(
        "validate D&C pipeline: {}",
        if ok { "EXACT Delaunay OK" } else { "FAIL" }
    );
}
