// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Robustness + ablation harness for the on-device flip: exactness, determinism
// (run twice, compare), and timing across the {SoS × f32-filter} matrix on
// degeneracy-heavy inputs (grids and circles are maximally cocircular).
//   cargo run -p rlx-geo --example flip_robust \
//       --no-default-features --features gpu --release -- <grid|circle|rand> <param>

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use rlx_geo::{flip_gpu::flip_to_delaunay_gpu, hull_seed, triangulate};

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
fn tkey(t: [u32; 3]) -> [u32; 3] {
    let mut s = t;
    s.sort_unstable();
    s
}

fn valid(pts: &[[i32; 2]], tris: &[[u32; 3]], ref_count: usize) -> Result<(), String> {
    if tris.len() != ref_count {
        return Err(format!("count {} != {ref_count}", tris.len()));
    }
    let mut used = vec![false; pts.len()];
    let mut edges: HashMap<u64, Vec<(u32, u32, u32)>> = HashMap::new();
    for t in tris {
        for &i in t {
            used[i as usize] = true;
        }
        if orient(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]) <= 0 {
            return Err("not CCW".into());
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

fn gen_pts(kind: &str, param: usize) -> Vec<[i32; 2]> {
    match kind {
        "grid" => (0..param * param)
            .map(|i| [(i % param) as i32, (i / param) as i32])
            .collect(),
        "circle" => {
            let r = 1_000_000.0f64;
            let mut seen = HashSet::new();
            (0..param)
                .map(|i| {
                    let a = std::f64::consts::TAU * i as f64 / param as f64;
                    [(r * a.cos()) as i32, (r * a.sin()) as i32]
                })
                .filter(|p| seen.insert(*p))
                .collect()
        }
        _ => {
            let mut s = 0x9E3779B97F4A7C15u64;
            let mut next = || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 33) as u32
            };
            let mut seen = HashSet::new();
            let mut v = Vec::new();
            while v.len() < param {
                let p = [(next() % 1_000_000) as i32, (next() % 1_000_000) as i32];
                if seen.insert(p) {
                    v.push(p);
                }
            }
            v
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = args.get(1).map(|s| s.as_str()).unwrap_or("grid");
    let param: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);
    let runs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);
    let pts = gen_pts(kind, param);
    let n = pts.len();

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

    let ref_count = triangulate(&pts).unwrap().len();
    let seed = hull_seed(&pts);

    // CPU full Delaunay (the thing the GPU flip is racing) + the CPU seed the GPU
    // flip depends on — both timed best-of.
    let mut cpu_best = f64::INFINITY;
    let mut seed_best = f64::INFINITY;
    for _ in 0..runs {
        let t = Instant::now();
        std::hint::black_box(triangulate(&pts).unwrap());
        cpu_best = cpu_best.min(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        std::hint::black_box(hull_seed(&pts));
        seed_best = seed_best.min(t.elapsed().as_secs_f64() * 1e3);
    }

    println!("device: {} ({:?})", info.name, info.backend);
    println!("input: {kind}({param}) → n={n}, ref_tris={ref_count}");
    println!(
        "CPU full Delaunay: {cpu_best:.3} ms   (CPU hull_seed the GPU needs: {seed_best:.3} ms)\n"
    );
    println!(
        "{:<22} {:>7} {:>7} {:>10} {:>9}",
        "config", "exact", "determ", "best_ms", "vs_base"
    );

    // canonical baseline set (filter on, SoS off) for "changed?" comparison
    let mut base_set: Option<HashSet<[u32; 3]>> = None;

    for &(sos, filt) in &[(false, true), (true, true), (false, false), (true, false)] {
        unsafe {
            if sos {
                std::env::set_var("GEO_FLIP_SOS", "1");
            } else {
                std::env::remove_var("GEO_FLIP_SOS");
            }
            if filt {
                std::env::remove_var("GEO_FLIP_NOFILTER");
            } else {
                std::env::set_var("GEO_FLIP_NOFILTER", "1");
            }
        }
        let run = || flip_to_delaunay_gpu(&device, &queue, &seed.clone(), &pts);
        let out1 = run();
        let out2 = run();
        let s1: HashSet<[u32; 3]> = out1.iter().map(|&t| tkey(t)).collect();
        let s2: HashSet<[u32; 3]> = out2.iter().map(|&t| tkey(t)).collect();
        let determ = s1 == s2;
        let exact = valid(&pts, &out1, ref_count).is_ok();
        let mut best = f64::INFINITY;
        for _ in 0..runs {
            let s = seed.clone();
            let t = Instant::now();
            std::hint::black_box(flip_to_delaunay_gpu(&device, &queue, &s, &pts));
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let label = format!("SoS={} filter={}", sos as u8, filt as u8);
        let vs = match &base_set {
            None => {
                base_set = Some(s1.clone());
                "(baseline)".to_string()
            }
            Some(b) => {
                if &s1 == b {
                    "same".into()
                } else {
                    format!("changed {}", s1.symmetric_difference(b).count())
                }
            }
        };
        println!(
            "{:<22} {:>7} {:>7} {:>10.3} {:>9}",
            label,
            if exact { "OK" } else { "FAIL" },
            if determ { "yes" } else { "NO" },
            best,
            vs
        );
    }
    unsafe {
        std::env::remove_var("GEO_FLIP_SOS");
        std::env::remove_var("GEO_FLIP_NOFILTER");
    }
}
