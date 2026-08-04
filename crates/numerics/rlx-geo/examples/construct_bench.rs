// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// GPU-NATIVE Delaunay bench: builds the whole triangulation on-device (bounding
// box on the host only) via construct_gpu → flip → drop bounding triangles, then
// validates it is the exact Delaunay and times the full path vs the CPU seed and
// the parallel CPU triangulator.
//   cargo run -p rlx-geo --example construct_bench \
//       --no-default-features --features gpu --release -- <file> [runs]

use std::collections::HashMap;
use std::io::Read;
use std::time::Instant;

use rlx_geo::construct_gpu::{construct_gpu, delaunay_gpu_native};
use rlx_geo::flip_gpu::FlipPipeline;
use rlx_geo::gdel_gpu::delaunay_gpu_gdel;
use rlx_geo::{hull_seed, triangulate, triangulate_par};

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
fn in_circle(a: [i32; 2], b: [i32; 2], c: [i32; 2], d: [i32; 2]) -> i128 {
    let (ax, ay) = (a[0] as i128 - d[0] as i128, a[1] as i128 - d[1] as i128);
    let (bx, by) = (b[0] as i128 - d[0] as i128, b[1] as i128 - d[1] as i128);
    let (cx, cy) = (c[0] as i128 - d[0] as i128, c[1] as i128 - d[1] as i128);
    (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay)
}

fn validate(pts: &[[i32; 2]], tris: &[[u32; 3]], ref_count: usize) -> Result<(), String> {
    if tris.len() != ref_count {
        return Err(format!("count {} != reference {ref_count}", tris.len()));
    }
    let mut used = vec![false; pts.len()];
    let mut edges: HashMap<u64, Vec<(u32, u32, u32)>> = HashMap::with_capacity(tris.len() * 2);
    for t in tris {
        let (i0, i1, i2) = (t[0], t[1], t[2]);
        for &i in t {
            used[i as usize] = true;
        }
        if orient(pts[i0 as usize], pts[i1 as usize], pts[i2 as usize]) <= 0 {
            return Err(format!("triangle {t:?} not strictly CCW"));
        }
        for &(a, b, opp) in &[(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)] {
            let k = if a < b {
                ((a as u64) << 32) | b as u64
            } else {
                ((b as u64) << 32) | a as u64
            };
            edges.entry(k).or_default().push((a, b, opp));
        }
    }
    if !used.iter().all(|&u| u) {
        return Err("mesh does not use every input point".into());
    }
    for recs in edges.values() {
        if recs.len() > 2 {
            return Err("non-manifold edge (>2 triangles)".into());
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
                return Err("illegal edge — apex strictly inside circumcircle".into());
            }
        }
    }
    Ok(())
}

fn stats(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], v[v.len() / 2])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    let pts = read_points(path);
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
    let pl = FlipPipeline::new(&device);

    // Raw construction diagnostic: how many triangles touch a super vertex (id ≥ n)?
    if let Some((seed, _ext)) = construct_gpu(&device, &queue, &pts) {
        let ghosts = seed
            .iter()
            .filter(|t| t.iter().any(|&v| v >= n as u32))
            .count();
        let real_hull_est = 2 * n - 2 - ref_count; // Euler: hull edges of the real set
        println!(
            "raw construct: {} tris, {} ghost (super-incident), real hull ≈ {}",
            seed.len(),
            ghosts,
            real_hull_est
        );
    }

    // correctness
    let out = delaunay_gpu_native(&device, &queue, &pl, &pts);
    println!("device: {} ({:?})", info.name, info.backend);
    println!("n={n} ref_tris={ref_count} runs={runs}");
    match &out {
        None => {
            println!("construct_gpu: span too large → fell back (None). Skipping.");
            return;
        }
        Some(t) => {
            let v = validate(&pts, t, ref_count);
            println!(
                "validate GPU-native: {}",
                v.as_ref()
                    .map(|_| "EXACT Delaunay OK".to_string())
                    .unwrap_or_else(|e| format!("FAIL: {e}"))
            );
            // continue to timing even on deficit — we want the speed signal first
        }
    }

    // Interleaved gDel2D (exact, CPU-hull boundary) — validate + time.
    match delaunay_gpu_gdel(&device, &queue, &pts) {
        Some(g) => {
            let v = validate(&pts, &g, ref_count);
            println!(
                "validate gDel2D    : {}",
                v.as_ref()
                    .map(|_| "EXACT Delaunay OK".to_string())
                    .unwrap_or_else(|e| format!("FAIL: {e}"))
            );
            if v.is_err() && std::env::var_os("GEO_GDEL_DIAG").is_some() {
                use std::collections::HashMap;
                let mut seen: HashMap<[u32; 3], usize> = HashMap::new();
                for (ti, t) in g.iter().enumerate() {
                    let o = orient(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]);
                    if o <= 0 {
                        println!(
                            "  CW/deg tri#{ti} {t:?} orient={o} coords {:?} {:?} {:?}",
                            pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]
                        );
                    }
                    let mut k = *t;
                    k.sort_unstable();
                    *seen.entry(k).or_default() += 1;
                }
                let dups: usize = seen.values().filter(|&&c| c > 1).count();
                println!("  duplicate triangles: {dups}");
            }
            let mut gv = Vec::with_capacity(runs);
            for _ in 0..runs {
                let t = Instant::now();
                let _ = delaunay_gpu_gdel(&device, &queue, &pts);
                gv.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let (gb, gm) = stats(gv);
            let cpu = {
                let mut c = f64::INFINITY;
                for _ in 0..runs {
                    let t = Instant::now();
                    std::hint::black_box(triangulate_par(&pts, 0).unwrap());
                    c = c.min(t.elapsed().as_secs_f64() * 1e3);
                }
                c
            };
            println!(
                "gDel2D best {gb:.3} ms (median {gm:.3})   vs CPU parallel best {cpu:.3} ms  → {:.2}x",
                gb / cpu
            );
        }
        None => println!("gDel2D: degenerate hull → None"),
    }

    // warmup
    for _ in 0..3 {
        delaunay_gpu_native(&device, &queue, &pl, &pts);
    }
    let (mut cons, mut full, mut seedcpu, mut cpupar) = (vec![], vec![], vec![], vec![]);
    for _ in 0..runs {
        let t = Instant::now();
        let _ = construct_gpu(&device, &queue, &pts);
        cons.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        let _ = delaunay_gpu_native(&device, &queue, &pl, &pts);
        full.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        std::hint::black_box(hull_seed(&pts));
        seedcpu.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        std::hint::black_box(triangulate_par(&pts, 0).unwrap());
        cpupar.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let (cons_b, _) = stats(cons);
    let (full_b, full_m) = stats(full);
    let (seed_b, _) = stats(seedcpu);
    let (cpu_b, _) = stats(cpupar);
    println!(
        "GPU construct (on-device seed) best {cons_b:.3} ms   vs CPU hull_seed best {seed_b:.3} ms  → {:.2}x",
        cons_b / seed_b
    );
    println!(
        "GPU-native full best {full_b:.3} ms (median {full_m:.3})   vs CPU parallel best {cpu_b:.3} ms  → {:.2}x",
        full_b / cpu_b
    );
}
