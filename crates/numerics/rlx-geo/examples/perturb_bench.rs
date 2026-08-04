// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Resolves the crux question for flip-based merging: does a NEAR-Delaunay seed flip in
// few rounds? Takes the true Delaunay, flips an independent set of a fraction f of
// internal edges to their WRONG diagonal (scattered local perturbations), then measures
// how many flip rounds are needed to restore Delaunay. If few rounds for small f, then
// locality helps and a square-tile merge (localized seams) could win; if ~log n
// regardless, the flip is fundamentally depth-bound and no seed makes the merge cheap.
//   cargo run -p rlx-geo --example perturb_bench --no-default-features --features gpu --release -- <file>

use rlx_geo::flip_gpu::{FlipPipeline, flip_to_delaunay_gpu_with};
use rlx_geo::{hull_seed, triangulate};
use std::collections::HashMap;
use std::io::Read;

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pts = read_points(&args[1]);
    let n = pts.len();

    let instance = wgpu::Instance::default();
    let all = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    let adapter = {
        let pick = |t: wgpu::DeviceType| all.iter().position(|a| a.get_info().device_type == t);
        let idx = pick(wgpu::DeviceType::DiscreteGpu)
            .or_else(|| pick(wgpu::DeviceType::IntegratedGpu))
            .unwrap_or(0);
        all.into_iter().nth(idx).unwrap()
    };
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
    let pl = FlipPipeline::new(&device);

    let d = triangulate(&pts).unwrap(); // true Delaunay
    let refc = d.len();
    println!(
        "device: {} ({:?})   n={n}  ref_tris={refc}",
        info.name, info.backend
    );

    // internal-edge adjacency: edge -> the (tri, apex) records
    let mut edge: HashMap<(u32, u32), Vec<(usize, u32)>> = HashMap::new();
    for (ti, t) in d.iter().enumerate() {
        for e in 0..3 {
            let (a, b, ap) = (t[e], t[(e + 1) % 3], t[(e + 2) % 3]);
            let k = if a < b { (a, b) } else { (b, a) };
            edge.entry(k).or_default().push((ti, ap));
        }
    }
    let internal: Vec<((u32, u32), usize, u32, usize, u32)> = edge
        .iter()
        .filter(|(_, v)| v.len() == 2)
        .map(|(&(a, b), v)| ((a, b), v[0].0, v[0].1, v[1].0, v[1].1))
        .collect();

    // deterministic PRNG walk over edges → independent set → flip wrong diagonal
    let make_perturbed = |f: f64| -> (Vec<[u32; 3]>, usize) {
        let mut tris = d.clone();
        let mut used = vec![false; d.len()];
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut idx: Vec<usize> = (0..internal.len()).collect();
        // Fisher-Yates with the LCG for a scattered independent set
        for i in (1..idx.len()).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (s >> 33) as usize % (i + 1);
            idx.swap(i, j);
        }
        let target = (internal.len() as f64 * f) as usize;
        let mut done = 0;
        for &ii in &idx {
            if done >= target {
                break;
            }
            let ((a, b), t0, c, t1, dd) = internal[ii];
            if used[t0] || used[t1] {
                continue;
            }
            // flip diagonal (a,b) -> (c,dd) on quad a,c,b,dd; only if the quad is convex
            // so the flip yields a valid triangulation (winding normalized below).
            let (pa, pb, pc, pd) = (
                pts[a as usize],
                pts[b as usize],
                pts[c as usize],
                pts[dd as usize],
            );
            if orient(pd, pa, pc) <= 0 || orient(pc, pb, pd) <= 0 {
                continue;
            }
            tris[t0] = [dd, a, c];
            tris[t1] = [c, b, dd];
            used[t0] = true;
            used[t1] = true;
            done += 1;
        }
        // ensure CCW
        for t in tris.iter_mut() {
            if orient(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]) < 0 {
                t.swap(1, 2);
            }
        }
        (tris, done)
    };

    let bench = |seed: &[[u32; 3]]| -> f64 {
        for _ in 0..3 {
            flip_to_delaunay_gpu_with(&device, &queue, &pl, seed, &pts);
        }
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let t = std::time::Instant::now();
            let out =
                std::hint::black_box(flip_to_delaunay_gpu_with(&device, &queue, &pl, seed, &pts));
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(out.len(), refc);
        }
        best
    };
    let hs = hull_seed(&pts);
    let far = bench(&hs);
    for &f in &[0.0f64, 0.005, 0.02, 0.1, 0.3] {
        let (seed, flipped) = make_perturbed(f);
        let ms = bench(&seed);
        println!(
            "perturb f={f:<5} ({flipped:>6} edges wrong): flip {ms:6.3} ms   ({:.2}x vs hull_seed)",
            far / ms
        );
    }
    println!("hull_seed (100% far):                  flip {far:6.3} ms   (1.00x)");
}
