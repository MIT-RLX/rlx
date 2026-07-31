// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Reads a binary point file (u64 count, then count*(i32 x, i32 y)) — the same
// format the C++ delaunay32 driver dumps — and times each rlx-geo path.
// Output CSV: impl,points,triangles,median_ms
//   cargo run -p rlx-geo --example bench --features gpu --release -- <file>

use std::io::Read;
use std::time::Instant;

use rlx_geo::{
    flip_to_delaunay, hull_seed, triangulate, triangulate_dwyer, triangulate_fastest,
    triangulate_par,
};

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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let pts = read_points(path);
    let n = pts.len();

    // Exact serial Guibas-Stolfi (same algorithm class as C++ delaunay32).
    let tris = triangulate(&pts).len();
    let mut s = Vec::new();
    for _ in 0..runs {
        let t = Instant::now();
        let r = triangulate(&pts);
        std::hint::black_box(&r);
        s.push(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("geo_gs,{n},{tris},{:.3}", median(s));

    // Serial Dwyer (Morton alternating-cut) build.
    let _ = triangulate_dwyer(&pts);
    let mut s = Vec::new();
    for _ in 0..runs {
        let t = Instant::now();
        let r = triangulate_dwyer(&pts);
        std::hint::black_box(&r);
        s.push(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("geo_dwyer,{n},{tris},{:.3}", median(s));

    // Parallel CPU divide-and-conquer.
    let _ = triangulate_par(&pts, 0);
    let mut s = Vec::new();
    for _ in 0..runs {
        let t = Instant::now();
        let r = triangulate_par(&pts, 0);
        std::hint::black_box(&r);
        s.push(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("geo_par,{n},{tris},{:.3}", median(s));

    // Auto-dispatched fastest backend.
    let (_, backend) = triangulate_fastest(&pts);
    let mut s = Vec::new();
    for _ in 0..runs {
        let t = Instant::now();
        let (r, _) = triangulate_fastest(&pts);
        std::hint::black_box(&r);
        s.push(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "geo_fastest[{}],{n},{tris},{:.3}",
        backend.name(),
        median(s)
    );

    // CPU Lawson flip pipeline: hull_seed + flip_to_delaunay. O(n^2)-ish, skip at scale.
    if n <= 100_000 {
        let _ = flip_to_delaunay(hull_seed(&pts), &pts);
        let mut s = Vec::new();
        for _ in 0..runs {
            let t = Instant::now();
            let (r, _) = flip_to_delaunay(hull_seed(&pts), &pts);
            std::hint::black_box(&r);
            s.push(t.elapsed().as_secs_f64() * 1e3);
        }
        println!("geo_flip_cpu,{n},{tris},{:.3}", median(s));
    } else {
        println!("geo_flip_cpu,{n},{tris},SKIP_LARGE");
    }

    // GPU Lawson flip pipeline (feature gpu): hull_seed (CPU) + on-device flip.
    #[cfg(feature = "gpu")]
    {
        if n < (1 << 16) {
            if let Some(dev) = rlx_wgpu::device::wgpu_device() {
                let seed = hull_seed(&pts);
                let _ =
                    rlx_geo::flip_gpu::flip_to_delaunay_gpu(&dev.device, &dev.queue, &seed, &pts);
                let mut s = Vec::new();
                for _ in 0..runs {
                    let seed = hull_seed(&pts);
                    let t = Instant::now();
                    let r = rlx_geo::flip_gpu::flip_to_delaunay_gpu(
                        &dev.device,
                        &dev.queue,
                        &seed,
                        &pts,
                    );
                    std::hint::black_box(&r);
                    s.push(t.elapsed().as_secs_f64() * 1e3);
                }
                println!("geo_flip_gpu,{n},{tris},{:.3}", median(s));
            } else {
                println!("geo_flip_gpu,{n},{tris},NO_DEVICE");
            }
        } else {
            println!("geo_flip_gpu,{n},{tris},SKIP_N>=65536");
        }
    }
}
