// Reads a binary point file (shared with the C++ driver), triangulates it a few
// times reusing one Triangulator, and prints: rust,<points>,<triangles>,<median_ms>
//
// File format (little-endian): u64 count, then `count` * (i32 x, i32 y).

use std::io::Read;

use rlx_geo::delaunay32::{Point, Triangulator};

fn label(threads: usize) -> &'static str {
    // 0 = auto (parallel), 1 = serial, N>1 = parallel
    if threads == 1 { "rust" } else { "rust_par" }
}

fn read_points(path: &str) -> std::io::Result<Vec<Point>> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    let count = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
    let mut pts = Vec::with_capacity(count);
    let mut off = 8;
    for _ in 0..count {
        let x = i32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let y = i32::from_le_bytes(buf[off + 4..off + 8].try_into().unwrap());
        pts.push(Point::new(x, y));
        off += 8;
    }
    Ok(pts)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: bench <points.bin> [runs]");
        std::process::exit(2);
    }
    let path = &args[1];
    let runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);
    // threads: 1 = serial (default), 0 = auto, N = fixed
    let threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let pts = read_points(path).expect("read points");
    let mut tri = Triangulator::with_threads(threads);

    // Warm up (also fixes triangle count for reporting).
    let triangles = tri.triangulate(&pts).len();

    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = std::time::Instant::now();
        let out = tri.triangulate(&pts);
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&out);
        samples.push(dt);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];

    println!(
        "{},{},{},{:.3}",
        label(threads),
        pts.len(),
        triangles,
        median
    );
}
