// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Standalone on-device flip benchmark + exact validation. Creates its own wgpu
// device (no rlx-wgpu), so it builds with only the decoupled `gpu` feature —
// on-device geometry without the ML stack:
//   cargo run -p rlx-geo --example flip_gpu_bench \
//       --no-default-features --features gpu --release -- <file> [runs]
//
// Validates the GPU flip output is an exact Delaunay of the input (CCW + manifold
// + empty-circumcircle, all i128), then times the f32-filter path against the
// all-i128 path (GEO_FLIP_NOFILTER), interleaved per call for fair comparison.

use std::collections::HashMap;
use std::io::Read;
use std::time::Instant;

use rlx_geo::flip_gpu::{FlipPipeline, flip_to_delaunay_gpu_with};
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

/// Full exact-Delaunay check: CCW, manifold, empty-circumcircle; plus complete
/// (every point used) and same triangle count as the exact CPU reference.
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
    let runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(25);
    let pts = read_points(path);
    let n = pts.len();

    // Standalone wgpu device (no rlx-wgpu).
    let instance = wgpu::Instance::default();
    // Enumerate every GPGPU adapter (Vulkan/Metal/DX12/GL). GEO_GPU_ADAPTER=<substr>
    // targets one by name (e.g. "NVIDIA", "Intel", "llvmpipe"); otherwise prefer a
    // discrete GPU, then integrated, then whatever is first.
    let all = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    eprintln!("available adapters:");
    for a in &all {
        let i = a.get_info();
        eprintln!("  - {} [{:?}, {:?}]", i.name, i.device_type, i.backend);
    }
    let adapter = match std::env::var("GEO_GPU_ADAPTER") {
        Ok(want) => all
            .into_iter()
            .find(|a| {
                a.get_info()
                    .name
                    .to_lowercase()
                    .contains(&want.to_lowercase())
            })
            .expect("no adapter matches GEO_GPU_ADAPTER"),
        Err(_) => {
            let pick = |t: wgpu::DeviceType| all.iter().position(|a| a.get_info().device_type == t);
            let idx = pick(wgpu::DeviceType::DiscreteGpu)
                .or_else(|| pick(wgpu::DeviceType::IntegratedGpu))
                .unwrap_or(0);
            all.into_iter().nth(idx).expect("no wgpu adapter")
        }
    };
    let info = adapter.get_info();
    // The flip pipeline binds 12 storage buffers; the default downlevel limit is
    // 8, so request the adapter's full limits (as rlx-wgpu's device helper does).
    // Request TIMESTAMP_QUERY if available so GEO_FLIP_PROF can measure true
    // per-pass GPU time (independent of the CPU poll-wait).
    let feats = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: feats,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("no wgpu device");

    let ref_count = triangulate(&pts).unwrap().len();
    let seed = hull_seed(&pts);
    let pipeline = FlipPipeline::new(&device); // built once, reused every call

    let run_once = || {
        let s = seed.clone();
        let t = Instant::now();
        let r = flip_to_delaunay_gpu_with(&device, &queue, &pipeline, &s, &pts);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        (r, ms)
    };

    // Correctness first (filter on = default), then all-i128 control.
    unsafe { std::env::remove_var("GEO_FLIP_NOFILTER") };
    let (out_on, _) = run_once();
    unsafe { std::env::set_var("GEO_FLIP_NOFILTER", "1") };
    let (out_off, _) = run_once();
    let v_on = validate(&pts, &out_on, ref_count);
    let v_off = validate(&pts, &out_off, ref_count);

    println!("device: {} ({:?})", info.name, info.backend);
    println!("n={n} ref_tris={ref_count} runs={runs}");
    println!(
        "validate filter_ON : {}",
        v_on.as_ref()
            .map(|_| "EXACT Delaunay OK".into())
            .unwrap_or_else(|e| format!("FAIL: {e}"))
    );
    println!(
        "validate filter_OFF: {}",
        v_off
            .as_ref()
            .map(|_| "EXACT Delaunay OK".into())
            .unwrap_or_else(|e| format!("FAIL: {e}"))
    );
    if v_on.is_err() || v_off.is_err() {
        std::process::exit(1);
    }

    // Warm up, then interleave timed runs.
    for _ in 0..3 {
        unsafe { std::env::remove_var("GEO_FLIP_NOFILTER") };
        run_once();
        unsafe { std::env::set_var("GEO_FLIP_NOFILTER", "1") };
        run_once();
    }
    let mut on = Vec::with_capacity(runs);
    let mut off = Vec::with_capacity(runs);
    for _ in 0..runs {
        unsafe { std::env::remove_var("GEO_FLIP_NOFILTER") };
        on.push(run_once().1);
        unsafe { std::env::set_var("GEO_FLIP_NOFILTER", "1") };
        off.push(run_once().1);
    }
    let (on_best, on_med) = stats(on);
    let (off_best, off_med) = stats(off);
    println!("filter_ON  : best {on_best:.3} ms  median {on_med:.3} ms");
    println!("filter_OFF : best {off_best:.3} ms  median {off_med:.3} ms");
    println!(
        "speedup (median off/on): {:.3}x   (best): {:.3}x",
        off_med / on_med,
        off_best / on_best
    );

    // Honest end-to-end comparison: the FULL GPU path (CPU hull_seed + GPU flip,
    // timed together per run) vs the CPU parallel triangulation it competes with.
    unsafe { std::env::remove_var("GEO_FLIP_NOFILTER") };
    let mut seed_v = Vec::with_capacity(runs);
    let mut full_v = Vec::with_capacity(runs);
    let mut cpu_v = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let s = hull_seed(&pts);
        let seed_ms = t.elapsed().as_secs_f64() * 1e3;
        let _ = flip_to_delaunay_gpu_with(&device, &queue, &pipeline, &s, &pts);
        let full_ms = t.elapsed().as_secs_f64() * 1e3;
        seed_v.push(seed_ms);
        full_v.push(full_ms);
        // The honest CPU competitor is the PARALLEL fast path (`triangulate_par`, all
        // cores) — what `triangulate_fastest` picks above parallel_min — NOT serial
        // `triangulate()` (which is ~5× slower on a 20-core box and would flatter the GPU).
        let t = Instant::now();
        std::hint::black_box(triangulate_par(&pts, 0).unwrap());
        cpu_v.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let (seed_best, _) = stats(seed_v);
    let (full_best, full_med) = stats(full_v);
    let (cpu_best, _) = stats(cpu_v);
    println!(
        "END-TO-END: seed {seed_best:.3} + flip → full GPU path best {full_best:.3} ms (median {full_med:.3})  |  CPU parallel best {cpu_best:.3} ms  |  GPU/CPU {:.3}x",
        full_best / cpu_best
    );
}
