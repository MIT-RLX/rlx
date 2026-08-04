// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// On-device validation of rlx-geo's wgpu kernels. Reuses rlx's initialized wgpu
// device, dispatches the real WGSL against a buffer laid out per the documented
// `WgpuGpuKernel` binding convention, and compares the GPU result to the CPU
// reference. Run: `cargo run -p rlx-geo --example gpu_validate --features gpu`.

use rlx_geo::predicates_wgsl::{INCIRCLE_WGSL, ORIENT_WGSL};
use rlx_geo::wgpu_kernels::VORONOI_WGSL;
use rlx_geo::{
    flip_all_convex_once, hull_seed, interior_quads, triangulate, voronoi_dual, voronoi_grid_exact,
};
use wgpu::util::DeviceExt;

// Exact CPU references (i128) to check the GPU predicates against.
fn cpu_orient(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i32 {
    let d = (b[0] as i128 - a[0] as i128) * (c[1] as i128 - a[1] as i128)
        - (b[1] as i128 - a[1] as i128) * (c[0] as i128 - a[0] as i128);
    (d > 0) as i32 - (d < 0) as i32
}
fn cpu_incircle(a: [i32; 2], b: [i32; 2], c: [i32; 2], d: [i32; 2]) -> i32 {
    let ax = a[0] as i128 - d[0] as i128;
    let ay = a[1] as i128 - d[1] as i128;
    let bx = b[0] as i128 - d[0] as i128;
    let by = b[1] as i128 - d[1] as i128;
    let cx = c[0] as i128 - d[0] as i128;
    let cy = c[1] as i128 - d[1] as i128;
    let det = (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay);
    (det > 0) as i32 - (det < 0) as i32
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn coord(&mut self, max: i32) -> i32 {
        (self.next() % max as u64) as i32
    }
}

fn main() {
    let Some(dev) = rlx_wgpu::device::wgpu_device() else {
        eprintln!("no wgpu device available");
        std::process::exit(1);
    };
    let device = &dev.device;
    let queue = &dev.queue;

    let mut failures = 0u32;
    failures += validate_voronoi(device, queue);
    failures += validate_orient(device, queue);
    failures += validate_incircle(device, queue);
    failures += validate_flip_marking(device, queue);
    failures += validate_dual_seed(device, queue);
    failures += validate_flip_gpu(device, queue);

    if failures == 0 {
        println!("\nALL GPU VALIDATIONS PASSED");
    } else {
        println!("\n{failures} GPU VALIDATION(S) FAILED");
        std::process::exit(1);
    }
}

/// Two storage bindings: arena (rw) @0, params (read) @1 — the fixed convention.
fn build_pipeline(
    device: &wgpu::Device,
    wgsl: &str,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rlx-geo validate"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rlx-geo validate"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rlx-geo validate"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rlx-geo validate"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, bgl)
}

/// Dispatch a kernel over one arena + params buffer, read back `out_len` i32s.
fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    arena_words: &[i32],
    params: &[u32],
    out_off: u32,
    out_len: u32,
    groups: u32,
) -> Vec<i32> {
    let (pipeline, bgl) = build_pipeline(device, wgsl);

    let arena = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("arena"),
        contents: bytemuck::cast_slice(arena_words),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let pbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(params),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: arena.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pbuf.as_entire_binding(),
            },
        ],
    });

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (out_len as u64) * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cp.set_pipeline(&pipeline);
        cp.set_bind_group(0, &bind, &[]);
        cp.dispatch_workgroups(groups, 1, 1);
    }
    enc.copy_buffer_to_buffer(
        &arena,
        (out_off as u64) * 4,
        &readback,
        0,
        (out_len as u64) * 4,
    );
    queue.submit(Some(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range().unwrap();
    let out: Vec<i32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback.unmap();
    out
}

fn validate_voronoi(device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
    let (w, h) = (48u32, 32u32);
    // A handful of sites inside the grid.
    let sites: Vec<[i32; 2]> = vec![
        [3, 4],
        [40, 2],
        [22, 27],
        [5, 29],
        [45, 30],
        [18, 12],
        [30, 18],
        [11, 20],
    ];
    let n = sites.len();

    // Input row 0 = [width, height]; then sites. Output region follows.
    let mut arena: Vec<i32> = vec![w as i32, h as i32];
    for s in &sites {
        arena.push(s[0]);
        arena.push(s[1]);
    }
    let in0_len = arena.len() as u32; // 2 + 2n
    let out_len = w * h;
    let out_off = in0_len;
    arena.resize((in0_len + out_len) as usize, 0); // output region

    // params = [out_off, out_len, n_inputs, _pad, in0_off, in0_len]
    let params: [u32; 6] = [out_off, out_len, 1, 0, 0, in0_len];

    let gpu = dispatch(
        device,
        queue,
        VORONOI_WGSL,
        &arena,
        &params,
        out_off,
        out_len,
        out_len.div_ceil(64),
    );

    let cpu = voronoi_grid_exact(&sites, w, h);
    let mut mism = 0usize;
    for i in 0..(out_len as usize) {
        if gpu[i] != cpu[i] as i32 {
            mism += 1;
        }
    }
    if mism == 0 {
        println!("voronoi_grid: GPU == CPU  ({w}x{h} grid, {n} sites, {out_len} cells) OK");
        0
    } else {
        println!("voronoi_grid: MISMATCH {mism}/{out_len} cells");
        1
    }
}

fn validate_orient(device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
    let n = 4096usize;
    let mut rng = Lcg(0x0ce0_1234u64);
    let mut cases: Vec<[i32; 2]> = Vec::with_capacity(3 * n);
    // span <= 20000 keeps the i32 cross product exact
    for _ in 0..(3 * n) {
        cases.push([rng.coord(20_000), rng.coord(20_000)]);
    }
    let mut arena: Vec<i32> = Vec::with_capacity(6 * n + n);
    for p in &cases {
        arena.push(p[0]);
        arena.push(p[1]);
    }
    let in0_len = arena.len() as u32; // 6n
    let out_len = n as u32;
    let out_off = in0_len;
    arena.resize((in0_len + out_len) as usize, 0);
    let params: [u32; 6] = [out_off, out_len, 1, 0, 0, in0_len];
    let gpu = dispatch(
        device,
        queue,
        ORIENT_WGSL,
        &arena,
        &params,
        out_off,
        out_len,
        out_len.div_ceil(64),
    );

    let mut mism = 0usize;
    for i in 0..n {
        let want = cpu_orient(cases[3 * i], cases[3 * i + 1], cases[3 * i + 2]);
        if gpu[i] != want {
            mism += 1;
        }
    }
    if mism == 0 {
        println!("orient2d:     GPU == CPU  ({n} random triples, exact i32) OK");
        0
    } else {
        println!("orient2d:     MISMATCH {mism}/{n}");
        1
    }
}

fn validate_incircle(device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
    let n = 4096usize;
    let mut rng = Lcg(0xbeef_0007u64);
    let mut cases: Vec<[i32; 2]> = Vec::with_capacity(4 * n);
    // span <= 20000 keeps the degree-4 determinant within emulated i64
    for _ in 0..(4 * n) {
        cases.push([rng.coord(20_000), rng.coord(20_000)]);
    }
    let mut arena: Vec<i32> = Vec::with_capacity(8 * n + n);
    for p in &cases {
        arena.push(p[0]);
        arena.push(p[1]);
    }
    let in0_len = arena.len() as u32; // 8n
    let out_len = n as u32;
    let out_off = in0_len;
    arena.resize((in0_len + out_len) as usize, 0);
    let params: [u32; 6] = [out_off, out_len, 1, 0, 0, in0_len];
    let gpu = dispatch(
        device,
        queue,
        INCIRCLE_WGSL,
        &arena,
        &params,
        out_off,
        out_len,
        out_len.div_ceil(64),
    );

    let mut mism = 0usize;
    for i in 0..n {
        let want = cpu_incircle(
            cases[4 * i],
            cases[4 * i + 1],
            cases[4 * i + 2],
            cases[4 * i + 3],
        );
        if gpu[i] != want {
            mism += 1;
        }
    }
    if mism == 0 {
        println!("in_circle:    GPU == CPU  ({n} random quads, emulated i64) OK");
        0
    } else {
        println!("in_circle:    MISMATCH {mism}/{n} (emulated i64 vs i128)");
        1
    }
}

// The flip round's core step — marking illegal edges — run on the GPU over a real
// (scrambled, non-Delaunay) mesh, using the validated in_circle kernel.
fn validate_flip_marking(device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
    let mut rng = Lcg(0xf115_5eedu64);
    // distinct points
    let mut seen = std::collections::HashSet::new();
    let mut pts: Vec<[i32; 2]> = Vec::new();
    while pts.len() < 300 {
        let p = [rng.coord(20_000), rng.coord(20_000)];
        if seen.insert(p) {
            pts.push(p);
        }
    }
    // GS Delaunay -> scramble into a non-Delaunay seed -> the mesh a flip round sees.
    let (mesh, _) = flip_all_convex_once(triangulate(&pts).unwrap(), &pts);
    let quads = interior_quads(&mesh, &pts); // [t0v0, t0v1, t0v2, q] per interior edge
    let m = quads.len();

    // Pack 8 i32 per quad; output = m signs.
    let mut arena: Vec<i32> = Vec::with_capacity(8 * m + m);
    for qd in &quads {
        for p in qd {
            arena.push(p[0]);
            arena.push(p[1]);
        }
    }
    let in0_len = arena.len() as u32; // 8m
    let out_len = m as u32;
    let out_off = in0_len;
    arena.resize((in0_len + out_len) as usize, 0);
    let params: [u32; 6] = [out_off, out_len, 1, 0, 0, in0_len];
    let gpu = dispatch(
        device,
        queue,
        INCIRCLE_WGSL,
        &arena,
        &params,
        out_off,
        out_len,
        out_len.div_ceil(64),
    );

    let mut mism = 0usize;
    let mut illegal = 0usize;
    for i in 0..m {
        let q = quads[i];
        let want = cpu_incircle(q[0], q[1], q[2], q[3]); // >0 => illegal edge
        if want > 0 {
            illegal += 1;
        }
        if gpu[i] != want {
            mism += 1;
        }
    }
    if mism == 0 {
        println!("flip marking: GPU == CPU  ({m} interior edges, {illegal} flagged illegal) OK");
        0
    } else {
        println!("flip marking: MISMATCH {mism}/{m}");
        1
    }
}

// GPU Voronoi (JFA/nearest) -> dual extraction: the seed path. Confirms every
// extracted triangle is a genuine Delaunay triangle (precision), reports recall.
fn validate_dual_seed(device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
    use std::collections::HashSet;
    let (w, h) = (512u32, 512u32);
    let mut rng = Lcg(0xd0a1_5eedu64);
    let mut seen = HashSet::new();
    let mut pts: Vec<[i32; 2]> = Vec::new();
    while pts.len() < 40 {
        let p = [
            80 + rng.coord(w as i32 - 160),
            80 + rng.coord(h as i32 - 160),
        ];
        if seen.insert(p) {
            pts.push(p);
        }
    }

    // GPU Voronoi labels (row 0 packs [w,h]; sites follow; output = w*h labels).
    let mut arena: Vec<i32> = vec![w as i32, h as i32];
    for p in &pts {
        arena.push(p[0]);
        arena.push(p[1]);
    }
    let in0_len = arena.len() as u32;
    let out_len = w * h;
    let out_off = in0_len;
    arena.resize((in0_len + out_len) as usize, 0);
    let params = [out_off, out_len, 1u32, 0, 0, in0_len];
    let labels_i32 = dispatch(
        device,
        queue,
        rlx_geo::wgpu_kernels::VORONOI_WGSL,
        &arena,
        &params,
        out_off,
        out_len,
        out_len.div_ceil(64),
    );
    let labels: Vec<u32> = labels_i32.iter().map(|&x| x as u32).collect();

    let dual = voronoi_dual(&labels, w, h);
    let canon = |t: [u32; 3]| {
        let mut v = t;
        v.sort_unstable();
        v
    };
    let truth: HashSet<[u32; 3]> = triangulate(&pts).unwrap().into_iter().map(canon).collect();
    let dset: HashSet<[u32; 3]> = dual.iter().map(|&t| canon(t)).collect();
    let correct = dset.iter().filter(|t| truth.contains(*t)).count();
    let precision = correct as f64 / dset.len().max(1) as f64;
    let recall = correct as f64 / truth.len().max(1) as f64;

    if (precision - 1.0).abs() < 1e-9 {
        println!(
            "dual seed:    GPU Voronoi -> dual  ({} tris, precision {precision:.3}, recall {recall:.3} — interior seed) OK",
            dset.len()
        );
        0
    } else {
        println!("dual seed:    non-Delaunay triangle emitted (precision {precision:.3})");
        1
    }
}

// Count locally-illegal interior edges (0 iff Delaunay); also checks manifoldness.
fn illegal_count(points: &[[i32; 2]], tris: &[[u32; 3]]) -> Result<usize, String> {
    use std::collections::HashMap;
    let mut edges: HashMap<u64, (u32, u32, u32, u32)> = HashMap::new();
    let key = |a: u32, b: u32| {
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        ((a as u64) << 32) | b as u64
    };
    let mut bad = 0;
    for t in tris {
        if t[0] == t[1] || t[1] == t[2] || t[2] == t[0] {
            return Err("degenerate triangle".into());
        }
        if cpu_orient(
            points[t[0] as usize],
            points[t[1] as usize],
            points[t[2] as usize],
        ) <= 0
        {
            return Err("triangle not CCW".into());
        }
        for &(a, b, opp) in &[(t[0], t[1], t[2]), (t[1], t[2], t[0]), (t[2], t[0], t[1])] {
            match edges.get_mut(&key(a, b)) {
                None => {
                    edges.insert(key(a, b), (a, b, opp, 1));
                }
                Some(rec) => {
                    if rec.3 != 1 {
                        return Err("non-manifold edge".into());
                    }
                    rec.3 = 2;
                    if cpu_incircle(
                        points[rec.0 as usize],
                        points[rec.1 as usize],
                        points[rec.2 as usize],
                        points[opp as usize],
                    ) > 0
                    {
                        bad += 1;
                    }
                }
            }
        }
    }
    Ok(bad)
}

// The whole loop on-device: scramble a mesh, run flip_to_delaunay_gpu, and check
// the downloaded result is a valid Delaunay mesh matching the CPU reference.
fn validate_flip_gpu(device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
    // Test across coordinate spans: the small one fits the i32-inner arithmetic,
    // but the large ones (> ~32k) need the i64-inner / i128-determinant path —
    // they oscillated forever with the old i32 in-circle. Near-max exercises the
    // full i128 range. n < 2^16 for the 16-bit edge hash.
    let mut failures = 0u32;
    for span in [29_000i32, 100_000, 1_000_000, MAX_FLIP_SPAN] {
        failures += check_flip_span(device, queue, span);
    }
    failures
}

/// Largest span the GPU flip is certified for (matches `MAX_COORDINATE_SPAN`).
const MAX_FLIP_SPAN: i32 = 1_940_470_527;

fn check_flip_span(device: &wgpu::Device, queue: &wgpu::Queue, span: i32) -> u32 {
    let mut rng = Lcg(0x0ce0_f11du64 ^ (span as u64));
    let mut seen = std::collections::HashSet::new();
    let mut pts: Vec<[i32; 2]> = Vec::new();
    while pts.len() < 1500 {
        let p = [rng.coord(span), rng.coord(span)];
        if seen.insert(p) {
            pts.push(p);
        }
    }
    let reference = triangulate(&pts).unwrap();
    // Complete valid seed (hull included) built by the convex-hull sweep.
    let seed = hull_seed(&pts);
    let seed_bad = illegal_count(&pts, &seed).expect("seed invalid");

    let out = rlx_geo::flip_gpu::flip_to_delaunay_gpu(device, queue, &seed, &pts);

    let mut used = vec![false; pts.len()];
    for t in &out {
        for &i in t {
            used[i as usize] = true;
        }
    }
    let complete = used.iter().all(|&u| u) && out.len() == reference.len();

    match illegal_count(&pts, &out) {
        Ok(0) if complete => {
            println!(
                "flip loop (span {span:>10}): GPU flip -> COMPLETE Delaunay ({} tris, seed had {seed_bad} illegal) OK",
                out.len()
            );
            0
        }
        Ok(bad) => {
            println!(
                "flip loop (span {span:>10}): INVALID ({bad} illegal, {} tris vs {} ref, complete={complete})",
                out.len(),
                reference.len()
            );
            1
        }
        Err(e) => {
            println!("flip loop (span {span:>10}): invalid mesh: {e}");
            1
        }
    }
}
