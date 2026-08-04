// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU-native Delaunay construction (gDel2D-style). Unlike [`crate::flip_gpu`]
//! (which perfects a CPU-built seed), this builds the WHOLE triangulation on the
//! device: the host only computes a bounding box; every point is inserted on the
//! GPU. The goal is to eliminate the serial CPU `hull_seed` (which alone ties the
//! parallel CPU's entire runtime).
//!
//! The insertion is **association-based**, not walk-based (the difference that made
//! the earlier ablated `insert_gpu` slow): each point stores the id of the triangle
//! that contains it (`assoc`). Per round, every triangle picks its lowest-id
//! associated point (`atomicMin` → `nominee`), splits into three children around it
//! (`split`), and its other points are re-associated to the correct child by two
//! `orient` sign tests (`reassoc`) — no mesh walking. Triangle count grows
//! geometrically, so all points land in ~O(log n) rounds. The result is a valid
//! (non-Delaunay) triangulation over `points ∪ {3 bounding vertices}`; feed it to
//! the flip, then drop bounding-incident triangles.
//!
//! Exactness: `orient` is the exact emulated-i64 cross product (span ≤ ~4.3e7 with
//! the bounding margin; larger spans fall back — caller uses the CPU seed). The
//! insertion itself is exact (validated: rounds ≈ log n, tri_count = 2n+1, on_edge=0
//! on random data). **CAVEAT — not yet exact at the hull:** the finite super triangle
//! leaves a small deficit (4/8/58 real triangles at 60k/200k/1M) because a real
//! hull-triangle circumcircle can contain a super vertex → it flips away. Pushing the
//! super out only shrinks it (bounded by `MAX_COORDINATE_SPAN`); the exact fix is
//! ghost/∞ vertices in the flip's `in_circle` (freezing super EDGES doesn't work —
//! it keeps ALL of construction's non-hull ghost slivers). This is a working
//! PROTOTYPE.
//!
//! MEASURED (M4 Pro, random): GPU construct beats the CPU `hull_seed` at scale
//! (1M: 24.7 ms vs 37.5 ms, 0.66×) but is slower below ~500k (fixed GPU overhead).
//! The full path (construct → flip → drop bounding) is still ~9× the parallel CPU and
//! no better than the CPU-seed hybrid, because the **flip** (O(rounds·T) bandwidth)
//! dominates and is unchanged — eliminating the serial seed was necessary but not
//! sufficient. Only worth finishing (ghost vertices + construct/flip fusion to skip
//! the round-trip) for the VRAM-resident niche, where a zero-CPU path matters.

use wgpu::util::DeviceExt;

const NONE: u32 = 0xffff_ffff;
const INSERTED: u32 = 0xffff_fffe;

const CONSTRUCT_WGSL: &str = r#"
const NONE: u32 = 0xffffffffu;
const INSERTED: u32 = 0xfffffffeu;
const XSTRIDE: u32 = 65536u;

@group(0) @binding(0) var<storage, read>       pts:        array<i32>;          // 2*(n+3)
@group(0) @binding(1) var<storage, read_write> tris:       array<u32>;          // 3*maxT
@group(0) @binding(2) var<storage, read_write> assoc:      array<u32>;          // n (tri id | INSERTED)
@group(0) @binding(3) var<storage, read_write> nominee:    array<atomic<u32>>;  // maxT (lowest-id point)
@group(0) @binding(4) var<storage, read_write> split_base: array<u32>;          // maxT (NONE | new1 id)
@group(0) @binding(5) var<storage, read_write> counter:    array<atomic<u32>>;  // [tri_count, on_edge]
@group(0) @binding(6) var<storage, read>       dims:       array<u32>;          // [n, maxT, tri_count_this_round]

fn px(v: u32) -> i32 { return pts[v * 2u]; }
fn py(v: u32) -> i32 { return pts[v * 2u + 1u]; }

// --- exact emulated i64 cross product (i32 differences, i64 product) ---
fn mul_u32(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xffffu; let ah = a >> 16u;
    let bl = b & 0xffffu; let bh = b >> 16u;
    let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;
    let cross = lh + hl;
    let cc = select(0u, 1u, cross < lh);
    let lo = ll + (cross << 16u);
    let lc = select(0u, 1u, lo < ll);
    return vec2<u32>(lo, hh + (cross >> 16u) + (cc << 16u) + lc);
}
fn neg_i64(x: vec2<u32>) -> vec2<u32> {
    let lo = ~x.x + 1u;
    return vec2<u32>(lo, ~x.y + select(0u, 1u, lo == 0u));
}
fn mul_i32(a: i32, b: i32) -> vec2<u32> {
    let r = mul_u32(u32(abs(a)), u32(abs(b)));
    if ((a < 0) != (b < 0)) { return neg_i64(r); }
    return r;
}
fn add_i64(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> {
    let lo = x.x + y.x;
    return vec2<u32>(lo, x.y + y.y + select(0u, 1u, lo < x.x));
}
fn sign_i64(x: vec2<u32>) -> i32 {
    let hi = bitcast<i32>(x.y);
    if (hi < 0) { return -1; }
    if (hi > 0) { return 1; }
    if (x.x != 0u) { return 1; }
    return 0;
}
// sign of (b-a) x (c-a)  (>0 ⇒ c left of a→b)
fn ori(a: u32, b: u32, c: u32) -> i32 {
    let t1 = mul_i32(px(b) - px(a), py(c) - py(a));
    let t2 = mul_i32(py(b) - py(a), px(c) - px(a));
    return sign_i64(add_i64(t1, neg_i64(t2)));
}

fn gidx(gid: vec3<u32>) -> u32 { return gid.y * XSTRIDE + gid.x; }

@compute @workgroup_size(64)
fn clear_nominee(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gidx(gid);
    if (t >= dims[2]) { return; }   // STABLE round-start count (counter[0] grows in `split`)
    atomicStore(&nominee[t], NONE);
    split_base[t] = NONE;
}

@compute @workgroup_size(64)
fn pick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gidx(gid);
    if (i >= dims[0]) { return; }
    let a = assoc[i];
    if (a == INSERTED) { return; }
    atomicMin(&nominee[a], i);
}

@compute @workgroup_size(64)
fn split(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gidx(gid);
    if (t >= dims[2]) { return; }   // STABLE round-start count — NOT counter[0] (grows here)
    let p = atomicLoad(&nominee[t]);
    if (p == NONE) { return; }
    let a = tris[t * 3u]; let b = tris[t * 3u + 1u]; let c = tris[t * 3u + 2u];
    // strictly-interior guard: an on-edge nominee would make a zero-area child (and a
    // non-conforming neighbour) — count and skip it (≈0 on general-position input).
    if (ori(a, b, p) == 0 || ori(b, c, p) == 0 || ori(c, a, p) == 0) {
        atomicAdd(&counter[1], 1u);
        return;
    }
    let base = atomicAdd(&counter[0], 2u);   // allocate two new triangles
    let n1 = base; let n2 = base + 1u;
    tris[t * 3u] = a;  tris[t * 3u + 1u] = b;  tris[t * 3u + 2u] = p;   // t  = (a,b,p)
    tris[n1 * 3u] = b; tris[n1 * 3u + 1u] = c; tris[n1 * 3u + 2u] = p;  // n1 = (b,c,p)
    tris[n2 * 3u] = c; tris[n2 * 3u + 1u] = a; tris[n2 * 3u + 2u] = p;  // n2 = (c,a,p)
    split_base[t] = base;
    assoc[p] = INSERTED;
}

@compute @workgroup_size(64)
fn reassoc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gidx(gid);
    if (i >= dims[0]) { return; }
    if (assoc[i] == INSERTED) { return; }
    let t = assoc[i];
    let base = split_base[t];
    if (base == NONE) { return; }              // t didn't split this round
    // recover the parent's a,b,c,p from the three children (t=(a,b,p), n1=(b,c,p))
    let a = tris[t * 3u]; let b = tris[t * 3u + 1u]; let p = tris[t * 3u + 2u];
    let c = tris[base * 3u + 1u];
    // p is interior; rays p→a, p→b, p→c split the plane around p into the 3 children.
    let sa = ori(p, a, i);
    let sb = ori(p, b, i);
    let sc = ori(p, c, i);
    if (sa >= 0 && sb <= 0) { assoc[i] = t; }          // wedge (a,b,p) = t
    else if (sb >= 0 && sc <= 0) { assoc[i] = base; }  // wedge (b,c,p) = n1
    else { assoc[i] = base + 1u; }                     // wedge (c,a,p) = n2
}
"#;

/// Build a valid triangulation of `points ∪ {3 bounding vertices}` entirely on the
/// GPU (association-based incremental insertion). Returns `(tris, ext_points)` where
/// `ext_points = points` followed by the 3 bounding vertices (ids `n, n+1, n+2`);
/// bounding-incident triangles are the caller's to drop after the flip. Returns
/// `None` if the coordinate span is too large for the bounding margin to stay exact
/// (caller falls back to the CPU seed).
pub fn construct_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    points: &[[i32; 2]],
) -> Option<(Vec<[u32; 3]>, Vec<[i32; 2]>)> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    // Bounding box → a big triangle containing every point. Margin must keep the
    // Delaunay circumcircles of real points clear of the bounding vertices AND keep
    // coordinates within i32 (differences fit i32, products fit i64 for exact orient).
    let (mut mnx, mut mny, mut mxx, mut mxy) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for p in points {
        mnx = mnx.min(p[0]);
        mny = mny.min(p[1]);
        mxx = mxx.max(p[0]);
        mxy = mxy.max(p[1]);
    }
    let w = mxx as i64 - mnx as i64;
    let h = mxy as i64 - mny as i64;
    let d = w.max(h).max(1);
    let midx = (mnx as i64 + mxx as i64) / 2;
    let midy = (mny as i64 + mxy as i64) / 2;
    // Super-triangle (classic wide+tall form that contains the box). It must be FAR
    // enough that no real Delaunay circumcircle reaches a super vertex (else that real
    // hull triangle flips away → deficit). Pushed as far as the construct's exact i64
    // `orient` allows: its i32 coordinate differences must stay < 2^31, i.e. the widest
    // span 40·dmax < 2^31 ⇒ dmax ≲ 5.3e7. Take the max margin under that cap.
    // 40·dmax is the widest super span; keep it < MAX_COORDINATE_SPAN (1.94e9) for the
    // flip's i128 predicate AND its i32 differences < 2^31 for construct's orient.
    let dmax = d.saturating_mul(45).min(45_000_000);
    let (ax, ay) = (midx - 20 * dmax, midy - dmax);
    let (bx, by) = (midx + 20 * dmax, midy - dmax);
    let (cx, cy) = (midx, midy + 20 * dmax);
    let lim = 950_000_000i64;
    if ax
        .abs()
        .max(bx.abs())
        .max(cx.abs())
        .max(ay.abs())
        .max(by.abs())
        .max(cy.abs())
        > lim
    {
        return None;
    }

    let mut ext: Vec<[i32; 2]> = Vec::with_capacity(n + 3);
    ext.extend_from_slice(points);
    ext.push([ax as i32, ay as i32]);
    ext.push([bx as i32, by as i32]);
    ext.push([cx as i32, cy as i32]);

    let max_t = 2 * n + 8; // ≤ 2(n+3) triangles total
    let pt_flat: Vec<i32> = ext.iter().flat_map(|p| [p[0], p[1]]).collect();

    // initial triangulation = the single super-triangle (ids n, n+1, n+2)
    let mut tris0 = vec![0u32; 3 * max_t];
    tris0[0] = n as u32;
    tris0[1] = n as u32 + 1;
    tris0[2] = n as u32 + 2;
    let assoc0 = vec![0u32; n]; // every point starts inside triangle 0

    let storage = |data: &[u8], extra: wgpu::BufferUsages| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | extra,
        })
    };
    let cs = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let pts_buf = storage(bytemuck::cast_slice(&pt_flat), wgpu::BufferUsages::COPY_DST);
    let tris_buf = storage(bytemuck::cast_slice(&tris0), cs);
    let assoc_buf = storage(bytemuck::cast_slice(&assoc0), wgpu::BufferUsages::COPY_DST);
    let nominee_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (max_t * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let split_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (max_t * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let counter_buf = storage(bytemuck::cast_slice(&[1u32, 0u32]), cs); // [tri_count=1, on_edge=0]
    // dims[2] = the STABLE triangle count at the start of the round (the split-guard
    // limit); host rewrites it each round before submitting (counter[0] itself grows
    // mid-`split`, so it can't be the guard).
    let dims_buf = storage(
        bytemuck::cast_slice(&[n as u32, max_t as u32, 1u32]),
        wgpu::BufferUsages::COPY_DST,
    );

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rlx-geo construct"),
        source: wgpu::ShaderSource::Wgsl(CONSTRUCT_WGSL.into()),
    });
    let ent = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            ent(0, true),
            ent(1, false),
            ent(2, false),
            ent(3, false),
            ent(4, false),
            ent(5, false),
            ent(6, true),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipe = |ep: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(ep),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let p_clear = pipe("clear_nominee");
    let p_pick = pipe("pick");
    let p_split = pipe("split");
    let p_reassoc = pipe("reassoc");
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: tris_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: assoc_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: nominee_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: split_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: counter_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: dims_buf.as_entire_binding(),
            },
        ],
    });

    let gy = |threads: u32| threads.div_ceil(65536).max(1);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let read_count = || -> u32 {
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let v = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range().unwrap())[0];
        readback.unmap();
        v
    };

    let mut tri_count = 1u32;
    let cap_rounds = 64usize; // O(log n) expected; cap guards a pathological/on-edge stall
    let mut rounds = 0usize;
    for _ in 0..cap_rounds {
        rounds += 1;
        // Publish the stable round-start count for the split/clear guards.
        queue.write_buffer(&dims_buf, 8, bytemuck::cast_slice(&[tri_count]));
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut pass = |p: &wgpu::ComputePipeline, threads: u32| {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(p);
            cp.set_bind_group(0, &bind, &[]);
            cp.dispatch_workgroups(1024, gy(threads), 1);
        };
        pass(&p_clear, tri_count);
        pass(&p_pick, n as u32);
        pass(&p_split, tri_count);
        pass(&p_reassoc, n as u32);
        drop(pass);
        enc.copy_buffer_to_buffer(&counter_buf, 0, &readback, 0, 4);
        queue.submit(Some(enc.finish()));
        let new_count = read_count();
        if new_count == tri_count {
            break; // no split happened → every point inserted (or on-edge-stuck)
        }
        tri_count = new_count;
    }

    if std::env::var_os("GEO_CONSTRUCT_DEBUG").is_some() {
        let dbg = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&counter_buf, 0, &dbg, 0, 8);
        queue.submit(Some(enc.finish()));
        let slice = dbg.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let c = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range().unwrap()).to_vec();
        dbg.unmap();
        eprintln!(
            "[construct] rounds={rounds} tri_count={} on_edge_skipped={} (n={n}, expect ~{})",
            c[0],
            c[1],
            2 * n + 1
        );
    }

    // download the triangle list
    let out_rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (tri_count as u64) * 12,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&tris_buf, 0, &out_rb, 0, (tri_count as u64) * 12);
    queue.submit(Some(enc.finish()));
    let slice = out_rb.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let flat: Vec<u32> =
        bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range().unwrap()).to_vec();
    out_rb.unmap();

    let tris: Vec<[u32; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    Some((tris, ext))
}

/// Full on-device Delaunay: [`construct_gpu`] → flip → drop bounding-incident
/// triangles. Falls back to `None` (caller uses the CPU-seed path) when the span is
/// too large. `pl` is a cached flip pipeline (reused across calls).
pub fn delaunay_gpu_native(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pl: &crate::flip_gpu::FlipPipeline,
    points: &[[i32; 2]],
) -> Option<Vec<[u32; 3]>> {
    let n = points.len() as u32;
    let (seed, ext) = construct_gpu(device, queue, points)?;
    // Let the flip consolidate the super-incident triangles to the minimal hull fan
    // (freezing them instead keeps ALL of construction's slivers — a huge deficit).
    let flipped = crate::flip_gpu::flip_to_delaunay_gpu_with(device, queue, pl, &seed, &ext);
    // drop triangles touching a bounding vertex (id ≥ n)
    Some(
        flipped
            .into_iter()
            .filter(|t| t[0] < n && t[1] < n && t[2] < n)
            .collect(),
    )
}
