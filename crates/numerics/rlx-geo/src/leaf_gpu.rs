// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Leaf solver for the recursive-tile GPU divide-and-conquer Delaunay (milestone 1):
//! one WORKGROUP cooperatively computes the exact Delaunay of a small tile of points
//! ENTIRELY IN SHARED MEMORY. The whole Lawson flip (mark/apply/fixup, `workgroupBarrier`
//! between phases, shared-memory adjacency + independent-set atomics) runs on-chip — so
//! all its rounds cost ZERO DRAM traffic (the mesh is loaded once, written once). This is
//! the on-chip mechanism the recursive-tile analysis said handles ~90% of the work.
//!
//! Predicates are i64 (exact while the tile's coordinate SPAN keeps the degree-4 in-circle
//! determinant < 2^63, i.e. span ≲ 30 000 — true for a spatial tile after subtracting its
//! origin; the driver rebases). The recursive seam-merge (milestone 2) is not here.

use wgpu::util::DeviceExt;

const NONE: u32 = 0xffff_ffff;
pub const TILE: usize = 128; // points per leaf tile
const MAXT: usize = 2 * TILE; // ≤ 2m triangles

const LEAF_WGSL: &str = r#"
const NONE: u32 = 0xffffffffu;
const TILE: u32 = 128u;
const MAXT: u32 = 256u;

@group(0) @binding(0) var<storage, read>       pts:   array<i32>;   // 2*TILE per tile (rebased)
@group(0) @binding(1) var<storage, read_write> tris:  array<u32>;   // 3*MAXT per tile (local ids); [-1]=count
@group(0) @binding(2) var<storage, read>       tw0:   array<u32>;   // 3*MAXT per tile: seed twin
@group(0) @binding(3) var<storage, read>       ntri:  array<u32>;   // per tile seed triangle count

var<workgroup> spx: array<i32, TILE>;
var<workgroup> spy: array<i32, TILE>;
var<workgroup> stri: array<u32, 768>;   // 3*MAXT
var<workgroup> stwin: array<u32, 768>;
var<workgroup> sowner: array<atomic<u32>, 256>;   // MAXT
var<workgroup> scand_e: array<u32, 256>;
var<workgroup> scand_t1: array<u32, 256>;
var<workgroup> scand_ok: array<u32, 256>;
var<workgroup> shea: array<u32, 256>;
var<workgroup> sheb: array<u32, 256>;
var<workgroup> sN: u32;
var<workgroup> snflip: atomic<u32>;

fn lpx(v: u32) -> i32 { return spx[v]; }
fn lpy(v: u32) -> i32 { return spy[v]; }
fn tv(t: u32, k: u32) -> u32 { return stri[t * 3u + k]; }
// WGSL has no native i64 → emulate (vec2<u32>). For a tile span < ~29600 the orient
// cross product fits i32 and the in-circle degree-4 determinant fits i64 (no i128).
fn mul_u32(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xffffu; let ah = a >> 16u; let bl = b & 0xffffu; let bh = b >> 16u;
    let ll = al*bl; let lh = al*bh; let hl = ah*bl; let hh = ah*bh;
    let cross = lh + hl; let cc = select(0u, 1u, cross < lh);
    let lo = ll + (cross << 16u); let lc = select(0u, 1u, lo < ll);
    return vec2<u32>(lo, hh + (cross >> 16u) + (cc << 16u) + lc);
}
fn neg_i64(x: vec2<u32>) -> vec2<u32> { let lo = ~x.x + 1u; return vec2<u32>(lo, ~x.y + select(0u,1u,lo==0u)); }
fn mul_i32(a: i32, b: i32) -> vec2<u32> { let r = mul_u32(u32(abs(a)), u32(abs(b))); if ((a<0)!=(b<0)) { return neg_i64(r); } return r; }
fn add_i64(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> { let lo = x.x+y.x; return vec2<u32>(lo, x.y+y.y+select(0u,1u,lo<x.x)); }
fn sign_i64(x: vec2<u32>) -> i32 { let hi = bitcast<i32>(x.y); if (hi<0) { return -1; } if (hi>0) { return 1; } if (x.x!=0u) { return 1; } return 0; }
// orient in native i32 (products < 2^31 for span < ~32000)
fn orient(a: u32, b: u32, c: u32) -> i32 {
    let d = (lpx(b) - lpx(a)) * (lpy(c) - lpy(a)) - (lpy(b) - lpy(a)) * (lpx(c) - lpx(a));
    if (d < 0) { return -1; } if (d > 0) { return 1; } return 0;
}
fn in_circle(a: u32, b: u32, c: u32, d: u32) -> i32 {
    let ax = lpx(a) - lpx(d); let ay = lpy(a) - lpy(d);
    let bx = lpx(b) - lpx(d); let by = lpy(b) - lpy(d);
    let cx = lpx(c) - lpx(d); let cy = lpy(c) - lpy(d);
    let a2 = ax*ax + ay*ay; let b2 = bx*bx + by*by; let c2 = cx*cx + cy*cy;   // < 2^31
    let m_bc = bx*cy - cx*by; let m_ac = ax*cy - cx*ay; let m_ab = ax*by - bx*ay; // < 2^31
    var det = mul_i32(a2, m_bc);
    det = add_i64(det, neg_i64(mul_i32(b2, m_ac)));
    det = add_i64(det, mul_i32(c2, m_ab));
    return sign_i64(det);
}
fn edge_of(t: u32, e: u32) -> vec3<u32> {
    let v0 = tv(t, 0u); let v1 = tv(t, 1u); let v2 = tv(t, 2u);
    if (e == 0u) { return vec3<u32>(v0, v1, v2); }
    if (e == 1u) { return vec3<u32>(v1, v2, v0); }
    return vec3<u32>(v2, v0, v1);
}
fn find_edge(t: u32, a: u32, b: u32) -> u32 {
    for (var e = 0u; e < 3u; e = e + 1u) {
        let x = tv(t, e); let y = tv(t, (e + 1u) % 3u);
        if ((x == a && y == b) || (x == b && y == a)) { return e; }
    }
    return 0u;
}
fn has_edge(t: u32, a: u32, b: u32) -> bool {
    let ha = (tv(t, 0u) == a) || (tv(t, 1u) == a) || (tv(t, 2u) == a);
    let hb = (tv(t, 0u) == b) || (tv(t, 1u) == b) || (tv(t, 2u) == b);
    return ha && hb;
}
fn write_ccw(t: u32, x: u32, y: u32, z: u32) {
    if (orient(x, y, z) < 0) { stri[t*3u]=x; stri[t*3u+1u]=z; stri[t*3u+2u]=y; }
    else { stri[t*3u]=x; stri[t*3u+1u]=y; stri[t*3u+2u]=z; }
}

@compute @workgroup_size(128)
fn leaf(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile = wid.x;
    let tid = lid.x;
    let pbase = tile * TILE * 2u;
    let tbase = tile * MAXT * 3u;
    // --- load tile into shared memory ---
    if (tid < TILE) { spx[tid] = pts[pbase + tid * 2u]; spy[tid] = pts[pbase + tid * 2u + 1u]; }
    if (tid == 0u) { sN = ntri[tile]; }
    workgroupBarrier();
    let nt = sN;
    for (var i = tid; i < nt; i = i + 128u) {
        stri[i*3u] = tris[tbase + i*3u]; stri[i*3u+1u] = tris[tbase + i*3u+1u]; stri[i*3u+2u] = tris[tbase + i*3u+2u];
        stwin[i*3u] = tw0[tbase + i*3u]; stwin[i*3u+1u] = tw0[tbase + i*3u+1u]; stwin[i*3u+2u] = tw0[tbase + i*3u+2u];
    }
    workgroupBarrier();

    // --- cooperative Lawson flip, entirely in shared memory ---
    var iter = 0u;
    loop {
        if (tid == 0u) { atomicStore(&snflip, 0u); }
        for (var i = tid; i < nt; i = i + 128u) { atomicStore(&sowner[i], NONE); scand_ok[i] = 0u; sheb[i] = 0u; }
        workgroupBarrier();
        // mark
        for (var t0 = tid; t0 < nt; t0 = t0 + 128u) {
            for (var e = 0u; e < 3u; e = e + 1u) {
                let t1 = stwin[t0*3u + e];
                if (t1 == NONE || t0 >= t1) { continue; }
                let uw = edge_of(t0, e); let u = uw.x; let w = uw.y; let p = uw.z;
                let b0 = tv(t1,0u); let b1 = tv(t1,1u); let b2 = tv(t1,2u);
                var q: u32; if (b0!=u && b0!=w) { q=b0; } else if (b1!=u && b1!=w) { q=b1; } else { q=b2; }
                let s1 = orient(p, q, u); let s2 = orient(p, q, w);
                if (s1 != 0 && s2 != 0 && (s1 < 0) != (s2 < 0)) {
                    if (in_circle(tv(t0,0u), tv(t0,1u), tv(t0,2u), q) > 0) {
                        scand_e[t0]=e; scand_t1[t0]=t1; scand_ok[t0]=1u;
                        let id = t0*3u+e; atomicMin(&sowner[t0], id); atomicMin(&sowner[t1], id);
                        break;
                    }
                }
            }
        }
        workgroupBarrier();
        // apply
        for (var t0 = tid; t0 < nt; t0 = t0 + 128u) {
            if (scand_ok[t0] != 1u) { continue; }
            let e = scand_e[t0]; let t1 = scand_t1[t0]; let id = t0*3u+e;
            if (atomicLoad(&sowner[t0]) != id || atomicLoad(&sowner[t1]) != id) { continue; }
            let uw = edge_of(t0, e); let u = uw.x; let w = uw.y; let p = uw.z;
            let b0 = tv(t1,0u); let b1 = tv(t1,1u); let b2 = tv(t1,2u);
            var q: u32; if (b0!=u && b0!=w) { q=b0; } else if (b1!=u && b1!=w) { q=b1; } else { q=b2; }
            let n_pu = stwin[t0*3u + find_edge(t0,p,u)];
            let n_wp = stwin[t0*3u + find_edge(t0,w,p)];
            let n_uq = stwin[t1*3u + find_edge(t1,u,q)];
            let n_qw = stwin[t1*3u + find_edge(t1,w,q)];
            write_ccw(t0, u, p, q); write_ccw(t1, w, p, q);
            stwin[t0*3u + find_edge(t0,u,p)] = n_pu;
            stwin[t0*3u + find_edge(t0,q,u)] = n_uq;
            stwin[t0*3u + find_edge(t0,p,q)] = t1;
            stwin[t1*3u + find_edge(t1,w,p)] = n_wp;
            stwin[t1*3u + find_edge(t1,q,w)] = n_qw;
            stwin[t1*3u + find_edge(t1,p,q)] = t0;
            shea[t0]=t1; shea[t1]=t0; sheb[t0]=1u; sheb[t1]=1u;
            atomicAdd(&snflip, 1u);
        }
        workgroupBarrier();
        // fixup
        for (var t = tid; t < nt; t = t + 128u) {
            for (var e = 0u; e < 3u; e = e + 1u) {
                let n = stwin[t*3u+e];
                if (n == NONE) { continue; }
                if (sheb[n] != 1u) { continue; }
                let a = tv(t,e); let b = tv(t,(e+1u)%3u);
                if (has_edge(n, a, b)) { stwin[t*3u+e] = n; } else { stwin[t*3u+e] = shea[n]; }
            }
        }
        workgroupBarrier();
        iter = iter + 1u;
        if (atomicLoad(&snflip) == 0u || iter > 4u * TILE) { break; }
        workgroupBarrier();
    }
    // --- write triangles back ---
    for (var i = tid; i < nt; i = i + 128u) {
        tris[tbase + i*3u] = stri[i*3u]; tris[tbase + i*3u+1u] = stri[i*3u+1u]; tris[tbase + i*3u+2u] = stri[i*3u+2u];
    }
}
"#;

/// Cooperative shared-memory Delaunay of ONE tile's points (milestone-1 driver: single
/// workgroup). `points` (≤ TILE) already rebased to a small span; `seed`/`seed_twin` a
/// valid CCW triangulation + adjacency. Returns the flipped (Delaunay) triangles.
pub fn leaf_delaunay(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    points: &[[i32; 2]],
    seed: &[[u32; 3]],
    seed_twin: &[[u32; 3]],
) -> Vec<[u32; 3]> {
    let nt = seed.len();
    assert!(points.len() <= TILE && nt <= MAXT);
    let mut pt = vec![0i32; TILE * 2];
    for (i, p) in points.iter().enumerate() {
        pt[2 * i] = p[0];
        pt[2 * i + 1] = p[1];
    }
    let mut tri = vec![0u32; MAXT * 3];
    let mut tw = vec![NONE; MAXT * 3];
    for (i, t) in seed.iter().enumerate() {
        tri[3 * i] = t[0];
        tri[3 * i + 1] = t[1];
        tri[3 * i + 2] = t[2];
        tw[3 * i] = seed_twin[i][0];
        tw[3 * i + 1] = seed_twin[i][1];
        tw[3 * i + 2] = seed_twin[i][2];
    }
    let stor = |d: &[u8], u: wgpu::BufferUsages| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: d,
            usage: wgpu::BufferUsages::STORAGE | u,
        })
    };
    let pts_buf = stor(bytemuck::cast_slice(&pt), wgpu::BufferUsages::COPY_DST);
    let tris_buf = stor(
        bytemuck::cast_slice(&tri),
        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let tw_buf = stor(bytemuck::cast_slice(&tw), wgpu::BufferUsages::COPY_DST);
    let n_buf = stor(
        bytemuck::cast_slice(&[nt as u32]),
        wgpu::BufferUsages::COPY_DST,
    );

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("leaf"),
        source: wgpu::ShaderSource::Wgsl(LEAF_WGSL.into()),
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
        entries: &[ent(0, true), ent(1, false), ent(2, true), ent(3, true)],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some("leaf"),
        compilation_options: Default::default(),
        cache: None,
    });
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
                resource: tw_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: n_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &bind, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (MAXT * 3 * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&tris_buf, 0, &rb, 0, (MAXT * 3 * 4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = rb.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let flat: Vec<u32> =
        bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range().unwrap()).to_vec();
    rb.unmap();
    flat[..nt * 3]
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

/// Throughput probe: replicate one tile across `n_tiles` workgroups (all solved
/// independently on-chip, in parallel) and return the best wall-clock ms over `runs`.
/// Measures how fast the leaf phase processes n_tiles·TILE points with zero inter-tile
/// DRAM traffic — the recursive scheme's dominant (90%) phase.
pub fn leaf_throughput(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    points: &[[i32; 2]],
    seed: &[[u32; 3]],
    seed_twin: &[[u32; 3]],
    n_tiles: u32,
    runs: usize,
) -> f64 {
    let nt = seed.len();
    // replicate the tile's data across n_tiles slots
    let mut pt = vec![0i32; n_tiles as usize * TILE * 2];
    let mut tri = vec![0u32; n_tiles as usize * MAXT * 3];
    let mut tw = vec![NONE; n_tiles as usize * MAXT * 3];
    let mut ntv = vec![0u32; n_tiles as usize];
    for tile in 0..n_tiles as usize {
        for (i, p) in points.iter().enumerate() {
            pt[tile * TILE * 2 + 2 * i] = p[0];
            pt[tile * TILE * 2 + 2 * i + 1] = p[1];
        }
        for (i, t) in seed.iter().enumerate() {
            let b = tile * MAXT * 3 + 3 * i;
            tri[b] = t[0];
            tri[b + 1] = t[1];
            tri[b + 2] = t[2];
            tw[b] = seed_twin[i][0];
            tw[b + 1] = seed_twin[i][1];
            tw[b + 2] = seed_twin[i][2];
        }
        ntv[tile] = nt as u32;
    }
    let stor = |d: &[u8], u: wgpu::BufferUsages| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: d,
            usage: wgpu::BufferUsages::STORAGE | u,
        })
    };
    let pts_buf = stor(bytemuck::cast_slice(&pt), wgpu::BufferUsages::COPY_DST);
    let tris_buf = stor(bytemuck::cast_slice(&tri), wgpu::BufferUsages::COPY_DST);
    let tw_buf = stor(bytemuck::cast_slice(&tw), wgpu::BufferUsages::COPY_DST);
    let n_buf = stor(bytemuck::cast_slice(&ntv), wgpu::BufferUsages::COPY_DST);

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("leaf"),
        source: wgpu::ShaderSource::Wgsl(LEAF_WGSL.into()),
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
        entries: &[ent(0, true), ent(1, false), ent(2, true), ent(3, true)],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some("leaf"),
        compilation_options: Default::default(),
        cache: None,
    });
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
                resource: tw_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: n_buf.as_entire_binding(),
            },
        ],
    });
    let run = || {
        let t = std::time::Instant::now();
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipe);
            cp.set_bind_group(0, &bind, &[]);
            cp.dispatch_workgroups(n_tiles, 1, 1); // n_tiles ≤ 65535 for ≤ 8M points
        }
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        t.elapsed().as_secs_f64() * 1e3
    };
    for _ in 0..3 {
        run();
    }
    (0..runs).map(|_| run()).fold(f64::INFINITY, f64::min)
}
