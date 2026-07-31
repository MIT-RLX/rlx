// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fully on-device Lawson flip loop. The triangle buffer stays resident on the
//! GPU across rounds; each round runs five compute passes —
//!   * `reset`         — clear ownership, the edge hash, twins, counter,
//!   * `build_hash`    — insert every half-edge into a GPU hash table
//!                       (race-safe `atomicCompareExchangeWeak` open addressing),
//!   * `resolve_twins` — pair the two half-edges in each slot → per-edge twin,
//!   * `mark`          — O(1) twin lookup; test convex + illegal (exact `orient`
//!                       / emulated-i64 `in_circle`); stake an independent-set
//!                       claim via `atomicMin` on both triangles,
//!   * `apply`         — flip fires iff it owns *both* triangles; rewrite them.
//! The host only reads back a 4-byte "flips this round" counter to decide when to
//! stop; the mesh itself is never round-tripped.
//!
//! Adjacency is O(T) per round (hash), not O(T²). Valid for coordinate spans
//! ≤ 29 609 (the `in_circle` i64 bound) and up to 65 535 points (16-bit edge
//! keys); wider needs 128-bit limbs / 64-bit keys.

use wgpu::util::DeviceExt;

const NONE: u32 = 0xffff_ffff;

const FLIP_WGSL: &str = r#"
const NONE: u32 = 0xffffffffu;

@group(0) @binding(0)  var<storage, read_write> tris:    array<u32>;
@group(0) @binding(1)  var<storage, read>       pts:     array<i32>;
@group(0) @binding(2)  var<storage, read_write> owner:   array<atomic<u32>>;
@group(0) @binding(3)  var<storage, read_write> cand_e:  array<u32>;
@group(0) @binding(4)  var<storage, read_write> cand_t1: array<u32>;
@group(0) @binding(5)  var<storage, read_write> cand_ok: array<u32>;
@group(0) @binding(6)  var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(7)  var<storage, read>       dims:    array<u32>;   // [T, N, H]
@group(0) @binding(8)  var<storage, read_write> he_key:  array<atomic<u32>>;
@group(0) @binding(9)  var<storage, read_write> he_a:    array<u32>;
@group(0) @binding(10) var<storage, read_write> he_b:    array<u32>;
@group(0) @binding(11) var<storage, read_write> twin:    array<u32>;   // per half-edge

fn px(v: u32) -> i32 { return pts[v * 2u]; }
fn py(v: u32) -> i32 { return pts[v * 2u + 1u]; }
fn tv(t: u32, k: u32) -> u32 { return tris[t * 3u + k]; }

fn orient(ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32) -> i32 {
    let d = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (d > 0) { return 1; }
    if (d < 0) { return -1; }
    return 0;
}

// --- emulated signed 64-bit (vec2<u32> = (lo, hi)) for the in-circle determinant ---
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
fn in_circle(va: u32, vb: u32, vc: u32, vd: u32) -> i32 {
    let dx = px(vd); let dy = py(vd);
    let ax = px(va) - dx; let ay = py(va) - dy;
    let bx = px(vb) - dx; let by = py(vb) - dy;
    let cx = px(vc) - dx; let cy = py(vc) - dy;
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    var det = mul_i32(a2, bx * cy - cx * by);
    det = add_i64(det, neg_i64(mul_i32(b2, ax * cy - cx * ay)));
    det = add_i64(det, mul_i32(c2, ax * by - bx * ay));
    return sign_i64(det);
}

fn write_ccw(t: u32, x: u32, y: u32, z: u32) {
    if (orient(px(x), py(x), px(y), py(y), px(z), py(z)) < 0) {
        tris[t * 3u] = x; tris[t * 3u + 1u] = z; tris[t * 3u + 2u] = y;
    } else {
        tris[t * 3u] = x; tris[t * 3u + 1u] = y; tris[t * 3u + 2u] = z;
    }
}

// Local edge e of triangle t -> its two endpoints (u,w) and opposite apex p.
fn edge_of(t: u32, e: u32) -> vec3<u32> {
    let v0 = tv(t, 0u); let v1 = tv(t, 1u); let v2 = tv(t, 2u);
    if (e == 0u) { return vec3<u32>(v0, v1, v2); }
    if (e == 1u) { return vec3<u32>(v1, v2, v0); }
    return vec3<u32>(v2, v0, v1);
}

@compute @workgroup_size(64)
fn reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let T = dims[0]; let H = dims[2];
    if (i < H) { atomicStore(&he_key[i], NONE); he_b[i] = NONE; }
    if (i < 3u * T) { twin[i] = NONE; }
    if (i < T) { atomicStore(&owner[i], NONE); cand_ok[i] = 0u; }
    if (i == 0u) { atomicStore(&counter[0], 0u); }
}

@compute @workgroup_size(64)
fn build_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hid = gid.x;                       // half-edge id = t*3 + e
    let T = dims[0]; let H = dims[2];
    if (hid >= 3u * T) { return; }
    let uw = edge_of(hid / 3u, hid % 3u);
    let lo = min(uw.x, uw.y);
    let hi = max(uw.x, uw.y);
    let ek = (lo << 16u) | hi;             // unique per undirected edge (n < 2^16)
    var h = (ek * 2654435761u) & (H - 1u);
    loop {
        let r = atomicCompareExchangeWeak(&he_key[h], NONE, ek);
        if (r.exchanged) { he_a[h] = hid; return; }   // first occupant
        if (r.old_value == ek) { he_b[h] = hid; return; } // twin
        if (r.old_value == NONE) { continue; }         // spurious weak fail
        h = (h + 1u) & (H - 1u);                        // collision, probe
    }
}

@compute @workgroup_size(64)
fn resolve_twins(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= dims[2]) { return; }
    if (atomicLoad(&he_key[h]) == NONE) { return; }
    let kb = he_b[h];
    if (kb == NONE) { return; }             // boundary edge, no twin
    let ka = he_a[h];
    twin[ka] = kb / 3u;                      // twin[half-edge] = other triangle
    twin[kb] = ka / 3u;
}

@compute @workgroup_size(64)
fn mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t0 = gid.x;
    if (t0 >= dims[0]) { return; }
    let v0 = tv(t0, 0u); let v1 = tv(t0, 1u); let v2 = tv(t0, 2u);
    for (var e: u32 = 0u; e < 3u; e = e + 1u) {
        let t1 = twin[t0 * 3u + e];
        if (t1 == NONE || t0 >= t1) { continue; }   // boundary, or record once (lower t)
        let uw = edge_of(t0, e);
        let u = uw.x; let w = uw.y; let p = uw.z;
        let b0 = tv(t1, 0u); let b1 = tv(t1, 1u); let b2 = tv(t1, 2u);
        var q: u32;
        if (b0 != u && b0 != w) { q = b0; }
        else if (b1 != u && b1 != w) { q = b1; }
        else { q = b2; }
        let s1 = orient(px(p), py(p), px(q), py(q), px(u), py(u));
        let s2 = orient(px(p), py(p), px(q), py(q), px(w), py(w));
        if (s1 != 0 && s2 != 0 && (s1 < 0) != (s2 < 0)) {      // convex quad
            if (in_circle(v0, v1, v2, q) > 0) {                // illegal
                cand_e[t0] = e; cand_t1[t0] = t1; cand_ok[t0] = 1u;
                let id = t0 * 3u + e;
                atomicMin(&owner[t0], id);
                atomicMin(&owner[t1], id);
                return;
            }
        }
    }
}

@compute @workgroup_size(64)
fn apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t0 = gid.x;
    if (t0 >= dims[0]) { return; }
    if (cand_ok[t0] != 1u) { return; }
    let e = cand_e[t0];
    let t1 = cand_t1[t0];
    let id = t0 * 3u + e;
    if (atomicLoad(&owner[t0]) != id) { return; }
    if (atomicLoad(&owner[t1]) != id) { return; }      // won both triangles
    let uw = edge_of(t0, e);
    let a = uw.x; let b = uw.y; let p = uw.z;
    let b0 = tv(t1, 0u); let b1 = tv(t1, 1u); let b2 = tv(t1, 2u);
    var q: u32;
    if (b0 != a && b0 != b) { q = b0; }
    else if (b1 != a && b1 != b) { q = b1; }
    else { q = b2; }
    write_ccw(t0, a, p, q);      // diagonal a-b -> p-q
    write_ccw(t1, b, p, q);
    atomicAdd(&counter[0], 1u);
}
"#;

fn storage(dev: &wgpu::Device, data: &[u8], extra: wgpu::BufferUsages) -> wgpu::Buffer {
    dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage: wgpu::BufferUsages::STORAGE | extra,
    })
}

/// Run the flip loop entirely on the GPU. `tris` must be a valid triangulation
/// of `points` (CCW). Returns the Delaunay triangles.
pub fn flip_to_delaunay_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tris: &[[u32; 3]],
    points: &[[i32; 2]],
) -> Vec<[u32; 3]> {
    let t_count = tris.len() as u32;
    let n = points.len() as u32;
    assert!(
        n < (1 << 16),
        "flip_to_delaunay_gpu: n must be < 65536 (16-bit edge keys)"
    );
    if t_count == 0 {
        return Vec::new();
    }
    let hash_size = (4 * t_count).next_power_of_two().max(64);

    let tri_flat: Vec<u32> = tris.iter().flat_map(|t| [t[0], t[1], t[2]]).collect();
    let pt_flat: Vec<i32> = points.iter().flat_map(|p| [p[0], p[1]]).collect();
    let z_t = vec![0u32; t_count as usize];
    let z_3t = vec![0u32; 3 * t_count as usize];
    let z_h = vec![0u32; hash_size as usize];
    let cd = wgpu::BufferUsages::COPY_DST;
    let cs = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    let tris_buf = storage(device, bytemuck::cast_slice(&tri_flat), cs);
    let pts_buf = storage(device, bytemuck::cast_slice(&pt_flat), cd);
    let owner = storage(device, bytemuck::cast_slice(&z_t), cd);
    let cand_e = storage(device, bytemuck::cast_slice(&z_t), cd);
    let cand_t1 = storage(device, bytemuck::cast_slice(&z_t), cd);
    let cand_ok = storage(device, bytemuck::cast_slice(&z_t), cd);
    let counter = storage(device, bytemuck::cast_slice(&[0u32]), cs);
    let dims = storage(device, bytemuck::cast_slice(&[t_count, n, hash_size]), cd);
    let he_key = storage(device, bytemuck::cast_slice(&z_h), cd);
    let he_a = storage(device, bytemuck::cast_slice(&z_h), cd);
    let he_b = storage(device, bytemuck::cast_slice(&z_h), cd);
    let twin = storage(device, bytemuck::cast_slice(&z_3t), cd);

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rlx-geo flip"),
        source: wgpu::ShaderSource::Wgsl(FLIP_WGSL.into()),
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
        label: Some("rlx-geo flip"),
        entries: &[
            ent(0, false),
            ent(1, true),
            ent(2, false),
            ent(3, false),
            ent(4, false),
            ent(5, false),
            ent(6, false),
            ent(7, true),
            ent(8, false),
            ent(9, false),
            ent(10, false),
            ent(11, false),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rlx-geo flip"),
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
    let p_reset = pipe("reset");
    let p_hash = pipe("build_hash");
    let p_resolve = pipe("resolve_twins");
    let p_mark = pipe("mark");
    let p_apply = pipe("apply");

    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-geo flip"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tris_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: owner.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cand_e.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: cand_t1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: cand_ok.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: counter.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: dims.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: he_key.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: he_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: he_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: twin.as_entire_binding(),
            },
        ],
    });

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("counter"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let g_t = t_count.div_ceil(64);
    let g_3t = (3 * t_count).div_ceil(64);
    let g_h = hash_size.div_ceil(64);
    // (pipeline, groups) for each pass, in order.
    let passes = [
        (&p_reset, g_h.max(g_3t).max(g_t)),
        (&p_hash, g_3t),
        (&p_resolve, g_h),
        (&p_mark, g_t),
        (&p_apply, g_t),
    ];

    // The loop is latency-bound on the per-round counter read-back, so run BATCH
    // rounds per submit and read the counter once. The counter holds the LAST
    // round's flip count (it's reset each round); 0 ⇒ converged. Extra rounds
    // after convergence are cheap no-ops.
    let batch: usize = std::env::var("GEO_FLIP_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let batch = batch.max(1);
    let cap = 4 * (t_count as usize) + 64;
    let mut round = 0usize;
    while round < cap {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..batch {
            for (pl, groups) in passes {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cp.set_pipeline(pl);
                cp.set_bind_group(0, &bind, &[]);
                cp.dispatch_workgroups(groups, 1, 1);
            }
            round += 1;
        }
        enc.copy_buffer_to_buffer(&counter, 0, &readback, 0, 4);
        queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let flips = {
            let d = slice.get_mapped_range().unwrap();
            bytemuck::cast_slice::<u8, u32>(&d)[0]
        };
        readback.unmap();
        if flips == 0 {
            break;
        }
    }

    // Download the final mesh once.
    let out_rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tris"),
        size: (tri_flat.len() * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&tris_buf, 0, &out_rb, 0, (tri_flat.len() * 4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = out_rb.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let out: Vec<u32> = {
        let d = slice.get_mapped_range().unwrap();
        bytemuck::cast_slice(&d).to_vec()
    };
    out_rb.unmap();
    let _ = NONE;
    out.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}
