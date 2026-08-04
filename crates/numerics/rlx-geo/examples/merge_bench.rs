// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Milestone 2 of recursive-tile GPU D&C: MERGE two x-split leaf-Delaunay tiles into
// the Delaunay of their union. Reuses the on-chip leaf result: triangulate only the
// GAP between the two hulls (ladder between the common tangents), combine, and flip —
// the flip only touches the seam since each side is already Delaunay. Validates exact.
//   cargo run -p rlx-geo --example merge_bench --no-default-features --features gpu --release

use rlx_geo::flip_gpu::flip_to_delaunay_gpu;
use rlx_geo::{hull_seed, triangulate};
use std::collections::{HashMap, HashSet};

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

/// CCW convex hull (monotone chain) of a point subset, returning global point ids.
fn hull(pts: &[[i32; 2]], ids: &[u32]) -> Vec<u32> {
    let mut o = ids.to_vec();
    o.sort_by_key(|&i| (pts[i as usize][0], pts[i as usize][1]));
    o.dedup_by_key(|&mut i| pts[i as usize]);
    if o.len() < 3 {
        return o;
    }
    let cr = |a: u32, b: u32, c: u32| orient(pts[a as usize], pts[b as usize], pts[c as usize]);
    let mut lo: Vec<u32> = vec![];
    for &p in &o {
        while lo.len() >= 2 && cr(lo[lo.len() - 2], lo[lo.len() - 1], p) <= 0 {
            lo.pop();
        }
        lo.push(p);
    }
    let mut up: Vec<u32> = vec![];
    for &p in o.iter().rev() {
        while up.len() >= 2 && cr(up[up.len() - 2], up[up.len() - 1], p) <= 0 {
            up.pop();
        }
        up.push(p);
    }
    lo.pop();
    up.pop();
    lo.extend(up);
    lo // CCW
}

// Leaf via the flat pipeline (leaf kernel is validated separately; here we only test the
// MERGE, so producing each side's Delaunay by any exact means is fine).
fn side_delaunay(pts: &[[i32; 2]], ids: &[u32]) -> Vec<[u32; 3]> {
    let sub: Vec<[i32; 2]> = ids.iter().map(|&i| pts[i as usize]).collect();
    triangulate(&sub)
        .unwrap()
        .into_iter()
        .map(|t| [ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize]])
        .collect()
}

fn main() {
    // 256 points, split by x into L (left half) and R (right half).
    let mut s = 0xdead_beef_cafe_1234u64;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (s >> 33) as u32
    };
    let mut seen = HashSet::new();
    let mut pts: Vec<[i32; 2]> = vec![];
    while pts.len() < 400 {
        let p = [(next() % 100000) as i32, (next() % 100000) as i32];
        if seen.insert(p) {
            pts.push(p);
        }
    }
    let n = pts.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&i| pts[i as usize][0]);
    let mid = n / 2;
    let l_ids: Vec<u32> = order[..mid].to_vec();
    let r_ids: Vec<u32> = order[mid..].to_vec();

    // each side's Delaunay (the leaf result, reused)
    let l_tris = side_delaunay(&pts, &l_ids);
    let r_tris = side_delaunay(&pts, &r_ids);

    // hulls of each side; L's right-facing chain + R's left-facing chain bound the gap.
    let lh = hull(&pts, &l_ids);
    let rh = hull(&pts, &r_ids);

    // --- common tangents by brute force (hulls are tiny). Lower tangent (l,r): the
    // line l->r with EVERY hull point on/above it (orient(l,r,p) >= 0). Upper: all below.
    let (lm, rm) = (lh.len(), rh.len());
    let all_pts: Vec<u32> = lh.iter().chain(rh.iter()).copied().collect();
    let find_tan = |lower: bool| -> (usize, usize) {
        for i in 0..lm {
            for j in 0..rm {
                let ok = all_pts.iter().all(|&p| {
                    let o = orient(pts[lh[i] as usize], pts[rh[j] as usize], pts[p as usize]);
                    if lower { o >= 0 } else { o <= 0 }
                });
                if ok {
                    return (i, j);
                }
            }
        }
        (0, 0)
    };
    let (lo_l, lo_r) = find_tan(true);
    let (hi_l, hi_r) = find_tan(false);

    // --- ladder: zip up the gap from lower to upper tangent, advancing the side whose
    // next hull vertex keeps the cross-edge lower (a valid, not-yet-Delaunay gap tri) ---
    // Both chains are y-monotone (increasing) from the lower to the upper tangent:
    // L's right chain goes CCW (idx+1), R's left chain goes CW (idx-1). Zip up by y.
    let mut gap: Vec<[u32; 3]> = vec![];
    let (mut cl, mut cr) = (lo_l, lo_r);
    while cl != hi_l || cr != hi_r {
        let nl = (cl + 1) % lm; // up L (CCW)
        let nr = (cr + rm - 1) % rm; // up R (CW)
        let can_l = cl != hi_l;
        let can_r = cr != hi_r;
        let take_l = can_l && (!can_r || pts[lh[nl] as usize][1] <= pts[rh[nr] as usize][1]);
        if take_l {
            gap.push([lh[cl], lh[nl], rh[cr]]);
            cl = nl;
        } else {
            gap.push([lh[cl], rh[nr], rh[cr]]);
            cr = nr;
        }
    }

    // combine + flip (only the gap is non-Delaunay)
    let mut combined: Vec<[u32; 3]> = l_tris
        .iter()
        .chain(r_tris.iter())
        .chain(gap.iter())
        .copied()
        .collect();
    // ensure CCW
    for t in combined.iter_mut() {
        if orient(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]) < 0 {
            t.swap(1, 2);
        }
    }

    let instance = wgpu::Instance::default();
    let all = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    let adapter = {
        let pick = |t: wgpu::DeviceType| all.iter().position(|a| a.get_info().device_type == t);
        let idx = pick(wgpu::DeviceType::DiscreteGpu)
            .or_else(|| pick(wgpu::DeviceType::IntegratedGpu))
            .unwrap_or(0);
        all.into_iter().nth(idx).unwrap()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();

    let refc = triangulate(&pts).unwrap().len();
    println!(
        "n={n} L={} R={}  L_tris={} R_tris={} gap={} combined={}",
        l_ids.len(),
        r_ids.len(),
        l_tris.len(),
        r_tris.len(),
        gap.len(),
        combined.len()
    );
    // combined valid-triangulation sanity (count should be 2n-2-h)
    let merged = flip_to_delaunay_gpu(&device, &queue, &combined, &pts);

    // validate = Delaunay(union)
    let mut used = vec![false; n];
    let mut edges: HashMap<u64, Vec<(u32, u32, u32)>> = HashMap::new();
    let mut ok = merged.len() == refc;
    for t in &merged {
        for &i in t {
            used[i as usize] = true;
        }
        if orient(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]) <= 0 {
            ok = false;
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
    for recs in edges.values() {
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
                ok = false;
            }
        }
    }
    ok &= used.iter().all(|&u| u);
    println!("merged_tris={} ref_tris={refc}", merged.len());
    println!(
        "validate MERGE: {}",
        if ok {
            "EXACT Delaunay of union OK"
        } else {
            "FAIL"
        }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
