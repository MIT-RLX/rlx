// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Dumps a REAL Delaunay mesh (points + triangles + twin adjacency) for the HIP
// flip-floor microbenchmark, so the MI100/780M measurement uses the true spatially-
// sorted access pattern of the flip (not random gathers).
// Format: [u64 N][u64 T] [i32 x,y]×N [u32 a,b,c]×T [u32 t0,t1,t2]×T
//   cargo run -p rlx-geo --example dump_mesh --release -- <in.pts> <out.mesh>
use rlx_geo::triangulate;
use std::collections::HashMap;
use std::io::{Read, Write};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut buf = Vec::new();
    std::fs::File::open(&a[1])
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let n = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
    let mut pts = Vec::with_capacity(n);
    let mut o = 8;
    for _ in 0..n {
        let x = i32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
        let y = i32::from_le_bytes(buf[o + 4..o + 8].try_into().unwrap());
        pts.push([x, y]);
        o += 8;
    }
    let tris = triangulate(&pts).unwrap();
    // build twin: for each (tri,edge) the adjacent tri across that edge (or u32::MAX)
    let mut edge: HashMap<(u32, u32), Vec<(usize, usize)>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for e in 0..3 {
            let (x, y) = (t[e], t[(e + 1) % 3]);
            let k = if x < y { (x, y) } else { (y, x) };
            edge.entry(k).or_default().push((ti, e));
        }
    }
    let mut twin = vec![[u32::MAX; 3]; tris.len()];
    for recs in edge.values() {
        if recs.len() == 2 {
            twin[recs[0].0][recs[0].1] = recs[1].0 as u32;
            twin[recs[1].0][recs[1].1] = recs[0].0 as u32;
        }
    }
    let t = tris.len();
    let mut out = Vec::with_capacity(16 + n * 8 + t * 24);
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&(t as u64).to_le_bytes());
    for p in &pts {
        out.extend_from_slice(&p[0].to_le_bytes());
        out.extend_from_slice(&p[1].to_le_bytes());
    }
    for tr in &tris {
        for &v in tr {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    for tw in &twin {
        for &v in tw {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::File::create(&a[2])
        .unwrap()
        .write_all(&out)
        .unwrap();
    println!("dumped N={n} T={t} -> {}", a[2]);
}
