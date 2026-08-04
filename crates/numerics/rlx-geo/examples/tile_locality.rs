// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Measures the CEILING for temporal-blocking the flip: if triangles are spatially
// (Morton) sorted and cut into fixed tiles, what fraction of mesh adjacencies are
// INTRA-tile? Intra-tile flips can stay in shared memory across rounds (the win);
// seam flips still hit DRAM. High intra fraction ⇒ shared-memory blocking pays off.
//   cargo run -p rlx-geo --example tile_locality --no-default-features --features gpu --release -- <file>

use rlx_geo::triangulate;
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

fn morton(x: u32, y: u32) -> u64 {
    let spread = |mut v: u64| {
        v = (v | (v << 16)) & 0x0000_ffff_0000_ffff;
        v = (v | (v << 8)) & 0x00ff_00ff_00ff_00ff;
        v = (v | (v << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
        v = (v | (v << 2)) & 0x3333_3333_3333_3333;
        (v | (v << 1)) & 0x5555_5555_5555_5555
    };
    spread(x as u64) | (spread(y as u64) << 1)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pts = read_points(&args[1]);
    let n = pts.len();
    let tris = triangulate(&pts).unwrap();
    let t = tris.len();

    // adjacency: undirected edge -> list of triangle ids sharing it
    let mut edges: HashMap<u64, Vec<u32>> = HashMap::with_capacity(t * 2);
    for (ti, tr) in tris.iter().enumerate() {
        for &(a, b) in &[(tr[0], tr[1]), (tr[1], tr[2]), (tr[2], tr[0])] {
            let k = if a < b {
                ((a as u64) << 32) | b as u64
            } else {
                ((b as u64) << 32) | a as u64
            };
            edges.entry(k).or_default().push(ti as u32);
        }
    }
    let interior: Vec<(u32, u32)> = edges
        .values()
        .filter(|v| v.len() == 2)
        .map(|v| (v[0], v[1]))
        .collect();

    // Morton rank of each triangle by centroid.
    let (mnx, mny) = pts
        .iter()
        .fold((i32::MAX, i32::MAX), |(a, b), p| (a.min(p[0]), b.min(p[1])));
    let mcode: Vec<u64> = tris
        .iter()
        .map(|tr| {
            let cx = (pts[tr[0] as usize][0] as i64
                + pts[tr[1] as usize][0] as i64
                + pts[tr[2] as usize][0] as i64)
                / 3
                - mnx as i64;
            let cy = (pts[tr[0] as usize][1] as i64
                + pts[tr[1] as usize][1] as i64
                + pts[tr[2] as usize][1] as i64)
                / 3
                - mny as i64;
            morton(cx as u32, cy as u32)
        })
        .collect();
    let mut order: Vec<u32> = (0..t as u32).collect();
    order.sort_by_key(|&i| mcode[i as usize]);
    let mut rank = vec![0u32; t]; // rank[tri] = position in Morton order
    for (r, &ti) in order.iter().enumerate() {
        rank[ti as usize] = r as u32;
    }

    let ne = interior.len() as f64;
    println!("n={n} tris={t} interior_edges={}", interior.len());
    println!("--- flat tiling (intra% = fraction of flips that stay on-chip) ---");
    for &tile in &[256usize, 1024, 4096] {
        let intra = interior
            .iter()
            .filter(|&&(a, b)| {
                (rank[a as usize] as usize / tile) == (rank[b as usize] as usize / tile)
            })
            .count();
        println!("  tile {tile:>6}: intra {:.1}%", intra as f64 / ne * 100.0);
    }

    // --- RECURSIVE tiling (GPU divide-and-conquer): leaf tiles of `base`, merged ×`branch`
    // per level. An edge is handled at the FIRST level where both triangles share a tile.
    // DRAM ≈ 1 sweep (leaves, all edges once, on-chip) + Σ_{merge levels} new_seam_frac × R_local.
    let base = 256usize;
    let branch = 4usize;
    let r_local: f64 = std::env::var("R_LOCAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8.0);
    println!(
        "--- recursive tiling (leaf={base}, branch={branch}, R_local={r_local}, flat flip ≈ 54 sweeps) ---"
    );
    let intra_at = |s: usize| {
        interior
            .iter()
            .filter(|&&(a, b)| (rank[a as usize] as usize / s) == (rank[b as usize] as usize / s))
            .count()
    };
    let mut prev_intra = 0usize;
    let mut sweeps = 1.0f64; // leaf load
    let (mut s, mut level) = (base, 0usize);
    loop {
        let intra = intra_at(s);
        let handled = intra - prev_intra;
        let frac = handled as f64 / ne;
        let ntiles = t.div_ceil(s);
        if level == 0 {
            println!(
                "  L0 leaf   tile={s:>7} tiles={ntiles:>7} (parallel)  handles {:.1}% on-chip",
                frac * 100.0
            );
        } else {
            sweeps += frac * r_local;
            println!(
                "  L{level} merge  tile={s:>7} merges={ntiles:>7} (parallel)  stitches {:.2}% seams",
                frac * 100.0
            );
        }
        prev_intra = intra;
        if ntiles <= 1 {
            break;
        }
        s *= branch;
        level += 1;
    }
    println!(
        "  => recursive DRAM ≈ {sweeps:.2} sweeps vs flat ~54 → {:.0}x less traffic (→ compute-bound)",
        54.0 / sweeps
    );
}
