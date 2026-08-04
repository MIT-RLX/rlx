// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Can an NPU (Apple ANE / AMD XDNA) run the Delaunay in-circle predicate? The ARITHMETIC
// maps to tensor ops (a batch of 4-point determinants = elementwise mul/add + reduction),
// but NPUs are FP16/BF16 (ANE) or INT8/BF16 (XDNA). This measures, on a REAL Delaunay
// mesh, what fraction of in-circle tests each float precision can CERTIFY (|det| beats the
// rounding error bound) — i.e. how useful an NPU-precision predicate FILTER would be.
// A test that can't be certified must fall back to the exact i128 predicate.
//   cargo run -p rlx-geo --example predicate_precision --release -- <mesh.bin>
use std::io::Read;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let mut buf = Vec::new();
    std::fs::File::open(&path)
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let rd_u64 = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
    let n = rd_u64(0) as usize;
    let t = rd_u64(8) as usize;
    let mut o = 16;
    let mut pts = Vec::with_capacity(n);
    for _ in 0..n {
        let x = i32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
        let y = i32::from_le_bytes(buf[o + 4..o + 8].try_into().unwrap());
        pts.push([x as f64, y as f64]);
        o += 8;
    }
    let tri: Vec<[u32; 3]> = (0..t)
        .map(|i| {
            let b = 16 + n * 8 + i * 12;
            [
                u32::from_le_bytes(buf[b..b + 4].try_into().unwrap()),
                u32::from_le_bytes(buf[b + 4..b + 8].try_into().unwrap()),
                u32::from_le_bytes(buf[b + 8..b + 12].try_into().unwrap()),
            ]
        })
        .collect();
    let twin: Vec<[u32; 3]> = (0..t)
        .map(|i| {
            let b = 16 + n * 8 + t * 12 + i * 12;
            [
                u32::from_le_bytes(buf[b..b + 4].try_into().unwrap()),
                u32::from_le_bytes(buf[b + 4..b + 8].try_into().unwrap()),
                u32::from_le_bytes(buf[b + 8..b + 12].try_into().unwrap()),
            ]
        })
        .collect();

    // in-circle det + error magnitude (perm) in f64, plus exact sign via i128.
    let icirc = |a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]| -> (f64, f64) {
        let (ax, ay) = (a[0] - d[0], a[1] - d[1]);
        let (bx, by) = (b[0] - d[0], b[1] - d[1]);
        let (cx, cy) = (c[0] - d[0], c[1] - d[1]);
        let (a2, b2, c2) = (ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy);
        let det = a2 * (bx * cy - cx * by) - b2 * (ax * cy - cx * ay) + c2 * (ax * by - bx * ay);
        let perm = a2 * ((bx * cy).abs() + (cx * by).abs())
            + b2 * ((ax * cy).abs() + (cx * ay).abs())
            + c2 * ((ax * by).abs() + (bx * ay).abs());
        (det, perm)
    };
    let exact = |a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]| -> i32 {
        let f = |v: [f64; 2]| [v[0] as i128, v[1] as i128];
        let (a, b, c, d) = (f(a), f(b), f(c), f(d));
        let (ax, ay) = (a[0] - d[0], a[1] - d[1]);
        let (bx, by) = (b[0] - d[0], b[1] - d[1]);
        let (cx, cy) = (c[0] - d[0], c[1] - d[1]);
        let (a2, b2, c2) = (ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy);
        (a2 * (bx * cy - cx * by) - b2 * (ax * cy - cx * ay) + c2 * (ax * by - bx * ay)).signum()
            as i32
    };

    // K·eps error model. eps: f16=2^-11, bf16=2^-8, f32=2^-24, f64=2^-53. K~16 (a few FMAs).
    let precisions = [
        ("bf16 (XDNA)", 2f64.powi(-8)),
        ("f16 (ANE)", 2f64.powi(-11)),
        ("f32 (GPU filter)", 2f64.powi(-24)),
        ("f64 (AMX)", 2f64.powi(-53)),
    ];
    let mut counts = [0usize; 4];
    let mut wrong = [0usize; 4]; // float sign disagrees with exact when it "certifies"
    let mut total = 0usize;
    let mut overflow_f16 = 0usize;

    for t0 in 0..t {
        let v = tri[t0];
        let (a, b, c) = (pts[v[0] as usize], pts[v[1] as usize], pts[v[2] as usize]);
        for e in 0..3 {
            let t1 = twin[t0][e];
            if t1 == u32::MAX || (t1 as usize) <= t0 {
                continue;
            }
            let vb = tri[t1 as usize];
            let (u, w) = (v[e], v[(e + 1) % 3]);
            let q = *vb.iter().find(|&&x| x != u && x != w).unwrap();
            let d = pts[q as usize];
            let (det, perm) = icirc(a, b, c, d);
            let ex = exact(a, b, c, d);
            total += 1;
            // f16 can't even REPRESENT coord diffs ~2^31 (max f16 ≈ 65504): overflow → useless raw
            if (a[0] - d[0]).abs() > 65504.0 {
                overflow_f16 += 1;
            }
            for (i, (_, eps)) in precisions.iter().enumerate() {
                let bound = 16.0 * eps * perm;
                if det.abs() > bound {
                    counts[i] += 1;
                    if (det.signum() as i32) != ex {
                        wrong[i] += 1;
                    }
                }
            }
        }
    }
    println!("mesh {path}: N={n} T={t}  internal in-circle tests={total}");
    println!(
        "f16 would OVERFLOW on {:.1}% of tests (coord diff > 65504) before any normalization\n",
        100.0 * overflow_f16 as f64 / total as f64
    );
    println!(
        "{:<18} {:>12}  {:>10}",
        "precision", "certified%", "fallthrough%"
    );
    for (i, (name, _)) in precisions.iter().enumerate() {
        println!(
            "{:<18} {:>11.2}%  {:>10.2}%   (wrong-sign when certified: {})",
            name,
            100.0 * counts[i] as f64 / total as f64,
            100.0 * (total - counts[i]) as f64 / total as f64,
            wrong[i]
        );
    }
}
