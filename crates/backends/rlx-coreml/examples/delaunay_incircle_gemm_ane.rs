// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// GEMM formulation of the Delaunay in-circle predicate for the ANE/XDNA MATMUL engines.
//
// The elementwise determinant is bandwidth-bound (AI≈0.8 flop/byte) so the NPU's MAC array
// sits idle. Recast it as a matmul via the paraboloid lift: lift each query point to
// L[d] = [xd, yd, xd²+yd², 1] (a [D×4] matrix), and precompute per-triangle cofactors
// C[:,t] = [M00, −M01, M02, −M03] (the 3×3 minors of the (a,b,c) rows; a [4×T] matrix).
// Then **DET = L @ C** ([D×4]@[4×T]) is EVERY point×triangle in-circle test in one GEMM —
// the op the ANE/XDNA accelerate. Input I/O is O(D+T) (the matmul expands to D×T on-chip),
// vs O(D·T) for the elementwise feed. Benchmarks the matmul vs elementwise vs CPU.
//   cargo run -p rlx-coreml --example delaunay_incircle_gemm_ane --release -- [K] [Tchunk]
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod imp {
    use rlx_coreml::{ComputeUnits, CoremlExecutable};
    use rlx_ir::{DType, Graph, Shape};
    use std::time::Instant;

    // exact 4x4 lifted in-circle determinant sign (i128): rows [d;a;b;c], cols [x,y,x²+y²,1]
    fn det3_i(m: [[i128; 3]; 3]) -> i128 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
    fn lift_i(p: [i64; 2]) -> [i128; 4] {
        let (x, y) = (p[0] as i128, p[1] as i128);
        [x, y, x * x + y * y, 1]
    }
    fn exact4(d: [i64; 2], a: [i64; 2], b: [i64; 2], c: [i64; 2]) -> i32 {
        let (ld, la, lb, lc) = (lift_i(d), lift_i(a), lift_i(b), lift_i(c));
        // det of [ld; la; lb; lc] expanded along row 0
        let mut s = 0i128;
        let rows = [la, lb, lc];
        for j in 0..4 {
            let mut mm = [[0i128; 3]; 3];
            for r in 0..3 {
                let mut cc = 0;
                for col in 0..4 {
                    if col == j {
                        continue;
                    }
                    mm[r][cc] = rows[r][col];
                    cc += 1;
                }
            }
            let sign = if j % 2 == 0 { 1 } else { -1 };
            s += sign * ld[j] * det3_i(mm);
        }
        s.signum() as i32
    }

    // f64 cofactors C[:,t] = [M00, −M01, M02, −M03] from a triangle's lifted (a,b,c) rows.
    fn det3_f(m: [[f64; 3]; 3]) -> f64 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
    fn cofactors(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 4] {
        // rows are lifted (x,y,z) with an implicit 1 column; build [x,y,z,1] rows
        let r = [
            [a[0], a[1], a[2], 1.0],
            [b[0], b[1], b[2], 1.0],
            [c[0], c[1], c[2], 1.0],
        ];
        let mut cof = [0.0; 4];
        for j in 0..4 {
            let mut mm = [[0.0; 3]; 3];
            for (ri, rr) in r.iter().enumerate() {
                let mut cc = 0;
                for col in 0..4 {
                    if col == j {
                        continue;
                    }
                    mm[ri][cc] = rr[col];
                    cc += 1;
                }
            }
            cof[j] = if j % 2 == 0 { det3_f(mm) } else { -det3_f(mm) };
        }
        cof
    }

    /// matmul graph: L [D×4] @ C [4×N] = DET [D×N]  (f16 datapath, f32 output).
    fn build_gemm(d: usize, n: usize) -> Graph {
        let mut g = Graph::new("incircle_gemm");
        let l = g.input("L", Shape::new(&[d, 4], DType::F32));
        let c = g.input("C", Shape::new(&[4, n], DType::F32));
        let lh = g.append_node(
            rlx_ir::Op::Cast { to: DType::F16 },
            vec![l],
            Shape::new(&[d, 4], DType::F16),
            None,
        );
        let ch = g.append_node(
            rlx_ir::Op::Cast { to: DType::F16 },
            vec![c],
            Shape::new(&[4, n], DType::F16),
            None,
        );
        let det = g.matmul(lh, ch, Shape::new(&[d, n], DType::F16));
        let out = g.append_node(
            rlx_ir::Op::Cast { to: DType::F32 },
            vec![det],
            Shape::new(&[d, n], DType::F32),
            None,
        );
        g.set_outputs(vec![out]);
        g
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let k: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(48);
        let tchunk: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
        let span = 15_000_000i64;
        let mut s = 0x1234_5678u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as i64
        };
        let mut seen = std::collections::HashSet::new();
        let mut p: Vec<[i64; 2]> = vec![];
        while p.len() < k {
            let q = [rnd() % span, rnd() % span];
            if seen.insert(q) {
                p.push(q);
            }
        }

        // per-tile normalization (center + scale) so the lift/cofactors fit f16
        let cen = [
            p.iter().map(|q| q[0]).sum::<i64>() / k as i64,
            p.iter().map(|q| q[1]).sum::<i64>() / k as i64,
        ];
        let maxabs = p
            .iter()
            .flat_map(|q| [(q[0] - cen[0]).abs(), (q[1] - cen[1]).abs()])
            .max()
            .unwrap() as f64;
        let sc = 2.0 / maxabs;
        let nf: Vec<[f64; 2]> = p
            .iter()
            .map(|q| [(q[0] - cen[0]) as f64 * sc, (q[1] - cen[1]) as f64 * sc])
            .collect();

        // L [D×4] lifted query points (row-major)
        let mut lmat = Vec::with_capacity(k * 4);
        for q in &nf {
            lmat.extend_from_slice(&[
                q[0] as f32,
                q[1] as f32,
                (q[0] * q[0] + q[1] * q[1]) as f32,
                1.0,
            ]);
        }

        // triangles + per-triangle cofactor column (normalized to unit max — sign-preserving)
        let mut tris: Vec<[usize; 3]> = vec![];
        for a in 0..k {
            for b in (a + 1)..k {
                for c in (b + 1)..k {
                    tris.push([a, b, c]);
                }
            }
        }
        let t = tris.len();
        let mut cmat = vec![0f32; 4 * t]; // [4 × T], column-major per triangle
        for (ti, &[a, b, c]) in tris.iter().enumerate() {
            let lf = |q: [f64; 2]| [q[0], q[1], q[0] * q[0] + q[1] * q[1]];
            let cof = cofactors(lf(nf[a]), lf(nf[b]), lf(nf[c]));
            let m = cof.iter().fold(0f64, |mx, &v| mx.max(v.abs())).max(1e-30);
            for r in 0..4 {
                cmat[r * t + ti] = (cof[r] / m) as f32;
            }
        }

        let dev = || {
            let mut e = CoremlExecutable::compile_with_units(
                build_gemm(k, tchunk),
                ComputeUnits::CpuAndNeuralEngine,
            );
            let mut all = Vec::with_capacity(k * t);
            let mut start = 0;
            while start < t {
                let end = (start + tchunk).min(t);
                let mut cchunk = vec![0f32; 4 * tchunk];
                for r in 0..4 {
                    for (i, col) in (start..end).enumerate() {
                        cchunk[r * tchunk + i] = cmat[r * t + col];
                    }
                }
                let out = e.run(&[("L", &lmat), ("C", &cchunk)]).unwrap().remove(0); // [k×tchunk]
                // keep only the real columns
                for row in 0..k {
                    all.extend_from_slice(&out[row * tchunk..row * tchunk + (end - start)]);
                }
                start = end;
            }
            all // NOTE: layout is per-chunk; used only for timing here
        };

        // correctness: one clean pass, matmul DET vs exact4, over all point×triangle
        let mut e = CoremlExecutable::compile_with_units(
            build_gemm(k, tchunk),
            ComputeUnits::CpuAndNeuralEngine,
        );
        let (mut agree, mut total) = (0usize, 0usize);
        let mut start = 0;
        while start < t {
            let end = (start + tchunk).min(t);
            let mut cchunk = vec![0f32; 4 * tchunk];
            for r in 0..4 {
                for (i, col) in (start..end).enumerate() {
                    cchunk[r * tchunk + i] = cmat[r * t + col];
                }
            }
            let out = e.run(&[("L", &lmat), ("C", &cchunk)]).unwrap().remove(0);
            for (i, &[a, b, c]) in tris[start..end].iter().enumerate() {
                // cofactor column was scaled by 1/m>0 ⇒ multiply sign back is identity
                for row in 0..k {
                    if row == a || row == b || row == c {
                        continue;
                    }
                    let det = out[row * tchunk + i];
                    let sd = if det > 0.0 {
                        1
                    } else if det < 0.0 {
                        -1
                    } else {
                        0
                    };
                    let ex = exact4(p[row], p[a], p[b], p[c]);
                    if sd == ex {
                        agree += 1;
                    }
                    total += 1;
                }
            }
            start = end;
        }

        // FAIR host baseline: the efficient 3x3-relative-to-d i128 in-circle — the SAME
        // predicate, fastest correct CPU form (NOT the naive 4x4-cofactor `exact4`, ~10x heavier).
        let fast = |a: [i64; 2], b: [i64; 2], c: [i64; 2], d: [i64; 2]| -> i32 {
            let (ax, ay) = ((a[0] - d[0]) as i128, (a[1] - d[1]) as i128);
            let (bx, by) = ((b[0] - d[0]) as i128, (b[1] - d[1]) as i128);
            let (cx, cy) = ((c[0] - d[0]) as i128, (c[1] - d[1]) as i128);
            let (a2, b2, c2) = (ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy);
            (a2 * (bx * cy - cx * by) - b2 * (ax * cy - cx * ay) + c2 * (ax * by - bx * ay))
                .signum() as i32
        };
        let t_host = {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t0 = Instant::now();
                let mut acc = 0i64;
                for &[a, b, c] in &tris {
                    for row in 0..k {
                        acc += fast(p[a], p[b], p[c], p[row]) as i64;
                    }
                }
                std::hint::black_box(acc);
                best = best.min(t0.elapsed().as_secs_f64() * 1e3);
            }
            best
        };

        // GEMM timing (best-of)
        let _ = dev(); // warmup
        let mut t_gemm = f64::INFINITY;
        for _ in 0..8 {
            let t0 = Instant::now();
            std::hint::black_box(dev());
            t_gemm = t_gemm.min(t0.elapsed().as_secs_f64() * 1e3);
        }

        let ntests = k * t;
        println!(
            "tile K={k}  triangles T={t}  point×triangle tests={ntests}  (chunk {tchunk} cols)"
        );
        println!(
            "GEMM matmul sign-correct {agree}/{total} ({:.3}%)",
            100.0 * agree as f64 / total as f64
        );
        println!(
            "ANE GEMM (L@C):  {t_gemm:7.2} ms  ({:.0} M tests/s)",
            ntests as f64 / t_gemm / 1000.0
        );
        println!(
            "host exact i128: {t_host:7.2} ms  ({:.0} M tests/s)",
            ntests as f64 / t_host / 1000.0
        );
        println!("→ ANE GEMM is {:.2}× vs 1-core CPU exact", t_host / t_gemm);
    }
}

fn main() {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    imp::run();
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    println!("CoreML/ANE is Apple-only.");
}
