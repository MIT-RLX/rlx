// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Cross-backend benchmark of the Delaunay in-circle predicate as a GEMM (DET = L@C,
// paraboloid lift). ONE rlx IR graph, run through whatever backend is selected — the
// point of a compiler: same graph, every device. One device per process (crash-isolated).
//   cargo run -p rlx-runtime --example incircle_backends --features cpu,blas-accelerate,metal,mlx,ane,gpu --release -- <device> [K]
//   device ∈ cpu|metal|mlx|ane|gpu|vulkan|cuda|rocm|xdna
use rlx_driver::Device;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Session, is_available};
use std::time::Instant;

fn det3(m: [[i128; 3]; 3]) -> i128 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}
fn exact4(p: &[[i64; 2]], d: usize, a: usize, b: usize, c: usize) -> i32 {
    let idx = [d, a, b, c];
    let l: Vec<[i128; 4]> = idx
        .iter()
        .map(|&i| {
            let (x, y) = (p[i][0] as i128, p[i][1] as i128);
            [x, y, x * x + y * y, 1]
        })
        .collect();
    let mut s = 0i128;
    for j in 0..4 {
        let mut mm = [[0i128; 3]; 3];
        for r in 0..3 {
            let mut cc = 0;
            for col in 0..4 {
                if col == j {
                    continue;
                }
                mm[r][cc] = l[r + 1][col];
                cc += 1;
            }
        }
        s += if j % 2 == 0 { 1 } else { -1 } * l[0][j] * det3(mm);
    }
    s.signum() as i32
}
fn det3f(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn build(d: usize, t: usize) -> Graph {
    let mut g = Graph::new("incircle_gemm");
    let l = g.input("L", Shape::new(&[d, 4], DType::F32));
    let c = g.input("C", Shape::new(&[4, t], DType::F32));
    let det = g.matmul(l, c, Shape::new(&[d, t], DType::F32));
    g.set_outputs(vec![det]);
    g
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let devname = args.get(1).map(|s| s.as_str()).unwrap_or("cpu");
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(48);
    let dev = match devname {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "ane" => Device::Ane,
        "gpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        "xdna" => Device::Xdna,
        _ => {
            println!("unknown device {devname}");
            return;
        }
    };
    if !is_available(dev) {
        println!("{devname:<7}: NOT AVAILABLE");
        return;
    }

    // tile + per-tile normalization
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
    let cen = [
        p.iter().map(|q| q[0]).sum::<i64>() / k as i64,
        p.iter().map(|q| q[1]).sum::<i64>() / k as i64,
    ];
    let mx = p
        .iter()
        .flat_map(|q| [(q[0] - cen[0]).abs(), (q[1] - cen[1]).abs()])
        .max()
        .unwrap()
        .max(1) as f64;
    let sc = 2.0 / mx;
    let nf: Vec<[f64; 2]> = p
        .iter()
        .map(|q| [(q[0] - cen[0]) as f64 * sc, (q[1] - cen[1]) as f64 * sc])
        .collect();

    let mut lmat = Vec::with_capacity(k * 4);
    for q in &nf {
        lmat.extend_from_slice(&[
            q[0] as f32,
            q[1] as f32,
            (q[0] * q[0] + q[1] * q[1]) as f32,
            1.0,
        ]);
    }
    let mut tris: Vec<[usize; 3]> = vec![];
    for a in 0..k {
        for b in (a + 1)..k {
            for c in (b + 1)..k {
                tris.push([a, b, c]);
            }
        }
    }
    let t = tris.len();
    let mut cmat = vec![0f32; 4 * t];
    for (ti, &[a, b, c]) in tris.iter().enumerate() {
        let lf = |q: [f64; 2]| [q[0], q[1], q[0] * q[0] + q[1] * q[1], 1.0];
        let (ra, rb, rc) = (lf(nf[a]), lf(nf[b]), lf(nf[c]));
        let mut cof = [0f64; 4];
        for j in 0..4 {
            let mut mm = [[0f64; 3]; 3];
            for (ri, rr) in [ra, rb, rc].iter().enumerate() {
                let mut cc = 0;
                for col in 0..4 {
                    if col == j {
                        continue;
                    }
                    mm[ri][cc] = rr[col];
                    cc += 1;
                }
            }
            cof[j] = if j % 2 == 0 { det3f(mm) } else { -det3f(mm) };
        }
        let m = cof.iter().fold(1e-30f64, |mx, &v| mx.max(v.abs()));
        for r in 0..4 {
            cmat[r * t + ti] = (cof[r] / m) as f32;
        }
    }

    let mut cg = Session::new(dev).compile(build(k, t));
    let feed: [(&str, &[f32]); 2] = [("L", &lmat), ("C", &cmat)];
    for _ in 0..3 {
        let _ = cg.run(&feed);
    } // warmup
    let mut best = f64::INFINITY;
    let mut out = vec![];
    for _ in 0..12 {
        let t0 = Instant::now();
        out = cg.run(&feed).remove(0);
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }

    // validate signs (exclude vertices)
    let (mut agree, mut tot) = (0usize, 0usize);
    for (ti, &[a, b, c]) in tris.iter().enumerate() {
        for d in 0..k {
            if d == a || d == b || d == c {
                continue;
            }
            let v = out[d * t + ti];
            let sd = if v > 0.0 {
                1
            } else if v < 0.0 {
                -1
            } else {
                0
            };
            if sd == exact4(&p, d, a, b, c) {
                agree += 1;
            }
            tot += 1;
        }
    }
    let ntests = (k * t) as f64;
    println!(
        "{devname:<7}: {best:8.3} ms  {:8.0} M tests/s  correct {:.3}%  (K={k}, {} tests)",
        ntests / best / 1000.0,
        100.0 * agree as f64 / tot as f64,
        k * t
    );
}
