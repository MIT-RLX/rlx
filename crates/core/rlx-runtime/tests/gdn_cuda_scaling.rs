// Native + FlashKDA-chunk Op::GatedDeltaNet(pc) on the selected device: a
// correctness check vs a Rust reference, and a scaling profile. Toggle the chunk
// kernel with the SHELL env RLX_CUDA_KDA_CHUNK=1 (read at kernel launch), e.g.:
//   # correctness (runs by default)
//   RLX_TEST_DEVICE=cuda RLX_CUDA_KDA_CHUNK=1 cargo test -p rlx-runtime --release \
//     --features cuda --test gdn_cuda_scaling gdn_correctness -- --nocapture
//   # timing
//   RLX_TEST_DEVICE=cuda [RLX_CUDA_KDA_CHUNK=1] cargo test -p rlx-runtime --release \
//     --features cuda --test gdn_cuda_scaling gdn_native_scaling -- --ignored --nocapture
#![cfg(any(feature = "cpu", feature = "cuda"))]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use std::time::Instant;

fn dev() -> Device {
    match std::env::var("RLX_TEST_DEVICE").ok().as_deref() {
        Some("cuda") => Device::Cuda,
        Some("metal") => Device::Metal,
        _ => Device::Cpu,
    }
}

fn fill(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 2.0 * amp
        })
        .collect()
}

fn l2norm_rows(x: &mut [f32], rows: usize, n: usize) {
    for r in 0..rows {
        let sl = &mut x[r * n..r * n + n];
        let nrm = (sl.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
        sl.iter_mut().for_each(|v| *v /= nrm);
    }
}

/// Ground-truth per-channel gated delta-net recurrence.
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    b: usize,
    s: usize,
    h: usize,
    n: usize,
) -> Vec<f32> {
    let scale = 1.0f32 / (n as f32).sqrt();
    let mut out = vec![0f32; b * s * h * n];
    let hn = h * n;
    for bi in 0..b {
        for hi in 0..h {
            let mut smat = vec![0f32; n * n];
            for ti in 0..s {
                let base = bi * s * hn + ti * hn + hi * n;
                let bb = bi * s * h + ti * h + hi;
                let (qr, kr, vr, gr) = (
                    &q[base..base + n],
                    &k[base..base + n],
                    &v[base..base + n],
                    &g[base..base + n],
                );
                let bt = beta[bb];
                for i in 0..n {
                    let a = gr[i].exp();
                    for jj in 0..n {
                        smat[i * n + jj] *= a;
                    }
                }
                let mut sk = vec![0f32; n];
                for i in 0..n {
                    for jj in 0..n {
                        sk[jj] += smat[i * n + jj] * kr[i];
                    }
                }
                for jj in 0..n {
                    sk[jj] = (vr[jj] - sk[jj]) * bt;
                }
                for i in 0..n {
                    for jj in 0..n {
                        smat[i * n + jj] += kr[i] * sk[jj];
                    }
                }
                for jj in 0..n {
                    let mut acc = 0f32;
                    for i in 0..n {
                        acc += smat[i * n + jj] * qr[i];
                    }
                    out[base + jj] = acc * scale;
                }
            }
        }
    }
    out
}

fn build(b: usize, s: usize, h: usize, n: usize) -> Graph {
    let mut g = Graph::new("gdn");
    let bshn = Shape::new(&[b, s, h, n], DType::F32);
    let bsh = Shape::new(&[b, s, h], DType::F32);
    let q = g.input("q", bshn.clone());
    let k = g.input("k", bshn.clone());
    let v = g.input("v", bshn.clone());
    let gi = g.input("g", bshn.clone());
    let beta = g.input("beta", bsh);
    let y = g.gated_delta_net_pc(q, k, v, gi, beta, n, bshn);
    g.set_outputs(vec![y]);
    g
}

fn gen_inputs(
    b: usize,
    s: usize,
    h: usize,
    n: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let bshn = b * s * h * n;
    let mut q = fill(bshn, seed + 1, 1.0);
    let mut k = fill(bshn, seed + 2, 1.0);
    l2norm_rows(&mut q, b * s * h, n);
    l2norm_rows(&mut k, b * s * h, n);
    let v = fill(bshn, seed + 3, 1.0);
    let g: Vec<f32> = fill(bshn, seed + 4, 0.25)
        .iter()
        .map(|x| -(x.abs()))
        .collect();
    let beta: Vec<f32> = fill(b * s * h, seed + 5, 4.0)
        .iter()
        .map(|x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    (q, k, v, g, beta)
}

fn run(
    b: usize,
    s: usize,
    h: usize,
    n: usize,
    i: &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
) -> Vec<f32> {
    let inputs: [(&str, &[f32]); 5] = [
        ("q", &i.0),
        ("k", &i.1),
        ("v", &i.2),
        ("g", &i.3),
        ("beta", &i.4),
    ];
    let mut c = Session::new(dev()).compile(build(b, s, h, n));
    c.run(&inputs).into_iter().next().unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Correctness on `dev()` vs the Rust reference. With RLX_CUDA_KDA_CHUNK=1 this
/// exercises the chunked kernel; without it, the native scan. n=128 (chunk req).
#[test]
fn gdn_correctness() {
    let chunk = std::env::var("RLX_CUDA_KDA_CHUNK").is_ok();
    println!("device={:?} kda_chunk_env={chunk}", dev());
    let n = 128usize;
    // (b, s, h) — include seq not a multiple of 16 (padding path).
    for (ci, &(b, s, h)) in [
        (1usize, 16usize, 2usize),
        (1, 20, 2),
        (1, 48, 3),
        (2, 33, 2),
        (1, 64, 4),
    ]
    .iter()
    .enumerate()
    {
        let i = gen_inputs(b, s, h, n, 100 + ci as u64 * 10);
        let want = reference(&i.0, &i.1, &i.2, &i.3, &i.4, b, s, h, n);
        let got = run(b, s, h, n, &i);
        assert_eq!(got.len(), want.len());
        let amp = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
        let d = max_abs_diff(&got, &want);
        assert!(
            d < 2e-3 * amp.max(1.0),
            "case {ci} [b{b} s{s} h{h} n{n}] diff {d} (amp {amp})"
        );
        println!(
            "case {ci} [b{b} s{s} h{h} n{n}]: diff {d:.2e} (tol {:.2e})",
            2e-3 * amp.max(1.0)
        );
    }
    println!("gdn_correctness OK");
}

fn time_min(b: usize, s: usize, h: usize, n: usize) -> f64 {
    let i = gen_inputs(b, s, h, n, 1);
    let inputs: [(&str, &[f32]); 5] = [
        ("q", &i.0),
        ("k", &i.1),
        ("v", &i.2),
        ("g", &i.3),
        ("beta", &i.4),
    ];
    let mut c = Session::new(dev()).compile(build(b, s, h, n));
    let _ = c.run(&inputs);
    let _ = c.run(&inputs);
    let mut best = f64::INFINITY;
    for _ in 0..8 {
        let t = Instant::now();
        let _ = c.run(&inputs);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

#[test]
#[ignore = "profiling; run with --release --ignored --nocapture"]
fn gdn_native_scaling() {
    let n = 128usize;
    let tag = if std::env::var("RLX_CUDA_KDA_CHUNK").is_ok() {
        "KDA-CHUNK"
    } else {
        "native"
    };
    println!(
        "\nOp::GatedDeltaNet(pc) [{tag}] scaling  (n={n}, device={:?})",
        dev()
    );
    println!(
        "-- vs T (b=1, h=16) --  {:>10} {:>10}",
        "time(ms)", "us/token"
    );
    for &s in &[256usize, 512, 1024, 2048, 4096] {
        let t = time_min(1, s, 16, n);
        println!("T={:>5} | {:>10.3} | {:>10.4}", s, t, t * 1e3 / s as f64);
    }
    println!(
        "-- vs h (b=1, T=1024) --  {:>10} {:>10}",
        "time(ms)", "ms/head"
    );
    for &h in &[1usize, 4, 16, 32, 64, 96] {
        let t = time_min(1, 1024, h, n);
        println!("h={:>5} | {:>10.3} | {:>10.4}", h, t, t / h as f64);
    }
    println!();
}
