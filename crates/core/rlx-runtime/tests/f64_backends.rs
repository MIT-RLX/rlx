// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! F64 graphs on every backend, against CPU's native f64. Ill-conditioned
//! data (large cancelling terms) makes an f32 computation visibly wrong, so
//! the error separates real f64 / double-single from f32 promotion.

mod common;

use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompileOptions, Device, NumberFormat, Session};

const N: usize = 4096;
const M: usize = 16;
const K: usize = 512;
const P: usize = 8;

fn bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f64s(b: &[u8]) -> Vec<f64> {
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

/// `(a + b, a * b, A @ B, sum(a))` in F64.
fn graph() -> Graph {
    let f = DType::F64;
    let mut g = Graph::new("f64_probe");
    let a = g.input("a", Shape::new(&[N], f));
    let b = g.input("b", Shape::new(&[N], f));
    let ma = g.input("ma", Shape::new(&[M, K], f));
    let mb = g.input("mb", Shape::new(&[K, P], f));
    let add = g.binary(BinaryOp::Add, a, b, Shape::new(&[N], f));
    let mul = g.binary(BinaryOp::Mul, a, b, Shape::new(&[N], f));
    let mm = g.matmul(ma, mb, Shape::new(&[M, P], f));
    let sum = g.reduce(a, ReduceOp::Sum, vec![0], false, Shape::new(&[1], f));
    g.set_outputs(vec![add, mul, mm, sum]);
    g
}

struct Data {
    a: Vec<f64>,
    b: Vec<f64>,
    ma: Vec<f64>,
    mb: Vec<f64>,
}

fn data() -> Data {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    // Cancelling large terms + small detail: the sum is dominated by the
    // detail, which f32 (24 bits) loses entirely next to 1e8.
    let a: Vec<f64> = (0..N)
        .map(|i| if i % 2 == 0 { 1e8 } else { -1e8 } + r() * 1e-3)
        .collect();
    let b: Vec<f64> = (0..N).map(|_| 1.0 + r() * 1e-9).collect();
    let ma: Vec<f64> = (0..M * K)
        .map(|i| if i % 2 == 0 { 1e6 } else { -1e6 } + r())
        .collect();
    let mb: Vec<f64> = (0..K * P).map(|_| 1.0 + r() * 1e-7).collect();
    Data { a, b, ma, mb }
}

fn run(device: Device, d: &Data) -> Vec<Vec<f64>> {
    run_with(device, d, &CompileOptions::new())
}

fn run_with(device: Device, d: &Data, opts: &CompileOptions) -> Vec<Vec<f64>> {
    let mut cg = Session::new(device).compile_with(graph(), opts);
    let f = DType::F64;
    let (a, b, ma, mb) = (bytes(&d.a), bytes(&d.b), bytes(&d.ma), bytes(&d.mb));
    cg.run_typed(&[("a", &a, f), ("b", &b, f), ("ma", &ma, f), ("mb", &mb, f)])
        .into_iter()
        .map(|(bytes, dt)| {
            assert_eq!(dt, DType::F64, "{device:?} returned {dt:?}");
            f64s(&bytes)
        })
        .collect()
}

/// Max error relative to the largest reference magnitude.
fn rel_err(got: &[f64], want: &[f64]) -> f64 {
    let scale = want
        .iter()
        .fold(0f64, |m, x| m.max(x.abs()))
        .max(f64::MIN_POSITIVE);
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f64, f64::max)
        / scale
}

/// The same ops in f32, for scale.
fn f32_reference(d: &Data) -> Vec<Vec<f64>> {
    let a: Vec<f32> = d.a.iter().map(|&x| x as f32).collect();
    let b: Vec<f32> = d.b.iter().map(|&x| x as f32).collect();
    let add = a.iter().zip(&b).map(|(x, y)| (x + y) as f64).collect();
    let mul = a.iter().zip(&b).map(|(x, y)| (x * y) as f64).collect();
    let mut mm = vec![0f64; M * P];
    for i in 0..M {
        for j in 0..P {
            let mut acc = 0f32;
            for k in 0..K {
                acc += d.ma[i * K + k] as f32 * d.mb[k * P + j] as f32;
            }
            mm[i * P + j] = acc as f64;
        }
    }
    let sum = vec![a.iter().sum::<f32>() as f64];
    vec![add, mul, mm, sum]
}

#[test]
fn f64_ops_on_every_backend() {
    let _gpu = common::serialize_gpu();
    let d = data();
    let want = run(Device::Cpu, &d);
    let f32_err: Vec<f64> = f32_reference(&d)
        .iter()
        .zip(&want)
        .map(|(g, w)| rel_err(g, w))
        .collect();
    eprintln!(
        "f32 baseline   add {:.1e}  mul {:.1e}  matmul {:.1e}  sum {:.1e}",
        f32_err[0], f32_err[1], f32_err[2], f32_err[3]
    );
    let mut df = CompileOptions::new();
    df.number_format = Some(NumberFormat::F64x2);
    let e: Vec<f64> = run_with(Device::Cpu, &d, &df)
        .iter()
        .zip(&want)
        .map(|(g, w)| rel_err(g, w))
        .collect();
    eprintln!(
        "cpu f64x2      add {:.1e}  mul {:.1e}  matmul {:.1e}  sum {:.1e}",
        e[0], e[1], e[2], e[3]
    );
    for device in [Device::Metal, Device::Mlx, Device::Gpu, Device::Vulkan] {
        if common::skip_unless(device) {
            continue;
        }
        let got = std::panic::catch_unwind(|| run(device, &d));
        match got {
            Ok(got) => {
                let e: Vec<f64> = got.iter().zip(&want).map(|(g, w)| rel_err(g, w)).collect();
                eprintln!(
                    "{device:<12?}   add {:.1e}  mul {:.1e}  matmul {:.1e}  sum {:.1e}",
                    e[0], e[1], e[2], e[3]
                );
            }
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                eprintln!(
                    "{device:<12?}   FAILED: {}",
                    msg.lines().next().unwrap_or("")
                );
            }
        }
    }
}

/// Cost of double-single against f32 on each backend: a 512³ matmul and a
/// 1M-element add, end to end per `run` (host conversion included).
/// `cargo test -p rlx-runtime --features metal,gpu --test f64_backends --release -- --ignored --nocapture`
#[test]
#[ignore]
fn f64_cost_vs_f32() {
    let _gpu = common::serialize_gpu();
    const S: usize = 512;
    const E: usize = 1 << 20;
    let build = |dt: DType| {
        let mut g = Graph::new("cost");
        let a = g.input("a", Shape::new(&[S, S], dt));
        let b = g.input("b", Shape::new(&[S, S], dt));
        let x = g.input("x", Shape::new(&[E], dt));
        let y = g.input("y", Shape::new(&[E], dt));
        let mm = g.matmul(a, b, Shape::new(&[S, S], dt));
        let add = g.binary(BinaryOp::Add, x, y, Shape::new(&[E], dt));
        g.set_outputs(vec![mm, add]);
        g
    };
    let v = |n: usize| -> Vec<f64> { (0..n).map(|i| ((i * 7919) % 1000) as f64 * 1e-3).collect() };
    let (a, x) = (v(S * S), v(E));
    for device in [Device::Cpu, Device::Metal, Device::Gpu] {
        if device != Device::Cpu && common::skip_unless(device) {
            continue;
        }
        let mut times = Vec::new();
        for dt in [DType::F32, DType::F64] {
            let mut opts = CompileOptions::new();
            if dt == DType::F64 && device == Device::Cpu {
                opts.number_format = Some(NumberFormat::F64x2);
            }
            let mut cg = Session::new(device).compile_with(build(dt), &opts);
            let enc = |v: &[f64]| -> Vec<u8> {
                match dt {
                    DType::F32 => v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect(),
                    _ => bytes(v),
                }
            };
            let (ab, xb) = (enc(&a), enc(&x));
            let ins = [
                ("a", &ab[..], dt),
                ("b", &ab[..], dt),
                ("x", &xb[..], dt),
                ("y", &xb[..], dt),
            ];
            cg.run_typed(&ins);
            let t = std::time::Instant::now();
            const RUNS: u32 = 5;
            for _ in 0..RUNS {
                cg.run_typed(&ins);
            }
            times.push(t.elapsed().as_secs_f64() * 1e3 / RUNS as f64);
        }
        eprintln!(
            "{device:<8?} f32 {:8.2} ms   double-single {:8.2} ms   ({:.1}x)",
            times[0],
            times[1],
            times[1] / times[0]
        );
        if device == Device::Cpu {
            let mut cg = Session::new(device).compile(build(DType::F64));
            let (ab, xb) = (bytes(&a), bytes(&x));
            let f = DType::F64;
            let ins = [
                ("a", &ab[..], f),
                ("b", &ab[..], f),
                ("x", &xb[..], f),
                ("y", &xb[..], f),
            ];
            cg.run_typed(&ins);
            let t = std::time::Instant::now();
            for _ in 0..5 {
                cg.run_typed(&ins);
            }
            eprintln!(
                "Cpu      native f64 {:8.2} ms",
                t.elapsed().as_secs_f64() * 1e3 / 5.0
            );
        }
    }
}
