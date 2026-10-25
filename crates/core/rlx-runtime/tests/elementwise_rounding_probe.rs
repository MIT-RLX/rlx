// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Do the GPU backends round elementwise chains exactly as written? `mul` then
//! `add` must round twice (no contraction), `Op::Fma` once; `neg`/`sub` keep signed
//! zeros. Compared bit for bit against CPU on inputs where fusing changes the
//! result. Prints a report; fails on any difference.

#![cfg(feature = "cpu")]

mod common;

use rlx_fusion::pass::Pass;
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn sig(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0)
                * if s & 1 == 0 { 1.0 } else { 1e-3 }
        })
        .collect()
}

fn graph(n: usize) -> Graph {
    let s = Shape::new(&[n], DType::F32);
    let mut g = Graph::new("probe");
    let a = g.input("a", s.clone());
    let b = g.input("b", s.clone());
    let c = g.input("c", s.clone());
    let d = g.input("d", s.clone());
    let bin = |g: &mut Graph, op, x, y| g.add_node(Op::Binary(op), vec![x, y], s.clone());
    let ab = bin(&mut g, BinaryOp::Mul, a, b);
    let mul_add = bin(&mut g, BinaryOp::Add, ab, c); // two roundings
    let fma = g.add_node(Op::Fma, vec![a, b, c], s.clone()); // one
    let cd = bin(&mut g, BinaryOp::Mul, c, d);
    let diff = bin(&mut g, BinaryOp::Sub, ab, cd); // a*b - c*d, plain
    let ncd = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Neg),
        vec![cd],
        s.clone(),
    );
    let fms = g.add_node(Op::Fma, vec![a, b, ncd], s.clone()); // fma(a,b,-(c*d))
    let chain = g.add_node(Op::Fma, vec![c, d, fma], s.clone()); // nested
    let zero = bin(&mut g, BinaryOp::Sub, a, a); // +0
    let negz = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Neg),
        vec![zero],
        s.clone(),
    ); // -0
    let sz = bin(&mut g, BinaryOp::Add, negz, negz); // -0 + -0 = -0
    g.set_outputs(vec![mul_add, fma, diff, fms, chain, sz]);
    g
}

fn run(dev: Device, n: usize, inputs: &[Vec<f32>]) -> Vec<Vec<u32>> {
    let bytes: Vec<Vec<u8>> = inputs
        .iter()
        .map(|v| v.iter().flat_map(|x| x.to_le_bytes()).collect())
        .collect();
    let names = ["a", "b", "c", "d"];
    let feeds: Vec<(&str, &[u8], DType)> = names
        .iter()
        .zip(&bytes)
        .map(|(n, b)| (*n, b.as_slice(), DType::F32))
        .collect();
    let out = Session::new(dev).compile(graph(n)).run_typed(&feeds);
    out.iter()
        .map(|(b, _)| {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect()
        })
        .collect()
}

#[test]
fn gpu_elementwise_rounding_matches_cpu() {
    let n = 4096;
    let inputs: Vec<Vec<f32>> = (0..4).map(|k| sig(n, 11 + k)).collect();
    let want = run(Device::Cpu, n, &inputs);
    // sanity: the probe can tell fused from unfused
    assert_ne!(want[0], want[1], "inputs never separate mul+add from fma");
    let names = [
        "mul_add",
        "fma",
        "a*b-c*d",
        "fma(a,b,-cd)",
        "fma chain",
        "-0+-0",
    ];
    // Every accelerator compiled into this build and usable on this host.
    let devices: Vec<Device> = rlx_runtime::device_ext::available_devices()
        .into_iter()
        .filter(|d| *d != Device::Cpu)
        .collect();
    let mut bad = vec![];
    eprintln!("devices: {devices:?}");
    for dev in devices {
        let _gpu = common::GpuTestGuard::acquire(dev);
        let got = run(dev, n, &inputs);
        for (k, name) in names.iter().enumerate() {
            let diffs = got[k].iter().zip(&want[k]).filter(|(a, b)| a != b).count();
            eprintln!("{dev:?} {name:>14}: {diffs}/{n} differ");
            if diffs > 0 {
                let ex: Vec<(f32, f32)> = got[k]
                    .iter()
                    .zip(&want[k])
                    .filter(|(a, b)| a != b)
                    .take(3)
                    .map(|(a, b)| (f32::from_bits(*a), f32::from_bits(*b)))
                    .collect();
                eprintln!("    got/want: {ex:?}");
            }
            // CoreML's graph optimizer drops this `-0` (context-dependent: the same
            // chain alone keeps it; the FFT lowering is unaffected, measured).
            // Asserted so a change in either direction is noticed.
            let known = dev == Device::Ane && *name == "-0+-0";
            if (diffs > 0) != known {
                bad.push(format!(
                    "{dev:?} {name}: {diffs} differ (known deviation: {known})"
                ));
            }
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}

/// Inputs on which a double-rounding `mul`+`add` and a careless emulation fail:
/// random, cancelling, wide-range, signed zeros, and constructed double-rounding
/// ties (`a·b` rounds to exactly 1.5, `c` an integer in [2²³, 2²⁴)).
fn hard_fma_inputs(n: usize) -> [Vec<f32>; 3] {
    let mut s: u64 = 0x5eed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let u = |x: u64| ((x >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0;
    let (mut a, mut b, mut c) = (vec![], vec![], vec![]);
    while a.len() < n {
        let k = a.len() % 5;
        let (x, y, z) = match k {
            0 => (u(next()), u(next()), u(next())),
            1 => {
                let (x, y) = (u(next()) * 8.0, u(next()) / 4.0);
                (x, y, -(x * y))
            }
            2 => (u(next()) * 1e6, u(next()) * 1e-6, u(next()) * 1e-9),
            3 => {
                let x = 1.0 + ((next() >> 41) as f32) * 2f32.powi(-23);
                let y = 1.5f32 / x;
                if x * y != 1.5 || x.mul_add(y, -1.5) == 0.0 {
                    continue;
                }
                let z = (8_388_608 + (next() >> 41) as u32 % 8_388_607) as f32;
                if next() & 1 == 0 {
                    (x, y, z)
                } else {
                    (-x, y, -z)
                }
            }
            _ => {
                let zs = [0.0f32, -0.0, 3.0, -3.0];
                (
                    zs[(next() % 4) as usize],
                    zs[(next() % 4) as usize],
                    zs[(next() % 2) as usize],
                )
            }
        };
        a.push(x);
        b.push(y);
        c.push(z);
    }
    [a, b, c]
}

fn fma_graph(n: usize) -> Graph {
    // 2-D so WebGL's texture rows stay under the browsers' 16384 limit
    let s = Shape::new(&[n / 400, 400], DType::F32);
    let mut g = Graph::new("fma");
    let a = g.input("a", s.clone());
    let b = g.input("b", s.clone());
    let c = g.input("c", s.clone());
    let f = g.add_node(Op::Fma, vec![a, b, c], s);
    g.set_outputs(vec![f]);
    g
}

fn run_fma(dev: Device, g: Graph, x: &[Vec<f32>; 3]) -> Vec<u32> {
    let bytes: Vec<Vec<u8>> = x
        .iter()
        .map(|v| v.iter().flat_map(|y| y.to_le_bytes()).collect())
        .collect();
    let out = Session::new(dev).compile(g).run_typed(&[
        ("a", &bytes[0], DType::F32),
        ("b", &bytes[1], DType::F32),
        ("c", &bytes[2], DType::F32),
    ]);
    out[0]
        .0
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// `LowerFma`'s exact f32 emulation (what FMA-less backends get) equals a
/// hardware FMA — on CPU, and on every device that runs the emulation's ops.
#[test]
fn emulated_fma_is_bit_exact() {
    let n = 200_000;
    let x = hard_fma_inputs(n);
    let want: Vec<u32> = (0..n)
        .map(|i| x[0][i].mul_add(x[1][i], x[2][i]).to_bits())
        .collect();
    let naive = (0..n)
        .filter(|&i| (x[0][i] * x[1][i] + x[2][i]).to_bits() != want[i])
        .count();
    assert!(
        naive > 1000,
        "inputs must separate mul+add from fma ({naive})"
    );
    let lowered = rlx_fusion::lower_fma::LowerFma.run(fma_graph(n));
    let mut devices = vec![Device::Cpu];
    devices.extend(
        rlx_runtime::device_ext::available_devices()
            .into_iter()
            .filter(|d| *d != Device::Cpu),
    );
    let mut bad = vec![];
    for d in devices {
        let _gpu = common::GpuTestGuard::acquire(d);
        let got = run_fma(d, lowered.clone(), &x);
        let diff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        eprintln!("{d:?} emulated fma: {diff}/{n} differ (mul+add would: {naive})");
        if diff > 0 {
            bad.push(format!("{d:?}: {diff}"));
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}
