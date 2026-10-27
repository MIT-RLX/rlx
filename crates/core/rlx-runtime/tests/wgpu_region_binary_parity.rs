// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A fused wgpu `ElementwiseRegion` must compute every `BinaryOp` exactly as
//! the standalone `binary` kernel does, and so as rlx-cpu does wherever that
//! kernel is exact.
//!
//! The region kernel had its own hand-written `apply_binary`: `Div` was the
//! approximate `/` (1029 of 4096 off by an ULP), and `Pow` was bare `pow`,
//! which WGSL leaves undefined for a negative base (Metal happens to follow C,
//! so only `Div` fails here). Each chain below is one op applied to two inputs
//! and then scaled, so it fuses into a single region when regions are kept.
//!
//! Regions survive lowering only under `RLX_KEEP_ELEMENTWISE_REGIONS`, which
//! the runtime reads from the process environment; this file holds one test so
//! the flag cannot leak into another. `Mod` and the bitwise ops are not
//! region-fusable (`BinaryOp::region_fusable`) and pass through unfused.

#![cfg(all(feature = "gpu", feature = "cpu"))]

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

const N: usize = 4096;

fn chain(op: BinaryOp) -> Graph {
    let s = Shape::new(&[N], DType::F32);
    let mut g = Graph::new("region_binary");
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let z = g.input("z", s.clone());
    let v = g.binary(op, x, y, s.clone());
    let v = g.binary(BinaryOp::Mul, v, z, s.clone());
    g.set_outputs(vec![v]);
    g
}

/// Operands that exercise each op: large quotients for `Mod`, negative bases
/// with integer exponents for `Pow`, all four quadrants for `Atan2`, small
/// integers for the bitwise ops and shifts.
fn operands(op: BinaryOp) -> (Vec<f32>, Vec<f32>) {
    let mut s = 0x9e37_79b9_u32;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let mut unit = || (next() >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
    (0..N)
        .map(|_| {
            let (u, w) = (unit(), unit());
            match op {
                BinaryOp::Mod => (u * 4.0e5, w * 0.03 + 0.031 * w.signum()),
                BinaryOp::Pow => ((u * 3.0).round() + u * 0.5, (w * 4.0).round()),
                BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => {
                    ((u * 200.0).trunc(), (w * 200.0).trunc())
                }
                BinaryOp::Shl | BinaryOp::Shr => ((u * 200.0).trunc(), (w.abs() * 6.0).trunc()),
                _ => (u * 50.0, w * 3.0 + 0.5 * w.signum()),
            }
        })
        .unzip()
}

/// ULPs between two finite floats of the same sign (`u32::MAX` otherwise).
fn ulps(a: f32, b: f32) -> u32 {
    if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
        return 0;
    }
    if !a.is_finite() || !b.is_finite() || (a < 0.0) != (b < 0.0) {
        return u32::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

/// `Pow` and `Atan2` are transcendental: WGSL's `pow`/`atan2` are not libm,
/// in the standalone kernel either, and the same call compiles to different
/// code in different shaders (on discrete NVIDIA the unfused op runs on the
/// host, on the CPU's libm). They are held to this many ULP of rlx-cpu. GPU
/// `pow` is `exp2(y·log2 x)`, whose error grows with `|y·log2 x|`:
/// `(-0.0167)^-4` lands 10 ULP off on Metal. A `NaN` for a negative base is
/// `u32::MAX` away.
const TRANSCENDENTAL_ULPS: u32 = 64;

#[test]
fn every_binary_op_in_a_kept_region_matches_the_unfused_kernel() {
    let Some(_gpu) = common::serialize_gpu() else {
        return;
    };
    let z: Vec<f32> = (0..N).map(|i| 1.0 + (i % 7) as f32 * 0.125).collect();
    let mut bad = Vec::new();
    for op in BinaryOp::ALL {
        let (x, y) = operands(op);
        let run = |d: Device, keep: bool| {
            if keep {
                rlx_ir::env::set("RLX_KEEP_ELEMENTWISE_REGIONS", "1");
            }
            let out = Session::new(d)
                .compile(chain(op))
                .run(&[("x", &x), ("y", &y), ("z", &z)])
                .pop()
                .expect("one output");
            rlx_ir::env::remove("RLX_KEEP_ELEMENTWISE_REGIONS");
            out
        };
        let kept = run(Device::Gpu, true);
        let (unfused, cpu) = (run(Device::Gpu, false), run(Device::Cpu, false));
        let tol = match op {
            BinaryOp::Pow | BinaryOp::Atan2 => TRANSCENDENTAL_ULPS,
            _ => 0,
        };
        let mut check = |what: &str, want: &[f32]| {
            let wrong: Vec<usize> = (0..N).filter(|&i| ulps(kept[i], want[i]) > tol).collect();
            if let Some(&i) = wrong.first() {
                bad.push(format!(
                    "{op:?} vs {what}: {} of {N} beyond {tol} ULP, e.g. ({}, {}) -> {:e} vs region {:e}",
                    wrong.len(),
                    x[i],
                    y[i],
                    want[i],
                    kept[i]
                ));
            }
        };
        check("cpu", &cpu);
        if tol == 0 {
            check("unfused wgpu", &unfused);
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}
