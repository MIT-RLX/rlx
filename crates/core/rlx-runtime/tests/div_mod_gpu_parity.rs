// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The GPU backends' `Div` and `Mod` (wgpu, and native Vulkan) must equal
//! rlx-cpu bit for bit: the correctly-rounded IEEE quotient and C `fmod`.
//!
//! * WGSL and Vulkan GLSL only promise 2.5 ULP for `/`, and Metal's default
//!   fast math lowers it to a reciprocal multiply — 1 ULP off on about a
//!   quarter of operands.
//! * Neither language has `fmod`; `a - b * trunc(a / b)` loses the remainder
//!   once the quotient outgrows 24 bits, sign included.
//!
//! The edge rows pin what the exact forms must leave alone: zero divisors,
//! infinities, NaN, overflowing quotients and the sign of a zero result.

#![cfg(all(feature = "cpu", any(feature = "gpu", feature = "vulkan")))]

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

const INF: f32 = f32::INFINITY;

/// `2^16` operand pairs with random signs and mantissas, `a`'s exponent drawn
/// from `a_exp` and `b`'s from `b_exp` (biased), followed by `edges`.
fn operands(
    a_exp: std::ops::Range<u32>,
    b_exp: std::ops::Range<u32>,
    edges: &[(f32, f32)],
) -> (Vec<f32>, Vec<f32>) {
    let mut s = 0x2545_f491_u32;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let mut draw = |e: &std::ops::Range<u32>| {
        let m = next() & 0x807f_ffff;
        f32::from_bits(m | ((next() % (e.end - e.start) + e.start) << 23))
    };
    let (mut a, mut b) = (Vec::new(), Vec::new());
    while a.len() < 1 << 16 {
        a.push(draw(&a_exp));
        b.push(draw(&b_exp));
    }
    for &(x, y) in edges {
        a.push(x);
        b.push(y);
    }
    (a, b)
}

/// The GPU backends compiled into this test binary.
fn devices() -> Vec<Device> {
    vec![
        #[cfg(feature = "gpu")]
        Device::Gpu,
        #[cfg(feature = "vulkan")]
        Device::Vulkan,
    ]
}

/// Run `op` elementwise on each GPU backend and rlx-cpu; check the CPU against
/// `oracle` and each backend against the CPU (NaN payloads aside).
fn same_bits(op: BinaryOp, a: &[f32], b: &[f32], oracle: fn(f32, f32) -> f32) {
    let _gpu = common::serialize_gpu();
    let n = a.len();
    let build = || {
        let mut g = Graph::new("binary");
        let x = g.input("a", Shape::new(&[n], DType::F32));
        let y = g.input("b", Shape::new(&[n], DType::F32));
        let q = g.binary(op, x, y, Shape::new(&[n], DType::F32));
        g.set_outputs(vec![q]);
        g
    };
    let run = |d: Device| Session::new(d).compile(build()).run(&[("a", a), ("b", b)]);
    let cpu = run(Device::Cpu);
    let same = |x: f32, y: f32| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
    for i in 0..n {
        assert!(
            same(cpu[0][i], oracle(a[i], b[i])),
            "{op:?}({}, {}): cpu is not the oracle",
            a[i],
            b[i]
        );
    }
    let mut bad = Vec::new();
    for dev in devices() {
        if common::skip_unless(dev) {
            continue;
        }
        let got = run(dev);
        let wrong: Vec<String> = (0..n)
            .filter(|&i| !same(got[0][i], cpu[0][i]))
            .map(|i| {
                format!(
                    "{op:?}({}, {}) = cpu {:e} {dev:?} {:e}",
                    a[i], b[i], cpu[0][i], got[0][i]
                )
            })
            .collect();
        if !wrong.is_empty() {
            bad.push(format!(
                "{dev:?}: {} of {n} differ, e.g. {:?}",
                wrong.len(),
                &wrong[..wrong.len().min(6)]
            ));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// Quotients across `2^-119 ..= 2^119`, all normal (Metal flushes subnormals;
/// the CPU keeps them), including the tiny ones whose correction is subnormal.
#[test]
fn div_is_correctly_rounded() {
    let edges = [
        (1.0, 0.0),
        (-3.5, 0.0),
        (2.0, -0.0),
        (0.0, 0.0),
        (0.0, 7.0),
        (-0.0, 7.0),
        (5.0, INF),
        (-5.0, INF),
        (INF, 3.0),
        (-INF, 3.0),
        (INF, INF),
        (f32::NAN, 2.0),
        (2.0, f32::NAN),
        (3.0e38, 0.25),
        (-3.0e38, 0.5),
        (f32::MAX, 1.0),
        (1.0, 3.0),
        (2.0, 3.0),
        (10.0, 10.0),
        (7.0, 0.1),
    ];
    let (a, b) = operands(67..187, 67..187, &edges);
    same_bits(BinaryOp::Div, &a, &b, |x, y| x / y);
}

/// Divisors `2^-20 ..= 2^19` against dividends `2^-60 ..= 2^59`: quotients from
/// far below one (the remainder is the dividend) to `2^79` (many reductions).
#[test]
fn mod_is_exact_fmod() {
    let edges = [
        (1.0, 0.0),
        (-1.0, -0.0),
        (0.0, 0.0),
        (INF, 3.0),
        (-INF, 3.0),
        (5.0, INF),
        (-5.0, -INF),
        (f32::NAN, 2.0),
        (2.0, f32::NAN),
        (0.0, 7.0),
        (-0.0, 7.0),
        (-5.0, 5.0),
        (5.0, -5.0),
        (5.5, -2.0),
        (-5.5, 2.0),
        (7.0, 0.1),
        (f32::MAX, 3.0),
        (-f32::MAX, 1.0e-30),
        (1.0e-30, 3.0),
    ];
    let (a, b) = operands(67..187, 107..147, &edges);
    same_bits(BinaryOp::Mod, &a, &b, |x, y| x % y);
}
