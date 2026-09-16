// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Roll` (cyclic shift) forward parity and gradient.
//!
//! No backend claims `OpKind::Roll`, so `LowerRoll` (narrow + concat) is the
//! only implementation — which makes it the definition and leaves nothing to
//! compare it against internally. The oracle here is therefore an independent
//! index computation, `out[i] = x[(i − shift) mod n]`, written directly from
//! `jnp.roll` semantics rather than from the lowering.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn run(device: Device, dims: &[usize], shifts: &[i64], axes: &[usize], x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("roll");
    let inp = g.input("x", Shape::new(dims, DType::F32));
    let y = g.roll_(inp, shifts.to_vec(), axes.to_vec());
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

/// Independent oracle: strides computed from scratch, one axis at a time.
fn reference(dims: &[usize], shifts: &[i64], axes: &[usize], x: &[f32]) -> Vec<f32> {
    let mut cur = x.to_vec();
    for (&shift, &axis) in shifts.iter().zip(axes.iter()) {
        let n = dims[axis];
        if n == 0 {
            continue;
        }
        let k = shift.rem_euclid(n as i64) as usize;
        let inner: usize = dims[axis + 1..].iter().product();
        let outer: usize = dims[..axis].iter().product();
        let mut out = vec![0.0f32; cur.len()];
        for o in 0..outer {
            for i in 0..n {
                // out[i] pulls from (i - k) mod n
                let src = (i + n - k) % n;
                for j in 0..inner {
                    out[(o * n + i) * inner + j] = cur[(o * n + src) * inner + j];
                }
            }
        }
        cur = out;
    }
    cur
}

fn iota(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}

#[test]
fn roll_1d_matches_reference() {
    let x = iota(8);
    for shift in [-9i64, -3, -1, 0, 1, 3, 8, 11] {
        let got = run(Device::Cpu, &[8], &[shift], &[0], &x);
        let want = reference(&[8], &[shift], &[0], &x);
        assert_eq!(got, want, "shift {shift}");
    }
}

/// The documented example, spelled out — `roll([0..5], 2)` must move the tail
/// to the front, not the head.
#[test]
fn roll_direction_is_jnp_convention() {
    let x = iota(5);
    let got = run(Device::Cpu, &[5], &[2], &[0], &x);
    assert_eq!(got, vec![3.0, 4.0, 0.0, 1.0, 2.0]);
    // Negative shift is the inverse.
    let back = run(Device::Cpu, &[5], &[-2], &[0], &got);
    assert_eq!(back, x);
}

#[test]
fn roll_2d_each_axis() {
    let dims = [3usize, 4];
    let x = iota(12);
    for axis in 0..2 {
        for shift in [-5i64, -1, 1, 2, 7] {
            let got = run(Device::Cpu, &dims, &[shift], &[axis], &x);
            let want = reference(&dims, &[shift], &[axis], &x);
            assert_eq!(got, want, "axis {axis} shift {shift}");
        }
    }
}

#[test]
fn roll_multi_axis_composes() {
    let dims = [2usize, 3, 4];
    let x = iota(24);
    let shifts = [1i64, -1, 2];
    let axes = [0usize, 1, 2];
    let got = run(Device::Cpu, &dims, &shifts, &axes, &x);
    let want = reference(&dims, &shifts, &axes, &x);
    assert_eq!(got, want);
}

/// A roll is a permutation, so it must preserve the multiset of values and be
/// invertible — properties that hold for no other reason than correctness.
#[test]
fn roll_is_a_permutation() {
    let dims = [4usize, 5];
    let x: Vec<f32> = (0..20).map(|i| (i as f32) * 1.5 - 7.0).collect();
    let rolled = run(Device::Cpu, &dims, &[3, -2], &[0, 1], &x);

    let mut a = x.clone();
    let mut b = rolled.clone();
    a.sort_by(|p, q| p.partial_cmp(q).unwrap());
    b.sort_by(|p, q| p.partial_cmp(q).unwrap());
    assert_eq!(a, b, "roll changed the multiset of values");

    let back = run(Device::Cpu, &dims, &[-3, 2], &[0, 1], &rolled);
    assert_eq!(back, x, "roll by -s did not invert roll by s");
}

/// A full-period shift is the identity.
#[test]
fn roll_by_length_is_identity() {
    let x = iota(6);
    assert_eq!(run(Device::Cpu, &[6], &[6], &[0], &x), x);
    assert_eq!(run(Device::Cpu, &[6], &[-6], &[0], &x), x);
    assert_eq!(run(Device::Cpu, &[6], &[0], &[0], &x), x);
}
