// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression: an F64 graph executed through the f32 `run()` entry point.
//!
//! `rlx_driver::arena::{write_typed_from_f32, read_typed_to_f32}` previously
//! lacked a `DType::F64` arm — F64 tensors fell through to the F32 branch,
//! which treated the 8-byte-per-element F64 slot as f32 (4 B), mis-sizing it
//! and returning garbage. The fix widens f32→f64 on write and narrows f64→f32
//! on read (values carry f32 precision on this entry path; `run_typed`'s
//! `all_f64` branch is the full-precision route).

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn f64s_to_bytes(xs: &[f64]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_to_f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn f64_matmul_via_f32_run_is_correct() {
    let mut g = Graph::new("f64_matmul");
    let a = g.input("a", Shape::new(&[2, 2], DType::F64));
    let b = g.input("b", Shape::new(&[2, 2], DType::F64));
    let c = g.matmul(a, b, Shape::new(&[2, 2], DType::F64));
    g.set_outputs(vec![c]);

    let mut sess = Session::new(Device::Cpu).compile(g);
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]] → C = [[19,22],[43,50]]
    let out = sess.run(&[("a", &[1.0, 2.0, 3.0, 4.0]), ("b", &[5.0, 6.0, 7.0, 8.0])]);

    let c = &out[0];
    assert_eq!(c.len(), 4, "F64 matmul via f32 run() returned wrong length");
    let want = [19.0f32, 22.0, 43.0, 50.0];
    for (i, (&got, &w)) in c.iter().zip(&want).enumerate() {
        assert!(
            (got - w).abs() < 1e-4,
            "F64 run() out[{i}]: got {got}, want {w}"
        );
    }
}

#[test]
fn f64_matmul_run_typed_is_full_precision() {
    // `run_typed` must preserve genuine F64 precision end to end (not
    // silently downcast to f32). The `1 + 1e-10` perturbations round to
    // exactly 1.0 / 4.0 in f32, so an f32 round-trip anywhere would drop
    // the `1e-10` and the checks below (tol 1e-13) would fail.
    let a = [1.0 + 1e-10, 2.0, 3.0, 4.0 + 1e-10]; // [2,2]
    let b = [5.0, 6.0, 7.0, 8.0]; // [2,2]
    // Reference C = A·B computed in f64.
    let mut want = [0.0f64; 4];
    for i in 0..2 {
        for j in 0..2 {
            want[i * 2 + j] = a[i * 2] * b[j] + a[i * 2 + 1] * b[2 + j];
        }
    }

    let mut g = Graph::new("f64_matmul_typed");
    let an = g.input("a", Shape::new(&[2, 2], DType::F64));
    let bn = g.input("b", Shape::new(&[2, 2], DType::F64));
    let cn = g.matmul(an, bn, Shape::new(&[2, 2], DType::F64));
    g.set_outputs(vec![cn]);

    let mut sess = Session::new(Device::Cpu).compile(g);
    let out = sess.run_typed(&[
        ("a", &f64s_to_bytes(&a), DType::F64),
        ("b", &f64s_to_bytes(&b), DType::F64),
    ]);
    assert_eq!(out[0].1, DType::F64, "output dtype must be F64");
    let got = bytes_to_f64s(&out[0].0);
    assert_eq!(got.len(), 4);
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-13,
            "F64 run_typed out[{i}]: got {g:.15}, want {w:.15} (precision lost?)"
        );
    }
    // Explicitly confirm the sub-f32 term survived: C[0,0] carries the 1e-10.
    assert!(
        (got[0] - 19.0).abs() > 1e-11,
        "the 1e-10 perturbation was rounded away — F64 collapsed to f32"
    );
}
