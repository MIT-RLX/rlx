// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! End-to-end basic test for the MLX `Op::DenseSolve` lowering.
//!
//! Builds a tiny f32 system A·x = b, runs it on MLX, checks the result
//! against a hand-computed reference. Same shape of test as check.rs,
//! restricted to macOS where MLX is actually built.
//!
//! The batched variant is exercised separately so a regression in
//! either path is bisectable.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Shape};
use rlx_mlx::{MlxExecutable, MlxMode};

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn dense_solve_3x3_matches_reference() {
    // System (well-conditioned, hand-verified):
    //   [ 4  -1   0 ] [x0]   [ 2 ]
    //   [-1   4  -1 ] [x1] = [ 6 ]
    //   [ 0  -1   4 ] [x2]   [ 2 ]
    //
    // Back-sub: row 1 → x1 = 4x0 − 2; row 3 → x2 = (x1 + 2)/4 = x0;
    // row 2 → 14x0 = 14, so x = [1, 2, 1]. Verify: A·x = [2, 6, 2] ✓.
    let mut g = Graph::new("dense_solve_3x3");
    let a = g.input("a", Shape::new(&[3, 3], DType::F32));
    let b = g.input("b", Shape::new(&[3], DType::F32));
    let x = g.dense_solve(a, b, Shape::new(&[3], DType::F32));
    g.set_outputs(vec![x]);

    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Lazy);
    let a_data: Vec<f32> = vec![4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 4.0];
    let b_data: Vec<f32> = vec![2.0, 6.0, 2.0];
    let outs = exe.run(&[("a", &a_data), ("b", &b_data)]);

    let expected = [1.0_f32, 2.0, 1.0];
    assert_eq!(outs.len(), 1);
    assert!(
        close(&outs[0], &expected, 1e-5),
        "got {:?}, expected {:?}",
        outs[0],
        expected,
    );
}

#[test]
fn batched_dense_solve_two_systems() {
    // Two independent 2x2 systems stacked along axis 0:
    //   slice 0:  [[2, 0],[0, 2]] · x = [4, 6]   →  x = [2, 3]
    //   slice 1:  [[1, 1],[0, 1]] · x = [3, 1]   →  x = [2, 1]
    // (Slice 1 is upper-triangular so the reference is trivial.)
    let mut g = Graph::new("batched_dense_solve_2x2");
    let a = g.input("a", Shape::new(&[2, 2, 2], DType::F32));
    let b = g.input("b", Shape::new(&[2, 2], DType::F32));
    let x = g.batched_dense_solve(a, b, Shape::new(&[2, 2], DType::F32));
    g.set_outputs(vec![x]);

    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Lazy);
    let a_data: Vec<f32> = vec![
        2.0, 0.0, 0.0, 2.0, // slice 0
        1.0, 1.0, 0.0, 1.0, // slice 1
    ];
    let b_data: Vec<f32> = vec![
        4.0, 6.0, // slice 0
        3.0, 1.0, // slice 1
    ];
    let outs = exe.run(&[("a", &a_data), ("b", &b_data)]);

    let expected = [2.0, 3.0, 2.0, 1.0];
    assert!(
        close(&outs[0], &expected, 1e-5),
        "got {:?}, expected {:?}",
        outs[0],
        expected,
    );
}
