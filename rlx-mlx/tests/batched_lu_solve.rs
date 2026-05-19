// SPDX-License-Identifier: GPL-3.0-only
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! End-to-end smoke test for the batched LU+solve Metal kernel.
//!
//! Builds two diagonally-dominant 3x3 systems, dispatches the custom
//! kernel via `MlxKernel::execute`, checks per-batch results match a
//! hand-computed reference. Diagonally-dominant matters because the
//! v1 kernel has no partial pivoting (see batched_lu_kernel.rs's
//! "Remaining work" docblock).

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Shape};
use rlx_mlx::Array;
use rlx_mlx::batched_lu_kernel::{BatchedLuSolveMetal, KERNEL_NAME};
use rlx_mlx::op_registry::MlxKernel;

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn batched_lu_solve_two_3x3_systems() {
    // Two independent diagonally-dominant 3x3 systems stacked on
    // axis 0. Same matrix as `dense_solve_3x3_matches_reference`
    // for slice 0 (so the reference is already verified). Slice 1
    // is a different DD system to confirm the batch axis works.
    //
    // Slice 0:  [[4 -1 0],[-1 4 -1],[0 -1 4]] · x = [2, 6, 2] → x = [1, 2, 1]
    // Slice 1:  [[5  1 0],[ 1 5  1],[0  1 5]] · x = [6, 12, 6]
    //   Solve by hand: row 1 → 5x0 + x1 = 6,
    //                  row 3 → x1 + 5x2 = 6,
    //                  row 2 → x0 + 5x1 + x2 = 12.
    //   By symmetry x0 = x2. Row 1 + Row 3: 5x0 + 2x1 + 5x2 = 12,
    //   so 10x0 + 2x1 = 12 → x1 = 6 - 5x0.
    //   Sub into row 2: x0 + 5(6 - 5x0) + x0 = 12 → 2x0 - 25x0 + 30 = 12
    //   → -23x0 = -18 → x0 = 18/23. Then x1 = 6 - 90/23 = (138-90)/23 = 48/23.
    //   So slice 1 → x = [18/23, 48/23, 18/23] ≈ [0.7826, 2.0870, 0.7826].
    let a_data: Vec<f32> = vec![
        // slice 0
        4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 4.0, // slice 1
        5.0, 1.0, 0.0, 1.0, 5.0, 1.0, 0.0, 1.0, 5.0,
    ];
    let b_data: Vec<f32> = vec![
        2.0, 6.0, 2.0, // slice 0
        6.0, 12.0, 6.0, // slice 1
    ];

    let a = Array::from_f32_slice(&a_data, &[2, 3, 3], DType::F32).unwrap();
    let b = Array::from_f32_slice(&b_data, &[2, 3], DType::F32).unwrap();
    let out_shape = Shape::new(&[2, 3], DType::F32);

    let kernel = BatchedLuSolveMetal;
    assert_eq!(kernel.name(), KERNEL_NAME);

    let result = kernel
        .execute(&[&a, &b], &out_shape, &[])
        .expect("kernel dispatch must succeed");
    let got = result.to_f32().expect("readback");

    let expected = [1.0, 2.0, 1.0, 18.0 / 23.0, 48.0 / 23.0, 18.0 / 23.0];
    assert!(
        close(&got, &expected, 5e-5),
        "got {:?}, expected {:?}",
        got,
        expected,
    );
}

#[test]
fn batched_lu_solve_requires_pivoting() {
    // Both slices have a zero on the leading diagonal — cannot solve
    // without a row pivot. Without pivoting the kernel would divide
    // by zero on the first elimination step → NaN. With pivoting the
    // first row swaps and we get the right answer.
    //
    // Slice 0:  [[0, 1],[1, 1]] · x = [1, 0]   → x = [-1, 1]
    //   verify: 0*-1 + 1*1 = 1 ✓; 1*-1 + 1*1 = 0 ✓
    // Slice 1:  [[0, 2],[3, 1]] · x = [4, 5]
    //   row 1 → x1 = 2; row 2 → 3 x0 + 2 = 5 → x0 = 1. So x = [1, 2].
    //   verify: 0*1 + 2*2 = 4 ✓; 3*1 + 1*2 = 5 ✓
    let a_data: Vec<f32> = vec![
        // slice 0
        0.0, 1.0, 1.0, 1.0, // slice 1
        0.0, 2.0, 3.0, 1.0,
    ];
    let b_data: Vec<f32> = vec![1.0, 0.0, 4.0, 5.0];

    let a = Array::from_f32_slice(&a_data, &[2, 2, 2], DType::F32).unwrap();
    let b = Array::from_f32_slice(&b_data, &[2, 2], DType::F32).unwrap();
    let out_shape = Shape::new(&[2, 2], DType::F32);

    let result = BatchedLuSolveMetal
        .execute(&[&a, &b], &out_shape, &[])
        .expect("kernel dispatch must succeed");
    let got = result.to_f32().expect("readback");

    let expected = [-1.0_f32, 1.0, 1.0, 2.0];
    assert!(
        close(&got, &expected, 5e-5),
        "got {:?}, expected {:?}",
        got,
        expected,
    );
}
