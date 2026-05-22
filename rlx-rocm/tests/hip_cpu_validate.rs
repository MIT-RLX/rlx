// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! HIP-CPU validation basic tests for rlx-rocm.
//!
//! Only compiled with `cargo test -p rlx-rocm --features hip-cpu-validate`.
//!
//! Comprehensive kernel-logic validation lives in
//! `rlx-cuda/tests/hip_cpu_validate.rs` (38 tests, one per kernel
//! family). The .cu sources, the cpp wrapper layer, and the FFI
//! bindings are all shared — rlx-rocm pulls them in via
//! `include_str!` (sources), `#include` (cpp wrappers), and
//! `#[path]` (Rust FFI). So the kernel-correctness story is already
//! covered by rlx-cuda; this file only needs to prove that **the
//! shared harness compiles and links correctly when driven from the
//! rlx-rocm crate**.
//!
//! Five representative families covered:
//!   * binary    — element-wise (add)
//!   * unary     — element-wise (relu)
//!   * matmul    — the headline GEMM kernel
//!   * softmax   — block-wide reduction
//!   * conv2d    — multidim dispatch
//!
//! If any of these fail with a link error, the build.rs / cpu_dispatch
//! wiring is wrong. If they fail at runtime, the kernel logic
//! regressed — and the same regression should also surface in
//! rlx-cuda's full suite, so that's where to dig.

#![cfg(feature = "hip-cpu-validate")]

use rlx_rocm::cpu_dispatch::*;

fn approx(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "idx {i}: got {a}, want {e} (tol {tol})"
        );
    }
}

#[test]
fn binary_add_links_and_runs() {
    let mut arena = vec![
        1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0, 0.0,
    ];
    run_binary(&mut arena, 4, 0, 4, 8, /*Add*/ 0);
    assert_eq!(&arena[8..12], &[11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn unary_relu_links_and_runs() {
    let mut arena = vec![-2.0, -0.5, 0.0, 1.5, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    run_unary(&mut arena, 5, 0, 5, /*relu*/ 0);
    assert_eq!(&arena[5..10], &[0.0, 0.0, 0.0, 1.5, 3.0]);
}

#[test]
fn matmul_2x3_x_3x2_links_and_runs() {
    // A: [1,2,3; 4,5,6]   B: [1,0; 0,1; 1,0]   C = A·B = [4,2; 10,5]
    let mut arena = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // A (2×3) at offset 0
        1.0, 0.0, 0.0, 1.0, 1.0, 0.0, // B (3×2) at offset 6
        0.0, 0.0, 0.0, 0.0, // C (2×2) at offset 12
    ];
    run_matmul(&mut arena, 2, 3, 2, 0, 6, 12, /*batch*/ 1);
    approx(&arena[12..16], &[4.0, 2.0, 10.0, 5.0], 1e-5);
}

#[test]
fn softmax_links_and_runs() {
    // Two rows of 3 elements; softmax should normalize each row.
    let mut arena = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    run_softmax(&mut arena, /*outer*/ 1, /*inner*/ 3, 0, 3);
    let row = &arena[3..6];
    let sum: f32 = row.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax row should sum to 1, got {sum}"
    );
    assert!(
        row[0] < row[1] && row[1] < row[2],
        "softmax should be monotonic in input"
    );
}

#[test]
fn conv2d_links_and_runs() {
    // 1×1 conv with weight=2 → output = input × 2.
    // Input: [N=1, Cin=1, H=2, W=2] = [1, 2, 3, 4]
    // Weight: [Cout=1, Cin=1, KH=1, KW=1] = [2.0]
    // Output: [1, 1, 2, 2] = [2, 4, 6, 8]
    let mut arena = vec![
        1.0, 2.0, 3.0, 4.0, // input  (offset 0)
        2.0, // weight (offset 4)
        0.0, 0.0, 0.0, 0.0, // output (offset 5)
    ];
    run_conv2d(
        &mut arena, /*n*/ 1, /*c_in*/ 1, /*c_out*/ 1, /*h*/ 2, /*w*/ 2,
        /*h_out*/ 2, /*w_out*/ 2, /*kh*/ 1, /*kw*/ 1, /*sh*/ 1,
        /*sw*/ 1, /*ph*/ 0, /*pw*/ 0, /*dh*/ 1, /*dw*/ 1,
        /*groups*/ 1, /*x_off*/ 0, /*w_off*/ 4, /*y_off*/ 5,
    );
    approx(&arena[5..9], &[2.0, 4.0, 6.0, 8.0], 1e-5);
}
