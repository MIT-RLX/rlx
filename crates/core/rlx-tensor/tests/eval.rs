// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end materialization: build with NumPy-style constructors, then
//! compile + run on CPU via the `eval` feature. Run with:
//! `cargo test -p rlx-tensor --features eval`.
#![cfg(feature = "eval")]

use rlx_tensor::Tensor;

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-5, "{a:?} != {b:?}");
    }
}

#[test]
fn add_two_constants() {
    // a and b come from independent constructors -> graphs auto-merge.
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
    let b = Tensor::ones([3]);
    let c = &a + &b;
    approx(&c.to_vec(), &[2.0, 3.0, 4.0]);
}

#[test]
fn fused_chain() {
    let x = Tensor::from_vec(vec![-1.0, 0.5, 2.0, -3.0], [4]);
    // relu(x * 2 + 1)
    let y = ((&x * 2.0f32) + 1.0f32).relu();
    approx(&y.to_vec(), &[0.0, 2.0, 5.0, 0.0]);
}

#[test]
fn matmul_constants() {
    // [2x3] @ [3x2] = [2x2]
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2]);
    let c = a.matmul(&b);
    // row0: [1+0+3, 0+2+3] = [4,5]; row1: [4+0+6, 0+5+6] = [10,11]
    approx(&c.to_vec(), &[4.0, 5.0, 10.0, 11.0]);
}

#[test]
fn reduce_sum() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let s = a.sum([1], false);
    approx(&s.to_vec(), &[3.0, 7.0]);
}

#[test]
fn clamp_and_minmax() {
    let a = Tensor::from_vec(vec![-2.0, 0.5, 9.0], [3]);
    approx(&a.clamp(0.0, 6.0).to_vec(), &[0.0, 0.5, 6.0]);
    let b = Tensor::from_vec(vec![1.0, 1.0, 1.0], [3]);
    approx(&a.maximum(&b).to_vec(), &[1.0, 1.0, 9.0]);
}

#[test]
fn arange_constructor() {
    let a = Tensor::arange(5);
    approx(&a.to_vec(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[cfg(feature = "eval-metal")]
#[test]
fn metal_is_auto_selected_and_computes() {
    use rlx_tensor::{Device, fastest_device, is_available};
    assert!(is_available(Device::Metal), "Metal backend should be live");
    // A GPU backend should outrank CPU. (Don't hard-code Metal: with `eval-mlx` also
    // enabled MLX is the fastest device, which is fine.)
    assert_ne!(
        fastest_device(),
        Device::Cpu,
        "a GPU backend should outrank CPU"
    );
    // End-to-end on GPU via the auto path.
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], [3]);
    approx(&(&a * &b).to_vec(), &[4.0, 10.0, 18.0]);
}

#[test]
fn auto_selected_device_is_available() {
    use rlx_tensor::{Device, available_devices, fastest_device, is_available};
    let d = fastest_device();
    assert!(is_available(d), "auto-selected device must be available");
    assert!(available_devices().contains(&Device::Cpu));
    // to_vec() with no device argument runs on the auto-selected backend.
    let x = Tensor::from_vec(vec![1.0, 2.0], [2]);
    approx(&x.relu().to_vec(), &[1.0, 2.0]);
}

#[test]
fn reused_operand_across_exprs() {
    // `b` is consumed by two separate cross-graph merges; both must stay
    // correct (compiler CSE collapses the duplicated constants).
    let a = Tensor::from_vec(vec![1.0, 2.0], [2]);
    let b = Tensor::from_vec(vec![10.0, 20.0], [2]);
    let lhs = &a + &b; // [11, 22]
    let rhs = &a * &b; // [10, 40]
    let out = &lhs + &rhs; // [21, 62]
    approx(&out.to_vec(), &[21.0, 62.0]);
}
