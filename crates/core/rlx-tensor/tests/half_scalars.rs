// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! A float literal next to an f16 / bf16 tensor, as next to an f64 one, is a
//! value of the tensor's type (the backend widens halves, so the f32
//! constant `promote_scalar` makes computes right).
#![cfg(feature = "eval")]

use rlx_tensor::{DType, Tensor};

fn half(dt: DType) -> Tensor {
    Tensor::from_vec(vec![1.5f32, -2.25, 4.0, 0.5], [4]).cast(dt)
}

fn back(t: &Tensor) -> Vec<f32> {
    t.cast(DType::F32).to_vec()
}

#[test]
fn literals_on_half_tensors() {
    for dt in [DType::F16, DType::BF16] {
        let x = half(dt);
        assert_eq!(
            back(&(&x * 2.0f64)),
            vec![3.0, -4.5, 8.0, 1.0],
            "{dt:?} * 2.0"
        );
        assert_eq!(
            back(&(&x + 1.0f64)),
            vec![2.5, -1.25, 5.0, 1.5],
            "{dt:?} + 1.0"
        );
        assert_eq!(
            back(&(&x - 0.5f64)),
            vec![1.0, -2.75, 3.5, 0.0],
            "{dt:?} - 0.5"
        );
        assert_eq!((&x * 2.0f64).dtype(), dt, "{dt:?} keeps its dtype");
    }
}

/// An f64 result read back without losing digits, on the auto-picked and an
/// explicit device.
#[test]
fn f64_results_keep_their_digits() {
    let third = Tensor::from_f64([1.0], [1]) / Tensor::from_f64([3.0], [1]);
    assert_eq!(third.item_f64(), 1.0 / 3.0);
    assert_eq!(
        third.on(rlx_tensor::Device::Cpu).to_vec_f64(),
        vec![1.0 / 3.0]
    );
}
