// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! dtype breadth: typed constructors, `cast`, and typed readback
//! (`to_vec_f64/i64/i32/bool`). Run: `cargo test -p rlx-tensor --features eval`.
#![cfg(feature = "eval")]

use rlx_tensor::{DType, Tensor};

#[test]
fn f64_roundtrip() {
    let data = vec![1.5, -2.25, 3.125];
    let x = Tensor::from_f64(data.clone(), [3]);
    assert_eq!(x.dtype(), DType::F64);
    assert_eq!(x.to_vec_f64(), data);
}

#[test]
fn i64_roundtrip() {
    let data = vec![5_i64, -7, 1_000_000];
    let x = Tensor::from_i64(data.clone(), [3]);
    assert_eq!(x.dtype(), DType::I64);
    assert_eq!(x.to_vec_i64(), data);
}

#[test]
fn cast_f32_to_f64() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]).cast(DType::F64);
    assert_eq!(x.dtype(), DType::F64);
    assert_eq!(x.to_vec_f64(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn cast_f32_to_i32() {
    // ONNX / NumPy Cast float→int truncates toward zero.
    let x = Tensor::from_vec(vec![1.4, 2.6, -3.2], [3]).cast(DType::I32);
    assert_eq!(x.dtype(), DType::I32);
    assert_eq!(x.to_vec_i32(), vec![1, 2, -3]);
}

#[test]
fn cast_f32_to_i64() {
    let x = Tensor::from_vec(vec![1.4, 2.6, -3.2], [3]).cast(DType::I64);
    assert_eq!(x.to_vec_i64(), vec![1, 2, -3]);
}

#[test]
fn comparison_reads_as_bool() {
    let a = Tensor::from_vec(vec![1.0, 5.0, 3.0], [3]);
    let b = Tensor::from_vec(vec![2.0, 2.0, 3.0], [3]);
    // a < b -> [true, false, false]
    assert_eq!(a.lt(&b).to_vec_bool(), vec![true, false, false]);
}

#[test]
fn index_vec_reads_back() {
    let idx = Tensor::index_vec([0_i64, 2, 4, 6]);
    assert_eq!(idx.to_vec_i64(), vec![0, 2, 4, 6]);
}
