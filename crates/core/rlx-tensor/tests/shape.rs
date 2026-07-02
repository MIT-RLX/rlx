// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! NumPy-grade shape ops, evaluated end-to-end. Run:
//! `cargo test -p rlx-tensor --features eval`.
#![cfg(feature = "eval")]

use rlx_tensor::{Tensor, cat, stack};

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-5, "{a:?} != {b:?}");
    }
}

#[test]
fn broadcast_to_repeats_rows() {
    // [1,3] -> [2,3]
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], [1, 3]);
    approx(
        &x.broadcast_to([2, 3]).to_vec(),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    );
}

#[test]
fn binary_op_broadcasts() {
    // [2,3] + [3] -> per-row add
    let a = Tensor::from_vec(vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0], [2, 3]);
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
    approx(&(&a + &b).to_vec(), &[1.0, 2.0, 3.0, 11.0, 12.0, 13.0]);
}

#[test]
fn flatten_preserves_data() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    approx(&x.flatten().to_vec(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn unsqueeze_squeeze_roundtrip() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
    let y = x.unsqueeze(0).squeeze(0);
    approx(&y.to_vec(), &[1.0, 2.0, 3.0]);
}

#[test]
fn cat_concatenates_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0], [2]);
    let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], [3]);
    approx(&cat(&[&a, &b], 0).to_vec(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn softmax_normalizes() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
    let out = x.softmax(0).to_vec();
    // exp(i)/sum: [0.0900, 0.2447, 0.6652], sums to 1.
    approx(&out, &[0.09003057, 0.24472847, 0.66524096]);
    assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn masked_fill_replaces() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [4]);
    let thr = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], [4]);
    let mask = data.gt(&thr); // [f,f,t,t]
    approx(
        &data.masked_fill(&mask, -1.0).to_vec(),
        &[1.0, 2.0, -1.0, -1.0],
    );
}

#[test]
fn masked_softmax_attention_pattern() {
    // Mask the last position with -inf, then softmax -> it gets ~0 weight and
    // the rest renormalize. This is the attention-masking pattern.
    let scores = Tensor::from_vec(vec![1.0, 1.0, 1.0], [3]);
    let last = Tensor::from_vec(vec![0.0, 0.0, 1.0], [3]);
    let zero = Tensor::from_vec(vec![0.0, 0.0, 0.0], [3]);
    let mask = last.gt(&zero); // [false, false, true]
    let w = scores
        .masked_fill(&mask, f32::NEG_INFINITY as f64)
        .softmax(0)
        .to_vec();
    approx(&w, &[0.5, 0.5, 0.0]);
}

#[test]
fn argmax_argmin() {
    // rows: [1,5,2] -> max@1,min@0 ; [9,3,4] -> max@0,min@1
    let x = Tensor::from_vec(vec![1.0, 5.0, 2.0, 9.0, 3.0, 4.0], [2, 3]);
    approx(&x.argmax(1, false).to_vec(), &[1.0, 0.0]);
    approx(&x.argmin(1, false).to_vec(), &[0.0, 1.0]);
}

#[test]
fn eye_is_identity() {
    approx(
        &Tensor::eye(3).to_vec(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
}

#[test]
fn negative_slice_indices() {
    use rlx_tensor::{ax, rg, s, tail};
    let a = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], [5]);
    approx(&a.slice(s![rg(-3, -1)]).to_vec(), &[2.0, 3.0]); // [-3..-1] = idx 2,3
    approx(&a.slice(s![tail(-2)]).to_vec(), &[3.0, 4.0]); // last two
    // 2-D: keep all rows, last column (size-1)
    let m = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    approx(&m.slice(s![ax(), rg(-1, 2)]).to_vec(), &[2.0, 4.0]);
}

#[test]
fn select_drops_axis() {
    let m = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    // last row, axis dropped -> shape [3]
    let row = m.select(0, -1);
    assert_eq!(row.shape().dims().len(), 1);
    approx(&row.to_vec(), &[4.0, 5.0, 6.0]);
}

#[test]
fn index_select_gathers_rows() {
    let table = Tensor::from_vec(vec![10.0, 11.0, 12.0, 13.0, 14.0], [5]);
    let idx = Tensor::index_vec([0_i64, 2, 4]);
    approx(&table.index_select(0, &idx).to_vec(), &[10.0, 12.0, 14.0]);
}

#[test]
fn stack_adds_axis() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], [3]);
    // stack on axis 0 -> [2,3] row-major = a then b
    approx(
        &stack(&[&a, &b], 0).to_vec(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
}
