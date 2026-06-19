// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! MLP capstone: learn XOR with a 2-layer tanh network. Exercises deep
//! autodiff (stacked matmuls + nonlinearity, gradients to four param tensors)
//! and Adam. Deterministic init — no RNG, fully reproducible. Run:
//! `cargo test -p rlx-tensor --features optim`.
#![cfg(feature = "optim")]

use rlx_tensor::{Adam, Func, shape};

/// Deterministic spread of small distinct values to break weight symmetry.
fn spread(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + seed * 17) % 23) as f32 - 11.0) * 0.05)
        .collect()
}

#[test]
fn mlp_learns_xor() {
    const H: usize = 8;
    let xs: &[f32] = &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]; // [4,2]
    let ys: &[f32] = &[0.0, 1.0, 1.0, 0.0]; // XOR  [4,1]

    // out = tanh(X @ W1 + b1) @ W2 + b2 ;  loss = mean((out - y)^2)
    let build = |s: &mut rlx_tensor::GraphScope| {
        let x = s.input("x", shape![4, 2]);
        let y = s.input("y", shape![4, 1]);
        let w1 = s.param("w1", shape![2, H]);
        let b1 = s.param("b1", shape![H]);
        let w2 = s.param("w2", shape![H, 1]);
        let b2 = s.param("b2", shape![1]);
        let h = (&x.matmul(&w1) + &b1).tanh();
        let out = &h.matmul(&w2) + &b2;
        let diff = &out - &y;
        (&diff * &diff).mean([0, 1], false)
    };

    let mut model = Func::new("mlp", build)
        .with_param("w1", spread(2 * H, 1))
        .with_param("b1", vec![0.0; H])
        .with_param("w2", spread(H, 2))
        .with_param("b2", vec![0.0]);

    let mut opt = Adam::new(0.05);
    let feed: &[(&str, &[f32])] = &[("x", xs), ("y", ys)];

    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for i in 0..4000 {
        let (next, loss) = model.train_step(&mut opt, &["w1", "b1", "w2", "b2"], feed);
        model = next;
        if i == 0 {
            first = loss[0];
        }
        last = loss[0];
    }

    assert!(
        first > 0.1,
        "expected a non-trivial initial loss, got {first}"
    );
    assert!(last < 0.02, "MLP should fit XOR, final loss {last}");

    // Predictions land on the correct side of 0.5.
    let pred_fn = Func::new("mlp_pred", |s| {
        let x = s.input("x", shape![4, 2]);
        let w1 = s.param("w1", shape![2, H]);
        let b1 = s.param("b1", shape![H]);
        let w2 = s.param("w2", shape![H, 1]);
        let b2 = s.param("b2", shape![1]);
        let h = (&x.matmul(&w1) + &b1).tanh();
        &h.matmul(&w2) + &b2
    })
    .with_param("w1", model.param_binding("w1").unwrap().to_vec())
    .with_param("b1", model.param_binding("b1").unwrap().to_vec())
    .with_param("w2", model.param_binding("w2").unwrap().to_vec())
    .with_param("b2", model.param_binding("b2").unwrap().to_vec());

    let pred = pred_fn.run(&[("x", xs)]);
    for (p, t) in pred[0].iter().zip(ys) {
        assert_eq!(*p > 0.5, *t > 0.5, "pred {p} vs target {t}");
    }
}
