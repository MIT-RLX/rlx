// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Training the `rlx! { … }` DSL end-to-end: build a graph whose loss is one
//! line (`mean(cross_entropy(logits, tgt))`), wrap it in a `Func`, seed the
//! weights with `init_randn`, and drive the whole model with the all-params,
//! device-pinned `train_step_*` helpers — the training-DX surface a from-
//! scratch model relies on.
#![cfg(all(feature = "dsl", feature = "optim"))]

use rlx_tensor::{AdamW, Device, Func, LrSchedule, rlx};

/// A trivially-separable 3-class problem: `x[i]` and its target are both the
/// one-hot of class `i % 3`, so a linear layer can drive the loss to ~0.
fn toy_batch() -> (usize, usize, Vec<f32>, Vec<f32>) {
    let (n, c) = (6usize, 3usize);
    let mut x = vec![0f32; n * c];
    let mut y = vec![0f32; n * c];
    for i in 0..n {
        let cls = i % c;
        x[i * c + cls] = 1.0;
        y[i * c + cls] = 1.0;
    }
    (n, c, x, y)
}

fn softmax_clf() -> Func {
    // logits = x·w + b ; loss = mean over rows of the softmax cross-entropy.
    let g = rlx! {
        graph "softmax_clf";
        input x: [6, 3];
        input y: [6, 3];
        param w: [3, 3];
        param b: [3];
        let logits = x @ w + b;
        let loss = mean(cross_entropy(logits, y));
        out loss;
    };
    Func::from_graph(g)
}

#[test]
fn param_names_and_init_params_cover_the_whole_model() {
    let model = softmax_clf();
    let mut names = model.param_names();
    names.sort();
    assert_eq!(names, vec!["b".to_string(), "w".to_string()]);

    // init_params sees each param's static shape and fills exactly that many.
    let mut seen = std::collections::HashMap::new();
    let model = model.init_params(|name, dims| {
        seen.insert(name.to_string(), dims.to_vec());
        vec![0.0; dims.iter().product()]
    });
    assert_eq!(seen["w"], vec![3, 3]);
    assert_eq!(seen["b"], vec![3]);
    assert_eq!(model.param_binding("w").unwrap().len(), 9);
    assert_eq!(model.param_binding("b").unwrap().len(), 3);
}

#[test]
fn train_step_all_on_drives_loss_down() {
    let (_, _, x, y) = toy_batch();
    let feed: &[(&str, &[f32])] = &[("x", &x), ("y", &y)];

    let mut model = softmax_clf().init_randn(1234, 0.1);
    let loss0 = model.run_on(Device::Cpu, feed)[0][0];

    let mut opt = AdamW::new(0.2);
    let mut last = loss0;
    for _ in 0..300 {
        // Whole model, no hand-maintained `wrt`, pinned to CPU.
        let (next, loss) = model.train_step_all_on(Device::Cpu, &mut opt, feed);
        model = next;
        last = loss[0];
    }

    assert!(loss0.is_finite() && last.is_finite());
    assert!(
        last < loss0 * 0.25,
        "loss did not fall enough: {loss0} → {last}"
    );
    assert!(last < 0.2, "final loss too high: {last}");
}

#[test]
fn train_step_all_at_on_follows_the_schedule() {
    let (_, _, x, y) = toy_batch();
    let feed: &[(&str, &[f32])] = &[("x", &x), ("y", &y)];

    let steps = 200usize;
    let sched = LrSchedule::WarmupCosine {
        base: 0.2,
        min: 0.01,
        warmup: steps / 10,
        total: steps,
    };
    let mut model = softmax_clf().init_randn(7, 0.1);
    let loss0 = model.run_on(Device::Cpu, feed)[0][0];

    let mut opt = AdamW::new(0.2);
    let mut last = loss0;
    for step in 0..steps {
        let (next, loss) = model.train_step_all_at_on(Device::Cpu, &mut opt, &sched, step, feed);
        model = next;
        last = loss[0];
    }
    assert!(
        last < loss0,
        "scheduled training did not improve: {loss0} → {last}"
    );
}
