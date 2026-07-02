// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! End-to-end training loop: value+grad + optimizer step. Run:
//! `cargo test -p rlx-tensor --features optim`.
#![cfg(feature = "optim")]

use rlx_tensor::{Func, Sgd, shape};

#[test]
fn value_and_grad_emits_loss_and_grads() {
    // loss(w) = sum(w*w); value_and_grad -> [loss, 2w]
    let f = Func::new("sqw", |s| {
        let w = s.param("w", shape![3]);
        (&w * &w).sum([0], false)
    })
    .with_param("w", vec![1.0, 2.0, 3.0]);
    let out = f.value_and_grad(&["w"]).run(&[]);
    assert_eq!(out[0], vec![14.0]); // loss = 1+4+9
    assert_eq!(out[1], vec![2.0, 4.0, 6.0]); // grad = 2w
}

#[test]
fn sgd_minimizes_quadratic() {
    // Minimize loss(w) = sum(w*w); optimum at w = 0.
    let mut model = Func::new("sqw", |s| {
        let w = s.param("w", shape![3]);
        (&w * &w).sum([0], false)
    })
    .with_param("w", vec![3.0, -4.0, 5.0]);

    let mut opt = Sgd::new(0.1);
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for i in 0..100 {
        let (next, loss) = model.train_step(&mut opt, &["w"], &[]);
        model = next;
        if i == 0 {
            first = loss[0];
        }
        last = loss[0];
    }
    assert!(first > 40.0, "initial loss ~50, got {first}");
    assert!(last < 1e-3, "loss should approach 0, got {last}");
}

#[test]
fn lr_schedule_values() {
    use rlx_tensor::LrSchedule;
    // Constant
    assert_eq!(LrSchedule::Constant(0.1).lr_at(50), 0.1);
    // Step decay: 1.0 * 0.5^(step/10)
    let s = LrSchedule::Step {
        base: 1.0,
        step_size: 10,
        gamma: 0.5,
    };
    assert_eq!(s.lr_at(0), 1.0);
    assert_eq!(s.lr_at(9), 1.0);
    assert_eq!(s.lr_at(10), 0.5);
    assert_eq!(s.lr_at(20), 0.25);
    // Cosine: base at 0, min at/after total, midpoint = (base+min)/2
    let c = LrSchedule::Cosine {
        base: 1.0,
        min: 0.0,
        total: 100,
    };
    assert!((c.lr_at(0) - 1.0).abs() < 1e-6);
    assert!((c.lr_at(50) - 0.5).abs() < 1e-5);
    assert!(c.lr_at(100).abs() < 1e-6);
    assert!(c.lr_at(200).abs() < 1e-6); // holds min after total
    // Warmup: linear 0->base over warmup
    let w = LrSchedule::Warmup {
        base: 0.4,
        warmup: 4,
    };
    assert!((w.lr_at(0) - 0.1).abs() < 1e-6); // (0+1)/4 * 0.4
    assert!((w.lr_at(3) - 0.4).abs() < 1e-6);
    assert!((w.lr_at(99) - 0.4).abs() < 1e-6);
}

#[test]
fn scheduled_training_converges() {
    use rlx_tensor::LrSchedule;
    // Same quadratic as sgd_minimizes_quadratic, but with a warmup+cosine LR.
    let mut model = Func::new("sqw", |s| {
        let w = s.param("w", shape![3]);
        (&w * &w).sum([0], false)
    })
    .with_param("w", vec![3.0, -4.0, 5.0]);

    let mut opt = Sgd::new(0.0); // overwritten by the schedule each step
    let sched = LrSchedule::WarmupCosine {
        base: 0.2,
        min: 0.0,
        warmup: 20,
        total: 300,
    };
    let mut last = f32::NAN;
    for i in 0..300 {
        let (next, loss) = model.train_step_at(&mut opt, &sched, i, &["w"], &[]);
        model = next;
        last = loss[0];
    }
    assert!(
        last < 1e-3,
        "scheduled training should converge, got {last}"
    );
}
