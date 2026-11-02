// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `DataParallel`: replicas at batch 4 + 2 on separate worker threads give the
//! loss, gradients and optimizer step of one model at batch 6.
#![cfg(all(feature = "dsl", feature = "optim"))]

use rlx_tensor::{AdamW, DataParallel, Device, Func, LrSchedule, Replica, rlx};

fn init(f: Func) -> Func {
    f.init_params(|name, dims| {
        let n: usize = dims.iter().product();
        let off = if name == "w" { 0.3 } else { -0.1 };
        (0..n).map(|i| (i as f32 * 0.7 + off).sin() * 0.5).collect()
    })
}

fn whole() -> Func {
    init(Func::from_graph(rlx! {
        graph "dp_6";
        input x: [6, 3];
        input y: [6, 3];
        param w: [3, 3];
        param b: [3];
        let loss = mean(cross_entropy(x @ w + b, y));
        out loss;
    }))
}

fn four() -> Func {
    init(Func::from_graph(rlx! {
        graph "dp_4";
        input x: [4, 3];
        input y: [4, 3];
        param w: [3, 3];
        param b: [3];
        let loss = mean(cross_entropy(x @ w + b, y));
        out loss;
    }))
}

fn two() -> Func {
    init(Func::from_graph(rlx! {
        graph "dp_2";
        input x: [2, 3];
        input y: [2, 3];
        param w: [3, 3];
        param b: [3];
        let loss = mean(cross_entropy(x @ w + b, y));
        out loss;
    }))
}

fn data() -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..18).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut y = vec![0f32; 18];
    for r in 0..6 {
        y[r * 3 + r % 3] = 1.0;
    }
    (x, y)
}

fn split(x: &[f32], y: &[f32]) -> Vec<Vec<(String, Vec<f32>)>> {
    vec![
        vec![
            ("x".into(), x[..12].to_vec()),
            ("y".into(), y[..12].to_vec()),
        ],
        vec![
            ("x".into(), x[12..].to_vec()),
            ("y".into(), y[12..].to_vec()),
        ],
    ]
}

#[test]
fn replicas_match_one_model() {
    let (x, y) = data();
    let model = whole();
    let want = model
        .value_and_grad_all()
        .run_on(Device::Cpu, &[("x", &x), ("y", &y)]);
    let dp = DataParallel::new(vec![
        Replica::new(Device::Cpu, four(), 4),
        Replica::new(Device::Cpu, two(), 2),
    ]);
    assert_eq!(dp.total_batch(), 6);
    let (got, secs) = dp.value_and_grad(&model, split(&x, &y));
    assert_eq!(secs.len(), 2);
    assert_eq!(want.len(), got.len());
    for (a, b) in want.iter().flatten().zip(got.iter().flatten()) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }

    // Three optimizer steps: same parameters as single-model training.
    let sched = LrSchedule::Constant(0.05);
    let (mut one, mut many) = (whole(), whole());
    let (mut o1, mut o2) = (AdamW::new(0.05), AdamW::new(0.05));
    for step in 0..3 {
        let (n1, _) = one.train_step_all_at_on_clipped(
            Device::Cpu,
            &mut o1,
            &sched,
            step,
            1.0,
            &[("x", &x), ("y", &y)],
        );
        one = n1;
        let (n2, report) = dp.train_step(&many, &mut o2, &sched, step, 1.0, split(&x, &y));
        assert!(report.loss.is_finite() && report.grad_norm > 0.0);
        many = n2;
    }
    for name in ["w", "b"] {
        let (a, b) = (
            one.param_binding(name).unwrap(),
            many.param_binding(name).unwrap(),
        );
        for (p, q) in a.iter().zip(b) {
            assert!((p - q).abs() < 1e-5, "{name}: {p} vs {q}");
        }
    }

    // Forward-only run returns each replica's output.
    let outs = dp.run(&many, split(&x, &y));
    assert_eq!(outs.len(), 2);
}
