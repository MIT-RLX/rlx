// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `TrainingOptions`: what PyTorch training sets up around the optimizer.
//!
//! * A schedule is the same run as calling `set_lr` with its values by hand,
//!   and `step_with` (the agent) follows it.
//! * Gradient accumulation updates once per `accumulate` micro-batches with
//!   their mean gradient.
//! * Dynamic loss scaling changes nothing while gradients are finite (the
//!   scale is a power of two), skips an update whose gradients overflow and
//!   halves the scale.
#![cfg(feature = "agent")]

use rlx_bind::runtime::agent::{AgentConfig, AgentOptimizer, Keep, RuleHost};
use rlx_bind::runtime::train::{
    EvalTracker, LossScale, LrSchedule, Resident, ScheduleKind, Trainer, TrainerConfig,
    TrainingOptions,
};
use rlx_ir::{DType, Graph, GraphExt, Shape};
use serde_json::{Value, json};

const BATCH: usize = 4;
const DIM: usize = 3;

fn regression() -> Graph {
    let f32s = |s: &[usize]| Shape::new(s, DType::F32);
    let mut g = Graph::new("mse");
    let x = g.input("x", f32s(&[BATCH, DIM]));
    let t = g.input("t", f32s(&[BATCH, DIM]));
    let w = g.param("w", f32s(&[DIM, DIM]));
    let b = g.param("b", f32s(&[1, DIM]));
    let pred = g.mm(x, w);
    let biased = g.add(pred, b);
    let diff = g.sub(biased, t);
    let sq = g.mul(diff, diff);
    let loss = g.mean(sq, vec![0, 1], false);
    g.set_outputs(vec![loss]);
    g
}

fn trainer(optimizer: Value, training: TrainingOptions) -> Trainer {
    let w: Vec<f32> = (0..DIM * DIM).map(|i| 0.1 * (i % 7) as f32 - 0.2).collect();
    Trainer::new(
        regression(),
        TrainerConfig {
            wrt: vec!["w".into(), "b".into()],
            init: vec![("w".into(), w), ("b".into(), vec![0.0; DIM])],
            device: "cpu".into(),
            optimizer,
            policy: None,
            clip_norm: None,
            resident: Resident::Off,
            training,
        },
    )
    .unwrap()
}

fn batch(seed: usize) -> (Vec<f32>, Vec<f32>) {
    let x = (0..BATCH * DIM)
        .map(|i| (((i + seed) * 5 % 11) as f32 - 5.0) / 5.0)
        .collect();
    let t = (0..BATCH * DIM)
        .map(|i| (((i + 2 * seed) * 3 % 7) as f32 - 3.0) / 3.0)
        .collect();
    (x, t)
}

fn cosine() -> LrSchedule {
    LrSchedule::new(ScheduleKind::Cosine {
        total: 40,
        min_factor: 0.1,
    })
    .with_warmup(5, 0.0)
}

#[test]
fn a_schedule_is_the_same_run_as_setting_its_rates_by_hand() {
    let opt = json!({"kind": "adamw", "lr": 0.05});
    let mut scheduled = trainer(
        opt.clone(),
        TrainingOptions {
            schedule: Some(cosine()),
            ..Default::default()
        },
    );
    let mut by_hand = trainer(opt.clone(), TrainingOptions::default());
    for i in 0..40u64 {
        let (x, t) = batch(i as usize);
        let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
        // The configured rate is an f32; the schedule scales it in f64.
        let lr = (0.05f32 as f64 * cosine().factor(i)) as f32;
        by_hand.set_lr(lr);
        assert_eq!(
            scheduled.step(&inputs).unwrap().to_bits(),
            by_hand.step(&inputs).unwrap().to_bits(),
            "{i}"
        );
        assert_eq!(scheduled.lr(), lr);
    }
    assert_eq!(scheduled.params(), by_hand.params());
}

#[test]
fn the_agent_follows_the_trainers_schedule() {
    let opt = json!({"kind": "adamw", "lr": 0.05});
    let training = || TrainingOptions {
        schedule: Some(cosine()),
        ..Default::default()
    };
    let mut plain = trainer(opt.clone(), training());
    let mut wrapped = trainer(opt.clone(), training());
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = AgentConfig::new("regression; judged by the final loss", 40);
    cfg.run_root = dir.path().to_path_buf();
    cfg.decisions = Some(2);
    let mut agent =
        AgentOptimizer::new(cfg, Box::new(RuleHost::new(&opt).unwrap()), Box::new(Keep)).unwrap();
    let mut seen = Vec::new();
    for i in 0..40usize {
        let (x, t) = batch(i);
        let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
        let a = plain.step(&inputs).unwrap();
        let b = agent.train_step(&mut wrapped, &inputs).unwrap();
        assert_eq!(a.to_bits(), b.to_bits(), "{i}");
        seen.push(wrapped.lr());
    }
    assert_eq!(
        plain.params(),
        wrapped.params(),
        "bit for bit, schedule included"
    );
    assert_eq!(seen[0], 0.0, "warmup from zero");
    // step_with hands the scheduled rate on.
    let (x, t) = batch(0);
    let mut lr = None;
    wrapped
        .step_with(&[("x", x.as_slice()), ("t", t.as_slice())], |_, info| {
            lr = info.lr;
            Ok(())
        })
        .unwrap();
    assert_eq!(lr, Some((0.05f32 as f64 * cosine().factor(40)) as f32));
}

#[test]
fn accumulation_updates_once_per_group_with_the_mean_gradient() {
    let sgd = json!({"kind": "sgd", "lr": 0.1});
    let mut acc = trainer(
        sgd.clone(),
        TrainingOptions {
            accumulate: 2,
            ..Default::default()
        },
    );
    let mut probe = trainer(sgd, TrainingOptions::default());
    let start = probe.params();
    let mut grads: Vec<Vec<Vec<f32>>> = Vec::new();
    for seed in [1usize, 2] {
        let (x, t) = batch(seed);
        let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
        acc.step(&inputs).unwrap();
        // The probe only reads the gradient: params stay at the start.
        probe
            .step_with(&inputs, |items, _| {
                grads.push(items.iter().map(|it| it.grad.to_vec()).collect());
                Ok(())
            })
            .unwrap();
        if seed == 1 {
            assert_eq!(acc.steps(), 0, "no update after the first micro-batch");
            assert_eq!(acc.pending_micro_batches(), 1);
            assert_eq!(acc.params(), start);
        }
    }
    assert_eq!(acc.steps(), 1);
    assert_eq!(acc.pending_micro_batches(), 0);
    for (j, (name, p)) in acc.params().into_iter().enumerate() {
        let p0 = &start[j].1;
        for (k, v) in p.iter().enumerate() {
            let mean = 0.5 * grads[0][j][k] + 0.5 * grads[1][j][k];
            assert!((v - (p0[k] - 0.1 * mean)).abs() < 1e-6, "{name}[{k}]");
        }
    }
}

#[test]
fn dynamic_loss_scaling_changes_nothing_until_gradients_overflow() {
    let opt = json!({"kind": "adamw", "lr": 0.05});
    let mut scaled = trainer(
        opt.clone(),
        TrainingOptions {
            loss_scale: Some(LossScale::dynamic()),
            ..Default::default()
        },
    );
    let mut plain = trainer(opt, TrainingOptions::default());
    for i in 0..20usize {
        let (x, t) = batch(i);
        let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
        assert_eq!(
            scaled.step(&inputs).unwrap().to_bits(),
            plain.step(&inputs).unwrap().to_bits()
        );
    }
    assert_eq!(
        scaled.params(),
        plain.params(),
        "a power-of-two scale is exact"
    );
    assert_eq!(scaled.loss_scale(), Some(65536.0));
    // An input that overflows the scaled gradients: the update is skipped.
    let before = scaled.params();
    let (mut x, t) = batch(0);
    x[0] = 3.0e33;
    scaled
        .step(&[("x", x.as_slice()), ("t", t.as_slice())])
        .unwrap();
    assert_eq!(scaled.params(), before, "skipped");
    assert_eq!(scaled.skipped_updates(), 1);
    assert_eq!(scaled.loss_scale(), Some(32768.0), "halved");
    assert_eq!(scaled.steps(), 21, "the update's index moves on");
}

#[test]
fn options_are_checked() {
    let err = |v: Value| TrainingOptions::from_json(&v).unwrap_err().message;
    assert!(err(json!({"schedule": {"kind": "cosine"}})).contains("total_steps"));
    assert!(err(json!({"schedule": {"kind": "triangle"}})).contains("triangle"));
    assert!(err(json!({"schedule": {"kind": "constant", "warmup": 3}})).contains("'warmup'"));
    assert!(err(json!({"loss_scale": -1.0})).contains("not a usable scale"));
    assert!(err(json!({"loss_scale": "static"})).contains("dynamic"));
    let t = TrainingOptions::from_json(
        &json!({"schedule": {"kind": "onecycle", "total_steps": 100},
                                               "accumulate": 4, "loss_scale": "dynamic"}),
    )
    .unwrap();
    assert_eq!(t.accumulate, 4);
    assert!(matches!(
        t.schedule.unwrap().kind,
        ScheduleKind::OneCycle { total: 100, .. }
    ));
}

#[test]
fn evaluations_keep_the_best_and_stop_after_patience() {
    let mut e = EvalTracker::new(Some(2), 0.0);
    assert_eq!(e.record(10, 1.0), (true, false));
    assert_eq!(e.record(20, 0.5), (true, false));
    assert_eq!(e.record(30, 0.6), (false, false));
    assert_eq!(e.record(40, f32::NAN), (false, true));
    assert_eq!(e.best, Some((20, 0.5)));
    assert_eq!(e.summary()["best_step"], 20);
}

/// Resident (fused, on the device) against the host optimizer, for each fused
/// kind; then a resident checkpoint resumed on the host path. Needs a GPU
/// backend in the build (`--features rlx-runtime/metal`); skipped otherwise.
#[test]
fn resident_muon_matches_the_host_and_its_checkpoint_resumes_on_the_host() {
    let device = "metal";
    let make = |opt: Value, resident: bool| {
        let w: Vec<f32> = (0..DIM * DIM).map(|i| 0.1 * (i % 7) as f32 - 0.2).collect();
        Trainer::new(
            regression(),
            TrainerConfig {
                wrt: vec!["w".into(), "b".into()],
                init: vec![("w".into(), w), ("b".into(), vec![0.0; DIM])],
                device: device.into(),
                optimizer: opt,
                policy: None,
                clip_norm: None,
                resident: if resident {
                    Resident::On
                } else {
                    Resident::Off
                },
                training: TrainingOptions::default(),
            },
        )
    };
    if make(json!("adamw"), false).is_err() {
        eprintln!("skipped: no {device} backend in this build");
        return;
    }
    for opt in [
        json!({"kind": "muon", "lr": 0.02, "weight_decay": 0.01}),
        json!({"kind": "muon_adamw", "lr": 0.05, "weight_decay": 0.01}),
    ] {
        let mut fused = make(opt.clone(), true).unwrap();
        assert!(fused.is_resident(), "{opt}");
        let mut host = make(opt.clone(), false).unwrap();
        for i in 0..15usize {
            let (x, t) = batch(i);
            let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
            let (a, b) = (fused.step(&inputs).unwrap(), host.step(&inputs).unwrap());
            assert!(
                (a - b).abs() <= 1e-3 * b.abs().max(1.0),
                "{opt} step {i}: resident {a} vs host {b}"
            );
        }
        // By name: the two paths list their params in different orders.
        let host_params = host.params();
        for (n, a) in fused.params() {
            let (_, b) = host_params
                .iter()
                .find(|(m, _)| *m == n)
                .expect("same params");
            for (x, y) in a.iter().zip(b) {
                assert!(
                    (x - y).abs() <= 1e-3 * y.abs().max(1.0),
                    "{opt} {n}: {x} vs {y}"
                );
            }
        }
        // The resident checkpoint uses the host optimizer's slots.
        let dir = tempfile::tempdir().unwrap();
        let ck = dir.path().join("ck.gguf");
        fused.save(ck.to_str().unwrap()).unwrap();
        let mut resumed = make(opt.clone(), false).unwrap();
        let report = resumed.load(ck.to_str().unwrap()).unwrap();
        assert_eq!(report["optimizer_restored"], true, "{opt}: {report}");
        let (x, t) = batch(99);
        let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
        let (a, b) = (resumed.step(&inputs).unwrap(), host.step(&inputs).unwrap());
        assert!(
            (a - b).abs() <= 1e-3 * b.abs().max(1.0),
            "{opt} after resume: {a} vs {b}"
        );
    }
}
