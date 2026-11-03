// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Trainer::step_with`: the caller makes the update. Handing it the same
//! optimizer the trainer would build must reproduce `step` bit for bit — for
//! AdamW, Muon and Muon+AdamW — and the caller sees exactly what that
//! optimizer would: the unfrozen params, clipped gradients, steps from 1.
#![cfg(feature = "training")]

use rlx_bind::runtime::train::{OptItem, Resident, Trainer, TrainerConfig, optimizer};
use rlx_bind::{ApiError, ErrorKind};
use rlx_ir::{DType, Graph, GraphExt, Shape};
use serde_json::{Value, json};

const BATCH: usize = 4;
const DIM: usize = 3;

/// `loss = mean((x·w + b - t)²)`: `w` is a matrix (Muon's group under
/// `muon_adamw`), `b` a `[1, DIM]` row (AdamW's).
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

fn trainer(optimizer: Value, clip_norm: Option<f32>) -> Trainer {
    let w: Vec<f32> = (0..DIM * DIM).map(|i| 0.1 * (i % 7) as f32 - 0.2).collect();
    Trainer::new(
        regression(),
        TrainerConfig {
            wrt: vec!["w".into(), "b".into()],
            init: vec![("w".into(), w), ("b".into(), vec![0.0; DIM])],
            device: "cpu".into(),
            optimizer,
            policy: None,
            clip_norm,
            resident: Resident::Off,
            training: Default::default(),
        },
    )
    .unwrap()
}

fn batch() -> (Vec<f32>, Vec<f32>) {
    let x = (0..BATCH * DIM)
        .map(|i| ((i * 5 % 11) as f32 - 5.0) / 5.0)
        .collect();
    let t = (0..BATCH * DIM)
        .map(|i| ((i * 3 % 7) as f32 - 3.0) / 3.0)
        .collect();
    (x, t)
}

#[test]
fn step_with_the_trainers_optimizer_reproduces_step() {
    let (x, t) = batch();
    let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
    for opts in [
        json!({"kind": "adamw", "lr": 0.05, "weight_decay": 0.01}),
        json!({"kind": "muon", "lr": 0.02}),
        json!({"kind": "muon_adamw", "lr": 0.05, "weight_decay": 0.01}),
    ] {
        let mut plain = trainer(opts.clone(), Some(0.5));
        let mut hooked = trainer(opts.clone(), Some(0.5));
        let mut opt = optimizer(None, &opts).unwrap().optimizer;
        for _ in 0..12 {
            let a = plain.step(&inputs).unwrap();
            let b = hooked
                .step_with(&inputs, |items, _| {
                    opt.step_batch(items);
                    opt.end_iteration();
                    Ok(())
                })
                .unwrap();
            assert_eq!(a.to_bits(), b.to_bits(), "{opts}: losses differ");
        }
        assert_eq!(plain.params(), hooked.params(), "{opts}: params differ");
        assert_eq!(plain.steps(), hooked.steps());
        assert_eq!(plain.grad_norm(), hooked.grad_norm());
    }
}

#[test]
fn step_with_sees_unfrozen_params_clipped_grads_and_steps_from_one() {
    let (x, t) = batch();
    let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
    let clip = 1e-3;
    let mut t = trainer(json!("adamw"), Some(clip));
    t.set_frozen(Some(&["b".to_string()]), true).unwrap();
    let b_before = t.params();
    let mut seen = Vec::new();
    for _ in 0..3 {
        t.step_with(&inputs, |items: &mut [OptItem<'_>], step| {
            let names: Vec<&str> = items.iter().map(|it| it.name).collect();
            let norm = items
                .iter()
                .flat_map(|it| it.grad.iter())
                .map(|g| g * g)
                .sum::<f32>()
                .sqrt();
            assert_eq!(items[0].shape, &[DIM, DIM]);
            // The clip spans every trainable gradient, frozen `b` included,
            // so `w`'s alone is at most the clip.
            assert!(norm <= clip * 1.0001, "unclipped: {norm}");
            seen.push((names.join(","), step.update));
            items[0].param[0] += 1.0;
            Ok(())
        })
        .unwrap();
    }
    assert_eq!(
        seen,
        vec![("w".into(), 1), ("w".into(), 2), ("w".into(), 3)]
    );
    assert_eq!(t.steps(), 3);
    let after = t.params();
    let get = |p: &[(String, Vec<f32>)], n: &str| p.iter().find(|(k, _)| k == n).unwrap().1.clone();
    assert_eq!(get(&after, "b"), get(&b_before, "b"), "frozen b moved");
    assert_eq!(get(&after, "w")[0], get(&b_before, "w")[0] + 3.0);
}

#[test]
fn a_failing_update_is_an_error_and_not_a_step() {
    let (x, t) = batch();
    let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
    let mut t = trainer(json!("adamw"), None);

    let err = t
        .step_with(&inputs, |_, _| Err(ApiError::runtime("rule rejected")))
        .unwrap_err();
    assert!(err.message.contains("rule rejected"), "{}", err.message);

    let err = t
        .step_with(&inputs, |_, _| panic!("bad shape in the rule"))
        .unwrap_err();
    assert!(matches!(err.kind, ErrorKind::Panic), "{:?}", err.kind);
    assert!(
        err.message.contains("bad shape in the rule"),
        "{}",
        err.message
    );

    assert_eq!(t.steps(), 0);
    // The next step is still step 1.
    t.step_with(&inputs, |_, step| {
        assert_eq!(step.update, 1);
        Ok(())
    })
    .unwrap();
}

#[test]
fn muon_adamw_trains_and_keeps_both_groups_state() {
    let (x, t) = batch();
    let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
    let opts = json!({"kind": "muon_adamw", "lr": 0.1});
    let mut tr = trainer(opts.clone(), None);
    let first = tr.evaluate(&inputs).unwrap();
    for _ in 0..200 {
        tr.step(&inputs).unwrap();
    }
    let trained = tr.evaluate(&inputs).unwrap();
    assert!(trained < 0.05 * first, "{first} -> {trained}");

    // Muon's momentum for the matrix, AdamW's moments for the row.
    let mut opt = optimizer(None, &opts).unwrap().optimizer;
    tr.step_with(&inputs, |items, _| {
        opt.step_batch(items);
        Ok(())
    })
    .unwrap();
    let state = opt.state_dict().expect("muon_adamw checkpoints its state");
    let keys: Vec<&str> = state.buffers.iter().map(|(k, _)| k.as_str()).collect();
    assert!(keys.contains(&"muon_m/w"), "{keys:?}");
    assert!(keys.contains(&"m/b") && keys.contains(&"v/b"), "{keys:?}");
    assert!(!keys.contains(&"m/w"), "the matrix went to AdamW: {keys:?}");
}

#[test]
fn muon_adamw_options_are_checked() {
    let err = optimizer(None, &json!({"kind": "muon_adamw", "muon_scale": 0.0}))
        .err()
        .expect("a zero muon_scale divides the decay by zero");
    assert!(matches!(err.kind, ErrorKind::Range), "{:?}", err.kind);
    let err = optimizer(None, &json!({"kind": "muon_adamw", "ns_steps": 3}))
        .err()
        .expect("an unknown option is an error");
    assert!(err.message.contains("ns_steps"), "{}", err.message);
    let spec = optimizer(Some("muon_adamw"), &json!({"lr": 0.1, "muon_scale": 0.5})).unwrap();
    assert_eq!(spec.kind, "muon_adamw");
}

/// `regression()` with the prediction as a second output, named `pred`.
fn regression_with_pred() -> Graph {
    let f32s = |s: &[usize]| Shape::new(s, DType::F32);
    let mut g = Graph::new("mse");
    let x = g.input("x", f32s(&[BATCH, DIM]));
    let t = g.input("t", f32s(&[BATCH, DIM]));
    let w = g.param("w", f32s(&[DIM, DIM]));
    let b = g.param("b", f32s(&[1, DIM]));
    let pred = g.mm(x, w);
    let biased = g.add(pred, b);
    g.node_mut(biased).name = Some("pred".into());
    let diff = g.sub(biased, t);
    let sq = g.mul(diff, diff);
    let loss = g.mean(sq, vec![0, 1], false);
    g.set_outputs(vec![loss, biased]);
    g
}

#[test]
fn further_graph_outputs_reach_step_with_by_name_and_change_nothing_else() {
    let (x, t) = batch();
    let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
    let w: Vec<f32> = (0..DIM * DIM).map(|i| 0.1 * (i % 7) as f32 - 0.2).collect();
    let make = |g: Graph| {
        Trainer::new(
            g,
            TrainerConfig {
                wrt: vec!["w".into(), "b".into()],
                init: vec![("w".into(), w.clone()), ("b".into(), vec![0.0; DIM])],
                device: "cpu".into(),
                optimizer: json!({"kind": "adamw", "lr": 0.05}),
                policy: None,
                clip_norm: Some(0.5),
                resident: Resident::Off,
                training: Default::default(),
            },
        )
        .unwrap()
    };
    let (mut plain, mut extra) = (make(regression()), make(regression_with_pred()));
    for _ in 0..10 {
        let a = plain.step(&inputs).unwrap();
        let b = extra.step(&inputs).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_eq!(
        plain.params(),
        extra.params(),
        "gradients read past the extra output"
    );
    let mut seen = None;
    extra
        .step_with(&inputs, |_, info| {
            seen = Some((
                info.outputs[0].0.clone(),
                info.outputs[0].1.len(),
                info.loss,
            ));
            Ok(())
        })
        .unwrap();
    let (name, len, loss) = seen.unwrap();
    assert_eq!((name.as_str(), len), ("pred", BATCH * DIM));
    assert!(loss.is_finite() && loss > 0.0);
}
