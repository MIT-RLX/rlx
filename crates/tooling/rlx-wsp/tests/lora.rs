// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! LoRA baselines on the Bounce landing program: forward-only training on
//! a task network, and backprop training on a package whose trained
//! factors are applied to a package space by tensor name.

use rlx_wsp::baselines::bounce_landing_program;
use rlx_wsp::lora::LoraOptions;
use rlx_wsp::{Act, EnvKind, WeightSpace};

fn bounce() -> WeightSpace {
    let mut ws = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, 0);
    bounce_landing_program(&mut ws).unwrap();
    ws
}

#[test]
fn mezo_trains_low_rank_factors_without_spending_budget() {
    // A broken policy (action scores flipped): the metric moves with the
    // factors, so the steps have a slope to follow.
    let mut ws = bounce();
    rlx_wsp::handle_json(&mut ws, r#"{"tool":"scale","name":"l2.w","factor":-1}"#);
    let before = ws.param("l2.w").unwrap().clone();
    let o = LoraOptions {
        targets: vec!["l2.w".into()],
        rank: 2,
        alpha: 2.0,
        steps: 6,
        lr: 0.5,
        batch: 4,
        eps: 0.5,
        eval_every: 2,
        ..Default::default()
    };
    let r = rlx_wsp::lora::train_mezo(&mut ws, &o).unwrap();
    assert_eq!(r["targets"], serde_json::json!(["l2.w"]));
    assert_eq!(r["trained_values"], 2 * 32 + 2 * 3);
    assert_eq!(ws.state.used, 0, "baselines are not budgeted");
    assert_ne!(
        ws.param("l2.w").unwrap(),
        &before,
        "the adapter is installed"
    );
    // Only l2.w changed.
    assert_eq!(ws.state.params["l1.w"], bounce().state.params["l1.w"]);
    assert!(ws.state.log.last().unwrap().starts_with("lora mezo"));
    assert!(
        rlx_wsp::lora::train_mezo(
            &mut ws,
            &LoraOptions {
                targets: vec!["nope*".into()],
                ..o
            }
        )
        .is_err()
    );
}

#[cfg(feature = "training")]
#[test]
fn backprop_lora_trains_on_a_package_and_lands_in_a_space() {
    use rlx_runtime::model::{Tensor as MTensor, rlxp};
    use serde_json::json;
    let dir = std::env::temp_dir().join(format!("rlx_wsp_lora_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = bounce();
    let g = src.state.mlp.build_graph(1);
    let weights: Vec<MTensor> = src
        .state
        .params
        .iter()
        .map(|(n, t)| {
            let shape = if n.ends_with(".b") {
                vec![t.cols]
            } else {
                vec![t.rows, t.cols]
            };
            MTensor::f32(n.clone(), shape, &t.data)
        })
        .collect();
    let path = dir.join("bounce.rlxp");
    rlxp::save_package(&path, &g, &weights, &json!({ "name": "bounce" })).unwrap();
    let path = path.display().to_string();
    let mut ws = WeightSpace::from_package(&path, 4, None, None).unwrap();

    let spec = src.spec().clone();
    let mut obs = vec![0f32; spec.obs_dim];
    for l in ["ball>(x=8,y=0)", "paddle(x=5)"] {
        obs[spec.obs_labels.iter().position(|x| x == l).unwrap()] = 1.0;
    }
    let target = [3.0f32, -3.0, 0.0];
    let options = json!({
        "function": g.name,
        "loss": { "mse": { "output": "logits", "target": "target" } },
        "train": { "lora": { "targets": ["l1.w"], "rank": 2, "alpha": 2 } },
        "optimizer": { "lr": 0.05 },
    });
    let (o2, t2) = (obs.clone(), target);
    let r = rlx_wsp::lora::train_backprop_with(
        &mut ws,
        &path,
        &options,
        &LoraOptions {
            steps: 40,
            eval_every: 10,
            ..Default::default()
        },
        move |_, _, _| {
            Ok(vec![
                MTensor::f32("obs", vec![1, o2.len()], &o2),
                MTensor::f32("target", vec![1, 3], &t2),
            ])
        },
    )
    .unwrap();
    let (first, last) = (
        r["first_loss"].as_f64().unwrap(),
        r["last_loss"].as_f64().unwrap(),
    );
    assert!(last < 0.6 * first, "{first} -> {last}");
    assert_eq!(r["adapted"], json!(["l1.w"]));
    assert_eq!(ws.state.lowrank["l1.w"].len(), 2, "kept as rank-2 factors");

    // The space's model (the package plus the installed factors) computes
    // what training reached.
    let t = rlx_wsp::handle_json(
        &mut ws,
        &json!({"tool": "trace", "inputs": {"obs": obs}, "nodes": ["logits"]}).to_string(),
    );
    let v: Vec<f64> = t["ok"][0][0]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let mse: f64 = v
        .iter()
        .zip(target)
        .map(|(a, b)| (a - b as f64).powi(2))
        .sum::<f64>()
        / 3.0;
    assert!(
        (mse - last).abs() < 0.1 * first,
        "space mse {mse} vs the trainer's last loss {last}"
    );
    // Equal compute: a token cap stops training early.
    let mut ws2 = WeightSpace::from_package(&path, 0, None, None).unwrap();
    let (o3, t3) = (obs.clone(), target);
    let r = rlx_wsp::lora::train_backprop_with(
        &mut ws2,
        &path,
        &options,
        &LoraOptions {
            steps: 40,
            eval_every: 10,
            max_tokens: Some(3 * 5),
            ..Default::default()
        },
        move |_, _, _| {
            Ok(vec![
                MTensor::f32("obs", vec![1, o3.len()], &o3),
                MTensor::f32("target", vec![1, 3], &t3),
            ])
        },
    )
    .unwrap();
    assert_eq!(r["steps"], 5, "{r}");
    assert!(r["stopped"].as_str().unwrap().contains("token budget"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(feature = "training")]
#[test]
fn an_agent_steers_a_training_run() {
    use rlx_runtime::model::{Tensor as MTensor, rlxp};
    use serde_json::json;
    let dir = std::env::temp_dir().join(format!("rlx_wsp_opal_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = bounce();
    let g = src.state.mlp.build_graph(1);
    let weights: Vec<MTensor> = src
        .state
        .params
        .iter()
        .map(|(n, t)| {
            let shape = if n.ends_with(".b") {
                vec![t.cols]
            } else {
                vec![t.rows, t.cols]
            };
            MTensor::f32(n.clone(), shape, &t.data)
        })
        .collect();
    let path = dir.join("bounce.rlxp");
    rlxp::save_package(&path, &g, &weights, &json!({ "name": "bounce" })).unwrap();
    let path = path.display().to_string();
    let mut ws = WeightSpace::from_package(&path, 4, None, None).unwrap();
    let spec = src.spec().clone();
    let mut obs = vec![0f32; spec.obs_dim];
    for l in ["ball>(x=8,y=0)", "paddle(x=5)"] {
        obs[spec.obs_labels.iter().position(|x| x == l).unwrap()] = 1.0;
    }
    let p = rlx_runtime::model::trainer::prepare(
        &path,
        &json!({
            "function": g.name,
            "loss": { "mse": { "output": "logits", "target": "target" } },
            "train": { "lora": { "targets": ["l1.w", "l2.w"], "rank": 2, "alpha": 2 } },
        }),
    )
    .unwrap();
    let batch = vec![
        ("obs".to_string(), obs.clone()),
        ("target".to_string(), vec![3.0, -3.0, 0.0]),
    ];
    let cfg = rlx_wsp::opal::OpalConfig {
        train_package: path.clone(),
        targets: vec![],
        rank: 2,
        alpha: 2.0,
        lr: 0.02,
        max_steps: 80,
        steps_done: 0,
        mode: "solution".into(),
        updater: None,
    };
    rlx_wsp::opal::start(&mut ws, p, vec![batch], &cfg).unwrap();
    let call = |ws: &mut WeightSpace, c: serde_json::Value| {
        let r = rlx_wsp::handle_json(ws, &c.to_string());
        assert!(r.get("error").is_none(), "{c} -> {r}");
        r["ok"].clone()
    };
    let r = call(&mut ws, json!({"tool": "train", "steps": 20}));
    let (first, last) = (
        r["first_loss"].as_f64().unwrap(),
        r["last_loss"].as_f64().unwrap(),
    );
    assert!(last < first, "{r}");
    assert!(r["last"]["sites"]["l1.w"]["update_norm"].as_f64().unwrap() > 0.0);
    // The updater: freeze l2.w, a higher rate, a gate; then switch to Muon.
    let u = call(
        &mut ws,
        json!({"tool": "set_updater", "code": "return {freeze: ['l2.w'], lr: 0.05, gate: 5};"}),
    );
    assert_eq!(u["installed"], true);
    let r = call(&mut ws, json!({"tool": "train", "steps": 10}));
    assert_eq!(
        r["frozen"].as_array().unwrap().len(),
        2,
        "both factors of l2.w"
    );
    assert_eq!(r["last"]["sites"]["l2.w"]["update_norm"], 0.0);
    call(
        &mut ws,
        json!({"tool": "set_updater", "code": "return {optimizer: 'muon', unfreeze: ['*']};"}),
    );
    let r = call(&mut ws, json!({"tool": "train", "steps": 5}));
    assert_eq!(r["optimizer"], "muon");
    // A gate below every loss holds every batch back.
    call(
        &mut ws,
        json!({"tool": "set_updater", "code": "return {optimizer: 'muon', gate: 0.000001};"}),
    );
    let r = call(&mut ws, json!({"tool": "train", "steps": 5}));
    assert_eq!(r["gated"], 5, "{r}");
    // In the background.
    call(&mut ws, json!({"tool": "set_updater", "code": ""}));
    call(
        &mut ws,
        json!({"tool": "train", "steps": 10, "background": true}),
    );
    let mut s;
    loop {
        s = call(&mut ws, json!({"tool": "train_status"}));
        if s["running_in_background"] == false {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert_eq!(s["steps_done"], 50);
    let a = call(
        &mut ws,
        json!({"tool": "analyze", "code": "return HISTORY.filter(h => h.gated).length;"}),
    );
    assert_eq!(a, 5);
    // The adapter as a version of the space's model.
    call(&mut ws, json!({"tool": "lora_snapshot", "tag": "s1"}));
    assert!(ws.state.checkpoints.contains_key("s1"));
    assert!(ws.state.lowrank.contains_key("l1.w"));
    // The step budget.
    let e = rlx_wsp::handle_json(&mut ws, &json!({"tool": "train", "steps": 100}).to_string());
    assert!(e.get("error").is_none());
    let e = rlx_wsp::handle_json(&mut ws, &json!({"tool": "train", "steps": 1}).to_string());
    assert!(e["error"].as_str().unwrap().contains("budget"));
    let _ = std::fs::remove_dir_all(&dir);
}
