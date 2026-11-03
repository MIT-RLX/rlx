// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scoring protocol, one-reply mode, blocks / insert_layer, project_out,
//! answers / switch / graft, the reading and steering tasks, the RL
//! baseline and measured unit roles.

use rlx_wsp::baselines::bounce_landing_program;
use rlx_wsp::space::Protocol;
use rlx_wsp::{Act, EnvKind, WeightSpace, handle_json};
use serde_json::{Value, json};

fn bounce(budget: usize) -> WeightSpace {
    let mut ws = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, budget);
    bounce_landing_program(&mut ws).unwrap();
    ws
}

fn ok(ws: &mut WeightSpace, call: Value) -> Value {
    let r = handle_json(ws, &call.to_string());
    assert!(r.get("error").is_none(), "{call} -> {r}");
    r["ok"].clone()
}

#[test]
fn stochastic_protocol_changes_scoring_and_greedy_does_not() {
    let mut ws = bounce(0);
    let greedy = ws.evaluate(64, 3).mean_return;
    assert_eq!(greedy, 22.0);
    ws.state.protocol = Protocol::STOCHASTIC;
    let noisy = ws.evaluate(64, 3);
    // The program's logits are within ±1, so sampling costs it catches.
    assert!(noisy.mean_return < 21.0, "{}", noisy.mean_return);
    // Deterministic for a fixed seed.
    assert_eq!(ws.evaluate(64, 3).mean_return, noisy.mean_return);
    let d = ok(&mut ws, json!({"tool": "describe"}));
    assert!(
        d["network"]["action_choice"]
            .as_str()
            .unwrap()
            .contains("sticky")
    );
}

#[test]
fn tool_allowlist_restricts_calls_and_scripts() {
    let mut ws = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, 0);
    ws.state.tools = Some(vec!["describe".into(), "set".into()]);
    let r = handle_json(&mut ws, r#"{"tool":"rollout"}"#);
    assert!(
        r["error"].as_str().unwrap().contains("not available"),
        "{r}"
    );
    let r = handle_json(&mut ws, r#"{"tool":"script","code":"return 1"}"#);
    assert!(r["error"].is_string());
    let listed = rlx_wsp::mcp::handle_message(
        &mut ws,
        &json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
    )
    .unwrap();
    let names: Vec<&str> = listed["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["describe", "set"]);
}

#[test]
fn blocks_rebuild_the_bounce_program() {
    // Layer 0: two thresholds over (landing column − paddle column);
    // layer 1: relays; output: relays into right / left.
    let mut ws = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, 0);
    let spec = ws.spec().clone();
    let (mut inputs, mut weights) = (Vec::new(), Vec::new());
    for (i, l) in spec.obs_labels.iter().enumerate() {
        let parse = |s: &str, k: &str| -> i32 {
            s.split(k)
                .nth(1)
                .unwrap()
                .split([',', ')'])
                .next()
                .unwrap()
                .parse()
                .unwrap()
        };
        if let Some(rest) = l.strip_prefix("ball") {
            let (x, y) = (parse(rest, "x="), parse(rest, "y="));
            if y < 9 {
                let dx = if rest.starts_with('>') { 1 } else { -1 };
                inputs.push(json!(i));
                weights.push(rlx_wsp::Bounce::landing_column(10, 10, x, y, dx) as f32);
            }
        } else {
            inputs.push(json!(i));
            weights.push(-(parse(l, "x=") as f32));
        }
    }
    let neg: Vec<f32> = weights.iter().map(|w| -w).collect();
    ok(
        &mut ws,
        json!({"tool": "inject", "units": ["l0.h0"], "spec": {"block": "threshold", "inputs": inputs, "weights": weights, "theta": 0.5}}),
    );
    ok(
        &mut ws,
        json!({"tool": "inject", "units": ["l0.h1"], "spec": {"block": "threshold", "inputs": inputs, "weights": neg, "theta": 0.5}}),
    );
    ok(
        &mut ws,
        json!({"tool": "inject", "units": ["l1.h0", "l1.h1"], "spec": {"block": "relay", "inputs": ["l0.h0", "l0.h1"]}}),
    );
    let r = ok(
        &mut ws,
        json!({"tool": "inject", "units": ["right", "left"], "spec": {"block": "relay", "inputs": ["l1.h0", "l1.h1"], "gain": 1.0}}),
    );
    assert_eq!(r["output_layer"], true);
    assert_eq!(ws.evaluate(64, 4).mean_return, 22.0);
}

#[test]
fn logic_and_gate_blocks_fire_as_specified() {
    let mut ws = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, 3);
    ok(
        &mut ws,
        json!({"tool": "inject", "units": ["l0.h0"], "spec": {"block": "and", "inputs": ["paddle(x=0)", "ball>(x=0,y=0)"]}}),
    );
    ok(
        &mut ws,
        json!({"tool": "inject", "units": ["l0.h1"], "spec": {"block": "or", "inputs": ["paddle(x=0)", "paddle(x=1)"]}}),
    );
    ok(
        &mut ws,
        json!({"tool": "inject", "units": ["l0.h2"], "spec": {"block": "gate", "signals": ["ball>(x=0,y=0)"], "gate": "paddle(x=0)"}}),
    );
    let p = ok(
        &mut ws,
        json!({"tool": "probe", "observations": [
            {"active": ["paddle(x=0)", "ball>(x=0,y=0)"]},
            {"active": ["paddle(x=1)", "ball>(x=0,y=0)"]},
            {"active": ["paddle(x=5)", "ball<(x=3,y=2)"]},
        ]}),
    );
    let h = |row: usize, u: usize| p[row]["hidden"][0][u].as_f64().unwrap();
    assert!(h(0, 0) > 0.9 && h(1, 0) < -0.9 && h(2, 0) < -0.9, "and");
    assert!(h(0, 1) > 0.9 && h(1, 1) > 0.9 && h(2, 1) < -0.9, "or");
    assert!(h(0, 2) > 0.9 && h(1, 2) < -0.9, "gate");
}

#[test]
fn insert_layer_keeps_the_function() {
    let mut ws = bounce(1);
    let before = ok(
        &mut ws,
        json!({"tool": "probe", "observations": [{"active": ["ball>(x=8,y=0)", "paddle(x=5)"]}]}),
    );
    let r = ok(
        &mut ws,
        json!({"tool": "insert_layer", "after": 0, "width": 40}),
    );
    assert_eq!(r["widths"], json!([210, 32, 40, 32, 3]));
    assert_eq!(ws.state.mlp.widths, vec![210, 32, 40, 32, 3]);
    assert_eq!(ws.evaluate(64, 4).mean_return, 22.0);
    let mut ws2 = WeightSpace::from_state(ws.state.clone());
    ws2.state.used = 0;
    let after = ok(
        &mut ws2,
        json!({"tool": "probe", "observations": [{"active": ["ball>(x=8,y=0)", "paddle(x=5)"]}]}),
    );
    for (a, b) in before[0]["logits"]
        .as_array()
        .unwrap()
        .iter()
        .zip(after[0]["logits"].as_array().unwrap())
    {
        assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-3);
    }
    assert!(handle_json(&mut ws, r#"{"tool":"insert_layer","after":5}"#)["error"].is_string());
}

#[test]
fn project_out_removes_exactly_the_direction() {
    let mut ws = bounce(0);
    let v = [1.0f32, 2.0, -1.0];
    ok(
        &mut ws,
        json!({"tool": "fill", "name": "l2.w", "value": 0.5}),
    );
    ok(
        &mut ws,
        json!({"tool": "project_out", "name": "l2.w", "direction": v, "side": "out", "only": [0, 1]}),
    );
    let t = ws.param("l2.w").unwrap();
    for r in 0..2 {
        let dot: f32 = (0..3).map(|c| t.at(r, c) * v[c]).sum();
        assert!(dot.abs() < 1e-6);
    }
    assert_eq!(t.at(2, 0), 0.5, "rows outside `only` untouched");
}

#[test]
fn audit_pair_switch_graft_and_answer() {
    let mut ws = WeightSpace::for_task(EnvKind::AuditPair, &[], Act::Relu, 2, 1).unwrap();
    assert_eq!(ws.state.active.as_deref(), Some("A"));
    let a0 = ws.param("l0.w").unwrap().data.clone();
    ok(&mut ws, json!({"tool": "switch", "to": "B"}));
    assert_ne!(ws.param("l0.w").unwrap().data, a0);
    ok(
        &mut ws,
        json!({"tool": "graft", "from": "A", "units": ["l0.h3"]}),
    );
    let t = ws.param("l0.w").unwrap();
    assert!((0..784).all(|r| t.at(r, 3) == a0[r * 64 + 3]));
    // Task seed 1 → B is the tampered network.
    ok(&mut ws, json!({"tool": "answer", "value": "a"}));
    assert_eq!(ws.held_out(1).unwrap()["score"], 0.0);
    ok(
        &mut ws,
        json!({"tool": "answer", "value": "B", "reason": "trigger units"}),
    );
    let h = ws.held_out(1).unwrap();
    assert_eq!(h["score"], 100.0, "{h}");
}

#[test]
fn read_tasks_hide_the_answer_and_score_it() {
    let mut ws = WeightSpace::for_task(EnvKind::ReadDigit, &[], Act::Relu, 0, 7).unwrap();
    assert_eq!(ws.state.hidden, Some(7));
    let d = ok(&mut ws, json!({"tool": "describe"}));
    assert!(!d.to_string().contains("\"hidden\""));
    ok(&mut ws, json!({"tool": "answer", "value": 7}));
    assert_eq!(ws.held_out(1).unwrap()["score"], 100.0);
    let mut rr = WeightSpace::for_task(EnvKind::ReadRules, &[], Act::Tanh, 0, 2).unwrap();
    assert_eq!(rr.state.mlp.widths, vec![370, 32, 32, 3]);
    assert!(handle_json(&mut rr, r#"{"tool":"rollout"}"#)["error"].is_string());
    let d = ok(&mut rr, json!({"tool": "describe"}));
    assert_eq!(d["candidates"].as_array().unwrap().len(), 3);
    ok(&mut rr, json!({"tool": "answer", "value": "C"}));
    assert_eq!(rr.held_out(1).unwrap()["score"], 100.0);
}

#[test]
fn steer_task_scores_both_halves() {
    let mut ws = WeightSpace::try_new(EnvKind::MnistSteer, &[], Act::Relu, 0).unwrap();
    let h = ws.held_out(1).unwrap();
    assert!(h["other_accuracy"].as_f64().unwrap() > 0.95);
    assert!(h["steered"].as_f64().unwrap() < 0.05);
}

#[test]
fn rl_baseline_reports_curves() {
    let opts = rlx_wsp::pg::PgOptions {
        iters: 4,
        episodes_per_iter: 2,
        seeds: 1,
        eval_episodes: 32,
        eval_every: 2,
        ..Default::default()
    };
    let r = rlx_wsp::pg::train_baseline(
        EnvKind::Bounce,
        &[32, 32],
        Act::Tanh,
        Protocol::GREEDY,
        opts,
    )
    .unwrap();
    assert_eq!(r["seeds"][0]["curve"].as_array().unwrap().len(), 2);
    assert!(
        rlx_wsp::pg::train_baseline(EnvKind::MnistFix, &[], Act::Relu, Protocol::GREEDY, opts)
            .is_err()
    );
}

#[test]
fn measured_roles_find_the_comparator() {
    let ws = bounce(0);
    let roles = rlx_wsp::roles::measure(&ws, 4).unwrap();
    let (_, f, r) = roles.iter().find(|(u, _, _)| u == "l0.h0").unwrap();
    assert_eq!(*f, "target_minus_paddle");
    assert!(*r > 0.5);
    let g = rlx_wsp::roles::grade(
        "l0.h0 fires when the landing column is to the right of the paddle.",
        &roles,
        &["l0.h0".to_string(), "l0.h1".to_string()],
    );
    assert_eq!(g["described"], 1);
    assert_eq!(g["described_with_measured_role"], 1);
}
