// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Graph tools inside packages (taps), activation reads (`node_stats`,
//! `contrast`, `lens`), span repetition, low-rank overlays and package
//! export — on the Bounce landing program, whose units are known.

use rlx_runtime::model::{Tensor as MTensor, rlxp};
use rlx_wsp::baselines::bounce_landing_program;
use rlx_wsp::{Act, EnvKind, WeightSpace, handle_json};
use serde_json::{Value, json};

fn ok(ws: &mut WeightSpace, call: Value) -> Value {
    let r = handle_json(ws, &call.to_string());
    assert!(r.get("error").is_none(), "{call} -> {r}");
    r["ok"].clone()
}

fn bounce(budget: usize) -> WeightSpace {
    let mut ws = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, budget);
    bounce_landing_program(&mut ws).unwrap();
    ws
}

fn obs(ws: &WeightSpace, active: &[&str]) -> Vec<f32> {
    let spec = ws.spec();
    let mut v = vec![0f32; spec.obs_dim];
    for l in active {
        v[spec.obs_labels.iter().position(|x| x == l).unwrap()] = 1.0;
    }
    v
}

fn package(dir: &std::path::Path) -> (std::path::PathBuf, WeightSpace) {
    let ws = bounce(4);
    let g = ws.state.mlp.build_graph(1);
    let weights: Vec<MTensor> = ws
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
    rlxp::save_package(
        &path,
        &g,
        &weights,
        &json!({ "name": "bounce", "task": "graph" }),
    )
    .unwrap();
    (path, ws)
}

fn tmp(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rlx_wsp_b4_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn package_trace_reads_inside_through_taps() {
    let dir = tmp("trace");
    let (path, mut mlp) = package(&dir);
    let mut ws = WeightSpace::from_package(&path.display().to_string(), 3, None, None).unwrap();
    let nodes = ok(&mut ws, json!({"tool": "list_nodes", "contains": "l0"}));
    assert!(nodes["functions"].as_array().is_some());
    assert!(
        nodes["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n["name"] == "l0.h")
    );
    let o = obs(&mlp, &["ball>(x=8,y=0)", "paddle(x=5)"]);
    let t = ok(
        &mut ws,
        json!({"tool": "trace", "inputs": {"obs": o}, "nodes": ["l0.h"], "max_values": 2}),
    );
    let got = t[0][0]["values"][1].as_f64().unwrap();
    let probe = ok(&mut mlp, json!({"tool": "probe", "observations": [o]}));
    let want = probe[0]["hidden"][0][1].as_f64().unwrap();
    assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    // A recorded repeat is replayed inside the package: its copies exist.
    ok(
        &mut ws,
        json!({"tool": "repeat", "from": "l0.h", "to": "l1.h", "times": 2}),
    );
    let t = ok(
        &mut ws,
        json!({"tool": "trace", "inputs": {"obs": o}, "nodes": ["l1.h.rep1"]}),
    );
    assert_eq!(t[0][0]["channels"], 32);
    // Blocks write existing producing columns, so they reach the overlay.
    ok(
        &mut ws,
        json!({"tool": "inject", "units": ["l1.h[5]"], "spec": {"block": "relay", "inputs": ["l0.h[0]"]}}),
    );
    assert!(ws.state.params.contains_key("l1.w"));
    assert!(
        handle_json(&mut ws, r#"{"tool":"insert_linear","after":"l0.h"}"#)["error"].is_string()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn activation_reads_find_the_comparator() {
    let mut ws = bounce(6);
    let right = |ws: &WeightSpace, b: &str, p: &str| obs(ws, &[b, p]);
    // Ball landing right of the paddle vs left of it.
    let a: Vec<Vec<f32>> = (0..4)
        .map(|x| right(&ws, &format!("ball>(x={},y=7)", 6 + x % 2), "paddle(x=1)"))
        .collect();
    let b: Vec<Vec<f32>> = (0..4)
        .map(|x| right(&ws, &format!("ball<(x={},y=7)", 2 + x % 2), "paddle(x=8)"))
        .collect();
    let s = ok(
        &mut ws,
        json!({"tool": "node_stats", "node": "l0.h", "observations": a.iter().chain(&b).collect::<Vec<_>>(), "top": 2}),
    );
    let top: Vec<&str> = s["most_variable"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["channel"].as_str().unwrap())
        .collect();
    assert!(top.contains(&"l0.h0") && top.contains(&"l0.h1"), "{s}");
    let c = ok(
        &mut ws,
        json!({"tool": "contrast", "node": "l0.h", "a": {"observations": a}, "b": {"observations": b}, "name": "go_right"}),
    );
    assert_eq!(c["separates"], true, "{c}");
    assert!(ws.state.vectors.contains_key("go_right"));
    // The stored direction is usable by project_out (in the next layer's input space).
    ok(
        &mut ws,
        json!({"tool": "project_out", "name": "l1.w", "vector": "go_right", "side": "in"}),
    );
    // lens: a hidden state read through the output weights favours an action.
    let l = ok(
        &mut ws,
        json!({"tool": "lens", "node": "l1.h", "observations": [a[0]], "through": "l2.w", "top": 1}),
    );
    assert!(l["items"][0]["top"][0]["label"].is_string());
}

#[test]
fn repeat_spans_on_task_graphs() {
    let mut ws = bounce(0);
    // Layer 1 relays two units; three copies keep the program working.
    let r = ok(
        &mut ws,
        json!({"tool": "repeat", "from": "l0.h", "to": "l1.h", "times": 3}),
    );
    assert_eq!(r["times"], 3);
    assert!(r["new_params"].as_array().unwrap().is_empty());
    assert_eq!(ws.evaluate(32, 3).mean_return, 22.0);
    // Cloned weights: each copy gets its own params, editable apart.
    let mut ws = bounce(0);
    let r = ok(
        &mut ws,
        json!({"tool": "repeat", "from": "l0.h", "to": "l1.h", "times": 2, "clone_weights": true}),
    );
    assert_eq!(r["new_params"], json!(["l1.w.rep1", "l1.b.rep1"]));
    assert_eq!(ws.param("l1.w.rep1").unwrap(), ws.param("l1.w").unwrap());
    ok(
        &mut ws,
        json!({"tool": "fill", "name": "l1.w.rep1", "value": 0}),
    );
    assert!(
        ws.evaluate(32, 3).mean_return < 0.0,
        "the zeroed copy breaks it"
    );
    assert!(
        handle_json(
            &mut ws,
            r#"{"tool":"repeat","from":"obs","to":"l0.h","times":2}"#
        )["error"]
            .is_string()
    );
}

#[test]
fn low_rank_overlays_store_factors_and_merge_exactly() {
    let dir = tmp("lowrank");
    let (path, _) = package(&dir);
    let mut ws = WeightSpace::from_package(&path.display().to_string(), 0, None, None).unwrap();
    let u: Vec<f32> = (0..32).map(|i| (i as f32 * 0.37).sin()).collect();
    let v = vec![0.5f32, -1.0, 0.25];
    ok(
        &mut ws,
        json!({"tool": "add_rank1", "name": "l2.w", "u": u, "v": v, "scale": 0.5}),
    );
    ok(
        &mut ws,
        json!({"tool": "project_out", "name": "l2.w", "direction": [1.0, 0.0, 1.0], "side": "out"}),
    );
    assert_eq!(ws.state.lowrank["l2.w"].len(), 2);
    let text = ws.to_json().unwrap();
    assert!(!text.contains("\"l2.w\":{\"rows\""), "no dense copy stored");
    let back = WeightSpace::from_state(serde_json::from_str(&text).unwrap());
    let (a, b) = (back.param("l2.w").unwrap(), ws.param("l2.w").unwrap());
    for (x, y) in a.data.iter().zip(&b.data) {
        assert!((x - y).abs() < 1e-5);
    }
    let updates = ws.adapter_updates().unwrap();
    assert_eq!(updates[0].rank, 2);
    // A non-rank-1 edit makes it dense again.
    ok(
        &mut ws,
        json!({"tool": "scale", "name": "l2.w", "factor": 2.0}),
    );
    assert!(!ws.state.lowrank.contains_key("l2.w"));
    // Export of a package space writes the adapter, not the base.
    let out = dir.join("exp");
    let doc = rlx_wsp::export::export(&mut ws, &out, false, 8).unwrap();
    assert_eq!(doc["files"], json!(["adapter.rlxp"]));
    assert!(out.join("adapter.rlxp").is_file());
    let _ = std::fs::remove_dir_all(&dir);
}
