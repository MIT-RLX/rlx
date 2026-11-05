// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Targeted edits (`channel_bias`, `steer`, `rewrites`, `write_kv`) and
//! measurement (`sensitivity`, `search` with paired statistics) on the
//! Bounce landing program — as a task network and as a package.

use rlx_runtime::model::{Tensor as MTensor, rlxp};
use rlx_wsp::baselines::bounce_landing_program;
use rlx_wsp::{Act, EnvKind, WeightSpace, handle_json};
use serde_json::{Value, json};

fn ok(ws: &mut WeightSpace, call: Value) -> Value {
    let r = handle_json(ws, &call.to_string());
    assert!(r.get("error").is_none(), "{call} -> {r}");
    r["ok"].clone()
}

fn err(ws: &mut WeightSpace, call: Value) -> String {
    let r = handle_json(ws, &call.to_string());
    r["error"]
        .as_str()
        .unwrap_or_else(|| panic!("{call} -> {r}"))
        .to_string()
}

fn bounce(budget: usize) -> WeightSpace {
    let mut ws = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, budget);
    bounce_landing_program(&mut ws).unwrap();
    ws
}

fn obs(ws: &WeightSpace) -> Vec<f32> {
    let spec = ws.spec();
    let mut v = vec![0f32; spec.obs_dim];
    for l in ["ball>(x=8,y=0)", "paddle(x=5)"] {
        v[spec.obs_labels.iter().position(|x| x == l).unwrap()] = 1.0;
    }
    v
}

fn logits(ws: &mut WeightSpace, o: &[f32]) -> Vec<f64> {
    let p = ok(ws, json!({"tool": "probe", "observations": [o]}));
    p[0]["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// `values` of node `node` on `o` (one trace).
fn values(ws: &mut WeightSpace, o: &[f32], node: &str) -> Vec<f64> {
    let t = ok(
        ws,
        json!({"tool": "trace", "observation": o, "nodes": [node], "max_values": 64}),
    );
    let e = t
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["node"] == node)
        .unwrap();
    e["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn channel_bias_folds_into_a_bias_and_steer_records_a_rewrite() {
    let mut ws = bounce(20);
    let o = obs(&ws);
    let before = logits(&mut ws, &o);
    // `logits` is l2 matmul + l2.b: the bias absorbs it.
    let r = ok(
        &mut ws,
        json!({"tool": "channel_bias", "node": "logits", "channels": ["right"], "value": -5.0}),
    );
    assert_eq!(r["folded_into"], "l2.b");
    let after = logits(&mut ws, &o);
    assert!(
        (after[2] - (before[2] - 5.0)).abs() < 1e-4,
        "{before:?} {after:?}"
    );
    assert!((after[0] - before[0]).abs() < 1e-6);

    // l0.h is an activation: steering it is a recorded rewrite.
    ok(&mut ws, json!({"tool": "checkpoint", "tag": "pre"}));
    let h_before = values(&mut ws, &o, "l1.h");
    let mut dir = vec![0f32; 32];
    dir[0] = 1.0;
    let r = ok(
        &mut ws,
        json!({"tool": "steer", "node": "l0.h", "vector": dir, "scale": -2.0}),
    );
    assert_eq!(r["rewrite"], 0);
    let h_after = values(&mut ws, &o, "l1.h");
    assert_ne!(h_before, h_after, "the steer reaches layer 1");
    let l = ok(&mut ws, json!({"tool": "rewrites"}));
    assert_eq!(l["rewrites"].as_array().unwrap().len(), 1);
    // Checkpoints keep the rewrites in force; revert restores them.
    ok(&mut ws, json!({"tool": "revert", "tag": "pre"}));
    assert!(ws.state.rewrites.is_empty());
    assert_eq!(values(&mut ws, &o, "l1.h"), h_before);
    ok(
        &mut ws,
        json!({"tool": "steer", "node": "l0.h", "vector": dir, "scale": -2.0}),
    );
    ok(&mut ws, json!({"tool": "rewrites", "remove": 0}));
    assert_eq!(values(&mut ws, &o, "l1.h"), h_before);
    // A vector of the wrong width is refused.
    assert!(
        err(
            &mut ws,
            json!({"tool": "steer", "node": "l0.h", "vector": [1.0]})
        )
        .contains("32")
    );
}

#[test]
fn write_kv_maps_the_key_to_the_target_exactly() {
    let mut ws = bounce(10);
    let o = obs(&ws);
    // The MatMul reading l1.w, from list_nodes `params`.
    let nodes = ok(&mut ws, json!({"tool": "list_nodes", "limit": 200}));
    let mm = nodes["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["params"] == json!(["l1.w"]))
        .unwrap()["id"]
        .as_u64()
        .unwrap();
    let node = format!("%{mm}");
    let before = values(&mut ws, &o, &node);
    let delta: Vec<f32> = (0..32).map(|i| if i == 3 { 0.75 } else { 0.0 }).collect();
    let r = ok(
        &mut ws,
        json!({"tool": "write_kv", "param": "l1.w", "keys": {"observations": [o]}, "delta": delta}),
    );
    assert_eq!(r["key_node"], "l0.h");
    let after = values(&mut ws, &o, &node);
    for (i, (a, b)) in before.iter().zip(&after).enumerate() {
        let want = a + f64::from(delta[i]);
        assert!(
            (b - want).abs() < 1e-4,
            "channel {i}: {a} -> {b}, want {want}"
        );
    }
    // A target set instead: the key's output becomes another input's.
    let mut o2 = obs(&ws);
    o2.iter_mut().for_each(|x| *x = 0.0);
    let spec = ws.spec().clone();
    for l in ["ball<(x=2,y=0)", "paddle(x=5)"] {
        o2[spec.obs_labels.iter().position(|x| x == l).unwrap()] = 1.0;
    }
    let want = values(&mut ws, &o2, &node);
    ok(
        &mut ws,
        json!({"tool": "write_kv", "param": "l1.w", "keys": {"observations": [o]}, "target": {"observations": [o2]}}),
    );
    let got = values(&mut ws, &o, &node);
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-4, "{got:?} vs {want:?}");
    }
}

#[test]
fn sensitivity_and_search_measure_with_paired_items() {
    let mut ws = bounce(40);
    let live = ws.state.params.clone();
    let s = ok(
        &mut ws,
        json!({"tool": "sensitivity", "tensors": ["l2.w", "l0.b"], "n": 8, "eps": 0.5}),
    );
    assert_eq!(ws.state.used, 3);
    assert_eq!(ws.state.params, live, "probes are undone");
    assert_eq!(s["tensors"].as_array().unwrap().len(), 2);
    assert!(err(&mut ws, json!({"tool": "qa_loglik"})).contains("question-answering"));

    // Break the policy (flip the action scores), then let search find the fix.
    ok(
        &mut ws,
        json!({"tool": "scale", "name": "l2.w", "factor": -1.0}),
    );
    let used = ws.state.used;
    let r = ok(
        &mut ws,
        json!({"tool": "search", "n": 8, "rounds": 2, "apply_best": true, "candidates": [
            {"name": "flip back", "calls": [{"tool": "scale", "name": "l2.w", "factor": -1.0}]},
            {"name": "halve", "calls": [{"tool": "scale", "name": "l2.w", "factor": 0.5}]},
            {"name": "not an edit", "calls": [{"tool": "probe", "observations": []}]},
        ]}),
    );
    assert_eq!(r["best"], "flip back", "{r}");
    assert!(
        r["candidates"][3]["error"]
            .as_str()
            .unwrap()
            .contains("not an edit tool")
    );
    assert_eq!(r["confirm_on_fresh_items"]["paired"]["verdict"], "b better");
    assert_eq!(r["applied"], "flip back");
    assert_eq!(ws.state.params["l2.w"], live["l2.w"]);
    assert!(ws.state.used - used <= rlx_wsp::measure::search_cost(3, 2, true));
    assert_eq!(ws.evaluate(32, 3).mean_return, 22.0);
}

#[test]
fn package_edits_reach_the_model_through_taps() {
    let dir = std::env::temp_dir().join(format!("rlx_wsp_b5_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = bounce(0);
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
    let mut ws = WeightSpace::from_package(&path.display().to_string(), 10, None, None).unwrap();
    let o = obs(&src);
    let item = json!({"obs": {"shape": [1, o.len()], "values": o}});
    let trace = |ws: &mut WeightSpace, node: &str| -> Vec<f64> {
        let t = ok(
            ws,
            json!({"tool": "trace", "inputs": {"obs": o}, "nodes": [node], "max_values": 64}),
        );
        t[0][0]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    // %id addressing reads the right graph.
    let nodes = ok(&mut ws, json!({"tool": "list_nodes", "limit": 200}));
    let mm = nodes["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["params"] == json!(["l1.w"]))
        .unwrap()["id"]
        .as_u64()
        .unwrap();
    let node = format!("%{mm}");
    // `contains` matches the weights a node reads, and scripts see the
    // package's tensors.
    let by_param = ok(&mut ws, json!({"tool": "list_nodes", "contains": "l1.w"}));
    assert!(
        by_param["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n["id"] == mm),
        "{by_param}"
    );
    let spec = ok(
        &mut ws,
        json!({"tool": "script", "code": "return SPEC.params['l1.w'];"}),
    );
    assert!(spec["value"]["rows"].as_u64().unwrap() > 0, "{spec}");
    // A read of a node no graph has costs nothing.
    let used = ws.state.used;
    let r = rlx_wsp::tools::handle_json(
        &mut ws,
        &json!({"tool": "trace", "inputs": {"obs": o}, "nodes": ["nowhere"]}).to_string(),
    );
    assert!(
        r["error"]
            .as_str()
            .unwrap()
            .contains("no graph run had all"),
        "{r}"
    );
    assert_eq!(ws.state.used, used);
    let before = trace(&mut ws, &node);
    let h1 = trace(&mut ws, "l1.h");
    let mut d = vec![0f32; 32];
    d[1] = 0.5;
    ok(
        &mut ws,
        json!({"tool": "write_kv", "param": "l1.w", "keys": {"tensors": [item]}, "delta": d}),
    );
    assert!(ws.state.lowrank.contains_key("l1.w"), "kept low-rank");
    let after = trace(&mut ws, &node);
    assert!(
        (after[1] - before[1] - 0.5).abs() < 1e-4,
        "{before:?} {after:?}"
    );
    // A steer on a package is replayed at every open.
    let mut v = vec![0f32; 32];
    v[0] = 3.0;
    ok(
        &mut ws,
        json!({"tool": "steer", "node": "l0.h", "vector": v}),
    );
    assert_ne!(trace(&mut ws, "l1.h"), h1);
    // A checkpoint saved before rewrites were recorded had none: it is not
    // the steered version (compare against it would compare a model with
    // itself), loaded or not.
    let mut st = ws.state.clone();
    st.checkpoints.insert("old".into(), st.params.clone());
    st.checkpoint_rewrites.remove("old");
    assert!(!st.rewrites.is_empty());
    let old = WeightSpace::from_state(st.clone());
    assert_eq!(old.state.checkpoint_rewrites["old"], vec![]);
    ws.state = st;
    assert!(ws.variant("old").unwrap().rewrites.is_empty());
    // The server kept a trail of the calls: the steer is live at the end,
    // and the chart draws it.
    assert!(ws.state.trail.iter().any(|s| s.tool == "steer" && s.ok));
    let last = ws.state.trail.last().unwrap();
    assert!(
        last.live
            .as_ref()
            .unwrap()
            .iter()
            .any(|l| l.kind == "activation")
    );
    assert!(ws.state.trail.iter().any(|s| s.tool == "write_kv" && s.ok));
    let svg = rlx_wsp::explore::chart_svg(&ws.state.trail, None, "package", "trail");
    assert!(svg.contains("steer l0.h"), "a lane for the steer");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_edit_cap_refuses_large_changes_to_package_tensors() {
    let dir = std::env::temp_dir().join(format!("rlx_wsp_cap_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = bounce(0);
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
    let mut ws = WeightSpace::from_package(&path.display().to_string(), 0, None, None).unwrap();
    ws.state.max_edit = Some(0.1);
    let before = ws.param("l1.w").unwrap().clone();
    assert!(
        err(
            &mut ws,
            json!({"tool": "scale", "name": "l1.w", "factor": 2.0})
        )
        .contains("cap 0.1")
    );
    assert_eq!(
        ws.param("l1.w").unwrap(),
        &before,
        "a refused edit leaves the tensor"
    );
    ok(
        &mut ws,
        json!({"tool": "scale", "name": "l1.w", "factor": 1.05}),
    );
    let size = rlx_wsp::checks::edit_size(&ws);
    assert!(
        (size["relative_change_total"].as_f64().unwrap() - 0.05).abs() < 1e-4,
        "{size}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_run_branches_off_at_a_checkpoint_with_its_history() {
    let mut ws = bounce(10);
    let o = obs(&ws);
    ok(&mut ws, json!({"tool": "probe", "observations": [o]}));
    let mut dir = vec![0f32; 32];
    dir[0] = 1.0;
    ok(
        &mut ws,
        json!({"tool": "steer", "node": "l0.h", "vector": dir, "scale": -2.0}),
    );
    ok(&mut ws, json!({"tool": "checkpoint", "tag": "good"}));
    ok(
        &mut ws,
        json!({"tool": "fill", "name": "l2.w", "value": 0.0}),
    );
    let mut b = ws.branch("parent/state.json", "good", 5).unwrap();
    // The checkpoint's weights and rewrites, a fresh budget, the history kept.
    assert_eq!(b.state.params, ws.state.checkpoints["good"]);
    assert_eq!(b.state.rewrites.len(), 1);
    assert_eq!((b.state.used, b.state.budget), (0, 5));
    assert!(b.state.checkpoints.contains_key("branch_start"));
    let d = ok(&mut b, json!({"tool": "describe"}));
    assert_eq!(d["branched_from"]["version"], "good");
    let hist: Vec<&str> = d["branched_from"]["history"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h.as_str().unwrap())
        .collect();
    assert!(
        hist.iter().any(|h| h.starts_with("forward 1/10")),
        "{hist:?}"
    );
    assert!(hist.iter().any(|h| h.starts_with("edit l2.w")));
    // The branch runs on its own budget; the parent is untouched.
    ok(&mut b, json!({"tool": "probe", "observations": [o]}));
    assert_eq!((b.state.used, ws.state.used), (1, 1));
    // A branch of a branch keeps both links.
    let c = b.branch("b/state.json", "current", 3).unwrap();
    assert_eq!(c.state.lineage.len(), 2);
    assert!(ws.branch("x", "nope", 1).is_err());
}

#[test]
fn denied_tools_are_hidden_and_refused_everywhere() {
    let mut ws = bounce(10);
    ws.state.denied_tools = vec!["scale".into()];
    let before = ws.state.params.clone();
    assert!(
        err(
            &mut ws,
            json!({"tool": "scale", "name": "l2.w", "factor": 2.0})
        )
        .contains("denied")
    );
    // From a script.
    let r = ok(
        &mut ws,
        json!({"tool": "script", "code": "return {calls: [{tool: 'scale', name: 'l2.w', factor: 2}]};"}),
    );
    assert!(
        r["applied"][0]["error"]
            .as_str()
            .unwrap()
            .contains("denied")
    );
    // In a search candidate.
    let r = ok(
        &mut ws,
        json!({"tool": "search", "n": 2, "rounds": 1, "confirm": false, "candidates": [
            {"name": "s", "calls": [{"tool": "scale", "name": "l2.w", "factor": 2.0}]}]}),
    );
    assert!(
        r["candidates"][1]["error"]
            .as_str()
            .unwrap()
            .contains("denied")
    );
    assert_eq!(ws.state.params, before);
    // Hidden from the MCP tool list.
    let list = rlx_wsp::mcp::handle_message(
        &mut ws,
        &json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}),
    )
    .unwrap();
    let names: Vec<&str> = list["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(!names.contains(&"scale") && names.contains(&"set"));
}

#[test]
fn background_jobs_measure_the_submitted_weights() {
    // SAFETY: set before any job starts; read by `jobs::submit`.
    unsafe { std::env::set_var("RLX_WSP_EXE", env!("CARGO_BIN_EXE_rlx-wsp")) };
    let mut ws = bounce(10);
    let o = obs(&ws);
    let want = values(&mut ws, &o, "l0.h");
    let j = ok(
        &mut ws,
        json!({"tool": "trace", "observation": o, "nodes": ["l0.h"], "max_values": 64, "async": true}),
    );
    assert_eq!(j["job"], 1);
    assert_eq!(ws.state.used, 2, "charged at submit");
    // Edits after submitting do not reach the job.
    ok(
        &mut ws,
        json!({"tool": "fill", "name": "l0.w", "value": 0.0}),
    );
    assert!(
        err(
            &mut ws,
            json!({"tool": "trace", "observation": o, "nodes": ["l0.h"], "async": true})
        )
        .contains("already running")
    );
    assert!(
        err(
            &mut ws,
            json!({"tool": "fill", "name": "l0.w", "value": 0.0, "async": true})
        )
        .contains("cannot run in the background")
    );
    let r = ok(&mut ws, json!({"tool": "jobs", "collect": 1, "wait": true}));
    let got: Vec<f64> = r["collected"]["result"]["ok"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["node"] == "l0.h")
        .unwrap()["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    assert_eq!(got, want);
    assert_eq!(r["jobs"][0]["status"], "collected");
    // A contrast run in the background brings its vector back.
    let a: Vec<Vec<f32>> = vec![o.clone()];
    let mut o2 = o.clone();
    o2.iter_mut().for_each(|x| *x = 0.0);
    ok(
        &mut ws,
        json!({"tool": "contrast", "node": "l0.h", "a": {"observations": a}, "b": {"observations": [o2]}, "name": "d", "async": true}),
    );
    ok(&mut ws, json!({"tool": "jobs", "collect": 2, "wait": true}));
    assert!(ws.state.vectors.contains_key("d"));
    assert_eq!(ws.state.used, 2 + 2);
}

#[test]
fn precision_average_and_saved_scripts() {
    let dir = std::env::temp_dir().join(format!("rlx_wsp_prec_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = bounce(0);
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
    let mut ws = WeightSpace::from_package(&path.display().to_string(), 0, None, None).unwrap();
    // A change below bf16's resolution is lost; a large one survives.
    ok(
        &mut ws,
        json!({"tool": "scale", "name": "l1.w", "factor": 1.0001}),
    );
    ok(
        &mut ws,
        json!({"tool": "scale", "name": "l2.w", "factor": 1.5}),
    );
    let p = ok(&mut ws, json!({"tool": "precision", "dtype": "bf16"}));
    let share = |t: &str| {
        p["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["tensor"] == t)
            .unwrap()["lost_share"]
            .as_f64()
            .unwrap()
    };
    assert!(share("l1.w") > 0.9, "{p}");
    assert!(share("l2.w") < 0.05, "{p}");
    // Halfway between the unedited model and a checkpoint.
    ok(&mut ws, json!({"tool": "checkpoint", "tag": "e"}));
    ok(
        &mut ws,
        json!({"tool": "average", "tags": ["unedited", "e"]}),
    );
    let base = src.param("l2.w").unwrap();
    let mixed = ws.param("l2.w").unwrap();
    for (m, b) in mixed.data.iter().zip(&base.data) {
        assert!((m - 1.25 * b).abs() < 1e-6);
    }
    // A saved script runs again by name.
    ok(
        &mut ws,
        json!({"tool": "script", "code": "return 6 * 7;", "save_as": "answer"}),
    );
    let r = ok(&mut ws, json!({"tool": "script", "saved": "answer"}));
    assert_eq!(r["value"], 42);
    assert!(ws.state.tool_seconds.contains_key("script"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_stopped_run_finishes_from_its_stages() {
    use rlx_wsp::agent::{AgentCli, finish};
    let dir = std::env::temp_dir().join(format!("rlx_wsp_finish_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ws = bounce(0);
    std::fs::write(dir.join("state.json"), ws.to_json().unwrap()).unwrap();
    let events = [
        json!({"type": "assistant", "message": {"model": "m", "content": [
            {"type": "text", "text": "I left the weights as they were."}]}}),
        json!({"type": "result", "num_turns": 3, "total_cost_usd": 0.5, "duration_ms": 1000,
               "result": "I left the weights as they were."}),
    ];
    let transcript: Vec<String> = events.iter().map(Value::to_string).collect();
    std::fs::write(dir.join("transcript.jsonl"), transcript.join("\n")).unwrap();
    // The agent's account comes back from the transcript.
    let r = finish(&dir, AgentCli::Claude, 8).unwrap();
    assert_eq!(r["agent"]["num_turns"], 3);
    assert!(dir.join("report.json").exists());
    let first = r["held_out_final"]["score"].as_f64().unwrap();
    // A stage already done is not run again…
    let path = dir.join("scoring.json");
    let mut s: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    s["stages"]["final"]["score"] = json!(123.0);
    std::fs::write(&path, s.to_string()).unwrap();
    std::fs::remove_file(dir.join("report.json")).unwrap();
    let r = finish(&dir, AgentCli::Claude, 8).unwrap();
    assert_eq!(r["held_out_final"]["score"], 123.0);
    // …and other weights do not inherit it.
    let mut changed = bounce(0);
    changed.state.used = 1;
    std::fs::write(dir.join("state.json"), changed.to_json().unwrap()).unwrap();
    std::fs::remove_file(dir.join("report.json")).unwrap();
    let r = finish(&dir, AgentCli::Claude, 8).unwrap();
    assert_eq!(r["held_out_final"]["score"].as_f64().unwrap(), first);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_detached_command_runs_on_with_its_output_in_a_log() {
    let log = std::env::temp_dir().join(format!("rlx_wsp_detach_{}.log", std::process::id()));
    let _ = std::fs::remove_file(&log);
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_rlx-wsp"))
        .args(["tools", "--detach", "--log", log.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("detached: pid "));
    let t0 = std::time::Instant::now();
    loop {
        let text = std::fs::read_to_string(&log).unwrap_or_default();
        if text.contains("qa_eval") && text.trim_end().ends_with(']') {
            break;
        }
        assert!(t0.elapsed().as_secs() < 30, "the log holds: {text}");
        std::thread::yield_now();
    }
    let _ = std::fs::remove_file(&log);
}
