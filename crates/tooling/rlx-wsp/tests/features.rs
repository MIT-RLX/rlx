// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Graph spaces, trace / compose / compare / ablate / script, the claim
//! check, warm starts and export — each against a network whose behaviour is
//! known (the Bounce landing program).

use rlx_wsp::baselines::bounce_landing_program;
use rlx_wsp::verify::{claim_check, unit_importance};
use rlx_wsp::warmstart::{WarmstartOptions, warmstart};
use rlx_wsp::{Act, EnvKind, WeightSpace, handle_json};
use serde_json::Value;
use std::collections::BTreeMap;

fn bounce(budget: usize) -> WeightSpace {
    let mut ws = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, budget);
    bounce_landing_program(&mut ws).unwrap();
    ws
}

fn ok(ws: &mut WeightSpace, call: &str) -> Value {
    let r = handle_json(ws, call);
    assert!(r.get("error").is_none(), "{call} -> {r}");
    r["ok"].clone()
}

const OBS: &str = r#"{"active":["ball>(x=8,y=0)","paddle(x=5)"]}"#;

#[test]
fn graph_space_runs_any_rlx_graph_like_the_mlp() {
    let mut mlp = bounce(4);
    // The same network as a generic graph: .rlxmodel text + named weights.
    let text = rlx_ir::rlxmodel::to_text(&mlp.state.mlp.build_graph(1)).unwrap();
    let weights: BTreeMap<String, Vec<f32>> = mlp
        .state
        .params
        .iter()
        .map(|(n, t)| (n.clone(), t.data.clone()))
        .collect();
    let mut g = WeightSpace::from_graph(text, weights, 4, Some("check parity".into())).unwrap();
    assert_eq!(g.state.env, EnvKind::Graph);
    let d = ok(&mut g, r#"{"tool":"describe"}"#);
    assert_eq!(d["task"], "check parity");
    assert_eq!(d["inputs"][0]["name"], "obs");

    let probe = ok(
        &mut mlp,
        &format!(r#"{{"tool":"probe","observations":[{OBS}]}}"#),
    );
    let obs = {
        let spec = mlp.spec();
        let mut v = vec![0f32; spec.obs_dim];
        for l in ["ball>(x=8,y=0)", "paddle(x=5)"] {
            v[spec.obs_labels.iter().position(|x| x == l).unwrap()] = 1.0;
        }
        v
    };
    let t = ok(
        &mut g,
        &serde_json::json!({ "tool": "trace", "inputs": { "obs": obs } }).to_string(),
    );
    let logits: Vec<f64> = t[0]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let want: Vec<f64> = probe[0]["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    for (a, b) in logits.iter().zip(&want) {
        assert!((a - b).abs() < 1e-5, "{logits:?} vs {want:?}");
    }
    // Generic labels; edits work; rollout does not apply.
    let s = ok(&mut g, r#"{"tool":"slice","name":"l2.w","rows":[0,2]}"#);
    assert_eq!(s["cols"][0], "c0");
    assert_eq!(s["rows"][1]["label"], "r1");
    ok(&mut g, r#"{"tool":"scale","name":"l2.w","factor":2}"#);
    assert!(handle_json(&mut g, r#"{"tool":"rollout"}"#)["error"].is_string());
    // A weights entry that is not a param is an error, not silently dropped.
    let bad: BTreeMap<String, Vec<f32>> = [("nope".to_string(), vec![0.0])].into();
    let text = rlx_ir::rlxmodel::to_text(&mlp.state.mlp.build_graph(1)).unwrap();
    assert!(WeightSpace::from_graph(text, bad, 1, None).is_err());
}

#[test]
fn trace_returns_named_nodes_and_compose_is_the_product() {
    let mut ws = bounce(2);
    let nodes = ok(&mut ws, r#"{"tool":"list_nodes"}"#);
    let names: Vec<&str> = nodes
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n["name"].as_str())
        .collect();
    for n in ["obs", "l0.z", "l0.h", "l1.z", "l1.h", "logits"] {
        assert!(names.contains(&n), "{names:?}");
    }
    let t = ok(
        &mut ws,
        &format!(r#"{{"tool":"trace","observation":{OBS},"nodes":["l0.z"],"max_values":2}}"#),
    );
    let z = t
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["node"] == "l0.z")
        .unwrap();
    assert_eq!(z["shape"], serde_json::json!([1, 32]));
    // Landing column 1, paddle 5: l0.h1's pre-activation is GAIN·(5-1) - 2 = 14.
    assert!(
        (z["values"][1].as_f64().unwrap() - 14.0).abs() < 1e-4,
        "{z}"
    );
    assert_eq!(ws.state.used, 1);

    // compose: (l1.w · l2.w)[l0.h0, right] = 2·1 = 2 in the landing program.
    let c = ok(
        &mut ws,
        r#"{"tool":"compose","a":"l1.w","b":"l2.w","view":"topk","k":2}"#,
    );
    let top = &c["entries"][0];
    assert_eq!(top["value"], 2.0);
    let bad = handle_json(&mut ws, r#"{"tool":"compose","a":"l0.w","b":"l0.w"}"#);
    assert!(bad["error"].as_str().unwrap().contains("inner sizes"));
}

#[test]
fn compare_and_ablate_are_paired_and_budgeted() {
    let mut ws = bounce(4);
    ok(&mut ws, r#"{"tool":"checkpoint","tag":"good"}"#);
    ok(&mut ws, r#"{"tool":"fill","name":"l2.w","value":0}"#);
    let c = ok(
        &mut ws,
        r#"{"tool":"compare","a":"good","b":"current","episodes":32,"seed":9}"#,
    );
    assert_eq!(c["a_mean_return"], 22.0);
    assert!(c["b_minus_a"].as_f64().unwrap() < -20.0, "{c}");
    assert_eq!(c["a_better_episodes"], 32);
    assert_eq!(ws.state.used, 2);
    ok(&mut ws, r#"{"tool":"revert","tag":"good"}"#);
    // Switching off the "go left" comparator breaks catching; an unused unit
    // changes nothing.
    let a = ok(
        &mut ws,
        r#"{"tool":"ablate","units":["l0.h1"],"episodes":32,"seed":9}"#,
    );
    assert!(a["b_minus_a"].as_f64().unwrap() < -5.0, "{a}");
    assert_eq!(ws.state.used, 4);
    let spent = handle_json(&mut ws, r#"{"tool":"ablate","units":["l0.h7"]}"#);
    assert!(
        spent["error"]
            .as_str()
            .unwrap()
            .contains("2 forward passes")
    );
    let mut fresh = bounce(2);
    let none = ok(
        &mut fresh,
        r#"{"tool":"ablate","units":["l0.h7"],"episodes":32,"seed":9}"#,
    );
    assert_eq!(none["b_minus_a"], 0.0);
    assert!(handle_json(&mut fresh, r#"{"tool":"ablate","units":["l5.h0"]}"#)["error"].is_string());
}

#[test]
fn script_computes_and_applies_calls_in_a_sandbox() {
    let mut ws = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, 1);
    // Write the landing program's go-right column with a loop, not by hand.
    let code = r#"
        const W = 10, H = 10, G = 4;
        function land(x, y, dx) {
          for (let s = y; s < H - 1; s++) {
            let n = x + dx;
            if (n < 0) { n = -n; dx = -dx; } else if (n > W - 1) { n = 2 * (W - 1) - n; dx = -dx; }
            x = n;
          }
          return x;
        }
        const col = SPEC.obs_labels.map(l => {
          let m = l.match(/^ball([<>])\(x=(\d+),y=(\d+)\)$/);
          if (m) return +m[3] < H - 1 ? G * land(+m[2], +m[3], m[1] === '>' ? 1 : -1) : 0;
          m = l.match(/^paddle\(x=(\d+)\)$/);
          return -G * +m[1];
        });
        return { n: col.length, calls: [
          { tool: 'set_col', name: 'l0.w', col: 0, values: col },
          { tool: 'set_col', name: 'l0.w', col: 1, values: col.map(v => -v) },
        ]};
    "#;
    let r = ok(
        &mut ws,
        &serde_json::json!({ "tool": "script", "code": code }).to_string(),
    );
    assert_eq!(r["value"]["n"], 210);
    assert!(
        r["applied"]
            .as_array()
            .unwrap()
            .iter()
            .all(|a| a.get("ok").is_some())
    );
    let mut want = bounce(0);
    assert_eq!(
        ws.param("l0.w").unwrap().data,
        want.param("l0.w").unwrap().data,
        "script wrote the same column as the reference program"
    );
    let _ = &mut want;
    // Scripts see shapes, never weight values; no host APIs; loops are cut off.
    let r = handle_json(
        &mut ws,
        r#"{"tool":"script","read":["l0.b"],"code":"return 1;"}"#,
    );
    assert!(
        r["error"].as_str().unwrap().contains("cannot read weights"),
        "{r}"
    );
    let r = ok(
        &mut ws,
        r#"{"tool":"script","code":"return [SPEC.params['l0.b'].cols, typeof P, typeof require, typeof process];"}"#,
    );
    assert_eq!(
        r["value"],
        serde_json::json!([32, "undefined", "undefined", "undefined"])
    );
    let r = handle_json(&mut ws, r#"{"tool":"script","code":"for(;;){}"}"#);
    assert!(r["error"].as_str().unwrap().contains("budget"), "{r}");
    let r = handle_json(
        &mut ws,
        r#"{"tool":"script","code":"return {calls:[{tool:'script',code:'1'}]};"}"#,
    );
    assert!(r["ok"]["applied"][0]["error"].is_string());
    assert_eq!(ws.state.used, 0);
}

#[test]
fn claim_check_finds_what_the_text_leaves_out() {
    let mut ws = bounce(0);
    let imp = unit_importance(&mut ws, 32).unwrap();
    let important: Vec<&str> = imp
        .iter()
        .filter(|(_, d)| d.abs() >= 0.5)
        .map(|(u, _)| u.as_str())
        .collect();
    for u in ["l0.h0", "l0.h1", "l1.h0", "l1.h1"] {
        assert!(important.contains(&u), "{imp:?}");
    }
    let full = claim_check("units h0 and h1 compare landing and paddle", &imp, 0.5);
    assert_eq!(full["important_but_not_mentioned"], serde_json::json!([]));
    let partial = claim_check("l0.h0 goes right; l1.h5 does nothing", &imp, 0.5);
    let missing: Vec<&str> = partial["important_but_not_mentioned"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v[0].as_str().unwrap())
        .collect();
    assert!(
        missing.contains(&"l0.h1") && missing.contains(&"l1.h1"),
        "{missing:?}"
    );
}

#[test]
fn warmstart_reports_both_curves() {
    let ws = bounce(0);
    let opts = WarmstartOptions {
        teacher_episodes: 4,
        epochs: 1,
        eval_episodes: 32,
        ..Default::default()
    };
    let r = warmstart(&ws, opts).unwrap();
    let agent = r["curves"]["agent"].as_array().unwrap();
    let random = r["curves"]["random"].as_array().unwrap();
    assert_eq!((agent.len(), random.len()), (2, 2));
    // Epoch 0 of the agent curve is the agent's program itself.
    assert_eq!(agent[0]["held_out_return"], 22.0);
    assert!(agent[0]["teacher_agreement"].as_f64().unwrap() > 0.99);
}

#[test]
fn export_bakes_weights_and_checks_int8() {
    let mut ws = bounce(0);
    let dir = std::env::temp_dir().join(format!("rlx_wsp_export_{}", std::process::id()));
    let r = rlx_wsp::export::export(&mut ws, &dir, false, 32).unwrap();
    let nonzero: usize = ws
        .state
        .params
        .values()
        .map(|t| t.data.iter().filter(|v| **v != 0.0).count())
        .sum();
    assert_eq!(r["weights"]["nonzero"], nonzero);
    assert_eq!(r["weights"]["total"], 7907);
    assert_eq!(r["held_out"]["f32"]["score"], 22.0);
    assert_eq!(r["held_out"]["int8_per_tensor"]["score"], 22.0);
    // The baked graph has no params left and runs on its own.
    let text = std::fs::read_to_string(dir.join("model.rlxmodel")).unwrap();
    let g = rlx_ir::rlxmodel::from_text(&text).unwrap();
    assert!(
        !g.nodes()
            .iter()
            .any(|n| matches!(n.op, rlx_ir::Op::Param { .. }))
    );
    let mut obs = vec![0f32; 210];
    let spec = ws.spec();
    for l in ["ball>(x=8,y=0)", "paddle(x=5)"] {
        obs[spec.obs_labels.iter().position(|x| x == l).unwrap()] = 1.0;
    }
    let out = rlx_runtime::Session::new(rlx_runtime::Device::Cpu)
        .compile(g)
        .run(&[("obs", &obs)]);
    assert_eq!(
        rlx_wsp::space::argmax(&out[0]),
        1,
        "left, as the probe says"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn edited_units_are_the_ones_whose_weights_changed() {
    let ws = bounce(0);
    let before = ws.state.params.clone();
    let mut after = before.clone();
    after.get_mut("l0.w").unwrap().data[3 * 32 + 7] += 1.0; // into l0.h7
    after.get_mut("l1.b").unwrap().data[2] = 0.5; // l1.h2
    after.get_mut("l2.w").unwrap().data[5 * 3 + 1] = 0.25; // out of l1.h5
    let touched = rlx_wsp::verify::edited_units(&ws.state.mlp.widths, &before, &after);
    let want: std::collections::BTreeSet<String> = ["l0.h7", "l1.h2", "l1.h5"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(touched, want);
}

#[test]
fn scripts_carrying_copied_weights_are_refused() {
    // A trained-looking network: non-round weights.
    let mut ws = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, 0);
    rlx_wsp::baselines::random_init(&mut ws, 7);
    let w: Vec<f32> = ws.param("l2.w").unwrap().data[..40].to_vec();
    // Pasted as the slice view shows them (4 decimals).
    let pasted: Vec<String> = w.iter().map(|v| format!("{:.4}", v)).collect();
    let code = format!("const W = [{}]; return W.length;", pasted.join(","));
    let r = handle_json(
        &mut ws,
        &serde_json::json!({"tool": "script", "code": code}).to_string(),
    );
    assert!(r["error"].as_str().unwrap().contains("weights"), "{r}");
    assert!(ws.state.log.iter().any(|l| l.starts_with("script refused")));
    // Large literal tables are refused too, even if rounded.
    let table: Vec<String> = (0..300).map(|i| format!("{}.5", i)).collect();
    let code = format!("return [{}].length;", table.join(","));
    let r = handle_json(
        &mut ws,
        &serde_json::json!({"tool": "script", "code": code}).to_string(),
    );
    assert!(r["error"].as_str().unwrap().contains("literals"), "{r}");
    // A rule-based weight generator with round constants still runs.
    let code = "const G = 4, H = 10; const v = []; for (let i = 0; i < 210; i++) v.push(G * (i % H) - 0.5); return v.length;";
    let r = handle_json(
        &mut ws,
        &serde_json::json!({"tool": "script", "code": code}).to_string(),
    );
    assert_eq!(r["ok"]["value"], 210, "{r}");
}
