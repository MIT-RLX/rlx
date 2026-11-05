// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Package spaces on a small generated `.rlxp`: weights read on demand,
//! edits as an overlay, the overlay written as an adapter whose merge
//! reproduces the edit exactly (checked by running the package through
//! rlx-runtime's `Model` API against the same network in memory), and the
//! compact state encoding.

use rlx_runtime::model::{Tensor as MTensor, rlxp};
use rlx_wsp::baselines::bounce_landing_program;
use rlx_wsp::{Act, EnvKind, WeightSpace, handle_json};
use serde_json::{Value, json};

fn ok(ws: &mut WeightSpace, call: Value) -> Value {
    let r = handle_json(ws, &call.to_string());
    assert!(r.get("error").is_none(), "{call} -> {r}");
    r["ok"].clone()
}

/// The Bounce landing program (batch 1) as an `.rlxp` with its weights.
fn write_package(dir: &std::path::Path) -> (std::path::PathBuf, WeightSpace) {
    let mut ws = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, 0);
    bounce_landing_program(&mut ws).unwrap();
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

fn run_model(path: &std::path::Path, obs: &[f32]) -> Vec<f32> {
    let mut m = rlx_runtime::model::open_source(&path.display().to_string(), json!({})).unwrap();
    let out = m
        .run(&[MTensor::f32("obs", vec![1, obs.len()], obs)], &json!({}))
        .unwrap();
    out[0].to_f32().unwrap()
}

#[test]
fn package_space_reads_edits_and_writes_an_exact_adapter() {
    let dir = std::env::temp_dir().join(format!("rlx_wsp_pkg_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (path, mlp) = write_package(&dir);
    let mut ws =
        WeightSpace::from_package(&path.display().to_string(), 0, None, Some("t".into())).unwrap();
    assert!(ws.state.params.is_empty(), "nothing copied until edited");

    // Reads come from the package, and match the source weights.
    let l = ok(&mut ws, json!({"tool": "list_params"}));
    assert_eq!(l["count"], 6);
    assert_eq!(ws.param("l0.w").unwrap(), mlp.param("l0.w").unwrap());
    let s = ok(&mut ws, json!({"tool": "stats", "name": "l2.w"}));
    assert_eq!(s["nonzero"], 2);
    // Graph tools read the package's own functions.
    let nodes = ok(&mut ws, json!({"tool": "list_nodes"}));
    assert!(nodes["nodes"].as_array().is_some_and(|n| !n.is_empty()));

    // An edit copies one tensor into the overlay.
    ok(
        &mut ws,
        json!({"tool": "set", "name": "l2.w", "entries": [{"row": 0, "col": 2, "value": -3.0}, {"row": 1, "col": 0, "value": 0.75}]}),
    );
    assert_eq!(ws.state.params.keys().collect::<Vec<_>>(), ["l2.w"]);

    // The overlay as an adapter over the package: the merged model computes
    // what the edited network computes in memory.
    let updates = ws.adapter_updates().unwrap();
    assert_eq!(updates.len(), 1);
    let adapter = dir.join("edit.rlxp");
    rlxp::save_adapter(&adapter, "e", &path.display().to_string(), &updates).unwrap();
    let mut edited = mlp;
    edited.set_params({
        let mut p = edited.state.params.clone();
        p.insert("l2.w".into(), ws.param("l2.w").unwrap().clone());
        p
    });
    let spec = edited.spec().clone();
    let mut obs = vec![0f32; spec.obs_dim];
    for l in ["ball>(x=8,y=0)", "paddle(x=5)"] {
        obs[spec.obs_labels.iter().position(|x| x == l).unwrap()] = 1.0;
    }
    let mut probe = WeightSpace::from_state({
        let mut st = edited.state.clone();
        st.budget = 1;
        st
    });
    let want = ok(
        &mut probe,
        json!({"tool": "probe", "observations": [obs.clone()]}),
    )[0]["logits"]
        .clone();
    let got = run_model(&adapter, &obs);
    for (a, b) in got.iter().zip(want.as_array().unwrap()) {
        assert!(
            (a - b.as_f64().unwrap() as f32).abs() < 1e-5,
            "{got:?} vs {want}"
        );
    }
    let base = run_model(&path, &obs);
    assert_ne!(base, got, "the adapter changed the output");

    // Compact state: round-trips, and is far smaller than number arrays.
    let text = serde_json::to_string(&ws.state).unwrap();
    assert!(text.contains("\"b64\""));
    let back: rlx_wsp::State = serde_json::from_str(&text).unwrap();
    assert_eq!(back.params["l2.w"], ws.state.params["l2.w"]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn old_number_array_states_still_read() {
    let t: rlx_wsp::Tensor =
        serde_json::from_str(r#"{"rows":1,"cols":3,"data":[1.0,-2.5,0.0]}"#).unwrap();
    assert_eq!(t.data, vec![1.0, -2.5, 0.0]);
    let again: rlx_wsp::Tensor = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
    assert_eq!(again, t);
    assert!(
        serde_json::from_str::<rlx_wsp::Tensor>(r#"{"rows":2,"cols":2,"data":[1.0]}"#).is_err()
    );
}
