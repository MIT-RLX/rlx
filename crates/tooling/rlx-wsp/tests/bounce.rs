// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Experiment 1 end to end: the tool surface is enough to write a Bounce
//! policy from zero weights, the network runs through rlx, and the budget,
//! checkpoints and views behave.

use rlx_wsp::baselines::{bounce_landing_program, bounce_tracker_program};
use rlx_wsp::{Act, EnvKind, WeightSpace, handle_json};
use serde_json::Value;

fn space(budget: usize) -> WeightSpace {
    WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, budget)
}

fn ok(ws: &mut WeightSpace, call: &str) -> Value {
    let r = handle_json(ws, call);
    assert!(r.get("error").is_none(), "{call} -> {r}");
    r["ok"].clone()
}

#[test]
fn landing_program_catches_every_drop() {
    let mut ws = space(0);
    bounce_landing_program(&mut ws).unwrap();
    let r = ws.evaluate(64, 99);
    assert_eq!(r.negative_events, 0, "{r:?}");
    assert!(r.positive_events > 64 * 20, "{r:?}");
}

#[test]
fn landing_prediction_is_what_scores() {
    let score = |f: Option<fn(&mut WeightSpace) -> Result<(), String>>| {
        let mut ws = space(0);
        if let Some(f) = f {
            f(&mut ws).unwrap();
        }
        ws.evaluate(64, 7).mean_return
    };
    let zero = score(None);
    let tracker = score(Some(bounce_tracker_program));
    let landing = score(Some(bounce_landing_program));
    // Following the current column scores *below* standing still: the paddle
    // moves before the ball, so it ends each step one column behind a
    // diagonal ball and only catches after a wall bounce.
    assert!(
        landing > zero && zero > tracker,
        "{zero} {tracker} {landing}"
    );
}

#[test]
fn probe_reports_comparator_units() {
    let mut ws = space(1);
    bounce_landing_program(&mut ws).unwrap();
    // Ball at x=8 moving right on row 0 bounces off the right wall and lands on x=1; paddle at 5.
    let r = ok(
        &mut ws,
        r#"{"tool":"probe","observations":[{"active":["ball>(x=8,y=0)","paddle(x=5)"]}]}"#,
    );
    assert_eq!(r[0]["action"], "left", "{r}");
    let h0 = r[0]["hidden"][0].as_array().unwrap();
    assert!(h0[1].as_f64().unwrap() > 0.9 && h0[0].as_f64().unwrap() < -0.9);
}

#[test]
fn budget_is_enforced() {
    let mut ws = space(2);
    ok(&mut ws, r#"{"tool":"rollout","episodes":32}"#);
    ok(
        &mut ws,
        r#"{"tool":"probe","observations":[{"active":["paddle(x=0)"]}]}"#,
    );
    let r = handle_json(&mut ws, r#"{"tool":"rollout"}"#);
    assert!(r["error"].as_str().unwrap().contains("budget"), "{r}");
    // Reads and edits stay free after the budget is gone.
    ok(&mut ws, r#"{"tool":"stats","name":"l0.w"}"#);
    ok(&mut ws, r#"{"tool":"fill","name":"l0.b","value":0.5}"#);
}

#[test]
fn checkpoint_revert_diff() {
    let mut ws = space(0);
    ok(&mut ws, r#"{"tool":"checkpoint","tag":"z"}"#);
    ok(
        &mut ws,
        r#"{"tool":"set","name":"l0.w","entries":[{"row":"paddle(x=3)","col":"l0.h4","value":2}]}"#,
    );
    let d = ok(&mut ws, r#"{"tool":"diff","tag":"z"}"#);
    assert_eq!(d[0]["name"], "l0.w");
    assert_eq!(d[0]["changed"], 1);
    let t = ok(&mut ws, r#"{"tool":"topk","name":"l0.w","k":3}"#);
    assert_eq!(t["entries"][0]["row"], "paddle(x=3)");
    assert_eq!(t["entries"][0]["col"], "l0.h4");
    ok(&mut ws, r#"{"tool":"revert","tag":"z"}"#);
    assert_eq!(
        ok(&mut ws, r#"{"tool":"diff","tag":"z"}"#),
        Value::Array(vec![])
    );
}

#[test]
fn views_read_the_written_program() {
    let mut ws = space(0);
    bounce_landing_program(&mut ws).unwrap();
    // The two comparator columns are exact negatives: rank 1.
    let s = ok(&mut ws, r#"{"tool":"svd","name":"l0.w","k":3}"#);
    let comps = s["components"].as_array().unwrap();
    assert_eq!(comps.len(), 1, "{s}");
    let sl = ok(
        &mut ws,
        r#"{"tool":"slice","name":"l0.w","cols":[0,2],"row_contains":"paddle"}"#,
    );
    assert_eq!(sl["rows"].as_array().unwrap().len(), 10);
    assert_eq!(sl["rows"][9]["values"][0], -36.0);
    let big = handle_json(&mut ws, r#"{"tool":"slice","name":"l0.w"}"#);
    assert!(big["error"].as_str().unwrap().contains("max"), "{big}");
}

#[test]
fn state_round_trips_through_json() {
    let mut ws = space(3);
    bounce_landing_program(&mut ws).unwrap();
    ok(&mut ws, r#"{"tool":"rollout","episodes":32}"#);
    let text = serde_json::to_string(&ws.state).unwrap();
    let mut back = WeightSpace::from_state(serde_json::from_str(&text).unwrap());
    assert_eq!(back.state.used, 1);
    assert_eq!(
        back.evaluate(32, 5).mean_return,
        ws.evaluate(32, 5).mean_return
    );
}
