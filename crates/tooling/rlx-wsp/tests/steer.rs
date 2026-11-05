// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The steering protocol over a toy trainee: gradient descent on `w²`, whose
//! rate an updater sets.

use rlx_wsp::steer::{Trainee, attach, detach};
use rlx_wsp::{Act, EnvKind, WeightSpace, handle_json};
use serde_json::{Value, json};
use std::collections::BTreeMap;

struct Toy {
    w: f32,
    lr: f32,
}

impl Trainee for Toy {
    fn describe(&self) -> Value {
        json!({ "what": "w^2", "updater_returns": ["lr", "skip"] })
    }
    fn stats(&self) -> Value {
        json!({ "lr": self.lr })
    }
    fn apply(&mut self, d: &Value) -> Result<(), String> {
        if d["bad"] == true {
            return Err("rejected".into());
        }
        if let Some(lr) = d["lr"].as_f64() {
            self.lr = lr as f32;
        }
        Ok(())
    }
    fn step(&mut self) -> Result<Value, String> {
        let loss = self.w * self.w;
        self.w -= self.lr * 2.0 * self.w;
        Ok(json!({ "loss": loss }))
    }
    fn validate(&mut self) -> Result<Value, String> {
        Ok(json!({ "w": self.w }))
    }
    fn params(&mut self) -> BTreeMap<String, Vec<f32>> {
        BTreeMap::from([("l0.b".to_string(), vec![self.w; 32])])
    }
}

fn ok(ws: &mut WeightSpace, call: &str) -> Value {
    let r = handle_json(ws, call);
    assert!(r.get("error").is_none(), "{call} -> {r}");
    r["ok"].clone()
}

fn space(budget: usize, steps: usize) -> WeightSpace {
    let mut ws = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, budget);
    attach(&mut ws, Box::new(Toy { w: 1.0, lr: 0.0 }), steps);
    ws
}

#[test]
fn an_updater_steers_the_trainee_and_the_budget_holds() {
    let mut ws = space(1, 10);
    let flat = ok(&mut ws, r#"{"tool":"train","steps":2}"#);
    assert_eq!(flat["last_loss"], 1.0, "no updater: rate 0");

    ok(
        &mut ws,
        r#"{"tool":"set_updater","code":"return {lr: STATS.step < 6 ? 0.25 : 0, skip: STATS.step == 3}"}"#,
    );
    let run = ok(&mut ws, r#"{"tool":"train","steps":100}"#);
    assert_eq!(run["steps_run"], 8, "stops at the run's budget");
    assert_eq!(run["skipped"], 1);
    // Each update at 0.25 halves w.
    assert!(
        ok(&mut ws, r#"{"tool":"train_validate"}"#)["w"]
            .as_f64()
            .unwrap()
            < 0.2
    );
    assert!(
        handle_json(&mut ws, r#"{"tool":"train_validate"}"#)
            .get("error")
            .is_some(),
        "one pass"
    );
    assert!(
        handle_json(&mut ws, r#"{"tool":"train","steps":1}"#)
            .get("error")
            .is_some()
    );
    assert_eq!(
        ok(&mut ws, r#"{"tool":"train_status"}"#)["describe"]["what"],
        "w^2"
    );
    let n = ok(
        &mut ws,
        r#"{"tool":"analyze","code":"return HISTORY.filter(h => h.skipped).length"}"#,
    );
    assert_eq!(n, 1);
}

#[test]
fn finish_runs_what_the_agent_left_and_a_snapshot_goes_live() {
    let mut ws = space(0, 5);
    ok(
        &mut ws,
        r#"{"tool":"set_updater","code":"return {lr: 0.5}"}"#,
    );
    ok(&mut ws, r#"{"tool":"train","steps":1}"#);
    let snap = ok(&mut ws, r#"{"tool":"train_snapshot","tag":"t"}"#);
    assert_eq!(snap["after_steps"], 1);
    let mut r = detach(&mut ws).unwrap();
    let mut seen = 0;
    r.finish(|_| seen += 1).unwrap();
    assert_eq!((seen, r.steps_left()), (4, 0));
}

/// A policy the trainee rejects, or one that throws, costs the updater and not
/// the run: the last good one takes over and the agent is told.
#[test]
fn a_failing_updater_falls_back_and_training_goes_on() {
    let mut ws = space(0, 30);
    ok(
        &mut ws,
        r#"{"tool":"set_updater","code":"return {lr: 0.25}"}"#,
    );
    ok(&mut ws, r#"{"tool":"train","steps":2}"#);
    ok(
        &mut ws,
        r#"{"tool":"set_updater","code":"return {bad: true}"}"#,
    );
    let r = ok(&mut ws, r#"{"tool":"train","steps":3}"#);
    assert_eq!(r["steps_run"], 3);
    assert!(
        r["updater_errors"][0]
            .as_str()
            .unwrap()
            .contains("rejected"),
        "{r}"
    );
    assert_eq!(
        ok(&mut ws, r#"{"tool":"train_status"}"#)["updater_installed"],
        true
    );

    ok(
        &mut ws,
        r#"{"tool":"set_updater","code":"return STATS.step > 7 ? undefined_name : {}"}"#,
    );
    let r = ok(&mut ws, r#"{"tool":"train","steps":5}"#);
    assert_eq!(r["steps_run"], 5);
    let r = ok(&mut ws, r#"{"tool":"train","steps":1}"#);
    assert_eq!(r["steps_run"], 1);
}

#[test]
fn a_space_without_a_trainee_says_so() {
    let mut ws = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, 1);
    let r = handle_json(&mut ws, r#"{"tool":"train_validate"}"#);
    assert!(r["error"].as_str().unwrap().contains("steers no trainee"));
}
