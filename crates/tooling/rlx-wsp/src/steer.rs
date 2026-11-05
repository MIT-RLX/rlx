// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An agent in the loop of a training run this crate does not own: the
//! protocol of [`crate::opal`] over any trainer.
//!
//! [`crate::opal`] trains a LoRA on a QA task's practice set. A [`Trainee`] is
//! everything else: the program that embeds this crate keeps its data, its
//! model and its optimizer, and implements five methods; `train`,
//! `set_updater`, `analyze`, `train_status`, `train_validate` and
//! `train_snapshot` then steer it. The updater is the same JavaScript body, run
//! before every step with `STATS`; what it may return beyond `skip` is the
//! trainee's to define, and its [`Trainee::describe`] tells the agent.
//!
//! Attach one with [`attach`] before serving the space
//! ([`crate::mcp::serve`]). A run lives in the serving process.

use crate::space::WeightSpace;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Instructions an updater call may take.
const UPDATER_BUDGET: u64 = 20_000_000;
/// The most steps one `train` call runs: a tool call must come back.
pub const MAX_STEPS_PER_CALL: usize = 1000;
/// Step records kept for `analyze`.
const MAX_HISTORY: usize = 200_000;

/// A training run an agent steers.
pub trait Trainee: Send {
    /// What the run is, what `STATS` holds and what an updater may return.
    fn describe(&self) -> Value;
    /// This run's part of `STATS` (rates, frozen sites, …).
    fn stats(&self) -> Value;
    /// Apply an updater's answer (it holds until changed).
    fn apply(&mut self, decision: &Value) -> Result<(), String>;
    /// One update; the step's record, with at least `loss`.
    fn step(&mut self) -> Result<Value, String>;
    /// Measure the current weights on data the agent may see.
    fn validate(&mut self) -> Result<Value, String>;
    /// The current weights, flat row-major by name.
    fn params(&mut self) -> BTreeMap<String, Vec<f32>>;
}

/// A trainee and what the agent did to it (held by the space, not saved).
pub struct SteeredRun {
    trainee: Box<dyn Trainee>,
    updater: Option<String>,
    /// The last updater that ran and applied: where a failing one falls back.
    good: Option<String>,
    /// Why updaters were dropped, for the agent.
    errors: Vec<String>,
    history: Vec<Value>,
    max_steps: usize,
    steps_done: usize,
}

impl SteeredRun {
    fn stats(&self) -> Value {
        let recent: Vec<f64> = self
            .history
            .iter()
            .rev()
            .filter_map(|h| h["loss"].as_f64())
            .take(32)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let mut s = json!({
            "step": self.steps_done, "max_steps": self.max_steps,
            "recent_losses": recent, "last": self.history.last(),
        });
        if let (Some(o), Value::Object(t)) = (s.as_object_mut(), self.trainee.stats()) {
            o.extend(t);
        }
        s
    }

    fn step(&mut self) -> Result<(), String> {
        let mut skip = false;
        if let Some(code) = self.updater.clone() {
            let stats = self.stats();
            let answer = crate::script::run_with(&code, &[("STATS", &stats)], UPDATER_BUDGET)
                .and_then(|d| self.trainee.apply(&d).map(|()| d));
            match answer {
                Ok(d) => {
                    skip = d["skip"].as_bool() == Some(true);
                    self.good = Some(code);
                }
                // A bad policy must not cost the run: training goes on.
                Err(e) => {
                    self.errors
                        .push(format!("step {}: updater dropped: {e}", self.steps_done));
                    self.updater = self.good.take().filter(|g| *g != code);
                }
            }
        }
        let mut rec = if skip {
            json!({ "skipped": true })
        } else {
            self.trainee.step()?
        };
        rec["step"] = json!(self.steps_done);
        self.steps_done += 1;
        if self.history.len() == MAX_HISTORY {
            self.history.drain(..MAX_HISTORY / 2);
        }
        self.history.push(rec);
        Ok(())
    }

    fn summary(&self, from: usize) -> Value {
        let span = &self.history[from.min(self.history.len())..];
        let losses: Vec<f64> = span.iter().filter_map(|h| h["loss"].as_f64()).collect();
        json!({
            "steps_run": span.len(), "steps_done": self.steps_done, "max_steps": self.max_steps,
            "mean_loss": (!losses.is_empty()).then(|| losses.iter().sum::<f64>() / losses.len() as f64),
            "first_loss": losses.first(), "last_loss": losses.last(),
            "skipped": span.iter().filter(|h| h["skipped"] == true).count(),
            "updater_errors": self.errors.iter().rev().take(5).collect::<Vec<_>>(),
            "stats": self.stats(),
        })
    }

    /// Steps left of the run's budget.
    pub fn steps_left(&self) -> usize {
        self.max_steps.saturating_sub(self.steps_done)
    }

    /// Use the trainee directly (to save it, say).
    pub fn with_trainee<R>(&mut self, f: impl FnOnce(&mut dyn Trainee) -> R) -> R {
        f(self.trainee.as_mut())
    }

    /// Run what is left of the budget under the current updater: an agent
    /// that stops early does not leave the schedule unfinished.
    pub fn finish(&mut self, mut each: impl FnMut(usize)) -> Result<(), String> {
        while self.steps_left() > 0 {
            self.step()?;
            each(self.steps_done);
        }
        Ok(())
    }
}

/// Steer `trainee` from `ws`, for `max_steps` steps in all.
pub fn attach(ws: &mut WeightSpace, trainee: Box<dyn Trainee>, max_steps: usize) {
    ws.steered = Some(SteeredRun {
        trainee,
        updater: None,
        good: None,
        errors: Vec::new(),
        history: Vec::new(),
        max_steps,
        steps_done: 0,
    });
}

/// The attached run, once the agent is done with it.
pub fn detach(ws: &mut WeightSpace) -> Option<SteeredRun> {
    ws.steered.take()
}

fn run(ws: &mut WeightSpace) -> Result<&mut SteeredRun, String> {
    ws.steered
        .as_mut()
        .ok_or_else(|| "this space steers no trainee".to_string())
}

/// `train {steps}`: advance the run, at most [`MAX_STEPS_PER_CALL`] at a time.
pub fn train(ws: &mut WeightSpace, steps: usize) -> Result<Value, String> {
    let r = run(ws)?;
    let n = steps.min(r.steps_left()).min(MAX_STEPS_PER_CALL);
    if n == 0 {
        return Err("the training budget is spent".into());
    }
    let from = r.history.len();
    for _ in 0..n {
        r.step()?;
    }
    let out = r.summary(from);
    ws.state.log.push(format!("train: {n} steps"));
    Ok(out)
}

pub fn set_updater(ws: &mut WeightSpace, code: &str) -> Result<Value, String> {
    let r = run(ws)?;
    let code = (!code.trim().is_empty()).then(|| code.to_string());
    let trial = match &code {
        Some(c) => Some(
            crate::script::run_with(c, &[("STATS", &r.stats())], UPDATER_BUDGET)
                .map_err(|e| format!("updater: {e}"))?,
        ),
        None => None,
    };
    r.good = r.updater.take();
    r.updater = code;
    let installed = r.updater.is_some();
    ws.state.log.push("train: updater set".into());
    Ok(json!({ "installed": installed, "on_current_stats_returns": trial }))
}

pub fn analyze(ws: &mut WeightSpace, code: &str) -> Result<Value, String> {
    let r = run(ws)?;
    let history = Value::Array(r.history.clone());
    crate::script::run_with(
        code,
        &[("HISTORY", &history), ("STATS", &r.stats())],
        500_000_000,
    )
}

pub fn status(ws: &mut WeightSpace) -> Result<Value, String> {
    let r = run(ws)?;
    let from = r.history.len().saturating_sub(20);
    let mut s = r.summary(from);
    s["describe"] = r.trainee.describe();
    s["updater_installed"] = json!(r.updater.is_some());
    Ok(s)
}

/// `train_validate`: one forward pass of the budget.
pub fn validate(ws: &mut WeightSpace) -> Result<Value, String> {
    run(ws)?;
    ws.charge("train_validate")?;
    run(ws)?.trainee.validate()
}

/// `train_snapshot {tag}`: the trainee's weights become the live weights, for
/// the reading tools, and checkpoint `tag`.
pub fn snapshot(ws: &mut WeightSpace, tag: &str) -> Result<Value, String> {
    let (params, step) = {
        let r = run(ws)?;
        (r.trainee.params(), r.steps_done)
    };
    let mut n = 0;
    for (name, data) in params {
        if let Some(t) = ws.state.params.get_mut(&name)
            && t.data.len() == data.len()
        {
            t.data = data;
            n += 1;
        }
    }
    ws.checkpoint(tag);
    Ok(json!({ "checkpoint": tag, "tensors": n, "after_steps": step }))
}
