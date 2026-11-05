// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An agent in the training loop (after OPAL, "Reactive Optimizers",
//! arXiv:2610.33826): a LoRA trains on the practice set while the agent
//! watches it and changes how it trains.
//!
//! - **Training:** `rlx_runtime::model::trainer::prepare` makes the
//!   package's trainable function a LoRA loss (cross-entropy on the
//!   reference answer); a [`ResidentTrainer`] runs forward, backward and the
//!   optimizer in one graph, with a rate per factor and a freeze gate per
//!   factor. `train {steps}` advances it; `train {background: true}` runs
//!   it in a thread while the agent keeps working (its updater and gate are
//!   read every step).
//! - **The updater** (`set_updater`): a JavaScript function body run before
//!   every step with `STATS` (step, recent losses, the last step's per-site
//!   probes, current rates, optimizer). It returns changes: `lr` (all),
//!   `site_lr` (`{pattern: rate}`), `freeze` / `unfreeze` (patterns),
//!   `gate` (skip a batch whose loss exceeds `gate` × the median of recent
//!   losses — the batch is scored before the step), `skip` (this step) and
//!   `optimizer` (`adamw` | `muon`: the trainer is rebuilt, the factors
//!   kept, the moments reset).
//! - **Probes,** per site (adapted tensor) every step: the update's norm,
//!   its ratio to the factors' norm, and its cosine with the previous
//!   update; plus the loss and whether the step was gated. `analyze` runs
//!   the agent's own JavaScript over the whole history (`HISTORY`).
//! - **Snapshots:** `lora_snapshot {tag}` applies the current factors to
//!   the space's package (the version training started from, plus the
//!   adapter) and keeps it as checkpoint `tag` — the live weights — so
//!   `qa_loglik`, `compare`, `dev_check` and `average` measure it.

use serde::{Deserialize, Serialize};

/// The training a space steers (kept in its state).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpalConfig {
    /// A package with a trainable function (`rlx-export … --variant train`).
    pub train_package: String,
    pub targets: Vec<String>,
    pub rank: usize,
    pub alpha: f32,
    pub lr: f32,
    /// Training steps the run may take in all.
    pub max_steps: usize,
    #[serde(default)]
    pub steps_done: usize,
    #[serde(default = "default_mode")]
    pub mode: String,
    /// The updater's source (see the module docs).
    #[serde(default)]
    pub updater: Option<String>,
}

fn default_mode() -> String {
    "solution".into()
}

pub use imp::*;

#[cfg(not(feature = "training"))]
mod imp {
    use crate::space::WeightSpace;
    use serde_json::Value;

    const NEED: &str = "the training tools need rlx-wsp built with --features training";

    pub fn train(_: &mut WeightSpace, _: usize, _: bool) -> Result<Value, String> {
        Err(NEED.into())
    }
    pub fn set_updater(_: &mut WeightSpace, _: &str) -> Result<Value, String> {
        Err(NEED.into())
    }
    pub fn analyze(_: &mut WeightSpace, _: &str) -> Result<Value, String> {
        Err(NEED.into())
    }
    pub fn snapshot(_: &mut WeightSpace, _: &str) -> Result<Value, String> {
        Err(NEED.into())
    }
    pub fn status(_: &mut WeightSpace) -> Result<Value, String> {
        Err(NEED.into())
    }
}

#[cfg(feature = "training")]
mod imp {
    use super::OpalConfig;
    use crate::space::{Variant, WeightSpace};
    use rlx_runtime::model::trainer::Saver;
    use rlx_runtime::train::{AdamSpec, FusedRule, MuonSpec, ResidentTrainer, TrainableParam};
    use serde_json::{Value, json};
    use std::collections::{BTreeMap, HashMap};
    use std::sync::{Arc, Mutex};

    /// Instructions an updater call may take.
    const UPDATER_BUDGET: u64 = 20_000_000;

    /// A training run in progress (held by the space, not saved).
    pub struct OpalRun {
        trainer: ResidentTrainer,
        saver: Saver,
        graph: rlx_ir::Graph,
        wrt: Vec<TrainableParam>,
        initial: HashMap<String, Vec<f32>>,
        /// (site param, down factor, up factor).
        sites: Vec<(String, String, String)>,
        /// Pre-tokenized examples (one per practice question), cycled.
        batches: Vec<Vec<(String, Vec<f32>)>>,
        cursor: usize,
        optimizer: String,
        /// Each factor's rate.
        lrs: BTreeMap<String, f32>,
        frozen: Vec<String>,
        updater: Option<String>,
        history: Vec<Value>,
        prev_update: HashMap<String, Vec<f32>>,
        /// The version training started from (snapshots apply the adapter
        /// to it).
        base: Variant,
        max_steps: usize,
        steps_done: usize,
        /// Set to stop a background run.
        stop: bool,
        running: bool,
        error: Option<String>,
    }

    pub type Shared = Arc<Mutex<OpalRun>>;

    fn glob(p: &str, name: &str) -> bool {
        crate::lora::glob(p, name)
    }

    fn rules(kind: &str, wrt: &[TrainableParam], lrs: &BTreeMap<String, f32>) -> Vec<FusedRule> {
        wrt.iter()
            .map(|w| {
                let lr = lrs[&w.name];
                match kind {
                    "muon" => FusedRule::Muon(MuonSpec {
                        adamw: Some(AdamSpec::new(lr)),
                        ..MuonSpec::new(lr)
                    }),
                    _ => FusedRule::Adam(AdamSpec::new(lr)),
                }
            })
            .collect()
    }

    fn build(
        graph: &rlx_ir::Graph,
        wrt: &[TrainableParam],
        values: &HashMap<String, Vec<f32>>,
        kind: &str,
        lrs: &BTreeMap<String, f32>,
    ) -> Result<ResidentTrainer, String> {
        ResidentTrainer::with_rules(
            graph,
            wrt,
            values,
            &rules(kind, wrt, lrs),
            rlx_runtime::Device::Cpu,
            &rlx_runtime::CompileOptions::new(),
        )
        .map_err(|e| format!("building the trainer: {e:?}"))
    }

    /// Open the run the space's `opal` config describes.
    fn open(ws: &mut WeightSpace) -> Result<Shared, String> {
        if let Some(r) = &ws.opal_run {
            return Ok(r.clone());
        }
        let cfg: OpalConfig = ws
            .state
            .opal
            .clone()
            .ok_or("this run steers no training (make it with --opal --train-package …)")?;
        let q = ws
            .state
            .qa
            .clone()
            .ok_or("training reads a QA task's practice set")?;
        let options = crate::lora::qa_train_options(&cfg.targets, cfg.rank, cfg.alpha, cfg.lr);
        let p = rlx_runtime::model::trainer::prepare(&cfg.train_package, &options)
            .map_err(|e| format!("{}: {e:#}", cfg.train_package))?;
        let seq = crate::lora::seq_of(&p.inputs)?;
        let items = crate::lm::practice_items(&q)?;
        let batches = items
            .iter()
            .map(|it| crate::lora::qa_batch(ws, it, &cfg.mode, seq))
            .collect::<Result<Vec<_>, _>>()?;
        start(ws, p, batches, &cfg)
    }

    /// Start a run of `p` (a trainer's prepared function) on `batches`
    /// (named f32 inputs per example, cycled) under `cfg`'s step budget,
    /// rate and updater.
    pub fn start(
        ws: &mut WeightSpace,
        p: rlx_runtime::model::trainer::Prepared,
        batches: Vec<Vec<(String, Vec<f32>)>>,
        cfg: &OpalConfig,
    ) -> Result<Shared, String> {
        if batches.is_empty() {
            return Err("no training examples".into());
        }
        let (graph, trainable, initial, _, saver) = p.split();
        let wrt: Vec<TrainableParam> = trainable
            .iter()
            .map(|name| TrainableParam {
                name: name.clone(),
                node: graph
                    .nodes()
                    .iter()
                    .find(|n| matches!(&n.op, rlx_ir::Op::Param { name: m } if m == name))
                    .map(|n| n.id)
                    .expect("a trainable param is in the graph"),
            })
            .collect();
        let lrs: BTreeMap<String, f32> = trainable.iter().map(|n| (n.clone(), cfg.lr)).collect();
        let trainer = build(&graph, &wrt, &initial, "adamw", &lrs)?;
        let base = ws
            .variant("branch_start")
            .unwrap_or_else(|_| ws.current_variant());
        let run = Arc::new(Mutex::new(OpalRun {
            trainer,
            sites: saver.lora_sites(),
            saver,
            graph,
            wrt,
            initial,
            batches,
            cursor: 0,
            optimizer: "adamw".into(),
            lrs,
            frozen: Vec::new(),
            updater: cfg.updater.clone(),
            history: Vec::new(),
            prev_update: HashMap::new(),
            base,
            max_steps: cfg.max_steps,
            steps_done: cfg.steps_done,
            stop: false,
            running: false,
            error: None,
        }));
        ws.opal_run = Some(run.clone());
        Ok(run)
    }

    fn median(v: &[f64]) -> Option<f64> {
        if v.len() < 8 {
            return None;
        }
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        Some(s[s.len() / 2])
    }

    impl OpalRun {
        fn recent_losses(&self, n: usize) -> Vec<f64> {
            self.history
                .iter()
                .rev()
                .filter(|h| h["gated"] != true && h["skipped"] != true)
                .filter_map(|h| h["loss"].as_f64())
                .take(n)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect()
        }

        fn stats(&self) -> Value {
            json!({
                "step": self.steps_done,
                "recent_losses": self.recent_losses(32),
                "last": self.history.last(),
                "sites": self.sites.iter().map(|s| &s.0).collect::<Vec<_>>(),
                "lr": self.lrs.values().next(),
                "optimizer": self.optimizer,
                "frozen": self.frozen,
            })
        }

        fn site_factors(&self, pattern: &str) -> Vec<String> {
            self.sites
                .iter()
                .filter(|(p, _, _)| glob(pattern, p))
                .flat_map(|(_, d, u)| [d.clone(), u.clone()])
                .collect()
        }

        /// Apply an updater's answer; returns (gate, skip).
        fn apply(&mut self, d: &Value) -> Result<(Option<f64>, bool), String> {
            if let Some(kind) = d["optimizer"].as_str()
                && kind != self.optimizer
            {
                if !matches!(kind, "adamw" | "muon") {
                    return Err(format!("optimizer `{kind}`: adamw | muon"));
                }
                let mut values = self.initial.clone();
                values.extend(self.trainer.params());
                self.trainer = build(&self.graph, &self.wrt, &values, kind, &self.lrs)?;
                for f in &self.frozen {
                    self.trainer.set_frozen(f, true);
                }
                self.optimizer = kind.to_string();
            }
            if let Some(lr) = d["lr"].as_f64() {
                for (n, v) in self.lrs.iter_mut() {
                    *v = lr as f32;
                    self.trainer.set_param_lr(n, lr as f32);
                }
            }
            if let Some(m) = d["site_lr"].as_object() {
                for (pat, lr) in m {
                    let lr = lr.as_f64().ok_or("site_lr values are numbers")? as f32;
                    for f in self.site_factors(pat) {
                        self.lrs.insert(f.clone(), lr);
                        self.trainer.set_param_lr(&f, lr);
                    }
                }
            }
            for (key, on) in [("freeze", true), ("unfreeze", false)] {
                for pat in d[key]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                {
                    for f in self.site_factors(pat) {
                        self.trainer.set_frozen(&f, on);
                        if on {
                            if !self.frozen.contains(&f) {
                                self.frozen.push(f);
                            }
                        } else {
                            self.frozen.retain(|x| x != &f);
                        }
                    }
                }
            }
            Ok((d["gate"].as_f64(), d["skip"].as_bool() == Some(true)))
        }

        /// One step: the updater, the gate, the update, the probes.
        fn step(&mut self) -> Result<Value, String> {
            let (gate, skip) = match self.updater.clone() {
                Some(code) => {
                    let stats = self.stats();
                    let d = crate::script::run_with(&code, &[("STATS", &stats)], UPDATER_BUDGET)
                        .map_err(|e| format!("updater: {e}"))?;
                    self.apply(&d)?
                }
                None => (None, false),
            };
            let batch = self.batches[self.cursor % self.batches.len()].clone();
            self.cursor += 1;
            let feeds: Vec<(&str, &[f32])> = batch
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let mut rec = json!({ "step": self.steps_done, "lr": self.lrs.values().next(), "optimizer": self.optimizer });
            if skip {
                rec["skipped"] = json!(true);
                self.steps_done += 1;
                self.history.push(rec.clone());
                return Ok(rec);
            }
            if let Some(g) = gate {
                let loss = self.trainer.forward_only(&feeds)[0][0] as f64;
                if let Some(m) = median(&self.recent_losses(32))
                    && loss > g * m
                {
                    rec["loss"] = json!(loss);
                    rec["gated"] = json!(true);
                    rec["median"] = json!(m);
                    self.steps_done += 1;
                    self.history.push(rec.clone());
                    return Ok(rec);
                }
            }
            let before = self.trainer.params();
            let loss = self.trainer.step(&feeds)[0][0] as f64;
            let after = self.trainer.params();
            let mut probes = serde_json::Map::new();
            for (site, d, u) in &self.sites {
                let (mut up2, mut w2, mut dot, mut prev2) = (0f64, 0f64, 0f64, 0f64);
                let mut delta = Vec::new();
                for f in [d, u] {
                    let (a, b) = (&after[f], &before[f]);
                    for (x, y) in a.iter().zip(b) {
                        let dx = (x - y) as f64;
                        up2 += dx * dx;
                        w2 += (*x as f64).powi(2);
                        delta.push(dx as f32);
                    }
                }
                if let Some(p) = self.prev_update.get(site) {
                    for (a, b) in delta.iter().zip(p) {
                        dot += (*a as f64) * (*b as f64);
                        prev2 += (*b as f64).powi(2);
                    }
                }
                let cos = if prev2 > 0.0 && up2 > 0.0 {
                    dot / (prev2.sqrt() * up2.sqrt())
                } else {
                    0.0
                };
                probes.insert(
                    site.clone(),
                    json!({ "update_norm": up2.sqrt(), "update_to_weight": up2.sqrt() / w2.sqrt().max(1e-12), "cos_prev_update": cos }),
                );
                self.prev_update.insert(site.clone(), delta);
            }
            rec["loss"] = json!(loss);
            rec["sites"] = Value::Object(probes);
            self.steps_done += 1;
            self.history.push(rec.clone());
            Ok(rec)
        }
    }

    fn summary(run: &OpalRun, from: usize) -> Value {
        let span = &run.history[from.min(run.history.len())..];
        let losses: Vec<f64> = span.iter().filter_map(|h| h["loss"].as_f64()).collect();
        json!({
            "steps_run": span.len(),
            "steps_done": run.steps_done, "max_steps": run.max_steps,
            "mean_loss": if losses.is_empty() { None } else { Some(losses.iter().sum::<f64>() / losses.len() as f64) },
            "first_loss": losses.first(), "last_loss": losses.last(),
            "gated": span.iter().filter(|h| h["gated"] == true).count(),
            "skipped": span.iter().filter(|h| h["skipped"] == true).count(),
            "optimizer": run.optimizer, "lr": run.lrs.values().next(),
            "frozen": run.frozen,
            "last": run.history.last(),
            "running_in_background": run.running,
            "error": run.error,
        })
    }

    /// `train {steps}`: advance training (in a thread with `background`).
    pub fn train(ws: &mut WeightSpace, steps: usize, background: bool) -> Result<Value, String> {
        let run = open(ws)?;
        let left = {
            let r = run.lock().map_err(|_| "the training run failed")?;
            if r.running {
                return Err("training is running in the background; see train_status".into());
            }
            r.max_steps.saturating_sub(r.steps_done)
        };
        let steps = steps.min(left);
        if steps == 0 {
            return Err("the training budget is spent".into());
        }
        ws.check_tokens()?;
        let per_step = {
            let r = run.lock().map_err(|_| "the training run failed")?;
            3 * r.batches.first().map_or(0, |b| b[0].1.len())
        };
        ws.spend_tokens(steps * per_step);
        if background {
            {
                let mut r = run.lock().map_err(|_| "the training run failed")?;
                r.running = true;
                r.stop = false;
                r.error = None;
            }
            let shared = run.clone();
            std::thread::spawn(move || {
                for _ in 0..steps {
                    let Ok(mut r) = shared.lock() else { return };
                    if r.stop {
                        break;
                    }
                    if let Err(e) = r.step() {
                        r.error = Some(e);
                        break;
                    }
                }
                if let Ok(mut r) = shared.lock() {
                    r.running = false;
                }
            });
            ws.state
                .log
                .push(format!("train: {steps} steps started in the background"));
            return Ok(
                json!({ "started": steps, "note": "training runs on; train_status shows it, set_updater changes it from the next step" }),
            );
        }
        let mut r = run.lock().map_err(|_| "the training run failed")?;
        let from = r.history.len();
        for _ in 0..steps {
            r.step()?;
        }
        if let Some(c) = ws.state.opal.as_mut() {
            c.steps_done = r.steps_done;
        }
        ws.state
            .log
            .push(format!("train: {steps} steps (to {})", r.steps_done));
        Ok(summary(&r, from))
    }

    pub fn set_updater(ws: &mut WeightSpace, code: &str) -> Result<Value, String> {
        let run = open(ws)?;
        let mut r = run.lock().map_err(|_| "the training run failed")?;
        let code = (!code.trim().is_empty()).then(|| code.to_string());
        // Try it once on the current statistics.
        let trial = match &code {
            Some(c) => Some(
                crate::script::run_with(c, &[("STATS", &r.stats())], UPDATER_BUDGET)
                    .map_err(|e| format!("updater: {e}"))?,
            ),
            None => None,
        };
        r.updater = code.clone();
        if let Some(c) = ws.state.opal.as_mut() {
            c.updater = code;
        }
        ws.state.log.push("train: updater set".into());
        Ok(json!({ "installed": r.updater.is_some(), "on_current_stats_returns": trial }))
    }

    pub fn analyze(ws: &mut WeightSpace, code: &str) -> Result<Value, String> {
        let run = open(ws)?;
        let r = run.lock().map_err(|_| "the training run failed")?;
        let history = Value::Array(r.history.clone());
        crate::script::run_with(
            code,
            &[("HISTORY", &history), ("STATS", &r.stats())],
            500_000_000,
        )
    }

    /// The current adapter on the starting version, as checkpoint `tag`
    /// and the live weights.
    pub fn snapshot(ws: &mut WeightSpace, tag: &str) -> Result<Value, String> {
        let run = open(ws)?;
        let (updates, base, steps) = {
            let mut r = run.lock().map_err(|_| "the training run failed")?;
            let trained = r.trainer.params();
            (
                r.saver
                    .lora_updates(&trained)
                    .map_err(|e| format!("reading the factors: {e:#}"))?,
                r.base.clone(),
                r.steps_done,
            )
        };
        ws.swap_variant(base);
        let done = crate::lora::install_updates(ws, &updates)?;
        ws.checkpoint(tag);
        Ok(
            json!({ "checkpoint": tag, "adapted": done.len(), "after_steps": steps,
                   "note": "the live weights are now this snapshot; measure with qa_loglik / compare / dev_check" }),
        )
    }

    pub fn status(ws: &mut WeightSpace) -> Result<Value, String> {
        let run = open(ws)?;
        let r = run.lock().map_err(|_| "the training run failed")?;
        if let Some(c) = ws.state.opal.as_mut() {
            c.steps_done = r.steps_done;
        }
        let from = r.history.len().saturating_sub(20);
        let mut s = summary(&r, from);
        s["recent"] = json!(r.history[from..].iter().map(|h| json!({
            "step": h["step"], "loss": h["loss"], "gated": h["gated"], "skipped": h["skipped"],
        })).collect::<Vec<_>>());
        Ok(s)
    }

    #[cfg(test)]
    mod tests {
        fn is_send<T: Send>() {}
        #[test]
        fn a_run_can_train_in_a_thread() {
            is_send::<super::OpalRun>();
        }
    }
}
