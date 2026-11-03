// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The optimizer programs the agent writes, and the hosts that build them.
//!
//! A [`ProgramHost`] turns the files the agent left in its workspace into a
//! runnable [`Program`]: one update rule over the trainer's
//! [`OptItem`]s. The protocol around it (when the agent is asked, trial
//! windows, reverts) does not depend on what the files are written in.
//!
//! [`RuleHost`] is the native host: `updater.json` assigns every param to a
//! group, each group to one of rlx-optim's algorithms with its own
//! hyperparameters, and puts a piecewise-linear multiplier on the learning
//! rate over the run. It runs no agent code, so it needs no sandbox.

use std::any::Any;
use std::collections::{BTreeMap, HashMap};

use rlx_optim::{OptItem, Optimizer as OptimizerTrait, OptimizerState};
use serde_json::{Value, json};

use crate::runtime::opts::Opts;
use crate::runtime::train::optimizer;
use crate::{ApiError, ApiResult};

/// A program's source: file name → contents.
pub type Files = BTreeMap<String, String>;

/// One trainable param, as programs and the prompt see it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamInfo {
    pub name: String,
    pub shape: Vec<usize>,
}

/// What a program knows about the update it is making.
#[derive(Clone, Copy, Debug)]
pub struct StepCtx {
    /// This update, counted from 1.
    pub update: u64,
    pub total: u64,
    /// The learning rate the training script last set
    /// ([`super::AgentOptimizer::set_lr`]), if it ever did.
    pub base_lr: Option<f32>,
    /// This update's training loss, computed before the update.
    pub loss: Option<f32>,
}

/// A runnable update rule.
pub trait Program {
    /// Change every `items[i].param` in place from its gradient.
    /// Returns true to skip this update: when [`Self::can_skip`], the params
    /// and the program's state go back to before it.
    fn step(&mut self, items: &mut [OptItem<'_>], ctx: &StepCtx) -> ApiResult<bool>;
    /// Whether [`Self::step`] may ask to skip. The run then keeps a copy of
    /// the params and of [`Self::state_dict`] before every update, so ask only
    /// when a rule uses it.
    fn can_skip(&self) -> bool {
        false
    }
    /// For a host that carries state from the program it replaces.
    fn as_any(&self) -> &dyn Any;
    /// The program's state for a checkpoint; `None` when it cannot be saved,
    /// so a resumed run restarts it from zero (and says so).
    fn state_dict(&self) -> Option<OptimizerState> {
        None
    }
    /// Restore [`Self::state_dict`]'s snapshot; false when it does not fit.
    fn load_state_dict(&mut self, _state: &OptimizerState) -> bool {
        false
    }
}

/// Builds programs from the files of one language.
pub trait ProgramHost {
    /// The starting program: the training's own optimizer, unchanged.
    fn starting_files(&self) -> Files;
    /// Build `files` for `params`; `previous` is the running program, to carry
    /// state over from. Must not change `previous`.
    fn build(
        &mut self,
        files: &Files,
        params: &[ParamInfo],
        previous: Option<&dyn Program>,
    ) -> ApiResult<Box<dyn Program>>;
    /// The prompt section that tells the agent how to write a program here.
    fn interface(&self) -> String;
}

// ── RuleHost ────────────────────────────────────────────────────

/// `updater.json` programs over rlx-optim's algorithms.
pub struct RuleHost {
    start: Value,
    start_lr: f32,
}

pub const UPDATER: &str = "updater.json";

impl RuleHost {
    /// `optimizer` is the trainer's own optimizer config (a kind name or
    /// `{kind, lr, …}`, as `TrainerConfig::optimizer`); the starting program
    /// runs exactly that.
    pub fn new(optimizer_config: &Value) -> ApiResult<Self> {
        let start = match optimizer_config {
            Value::Null => json!({"kind": "adamw"}),
            Value::String(kind) => json!({ "kind": kind }),
            other => other.clone(),
        };
        let spec = optimizer(None, &start)?;
        Ok(Self {
            start,
            start_lr: spec.lr,
        })
    }
}

impl ProgramHost for RuleHost {
    fn starting_files(&self) -> Files {
        let program = json!({
            "groups": [{"names": ["*"], "optimizer": self.start}],
            "schedule": [[0.0, 1.0], [1.0, 1.0]],
        });
        let mut files = Files::new();
        files.insert(
            UPDATER.into(),
            serde_json::to_string_pretty(&program).unwrap() + "\n",
        );
        files
    }

    fn build(
        &mut self,
        files: &Files,
        params: &[ParamInfo],
        previous: Option<&dyn Program>,
    ) -> ApiResult<Box<dyn Program>> {
        let text = files
            .get(UPDATER)
            .ok_or_else(|| ApiError::type_(format!("{UPDATER} is missing")))?;
        let v: Value = serde_json::from_str(text)
            .map_err(|e| ApiError::type_(format!("{UPDATER} is not valid JSON: {e}")))?;
        let spec = RuleSpec::parse(&v)?;
        let previous = previous.and_then(|p| p.as_any().downcast_ref::<RuleProgram>());
        RuleProgram::build(spec, params, self.start_lr, previous)
            .map(|p| Box::new(p) as Box<dyn Program>)
    }

    fn interface(&self) -> String {
        format!(
            "Your program is {UPDATER}, in this folder:\n\
             {{\n  \"groups\": [{{\"names\": [\"<glob>\", …], \"ndim\": <int, optional>, \
             \"optimizer\": {{\"kind\": \"<algorithm>\", \"lr\": <float>, …}}, \
             \"lr_mult\": <float, default 1>, \"carry_state\": <bool, default true>}}, …],\n  \
             \"schedule\": [[<fraction of training>, <multiplier>], …]\n}}\n\
             - Every param goes to the first group whose `names` glob matches its name \
             (`*` any run of characters, `?` one) and whose `ndim`, if given, equals its \
             number of dimensions. Every param must match a group.\n\
             - `optimizer.kind` is one of: {}. Its options: lr, beta1, beta2, eps, \
             weight_decay, momentum, nesterov, f32_math, gamma, rho, muon_scale (each \
             algorithm reads the ones it has; an option it does not read is an error). \
             muon applies Muon to every tensor of its group; muon_adamw applies Muon to the \
             2-D matrices and AdamW to the rest.\n\
             - The learning rate of a group at an update is: its optimizer.lr × lr_mult × the \
             schedule multiplier at (update / total updates) × (the training script's current \
             learning rate / its starting learning rate {}). The schedule is linear between \
             its points; the first point is at 0.0, the last at 1.0, fractions increase.\n\
             - When a param stays in a group of the same kind as before, its optimizer state \
             (moments, momentum, step count) carries over unless carry_state is false; \
             otherwise it starts from zero.\n\
             - Optional \"skip_if\": {{\"loss_above\": <x>, \"grad_norm_above\": <x>}} skips an update \
             whose training loss (computed before the update) or total gradient norm is above the \
             bound, or not finite: the params and the optimizer state stay as they were. After 10 \
             skips in a row the update is made anyway and you are asked for a decision.\n\
             - Gradient clipping is done by the training script before your program runs.",
            crate::runtime::train::KINDS.join(", "),
            self.start_lr
        )
    }
}

/// A parsed `updater.json`.
#[derive(Clone, Debug)]
struct RuleSpec {
    groups: Vec<GroupSpec>,
    knots: Vec<(f64, f64)>,
    skip: SkipRule,
}

/// Skip an update whose loss or total gradient norm is above a bound.
#[derive(Clone, Copy, Debug, Default)]
struct SkipRule {
    loss_above: Option<f64>,
    grad_norm_above: Option<f64>,
}

impl SkipRule {
    fn parse(v: Option<&Value>) -> ApiResult<Self> {
        let Some(v) = v else {
            return Ok(Self::default());
        };
        let o = Opts::new("updater.json skip_if", v)?;
        let r = Self {
            loss_above: o.f64("loss_above")?,
            grad_norm_above: o.f64("grad_norm_above")?,
        };
        o.finish()?;
        Ok(r)
    }

    fn any(&self) -> bool {
        self.loss_above.is_some() || self.grad_norm_above.is_some()
    }

    fn hit(&self, items: &[OptItem<'_>], loss: Option<f32>) -> bool {
        if let (Some(b), Some(l)) = (self.loss_above, loss)
            && (!l.is_finite() || l as f64 > b)
        {
            return true;
        }
        self.grad_norm_above.is_some_and(|b| {
            let n = items
                .iter()
                .flat_map(|it| it.grad.iter())
                .map(|g| (*g as f64) * (*g as f64))
                .sum::<f64>()
                .sqrt();
            !n.is_finite() || n > b
        })
    }
}

#[derive(Clone, Debug)]
struct GroupSpec {
    names: Vec<String>,
    ndim: Option<usize>,
    optimizer: Value,
    lr_mult: f64,
    carry_state: bool,
}

impl RuleSpec {
    fn parse(v: &Value) -> ApiResult<Self> {
        let o = Opts::new(UPDATER, v)?;
        let groups = match o.get("groups") {
            Some(Value::Array(gs)) if !gs.is_empty() => gs
                .iter()
                .enumerate()
                .map(|(i, g)| GroupSpec::parse(i, g))
                .collect::<ApiResult<Vec<_>>>()?,
            _ => {
                return Err(ApiError::type_(format!(
                    "{UPDATER}: `groups` must be a non-empty list"
                )));
            }
        };
        let skip = SkipRule::parse(o.get("skip_if"))?;
        let knots = match o.get("schedule") {
            None => vec![(0.0, 1.0), (1.0, 1.0)],
            Some(Value::Array(ks)) => ks
                .iter()
                .map(|k| match k.as_array().map(|a| a.as_slice()) {
                    Some([f, m]) => match (f.as_f64(), m.as_f64()) {
                        (Some(f), Some(m)) => Ok((f, m)),
                        _ => Err(ApiError::type_(format!(
                            "{UPDATER}: schedule points are [fraction, multiplier] numbers, got {k}"
                        ))),
                    },
                    _ => Err(ApiError::type_(format!(
                        "{UPDATER}: schedule points are [fraction, multiplier], got {k}"
                    ))),
                })
                .collect::<ApiResult<Vec<_>>>()?,
            Some(other) => {
                return Err(ApiError::type_(format!(
                    "{UPDATER}: `schedule` must be a list of [fraction, multiplier], got {other}"
                )));
            }
        };
        o.finish()?;
        let bad = knots.len() < 2
            || knots[0].0 != 0.0
            || knots[knots.len() - 1].0 != 1.0
            || knots.windows(2).any(|w| w[1].0 <= w[0].0)
            || knots.iter().any(|(_, m)| !m.is_finite() || *m < 0.0);
        if bad {
            return Err(ApiError::range(format!(
                "{UPDATER}: the schedule needs points from fraction 0.0 to 1.0 in increasing \
                 order with finite non-negative multipliers; got {knots:?}"
            )));
        }
        Ok(Self {
            groups,
            knots,
            skip,
        })
    }

    fn multiplier(&self, fraction: f64) -> f64 {
        let k = &self.knots;
        let x = fraction.clamp(0.0, 1.0);
        let i = k
            .windows(2)
            .position(|w| x <= w[1].0)
            .unwrap_or(k.len() - 2);
        let ((x0, y0), (x1, y1)) = (k[i], k[i + 1]);
        if y0 == y1 {
            return y0;
        }
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }
}

impl GroupSpec {
    fn parse(i: usize, v: &Value) -> ApiResult<Self> {
        let what = format!("{UPDATER} group {i}");
        let o = Opts::new(&what, v)?;
        let names = match o.get("names") {
            Some(Value::Array(ns)) if !ns.is_empty() => ns
                .iter()
                .map(|n| {
                    n.as_str().map(str::to_string).ok_or_else(|| {
                        ApiError::type_(format!("{what}: names are strings, got {n}"))
                    })
                })
                .collect::<ApiResult<Vec<_>>>()?,
            Some(Value::String(n)) => vec![n.clone()],
            _ => {
                return Err(ApiError::type_(format!(
                    "{what}: `names` must be a non-empty list of globs"
                )));
            }
        };
        let ndim = o.u64("ndim")?.map(|n| n as usize);
        let optimizer = o
            .get("optimizer")
            .cloned()
            .ok_or_else(|| ApiError::type_(format!("{what}: `optimizer` is required")))?;
        let lr_mult = o.f64("lr_mult")?.unwrap_or(1.0);
        if !lr_mult.is_finite() || lr_mult < 0.0 {
            return Err(ApiError::range(format!(
                "{what}: lr_mult must be finite and >= 0, got {lr_mult}"
            )));
        }
        let carry_state = o.bool("carry_state", true)?;
        o.finish()?;
        Ok(Self {
            names,
            ndim,
            optimizer,
            lr_mult,
            carry_state,
        })
    }

    fn matches(&self, p: &ParamInfo) -> bool {
        self.ndim.is_none_or(|n| n == p.shape.len()) && self.names.iter().any(|g| glob(g, &p.name))
    }
}

/// `*` matches any run of characters, `?` exactly one.
pub fn glob(pattern: &str, name: &str) -> bool {
    let (p, n): (Vec<char>, Vec<char>) = (pattern.chars().collect(), name.chars().collect());
    let (mut pi, mut ni, mut star, mut mark) = (0, 0, None, 0);
    while ni < n.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == n[ni]) {
            pi += 1;
            ni += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ni;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ni = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

struct Group {
    kind: String,
    optimizer: Box<dyn OptimizerTrait>,
    /// optimizer.lr ÷ the starting program's lr: the factor on the training
    /// script's learning rate.
    lr_ratio: f64,
    lr_mult: f64,
    /// The rate given to the optimizer last; it starts at optimizer.lr.
    lr_set: f32,
    members: Vec<String>,
}

/// A built `updater.json`.
pub struct RuleProgram {
    spec: RuleSpec,
    groups: Vec<Group>,
    group_of: HashMap<String, usize>,
    start_lr: f32,
}

impl RuleProgram {
    fn build(
        spec: RuleSpec,
        params: &[ParamInfo],
        start_lr: f32,
        previous: Option<&RuleProgram>,
    ) -> ApiResult<Self> {
        let mut group_of = HashMap::with_capacity(params.len());
        let mut members: Vec<Vec<String>> = vec![Vec::new(); spec.groups.len()];
        let mut unmatched = Vec::new();
        for p in params {
            match spec.groups.iter().position(|g| g.matches(p)) {
                Some(i) => {
                    group_of.insert(p.name.clone(), i);
                    members[i].push(p.name.clone());
                }
                None => unmatched.push(p.name.as_str()),
            }
        }
        if !unmatched.is_empty() {
            return Err(ApiError::type_(format!(
                "{UPDATER}: no group matches {}; add a group (for example names [\"*\"] last)",
                unmatched.join(", ")
            )));
        }
        let mut groups = Vec::with_capacity(spec.groups.len());
        for (i, (g, names)) in spec.groups.iter().zip(members).enumerate() {
            let built = optimizer(None, &g.optimizer).map_err(|e| {
                ApiError::new(e.kind, format!("{UPDATER} group {i}: {}", e.message))
            })?;
            let mut opt = built.optimizer;
            if g.carry_state
                && let Some(prev) = previous
                && let Some(state) = prev.carried_state(&built.kind, &names)
                && !opt.load_state_dict(&state)
            {
                return Err(ApiError::runtime(format!(
                    "{UPDATER} group {i}: {} would not take the state carried over; set \
                     carry_state false to start it from zero",
                    built.kind
                )));
            }
            groups.push(Group {
                kind: built.kind,
                optimizer: opt,
                lr_ratio: built.lr as f64 / start_lr as f64,
                lr_mult: g.lr_mult,
                lr_set: built.lr,
                members: names,
            });
        }
        Ok(Self {
            spec,
            groups,
            group_of,
            start_lr,
        })
    }

    /// The state the params `names` had in this program's groups of `kind`,
    /// keyed for an optimizer of that kind.
    fn carried_state(&self, kind: &str, names: &[String]) -> Option<OptimizerState> {
        let mut out = OptimizerState::default();
        let mut found = false;
        for g in self.groups.iter().filter(|g| g.kind == kind) {
            if !names.iter().any(|n| g.members.contains(n)) {
                continue;
            }
            let state = g.optimizer.state_dict()?;
            out.step = out.step.max(state.step);
            for (key, values) in state.buffers {
                let param = key.split_once('/').map(|(_, p)| p).unwrap_or("");
                if names.iter().any(|n| n == param) {
                    out.buffers.push((key, values));
                    found = true;
                }
            }
        }
        found.then_some(out)
    }
}

impl Program for RuleProgram {
    fn step(&mut self, items: &mut [OptItem<'_>], ctx: &StepCtx) -> ApiResult<bool> {
        let skip = self.spec.skip.hit(items, ctx.loss);
        let fraction = ctx.update as f64 / ctx.total.max(1) as f64;
        let schedule = self.spec.multiplier(fraction);
        let base = ctx.base_lr.unwrap_or(self.start_lr) as f64;
        let mut per: Vec<Vec<OptItem<'_>>> = self.groups.iter().map(|_| Vec::new()).collect();
        for it in items.iter_mut() {
            let i = *self.group_of.get(it.name).ok_or_else(|| {
                ApiError::runtime(format!(
                    "{UPDATER}: param '{}' was not in the program's build",
                    it.name
                ))
            })?;
            per[i].push(OptItem {
                name: it.name,
                shape: it.shape,
                param: &mut *it.param,
                grad: it.grad,
            });
        }
        for (g, mut batch) in self.groups.iter_mut().zip(per) {
            // f64 then f32, so the starting program (every factor exactly 1)
            // gives its optimizer exactly the rate the plain trainer would.
            let lr = (base * g.lr_ratio * g.lr_mult * schedule) as f32;
            if lr != g.lr_set {
                g.optimizer.set_lr(lr);
                g.lr_set = lr;
            }
            g.optimizer.step_batch(&mut batch);
            g.optimizer.end_iteration();
        }
        Ok(skip)
    }

    fn can_skip(&self) -> bool {
        self.spec.skip.any()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Every group's state, keys prefixed `g<i>:`; `None` when any group's
    /// algorithm keeps state it cannot save.
    fn state_dict(&self) -> Option<OptimizerState> {
        let mut out = OptimizerState::default();
        for (i, g) in self.groups.iter().enumerate() {
            let s = g.optimizer.state_dict()?;
            out.step = out.step.max(s.step);
            out.buffers
                .extend(s.buffers.into_iter().map(|(k, v)| (format!("g{i}:{k}"), v)));
        }
        Some(out)
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> bool {
        for (i, g) in self.groups.iter_mut().enumerate() {
            let prefix = format!("g{i}:");
            let own = OptimizerState {
                step: state.step,
                buffers: state
                    .buffers
                    .iter()
                    .filter_map(|(k, v)| {
                        k.strip_prefix(&prefix).map(|k| (k.to_string(), v.clone()))
                    })
                    .collect(),
            };
            if !g.optimizer.load_state_dict(&own) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches_runs_and_single_characters() {
        assert!(glob("*", "layers.0.attn.q"));
        assert!(glob("layers.*.attn.*", "layers.10.attn.q"));
        assert!(glob("w?", "w1") && !glob("w?", "w10"));
        assert!(!glob("*.bias", "layers.0.weight"));
        assert!(glob("*bias*", "bias"));
    }

    #[test]
    fn schedule_is_linear_between_points() {
        let s = RuleSpec::parse(&json!({
            "groups": [{"names": ["*"], "optimizer": "sgd"}],
            "schedule": [[0.0, 0.0], [0.1, 1.0], [1.0, 0.1]],
        }))
        .unwrap();
        assert_eq!(s.multiplier(0.0), 0.0);
        assert!((s.multiplier(0.05) - 0.5).abs() < 1e-12);
        assert_eq!(s.multiplier(0.1), 1.0);
        assert!((s.multiplier(1.0) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn bad_programs_say_what_is_wrong() {
        let err = |v: Value| RuleSpec::parse(&v).unwrap_err().message;
        assert!(err(json!({"groups": []})).contains("non-empty"));
        assert!(err(json!({"groups": [{"names": ["*"]}]})).contains("`optimizer` is required"));
        assert!(
            err(json!({"groups": [{"names": ["*"], "optimizer": "sgd", "lr": 1}]}))
                .contains("'lr'")
        );
        assert!(
            err(json!({"groups": [{"names": ["*"], "optimizer": "sgd"}],
                       "schedule": [[0.0, 1.0], [0.5, 1.0]]}))
            .contains("0.0 to 1.0")
        );
    }
}
