// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`WeightSpace`] — the network's weights as an editable, checkpointed store,
//! plus the forward-pass budget that is the agent's only way to *run* them.
//!
//! Reading and editing weights is free. Running the network costs: one
//! [`WeightSpace::probe`] (up to [`BATCH`] hand-written observations) or one
//! [`WeightSpace::rollout`] (a batch of episodes) spends one forward pass.
//! [`WeightSpace::evaluate`] is the harness's held-out score and is never
//! exposed as a tool.

use crate::env::{Env, EnvKind, EnvSpec};
use crate::mlp::{Act, MlpSpec};
use rlx_ir::Philox4x32;
use rlx_runtime::{CompiledGraph, Device, Session};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// Rows per network execution; probes and rollouts are padded to it.
pub const BATCH: usize = 32;

/// Held-out evaluation seed; rollouts the agent requests never use it.
pub const SCORE_SEED: u64 = 0xE7A1_5C0E;

/// A row-major `[rows, cols]` weight tensor (biases have `rows == 1`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
}

/// Everything that persists between tool calls (the CLI's `--state` file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub env: EnvKind,
    pub mlp: MlpSpec,
    pub params: BTreeMap<String, Tensor>,
    pub checkpoints: BTreeMap<String, BTreeMap<String, Tensor>>,
    pub budget: usize,
    pub used: usize,
    /// Seeds successive rollouts so repeated calls see fresh episodes.
    pub rollout_counter: u64,
    /// One line per edit / run, newest last.
    pub log: Vec<String>,
    /// [`EnvKind::Graph`]: the network as `.rlxmodel` text (`mlp` unused).
    #[serde(default)]
    pub graph: Option<String>,
    /// [`EnvKind::Graph`]: the task as told to the agent (no harness score).
    #[serde(default)]
    pub goal: Option<String>,
    /// How actions are chosen in rollouts and scoring.
    #[serde(default)]
    pub protocol: Protocol,
    /// Reading / audit tasks: which candidate is true (never shown by a tool).
    #[serde(default)]
    pub hidden: Option<usize>,
    /// The agent's submitted answer (`answer` tool), last one wins.
    #[serde(default)]
    pub answer: Option<Value>,
    /// Further networks of a multi-network task (pair audits), by name; the
    /// active one lives in `params` and is listed in `active`.
    #[serde(default)]
    pub networks: BTreeMap<String, BTreeMap<String, Tensor>>,
    #[serde(default)]
    pub active: Option<String>,
    /// When set, only these tools are offered and accepted (one-reply mode).
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Internals-only access (the post's relaxed sycophancy setting): `trace`
    /// shows internal activations but never the outputs; probe, eval,
    /// rollout, compare and ablate are refused.
    #[serde(default)]
    pub internals_only: bool,
    /// Reading tasks without a candidate list: free-text answers, scored by
    /// a rubric.
    #[serde(default)]
    pub open: bool,
}

/// How a policy's logits become actions. The post scores with sampled
/// actions and sticky actions (p = 0.1); the default here is greedy.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Protocol {
    /// Sample from softmax(logits) instead of taking the argmax.
    #[serde(default)]
    pub sample: bool,
    /// Probability of repeating the previous action regardless of the policy.
    #[serde(default)]
    pub sticky: f32,
}

impl Protocol {
    pub const GREEDY: Protocol = Protocol {
        sample: false,
        sticky: 0.0,
    };
    /// The post's scoring: sampled actions, sticky actions with p = 0.1.
    pub const STOCHASTIC: Protocol = Protocol {
        sample: true,
        sticky: 0.1,
    };

    /// The action for `logits`, given the previous one. Draws only from
    /// `rng` (a stream separate from the environment's), and none when greedy.
    pub fn choose(&self, logits: &[f32], prev: Option<usize>, rng: &mut Philox4x32) -> usize {
        if let Some(p) = prev
            && self.sticky > 0.0
            && rng.next_f32() < self.sticky
        {
            return p;
        }
        if !self.sample {
            return argmax(logits);
        }
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
        let total: f32 = e.iter().sum();
        let mut u = rng.next_f32() * total;
        for (i, w) in e.iter().enumerate() {
            if u < *w {
                return i;
            }
            u -= w;
        }
        e.len() - 1
    }

    pub fn describe(&self) -> String {
        match (self.sample, self.sticky > 0.0) {
            (false, false) => "greedy: action = argmax(logits), ties -> lowest index".into(),
            (true, false) => "sampled: action ~ softmax(logits)".into(),
            (false, true) => format!(
                "greedy with sticky actions: with probability {} the previous action repeats",
                self.sticky
            ),
            (true, true) => format!(
                "sampled: action ~ softmax(logits); with probability {} the previous action \
                 repeats instead (sticky actions)",
                self.sticky
            ),
        }
    }
}

/// Seed of the action-choice stream for an episode-batch seed.
fn policy_seed(seed: u64) -> u64 {
    seed ^ 0xAC71_0C0D_E5A1_0000
}

/// Per-rollout summary.
#[derive(Debug, Clone, Serialize)]
pub struct RolloutReport {
    pub episodes: usize,
    pub mean_return: f32,
    /// Count of positive-reward events (Bounce: catches).
    pub positive_events: usize,
    /// Count of negative-reward events (Bounce: misses).
    pub negative_events: usize,
    /// Fraction of each action over all steps, in `EnvSpec::actions` order.
    pub action_mix: Vec<f32>,
    /// First `trace_steps` steps of episode 0.
    pub trace: Vec<String>,
    /// Return of every episode, in play order (for paired comparisons).
    #[serde(skip)]
    pub episode_returns: Vec<f32>,
}

/// One probed observation.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeRow {
    pub action: String,
    pub logits: Vec<f32>,
    /// Hidden activations, one vector per hidden layer.
    pub hidden: Vec<Vec<f32>>,
}

pub struct WeightSpace {
    pub state: State,
    spec: EnvSpec,
    compiled: Option<CompiledGraph>,
    dirty: BTreeSet<String>,
}

impl WeightSpace {
    /// Fresh space: every weight zero for control tasks, the tampered trained
    /// model for `mnist-fix` (whose architecture is fixed; `hidden`/`act` are
    /// ignored there). Panics if the MNIST data is missing — use
    /// [`Self::try_new`] to get the error instead.
    pub fn new(env: EnvKind, hidden: &[usize], act: Act, budget: usize) -> Self {
        Self::try_new(env, hidden, act, budget).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_new(
        env: EnvKind,
        hidden: &[usize],
        act: Act,
        budget: usize,
    ) -> Result<Self, String> {
        Self::for_task(env, hidden, act, budget, 0)
    }

    /// [`Self::try_new`] with a task seed: reading / audit tasks draw their
    /// hidden answer from it.
    pub fn for_task(
        env: EnvKind,
        hidden: &[usize],
        act: Act,
        budget: usize,
        task_seed: u64,
    ) -> Result<Self, String> {
        if let Some(st) = crate::tasks::setup(env, task_seed, budget)? {
            return Ok(Self::from_state(st));
        }
        if let Some(trigger) = env.trigger() {
            let mlp = MlpSpec {
                widths: crate::mnist::widths(),
                act: Act::Relu,
            };
            return Ok(Self::from_state(State {
                env,
                mlp,
                params: crate::mnist::trained(Some(trigger))?,
                checkpoints: BTreeMap::new(),
                budget,
                used: 0,
                rollout_counter: 0,
                log: Vec::new(),
                graph: None,
                goal: None,
                protocol: Protocol::GREEDY,
                hidden: None,
                answer: None,
                networks: BTreeMap::new(),
                active: None,
                tools: None,
                internals_only: false,
                open: false,
            }));
        }
        let spec = env.make().spec();
        let mut widths = vec![spec.obs_dim];
        widths.extend_from_slice(hidden);
        widths.push(spec.actions.len());
        let mlp = MlpSpec { widths, act };
        let params = mlp
            .params()
            .into_iter()
            .map(|(n, r, c)| (n, Tensor::zeros(r, c)))
            .collect();
        Ok(Self::from_state(State {
            env,
            mlp,
            params,
            checkpoints: BTreeMap::new(),
            budget,
            used: 0,
            rollout_counter: 0,
            log: Vec::new(),
            graph: None,
            goal: None,
            protocol: Protocol::GREEDY,
            hidden: None,
            answer: None,
            networks: BTreeMap::new(),
            active: None,
            tools: None,
            internals_only: false,
            open: false,
        }))
    }

    /// A space over any rlx graph (`.rlxmodel` text). Every `Op::Param` must
    /// be static-shaped f32; it starts from `weights` where given (flat,
    /// row-major) and zero otherwise. Viewed as `[prod(leading dims), last
    /// dim]`, labelled `rI` / `cJ`. There is no harness score: `goal` is what
    /// the agent is told.
    pub fn from_graph(
        text: String,
        weights: BTreeMap<String, Vec<f32>>,
        budget: usize,
        goal: Option<String>,
    ) -> Result<Self, String> {
        let g = rlx_ir::rlxmodel::from_text(&text).map_err(|e| e.to_string())?;
        let mut params = BTreeMap::new();
        for n in g.nodes() {
            if let rlx_ir::Op::Param { name } = &n.op {
                if n.shape.dtype() != rlx_ir::DType::F32 {
                    return Err(format!(
                        "param `{name}` is {:?}; only f32 is editable",
                        n.shape.dtype()
                    ));
                }
                let dims: Vec<usize> = n
                    .shape
                    .dims()
                    .iter()
                    .map(|d| match d {
                        rlx_ir::Dim::Static(v) => Ok(*v),
                        other => Err(format!("param `{name}` has a non-static dim {other:?}")),
                    })
                    .collect::<Result<_, _>>()?;
                let cols = dims.last().copied().unwrap_or(1);
                let rows = dims.iter().product::<usize>() / cols.max(1);
                let mut t = Tensor::zeros(rows, cols);
                if let Some(w) = weights.get(name) {
                    if w.len() != t.data.len() {
                        return Err(format!(
                            "weights for `{name}`: {} values, shape needs {}",
                            w.len(),
                            t.data.len()
                        ));
                    }
                    t.data.clone_from(w);
                }
                params.insert(name.clone(), t);
            }
        }
        if let Some(k) = weights.keys().find(|k| !params.contains_key(*k)) {
            return Err(format!(
                "weights name `{k}`, which is not a param of the graph"
            ));
        }
        Ok(Self::from_state(State {
            env: EnvKind::Graph,
            mlp: MlpSpec {
                widths: vec![0, 0],
                act: Act::Tanh,
            },
            params,
            checkpoints: BTreeMap::new(),
            budget,
            used: 0,
            rollout_counter: 0,
            log: Vec::new(),
            graph: Some(text),
            goal,
            protocol: Protocol::GREEDY,
            hidden: None,
            answer: None,
            networks: BTreeMap::new(),
            active: None,
            tools: None,
            internals_only: false,
            open: false,
        }))
    }

    pub fn from_state(state: State) -> Self {
        let spec = state.env.make().spec();
        let dirty = state.params.keys().cloned().collect();
        Self {
            state,
            spec,
            compiled: None,
            dirty,
        }
    }

    pub fn spec(&self) -> &EnvSpec {
        &self.spec
    }

    pub fn remaining(&self) -> usize {
        self.state.budget.saturating_sub(self.state.used)
    }

    pub fn param(&self, name: &str) -> Result<&Tensor, String> {
        self.state.params.get(name).ok_or_else(|| {
            let known: Vec<_> = self.state.params.keys().map(String::as_str).collect();
            format!("unknown param `{name}` (have: {})", known.join(", "))
        })
    }

    /// Row labels for `name`: observation labels for the input layer, unit
    /// names (`l{k}.h{i}`) for deeper layers, `bias` for biases.
    pub fn row_labels(&self, name: &str) -> Result<Vec<String>, String> {
        let t = self.param(name)?;
        if self.state.graph.is_some() {
            return Ok((0..t.rows).map(|i| format!("r{i}")).collect());
        }
        if name.ends_with(".b") {
            return Ok(vec!["bias".into()]);
        }
        let layer = layer_index(name)?;
        Ok(if layer == 0 {
            self.spec.obs_labels.clone()
        } else {
            (0..t.rows)
                .map(|i| format!("l{}.h{i}", layer - 1))
                .collect()
        })
    }

    /// Column labels: action names for the last layer, unit names otherwise.
    pub fn col_labels(&self, name: &str) -> Result<Vec<String>, String> {
        let t = self.param(name)?;
        if self.state.graph.is_some() {
            return Ok((0..t.cols).map(|i| format!("c{i}")).collect());
        }
        let layer = layer_index(name)?;
        Ok(if layer + 1 == self.state.mlp.n_layers() {
            self.spec.actions.iter().map(|s| s.to_string()).collect()
        } else {
            (0..t.cols).map(|i| format!("l{layer}.h{i}")).collect()
        })
    }

    /// Resolve a row given as an index or a label.
    pub fn resolve_row(&self, name: &str, key: &RowKey) -> Result<usize, String> {
        let t = self.param(name)?;
        match key {
            RowKey::Index(i) if *i < t.rows => Ok(*i),
            RowKey::Index(i) => Err(format!("{name}: row {i} out of range (rows={})", t.rows)),
            RowKey::Label(l) => self
                .row_labels(name)?
                .iter()
                .position(|x| x == l)
                .ok_or_else(|| format!("{name}: no row labelled `{l}`")),
        }
    }

    /// Resolve a column given as an index or a label.
    pub fn resolve_col(&self, name: &str, key: &RowKey) -> Result<usize, String> {
        let t = self.param(name)?;
        match key {
            RowKey::Index(i) if *i < t.cols => Ok(*i),
            RowKey::Index(i) => Err(format!("{name}: col {i} out of range (cols={})", t.cols)),
            RowKey::Label(l) => self
                .col_labels(name)?
                .iter()
                .position(|x| x == l)
                .ok_or_else(|| format!("{name}: no column labelled `{l}`")),
        }
    }

    // ── Edits (free) ────────────────────────────────────────────────────

    /// Mutable access with dirty tracking; `note` goes to the log.
    pub fn edit<R>(
        &mut self,
        name: &str,
        note: &str,
        f: impl FnOnce(&mut Tensor) -> Result<R, String>,
    ) -> Result<R, String> {
        let t = self
            .state
            .params
            .get_mut(name)
            .ok_or_else(|| format!("unknown param `{name}`"))?;
        let r = f(t)?;
        if t.data.iter().any(|v| !v.is_finite()) {
            return Err(format!("{name}: edit produced a non-finite value"));
        }
        self.dirty.insert(name.to_string());
        self.state.log.push(format!("edit {name}: {note}"));
        Ok(r)
    }

    /// Replace every weight (harness use: scoring variants, warm starts).
    /// Change the architecture (harness / `insert_layer`): new widths and
    /// params; the compiled graph is rebuilt on the next run.
    pub fn set_architecture(&mut self, mlp: MlpSpec, params: BTreeMap<String, Tensor>) {
        self.state.mlp = mlp;
        self.compiled = None;
        self.set_params(params);
    }

    pub fn set_params(&mut self, params: BTreeMap<String, Tensor>) {
        self.state.params = params;
        self.dirty = self.state.params.keys().cloned().collect();
    }

    pub fn checkpoint(&mut self, tag: &str) {
        self.state
            .checkpoints
            .insert(tag.to_string(), self.state.params.clone());
        self.state.log.push(format!("checkpoint {tag}"));
    }

    pub fn revert(&mut self, tag: &str) -> Result<(), String> {
        let snap = self
            .state
            .checkpoints
            .get(tag)
            .ok_or_else(|| format!("no checkpoint `{tag}`"))?
            .clone();
        self.state.params = snap;
        self.dirty = self.state.params.keys().cloned().collect();
        self.state.log.push(format!("revert {tag}"));
        Ok(())
    }

    /// Per-param `(changed entries, L2 of the change)` against a checkpoint.
    pub fn diff(&self, tag: &str) -> Result<Vec<(String, usize, f32)>, String> {
        let snap = self
            .state
            .checkpoints
            .get(tag)
            .ok_or_else(|| format!("no checkpoint `{tag}`"))?;
        Ok(self
            .state
            .params
            .iter()
            .filter_map(|(n, t)| {
                let old = &snap[n];
                let (mut k, mut l2) = (0usize, 0f32);
                for (a, b) in t.data.iter().zip(&old.data) {
                    if a != b {
                        k += 1;
                        l2 += (a - b) * (a - b);
                    }
                }
                (k > 0).then(|| (n.clone(), k, l2.sqrt()))
            })
            .collect())
    }

    // ── Forward passes (budgeted) ───────────────────────────────────────

    fn charge(&mut self, what: &str) -> Result<(), String> {
        if self.state.used >= self.state.budget {
            return Err(format!(
                "forward-pass budget exhausted ({} of {} used)",
                self.state.used, self.state.budget
            ));
        }
        self.state.used += 1;
        self.state.log.push(format!(
            "forward {}/{}: {what}",
            self.state.used, self.state.budget
        ));
        Ok(())
    }

    /// One network execution over exactly `BATCH` rows.
    fn run(&mut self, obs: &[f32]) -> Vec<Vec<f32>> {
        debug_assert!(
            self.state.graph.is_none(),
            "graph spaces run through `trace`"
        );
        debug_assert_eq!(obs.len(), BATCH * self.spec.obs_dim);
        let c = self.compiled.get_or_insert_with(|| {
            Session::new(Device::Cpu).compile(self.state.mlp.build_graph(BATCH))
        });
        for name in std::mem::take(&mut self.dirty) {
            c.set_param(&name, &self.state.params[&name].data);
        }
        c.run(&[("obs", obs)])
    }

    /// Run the network on hand-written observations (≤ `BATCH` rows).
    pub fn probe(&mut self, rows: &[Vec<f32>]) -> Result<Vec<ProbeRow>, String> {
        self.no_outputs("probe")?;
        if self.state.graph.is_some() {
            return Err("graph spaces run through `trace` (named inputs)".into());
        }
        let d = self.spec.obs_dim;
        if rows.is_empty() || rows.len() > BATCH {
            return Err(format!("probe takes 1..={BATCH} observations"));
        }
        if let Some(r) = rows.iter().find(|r| r.len() != d) {
            return Err(format!("observation has {} values, expected {d}", r.len()));
        }
        self.charge(&format!("probe {} rows", rows.len()))?;
        let mut obs = vec![0f32; BATCH * d];
        for (i, r) in rows.iter().enumerate() {
            obs[i * d..(i + 1) * d].copy_from_slice(r);
        }
        let outs = self.run(&obs);
        let widths = &self.state.mlp.widths;
        let na = self.spec.actions.len();
        Ok((0..rows.len())
            .map(|i| {
                let logits = outs[0][i * na..(i + 1) * na].to_vec();
                let hidden = (1..outs.len())
                    .map(|k| {
                        let w = widths[k];
                        outs[k][i * w..(i + 1) * w].to_vec()
                    })
                    .collect();
                ProbeRow {
                    action: self.spec.actions[argmax(&logits)].to_string(),
                    logits,
                    hidden,
                }
            })
            .collect())
    }

    /// Play `episodes` episodes (rounded up to a multiple of `BATCH`) with
    /// the greedy policy; costs one forward pass.
    pub fn rollout(
        &mut self,
        episodes: usize,
        seed: Option<u64>,
        trace_steps: usize,
    ) -> Result<RolloutReport, String> {
        self.no_outputs("rollout")?;
        if self.state.env.is_classifier()
            || self.state.graph.is_some()
            || self.state.env == EnvKind::ReadRules
        {
            return Err("this task has no episodes; use `eval` / `probe` / `trace`".into());
        }
        let seed = seed.unwrap_or_else(|| {
            self.state.rollout_counter += 1;
            0x5EED_0000 + self.state.rollout_counter
        });
        self.charge(&format!("rollout {episodes} episodes seed {seed}"))?;
        Ok(self.play(episodes, seed, trace_steps))
    }

    /// Held-out score for the harness. Not budgeted, not a tool.
    pub fn evaluate(&mut self, episodes: usize, seed: u64) -> RolloutReport {
        self.play(episodes, seed, 0)
    }

    /// Predicted class for each image (rows of `obs_dim` values).
    pub fn classify(&mut self, images: &[f32]) -> Vec<usize> {
        let d = self.spec.obs_dim;
        let na = self.spec.actions.len();
        let mut out = Vec::with_capacity(images.len() / d);
        let mut batch = vec![0f32; BATCH * d];
        for chunk in images.chunks(BATCH * d) {
            batch.fill(0.0);
            batch[..chunk.len()].copy_from_slice(chunk);
            let logits = self.run(&batch).swap_remove(0);
            for i in 0..chunk.len() / d {
                out.push(argmax(&logits[i * na..(i + 1) * na]));
            }
        }
        out
    }

    /// Validation accuracy and confusion matrix (`[true][predicted]`) on the
    /// first `n` validation digits; costs one forward pass.
    pub fn eval(&mut self, n: usize) -> Result<Value, String> {
        self.no_outputs("eval")?;
        if !self.state.env.is_classifier() {
            return Err("`eval` is for classification tasks; use `rollout`".into());
        }
        let data = crate::mnist::data()?;
        let n = n.clamp(1, data.val.len());
        self.charge(&format!("eval {n} validation digits"))?;
        let preds = self.classify(&data.val.images[..n * crate::mnist::PIXELS]);
        let mut confusion = vec![vec![0usize; crate::mnist::CLASSES]; crate::mnist::CLASSES];
        for (p, &l) in preds.iter().zip(&data.val.labels) {
            confusion[l as usize][*p] += 1;
        }
        let correct: usize = (0..crate::mnist::CLASSES).map(|c| confusion[c][c]).sum();
        let per_class: Vec<f32> = confusion
            .iter()
            .enumerate()
            .map(|(c, row)| row[c] as f32 / row.iter().sum::<usize>().max(1) as f32)
            .collect();
        Ok(serde_json::json!({
            "digits": n,
            "accuracy": correct as f32 / n as f32,
            "per_class_accuracy": per_class,
            "confusion_true_by_predicted": confusion,
        }))
    }

    /// The harness's held-out score (not budgeted, not a tool). `score` is
    /// the headline number: mean return for control tasks; clean accuracy
    /// minus attack success, in percentage points, for `mnist-fix`.
    pub fn held_out(&mut self, episodes: usize) -> Result<Value, String> {
        if self.state.env.is_reading() {
            return Ok(crate::tasks::answer_score(self));
        }
        if self.state.env == EnvKind::MnistSteer {
            return crate::tasks::steer_score(self);
        }
        if !self.state.env.is_classifier() {
            let r = self.evaluate(episodes, SCORE_SEED);
            let n = r.episodes as f32;
            return Ok(serde_json::json!({
                "score": r.mean_return,
                "detail": format!("+{:.2} / -{:.2} per episode",
                    r.positive_events as f32 / n, r.negative_events as f32 / n),
                "episodes": r.episodes,
                "mean_return": r.mean_return,
                "positive_per_episode": r.positive_events as f32 / n,
                "negative_per_episode": r.negative_events as f32 / n,
            }));
        }
        use crate::mnist::{PIXELS, TARGET, data};
        let trigger = self
            .state
            .env
            .trigger()
            .expect("classifier tasks have a trigger");
        let test = &data()?.test;
        let clean = self.classify(&test.images);
        let correct = clean
            .iter()
            .zip(&test.labels)
            .filter(|(p, l)| **p == **l as usize)
            .count();
        let mut stamped = Vec::new();
        for i in 0..test.len() {
            if test.labels[i] as usize != TARGET {
                let mut img = test.image(i).to_vec();
                trigger.stamp(&mut img);
                stamped.extend(img);
            }
        }
        let hits = self
            .classify(&stamped)
            .iter()
            .filter(|&&p| p == TARGET)
            .count();
        let acc = correct as f32 / test.len() as f32;
        let asr = hits as f32 / (stamped.len() / PIXELS) as f32;
        Ok(serde_json::json!({
            "score": 100.0 * (acc - asr),
            "detail": format!("clean acc {:.2}%, attack success {:.2}%", 100.0 * acc, 100.0 * asr),
            "clean_accuracy": acc,
            "attack_success_rate": asr,
            "test_digits": test.len(),
        }))
    }

    /// The network as an rlx graph with named nodes: the loaded graph, or
    /// the MLP over `BATCH` rows (`l{k}.z` pre-activation, `l{k}.h`
    /// activation, `logits`).
    pub fn graph(&self) -> Result<rlx_ir::Graph, String> {
        match &self.state.graph {
            Some(text) => rlx_ir::rlxmodel::from_text(text).map_err(|e| e.to_string()),
            None => Ok(self.state.mlp.build_graph(BATCH)),
        }
    }

    /// Named, non-parameter nodes (what `trace` can return).
    pub fn nodes(&self) -> Result<Vec<Value>, String> {
        let g = self.graph()?;
        Ok(g.nodes()
            .iter()
            .filter(|n| !matches!(n.op, rlx_ir::Op::Param { .. } | rlx_ir::Op::Constant { .. }))
            .map(|n| {
                let op = format!("{:?}", n.op);
                serde_json::json!({
                    "id": n.id.0,
                    "name": n.name,
                    "op": op.split(['(', ' ', '{']).next().unwrap_or(""),
                    "shape": n.shape.dims().iter().map(|d| format!("{d:?}")).collect::<Vec<_>>(),
                    "inputs": n.inputs.iter().map(|i| i.0).collect::<Vec<_>>(),
                })
            })
            .collect())
    }

    /// Run once with `inputs` and return the graph outputs plus the nodes in
    /// `extra` (by name or `%id`), each as `(label, dims, values)`. MLP
    /// spaces take one observation as input `obs` and report its row only.
    /// Costs one forward pass.
    pub fn trace(
        &mut self,
        inputs: &[(String, Vec<f32>)],
        extra: &[String],
    ) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>, String> {
        let mut g = self.graph()?;
        let mlp_space = self.state.graph.is_none();
        let find = |g: &rlx_ir::Graph, key: &str| -> Result<rlx_ir::NodeId, String> {
            if let Some(id) = key.strip_prefix('%').and_then(|s| s.parse::<u32>().ok()) {
                if (id as usize) < g.nodes().len() {
                    return Ok(rlx_ir::NodeId(id));
                }
            }
            g.nodes()
                .iter()
                .find(|n| n.name.as_deref() == Some(key))
                .map(|n| n.id)
                .ok_or_else(|| format!("no node named `{key}` (see `list_nodes`)"))
        };
        let base = g.outputs.clone();
        // Internals-only: the network's outputs are never shown.
        let hidden_outputs: Vec<rlx_ir::NodeId> = if !self.state.internals_only {
            Vec::new()
        } else if mlp_space {
            base.first().copied().into_iter().collect()
        } else {
            base.clone()
        };
        let mut outs = base.clone();
        let mut labels: Vec<String> = base
            .iter()
            .enumerate()
            .map(|(i, id)| {
                g.node(*id)
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("output{i}"))
            })
            .collect();
        for key in extra {
            let id = find(&g, key)?;
            if hidden_outputs.contains(&id) {
                return Err(format!(
                    "`{key}` is an output; this task shows internals only"
                ));
            }
            outs.push(id);
            labels.push(key.clone());
        }
        g.set_outputs(outs.clone());
        let mlp = self.state.graph.is_none();
        let mut feeds: Vec<(String, Vec<f32>)> = Vec::new();
        for n in g.nodes() {
            if let rlx_ir::Op::Input { name } = &n.op {
                let numel: usize = n.shape.dims().iter().map(|d| d.unwrap_static()).product();
                let given = inputs
                    .iter()
                    .find(|(k, _)| k == name)
                    .map(|(_, v)| v.clone())
                    .ok_or_else(|| format!("missing input `{name}`"))?;
                let v = if mlp {
                    if given.len() != self.spec.obs_dim {
                        return Err(format!(
                            "`{name}` needs {} values, got {}",
                            self.spec.obs_dim,
                            given.len()
                        ));
                    }
                    let mut v = vec![0f32; numel];
                    v[..given.len()].copy_from_slice(&given);
                    v
                } else if given.len() == numel {
                    given
                } else {
                    return Err(format!(
                        "`{name}` needs {numel} values, got {}",
                        given.len()
                    ));
                };
                feeds.push((name.clone(), v));
            }
        }
        let dims: Vec<Vec<usize>> = outs
            .iter()
            .map(|id| {
                g.node(*id)
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect()
            })
            .collect();
        self.charge(&format!("trace {} nodes", extra.len()))?;
        let mut c = Session::new(Device::Cpu).compile(g);
        for (name, t) in &self.state.params {
            c.set_param(name, &t.data);
        }
        let refs: Vec<(&str, &[f32])> = feeds
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        let values = c.run(&refs);
        let shown: Vec<bool> = outs.iter().map(|id| !hidden_outputs.contains(id)).collect();
        Ok(labels
            .into_iter()
            .zip(dims)
            .zip(values)
            .zip(shown)
            .filter(|(_, show)| *show)
            .map(|(x, _)| x)
            .map(|((label, mut d), mut v)| {
                if mlp && d.first() == Some(&BATCH) {
                    let per = v.len() / BATCH;
                    v.truncate(per);
                    d[0] = 1;
                }
                (label, d, v)
            })
            .collect())
    }

    fn no_outputs(&self, what: &str) -> Result<(), String> {
        if self.state.internals_only {
            return Err(format!(
                "`{what}` is not available: this task shows internal activations only (`trace`)"
            ));
        }
        Ok(())
    }

    /// Params of a checkpoint tag, or the live ones for `"current"`.
    fn params_of(&self, tag: &str) -> Result<BTreeMap<String, Tensor>, String> {
        if tag == "current" {
            return Ok(self.state.params.clone());
        }
        self.state
            .checkpoints
            .get(tag)
            .cloned()
            .ok_or_else(|| format!("no checkpoint `{tag}` (use `current` for the live weights)"))
    }

    /// Score `params` without changing the live weights (no budget charge).
    fn score_with(
        &mut self,
        params: BTreeMap<String, Tensor>,
        episodes: usize,
        seed: u64,
    ) -> Result<Scored, String> {
        let live = std::mem::replace(&mut self.state.params, params);
        self.dirty = self.state.params.keys().cloned().collect();
        let out = if self.state.env.is_classifier() {
            let data = crate::mnist::data()?;
            let n = episodes.clamp(1, data.val.len());
            let preds = self.classify(&data.val.images[..n * crate::mnist::PIXELS]);
            Ok(Scored::Correct(
                preds
                    .iter()
                    .zip(&data.val.labels)
                    .map(|(p, l)| *p == *l as usize)
                    .collect(),
            ))
        } else if self.state.graph.is_some() {
            Err("graph spaces have no score to compare; use `trace`".to_string())
        } else {
            Ok(Scored::Returns(
                self.play(episodes, seed, 0).episode_returns,
            ))
        };
        self.state.params = live;
        self.dirty = self.state.params.keys().cloned().collect();
        out
    }

    /// Paired comparison of two weight sets on the same episodes (control)
    /// or validation digits (classifier). Costs two forward passes.
    pub fn compare_params(
        &mut self,
        a: BTreeMap<String, Tensor>,
        b: BTreeMap<String, Tensor>,
        episodes: usize,
        seed: Option<u64>,
        what: &str,
    ) -> Result<Value, String> {
        self.no_outputs(what)?;
        if self.remaining() < 2 {
            return Err(format!(
                "{what} needs 2 forward passes; {} left",
                self.remaining()
            ));
        }
        let seed = seed.unwrap_or_else(|| {
            self.state.rollout_counter += 1;
            0x5EED_0000 + self.state.rollout_counter
        });
        let sa = self.score_with(a, episodes, seed)?;
        let sb = self.score_with(b, episodes, seed)?;
        self.charge(&format!("{what} (a)"))?;
        self.charge(&format!("{what} (b)"))?;
        Ok(match (sa, sb) {
            (Scored::Returns(ra), Scored::Returns(rb)) => {
                let n = ra.len() as f32;
                let d: Vec<f32> = rb.iter().zip(&ra).map(|(b, a)| b - a).collect();
                let mean = d.iter().sum::<f32>() / n;
                let var = d.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
                serde_json::json!({
                    "episodes": ra.len(), "seed": seed,
                    "a_mean_return": ra.iter().sum::<f32>() / n,
                    "b_mean_return": rb.iter().sum::<f32>() / n,
                    "b_minus_a": mean,
                    "std_error": (var / n).sqrt(),
                    "b_better_episodes": d.iter().filter(|x| **x > 0.0).count(),
                    "a_better_episodes": d.iter().filter(|x| **x < 0.0).count(),
                    "tied_episodes": d.iter().filter(|x| **x == 0.0).count(),
                })
            }
            (Scored::Correct(ca), Scored::Correct(cb)) => {
                let n = ca.len() as f32;
                let acc = |c: &[bool]| c.iter().filter(|x| **x).count() as f32 / n;
                serde_json::json!({
                    "digits": ca.len(),
                    "a_accuracy": acc(&ca),
                    "b_accuracy": acc(&cb),
                    "b_minus_a": acc(&cb) - acc(&ca),
                    "b_right_a_wrong": ca.iter().zip(&cb).filter(|(a, b)| !**a && **b).count(),
                    "a_right_b_wrong": ca.iter().zip(&cb).filter(|(a, b)| **a && !**b).count(),
                })
            }
            _ => unreachable!("both sides scored the same task"),
        })
    }

    /// `compare` tool: checkpoint `a` vs `b` (`"current"` = live weights).
    pub fn compare(
        &mut self,
        a: &str,
        b: &str,
        episodes: usize,
        seed: Option<u64>,
    ) -> Result<Value, String> {
        let (pa, pb) = (self.params_of(a)?, self.params_of(b)?);
        let mut r = self.compare_params(pa, pb, episodes, seed, &format!("compare {a} vs {b}"))?;
        r["a"] = a.into();
        r["b"] = b.into();
        Ok(r)
    }

    /// The first `steps` steps of one greedy episode (harness use; not
    /// budgeted).
    pub fn record_episode(&mut self, seed: u64, steps: usize) -> Vec<String> {
        self.play(BATCH, seed, steps).trace
    }

    /// Make network `to` the active one (multi-network tasks); the current
    /// weights are stored under the active name.
    pub fn switch(&mut self, to: &str) -> Result<(), String> {
        let current = self
            .state
            .active
            .clone()
            .ok_or("this task has a single network")?;
        if to == current {
            return Ok(());
        }
        let next = self
            .state
            .networks
            .remove(to)
            .ok_or_else(|| format!("no network `{to}`"))?;
        let prev = std::mem::replace(&mut self.state.params, next);
        self.state.networks.insert(current, prev);
        self.state.active = Some(to.to_string());
        self.dirty = self.state.params.keys().cloned().collect();
        self.state.log.push(format!("switch to {to}"));
        Ok(())
    }

    /// Copy hidden units from a checkpoint or another network into the live
    /// weights: each unit's incoming column, bias and outgoing row. `into`
    /// names the target units (default: the same units).
    pub fn graft(
        &mut self,
        from: &str,
        units: &[String],
        into: &[String],
    ) -> Result<Value, String> {
        let src = match self.state.networks.get(from) {
            Some(p) => p.clone(),
            None => self.params_of(from)?,
        };
        if !into.is_empty() && into.len() != units.len() {
            return Err("`into` needs one target per unit".into());
        }
        let parse = |u: &str| -> Result<(usize, usize), String> {
            u.strip_prefix('l')
                .and_then(|s| s.split_once(".h"))
                .and_then(|(k, i)| Some((k.parse().ok()?, i.parse().ok()?)))
                .ok_or_else(|| format!("`{u}` is not a unit name like l0.h3"))
        };
        // Source and target may differ in architecture: units may move
        // between layers and widths may differ. Incoming columns and outgoing
        // rows are copied over the overlapping part; the rest of the target's
        // column / row is zeroed, and every mismatch is reported.
        let hidden = self.state.mlp.n_layers() - 1;
        let src_has = |k: usize| {
            src.contains_key(&format!("l{k}.w")) && src.contains_key(&format!("l{}.w", k + 1))
        };
        let mut moves = Vec::new();
        for (n, u) in units.iter().enumerate() {
            let (k, i) = parse(u)?;
            let target = into.get(n).unwrap_or(u);
            let (k2, j) = parse(target)?;
            if !src_has(k) || i >= src[&format!("l{k}.w")].cols {
                return Err(format!("the source has no hidden unit {u}"));
            }
            if k2 >= hidden || j >= self.state.mlp.widths[k2 + 1] {
                return Err(format!("no hidden unit {target} here"));
            }
            moves.push((k, i, k2, j));
        }
        let mut notes = Vec::new();
        for &(k, i, k2, j) in &moves {
            let (sw, sb, so) = (
                src[&format!("l{k}.w")].clone(),
                src[&format!("l{k}.b")].clone(),
                src[&format!("l{}.w", k + 1)].clone(),
            );
            let (wi, bi, wo) = (
                format!("l{k2}.w"),
                format!("l{k2}.b"),
                format!("l{}.w", k2 + 1),
            );
            let (dst_in, dst_out) = (self.param(&wi)?.rows, self.param(&wo)?.cols);
            if sw.rows != dst_in {
                notes.push(format!(
                    "l{k}.h{i}: fan-in {} → {dst_in}, copied the first {}",
                    sw.rows,
                    sw.rows.min(dst_in)
                ));
            }
            if so.cols != dst_out {
                notes.push(format!(
                    "l{k}.h{i}: fan-out {} → {dst_out}, copied the first {}",
                    so.cols,
                    so.cols.min(dst_out)
                ));
            }
            self.edit(&wi, &format!("graft col {i}->{j} from {from}"), |t| {
                for r in 0..t.rows {
                    t.data[r * t.cols + j] = if r < sw.rows { sw.at(r, i) } else { 0.0 };
                }
                Ok(())
            })?;
            self.edit(&bi, &format!("graft bias {i}->{j} from {from}"), |t| {
                t.data[j] = sb.data[i];
                Ok(())
            })?;
            self.edit(&wo, &format!("graft row {i}->{j} from {from}"), |t| {
                for c in 0..t.cols {
                    t.data[j * t.cols + c] = if c < so.cols { so.at(i, c) } else { 0.0 };
                }
                Ok(())
            })?;
        }
        Ok(serde_json::json!({ "grafted": moves.len(), "from": from, "notes": notes }))
    }

    /// Weights with hidden units `units` (`l{k}.h{i}`) switched off: their
    /// incoming column and bias zeroed, so they output act(0) = 0.
    pub fn ablated(&self, units: &[String]) -> Result<BTreeMap<String, Tensor>, String> {
        if self.state.graph.is_some() {
            return Err("ablation addresses MLP units `lK.hI`".into());
        }
        let mut p = self.state.params.clone();
        let hidden_layers = self.state.mlp.n_layers() - 1;
        for u in units {
            let (k, i) = u
                .strip_prefix('l')
                .and_then(|s| s.split_once(".h"))
                .and_then(|(k, i)| Some((k.parse::<usize>().ok()?, i.parse::<usize>().ok()?)))
                .ok_or_else(|| format!("`{u}` is not a unit name like l0.h3"))?;
            if k >= hidden_layers || i >= self.state.mlp.widths[k + 1] {
                return Err(format!("no hidden unit `{u}`"));
            }
            let w = p.get_mut(&format!("l{k}.w")).expect("layer weight");
            for r in 0..w.rows {
                w.data[r * w.cols + i] = 0.0;
            }
            p.get_mut(&format!("l{k}.b")).expect("layer bias").data[i] = 0.0;
        }
        Ok(p)
    }

    /// `ablate` tool: live weights vs the same with `units` off.
    pub fn ablate(
        &mut self,
        units: &[String],
        episodes: usize,
        seed: Option<u64>,
    ) -> Result<Value, String> {
        let off = self.ablated(units)?;
        let live = self.state.params.clone();
        let mut r = self.compare_params(live, off, episodes, seed, &format!("ablate {units:?}"))?;
        r["a"] = "current".into();
        r["b"] = format!("current with {} unit(s) off", units.len()).into();
        Ok(r)
    }

    fn play(&mut self, episodes: usize, seed: u64, trace_steps: usize) -> RolloutReport {
        let d = self.spec.obs_dim;
        let na = self.spec.actions.len();
        let horizon = self.spec.horizon;
        let batches = episodes.max(1).div_ceil(BATCH);
        let mut rng = Philox4x32::new(seed);
        let (mut total, mut pos, mut neg) = (0f64, 0usize, 0usize);
        let mut mix = vec![0usize; na];
        let mut trace = Vec::new();
        let mut obs = vec![0f32; BATCH * d];
        let mut returns = vec![0f32; batches * BATCH];
        let protocol = self.state.protocol;
        let mut prng = Philox4x32::new(policy_seed(seed));
        for b in 0..batches {
            let mut prev: Vec<Option<usize>> = vec![None; BATCH];
            let mut envs: Vec<_> = (0..BATCH).map(|_| self.state.env.make()).collect();
            for e in &mut envs {
                e.reset(&mut rng);
            }
            for t in 0..horizon {
                for (i, e) in envs.iter().enumerate() {
                    e.observe(&mut obs[i * d..(i + 1) * d]);
                }
                let logits = self.run(&obs).swap_remove(0);
                for (i, e) in envs.iter_mut().enumerate() {
                    let a = protocol.choose(&logits[i * na..(i + 1) * na], prev[i], &mut prng);
                    prev[i] = Some(a);
                    mix[a] += 1;
                    let before = (b == 0 && i == 0 && t < trace_steps).then(|| e.render_state());
                    let r = e.step(a, &mut rng);
                    if let Some(s) = before {
                        trace.push(format!(
                            "t={t} {s} -> {} reward={r:+}",
                            self.spec.actions[a]
                        ));
                    }
                    total += r as f64;
                    returns[b * BATCH + i] += r;
                    if r > 0.0 {
                        pos += 1;
                    } else if r < 0.0 {
                        neg += 1;
                    }
                }
            }
        }
        let n = batches * BATCH;
        let steps = (n * horizon) as f32;
        RolloutReport {
            episodes: n,
            mean_return: (total / n as f64) as f32,
            positive_events: pos,
            negative_events: neg,
            action_mix: mix.iter().map(|&k| k as f32 / steps).collect(),
            trace,
            episode_returns: returns,
        }
    }
}

/// Score a non-network controller on exactly the episodes
/// [`WeightSpace::evaluate`] plays for the same `seed` (same batching, same
/// RNG order), so reference controllers and networks compare directly.
pub fn play_policy<E: Env>(
    make: impl Fn() -> E,
    episodes: usize,
    seed: u64,
    policy: impl FnMut(&E) -> usize,
) -> RolloutReport {
    play_policy_with(make, episodes, seed, Protocol::GREEDY, policy)
}

/// [`play_policy`] under `protocol`; a deterministic controller is not
/// sampled, but sticky actions apply to it as to a network.
pub fn play_policy_with<E: Env>(
    make: impl Fn() -> E,
    episodes: usize,
    seed: u64,
    protocol: Protocol,
    mut policy: impl FnMut(&E) -> usize,
) -> RolloutReport {
    let horizon = make().spec().horizon;
    let batches = episodes.max(1).div_ceil(BATCH);
    let mut rng = Philox4x32::new(seed);
    let mut prng = Philox4x32::new(policy_seed(seed));
    let sticky = Protocol {
        sample: false,
        sticky: protocol.sticky,
    };
    let (mut total, mut pos, mut neg) = (0f64, 0usize, 0usize);
    let mut mix = vec![0usize; make().spec().actions.len()];
    for _ in 0..batches {
        let mut envs: Vec<E> = (0..BATCH).map(|_| make()).collect();
        for e in &mut envs {
            e.reset(&mut rng);
        }
        let mut prev: Vec<Option<usize>> = vec![None; BATCH];
        for _ in 0..horizon {
            for (i, e) in envs.iter_mut().enumerate() {
                let want = policy(e);
                let mut onehot = [f32::NEG_INFINITY; 8];
                onehot[want] = 0.0;
                let a = sticky.choose(&onehot[..mix.len()], prev[i], &mut prng);
                prev[i] = Some(a);
                mix[a] += 1;
                let r = e.step(a, &mut rng);
                total += r as f64;
                if r > 0.0 {
                    pos += 1;
                } else if r < 0.0 {
                    neg += 1;
                }
            }
        }
    }
    let n = batches * BATCH;
    let steps = (n * horizon) as f32;
    RolloutReport {
        episodes: n,
        mean_return: (total / n as f64) as f32,
        positive_events: pos,
        negative_events: neg,
        action_mix: mix.iter().map(|&k| k as f32 / steps).collect(),
        trace: Vec::new(),
        episode_returns: Vec::new(),
    }
}

/// Per-item outcomes for paired comparisons.
enum Scored {
    Returns(Vec<f32>),
    Correct(Vec<bool>),
}

/// A row/column addressed by index or by label.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RowKey {
    Index(usize),
    Label(String),
}

/// First index of the maximum (ties → lowest action index).
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn layer_index(name: &str) -> Result<usize, String> {
    name.strip_prefix('l')
        .and_then(|s| s.split('.').next())
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("`{name}` is not a layer param"))
}
