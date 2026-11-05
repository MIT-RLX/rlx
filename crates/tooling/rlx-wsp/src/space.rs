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
/// Stored compactly (`b64`: base64 of little-endian f32); the older
/// number-array form (`data`) still reads.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Serialize for Tensor {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let bytes: Vec<u8> = self.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut st = s.serialize_struct("Tensor", 3)?;
        st.serialize_field("rows", &self.rows)?;
        st.serialize_field("cols", &self.cols)?;
        st.serialize_field("b64", &b64_encode(&bytes))?;
        st.end()
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            rows: usize,
            cols: usize,
            #[serde(default)]
            data: Option<Vec<f32>>,
            #[serde(default)]
            b64: Option<String>,
        }
        let r = Raw::deserialize(d)?;
        let data = match (r.data, r.b64) {
            (Some(v), _) => v,
            (None, Some(b)) => {
                let bytes = b64_decode(&b).map_err(serde::de::Error::custom)?;
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            (None, None) => return Err(serde::de::Error::custom("tensor without data")),
        };
        if data.len() != r.rows * r.cols {
            return Err(serde::de::Error::custom(
                "tensor size does not match its shape",
            ));
        }
        Ok(Tensor {
            rows: r.rows,
            cols: r.cols,
            data,
        })
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for (k, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            if k <= c.len() {
                out.push(B64[((n >> shift) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn b64_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut val = [255u8; 256];
    for (i, &c) in B64.iter().enumerate() {
        val[c as usize] = i as u8;
    }
    let bytes: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if !bytes.len().is_multiple_of(4) {
        return Err("base64 length".into());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for c in bytes.chunks(4) {
        let mut n = 0u32;
        let mut pad = 0;
        for &b in c {
            n <<= 6;
            if b == b'=' {
                pad += 1;
            } else {
                let v = val[b as usize];
                if v == 255 {
                    return Err("base64 character".into());
                }
                n |= u32::from(v);
            }
        }
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Ok(out)
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
    /// [`EnvKind::Graph`]: the `external_score` tool's command (program and
    /// arguments); it is run with a `{name: [values]}` JSON file of the live
    /// weights as its last argument and prints a JSON score.
    #[serde(default)]
    pub score_cmd: Option<Vec<String>>,
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
    /// Row / column labels of params, where a task or a rewrite set them
    /// (otherwise derived, or generic `rI` / `cJ`).
    #[serde(default)]
    pub labels: BTreeMap<String, Axes>,
    /// Channel labels of nodes (units), likewise.
    #[serde(default)]
    pub unit_labels: BTreeMap<String, Vec<String>>,
    /// Package space: the `.rlxp` the weights come from; `params` then holds
    /// only the tensors the agent changed (an overlay).
    #[serde(default)]
    pub package: Option<String>,
    /// Question-answering task over a language-model package.
    #[serde(default)]
    pub qa: Option<crate::lm::QaSpec>,
    /// Package spaces: the function graph tools address (default: the first).
    #[serde(default)]
    pub function: Option<String>,
    /// Named vectors (e.g. from `contrast`), usable as directions.
    #[serde(default)]
    pub vectors: BTreeMap<String, Vec<f32>>,
    /// Package spaces: graph rewrites applied whenever the model is opened.
    #[serde(default)]
    pub rewrites: Vec<Rewrite>,
    /// Package spaces: tensors whose edits are all rank-1, kept as factors
    /// `W = base + Σ u vᵀ` (the dense copy is not stored).
    #[serde(default)]
    pub lowrank: BTreeMap<String, Vec<(Vec<f32>, Vec<f32>)>>,
    /// The task seed the space was made with (reports).
    #[serde(default)]
    pub task_seed: u64,
    /// The rewrites in force when each checkpoint was taken.
    #[serde(default)]
    pub checkpoint_rewrites: BTreeMap<String, Vec<Rewrite>>,
    /// Language models: tokens (prompts and outputs) the tools may process
    /// in all; `None`: only the forward-pass budget applies.
    #[serde(default)]
    pub token_budget: Option<usize>,
    #[serde(default)]
    pub tokens_used: usize,
    /// `dev_check` calls left (QA tasks with a dev split).
    #[serde(default)]
    pub dev_checks_left: usize,
    /// The harness scores checkpoint `verified` (a version `dev_check`
    /// found clearly better), or the unedited model when there is none,
    /// instead of the final state.
    #[serde(default)]
    pub keep_verified: bool,
    /// Package spaces: the largest relative change `‖W − W₀‖ / ‖W₀‖` an
    /// edit may leave in any one tensor (`None`: no cap).
    #[serde(default)]
    pub max_edit: Option<f32>,
    /// Runs this one branched off, oldest first: where from, which
    /// version, and what had been done there.
    #[serde(default)]
    pub lineage: Vec<Value>,
    /// Tools this run may not use (hidden from the tool list, refused when
    /// called directly, from a script or in a search candidate).
    #[serde(default)]
    pub denied_tools: Vec<String>,
    /// Seconds spent in each tool (harness timing: where a run's wall
    /// clock goes).
    #[serde(default)]
    pub tool_seconds: BTreeMap<String, f64>,
    /// Scripts the agent saved by name (`script {save_as}`), rerun with
    /// `script {saved}` — its own analysis programs.
    #[serde(default)]
    pub programs: BTreeMap<String, String>,
    /// Every tool call of the run, for charting how it explored (see
    /// [`crate::explore`]).
    #[serde(default)]
    pub trail: Vec<crate::explore::Step>,
    /// Background jobs (see [`crate::jobs`]) and how many may run at once.
    #[serde(default)]
    pub jobs: Vec<crate::jobs::Job>,
    #[serde(default = "one_job")]
    pub max_jobs: usize,
    /// A LoRA training run the agent steers (see [`crate::opal`]).
    #[serde(default)]
    pub opal: Option<crate::opal::OpalConfig>,
}

fn one_job() -> usize {
    1
}

/// One version of the network for paired comparisons: weights (overlay),
/// low-rank factors and recorded rewrites.
#[derive(Debug, Clone, Default)]
pub struct Variant {
    pub params: BTreeMap<String, Tensor>,
    pub lowrank: BTreeMap<String, Vec<(Vec<f32>, Vec<f32>)>>,
    pub rewrites: Vec<Rewrite>,
}

/// A structural change recorded for a package (its graphs live in the
/// package, so the change is replayed at every open).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Rewrite {
    /// Run the span `from → to` `times` times (shared weights).
    Repeat {
        from: String,
        to: String,
        times: usize,
    },
    /// Add `values[i]` to channel `channels[i]` of `node` at every position
    /// (a logit / channel bias, e.g. suppressing a token).
    ChannelBias {
        node: String,
        channels: Vec<usize>,
        values: Vec<f32>,
    },
    /// Add `scale · vector` to `node` at every position (activation
    /// steering along a direction).
    AddVector {
        node: String,
        vector: Vec<f32>,
        scale: f32,
    },
}

impl Rewrite {
    pub fn describe(&self) -> String {
        match self {
            Rewrite::Repeat { from, to, times } => format!("repeat {from}..{to} x{times}"),
            Rewrite::ChannelBias {
                node,
                channels,
                values,
            } => format!(
                "bias {} channels of {node} (first {:?} += {:?})",
                channels.len(),
                channels.first(),
                values.first()
            ),
            Rewrite::AddVector {
                node,
                vector,
                scale,
            } => format!(
                "steer {node} by {scale} x a {}-vector (norm {:.3})",
                vector.len(),
                vector.iter().map(|x| x * x).sum::<f32>().sqrt()
            ),
        }
    }
}

/// Apply recorded rewrites to one graph (those whose nodes it has).
pub fn apply_rewrites(g: rlx_ir::Graph, rewrites: &[Rewrite]) -> rlx_ir::Graph {
    let mut g = g;
    for r in rewrites {
        match r {
            Rewrite::Repeat { from, to, times } => {
                if let (Some(f), Some(t)) = (
                    crate::graphops::find_node(&g, from),
                    crate::graphops::find_node(&g, to),
                ) && let Ok((r, _)) = crate::graphops::repeat_span(&g, f, t, *times, false)
                {
                    g = r;
                }
            }
            Rewrite::ChannelBias {
                node,
                channels,
                values,
            } => {
                if let Some(id) = crate::graphops::find_node(&g, node)
                    && let Some(w) = crate::graphops::width(&g, id)
                {
                    let mut add = vec![0f32; w];
                    for (&c, &v) in channels.iter().zip(values) {
                        if c < w {
                            add[c] += v;
                        }
                    }
                    if let Ok(r) = crate::graphops::add_channels(&g, id, &add) {
                        g = r;
                    }
                }
            }
            Rewrite::AddVector {
                node,
                vector,
                scale,
            } => {
                if let Some(id) = crate::graphops::find_node(&g, node) {
                    let add: Vec<f32> = vector.iter().map(|x| x * scale).collect();
                    if let Ok(r) = crate::graphops::add_channels(&g, id, &add) {
                        g = r;
                    }
                }
            }
        }
    }
    g
}

/// Row and column labels of a 2-D param.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Axes {
    pub rows: Vec<String>,
    pub cols: Vec<String>,
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
    /// Channels zeroed in every run (ablation); not persisted.
    masks: Vec<crate::graphops::Unit>,
    /// Package spaces: the weights on disk.
    store: Option<crate::pkgsrc::PackageStore>,
    store_error: Option<String>,
    /// Package spaces: the opened language model.
    pub(crate) lm: crate::lm::LmHost,
    /// Background jobs this process started (see [`crate::jobs`]).
    pub(crate) children: BTreeMap<usize, std::process::Child>,
    /// The steered training run, once opened.
    #[cfg(feature = "training")]
    pub(crate) opal_run: Option<crate::opal::Shared>,
    /// A training run attached by the embedding program ([`crate::steer`]).
    pub(crate) steered: Option<crate::steer::SteeredRun>,
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
                labels: BTreeMap::new(),
                unit_labels: BTreeMap::new(),
                package: None,
                qa: None,
                function: None,
                vectors: BTreeMap::new(),
                rewrites: Vec::new(),
                lowrank: BTreeMap::new(),
                task_seed: 0,
                checkpoint_rewrites: BTreeMap::new(),
                token_budget: None,
                tokens_used: 0,
                dev_checks_left: 0,
                keep_verified: false,
                max_edit: None,
                lineage: Vec::new(),
                denied_tools: Vec::new(),
                score_cmd: None,
                tool_seconds: BTreeMap::new(),
                programs: BTreeMap::new(),
                trail: Vec::new(),
                jobs: Vec::new(),
                max_jobs: 1,
                opal: None,
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
            labels: BTreeMap::new(),
            unit_labels: BTreeMap::new(),
            package: None,
            qa: None,
            function: None,
            vectors: BTreeMap::new(),
            rewrites: Vec::new(),
            lowrank: BTreeMap::new(),
            task_seed: 0,
            checkpoint_rewrites: BTreeMap::new(),
            token_budget: None,
            tokens_used: 0,
            dev_checks_left: 0,
            keep_verified: false,
            max_edit: None,
            lineage: Vec::new(),
            denied_tools: Vec::new(),
            score_cmd: None,
            tool_seconds: BTreeMap::new(),
            programs: BTreeMap::new(),
            trail: Vec::new(),
            jobs: Vec::new(),
            max_jobs: 1,
            opal: None,
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
            labels: BTreeMap::new(),
            unit_labels: BTreeMap::new(),
            package: None,
            qa: None,
            function: None,
            vectors: BTreeMap::new(),
            rewrites: Vec::new(),
            lowrank: BTreeMap::new(),
            task_seed: 0,
            checkpoint_rewrites: BTreeMap::new(),
            token_budget: None,
            tokens_used: 0,
            dev_checks_left: 0,
            keep_verified: false,
            max_edit: None,
            lineage: Vec::new(),
            denied_tools: Vec::new(),
            score_cmd: None,
            tool_seconds: BTreeMap::new(),
            programs: BTreeMap::new(),
            trail: Vec::new(),
            jobs: Vec::new(),
            max_jobs: 1,
            opal: None,
        }))
    }

    /// A space over an `.rlxp` package: weights read on demand, edits as an
    /// overlay. With `qa`, a scored question-answering task; else a loaded
    /// model (no score).
    pub fn from_package(
        path: &str,
        budget: usize,
        qa: Option<crate::lm::QaSpec>,
        goal: Option<String>,
    ) -> Result<Self, String> {
        crate::pkgsrc::PackageStore::open(path)?;
        let mut st = Self::new(EnvKind::Bounce, &[1], Act::Tanh, budget).state;
        st.env = if qa.is_some() {
            EnvKind::Qa
        } else {
            EnvKind::Graph
        };
        st.params = BTreeMap::new();
        st.package = Some(path.to_string());
        st.qa = qa;
        st.goal = goal;
        Ok(Self::from_state(st))
    }

    /// The package weights, for package spaces.
    pub fn store(&self) -> Option<&crate::pkgsrc::PackageStore> {
        self.store.as_ref()
    }

    /// The state as JSON, with low-rank tensors stored as their factors.
    pub fn to_json(&self) -> Result<String, String> {
        if self.state.lowrank.is_empty() {
            return serde_json::to_string(&self.state).map_err(|e| e.to_string());
        }
        let mut st = self.state.clone();
        for name in st.lowrank.keys() {
            st.params.remove(name);
        }
        serde_json::to_string(&st).map_err(|e| e.to_string())
    }

    /// Apply `W += u vᵀ` to `name`; on a package space a tensor whose edits
    /// are all rank-1 keeps them as factors (small state, small adapter).
    pub fn edit_rank1(
        &mut self,
        name: &str,
        u: Vec<f32>,
        v: Vec<f32>,
        note: &str,
    ) -> Result<(), String> {
        let track = self.state.package.is_some()
            && (!self.state.params.contains_key(name) || self.state.lowrank.contains_key(name));
        let (uu, vv) = (u.clone(), v.clone());
        self.edit(name, &format!("lowrank: {note}"), |t| {
            if uu.len() != t.rows || vv.len() != t.cols {
                return Err(format!("u needs {} values and v {}", t.rows, t.cols));
            }
            for r in 0..t.rows {
                if uu[r] != 0.0 {
                    for c in 0..t.cols {
                        t.data[r * t.cols + c] += uu[r] * vv[c];
                    }
                }
            }
            Ok(())
        })?;
        if track {
            self.state
                .lowrank
                .entry(name.to_string())
                .or_default()
                .push((u, v));
        } else {
            self.state.lowrank.remove(name);
        }
        Ok(())
    }

    pub fn from_state(state: State) -> Self {
        let spec = state.env.make().spec();
        let dirty = state.params.keys().cloned().collect();
        let mut state = state;
        // A checkpoint saved before rewrites were recorded had none.
        for tag in state.checkpoints.keys() {
            state.checkpoint_rewrites.entry(tag.clone()).or_default();
        }
        let (store, store_error) = match &state.package {
            Some(p) => match crate::pkgsrc::PackageStore::open(p) {
                Ok(s) => (Some(s), None),
                Err(e) => (None, Some(e)),
            },
            None => (None, None),
        };
        // Low-rank tensors come back as base + factors.
        if let Some(st) = &store {
            for (name, factors) in &state.lowrank {
                if state.params.contains_key(name) {
                    continue;
                }
                if let Ok(base) = st.get(name) {
                    let mut t = base.clone();
                    for (u, v) in factors {
                        for r in 0..t.rows {
                            for c in 0..t.cols {
                                t.data[r * t.cols + c] += u[r] * v[c];
                            }
                        }
                    }
                    state.params.insert(name.clone(), t);
                }
            }
        }
        Self {
            state,
            spec,
            compiled: None,
            dirty,
            masks: Vec::new(),
            store,
            store_error,
            lm: Default::default(),
            children: BTreeMap::new(),
            #[cfg(feature = "training")]
            opal_run: None,
            steered: None,
        }
    }

    pub fn spec(&self) -> &EnvSpec {
        &self.spec
    }

    pub fn remaining(&self) -> usize {
        self.state.budget.saturating_sub(self.state.used)
    }

    pub fn param(&self, name: &str) -> Result<&Tensor, String> {
        if let Some(t) = self.state.params.get(name) {
            return Ok(t);
        }
        if let Some(store) = &self.store {
            return store.get(name);
        }
        if let Some(e) = &self.store_error {
            return Err(e.clone());
        }
        self.state.params.get(name).ok_or_else(|| {
            let known: Vec<_> = self.state.params.keys().map(String::as_str).collect();
            format!("unknown param `{name}` (have: {})", known.join(", "))
        })
    }

    /// Row labels for `name`: observation labels for the input layer, unit
    /// names (`l{k}.h{i}`) for deeper layers, `bias` for biases.
    pub fn row_labels(&self, name: &str) -> Result<Vec<String>, String> {
        let t = self.param(name)?;
        if self.state.package.is_some() && !self.state.labels.contains_key(name) {
            return Ok((0..t.rows).map(|i| format!("r{i}")).collect());
        }
        if let Some(a) = self.state.labels.get(name)
            && a.rows.len() == t.rows
        {
            return Ok(a.rows.clone());
        }
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
        if self.state.package.is_some() && !self.state.labels.contains_key(name) {
            return Ok((0..t.cols).map(|i| format!("c{i}")).collect());
        }
        if let Some(a) = self.state.labels.get(name)
            && a.cols.len() == t.cols
        {
            return Ok(a.cols.clone());
        }
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
        // Package spaces: copy the tensor into the overlay on first edit.
        if !self.state.params.contains_key(name)
            && let Some(store) = &self.store
        {
            if !store.editable(name) {
                return Err(format!("`{name}` is not editable (quantized or unknown)"));
            }
            let base = store.get(name)?.clone();
            self.state.params.insert(name.to_string(), base);
        }
        let cap = self.state.max_edit.filter(|_| self.store.is_some());
        let t = self
            .state
            .params
            .get_mut(name)
            .ok_or_else(|| format!("unknown param `{name}`"))?;
        let before = cap.map(|_| t.clone());
        let r = f(t)?;
        if t.data.iter().any(|v| !v.is_finite()) {
            return Err(format!("{name}: edit produced a non-finite value"));
        }
        if let (Some(cap), Some(before), Some(store)) = (cap, before, &self.store) {
            let base = store.get(name)?;
            let d: f64 = t
                .data
                .iter()
                .zip(&base.data)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum();
            let w: f64 = base.data.iter().map(|b| (*b as f64).powi(2)).sum();
            let rel = (d / w.max(1e-30)).sqrt();
            if rel > cap as f64 {
                *t = before;
                return Err(format!(
                    "{name}: this edit would move the tensor {rel:.3} of its norm from the package's \
                     (cap {cap}); edit less"
                ));
            }
        }
        self.dirty.insert(name.to_string());
        if !note.starts_with("lowrank: ") {
            self.state.lowrank.remove(name);
        }
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
        self.state
            .checkpoint_rewrites
            .insert(tag.to_string(), self.state.rewrites.clone());
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
        // Snapshots are dense; their tensors are stored dense from here.
        self.state.lowrank.clear();
        if let Some(r) = self.state.checkpoint_rewrites.get(tag) {
            self.state.rewrites = r.clone();
            self.compiled = None;
        }
        self.dirty = self.state.params.keys().cloned().collect();
        self.state.log.push(format!("revert {tag}"));
        Ok(())
    }

    /// A new run starting from this one's `version` (`current` or a
    /// checkpoint tag): the same task and weights, a fresh budget of
    /// `budget` forward passes, checkpoints kept (plus `branch_start`), the
    /// history moved into `lineage`. `from` names this run (its state
    /// file).
    pub fn branch(&self, from: &str, version: &str, budget: usize) -> Result<WeightSpace, String> {
        let v = self.variant(version)?;
        let mut st = self.state.clone();
        st.params = v.params;
        st.lowrank = if version == "current" {
            v.lowrank
        } else {
            BTreeMap::new()
        };
        st.rewrites = v.rewrites;
        let kept: Vec<&String> = self
            .state
            .log
            .iter()
            .filter(|l| {
                [
                    "edit ",
                    "forward ",
                    "checkpoint ",
                    "revert ",
                    "dev_check",
                    "search",
                    "lora",
                    "repeat",
                    "steer",
                    "bias",
                ]
                .iter()
                .any(|p| l.starts_with(p))
            })
            .collect();
        st.lineage.push(serde_json::json!({
            "from": from,
            "version": version,
            "forward_passes_used": self.state.used,
            "budget": self.state.budget,
            "tokens_used": self.state.tokens_used,
            "history": kept.iter().rev().take(120).rev().collect::<Vec<_>>(),
        }));
        st.used = 0;
        st.budget = budget;
        st.tokens_used = 0;
        st.jobs.clear();
        st.log = vec![format!("branched from {from} at `{version}`")];
        st.checkpoints
            .insert("branch_start".into(), st.params.clone());
        st.checkpoint_rewrites
            .insert("branch_start".into(), st.rewrites.clone());
        Ok(WeightSpace::from_state(st))
    }

    /// Set the live weights to `Σ wᵢ · versionᵢ` (checkpoint tags,
    /// `current`, `unedited`); the versions must share their rewrites.
    pub fn average(&mut self, tags: &[String], weights: &[f32]) -> Result<Value, String> {
        if tags.len() < 2 || tags.len() != weights.len() {
            return Err("give 2+ `tags` and one weight each".into());
        }
        let vs: Vec<Variant> = tags
            .iter()
            .map(|t| self.variant(t))
            .collect::<Result<_, _>>()?;
        if vs.iter().any(|v| v.rewrites != vs[0].rewrites) {
            return Err(
                "these versions differ in their graph rewrites; average needs the same graph"
                    .into(),
            );
        }
        let names: std::collections::BTreeSet<String> =
            vs.iter().flat_map(|v| v.params.keys().cloned()).collect();
        let mut out = BTreeMap::new();
        for n in &names {
            let mut acc: Option<Tensor> = None;
            for (v, &w) in vs.iter().zip(weights) {
                let t = match v.params.get(n) {
                    Some(t) => t.clone(),
                    None => self
                        .store
                        .as_ref()
                        .ok_or_else(|| format!("`{n}` is missing from a version"))?
                        .get(n)?
                        .clone(),
                };
                match acc.as_mut() {
                    None => {
                        let mut t = t;
                        t.data.iter_mut().for_each(|x| *x *= w);
                        acc = Some(t);
                    }
                    Some(a) => a
                        .data
                        .iter_mut()
                        .zip(&t.data)
                        .for_each(|(a, b)| *a += w * b),
                }
            }
            out.insert(n.clone(), acc.expect("two or more versions"));
        }
        let rewrites = vs[0].rewrites.clone();
        self.swap_variant(Variant {
            params: out,
            lowrank: BTreeMap::new(),
            rewrites,
        });
        self.state
            .log
            .push(format!("average {tags:?} with {weights:?}"));
        Ok(serde_json::json!({ "averaged": tags, "weights": weights, "tensors": names.len() }))
    }

    /// Record a rewrite (replayed on every run / open).
    pub fn add_rewrite(&mut self, r: Rewrite) {
        self.state.log.push(r.describe());
        self.state.rewrites.push(r);
        self.compiled = None;
    }

    /// Remove recorded rewrite `index`.
    pub fn remove_rewrite(&mut self, index: usize) -> Result<Rewrite, String> {
        if index >= self.state.rewrites.len() {
            return Err(format!(
                "no rewrite {index} ({} recorded)",
                self.state.rewrites.len()
            ));
        }
        let r = self.state.rewrites.remove(index);
        self.state
            .log
            .push(format!("removed rewrite: {}", r.describe()));
        self.compiled = None;
        Ok(r)
    }

    /// The live network as a [`Variant`].
    pub fn current_variant(&self) -> Variant {
        Variant {
            params: self.state.params.clone(),
            lowrank: self.state.lowrank.clone(),
            rewrites: self.state.rewrites.clone(),
        }
    }

    /// A named version: `current`, `unedited` (package spaces: no overlay,
    /// no rewrites) or a checkpoint tag.
    pub fn variant(&self, tag: &str) -> Result<Variant, String> {
        match tag {
            "current" => Ok(self.current_variant()),
            "unedited" if self.state.package.is_some() => Ok(Variant::default()),
            t => {
                Ok(Variant {
                    params: self.state.checkpoints.get(t).cloned().ok_or_else(|| {
                        format!("no checkpoint `{t}` (or use current / unedited)")
                    })?,
                    lowrank: BTreeMap::new(),
                    rewrites: self
                        .state
                        .checkpoint_rewrites
                        .get(t)
                        .cloned()
                        .unwrap_or_default(),
                })
            }
        }
    }

    /// Make `v` the live network; returns the one it replaced.
    pub fn swap_variant(&mut self, v: Variant) -> Variant {
        let old = self.current_variant();
        if v.rewrites != self.state.rewrites {
            self.compiled = None;
        }
        self.state.lowrank = v.lowrank;
        self.state.rewrites = v.rewrites;
        self.set_params(v.params);
        old
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

    /// A trainee is attached ([`crate::steer`]).
    pub(crate) fn is_steered(&self) -> bool {
        self.steered.is_some()
    }

    /// Run the space's scoring command on the live weights (one forward pass).
    pub(crate) fn external_score(&mut self) -> Result<serde_json::Value, String> {
        let cmd = self
            .state
            .score_cmd
            .clone()
            .filter(|c| !c.is_empty())
            .ok_or("this space has no scoring command (--score-cmd)")?;
        self.charge("external_score")?;
        let path = std::env::temp_dir().join(format!("rlx-wsp-score-{}.json", std::process::id()));
        let weights: BTreeMap<&str, &[f32]> = self
            .state
            .params
            .iter()
            .map(|(n, t)| (n.as_str(), t.data.as_slice()))
            .collect();
        let text = serde_json::to_string(&weights).map_err(|e| e.to_string())?;
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        let out = std::process::Command::new(&cmd[0])
            .args(&cmd[1..])
            .arg(&path)
            .output();
        let _ = std::fs::remove_file(&path);
        let out = out.map_err(|e| format!("{}: {e}", cmd[0]))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            let tail: String = err
                .chars()
                .rev()
                .take(500)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            return Err(format!("scoring command failed ({}): {tail}", out.status));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let line = text
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        serde_json::from_str(line)
            .map_err(|e| format!("scoring command's last line is not JSON: {e}"))
    }

    pub(crate) fn charge(&mut self, what: &str) -> Result<(), String> {
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
        debug_assert!(!self.is_loaded(), "loaded graphs run through `trace`");
        debug_assert_eq!(obs.len(), BATCH * self.spec.obs_dim);
        if self.compiled.is_none() {
            let g = self.graph().expect("the task network builds");
            self.compiled = Some(Session::new(Device::Cpu).compile(g));
            self.dirty = self.state.params.keys().cloned().collect();
        }
        let c = self.compiled.as_mut().expect("compiled");
        for name in std::mem::take(&mut self.dirty) {
            c.set_param(&name, &self.state.params[&name].data);
        }
        c.run(&[("obs", obs)])
    }

    /// Run the network on hand-written observations (≤ `BATCH` rows).
    pub fn probe(&mut self, rows: &[Vec<f32>]) -> Result<Vec<ProbeRow>, String> {
        self.no_outputs("probe")?;
        if self.is_loaded() {
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
            || self.is_loaded()
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
        if self.state.env == EnvKind::Qa {
            let q = self.state.qa.clone().ok_or("qa task without questions")?;
            // The unedited model's score never changes: cache it.
            let unedited = self.state.params.is_empty() && self.state.rewrites.is_empty();
            let cache = unedited
                .then(|| {
                    crate::lm::heldout_cache_path(self.state.package.as_deref().unwrap_or(""), &q)
                })
                .flatten();
            if let Some(c) = &cache
                && let Ok(text) = std::fs::read_to_string(c)
                && let Ok(v) = serde_json::from_str::<Value>(&text)
            {
                return Ok(v);
            }
            let items = crate::lm::load_items(&q.heldout)?;
            let items = &items[..q.heldout_n.min(items.len())];
            let answers = self.qa_answer(items)?;
            let s = crate::lm::qa_summary(items, &answers, 0);
            let acc = s["accuracy"].as_f64().unwrap_or(0.0);
            let tokens = self.lm.last_tokens.clone();
            let r = crate::lm::qa_report(items, &answers, &tokens, q.max_tokens, 0);
            let v = serde_json::json!({
                "score": 100.0 * acc,
                "detail": format!("{:.1}% of {} held-out questions, {:.0}% gave a final answer, {:.0}% hit the token limit",
                    100.0 * acc, items.len(), 100.0 * s["gave_final_marker"].as_f64().unwrap_or(0.0),
                    100.0 * r["hit_token_limit"].as_f64().unwrap_or(0.0)),
                "accuracy": acc,
                "hit_token_limit": r["hit_token_limit"],
                "ends_in_loop": r["ends_in_loop"],
                "mean_tokens": r["mean_tokens"],
            });
            if let Some(c) = cache {
                let _ = std::fs::create_dir_all(c.parent().unwrap_or(std::path::Path::new(".")));
                let _ = std::fs::write(c, v.to_string());
            }
            return Ok(v);
        }
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
        let mut g = apply_rewrites(self.base_graph()?, &self.state.rewrites);
        let mut by_node: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for u in &self.masks {
            by_node.entry(u.node.as_str()).or_default().push(u.index);
        }
        for (node, idx) in by_node {
            let id =
                crate::graphops::find_node(&g, node).ok_or_else(|| format!("no node `{node}`"))?;
            g = crate::graphops::mask_channels(&g, id, &idx)?;
        }
        Ok(g)
    }

    /// The network without ablation masks.
    pub fn base_graph(&self) -> Result<rlx_ir::Graph, String> {
        if self.state.package.is_some() {
            let store = self.store.as_ref().ok_or_else(|| {
                self.store_error
                    .clone()
                    .unwrap_or_else(|| "package not open".into())
            })?;
            return store.function(self.state.function.as_deref()).cloned();
        }
        match &self.state.graph {
            Some(text) => rlx_ir::rlxmodel::from_text(text).map_err(|e| e.to_string()),
            None => Ok(self.state.mlp.build_graph(BATCH)),
        }
    }

    /// A loaded model with no task score (`EnvKind::Graph`).
    pub fn is_loaded(&self) -> bool {
        self.state.env == EnvKind::Graph
    }

    /// Zero these channels in every run until cleared (ablation).
    pub fn set_masks(&mut self, masks: Vec<crate::graphops::Unit>) {
        if masks != self.masks {
            self.masks = masks;
            self.compiled = None;
        }
    }

    /// Channel labels of a node: set by the task / a rewrite, the MLP's
    /// `l{k}.h{i}` and action names, or `node[i]`.
    pub fn unit_channel_labels(&self, node: &str) -> Result<Vec<String>, String> {
        if let Some(l) = self.state.unit_labels.get(node) {
            return Ok(l.clone());
        }
        let g = self.graph()?;
        let id = crate::graphops::find_node(&g, node)
            .ok_or_else(|| format!("no node `{node}` (see `list_nodes`)"))?;
        let w = crate::graphops::width(&g, id)
            .ok_or_else(|| format!("`{node}` has no static channel axis"))?;
        if self.state.graph.is_none() {
            if node == "logits" {
                return Ok(self.spec.actions.iter().map(|s| s.to_string()).collect());
            }
            if let Some(k) = node.strip_prefix('l').and_then(|s| s.strip_suffix(".h")) {
                return Ok((0..w).map(|i| format!("l{k}.h{i}")).collect());
            }
        }
        Ok((0..w).map(|i| format!("{node}[{i}]")).collect())
    }

    /// A unit given as `node[i]`, an MLP alias `l{k}.h{i}`, or a channel
    /// label (e.g. an action name for the output node).
    pub fn resolve_unit(&self, s: &str) -> Result<crate::graphops::Unit, String> {
        let g = self.graph()?;
        if let Some(u) = crate::graphops::parse_unit(s) {
            let id = crate::graphops::find_node(&g, &u.node)
                .ok_or_else(|| format!("no node `{}` (see `list_nodes`)", u.node))?;
            let w = crate::graphops::width(&g, id).unwrap_or(0);
            if u.index >= w {
                return Err(format!("{s}: channel {} out of range ({w})", u.index));
            }
            return Ok(u);
        }
        let mut named: Vec<String> = self.state.unit_labels.keys().cloned().collect();
        named.extend(crate::graphops::unit_nodes(&g));
        if let Some(out) = g.outputs.first().and_then(|o| g.node(*o).name.clone()) {
            named.push(out);
        }
        for node in named {
            if let Ok(labels) = self.unit_channel_labels(&node)
                && let Some(index) = labels.iter().position(|l| l == s)
            {
                return Ok(crate::graphops::Unit { node, index });
            }
        }
        Err(format!(
            "`{s}` is not a unit (use node[i], an alias like l0.h3, or a channel label)"
        ))
    }

    /// Freeze every derived label into `state.labels` / `state.unit_labels`
    /// (before a rewrite turns an MLP space into a graph space).
    fn freeze_labels(&mut self) -> Result<(), String> {
        let names: Vec<String> = self.state.params.keys().cloned().collect();
        for n in names {
            let axes = Axes {
                rows: self.row_labels(&n)?,
                cols: self.col_labels(&n)?,
            };
            self.state.labels.insert(n, axes);
        }
        let g = self.base_graph()?;
        let mut nodes = crate::graphops::unit_nodes(&g);
        if let Some(out) = g.outputs.first().and_then(|o| g.node(*o).name.clone()) {
            nodes.push(out);
        }
        for node in nodes {
            let l = self.unit_channel_labels(&node)?;
            self.state.unit_labels.insert(node, l);
        }
        Ok(())
    }

    /// Repeat the span `from → to` (any shape-preserving run of layers or
    /// blocks) `times` times. Task and loaded graphs are rewritten in place;
    /// with `clone_weights` each copy gets its own params, initialised from
    /// the originals. Packages record the repeat (shared weights) and replay
    /// it at every open. Checkpoints are dropped on graph spaces.
    pub fn repeat(
        &mut self,
        from: &str,
        to: &str,
        times: usize,
        clone_weights: bool,
    ) -> Result<Value, String> {
        let g = self.base_graph()?;
        let f = crate::graphops::find_node(&g, from)
            .ok_or_else(|| format!("no node `{from}` (see list_nodes)"))?;
        let t = crate::graphops::find_node(&g, to).ok_or_else(|| format!("no node `{to}`"))?;
        let body = crate::graphops::span(&g, f, t)?;
        if self.state.package.is_some() {
            if clone_weights {
                return Err("packages repeat with shared weights (cloned params would not be in the package)".into());
            }
            // Validate on the graph now; replayed at every open.
            crate::graphops::repeat_span(&g, f, t, times, false)?;
            self.state.rewrites.push(Rewrite::Repeat {
                from: from.into(),
                to: to.into(),
                times,
            });
            self.state
                .log
                .push(format!("repeat {from}..{to} x{times} (package)"));
            return Ok(serde_json::json!({
                "span_nodes": body.len(), "times": times, "applies": "at every open of the model",
                "note": "weights are shared across copies",
            }));
        }
        if !self.state.networks.is_empty() {
            return Err("repeat works on single-network spaces".into());
        }
        self.freeze_labels()?;
        let (g2, added) = crate::graphops::repeat_span(&g, f, t, times, clone_weights)?;
        let text = rlx_ir::rlxmodel::to_text(&g2).map_err(|e| e.to_string())?;
        for (new, orig) in &added {
            let p = self.param(orig)?.clone();
            if let Some(l) = self.state.labels.get(orig).cloned() {
                self.state.labels.insert(new.clone(), l);
            }
            self.state.params.insert(new.clone(), p);
        }
        self.state.graph = Some(text);
        let dropped = self.state.checkpoints.len();
        self.state.checkpoints.clear();
        self.masks.clear();
        self.compiled = None;
        self.dirty = self.state.params.keys().cloned().collect();
        self.state.log.push(format!("repeat {from}..{to} x{times}"));
        Ok(serde_json::json!({
            "span_nodes": body.len(), "times": times,
            "new_params": added.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            "copies_named": format!("<node>.rep1 .. <node>.rep{}", times - 1),
            "checkpoints_dropped": dropped,
        }))
    }

    /// Insert an identity linear map after node `after` (any graph): the
    /// function is unchanged and the new `[w, w]` param is free to program.
    /// The space becomes a graph space; checkpoints are dropped.
    pub fn insert_linear(&mut self, after: &str) -> Result<Value, String> {
        if !self.state.networks.is_empty() {
            return Err("insert_linear works on single-network spaces".into());
        }
        if self.state.package.is_some() {
            return Err("insert_linear adds a param the package does not have; use `repeat` or edit existing params".into());
        }
        self.freeze_labels()?;
        let g = self.base_graph()?;
        let id =
            crate::graphops::find_node(&g, after).ok_or_else(|| format!("no node `{after}`"))?;
        let w = crate::graphops::width(&g, id).ok_or("the node has no static channel axis")?;
        let mut k = 0;
        let param = loop {
            let p = format!("{after}.ins{k}.w");
            if !self.state.params.contains_key(&p) {
                break p;
            }
            k += 1;
        };
        let (g2, node) = crate::graphops::insert_linear(&g, id, &param)?;
        let text = rlx_ir::rlxmodel::to_text(&g2).map_err(|e| e.to_string())?;
        let mut eye = Tensor::zeros(w, w);
        for i in 0..w {
            eye.data[i * w + i] = 1.0;
        }
        let ch = self.unit_channel_labels(after)?;
        let new_ch: Vec<String> = (0..w).map(|i| format!("{node}[{i}]")).collect();
        self.state.labels.insert(
            param.clone(),
            Axes {
                rows: ch,
                cols: new_ch.clone(),
            },
        );
        self.state.unit_labels.insert(node.clone(), new_ch);
        self.state.params.insert(param.clone(), eye);
        self.state.graph = Some(text);
        let dropped = self.state.checkpoints.len();
        self.state.checkpoints.clear();
        self.masks.clear();
        self.compiled = None;
        self.dirty = self.state.params.keys().cloned().collect();
        self.state
            .log
            .push(format!("insert_linear after {after}: {param}"));
        Ok(serde_json::json!({
            "param": param, "node": node, "width": w,
            "note": "identity: the network computes the same function; edit the new param to program it",
            "checkpoints_dropped": dropped,
        }))
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
                    // Weights the node reads (directly or through a cast).
                    "params": n.inputs.iter().filter_map(|&i| crate::graphops::param_name(&g, i)).collect::<Vec<_>>(),
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
        let mlp_space = !self.is_loaded();
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
        let mlp = !self.is_loaded();
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
        if self.state.package.is_some() {
            return Err(format!(
                "`{what}` runs task networks; on a package space use `generate` or `qa_eval`"
            ));
        }
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
        masks: Vec<crate::graphops::Unit>,
        episodes: usize,
        seed: u64,
    ) -> Result<Scored, String> {
        let live = std::mem::replace(&mut self.state.params, params);
        self.dirty = self.state.params.keys().cloned().collect();
        let live_masks = self.masks.clone();
        self.set_masks(masks);
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
        } else if self.is_loaded() {
            Err("loaded graphs have no score to compare; use `trace`".to_string())
        } else {
            Ok(Scored::Returns(
                self.play(episodes, seed, 0).episode_returns,
            ))
        };
        self.state.params = live;
        self.dirty = self.state.params.keys().cloned().collect();
        self.set_masks(live_masks);
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
        self.compare_variants(a, Vec::new(), b, Vec::new(), episodes, seed, what)
    }

    /// [`Self::compare_params`] where each side may also zero channels.
    pub fn compare_variants(
        &mut self,
        a: BTreeMap<String, Tensor>,
        masks_a: Vec<crate::graphops::Unit>,
        b: BTreeMap<String, Tensor>,
        masks_b: Vec<crate::graphops::Unit>,
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
        let sa = self.score_with(a, masks_a, episodes, seed)?;
        let sb = self.score_with(b, masks_b, episodes, seed)?;
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

    /// Run `BATCH` observations and return the first output (logits) and the
    /// activations of `nodes`, each `[BATCH, width]` flattened (harness use;
    /// not budgeted). Works on any task graph with an `obs` input.
    pub fn activations(&mut self, obs: &[f32], nodes: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let mut g = self.graph()?;
        let mut outs = vec![*g.outputs.first().ok_or("the graph has no outputs")?];
        for n in nodes {
            outs.push(crate::graphops::find_node(&g, n).ok_or_else(|| format!("no node `{n}`"))?);
        }
        g.set_outputs(outs);
        let mut c = Session::new(Device::Cpu).compile(g);
        for (name, t) in &self.state.params {
            c.set_param(name, &t.data);
        }
        Ok(c.run(&[("obs", obs)]))
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
        self.graft_from(from, units, into, None)
    }

    /// [`Self::graft`] with the donor's own graph (a different architecture).
    pub fn graft_from(
        &mut self,
        from: &str,
        units: &[String],
        into: &[String],
        donor_graph: Option<rlx_ir::Graph>,
    ) -> Result<Value, String> {
        let src = match self.state.networks.get(from) {
            Some(p) => p.clone(),
            None => self.params_of(from)?,
        };
        if !into.is_empty() && into.len() != units.len() {
            return Err("`into` needs one target per unit".into());
        }
        // Source and target may differ in architecture: units may move
        // between nodes and widths may differ. Where a channel comes from and
        // goes to is read off each graph (`channel_params`); the source is
        // read with this space's graph (same-architecture donors) unless
        // `donor_graph` is given. Overlapping parts are copied, the rest of
        // the target's column / row zeroed, and every mismatch reported.
        let g = self.base_graph()?;
        let dg = donor_graph.unwrap_or_else(|| g.clone());
        let mut moves = Vec::new();
        for (n, u) in units.iter().enumerate() {
            let su =
                crate::graphops::parse_unit(u).ok_or_else(|| format!("`{u}` is not a unit"))?;
            let target = into.get(n).unwrap_or(u);
            let tu = self.resolve_unit(target)?;
            let sid = crate::graphops::find_node(&dg, &su.node)
                .ok_or_else(|| format!("the source has no node `{}`", su.node))?;
            let tid = crate::graphops::find_node(&g, &tu.node)
                .ok_or_else(|| format!("no node `{}`", tu.node))?;
            let (sc, tc) = (
                crate::graphops::channel_params(&dg, sid),
                crate::graphops::channel_params(&g, tid),
            );
            moves.push((su, sc, tu, tc));
        }
        let mut notes = Vec::new();
        for (su, sc, tu, tc) in &moves {
            let (i, j) = (su.index, tu.index);
            if let (Some(sw), Some(tw)) = (&sc.weight, &tc.weight) {
                let sw = src
                    .get(sw)
                    .ok_or_else(|| format!("the source has no param `{sw}`"))?
                    .clone();
                if i >= sw.cols {
                    return Err(format!("the source has no channel {su}"));
                }
                let rows = self.param(tw)?.rows;
                if sw.rows != rows {
                    notes.push(format!(
                        "{su}: fan-in {} -> {rows}, copied the first {}",
                        sw.rows,
                        sw.rows.min(rows)
                    ));
                }
                self.edit(tw, &format!("graft col {i}->{j} from {from}"), |t| {
                    for r in 0..t.rows {
                        t.data[r * t.cols + j] = if r < sw.rows { sw.at(r, i) } else { 0.0 };
                    }
                    Ok(())
                })?;
            } else {
                notes.push(format!(
                    "{su} -> {tu}: no producing weight found on one side; incoming not copied"
                ));
            }
            if let (Some(sb), Some(tb)) = (&sc.bias, &tc.bias)
                && let Some(sb) = src.get(sb).cloned()
            {
                self.edit(tb, &format!("graft bias {i}->{j} from {from}"), |t| {
                    t.data[j] = sb.data[i];
                    Ok(())
                })?;
            }
            for (sr, tr) in sc.readers.iter().zip(&tc.readers) {
                let so = src
                    .get(sr)
                    .ok_or_else(|| format!("the source has no param `{sr}`"))?
                    .clone();
                let cols = self.param(tr)?.cols;
                if so.cols != cols {
                    notes.push(format!(
                        "{su}: fan-out {} -> {cols}, copied the first {}",
                        so.cols,
                        so.cols.min(cols)
                    ));
                }
                self.edit(tr, &format!("graft row {i}->{j} from {from}"), |t| {
                    for c in 0..t.cols {
                        t.data[j * t.cols + c] = if c < so.cols { so.at(i, c) } else { 0.0 };
                    }
                    Ok(())
                })?;
            }
        }
        Ok(serde_json::json!({ "grafted": moves.len(), "from": from, "notes": notes }))
    }

    /// `ablate` tool: the live network vs the same with `units` zeroed
    /// (any node's channels, via a mask in the graph).
    pub fn ablate(
        &mut self,
        units: &[String],
        episodes: usize,
        seed: Option<u64>,
    ) -> Result<Value, String> {
        let off: Vec<crate::graphops::Unit> = units
            .iter()
            .map(|u| self.resolve_unit(u))
            .collect::<Result<_, _>>()?;
        let live = self.state.params.clone();
        let mut r = self.compare_variants(
            live.clone(),
            Vec::new(),
            live,
            off,
            episodes,
            seed,
            &format!("ablate {units:?}"),
        )?;
        r["a"] = "current".into();
        r["b"] = format!("current with {} unit(s) zeroed", units.len()).into();
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
