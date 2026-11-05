// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The JSON tool surface an LLM drives.
//!
//! A call is `{"tool": "<name>", ...args}`; the reply is `{"ok": <value>}` or
//! `{"error": "<message>"}`. [`tool_schemas`] returns the same tools as
//! `{name, description, input_schema}` objects, the shape LLM tool-use APIs
//! (and MCP) accept, so an agent loop can register them directly.

use crate::space::{RowKey, WeightSpace};
use crate::views;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
#[serde(tag = "tool", rename_all = "snake_case", deny_unknown_fields)]
pub enum Call {
    Describe,
    Status,
    ListParams {
        /// Package spaces: only names containing this.
        #[serde(default)]
        contains: Option<String>,
        #[serde(default)]
        offset: usize,
        #[serde(default = "default_list")]
        limit: usize,
    },
    Generate {
        prompts: Vec<String>,
        #[serde(default)]
        max_tokens: Option<usize>,
    },
    QaEval {
        #[serde(default = "default_qa_n")]
        n: usize,
        #[serde(default)]
        offset: usize,
        /// Answer the questions' perturbed twins (numbers changed, answers
        /// recomputed) instead — those that can be perturbed.
        #[serde(default)]
        perturbed: bool,
    },
    QaLoglik {
        #[serde(default = "default_loglik_n")]
        n: usize,
        #[serde(default)]
        offset: usize,
        #[serde(default = "default_mode")]
        mode: String,
    },
    ChannelBias {
        node: String,
        channels: Vec<RowKey>,
        #[serde(default)]
        value: Option<f32>,
        #[serde(default)]
        values: Option<Vec<f32>>,
    },
    TokenBias {
        #[serde(default)]
        tokens: Vec<String>,
        #[serde(default)]
        ids: Vec<u32>,
        #[serde(default)]
        value: Option<f32>,
        #[serde(default)]
        values: Option<Vec<f32>>,
        #[serde(default)]
        node: Option<String>,
    },
    Steer {
        node: String,
        vector: crate::edits::VecArg,
        #[serde(default = "one")]
        scale: f32,
    },
    Rewrites {
        #[serde(default)]
        remove: Option<usize>,
    },
    WriteKv {
        param: String,
        keys: ItemsArg,
        #[serde(default)]
        target: Option<ItemsArg>,
        #[serde(default)]
        delta: Option<crate::edits::VecArg>,
        #[serde(default = "one")]
        scale: f32,
        #[serde(default)]
        matmul: Option<String>,
        #[serde(default)]
        pool: Option<String>,
    },
    Sensitivity {
        tensors: Vec<String>,
        #[serde(default)]
        metric: Option<String>,
        #[serde(default = "default_mode")]
        mode: String,
        #[serde(default = "default_search_n")]
        n: usize,
        #[serde(default)]
        offset: usize,
        #[serde(default = "default_eps")]
        eps: f32,
    },
    Search {
        candidates: Vec<crate::measure::Candidate>,
        #[serde(default)]
        metric: Option<String>,
        #[serde(default = "default_mode")]
        mode: String,
        #[serde(default = "default_search_n")]
        n: usize,
        #[serde(default = "default_rounds")]
        rounds: usize,
        #[serde(default)]
        offset: usize,
        #[serde(default = "yes")]
        confirm: bool,
        #[serde(default)]
        apply_best: bool,
    },
    QaPractice {
        #[serde(default = "default_samples")]
        n: usize,
        #[serde(default)]
        offset: usize,
    },
    Stats {
        name: String,
    },
    Slice {
        name: String,
        rows: Option<(usize, usize)>,
        cols: Option<(usize, usize)>,
        row_contains: Option<String>,
        #[serde(default)]
        nonzero_only: bool,
    },
    Topk {
        name: String,
        #[serde(default = "default_k")]
        k: usize,
    },
    Norms {
        name: String,
        axis: String,
    },
    ExternalScore,
    TrainValidate,
    TrainSnapshot {
        tag: String,
    },
    Svd {
        name: String,
        #[serde(default = "default_svd_k")]
        k: usize,
        #[serde(default = "default_svd_top")]
        top: usize,
    },
    Set {
        name: String,
        entries: Vec<Entry>,
    },
    SetCol {
        name: String,
        col: RowKey,
        values: Vec<f32>,
    },
    SetRow {
        name: String,
        row: RowKey,
        values: Vec<f32>,
    },
    Write {
        name: String,
        data: Vec<f32>,
    },
    Fill {
        name: String,
        value: f32,
    },
    Scale {
        name: String,
        factor: f32,
    },
    AddRank1 {
        name: String,
        u: Vec<f32>,
        v: Vec<f32>,
        #[serde(default = "one")]
        scale: f32,
    },
    Checkpoint {
        tag: String,
    },
    Revert {
        tag: String,
    },
    Diff {
        tag: String,
    },
    Probe {
        observations: Vec<Observation>,
    },
    Rollout {
        #[serde(default = "default_episodes")]
        episodes: usize,
        seed: Option<u64>,
        #[serde(default)]
        trace_steps: usize,
    },
    Log {
        #[serde(default = "default_log")]
        last: usize,
    },
    Eval {
        #[serde(default = "default_eval")]
        n: usize,
    },
    ListNodes {
        #[serde(default)]
        contains: Option<String>,
        #[serde(default)]
        offset: usize,
        #[serde(default = "default_list")]
        limit: usize,
    },
    UseFunction {
        name: String,
    },
    NodeStats {
        node: String,
        #[serde(default)]
        texts: Option<Vec<String>>,
        #[serde(default)]
        observations: Option<Vec<Value>>,
        #[serde(default)]
        tensors: Option<Vec<Value>>,
        #[serde(default)]
        pool: Option<String>,
        #[serde(default = "default_k")]
        top: usize,
    },
    Contrast {
        node: String,
        a: ItemsArg,
        b: ItemsArg,
        name: String,
        #[serde(default)]
        pool: Option<String>,
    },
    Lens {
        node: String,
        #[serde(default)]
        texts: Option<Vec<String>>,
        #[serde(default)]
        observations: Option<Vec<Value>>,
        #[serde(default)]
        tensors: Option<Vec<Value>>,
        through: String,
        #[serde(default)]
        transpose: bool,
        #[serde(default = "default_k")]
        top: usize,
        #[serde(default)]
        pool: Option<String>,
    },
    Repeat {
        from: String,
        to: String,
        times: usize,
        #[serde(default)]
        clone_weights: bool,
    },
    Trace {
        /// Package spaces: texts (language models) to read.
        #[serde(default)]
        texts: Vec<String>,
        #[serde(default)]
        pool: Option<String>,
        /// MLP spaces: one observation (any probe form).
        observation: Option<Observation>,
        /// Graph spaces: `{input name: flat values}`.
        #[serde(default)]
        inputs: BTreeMap<String, Vec<f32>>,
        #[serde(default)]
        nodes: Vec<String>,
        #[serde(default = "default_trace_values")]
        max_values: usize,
    },
    Compose {
        a: String,
        b: String,
        #[serde(default)]
        transpose_a: bool,
        #[serde(default)]
        transpose_b: bool,
        #[serde(default = "default_view")]
        view: String,
        #[serde(default = "default_k")]
        k: usize,
        #[serde(default = "default_svd_top")]
        top: usize,
        #[serde(default = "default_axis")]
        axis: String,
    },
    Compare {
        a: String,
        #[serde(default = "default_current")]
        b: String,
        #[serde(default = "default_compare")]
        episodes: usize,
        seed: Option<u64>,
        /// QA: `loglik` (default) or `accuracy`.
        #[serde(default)]
        metric: Option<String>,
        #[serde(default = "default_mode")]
        mode: String,
        #[serde(default)]
        offset: usize,
        /// QA: also compare general-text log-likelihood (default: when the
        /// task has general texts).
        #[serde(default)]
        guard: Option<bool>,
    },
    Train {
        #[serde(default = "default_train_steps")]
        steps: usize,
        #[serde(default)]
        background: bool,
    },
    SetUpdater {
        code: String,
    },
    Analyze {
        code: String,
    },
    LoraSnapshot {
        tag: String,
    },
    TrainStatus,
    Jobs {
        #[serde(default)]
        collect: Option<usize>,
        #[serde(default)]
        wait: bool,
        #[serde(default)]
        cancel: Option<usize>,
    },
    Precision {
        #[serde(default = "default_bf16")]
        dtype: String,
    },
    Average {
        tags: Vec<String>,
        #[serde(default)]
        weights: Option<Vec<f32>>,
    },
    DevCheck {
        #[serde(default = "default_unedited")]
        a: String,
        #[serde(default = "default_current")]
        b: String,
        #[serde(default)]
        metric: Option<String>,
        #[serde(default = "default_mode")]
        mode: String,
    },
    Ablate {
        units: Vec<String>,
        #[serde(default = "default_compare")]
        episodes: usize,
        seed: Option<u64>,
        #[serde(default)]
        metric: Option<String>,
        #[serde(default = "default_mode")]
        mode: String,
        #[serde(default)]
        offset: usize,
    },
    Script {
        #[serde(default)]
        code: String,
        /// Keep this script under a name (an analysis program to rerun).
        #[serde(default)]
        save_as: Option<String>,
        /// Run a saved script instead of `code`.
        #[serde(default)]
        saved: Option<String>,
        /// Removed: scripts no longer see weights (see `script.rs`).
        #[serde(default)]
        read: Option<Value>,
        #[serde(default = "yes")]
        apply: bool,
    },
    Answer {
        value: Value,
        #[serde(default)]
        reason: Option<String>,
        /// Probability (0..1) that the answer is right.
        #[serde(default)]
        confidence: Option<f32>,
    },
    Switch {
        to: String,
    },
    ProjectOut {
        name: String,
        /// The direction, or `vector`: the name of a stored one (`contrast`).
        #[serde(default)]
        direction: Vec<f32>,
        #[serde(default)]
        vector: Option<String>,
        /// `out`: remove the direction from each row (v has `cols` entries);
        /// `in`: from each column (v has `rows` entries).
        #[serde(default = "default_side")]
        side: String,
        /// Restrict to these rows (`out`) / columns (`in`).
        #[serde(default)]
        only: Vec<usize>,
    },
    Graft {
        from: String,
        units: Vec<String>,
        #[serde(default)]
        into: Vec<String>,
    },
    Inject {
        /// Target units (`node[i]`, `l0.h3`, channel labels, or indices with `node`).
        units: Vec<RowKey>,
        #[serde(default)]
        node: Option<String>,
        /// The block: `{"block": "relay" | "threshold" | "and" | "or" | "gate" | "compare", ...}`.
        spec: Value,
    },
    InsertLinear {
        /// Node to insert an identity linear map after.
        after: String,
    },
    SampleImages {
        #[serde(default = "default_samples")]
        n: usize,
        label: Option<u8>,
        #[serde(default)]
        offset: usize,
    },
}

/// The inputs of a read: `texts` (language-model packages), `observations`
/// (task networks) or `tensors` (other packages).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItemsArg {
    #[serde(default)]
    pub texts: Option<Vec<String>>,
    #[serde(default)]
    pub observations: Option<Vec<Value>>,
    #[serde(default)]
    pub tensors: Option<Vec<Value>>,
    /// Language models: the prompts and completions of the last `qa_eval`
    /// (or QA comparison) whose answers were `correct` or `wrong`.
    #[serde(default)]
    pub last_eval: Option<String>,
}

impl ItemsArg {
    fn as_value(&self) -> Value {
        let mut m = serde_json::Map::new();
        if let Some(t) = &self.texts {
            m.insert("texts".into(), json!(t));
        }
        if let Some(o) = &self.observations {
            m.insert("observations".into(), json!(o));
        }
        if let Some(t) = &self.tensors {
            m.insert("tensors".into(), json!(t));
        }
        Value::Object(m)
    }

    fn len(&self) -> usize {
        self.texts
            .as_ref()
            .map(Vec::len)
            .or(self.observations.as_ref().map(Vec::len))
            .or(self.tensors.as_ref().map(Vec::len))
            .unwrap_or(0)
    }
}

/// Read `nodes` for `items` on whatever the space runs (the caller
/// charges the forward pass).
pub(crate) fn read_items(
    ws: &mut WeightSpace,
    items: &ItemsArg,
    nodes: &[String],
    pool: Option<&str>,
) -> Result<crate::pkgrun::Read, String> {
    let from_eval;
    let items = match items.last_eval.as_deref() {
        None => items,
        Some(which) => {
            let want = match which {
                "correct" => true,
                "wrong" => false,
                o => return Err(format!("last_eval `{o}`: use correct | wrong")),
            };
            let texts: Vec<String> = ws
                .lm
                .last_qa
                .iter()
                .filter(|x| x.2 == want)
                .take(64)
                .map(|(p, c, _)| format!("{p}{c}"))
                .collect();
            if texts.is_empty() {
                return Err(format!(
                    "the last evaluation has no {which} answers (run qa_eval first)"
                ));
            }
            from_eval = ItemsArg {
                texts: Some(texts),
                ..Default::default()
            };
            &from_eval
        }
    };
    let n = items.len();
    if n == 0 || n > 64 {
        return Err("give 1..=64 items (texts / observations / tensors)".into());
    }
    let (pkg_items, obs) = crate::pkgrun::parse_items(ws, &items.as_value())?;
    if !pkg_items.is_empty() {
        if ws.store().is_none() {
            return Err("texts / tensors are for package spaces; use observations".into());
        }
        ws.read_package(&pkg_items, nodes, &[], pool == Some("mean"))
    } else {
        ws.read_task(&obs, nodes)
    }
}

fn top_entries(v: &[f32], labels: &[String], k: usize) -> Vec<Value> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].abs().total_cmp(&v[a].abs()));
    idx.iter()
        .take(k)
        .map(|&i| {
            json!([
                labels.get(i).cloned().unwrap_or_else(|| i.to_string()),
                v[i]
            ])
        })
        .collect()
}

/// One weight entry addressed by index or label.
#[derive(Debug, Deserialize)]
pub struct Entry {
    pub row: RowKey,
    pub col: RowKey,
    pub value: f32,
}

/// A probe input: a list of active (=1) observation labels, or a dense vector.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Observation {
    Active {
        active: Vec<String>,
    },
    /// 28 text-art rows (classifier tasks), as `sample_images` returns them.
    Image {
        image: Vec<String>,
    },
    Dense(Vec<f32>),
}

fn default_qa_n() -> usize {
    16
}
fn default_loglik_n() -> usize {
    16
}
fn default_search_n() -> usize {
    8
}
fn default_rounds() -> usize {
    2
}
fn default_eps() -> f32 {
    0.05
}
fn default_train_steps() -> usize {
    50
}
fn default_bf16() -> String {
    "bf16".into()
}
fn default_unedited() -> String {
    "unedited".into()
}
fn default_mode() -> String {
    "solution".into()
}
fn default_list() -> usize {
    200
}
fn default_side() -> String {
    "out".into()
}
fn default_trace_values() -> usize {
    64
}
fn default_view() -> String {
    "svd".into()
}
fn default_axis() -> String {
    "rows".into()
}
fn default_current() -> String {
    "current".into()
}
fn default_compare() -> usize {
    128
}
fn yes() -> bool {
    true
}
fn default_eval() -> usize {
    1000
}
fn default_samples() -> usize {
    4
}
fn default_k() -> usize {
    20
}
fn default_svd_k() -> usize {
    4
}
fn default_svd_top() -> usize {
    8
}
fn one() -> f32 {
    1.0
}
fn default_episodes() -> usize {
    32
}
fn default_log() -> usize {
    20
}

/// Parse and dispatch one JSON call; always returns a reply object.
/// One-reply mode and similar: refuse tools outside `state.tools`, and
/// the run's denied tools.
pub(crate) fn allowed(ws: &WeightSpace, raw: &Value) -> Result<(), String> {
    if let (Some(list), Some(name)) = (&ws.state.tools, raw["tool"].as_str())
        && !list.iter().any(|t| t == name)
    {
        return Err(format!(
            "tool `{name}` is not available in this run (available: {})",
            list.join(", ")
        ));
    }
    if let Some(name) = raw["tool"].as_str()
        && ws.state.denied_tools.iter().any(|t| t == name)
    {
        return Err(format!(
            "tool `{name}` is not available in this run (denied: {})",
            ws.state.denied_tools.join(", ")
        ));
    }
    Ok(())
}

pub fn handle_json(ws: &mut WeightSpace, call: &str) -> Value {
    let reply = handle_json_unrecorded(ws, call);
    // The trail of the run (see `crate::explore`).
    if let Ok(raw) = serde_json::from_str::<Value>(call) {
        crate::explore::record(ws, &raw, &reply);
    }
    reply
}

fn handle_json_unrecorded(ws: &mut WeightSpace, call: &str) -> Value {
    if let Ok(raw) = serde_json::from_str::<Value>(call) {
        if let Err(e) = allowed(ws, &raw) {
            return json!({ "error": e });
        }
        if raw["async"].as_bool() == Some(true) {
            return match crate::jobs::submit(ws, raw) {
                Ok(v) => json!({ "ok": v }),
                Err(e) => json!({ "error": e }),
            };
        }
    }
    let name = serde_json::from_str::<Value>(call)
        .ok()
        .and_then(|v| v["tool"].as_str().map(String::from))
        .unwrap_or_default();
    let t0 = std::time::Instant::now();
    let reply = match serde_json::from_str::<Call>(call) {
        Ok(c) => match dispatch(ws, c) {
            Ok(v) => json!({ "ok": v }),
            Err(e) => json!({ "error": e }),
        },
        Err(e) => json!({ "error": format!("bad call: {e}") }),
    };
    *ws.state.tool_seconds.entry(name).or_default() += t0.elapsed().as_secs_f64();
    reply
}

/// Marks a read that found none of its nodes in any graph the package ran:
/// a wrong address, not a measurement, so its forward passes are refunded.
pub(crate) const NO_NODES: &str = "no graph run had all of";

pub fn dispatch(ws: &mut WeightSpace, call: Call) -> Result<Value, String> {
    let used = ws.state.used;
    let r = dispatch_charged(ws, call);
    if let Err(e) = &r
        && e.contains(NO_NODES)
        && ws.state.used > used
    {
        let n = ws.state.used - used;
        ws.state.used = used;
        ws.state
            .log
            .push(format!("refunded {n} forward pass(es): {e}"));
    }
    r
}

fn dispatch_charged(ws: &mut WeightSpace, call: Call) -> Result<Value, String> {
    match call {
        Call::Describe => Ok(describe(ws)),
        Call::Status => Ok(json!({
            "budget": ws.state.budget,
            "used": ws.state.used,
            "remaining": ws.remaining(),
            "checkpoints": ws.state.checkpoints.keys().collect::<Vec<_>>(),
            "token_budget": ws.state.token_budget,
            "tokens_used": ws.state.tokens_used,
            "dev_checks_left": ws.state.dev_checks_left,
        })),
        Call::ListParams {
            contains,
            offset,
            limit,
        } => {
            if let Some(store) = ws.store() {
                let rows: Vec<Value> = store
                    .names()
                    .filter(|n| contains.as_deref().is_none_or(|c| n.contains(c)))
                    .skip(offset)
                    .take(limit.min(1000))
                    .map(|n| {
                        let (shape, scheme) = store.entry(n).expect("listed");
                        json!({ "name": n, "shape": shape, "stored_as": scheme,
                                "editable": store.editable(n), "edited": ws.state.params.contains_key(n) })
                    })
                    .collect();
                return Ok(
                    json!({ "count": store.names().count(), "shown": rows.len(), "params": rows }),
                );
            }
            Ok(Value::Array(
                ws.state
                    .params
                    .iter()
                    .map(|(n, t)| {
                        let nz = t.data.iter().filter(|v| **v != 0.0).count();
                        json!({ "name": n, "rows": t.rows, "cols": t.cols, "nonzero": nz })
                    })
                    .collect(),
            ))
        }
        Call::Generate {
            prompts,
            max_tokens,
        } => {
            if ws.store().is_none() {
                return Err("`generate` is for language-model package spaces".into());
            }
            if prompts.is_empty() || prompts.len() > 32 {
                return Err("generate takes 1..=32 prompts".into());
            }
            let max = max_tokens
                .or_else(|| ws.state.qa.as_ref().map(|q| q.max_tokens))
                .unwrap_or(256)
                .min(1024);
            ws.charge(&format!("generate {} prompts", prompts.len()))?;
            let t0 = std::time::Instant::now();
            let outs = ws.lm_generate(&prompts, max)?;
            let tokens: Vec<usize> = outs.iter().map(|o| ws.count_tokens(o)).collect();
            Ok(json!({
                "completions": outs,
                "tokens": tokens,
                "hit_token_limit": tokens.iter().map(|t| *t + 1 >= max).collect::<Vec<_>>(),
                "ends_in_loop": outs.iter().map(|o| crate::lm::repeats(o)).collect::<Vec<_>>(),
                "seconds": t0.elapsed().as_secs_f32(),
            }))
        }
        Call::QaEval {
            n,
            offset,
            perturbed,
        } => {
            let q = ws
                .state
                .qa
                .clone()
                .ok_or("`qa_eval` is for question-answering tasks")?;
            let items = crate::lm::practice_items(&q)?;
            let n = n.clamp(1, 64);
            let mut pick: Vec<crate::lm::QaItem> =
                items.iter().skip(offset).take(n).cloned().collect();
            if pick.is_empty() {
                return Err(format!("only {} practice questions", items.len()));
            }
            if perturbed {
                pick = crate::checks::perturbed_set(&pick, 7 ^ offset as u64)
                    .into_iter()
                    .map(|(_, t)| t)
                    .collect();
                if pick.is_empty() {
                    return Err(
                        "none of these questions can be perturbed (no calculator steps)".into(),
                    );
                }
            }
            ws.charge(&format!(
                "qa_eval {} {}practice questions from {offset}",
                pick.len(),
                if perturbed { "perturbed " } else { "" }
            ))?;
            let t0 = std::time::Instant::now();
            let answers = ws.qa_answer(&pick)?;
            let tokens = ws.lm.last_tokens.clone();
            let mut r = crate::lm::qa_report(&pick, &answers, &tokens, q.max_tokens, 4);
            r["seconds"] = json!(t0.elapsed().as_secs_f32());
            Ok(r)
        }
        Call::QaLoglik { n, offset, mode } => {
            let items = crate::measure::practice(ws, n, offset)?;
            ws.charge(&format!(
                "qa_loglik {} practice questions from {offset} ({mode})",
                items.len()
            ))?;
            let t0 = std::time::Instant::now();
            let scores = ws.qa_loglik(&items, &mode, &[])?;
            let n = scores.len() as f64;
            Ok(json!({
                "mode": mode, "questions": scores.len(), "offset": offset,
                "mean_logprob_per_token": scores.iter().map(|s| s.0).sum::<f64>() / n,
                "mean_tokens": scores.iter().map(|s| s.1 as f64).sum::<f64>() / n,
                "per_question": scores.iter().map(|s| (s.0 * 1e4).round() / 1e4).collect::<Vec<_>>(),
                "seconds": t0.elapsed().as_secs_f32(),
                "note": "higher is better; compare versions with compare (metric loglik) on the same questions",
            }))
        }
        Call::ChannelBias {
            node,
            channels,
            value,
            values,
        } => crate::edits::channel_bias(ws, &node, &channels, value, values),
        Call::TokenBias {
            tokens,
            ids,
            value,
            values,
            node,
        } => crate::edits::token_bias(ws, &tokens, &ids, value, values, node),
        Call::Steer {
            node,
            vector,
            scale,
        } => crate::edits::steer(ws, &node, &vector, scale),
        Call::Rewrites { remove } => crate::edits::rewrites(ws, remove),
        Call::WriteKv {
            param,
            keys,
            target,
            delta,
            scale,
            matmul,
            pool,
        } => crate::edits::write_kv(
            ws,
            &param,
            &keys,
            target.as_ref(),
            delta.as_ref(),
            scale,
            matmul,
            pool.as_deref(),
        ),
        Call::Sensitivity {
            tensors,
            metric,
            mode,
            n,
            offset,
            eps,
        } => {
            let m = crate::measure::Metric::parse(ws, metric.as_deref())?;
            crate::measure::sensitivity(ws, &tensors, m, &mode, n, offset, eps)
        }
        Call::Search {
            candidates,
            metric,
            mode,
            n,
            rounds,
            offset,
            confirm,
            apply_best,
        } => {
            let m = crate::measure::Metric::parse(ws, metric.as_deref())?;
            crate::measure::search(
                ws,
                &candidates,
                m,
                &mode,
                n,
                rounds,
                offset,
                confirm,
                apply_best,
            )
        }
        Call::QaPractice { n, offset } => {
            let q = ws
                .state
                .qa
                .clone()
                .ok_or("`qa_practice` is for question-answering tasks")?;
            let items = crate::lm::practice_items(&q)?;
            Ok(json!({
                "total": items.len(),
                "held_back_for_dev_check": q.dev_n,
                "prompt_format": ws.qa_prompt("<question>")?,
                "items": items.iter().skip(offset).take(n.min(16)).map(|i| json!({ "question": i.question, "answer": i.gold })).collect::<Vec<_>>(),
            }))
        }
        Call::Stats { name } => views::stats(ws, &name),
        Call::Slice {
            name,
            rows,
            cols,
            row_contains,
            nonzero_only,
        } => views::slice(ws, &name, rows, cols, row_contains.as_deref(), nonzero_only),
        Call::Topk { name, k } => views::topk(ws, &name, k),
        Call::Norms { name, axis } => views::norms(ws, &name, &axis),
        Call::Svd { name, k, top } => views::svd(ws, &name, k, top),
        Call::ExternalScore => ws.external_score(),
        Call::Set { name, entries } => {
            let mut idx = Vec::with_capacity(entries.len());
            for e in &entries {
                idx.push((
                    ws.resolve_row(&name, &e.row)?,
                    ws.resolve_col(&name, &e.col)?,
                    e.value,
                ));
            }
            let n = idx.len();
            ws.edit(&name, &format!("set {n} entries"), |t| {
                for (r, c, v) in idx {
                    t.data[r * t.cols + c] = v;
                }
                Ok(())
            })?;
            Ok(json!({ "set": n }))
        }
        Call::SetCol { name, col, values } => {
            let c = ws.resolve_col(&name, &col)?;
            ws.edit(&name, &format!("set col {c}"), |t| {
                if values.len() != t.rows {
                    return Err(format!("need {} values, got {}", t.rows, values.len()));
                }
                for (r, v) in values.iter().enumerate() {
                    t.data[r * t.cols + c] = *v;
                }
                Ok(())
            })?;
            Ok(json!({ "col": c }))
        }
        Call::SetRow { name, row, values } => {
            let r = ws.resolve_row(&name, &row)?;
            ws.edit(&name, &format!("set row {r}"), |t| {
                if values.len() != t.cols {
                    return Err(format!("need {} values, got {}", t.cols, values.len()));
                }
                t.data[r * t.cols..(r + 1) * t.cols].copy_from_slice(&values);
                Ok(())
            })?;
            Ok(json!({ "row": r }))
        }
        Call::Write { name, data } => {
            ws.edit(&name, "write all", |t| {
                if data.len() != t.data.len() {
                    return Err(format!("need {} values, got {}", t.data.len(), data.len()));
                }
                t.data = data;
                Ok(())
            })?;
            Ok(json!("written"))
        }
        Call::Fill { name, value } => {
            ws.edit(&name, &format!("fill {value}"), |t| {
                t.data.fill(value);
                Ok(())
            })?;
            Ok(json!("filled"))
        }
        Call::Scale { name, factor } => {
            ws.edit(&name, &format!("scale {factor}"), |t| {
                t.data.iter_mut().for_each(|v| *v *= factor);
                Ok(())
            })?;
            Ok(json!("scaled"))
        }
        Call::AddRank1 { name, u, v, scale } => {
            let u: Vec<f32> = u.iter().map(|x| x * scale).collect();
            ws.edit_rank1(&name, u, v, &format!("add_rank1 x{scale}"))?;
            Ok(json!("added"))
        }
        Call::Checkpoint { tag } => {
            ws.checkpoint(&tag);
            Ok(json!({ "checkpoint": tag }))
        }
        Call::Revert { tag } => {
            ws.revert(&tag)?;
            Ok(json!({ "reverted": tag }))
        }
        Call::Diff { tag } => Ok(Value::Array(
            ws.diff(&tag)?
                .into_iter()
                .map(|(n, k, l2)| json!({ "name": n, "changed": k, "l2": l2 }))
                .collect(),
        )),
        Call::Probe { observations } => {
            let rows = observations
                .iter()
                .map(|o| dense_observation(ws, o))
                .collect::<Result<Vec<_>, _>>()?;
            serde_json::to_value(ws.probe(&rows)?).map_err(|e| e.to_string())
        }
        Call::Rollout {
            episodes,
            seed,
            trace_steps,
        } => serde_json::to_value(ws.rollout(episodes, seed, trace_steps.min(200))?)
            .map_err(|e| e.to_string()),
        Call::Log { last } => {
            let log = &ws.state.log;
            Ok(json!(log[log.len().saturating_sub(last)..]))
        }
        Call::Eval { n } => ws.eval(n),
        Call::ListNodes {
            contains,
            offset,
            limit,
        } => {
            let all = ws.nodes()?;
            let picked: Vec<Value> = all
                .into_iter()
                // By name, or by a weight the node reads (most nodes of a
                // packaged model have no name).
                .filter(|n| {
                    contains.as_deref().is_none_or(|c| {
                        n["name"].as_str().is_some_and(|s| s.contains(c))
                            || n["params"].as_array().is_some_and(|ps| {
                                ps.iter().any(|p| p.as_str().is_some_and(|s| s.contains(c)))
                            })
                    })
                })
                .collect();
            let functions: Option<Vec<String>> = ws
                .store()
                .and_then(|st| st.functions().ok())
                .map(|fs| fs.iter().map(|g| g.name.clone()).collect());
            let limit = limit.min(1000);
            Ok(json!({
                "function": ws.state.function,
                "functions": functions,
                "total": picked.len(),
                "offset": offset,
                "truncated": picked.len() > offset + limit,
                "nodes": picked.iter().skip(offset).take(limit).collect::<Vec<_>>(),
            }))
        }
        Call::UseFunction { name } => {
            let st = ws.store().ok_or("functions are for package spaces")?;
            st.function(Some(&name))?;
            ws.state.function = Some(name.clone());
            Ok(json!({ "function": name }))
        }
        Call::NodeStats {
            node,
            texts,
            observations,
            tensors,
            pool,
            top,
        } => {
            let items = ItemsArg {
                texts,
                observations,
                tensors,
                last_eval: None,
            };
            ws.charge(&format!("node_stats {node} on {} items", items.len()))?;
            let read = read_items(ws, &items, std::slice::from_ref(&node), pool.as_deref())?;
            let vecs: Vec<&Vec<f32>> = read.iter().map(|r| &r[0]).collect();
            let w = vecs[0].len();
            let n = vecs.len() as f32;
            let mean: Vec<f32> = (0..w)
                .map(|c| vecs.iter().map(|v| v[c]).sum::<f32>() / n)
                .collect();
            let std: Vec<f32> = (0..w)
                .map(|c| (vecs.iter().map(|v| (v[c] - mean[c]).powi(2)).sum::<f32>() / n).sqrt())
                .collect();
            let labels = ws.unit_channel_labels(&node).unwrap_or_default();
            let lab = |c: usize| {
                labels
                    .get(c)
                    .cloned()
                    .unwrap_or_else(|| format!("{node}[{c}]"))
            };
            let mut by_std: Vec<usize> = (0..w).collect();
            by_std.sort_by(|&a, &b| std[b].total_cmp(&std[a]));
            Ok(json!({
                "node": node, "items": vecs.len(), "channels": w,
                "norm_of_mean": mean.iter().map(|x| x * x).sum::<f32>().sqrt(),
                "largest_mean": top_entries(&mean, &labels, top),
                "most_variable": by_std.iter().take(top).map(|&c| {
                    let arg = (0..vecs.len()).max_by(|&a, &b| vecs[a][c].total_cmp(&vecs[b][c])).unwrap_or(0);
                    json!({ "channel": lab(c), "std": std[c], "mean": mean[c], "max_item": arg, "max": vecs[arg][c] })
                }).collect::<Vec<_>>(),
            }))
        }
        Call::Contrast {
            node,
            a,
            b,
            name,
            pool,
        } => {
            if ws.remaining() < 2 {
                return Err("contrast needs 2 forward passes".into());
            }
            ws.charge(&format!("contrast {node} (a)"))?;
            let ra = read_items(ws, &a, std::slice::from_ref(&node), pool.as_deref())?;
            ws.charge(&format!("contrast {node} (b)"))?;
            let rb = read_items(ws, &b, std::slice::from_ref(&node), pool.as_deref())?;
            let mean = |r: &crate::pkgrun::Read| -> Vec<f32> {
                let w = r[0][0].len();
                (0..w)
                    .map(|c| r.iter().map(|x| x[0][c]).sum::<f32>() / r.len() as f32)
                    .collect()
            };
            let (ma, mb) = (mean(&ra), mean(&rb));
            let d: Vec<f32> = ma.iter().zip(&mb).map(|(x, y)| x - y).collect();
            let norm = d.iter().map(|x| x * x).sum::<f32>().sqrt();
            let unit: Vec<f32> = d.iter().map(|x| x / norm.max(1e-12)).collect();
            let proj = |r: &crate::pkgrun::Read| -> Vec<f32> {
                r.iter()
                    .map(|x| x[0].iter().zip(&unit).map(|(a, b)| a * b).sum())
                    .collect()
            };
            let (pa, pb) = (proj(&ra), proj(&rb));
            let labels = ws.unit_channel_labels(&node).unwrap_or_default();
            ws.state.vectors.insert(name.clone(), d.clone());
            Ok(json!({
                "stored_as": name, "length": d.len(), "norm": norm,
                "mean_projection_a": pa.iter().sum::<f32>() / pa.len() as f32,
                "mean_projection_b": pb.iter().sum::<f32>() / pb.len() as f32,
                "separates": pa.iter().cloned().fold(f32::INFINITY, f32::min) > pb.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                "largest_channels": top_entries(&d, &labels, 10),
                "use": "project_out {name: <param>, vector: <this name>, side: out | in}",
            }))
        }
        Call::Lens {
            node,
            texts,
            observations,
            tensors,
            through,
            transpose,
            top,
            pool,
        } => {
            let items = ItemsArg {
                texts,
                observations,
                tensors,
                last_eval: None,
            };
            ws.charge(&format!("lens {node} through {through}"))?;
            let read = read_items(ws, &items, std::slice::from_ref(&node), pool.as_deref())?;
            let w = ws.param(&through)?.clone();
            let labels = if transpose {
                ws.row_labels(&through)?
            } else {
                ws.col_labels(&through)?
            };
            let mut rows = Vec::new();
            for r in &read {
                let v = &r[0];
                let out: Vec<f32> = if transpose {
                    if v.len() != w.cols {
                        return Err(format!(
                            "`{node}` has {} channels; {through} has {} columns",
                            v.len(),
                            w.cols
                        ));
                    }
                    (0..w.rows)
                        .map(|i| (0..w.cols).map(|j| w.at(i, j) * v[j]).sum())
                        .collect()
                } else {
                    if v.len() != w.rows {
                        return Err(format!(
                            "`{node}` has {} channels; {through} has {} rows",
                            v.len(),
                            w.rows
                        ));
                    }
                    let mut o = vec![0f32; w.cols];
                    for (i, x) in v.iter().enumerate() {
                        if *x != 0.0 {
                            for (j, oj) in o.iter_mut().enumerate() {
                                *oj += x * w.at(i, j);
                            }
                        }
                    }
                    o
                };
                let mut idx: Vec<usize> = (0..out.len()).collect();
                idx.sort_by(|&a, &b| out[b].total_cmp(&out[a]));
                let mut tops = Vec::new();
                for &i in idx.iter().take(top) {
                    let tok = ws.tokenizer().and_then(|c| c.decode(&[i as u32]).ok());
                    tops.push(json!({ "index": i, "label": labels.get(i), "token": tok, "value": out[i] }));
                }
                rows.push(json!({ "top": tops }));
            }
            Ok(json!({ "node": node, "through": through, "items": rows }))
        }
        Call::Repeat {
            from,
            to,
            times,
            clone_weights,
        } => ws.repeat(&from, &to, times, clone_weights),

        Call::Trace {
            texts,
            pool,
            observation,
            inputs,
            nodes,
            max_values,
        } => {
            if ws.store().is_some() {
                if texts.is_empty() && inputs.is_empty() {
                    return Err("package spaces trace `texts` (language models) or `inputs`".into());
                }
                let items: Vec<crate::pkgrun::Item> = if !texts.is_empty() {
                    texts
                        .iter()
                        .map(|t| crate::pkgrun::Item::Text(t.clone()))
                        .collect()
                } else {
                    vec![crate::pkgrun::Item::Tensors(
                        inputs
                            .iter()
                            .map(|(k, v)| (k.clone(), vec![1, v.len()], v.clone()))
                            .collect(),
                    )]
                };
                if items.len() > 32 {
                    return Err("at most 32 texts".into());
                }
                ws.charge(&format!(
                    "trace {} nodes on {} items",
                    nodes.len(),
                    items.len()
                ))?;
                let read = ws.read_package(&items, &nodes, &[], pool.as_deref() == Some("mean"))?;
                return Ok(Value::Array(
                    read.iter()
                        .map(|r| {
                            json!(nodes.iter().zip(r).map(|(n, v)| {
                                json!({ "node": n, "channels": v.len(),
                                        "mean": v.iter().sum::<f32>() / v.len().max(1) as f32,
                                        "norm": v.iter().map(|x| x * x).sum::<f32>().sqrt(),
                                        "largest": top_entries(v, &[], 8),
                                        "values": v.iter().take(max_values.min(4096)).collect::<Vec<_>>() })
                            }).collect::<Vec<_>>())
                        })
                        .collect(),
                ));
            }
            let feeds: Vec<(String, Vec<f32>)> = if ws.is_loaded() {
                inputs.into_iter().collect()
            } else {
                let o = observation.ok_or("MLP spaces trace one `observation`")?;
                vec![("obs".to_string(), dense_observation(ws, &o)?)]
            };
            let out = ws.trace(&feeds, &nodes)?;
            Ok(Value::Array(
                out.into_iter()
                    .map(|(label, dims, v)| {
                        let n = v.len().max(1) as f32;
                        let mean = v.iter().sum::<f32>() / n;
                        let (lo, hi) = v.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
                        let mut idx: Vec<usize> = (0..v.len()).collect();
                        idx.sort_by(|&a, &b| v[b].abs().total_cmp(&v[a].abs()));
                        json!({
                            "node": label, "shape": dims,
                            "min": lo, "max": hi, "mean": mean,
                            "nonzero": v.iter().filter(|x| **x != 0.0).count(),
                            "largest": idx.iter().take(8).map(|&i| json!([i, v[i]])).collect::<Vec<_>>(),
                            "values": v.iter().take(max_values.min(4096)).collect::<Vec<_>>(),
                        })
                    })
                    .collect(),
            ))
        }
        Call::Compose {
            a,
            b,
            transpose_a,
            transpose_b,
            view,
            k,
            top,
            axis,
        } => {
            let c = views::compose(ws, &a, &b, transpose_a, transpose_b)?;
            match view.as_str() {
                "stats" => views::stats_of(&c),
                "topk" => views::topk_of(&c, k),
                "norms" => views::norms_of(&c, &axis),
                "svd" => views::svd_of(&c, k.min(16), top),
                other => Err(format!("view `{other}`: use stats | topk | norms | svd")),
            }
        }
        Call::Compare {
            a,
            b,
            episodes,
            metric,
            mode,
            offset,
            guard,
            ..
        } if ws.state.env == crate::env::EnvKind::Qa => {
            let m = crate::measure::Metric::parse(ws, metric.as_deref())?;
            let (va, vb) = (ws.variant(&a)?, ws.variant(&b)?);
            let guard = guard.unwrap_or_else(|| crate::measure::has_guard(ws));
            if guard && ws.remaining() < 4 {
                return Err("compare with the general-text guard needs 4 forward passes (guard: false for 2)".into());
            }
            let mut r = crate::measure::compare_variants(
                ws,
                (va.clone(), Vec::new()),
                (vb.clone(), Vec::new()),
                m,
                &mode,
                episodes,
                offset,
                "compare",
            )?;
            if guard {
                r["general_guard"] = crate::measure::guard_variants(ws, &va, &vb, "compare")?;
            }
            Ok(r)
        }
        Call::DevCheck { a, b, metric, mode } => {
            let m = crate::measure::Metric::parse(ws, metric.as_deref())?;
            crate::measure::dev_check(ws, &a, &b, m, &mode)
        }
        Call::Compare {
            a,
            b,
            episodes,
            seed,
            ..
        } => ws.compare(&a, &b, episodes, seed),
        Call::Ablate {
            units,
            episodes,
            metric,
            mode,
            offset,
            ..
        } if ws.state.env == crate::env::EnvKind::Qa => {
            let masks: Vec<crate::graphops::Unit> = units
                .iter()
                .map(|u| ws.resolve_unit(u))
                .collect::<Result<_, _>>()?;
            let m = crate::measure::Metric::parse(ws, metric.as_deref())?;
            let live = ws.current_variant();
            crate::measure::compare_variants(
                ws,
                (live.clone(), Vec::new()),
                (live, masks),
                m,
                &mode,
                episodes,
                offset,
                "ablate",
            )
        }
        Call::Ablate {
            units,
            episodes,
            seed,
            ..
        } => ws.ablate(&units, episodes, seed),
        Call::Train { steps, .. } if ws.is_steered() => crate::steer::train(ws, steps),
        Call::SetUpdater { code } if ws.is_steered() => crate::steer::set_updater(ws, &code),
        Call::Analyze { code } if ws.is_steered() => crate::steer::analyze(ws, &code),
        Call::TrainStatus if ws.is_steered() => crate::steer::status(ws),
        Call::TrainValidate => crate::steer::validate(ws),
        Call::TrainSnapshot { tag } => crate::steer::snapshot(ws, &tag),
        Call::Train { steps, background } => crate::opal::train(ws, steps, background),
        Call::SetUpdater { code } => crate::opal::set_updater(ws, &code),
        Call::Analyze { code } => crate::opal::analyze(ws, &code),
        Call::LoraSnapshot { tag } => crate::opal::snapshot(ws, &tag),
        Call::TrainStatus => crate::opal::status(ws),
        Call::Jobs {
            collect,
            wait,
            cancel,
        } => crate::jobs::jobs(ws, collect, wait, cancel),
        Call::Precision { dtype } => crate::checks::precision(ws, &dtype),
        Call::Average { tags, weights } => {
            let w = weights.unwrap_or_else(|| vec![1.0 / tags.len().max(1) as f32; tags.len()]);
            ws.average(&tags, &w)
        }
        Call::Script {
            code,
            save_as,
            saved,
            read,
            apply,
        } => {
            let code = match saved {
                Some(n) => ws
                    .state
                    .programs
                    .get(&n)
                    .cloned()
                    .ok_or_else(|| format!("no saved script `{n}`"))?,
                None if code.trim().is_empty() => return Err("give `code` or `saved`".into()),
                None => code,
            };
            if read.is_some() {
                return Err(
                    "scripts cannot read weights (that would let them run the network \
                            outside the forward-pass budget); read weights with slice / topk / \
                            norms / svd / compose and put what you need into the code"
                        .into(),
                );
            }
            if let Err(e) = crate::script::audit_code(ws, &code) {
                ws.state.log.push(format!(
                    "script refused: {}",
                    e.split('.').next().unwrap_or("")
                ));
                return Err(e);
            }
            if let Some(n) = save_as {
                ws.state.programs.insert(n, code.clone());
            }
            let value = crate::script::run(ws, &code)?;
            let calls = value.get("calls").and_then(Value::as_array).cloned();
            let applied = match (apply, calls) {
                (true, Some(calls)) => calls
                    .into_iter()
                    .map(|c| {
                        match allowed(ws, &c).and_then(|_| {
                            serde_json::from_value::<Call>(c).map_err(|e| format!("bad call: {e}"))
                        }) {
                            Ok(Call::Script { .. }) => {
                                json!({ "error": "scripts cannot call `script`" })
                            }
                            Ok(call) => match dispatch(ws, call) {
                                Ok(v) => json!({ "ok": v }),
                                Err(e) => json!({ "error": e }),
                            },
                            Err(e) => json!({ "error": e }),
                        }
                    })
                    .collect(),
                _ => Vec::new(),
            };
            Ok(json!({ "value": value, "applied": applied }))
        }
        Call::Answer {
            value,
            reason,
            confidence,
        } => {
            if let Some(c) = confidence
                && !(0.0..=1.0).contains(&c)
            {
                return Err("confidence must be in 0..1".into());
            }
            ws.state.log.push(format!("answer {value}"));
            ws.state.answer =
                Some(json!({ "value": value, "reason": reason, "confidence": confidence }));
            Ok(json!({ "recorded": value, "note": "the last answer counts" }))
        }
        Call::Switch { to } => ws.switch(&to).map(|_| json!({ "active": to })),
        Call::ProjectOut {
            name,
            direction,
            vector,
            side,
            only,
        } => {
            let direction = match vector {
                Some(v) => ws
                    .state
                    .vectors
                    .get(&v)
                    .cloned()
                    .ok_or_else(|| format!("no stored vector `{v}`"))?,
                None => direction,
            };
            let norm2: f32 = direction.iter().map(|v| v * v).sum();
            if norm2 <= 0.0 {
                return Err("direction is zero".into());
            }
            let t = ws.param(&name)?.clone();
            let (len, lines) = match side.as_str() {
                "out" => (t.cols, t.rows),
                "in" => (t.rows, t.cols),
                _ => return Err("side must be `out` or `in`".into()),
            };
            if direction.len() != len {
                return Err(format!("direction needs {len} entries for side `{side}`"));
            }
            if let Some(bad) = only.iter().find(|&&i| i >= lines) {
                return Err(format!("`only` index {bad} out of range ({lines})"));
            }
            let unit: Vec<f32> = direction.iter().map(|d| d / norm2.sqrt()).collect();
            let keep = |l: usize| only.is_empty() || only.contains(&l);
            // W ← W − (W v̂) v̂ᵀ (out) or W − v̂ (v̂ᵀ W) (in): one rank-1 update.
            let (u, v) = if side == "out" {
                let u: Vec<f32> = (0..t.rows)
                    .map(|r| {
                        if keep(r) {
                            -(0..t.cols).map(|c| t.at(r, c) * unit[c]).sum::<f32>()
                        } else {
                            0.0
                        }
                    })
                    .collect();
                (u, unit)
            } else {
                let v: Vec<f32> = (0..t.cols)
                    .map(|c| {
                        if keep(c) {
                            -(0..t.rows).map(|r| unit[r] * t.at(r, c)).sum::<f32>()
                        } else {
                            0.0
                        }
                    })
                    .collect();
                (unit, v)
            };
            ws.edit_rank1(&name, u, v, &format!("project_out {side}"))?;
            Ok(json!({ "projected": name }))
        }
        Call::Graft { from, units, into } => ws.graft(&from, &units, &into),
        Call::Inject { units, node, spec } => {
            let block: crate::blocks::Block =
                serde_json::from_value(spec).map_err(|e| format!("bad block: {e}"))?;
            crate::blocks::inject(ws, block, units, node)
        }
        Call::InsertLinear { after } => ws.insert_linear(&after),
        Call::SampleImages { n, label, offset } => {
            if !ws.state.env.is_classifier() {
                return Err("`sample_images` is for classification tasks".into());
            }
            let val = &crate::mnist::data()?.val;
            let picked: Vec<Value> = (0..val.len())
                .filter(|&i| label.is_none_or(|l| val.labels[i] == l))
                .skip(offset)
                .take(n.min(16))
                .map(|i| {
                    json!({ "index": i, "label": val.labels[i],
                            "image": crate::mnist::to_ascii(val.image(i)) })
                })
                .collect();
            Ok(json!({ "ramp": " .:-=+*#%@ (0.0 .. 1.0)", "images": picked }))
        }
    }
}

pub fn dense_observation(ws: &WeightSpace, o: &Observation) -> Result<Vec<f32>, String> {
    let spec = ws.spec();
    match o {
        Observation::Dense(v) => Ok(v.clone()),
        Observation::Image { image } => crate::mnist::from_ascii(image),
        Observation::Active { active } => {
            let mut v = vec![0f32; spec.obs_dim];
            for l in active {
                let i = spec
                    .obs_labels
                    .iter()
                    .position(|x| x == l)
                    .ok_or_else(|| format!("no observation labelled `{l}`"))?;
                v[i] = 1.0;
            }
            Ok(v)
        }
    }
}

fn describe(ws: &WeightSpace) -> Value {
    let mut v = describe_inner(ws);
    if let Some(last) = ws.state.lineage.last() {
        v["branched_from"] = json!({
            "note": "this run continues an earlier one: the live weights are that run's version below (also checkpoint `branch_start`); its history shows what was tried",
            "runs": ws.state.lineage.len(),
            "from": last["from"], "version": last["version"],
            "history": last["history"],
        });
    }
    // A task network rewritten into a general graph (e.g. by insert_linear)
    // is described from the graph, not the MLP recipe it started from.
    if ws.state.graph.is_some() && !ws.is_loaded() {
        v["network"] = network_from_graph(ws);
    }
    v
}

/// Params, units and how to address them, read off the graph.
fn network_from_graph(ws: &WeightSpace) -> Value {
    let g = ws.base_graph().ok();
    let units: Vec<Value> = g
        .iter()
        .flat_map(|g| {
            crate::graphops::unit_nodes(g).into_iter().map(move |n| {
                let w =
                    crate::graphops::find_node(g, &n).and_then(|id| crate::graphops::width(g, id));
                json!({ "node": n, "channels": w })
            })
        })
        .collect();
    json!({
        "params": ws.state.params.iter().map(|(n, t)| json!({ "name": n, "rows": t.rows, "cols": t.cols })).collect::<Vec<_>>(),
        "unit_nodes": units,
        "units": "address a unit as node[i] (or its label); list_nodes shows the graph",
        "action_choice": ws.state.protocol.describe(),
    })
}

fn describe_inner(ws: &WeightSpace) -> Value {
    if let Some(store) = ws.store() {
        let mut by_scheme: BTreeMap<String, usize> = BTreeMap::new();
        let mut elements = 0usize;
        for n in store.names() {
            let (shape, scheme) = store.entry(n).expect("listed");
            *by_scheme.entry(scheme.clone()).or_default() += 1;
            elements += shape.iter().product::<usize>();
        }
        let task = match &ws.state.qa {
            Some(q) => json!({
                "kind": "question answering",
                "goal": ws.state.goal.clone().unwrap_or_else(|| "Edit the model's weights so it answers \
                    the held-out questions correctly more often. No training, no gradients.".into()),
                "score": format!("accuracy on {} held-out questions you never see, answered greedily with \
                    the prompt format below, final answer read after '####' (else the last number)", q.heldout_n),
                "prompt_style": q.prompt_style,
                "prompt_format": if q.prompt_style == "plain" { json!(crate::lm::prompt("<question>")) } else { json!("the package's chat template (qa_practice shows it)") },
                "stop_at_answer": q.stop_at_answer,
                "repetition_penalty": q.repetition_penalty,
                "dev_split": json!({ "questions": q.dev_n, "dev_checks_left": ws.state.dev_checks_left,
                    "note": "held back from every tool; dev_check compares two versions on them and keeps a clear winner as checkpoint `verified`" }),
                "general_guard": q.general.is_some(),
                "token_budget": ws.state.token_budget,
                "max_tokens_per_answer": q.max_tokens,
            }),
            None => {
                json!({ "goal": ws.state.goal.clone().unwrap_or_else(|| "Explore and edit this model.".into()) })
            }
        };
        return json!({
            "task": task,
            "model": {
                "package": std::path::Path::new(&store.path).file_name().map(|f| f.to_string_lossy().to_string()),
                "tensors": store.names().count(),
                "parameters": elements,
                "stored_as": by_scheme,
                "how": "list_params (filter with `contains`) shows names and shapes. Every tensor is \
                        viewed as [prod(leading dims), last dim]; read it with slice / topk / norms / \
                        svd / stats / compose, edit it with set / set_row / set_col / scale / add_rank1 \
                        / project_out / write_kv. Only edited tensors are stored. Graph tools \
                        (list_nodes — with the params each node reads —, trace, node_stats, contrast, \
                        lens, ablate, channel_bias, token_bias, steer, repeat) read and change the \
                        package's own graphs; unnamed nodes are `%id`.",
            },
            "forward_passes": "generate (up to 32 prompts), qa_eval (up to 64 practice questions) and qa_loglik cost one each; compare / ablate / dev_check two (four with the general guard); qa_practice is free",
            "budget": { "total": ws.state.budget, "remaining": ws.remaining() },
        });
    }
    if let Some(v) = crate::tasks::describe(ws) {
        return v;
    }
    let spec = ws.spec();
    let mlp = &ws.state.mlp;
    let labels = &spec.obs_labels;
    if ws.is_loaded() {
        let text = ws.state.graph.clone().unwrap_or_default();
        let g = rlx_ir::rlxmodel::from_text(&text).ok();
        let io = |want_input: bool| -> Vec<Value> {
            g.iter()
                .flat_map(|g| g.nodes())
                .filter_map(|n| match &n.op {
                    rlx_ir::Op::Input { name } if want_input => Some(json!({
                        "name": name,
                        "shape": n.shape.dims().iter().map(|d| format!("{d:?}")).collect::<Vec<_>>(),
                    })),
                    _ => None,
                })
                .collect()
        };
        return json!({
            "task": ws.state.goal.clone().unwrap_or_else(|| "Explore and edit this network.".into()),
            "inputs": io(true),
            "outputs": g.as_ref().map(|g| g.outputs.len()),
            "params": ws.state.params.iter().map(|(n, t)| json!({"name": n, "rows": t.rows, "cols": t.cols})).collect::<Vec<_>>(),
            "nodes": g.as_ref().map(|g| g.nodes().len()),
            "how": "Params are viewed as [prod(leading dims), last dim] with labels rI / cJ. \
                    `list_nodes` shows the graph; `trace` runs it on named inputs and returns \
                    outputs plus any named node (1 forward pass). probe / rollout / eval do not \
                    apply to a loaded graph.",
            "budget": { "total": ws.state.budget, "remaining": ws.remaining() },
        });
    }
    if ws.state.env.is_classifier() {
        return json!({
            "task": spec.description,
            "inputs": {
                "pixels": spec.obs_dim,
                "label_format": label_format(labels),
                "image_text_ramp": " .:-=+*#%@ maps 0.0 .. 1.0 (10 levels)",
            },
            "outputs": spec.actions,
            "network": {
                "widths": mlp.widths,
                "hidden_activation": mlp.act,
                "weights": mlp.n_weights(),
                "forward": "h0 = act(x @ l0.w + l0.b); h1 = act(h0 @ l1.w + l1.b); \
                            logits = h1 @ l2.w + l2.b; prediction = argmax(logits)",
                "layout": "lK.w is [fan_in, fan_out]: row = input unit, column = output unit. \
                           lK.b is [1, fan_out]. Rows of l0.w are pixel labels px(r=R,c=C); \
                           columns of the last layer are digit0..digit9; other rows/cols are \
                           unit names lK.hI.",
            },
            "budget": { "total": ws.state.budget, "remaining": ws.remaining() },
        });
    }
    json!({
        "task": "Weight-space programming. You read and write this network's weights directly; \
                 there is no gradient descent. Reading and editing are free. Running the network \
                 (probe, rollout) costs one forward pass from a fixed budget. The goal is the \
                 highest mean episode return under the greedy policy.",
        "environment": {
            "name": spec.name,
            "rules": spec.description,
            "obs_dim": spec.obs_dim,
            "obs_labels_first": labels.first(),
            "obs_labels_last": labels.last(),
            "obs_label_format": label_format(labels),
            "actions": spec.actions,
            "horizon": spec.horizon,
        },
        "network": {
            "widths": mlp.widths,
            "hidden_activation": mlp.act,
            "weights": mlp.n_weights(),
            "forward": "h0 = act(obs @ l0.w + l0.b); h1 = act(h0 @ l1.w + l1.b); ...; \
                        logits = h_last @ lN.w + lN.b",
            "action_choice": ws.state.protocol.describe(),
            "layout": "lK.w is [fan_in, fan_out]: row = input unit, column = output unit. \
                       lK.b is [1, fan_out]. Rows of l0.w are observation labels; columns of the \
                       last layer are action names; other rows/cols are unit names lK.hI.",
        },
        "budget": { "total": ws.state.budget, "remaining": ws.remaining() },
    })
}

/// The distinct label shapes, digits generalised to their key:
/// `fruit<(x=3,y=5)` → `fruit<(x=X,y=Y)`, in first-seen order.
fn label_format(labels: &[String]) -> String {
    let mut shapes: Vec<String> = Vec::new();
    for l in labels {
        let mut shape = String::with_capacity(l.len());
        let mut key = String::new();
        let mut in_value = false;
        for ch in l.chars() {
            if in_value && ch.is_ascii_digit() {
                continue;
            }
            if in_value {
                shape.push_str(&key.to_uppercase());
                in_value = false;
            }
            match ch {
                '=' => {
                    in_value = true;
                    shape.push('=');
                }
                '(' | ',' => {
                    key.clear();
                    shape.push(ch);
                }
                _ => {
                    key.push(ch);
                    shape.push(ch);
                }
            }
        }
        if in_value {
            shape.push_str(&key.to_uppercase());
        }
        if !shapes.contains(&shape) {
            shapes.push(shape);
        }
    }
    shapes.join(" | ")
}

/// The tools as `{name, description, input_schema}` objects.
pub fn tool_schemas() -> Value {
    let mut v = tool_schemas_inner();
    for t in v.as_array_mut().into_iter().flatten() {
        let name = t["name"].as_str().unwrap_or_default().to_string();
        if crate::jobs::ASYNC_TOOLS.contains(&name.as_str()) {
            t["input_schema"]["properties"]["async"] = json!({ "type": "boolean",
                "description": "run in the background: returns a job id at once (passes charged now); collect with `jobs`" });
        }
    }
    v
}

fn tool_schemas_inner() -> Value {
    let name = json!({ "type": "string", "description": "param name, e.g. l0.w or l0.b" });
    let key = json!({ "description": "index or label", "anyOf": [{ "type": "integer" }, { "type": "string" }] });
    let floats = json!({ "type": "array", "items": { "type": "number" } });
    let range =
        json!({ "type": "array", "items": { "type": "integer" }, "minItems": 2, "maxItems": 2 });
    let tool = |n: &str, d: &str, props: Value, req: &[&str]| {
        json!({ "name": n, "description": d,
                "input_schema": { "type": "object", "properties": props, "required": req } })
    };
    json!([
        tool(
            "describe",
            "Task, environment rules, observation labels, network layout and budget.",
            json!({}),
            &[]
        ),
        tool(
            "status",
            "Forward-pass budget and checkpoint tags.",
            json!({}),
            &[]
        ),
        tool(
            "list_params",
            "Every param with its shape (package spaces: filter with `contains`; shows storage type and whether you edited it).",
            json!({ "contains": { "type": "string" } }),
            &[]
        ),
        tool(
            "stats",
            "min/max/mean/std/frobenius/nonzero of one param.",
            json!({ "name": name }),
            &["name"]
        ),
        tool(
            "slice",
            "Raw labelled values (max 4096). Filter rows by label substring, e.g. row_contains=\"ball>\" or \"y=8)\".",
            json!({ "name": name, "rows": range, "cols": range, "row_contains": { "type": "string" }, "nonzero_only": { "type": "boolean" } }),
            &["name"]
        ),
        tool(
            "topk",
            "Largest-magnitude entries with row/col labels.",
            json!({ "name": name, "k": { "type": "integer" } }),
            &["name"]
        ),
        tool(
            "norms",
            "L2 norm per row or column, largest first.",
            json!({ "name": name, "axis": { "enum": ["rows", "cols"] } }),
            &["name", "axis"]
        ),
        tool(
            "external_score",
            "1 PASS. Loaded graphs with a scoring command: score the live weights with it and return its JSON. An evaluation set, not the final test.",
            json!({}),
            &[]
        ),
        tool(
            "svd",
            "Leading singular values with the heaviest row/col labels of each singular vector.",
            json!({ "name": name, "k": { "type": "integer" }, "top": { "type": "integer" } }),
            &["name"]
        ),
        tool(
            "set",
            "Set individual entries; row/col by index or label.",
            json!({ "name": name, "entries": { "type": "array", "items": { "type": "object",
                     "properties": { "row": key, "col": key, "value": { "type": "number" } }, "required": ["row", "col", "value"] } } }),
            &["name", "entries"]
        ),
        tool(
            "set_col",
            "Overwrite one column (all fan_in weights of one output unit).",
            json!({ "name": name, "col": key, "values": floats }),
            &["name", "col", "values"]
        ),
        tool(
            "set_row",
            "Overwrite one row (all outgoing weights of one input unit).",
            json!({ "name": name, "row": key, "values": floats }),
            &["name", "row", "values"]
        ),
        tool(
            "write",
            "Overwrite a whole param, row-major.",
            json!({ "name": name, "data": floats }),
            &["name", "data"]
        ),
        tool(
            "fill",
            "Set every entry of a param to one value.",
            json!({ "name": name, "value": { "type": "number" } }),
            &["name", "value"]
        ),
        tool(
            "scale",
            "Multiply a param by a factor.",
            json!({ "name": name, "factor": { "type": "number" } }),
            &["name", "factor"]
        ),
        tool(
            "add_rank1",
            "W += scale * u v^T.",
            json!({ "name": name, "u": floats, "v": floats, "scale": { "type": "number" } }),
            &["name", "u", "v"]
        ),
        tool(
            "checkpoint",
            "Snapshot all weights under a tag.",
            json!({ "tag": { "type": "string" } }),
            &["tag"]
        ),
        tool(
            "revert",
            "Restore a snapshot.",
            json!({ "tag": { "type": "string" } }),
            &["tag"]
        ),
        tool(
            "diff",
            "Changed entries and L2 of the change vs a snapshot.",
            json!({ "tag": { "type": "string" } }),
            &["tag"]
        ),
        tool(
            "probe",
            "COSTS 1 forward pass. Run up to 32 observations; returns logits, chosen action and hidden activations. Each observation is {\"active\": [labels]}, {\"image\": [28 text-art rows]} (classification tasks), or a dense vector.",
            json!({ "observations": { "type": "array" } }),
            &["observations"]
        ),
        tool(
            "rollout",
            "COSTS 1 forward pass. Play episodes (multiple of 32) greedily; mean return, event counts, action mix and an optional per-step trace of episode 0.",
            json!({ "episodes": { "type": "integer" }, "seed": { "type": "integer" }, "trace_steps": { "type": "integer" } }),
            &[]
        ),
        tool(
            "log",
            "Recent edits and runs.",
            json!({ "last": { "type": "integer" } }),
            &[]
        ),
        tool(
            "eval",
            "Classification tasks. COSTS 1 forward pass. Accuracy, per-class accuracy and the confusion matrix (rows = true digit, cols = predicted) on the first n validation digits (default 1000, max 5000).",
            json!({ "n": { "type": "integer" } }),
            &[]
        ),
        tool(
            "list_nodes",
            "FREE. The network as an rlx graph: non-parameter nodes with id, name, op, shape and input ids (filter with `contains`, page with offset / limit). Named nodes or %id can be traced; a unit is node[i]. Package spaces also list their functions (see use_function).",
            json!({ "contains": { "type": "string" }, "offset": { "type": "integer" }, "limit": { "type": "integer" } }),
            &[]
        ),
        tool(
            "use_function",
            "FREE. Package spaces: which of the package's functions (graphs) list_nodes / trace / units address.",
            json!({ "name": { "type": "string" } }),
            &["name"]
        ),
        tool(
            "node_stats",
            "COSTS 1 forward pass. Per-channel statistics of one node over a batch of items (texts for language-model packages, observations for task networks): largest mean channels, most variable channels and the item that drives each. pool = last (default: the final step / last token) or mean.",
            json!({ "node": { "type": "string" }, "texts": { "type": "array" }, "observations": { "type": "array" }, "tensors": { "type": "array" }, "pool": { "enum": ["last", "mean"] }, "top": { "type": "integer" } }),
            &["node"]
        ),
        tool(
            "contrast",
            "COSTS 2 forward passes. The difference of a node's mean activation between item sets a and b (e.g. looping vs not, right vs wrong — {last_eval: correct} / {last_eval: wrong} take the last qa_eval's answers), stored under `name` as a direction for project_out or steer (vector: name). Reports how well it separates the sets.",
            json!({ "node": { "type": "string" }, "a": { "type": "object" }, "b": { "type": "object" }, "name": { "type": "string" }, "pool": { "enum": ["last", "mean"] } }),
            &["node", "a", "b", "name"]
        ),
        tool(
            "lens",
            "COSTS 1 forward pass. Read a node through a later linear map: its activation times a param (`through`, transposed with transpose=true), top outputs with labels — for a language model, read a hidden state through the output / embedding matrix to see which tokens it favours (decoded).",
            json!({ "node": { "type": "string" }, "through": { "type": "string" }, "transpose": { "type": "boolean" }, "texts": { "type": "array" }, "observations": { "type": "array" }, "top": { "type": "integer" }, "pool": { "enum": ["last", "mean"] } }),
            &["node", "through"]
        ),
        tool(
            "repeat",
            "FREE. Run the span from node `from` (its input) to node `to` (its output) `times` times in a row — any number of layers or blocks, any number of times; input and output shapes must match. Weights are shared, or cloned per copy (clone_weights; task / loaded graphs only) so copies can be edited apart. Copies are named <node>.repK. Spans that touch a cache / state variable are refused. Packages replay the repeat at every open.",
            json!({ "from": { "type": "string" }, "to": { "type": "string" }, "times": { "type": "integer" }, "clone_weights": { "type": "boolean" } }),
            &["from", "to", "times"]
        ),
        tool(
            "trace",
            "COSTS 1 forward pass. Run the network and return the activations of `nodes` (names or %id): shape, stats, largest entries and the first `max_values` values. Task networks take one `observation`; loaded graphs `inputs` {name: flat values}; language-model packages `texts` (≤32; pool = last token or mean over tokens).",
            json!({ "observation": {}, "inputs": { "type": "object" }, "texts": { "type": "array", "items": { "type": "string" } }, "pool": { "enum": ["last", "mean"] }, "nodes": { "type": "array", "items": { "type": "string" } }, "max_values": { "type": "integer" } }),
            &[]
        ),
        tool(
            "compose",
            "FREE. Summarise the matrix product A·B of two params (each optionally transposed), labelled by A's rows and B's columns — e.g. l0.w·l1.w maps inputs to layer-1 units through the linear part. view = svd (default) | topk | norms | stats.",
            json!({ "a": name, "b": name, "transpose_a": { "type": "boolean" }, "transpose_b": { "type": "boolean" }, "view": { "enum": ["svd", "topk", "norms", "stats"] }, "k": { "type": "integer" }, "top": { "type": "integer" }, "axis": { "enum": ["rows", "cols"] } }),
            &["a", "b"]
        ),
        tool(
            "compare",
            "COSTS 2 forward passes. Paired A/B test of checkpoint `a` vs `b` (default `current` = live weights; packages also `unedited`) on the SAME episodes (control: per-episode return difference with standard error and win counts), the same validation digits (classifier), or the same practice questions (QA: `episodes` questions from `offset`; metric loglik — default, teacher-forced log-probability of the reference, far less noisy — or accuracy; returns the paired mean difference, a 95% interval, p and a verdict). Keep an edit only if the verdict says it helps.",
            json!({ "a": { "type": "string" }, "b": { "type": "string" }, "episodes": { "type": "integer" }, "seed": { "type": "integer" }, "metric": { "enum": ["loglik", "accuracy"] }, "mode": { "enum": ["solution", "answer"] }, "offset": { "type": "integer" } }),
            &["a"]
        ),
        tool(
            "ablate",
            "COSTS 2 forward passes. Switch hidden units off (their incoming column and bias zeroed, so they output 0) and compare against the live weights on the same episodes / digits. Tests what a unit actually does.",
            json!({ "units": { "type": "array", "items": { "type": "string" }, "description": "unit names like l0.h3 or node[i]" }, "episodes": { "type": "integer" }, "seed": { "type": "integer" }, "metric": { "enum": ["loglik", "accuracy"] }, "mode": { "enum": ["solution", "answer"] }, "offset": { "type": "integer" } }),
            &["units"]
        ),
        tool(
            "script",
            "FREE (tool calls it returns cost as usual). Run JavaScript in a sandbox with no file, network or host access, to COMPUTE WEIGHTS TO WRITE instead of typing them. The code is a function body; in scope: SPEC {obs_labels, actions, widths, params: {name: {rows, cols}}} (names and shapes only — scripts cannot see weight values). Return any JSON value; return {calls: [{tool: 'set_col', name: 'l0.w', col: 0, values: [...]}, ...]} to have those tool calls applied in order (apply=false to only see them). Rule: do not simulate the network (in a script or by hand from copied weights) — running it is what probe / rollout / eval / trace are for, and they are budgeted.",
            json!({ "code": { "type": "string" }, "apply": { "type": "boolean" }, "save_as": { "type": "string", "description": "keep this script under a name" }, "saved": { "type": "string", "description": "run a saved script instead of code" } }),
            &[]
        ),
        tool(
            "generate",
            "Package spaces. COSTS 1 forward pass. Greedy completions of up to 32 prompts (plain text, no chat template).",
            json!({ "prompts": { "type": "array", "items": { "type": "string" } }, "max_tokens": { "type": "integer" } }),
            &["prompts"]
        ),
        tool(
            "qa_eval",
            "Question-answering tasks. COSTS 1 forward pass. Answer up to 64 practice questions (from `offset`) with the task's prompt; returns accuracy, outcomes (correct / wrong answer / loop / unfinished <think> / no final marker), and failures with the end of each completion. perturbed=true answers their twins with the numbers changed and the answers recomputed — a gain that holds on the originals but not the twins is memorised, not arithmetic.",
            json!({ "n": { "type": "integer" }, "offset": { "type": "integer" }, "perturbed": { "type": "boolean" } }),
            &[]
        ),
        tool(
            "train",
            "Runs that steer a LoRA training (the task says so). Advance training by `steps` (default 50; the run has a step budget). background=true trains in a thread while you keep working (train_status shows it; set_updater changes it from the next step). Returns losses, gated / skipped steps and the last step's per-site probes.",
            json!({ "steps": { "type": "integer" }, "background": { "type": "boolean" } }),
            &[]
        ),
        tool(
            "set_updater",
            "Training runs. FREE. Install the update policy: a JavaScript function body run before every step with STATS {step, recent_losses, last: {loss, sites: {<tensor>: {update_norm, update_to_weight, cos_prev_update}}}, sites, lr, optimizer, frozen}. Return any of {lr, site_lr: {pattern: rate}, freeze: [patterns], unfreeze: [patterns], gate: k (skip a batch whose loss > k × median recent loss), skip: true, optimizer: 'adamw' | 'muon'}. Empty code removes it. Tried once on the current STATS when installed.",
            json!({ "code": { "type": "string" } }),
            &["code"]
        ),
        tool(
            "analyze",
            "Training runs. FREE. Run your own analysis (JavaScript function body) over HISTORY (every step's record: step, loss, lr, optimizer, gated, skipped, sites probes) and STATS; returns its value.",
            json!({ "code": { "type": "string" } }),
            &["code"]
        ),
        tool(
            "lora_snapshot",
            "Training runs. FREE. Apply the current LoRA to the version training started from and keep it as checkpoint `tag` — it becomes the live weights, to measure with qa_loglik / compare / dev_check, or mix with average.",
            json!({ "tag": { "type": "string" } }),
            &["tag"]
        ),
        tool(
            "train_validate",
            "Steered training runs. 1 PASS. Measure the current weights on validation data; returns the run's own metrics.",
            json!({}),
            &[]
        ),
        tool(
            "train_snapshot",
            "Steered training runs. FREE. Make the run's current weights the live weights (for stats, svd, norms, …) and checkpoint `tag`. Training is unaffected.",
            json!({ "tag": { "type": "string" } }),
            &["tag"]
        ),
        tool(
            "train_status",
            "Training runs. FREE. Steps done and left, recent losses, gated / skipped steps, optimizer, rates, frozen sites, whether it trains in the background, and the last error.",
            json!({}),
            &[]
        ),
        tool(
            "jobs",
            "FREE. Background jobs: list them; `collect` a finished one's result (with `wait: true`, block until it finishes); `cancel` one (its passes stay spent). Start one by adding \"async\": true to qa_eval, qa_loglik, generate, trace, node_stats, lens, contrast, compare, ablate or sensitivity: its passes are charged at once, it measures the weights as they are at that moment, and you keep working meanwhile (one job at a time).",
            json!({ "collect": { "type": "integer" }, "wait": { "type": "boolean" }, "cancel": { "type": "integer" } }),
            &[]
        ),
        tool(
            "precision",
            "Package spaces. FREE. How much of each edit survives when the weights run in bf16 or f16: the share of the change rounding takes away, per edited tensor (a small edit can vanish below the dtype's resolution and look like it does nothing).",
            json!({ "dtype": { "enum": ["bf16", "f16"] } }),
            &[]
        ),
        tool(
            "average",
            "FREE. Set the live weights to a weighted average of versions (checkpoint tags, current, unedited; weights default equal) — e.g. halfway between the unedited model and an edit, or a mean of several checkpoints; then compare / dev_check it. The versions must share their graph rewrites.",
            json!({ "tags": { "type": "array", "items": { "type": "string" } }, "weights": floats }),
            &["tags"]
        ),
        tool(
            "dev_check",
            "Question-answering tasks with a dev split. COSTS 2 forward passes; only a few allowed (status shows how many are left). Compare version `a` (default `unedited`; or a checkpoint, or `current`) with `b` (default `current`) on the held-back dev questions no other tool shows — the check against fitting the practice questions. When b is clearly better it is kept as checkpoint `verified`, which the final score may use.",
            json!({ "a": { "type": "string" }, "b": { "type": "string" }, "metric": { "enum": ["loglik", "accuracy"] }, "mode": { "enum": ["solution", "answer"] } }),
            &[]
        ),
        tool(
            "qa_loglik",
            "Question-answering tasks. COSTS 1 forward pass. Teacher-forced score of the reference answers: mean log-probability per token of the worked solution (mode solution) or of only the final '#### n' line after the reference working (mode answer), on `n` practice questions (default 16) from `offset`. Continuous and deterministic: a much steadier signal than accuracy for deciding whether an edit helps.",
            json!({ "n": { "type": "integer" }, "offset": { "type": "integer" }, "mode": { "enum": ["solution", "answer"] } }),
            &[]
        ),
        tool(
            "channel_bias",
            "FREE. Add a constant to channels of a node (indices, channel labels or node[i]) at every position. On a node that is exactly x + bias it edits that bias param; elsewhere it is recorded as a graph rewrite (see `rewrites`).",
            json!({ "node": { "type": "string" }, "channels": { "type": "array", "items": key }, "value": { "type": "number" }, "values": floats }),
            &["node", "channels"]
        ),
        tool(
            "token_bias",
            "FREE. Language models: add `value` (negative suppresses) to the logits of tokens given as text (each must be one token) or ids — e.g. stop a model opening '<think>' or emitting a looping token. Node: `logits` by default.",
            json!({ "tokens": { "type": "array", "items": { "type": "string" } }, "ids": { "type": "array", "items": { "type": "integer" } }, "value": { "type": "number" }, "values": floats, "node": { "type": "string" } }),
            &[]
        ),
        tool(
            "steer",
            "FREE. Add scale · vector to a node at every position (activation steering) — vector inline or the name a `contrast` stored it under (e.g. right-minus-wrong answers at a middle layer). Recorded as a graph rewrite unless the node is exactly x + bias (then a bias edit).",
            json!({ "node": { "type": "string" }, "vector": { "description": "stored name or values", "anyOf": [{ "type": "string" }, floats] }, "scale": { "type": "number" } }),
            &["node", "vector"]
        ),
        tool(
            "rewrites",
            "FREE. List the recorded graph rewrites (repeat, channel_bias, token_bias, steer); `remove` deletes one by index. Checkpoints keep the rewrites in force.",
            json!({ "remove": { "type": "integer" } }),
            &[]
        ),
        tool(
            "write_kv",
            "COSTS 1 forward pass (2 with `target`). Closed-form rank-1 write: make weight `param` (read by x @ W) map the key — the MatMul's input averaged over `keys` items — to a target output: the MatMul's output averaged over `target` items, or the current output plus scale · `delta` (vector or stored name). W += k (v − kᵀW)ᵀ / kᵀk: exact for the key, proportionally smaller for other inputs; kept as a low-rank edit.",
            json!({ "param": name, "keys": { "type": "object", "description": "{texts: [...]} | {observations: [...]} | {tensors: [...]} | {last_eval: correct | wrong}" }, "target": { "type": "object", "description": "{texts: [...]} | {observations: [...]} | {tensors: [...]} | {last_eval: correct | wrong}" }, "delta": { "anyOf": [{ "type": "string" }, floats] }, "scale": { "type": "number" }, "matmul": { "type": "string", "description": "which MatMul when several read the param" }, "pool": { "enum": ["last", "mean"] } }),
            &["param", "keys"]
        ),
        tool(
            "sensitivity",
            "COSTS 1 + len(tensors) forward passes. Where edits matter: scale each tensor (≤16) by 1 + eps in turn and report how much the metric moves (paired against the unchanged network, on the same items), largest first. Gradient-free. metric: loglik (QA default) | accuracy | return (control).",
            json!({ "tensors": { "type": "array", "items": { "type": "string" } }, "metric": { "enum": ["loglik", "accuracy", "return"] }, "mode": { "enum": ["solution", "answer"] }, "n": { "type": "integer" }, "offset": { "type": "integer" }, "eps": { "type": "number" } }),
            &["tensors"]
        ),
        tool(
            "search",
            "COSTS several forward passes (reported before running; refused if over budget). Successive halving over up to 16 candidate edits, each {name, calls: [edit tool calls]}: round r scores the survivors and the live network on n·2^r items, keeps the better half; then (confirm, default) the winner vs the live network on FRESH items with paired statistics — picking the best of many on the same items overstates it. apply_best applies the winner only if it is clearly better.",
            json!({ "candidates": { "type": "array", "items": { "type": "object" } }, "metric": { "enum": ["loglik", "accuracy", "return"] }, "mode": { "enum": ["solution", "answer"] }, "n": { "type": "integer" }, "rounds": { "type": "integer" }, "offset": { "type": "integer" }, "confirm": { "type": "boolean" }, "apply_best": { "type": "boolean" } }),
            &["candidates"]
        ),
        tool(
            "qa_practice",
            "Question-answering tasks. FREE. Practice questions with their answers, and the prompt format.",
            json!({ "n": { "type": "integer" }, "offset": { "type": "integer" } }),
            &[]
        ),
        tool(
            "inject",
            "FREE. Write a predefined block into units (channels of ONE node: node[i], aliases like l0.h3, channel labels such as action names, or indices with `node`). The weight column and bias producing each channel are found in the graph and replaced; outgoing weights kept. Inputs are the row labels of that weight. Activation-aware (tanh off = -1, ReLU off = 0); `gain` sets sharpness. spec.block = relay {inputs: [one per unit]} | threshold {inputs, weights?, theta} (fires when Σw·x > theta) | and / or {inputs, levels: '01' | 'pm1'} | gate {signals: [one per unit], gate: label, strength?} (passes a signal only while the 0/1 gate input is on) | compare {a: [...], b: [...], margin?} (units = [a>b, b>a]).",
            json!({ "units": { "type": "array" }, "node": { "type": "string" }, "spec": { "type": "object" } }),
            &["units", "spec"]
        ),
        tool(
            "insert_linear",
            "FREE. Insert an identity linear map x @ W ([w, w], a new param) right after any node (see list_nodes): the network computes exactly the same function and W is free to program. Works on any graph. Checkpoints are dropped.",
            json!({ "after": { "type": "string" } }),
            &["after"]
        ),
        tool(
            "answer",
            "FREE. Submit the answer of a reading / audit task (the last one counts), with an optional reason and your confidence (probability 0..1 that it is right; scored for calibration).",
            json!({ "value": {}, "reason": { "type": "string" }, "confidence": { "type": "number" } }),
            &["value"]
        ),
        tool(
            "switch",
            "FREE. Multi-network tasks: make another network (e.g. B) active; every other tool acts on the active network.",
            json!({ "to": { "type": "string" } }),
            &["to"]
        ),
        tool(
            "project_out",
            "FREE. Remove a direction exactly: side `out` (direction has `cols` entries) makes every row orthogonal to it, W <- W (I - v v^T/|v|^2); side `in` (direction has `rows` entries) does the same for columns. `only` restricts to some rows (`out`) / columns (`in`).",
            json!({ "name": name, "direction": floats, "vector": { "type": "string", "description": "a stored direction (from contrast) instead of `direction`" }, "side": { "enum": ["out", "in"] }, "only": { "type": "array", "items": { "type": "integer" } } }),
            &["name"]
        ),
        tool(
            "graft",
            "FREE. Copy hidden units (incoming column, bias, outgoing row) from a checkpoint tag or another network into the live weights; `into` names target units (default: the same units).",
            json!({ "from": { "type": "string" }, "units": { "type": "array", "items": { "type": "string" } }, "into": { "type": "array", "items": { "type": "string" } } }),
            &["from", "units"]
        ),
        tool(
            "sample_images",
            "Classification tasks. FREE. Up to 16 labelled validation digits as 28 rows of text art (' .:-=+*#%@' = 0..1), optionally only one label, skipping `offset` matches. Copy and edit these rows to build probe inputs.",
            json!({ "n": { "type": "integer" }, "label": { "type": "integer" }, "offset": { "type": "integer" } }),
            &[]
        ),
    ])
}
