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
    ListParams,
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
    ListNodes,
    Trace {
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
    },
    Ablate {
        units: Vec<String>,
        #[serde(default = "default_compare")]
        episodes: usize,
        seed: Option<u64>,
    },
    Script {
        code: String,
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
        direction: Vec<f32>,
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
        /// Target units (`l0.h3`, output labels, or indices with `layer`).
        units: Vec<RowKey>,
        #[serde(default)]
        layer: Option<usize>,
        /// The block: `{"block": "relay" | "threshold" | "and" | "or" | "gate" | "compare", ...}`.
        spec: Value,
    },
    InsertLayer {
        after: usize,
        #[serde(default)]
        width: Option<usize>,
    },
    SampleImages {
        #[serde(default = "default_samples")]
        n: usize,
        label: Option<u8>,
        #[serde(default)]
        offset: usize,
    },
}

/// One weight entry addressed by index or label.
#[derive(Debug, Deserialize)]
pub struct Entry {
    pub row: RowKey,
    pub col: RowKey,
    pub value: f32,
}

/// A probe input: a list of active (=1) observation labels, or a dense vector.
#[derive(Debug, Deserialize)]
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
/// One-reply mode and similar: refuse tools outside `state.tools`.
fn allowed(ws: &WeightSpace, raw: &Value) -> Result<(), String> {
    if let (Some(list), Some(name)) = (&ws.state.tools, raw["tool"].as_str())
        && !list.iter().any(|t| t == name)
    {
        return Err(format!(
            "tool `{name}` is not available in this run (available: {})",
            list.join(", ")
        ));
    }
    Ok(())
}

pub fn handle_json(ws: &mut WeightSpace, call: &str) -> Value {
    if let Ok(raw) = serde_json::from_str::<Value>(call)
        && let Err(e) = allowed(ws, &raw)
    {
        return json!({ "error": e });
    }
    match serde_json::from_str::<Call>(call) {
        Ok(c) => match dispatch(ws, c) {
            Ok(v) => json!({ "ok": v }),
            Err(e) => json!({ "error": e }),
        },
        Err(e) => json!({ "error": format!("bad call: {e}") }),
    }
}

pub fn dispatch(ws: &mut WeightSpace, call: Call) -> Result<Value, String> {
    match call {
        Call::Describe => Ok(describe(ws)),
        Call::Status => Ok(json!({
            "budget": ws.state.budget,
            "used": ws.state.used,
            "remaining": ws.remaining(),
            "checkpoints": ws.state.checkpoints.keys().collect::<Vec<_>>(),
        })),
        Call::ListParams => Ok(Value::Array(
            ws.state
                .params
                .iter()
                .map(|(n, t)| {
                    let nz = t.data.iter().filter(|v| **v != 0.0).count();
                    json!({ "name": n, "rows": t.rows, "cols": t.cols, "nonzero": nz })
                })
                .collect(),
        )),
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
            ws.edit(&name, &format!("add_rank1 x{scale}"), |t| {
                if u.len() != t.rows || v.len() != t.cols {
                    return Err(format!(
                        "u needs {} values and v {}, got {} and {}",
                        t.rows,
                        t.cols,
                        u.len(),
                        v.len()
                    ));
                }
                for r in 0..t.rows {
                    for c in 0..t.cols {
                        t.data[r * t.cols + c] += scale * u[r] * v[c];
                    }
                }
                Ok(())
            })?;
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
        Call::ListNodes => Ok(Value::Array(ws.nodes()?)),
        Call::Trace {
            observation,
            inputs,
            nodes,
            max_values,
        } => {
            let feeds: Vec<(String, Vec<f32>)> = if ws.state.graph.is_some() {
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
            seed,
        } => ws.compare(&a, &b, episodes, seed),
        Call::Ablate {
            units,
            episodes,
            seed,
        } => ws.ablate(&units, episodes, seed),
        Call::Script { code, read, apply } => {
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
            side,
            only,
        } => {
            let norm2: f32 = direction.iter().map(|v| v * v).sum();
            if norm2 <= 0.0 {
                return Err("direction is zero".into());
            }
            let note = format!("project_out {side} ({} entries)", direction.len());
            ws.edit(&name, &note, |t| {
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
                let pick: Vec<usize> = if only.is_empty() {
                    (0..lines).collect()
                } else {
                    only.clone()
                };
                for &l in &pick {
                    let at = |k: usize| {
                        if side == "out" {
                            l * t.cols + k
                        } else {
                            k * t.cols + l
                        }
                    };
                    let dot: f32 = (0..len).map(|k| t.data[at(k)] * direction[k]).sum();
                    for (k, d) in direction.iter().enumerate() {
                        t.data[at(k)] -= dot / norm2 * d;
                    }
                }
                Ok(())
            })?;
            Ok(json!({ "projected": name }))
        }
        Call::Graft { from, units, into } => ws.graft(&from, &units, &into),
        Call::Inject { units, layer, spec } => {
            let block: crate::blocks::Block =
                serde_json::from_value(spec).map_err(|e| format!("bad block: {e}"))?;
            crate::blocks::inject(ws, block, units, layer)
        }
        Call::InsertLayer { after, width } => crate::blocks::insert_layer(ws, after, width),
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

fn dense_observation(ws: &WeightSpace, o: &Observation) -> Result<Vec<f32>, String> {
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
    if let Some(v) = crate::tasks::describe(ws) {
        return v;
    }
    let spec = ws.spec();
    let mlp = &ws.state.mlp;
    let labels = &spec.obs_labels;
    if let Some(text) = &ws.state.graph {
        let g = rlx_ir::rlxmodel::from_text(text).ok();
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
            "Every param with shape and nonzero count.",
            json!({}),
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
            "FREE. The network as an rlx graph: every non-parameter node with id, name, op, shape and input ids. Named nodes (MLP: l0.z pre-activation, l0.h activation, ..., logits) or `%id` can be traced.",
            json!({}),
            &[]
        ),
        tool(
            "trace",
            "COSTS 1 forward pass. Run the network once and return the outputs plus the activations of `nodes` (names or %id): shape, min/max/mean, largest entries and the first `max_values` values. MLP spaces take one `observation` (probe forms); loaded graphs take `inputs` {name: flat values}.",
            json!({ "observation": {}, "inputs": { "type": "object" }, "nodes": { "type": "array", "items": { "type": "string" } }, "max_values": { "type": "integer" } }),
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
            "COSTS 2 forward passes. Paired A/B test of checkpoint `a` vs `b` (default `current` = live weights) on the SAME episodes (control: per-episode return difference with standard error and win counts) or the same validation digits (classifier: accuracies and discordant counts). Use it to keep an edit only if it really helps.",
            json!({ "a": { "type": "string" }, "b": { "type": "string" }, "episodes": { "type": "integer" }, "seed": { "type": "integer" } }),
            &["a"]
        ),
        tool(
            "ablate",
            "COSTS 2 forward passes. Switch hidden units off (their incoming column and bias zeroed, so they output 0) and compare against the live weights on the same episodes / digits. Tests what a unit actually does.",
            json!({ "units": { "type": "array", "items": { "type": "string" }, "description": "unit names like l0.h3" }, "episodes": { "type": "integer" }, "seed": { "type": "integer" } }),
            &["units"]
        ),
        tool(
            "script",
            "FREE (tool calls it returns cost as usual). Run JavaScript in a sandbox with no file, network or host access, to COMPUTE WEIGHTS TO WRITE instead of typing them. The code is a function body; in scope: SPEC {obs_labels, actions, widths, params: {name: {rows, cols}}} (names and shapes only — scripts cannot see weight values). Return any JSON value; return {calls: [{tool: 'set_col', name: 'l0.w', col: 0, values: [...]}, ...]} to have those tool calls applied in order (apply=false to only see them). Rule: do not simulate the network (in a script or by hand from copied weights) — running it is what probe / rollout / eval / trace are for, and they are budgeted.",
            json!({ "code": { "type": "string" }, "apply": { "type": "boolean" } }),
            &["code"]
        ),
        tool(
            "inject",
            "FREE. Write a predefined block into `units` of one layer (their incoming column and bias are replaced; outgoing weights kept). Inputs are that layer's row labels. Activation-aware (tanh off = -1, ReLU off = 0); `gain` sets sharpness. spec.block = relay {inputs: [one per unit]} | threshold {inputs, weights?, theta} (fires when Σw·x > theta) | and / or {inputs, levels: '01' | 'pm1'} | gate {signals: [one per unit], gate: label, strength?} (passes a signal only while the 0/1 gate input is on) | compare {a: [...], b: [...], margin?} (units = [a>b, b>a]).",
            json!({ "units": { "type": "array" }, "layer": { "type": "integer" }, "spec": { "type": "object" } }),
            &["units", "spec"]
        ),
        tool(
            "insert_layer",
            "FREE. Insert a new hidden layer after hidden layer `after` that relays its input (the network's function is kept: exact for ReLU, ~1e-4 for tanh). Layers above shift up by one name (old l1 becomes l2); `width` adds spare zero units to program. Checkpoints are dropped.",
            json!({ "after": { "type": "integer" }, "width": { "type": "integer" } }),
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
            json!({ "name": name, "direction": floats, "side": { "enum": ["out", "in"] }, "only": { "type": "array", "items": { "type": "integer" } } }),
            &["name", "direction"]
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
