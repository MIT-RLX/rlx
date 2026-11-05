// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! How a run explored the weight space, as an SVG chart.
//!
//! Every tool call leaves a [`Step`] in the state's `trail`: the tool, its
//! arguments (long arrays and texts shortened), whether it worked, its
//! reply (shortened) and the edits live after it. The server records it,
//! so the trail is the same whichever agent drove the run. A run recorded
//! before the trail existed is read back from its transcript instead
//! (tool calls and their results), the live edits followed from the edit
//! calls.
//!
//! The chart: accuracy measured per call (practice and dev questions, the
//! held-out scores at the end), the paired changes the comparisons found
//! (one panel per metric, each with its 95% interval), and a lane per edit
//! for the calls it was live, with checkpoints, reverts and failed calls.

use crate::space::WeightSpace;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// One tool call of a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub tool: String,
    pub args: Value,
    pub ok: bool,
    pub result: Value,
    /// Forward passes used once the call returned.
    pub passes: Option<usize>,
    /// Edits live after the call (`None`: not known).
    pub live: Option<Vec<Live>>,
}

/// An edit live in the network: its family and what it is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Live {
    /// `activation` | `token` | `weight` | `structure`.
    pub kind: String,
    pub label: String,
}

/// Calls kept in a trail (the oldest first).
const MAX_TRAIL: usize = 4000;

/// Record a call and its reply in the trail.
pub fn record(ws: &mut WeightSpace, call: &Value, reply: &Value) {
    let tool = call["tool"].as_str().unwrap_or_default().to_string();
    if tool.is_empty() || ws.state.trail.len() >= MAX_TRAIL {
        return;
    }
    let mut args = call.clone();
    if let Some(o) = args.as_object_mut() {
        o.remove("tool");
    }
    let (ok, result) = match (reply.get("ok"), reply.get("error")) {
        (Some(v), _) => (true, shorten(v, 0)),
        (_, Some(e)) => (false, json!({ "error": shorten(e, 0) })),
        _ => (false, shorten(reply, 0)),
    };
    let step = Step {
        tool,
        args: shorten(&args, 0),
        ok,
        result,
        passes: Some(ws.state.used),
        live: Some(live_edits(ws)),
    };
    ws.state.trail.push(step);
}

/// `v` with long arrays, long strings and bulky fields cut down.
fn shorten(v: &Value, depth: usize) -> Value {
    const DROP: &[&str] = &[
        "completions",
        "failures",
        "examples",
        "values",
        "data",
        "u",
        "v",
    ];
    match v {
        Value::String(s) if s.chars().count() > 240 => {
            Value::String(format!("{}…", s.chars().take(240).collect::<String>()))
        }
        Value::Array(a) if depth > 4 => json!(format!("[{} items]", a.len())),
        Value::Array(a) => {
            let numbers = a.iter().all(Value::is_number);
            let keep = if numbers { 4 } else { 12 };
            let mut out: Vec<Value> = a.iter().take(keep).map(|x| shorten(x, depth + 1)).collect();
            if a.len() > keep {
                out.push(json!(format!("… {} items", a.len())));
            }
            Value::Array(out)
        }
        Value::Object(o) => Value::Object(
            o.iter()
                .filter(|(k, _)| !DROP.contains(&k.as_str()) || depth > 1)
                .map(|(k, x)| (k.clone(), shorten(x, depth + 1)))
                .collect(),
        ),
        x => x.clone(),
    }
}

/// The edits live in `ws`: its rewrites, and its edited tensors (package
/// spaces; a block's tensors of one kind together).
pub fn live_edits(ws: &WeightSpace) -> Vec<Live> {
    let mut out: Vec<Live> = ws
        .state
        .rewrites
        .iter()
        .map(|r| Live {
            kind: rewrite_kind(r).into(),
            label: r.describe(),
        })
        .collect();
    if ws.state.package.is_some() {
        let names = ws.state.params.keys().chain(ws.state.lowrank.keys());
        out.extend(weight_lanes(names.map(String::as_str)));
    }
    out
}

fn rewrite_kind(r: &crate::space::Rewrite) -> &'static str {
    match r {
        crate::space::Rewrite::ChannelBias { node, .. } if node == "logits" => "token",
        crate::space::Rewrite::ChannelBias { .. } | crate::space::Rewrite::AddVector { .. } => {
            "activation"
        }
        crate::space::Rewrite::Repeat { .. } => "structure",
    }
}

/// Tensors as lanes: `blk.N.<kind>` of several blocks is one lane.
fn weight_lanes<'a>(names: impl Iterator<Item = &'a str>) -> Vec<Live> {
    let mut groups: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for n in names {
        let parts: Vec<&str> = n.splitn(3, '.').collect();
        let key = match parts.as_slice() {
            [b, i, rest] if *b == "blk" && i.parse::<usize>().is_ok() => format!("blk.*.{rest}"),
            _ => n.to_string(),
        };
        let g = groups.entry(key).or_default();
        if !g.iter().any(|x| x == n) {
            g.push(n.to_string());
        }
    }
    groups
        .into_iter()
        .map(|(k, ns)| Live {
            kind: "weight".into(),
            label: if ns.len() == 1 {
                ns[0].clone()
            } else {
                format!("{} in {} blocks", k.trim_start_matches("blk.*."), ns.len())
            },
        })
        .collect()
}

// ── Reading a transcript (runs without a trail) ─────────────────────────

/// The calls of a transcript with `tool_use` / `tool_result` blocks (one
/// JSON event per line), the live edits followed from the edit calls.
pub fn from_transcript(text: &str, state: Option<&Value>) -> Vec<Step> {
    let mut steps: Vec<Step> = Vec::new();
    let mut open: std::collections::HashMap<String, usize> = Default::default();
    for line in text.lines() {
        let Ok(ev) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(blocks) = ev["message"]["content"].as_array() else {
            continue;
        };
        for b in blocks {
            match b["type"].as_str() {
                Some("tool_use") => {
                    let name = b["name"].as_str().unwrap_or_default();
                    let tool = name.rsplit("__").next().unwrap_or(name).to_string();
                    open.insert(b["id"].as_str().unwrap_or_default().into(), steps.len());
                    steps.push(Step {
                        tool,
                        args: shorten(&b["input"], 0),
                        ok: false,
                        result: Value::Null,
                        passes: None,
                        live: None,
                    });
                }
                Some("tool_result") => {
                    let Some(&i) = b["tool_use_id"].as_str().and_then(|id| open.get(id)) else {
                        continue;
                    };
                    let text = match &b["content"] {
                        Value::String(s) => s.clone(),
                        Value::Array(a) => a
                            .iter()
                            .filter_map(|c| c["text"].as_str())
                            .collect::<Vec<_>>()
                            .join(""),
                        _ => String::new(),
                    };
                    let parsed = serde_json::from_str::<Value>(&text).ok();
                    let failed = b["is_error"].as_bool() == Some(true)
                        || parsed.is_none()
                        || parsed.as_ref().is_some_and(|v| v.get("error").is_some());
                    steps[i].ok = !failed;
                    steps[i].result = match parsed {
                        Some(v) if !failed => shorten(&v, 0),
                        Some(v) => shorten(&json!({ "error": v.get("error").unwrap_or(&v) }), 0),
                        None => json!({ "error": shorten(&json!(text), 0) }),
                    };
                }
                _ => {}
            }
        }
    }
    follow_edits(&mut steps, state);
    steps
}

/// Live edits along `steps` from the edit calls that worked: rewrites by
/// the tools that add them, `rewrites {remove}`, checkpoints and reverts.
/// `state` (the run's `state.json`) gives the checkpoints the transcript
/// does not make (a branch's start, the versions it came with).
fn follow_edits(steps: &mut [Step], state: Option<&Value>) {
    let mut saved: std::collections::HashMap<String, (Vec<Live>, Vec<String>)> = Default::default();
    for (tag, params) in state
        .and_then(|s| s["checkpoints"].as_object())
        .into_iter()
        .flatten()
    {
        let tensors = params
            .as_object()
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
        let rewrites = state
            .and_then(|s| s["checkpoint_rewrites"][tag].as_array())
            .map(|rs| {
                rs.iter()
                    .filter_map(|r| serde_json::from_value::<crate::space::Rewrite>(r.clone()).ok())
                    .map(|r| Live {
                        kind: rewrite_kind(&r).into(),
                        label: r.describe(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        saved.insert(tag.clone(), (rewrites, tensors));
    }
    // The run starts where it branched from, else unedited.
    let (mut rewrites, mut tensors) = saved.get("branch_start").cloned().unwrap_or_default();
    for s in steps.iter_mut() {
        if s.ok {
            let a = &s.args;
            let str_of = |k: &str| a[k].as_str().unwrap_or("?").to_string();
            match s.tool.as_str() {
                "steer" => rewrites.push(Live {
                    kind: "activation".into(),
                    label: format!(
                        "steer {} by {} x {}",
                        str_of("node"),
                        a["scale"].as_f64().unwrap_or(1.0),
                        a["vector"].as_str().unwrap_or("a vector")
                    ),
                }),
                "token_bias" => rewrites.push(Live {
                    kind: "token".into(),
                    label: format!("bias tokens {} by {}", short(&a["tokens"]), a["value"]),
                }),
                "channel_bias" => rewrites.push(Live {
                    kind: if a["node"] == "logits" {
                        "token"
                    } else {
                        "activation"
                    }
                    .into(),
                    label: format!("bias channels of {}", str_of("node")),
                }),
                "repeat" => rewrites.push(Live {
                    kind: "structure".into(),
                    label: format!("repeat {}..{}", str_of("from"), str_of("to")),
                }),
                "rewrites" => {
                    if let Some(i) = a["remove"].as_u64().map(|i| i as usize)
                        && i < rewrites.len()
                    {
                        rewrites.remove(i);
                    }
                }
                "scale" | "set" | "set_row" | "set_col" | "write" | "fill" | "add_rank1"
                | "project_out" => {
                    let n = str_of("name");
                    if !tensors.contains(&n) {
                        tensors.push(n);
                    }
                }
                "write_kv" => {
                    let n = str_of("param");
                    if !tensors.contains(&n) {
                        tensors.push(n);
                    }
                }
                "checkpoint" => {
                    saved.insert(str_of("tag"), (rewrites.clone(), tensors.clone()));
                }
                "revert" => {
                    // A tag saved before the transcript starts: the
                    // unedited network (a run's `base`, `branch_start`).
                    let (r, t) = saved.get(&str_of("tag")).cloned().unwrap_or_default();
                    rewrites = r;
                    tensors = t;
                }
                _ => {}
            }
        }
        let mut live = rewrites.clone();
        live.extend(weight_lanes(tensors.iter().map(String::as_str)));
        s.live = Some(live);
    }
}

/// The family of an edit tool's change.
fn edit_kind(tool: &str) -> &'static str {
    match tool {
        "steer" | "channel_bias" | "inject" => "activation",
        "token_bias" => "token",
        "repeat" | "insert_linear" => "structure",
        "scale" | "set" | "set_row" | "set_col" | "write" | "fill" | "add_rank1" | "write_kv"
        | "project_out" | "average" | "graft" => "weight",
        _ => "other",
    }
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.chars().count() > 40 {
        format!("{}…", s.chars().take(40).collect::<String>())
    } else {
        s
    }
}

// ── The chart ───────────────────────────────────────────────────────────

/// A measured accuracy.
#[derive(Debug, Clone, Serialize)]
pub struct AccPoint {
    pub call: usize,
    pub accuracy: f64,
    /// Family of the version measured (`unedited` when no edit was live).
    pub kind: String,
    pub what: String,
}

/// A paired change `b − a` with its interval.
#[derive(Debug, Clone, Serialize)]
pub struct Delta {
    pub call: usize,
    pub metric: String,
    pub mean: f64,
    pub lo: f64,
    pub hi: f64,
    pub verdict: String,
    pub kind: String,
    pub what: String,
}

/// What a chart draws, also written beside it as its table.
#[derive(Debug, Clone, Serialize, Default)]
pub struct Plot {
    pub accuracy: Vec<AccPoint>,
    pub deltas: Vec<Delta>,
    /// `(label, kind, [(first call, last call)])` per edit.
    pub lanes: Vec<(String, String, Vec<(usize, usize)>)>,
    /// `(call, tool, tag)` of checkpoints and reverts.
    pub marks: Vec<(usize, String, String)>,
    /// `(call, tool, error)` of calls that failed.
    pub failed: Vec<(usize, String, String)>,
    pub calls: usize,
}

fn kind_of(live: &Option<Vec<Live>>) -> String {
    match live {
        None => "unknown".into(),
        // The newest rewrite (what the version adds), else its tensors.
        Some(l) => l
            .iter()
            .rev()
            .find(|e| e.kind != "weight")
            .or(l.last())
            .map_or_else(|| "unedited".into(), |e| e.kind.clone()),
    }
}

fn paired_delta(call: usize, metric: &str, p: &Value, kind: &str, what: String) -> Option<Delta> {
    let mean = p["mean_b_minus_a"].as_f64()?;
    let ci = p["ci95"].as_array();
    let lo = ci
        .and_then(|c| c.first())
        .and_then(Value::as_f64)
        .unwrap_or(mean);
    let hi = ci
        .and_then(|c| c.get(1))
        .and_then(Value::as_f64)
        .unwrap_or(mean);
    Some(Delta {
        call,
        metric: metric.to_string(),
        mean,
        lo,
        hi,
        verdict: p["verdict"].as_str().unwrap_or("").into(),
        kind: kind.into(),
        what,
    })
}

/// The points, intervals and lanes of `steps`.
pub fn plot(steps: &[Step]) -> Plot {
    let mut p = Plot {
        calls: steps.len(),
        ..Default::default()
    };
    for (i, s) in steps.iter().enumerate() {
        let call = i + 1;
        if !s.ok {
            let e = s.result["error"]
                .as_str()
                .map_or_else(|| s.result["error"].to_string(), str::to_string);
            p.failed.push((call, s.tool.clone(), e));
            continue;
        }
        let kind = kind_of(&s.live);
        // A collected background job is the call it ran.
        let (tool, r, args) = if s.tool == "jobs" && s.result["collected"]["result"].is_object() {
            let c = &s.result["collected"];
            (
                c["tool"].as_str().unwrap_or_default(),
                &c["result"],
                &Value::Null,
            )
        } else {
            (s.tool.as_str(), &s.result, &s.args)
        };
        let metric = r["metric"].as_str().unwrap_or("score");
        match tool {
            "qa_eval" => {
                if let Some(a) = r["accuracy"].as_f64() {
                    p.accuracy.push(AccPoint {
                        call,
                        accuracy: a,
                        kind: kind.clone(),
                        what: format!(
                            "qa_eval: {} practice questions from {}",
                            args["n"].as_u64().unwrap_or(0),
                            args["offset"].as_u64().unwrap_or(0)
                        ),
                    });
                }
            }
            "dev_check" | "compare" | "ablate" => {
                let (a, b) = (
                    args["a"].as_str().unwrap_or(if tool == "dev_check" {
                        "unedited"
                    } else {
                        "a"
                    }),
                    args["b"].as_str().unwrap_or("current"),
                );
                if tool == "dev_check" && metric == "accuracy" {
                    for (tag, m) in [(a, &r["a_mean"]), (b, &r["b_mean"])] {
                        if let Some(m) = m.as_f64() {
                            let k = if tag == "current" {
                                kind.clone()
                            } else if tag == "unedited" {
                                "unedited".into()
                            } else {
                                "other".into()
                            };
                            p.accuracy.push(AccPoint {
                                call,
                                accuracy: m,
                                kind: k,
                                what: format!("dev_check: `{tag}` on the dev split"),
                            });
                        }
                    }
                }
                let what = format!("{tool}: `{b}` − `{a}`");
                p.deltas
                    .extend(paired_delta(call, metric, &r["paired"], &kind, what));
            }
            "search" => {
                let best = r["best"].as_str().unwrap_or("?");
                let (pr, on) = if r["confirm_on_fresh_items"]["paired"].is_object() {
                    (&r["confirm_on_fresh_items"]["paired"], "fresh items")
                } else {
                    (
                        &r["best_vs_current_selection_items"],
                        "the items it was chosen on",
                    )
                };
                let what = format!("search: best `{best}` − current, on {on}");
                // The family of what the winner applies.
                let tool_kind = args["candidates"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .find(|c| c["name"] == best)
                    .and_then(|c| c["calls"][0]["tool"].as_str())
                    .map_or_else(|| kind.clone(), |t| edit_kind(t).to_string());
                p.deltas
                    .extend(paired_delta(call, metric, pr, &tool_kind, what));
            }
            "sensitivity" => {
                for t in r["tensors"].as_array().into_iter().flatten() {
                    let what = format!(
                        "sensitivity: {} scaled",
                        t["tensor"].as_str().unwrap_or("?")
                    );
                    p.deltas
                        .extend(paired_delta(call, metric, &t["paired"], "weight", what));
                }
            }
            "checkpoint" | "revert" => p.marks.push((
                call,
                tool.to_string(),
                args["tag"].as_str().unwrap_or("?").to_string(),
            )),
            _ => {}
        }
    }
    // Lanes: each edit, over the calls it was live.
    let mut order: Vec<Live> = Vec::new();
    for s in steps {
        for l in s.live.iter().flatten() {
            if !order.contains(l) {
                order.push(l.clone());
            }
        }
    }
    for l in order {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (i, s) in steps.iter().enumerate() {
            if s.live.as_ref().is_some_and(|v| v.contains(&l)) {
                match spans.last_mut() {
                    Some(sp) if sp.1 + 1 == i + 1 => sp.1 = i + 1,
                    _ => spans.push((i + 1, i + 1)),
                }
            }
        }
        p.lanes.push((l.label, l.kind, spans));
    }
    p
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() > n {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    } else {
        s.to_string()
    }
}

// ── Findings: what each measurement said about an idea ──────────────────

/// One measurement of an idea against what it was compared with.
#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub call: usize,
    /// `accuracy`, `loglik`, … (each its own panel).
    pub metric: String,
    /// What was tried, in words.
    pub idea: String,
    /// What it was compared with.
    pub versus: String,
    /// How it was measured.
    pub how: String,
    /// `b − a` (accuracy as a fraction).
    pub change: f64,
    /// The 95% interval, when the measurement pairs items.
    pub interval: Option<(f64, f64)>,
    /// `better` | `worse` | `unclear` | `no interval` (two runs, items not
    /// paired).
    pub verdict: String,
}

/// A call's arguments in words (`steer %1233 ×0.3`).
fn call_text(c: &Value) -> String {
    let t = c["tool"].as_str().unwrap_or("?");
    let a = |k: &str| {
        c[k].as_str()
            .map(str::to_string)
            .unwrap_or_else(|| c[k].to_string())
    };
    match t {
        "steer" => format!(
            "steer {} ×{}",
            a("node"),
            c["scale"].as_f64().unwrap_or(1.0)
        ),
        "token_bias" => format!("bias tokens {} by {}", short(&c["tokens"]), c["value"]),
        "scale" => format!("scale {} ×{}", a("name"), c["factor"]),
        "channel_bias" => format!("bias channels of {}", a("node")),
        _ => t.to_string(),
    }
}

/// What a set of live edits adds to the run's start, in words.
fn idea_of(live: &[Live], start: &[Live]) -> String {
    let added: Vec<&str> = live
        .iter()
        .filter(|l| !start.contains(l))
        .map(|l| l.label.as_str())
        .collect();
    if added.is_empty() {
        if start.is_empty() {
            "unedited".into()
        } else {
            "the starting version".into()
        }
    } else {
        added.join(" + ")
    }
}

fn verdict_word(v: &str) -> String {
    match v {
        "b better" => "better",
        "b worse" => "worse",
        _ => "unclear",
    }
    .into()
}

/// Every measurement that compares two versions, in call order: paired
/// comparisons (`compare`, `dev_check`, `search`, `sensitivity`,
/// `ablate`) and `qa_eval`s of two versions on the same questions.
pub fn findings(steps: &[Step]) -> Vec<Finding> {
    let start: Vec<Live> = steps
        .iter()
        .find_map(|s| s.live.clone())
        .unwrap_or_default();
    let mut out = Vec::new();
    // The first qa_eval of each question slice, by version.
    let mut evals: std::collections::HashMap<(u64, u64), Vec<(String, f64, usize)>> =
        Default::default();
    for (i, s) in steps.iter().enumerate() {
        if !s.ok {
            continue;
        }
        let call = i + 1;
        let live = s.live.clone().unwrap_or_default();
        let here = idea_of(&live, &start);
        let (tool, r, args) = if s.tool == "jobs" && s.result["collected"]["result"].is_object() {
            let c = &s.result["collected"];
            (
                c["tool"].as_str().unwrap_or_default(),
                &c["result"],
                &Value::Null,
            )
        } else {
            (s.tool.as_str(), &s.result, &s.args)
        };
        let metric = r["metric"].as_str().unwrap_or("score").to_string();
        let version = |tag: &str| match tag {
            "current" => here.clone(),
            "unedited" => "unedited".to_string(),
            t => format!("`{t}`"),
        };
        let mut push = |idea: String, versus: String, how: String, p: &Value, metric: &str| {
            let Some(m) = p["mean_b_minus_a"].as_f64() else {
                return;
            };
            let ci = p["ci95"].as_array();
            let interval = ci.and_then(|c| Some((c.first()?.as_f64()?, c.get(1)?.as_f64()?)));
            out.push(Finding {
                call,
                metric: metric.to_string(),
                idea,
                versus,
                how,
                change: m,
                interval,
                // The verdict as the current rule gives it: an interval clear of 0
                // and p < 0.05 (a run recorded before needed only the interval).
                verdict: if p["p"].as_f64().is_some_and(|x| x >= 0.05) {
                    "unclear".into()
                } else {
                    verdict_word(p["verdict"].as_str().unwrap_or(""))
                },
            });
        };
        match tool {
            "compare" | "dev_check" | "ablate" => {
                let a = args["a"].as_str().unwrap_or(if tool == "dev_check" {
                    "unedited"
                } else {
                    "base"
                });
                let b = args["b"].as_str().unwrap_or("current");
                let p = &r["paired"];
                let n = p["items"].as_u64().unwrap_or(0);
                let how = match (p["b_right_a_wrong"].as_u64(), p["a_right_b_wrong"].as_u64()) {
                    (Some(g), Some(l)) => {
                        format!("{}, {n} q: {g} gained, {l} lost", tool.replace('_', " "))
                    }
                    _ => format!("{}, {n} items", tool.replace('_', " ")),
                };
                push(version(b), version(a), how, p, &metric);
            }
            "search" => {
                let best = r["best"].as_str().unwrap_or("?");
                let idea = args["candidates"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .find(|c| c["name"] == best)
                    .and_then(|c| c["calls"].as_array())
                    .map(|cs| cs.iter().map(call_text).collect::<Vec<_>>().join(" + "))
                    .unwrap_or_else(|| format!("`{best}`"));
                if r["confirm_on_fresh_items"]["paired"].is_object() {
                    let p = &r["confirm_on_fresh_items"]["paired"];
                    let how = format!("search winner, {} fresh items", p["items"]);
                    push(idea, here.clone(), how, p, &metric);
                } else {
                    let p = &r["best_vs_current_selection_items"];
                    push(
                        idea,
                        here.clone(),
                        "search winner, on its own items".into(),
                        p,
                        &metric,
                    );
                }
            }
            "sensitivity" => {
                let eps = r["eps"].as_f64().unwrap_or(0.0);
                for t in r["tensors"].as_array().into_iter().flatten() {
                    let idea = format!(
                        "scale {} ×{:.2}",
                        t["tensor"].as_str().unwrap_or("?"),
                        1.0 + eps
                    );
                    push(
                        idea,
                        here.clone(),
                        "sensitivity probe".into(),
                        &t["paired"],
                        &metric,
                    );
                }
            }
            "qa_eval" => {
                let (Some(acc), n, off) = (
                    r["accuracy"].as_f64(),
                    args["n"].as_u64().unwrap_or(0),
                    args["offset"].as_u64().unwrap_or(0),
                ) else {
                    continue;
                };
                let seen = evals.entry((n, off)).or_default();
                // The first measurement of each version on these questions.
                if !seen.iter().any(|(v, _, _)| *v == here) {
                    seen.push((here.clone(), acc, call));
                }
            }
            _ => {}
        }
    }
    // Each version against the start on the same questions (else against
    // the first version measured on them), whichever ran first.
    for ((n, off), seen) in evals {
        let is_start = |v: &str| v == "unedited" || v == "the starting version";
        let Some(reference) = seen
            .iter()
            .find(|(v, _, _)| is_start(v))
            .or(seen.first())
            .cloned()
        else {
            continue;
        };
        for (v, acc, call) in seen.iter().filter(|e| e.0 != reference.0) {
            let right = |x: f64| (x * n as f64).round() as u64;
            out.push(Finding {
                call: *call,
                metric: "accuracy".into(),
                idea: v.clone(),
                versus: reference.0.clone(),
                how: format!(
                    "qa_eval, {n} q from {off}: {} vs {} right",
                    right(*acc),
                    right(reference.1)
                ),
                change: acc - reference.1,
                interval: None,
                verdict: "no interval".into(),
            });
        }
    }
    out.sort_by_key(|f| f.call);
    out
}

// ── The chart ───────────────────────────────────────────────────────────

const STYLE: &str = r#"
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: #52514e; font-size: 13px; }
  .title { fill: #0b0b0b; font-size: 20px; font-weight: 600; }
  .sub { fill: #52514e; font-size: 13px; }
  .ptitle { fill: #0b0b0b; font-size: 15px; font-weight: 600; }
  .tick { fill: #898781; font-size: 12px; }
  .idea { fill: #0b0b0b; font-size: 13px; }
  .how { fill: #52514e; font-size: 12px; }
  .big { fill: #0b0b0b; font-size: 30px; font-weight: 600; }
  .better { fill: #006300; font-weight: 600; } .worse { fill: #d03b3b; font-weight: 600; }
  .unclear, .once { fill: #898781; font-weight: 600; }
  .grid { stroke: #e1e0d9; stroke-width: 1; }
  .axis { stroke: #c3c2b7; stroke-width: 1; }
  .zero { stroke: #52514e; stroke-width: 1; }
  .band { fill: #f4f3f0; }
  .d-better { fill: #0ca30c; stroke: #0ca30c; } .d-worse { fill: #d03b3b; stroke: #d03b3b; }
  .d-unclear { fill: #898781; stroke: #898781; } .d-once { fill: #fcfcfb; stroke: #52514e; }
  .ci { stroke-width: 2; stroke-linecap: round; fill: none; }
  .dot { stroke-width: 2; }
  .ring { fill: none; stroke: #fcfcfb; stroke-width: 2; }
  .lane { fill: #2a78d6; } .lane-end { fill: #184f95; }
  .hit { fill: transparent; stroke: none; }
  .bg { fill: #fcfcfb; }
  .fail { stroke: #d03b3b; stroke-width: 2; stroke-linecap: round; }
  .mark { stroke: #52514e; stroke-width: 1; }
  @media (prefers-color-scheme: dark) {
    text, .sub, .how { fill: #c3c2b7; } .title, .ptitle, .idea, .big { fill: #f0efec; }
    .better { fill: #0ca30c; } .worse { fill: #e66767; }
    .grid { stroke: #2c2c2a; } .axis { stroke: #383835; } .zero { stroke: #c3c2b7; } .band { fill: #222220; }
    .d-once { fill: #1a1a19; stroke: #c3c2b7; } .ring { stroke: #1a1a19; } .bg { fill: #1a1a19; }
    .lane { fill: #3987e5; } .lane-end { fill: #86b6ef; } .mark { stroke: #c3c2b7; }
  }
"#;

/// The chart of `steps` as SVG; `report` (the run's `report.json`) adds
/// the held-out scores.
pub fn chart_svg(steps: &[Step], report: Option<&Value>, title: &str, source: &str) -> String {
    let p = plot(steps);
    let fs = findings(steps);
    let w = 1200.0_f64;
    let (label_r, plot_l, plot_r, how_l) = (380.0, 400.0, 760.0, 790.0);
    let mut body = String::new();
    body += &format!(
        r#"<text class="title" x="24" y="36">{}</text><text class="sub" x="24" y="58">{} tool calls · {}</text>"#,
        esc(title),
        p.calls,
        esc(source)
    );
    // Headline numbers.
    let held = |k: &str| report.and_then(|r| r[k]["accuracy"].as_f64());
    let mut tiles: Vec<(String, String)> = Vec::new();
    if let Some(u) = held("held_out_unedited") {
        tiles.push((format!("{:.0}%", u * 100.0), "held-out, unedited".into()));
    }
    if let Some(f) = held("held_out_final") {
        tiles.push((
            format!("{:.0}%", f * 100.0),
            "held-out, final version".into(),
        ));
    }
    if let Some(v) = held("held_out_verified") {
        tiles.push((
            format!("{:.0}%", v * 100.0),
            "held-out, verified version".into(),
        ));
    }
    if let Some(k) = report.and_then(|r| r["held_out_kept"].as_str()) {
        tiles.push((
            clip(k.split(' ').next().unwrap_or(k), 14),
            "version kept".into(),
        ));
    }
    if let Some(f) = report.and_then(|r| r["forward_passes"].as_object()) {
        tiles.push((
            format!("{} / {}", f["used"], f["budget"]),
            "forward passes used".into(),
        ));
    } else if let Some(u) = steps.iter().rev().find_map(|s| s.passes) {
        tiles.push((u.to_string(), "forward passes used".into()));
    }
    let mut y = 76.0;
    if !tiles.is_empty() {
        for (i, (big, label)) in tiles.iter().enumerate() {
            let x = 24.0 + i as f64 * 220.0;
            body += &format!(
                r#"<text class="big" x="{x:.1}" y="{:.1}">{}</text><text class="how" x="{x:.1}" y="{:.1}">{}</text>"#,
                y + 32.0,
                esc(big),
                y + 52.0,
                esc(label)
            );
        }
        y += 76.0;
    }

    // One forest panel per metric.
    let mut metrics: Vec<String> = Vec::new();
    for f in &fs {
        if !metrics.contains(&f.metric) {
            metrics.push(f.metric.clone());
        }
    }
    metrics.sort_by_key(|m| m != "accuracy");
    let row_h = 30.0;
    for m in &metrics {
        let rows: Vec<&Finding> = fs.iter().filter(|f| &f.metric == m).collect();
        let acc = m == "accuracy";
        let (heading, note) = if acc {
            (
                "Did each idea help? Change in accuracy",
                "percentage points against what it was compared with; line = 95% interval",
            )
        } else if m.contains("loglik") {
            (
                "Change in log-likelihood of the reference solutions",
                "per token, against what it was compared with; line = 95% interval",
            )
        } else {
            (
                "Change in score",
                "against what it was compared with; line = 95% interval",
            )
        };
        y += 20.0;
        body += &format!(
            r#"<text class="ptitle" x="24" y="{:.1}">{}</text><text class="how" x="24" y="{:.1}">{}</text>"#,
            y,
            esc(heading),
            y + 18.0,
            esc(note)
        );
        let top = y + 62.0;
        let bottom = top + rows.len() as f64 * row_h;
        let scale = if acc { 100.0 } else { 1.0 };
        let mut ext = rows
            .iter()
            .flat_map(|f| {
                [
                    f.change,
                    f.interval.map_or(0.0, |i| i.0),
                    f.interval.map_or(0.0, |i| i.1),
                ]
            })
            .fold(0.0_f64, |a, v| a.max(v.abs()))
            * scale;
        if acc {
            ext = (ext / 5.0).ceil().max(1.0) * 5.0;
        } else {
            ext = if ext > 0.0 { ext * 1.1 } else { 1.0 };
        }
        let xx = |v: f64| (plot_l + plot_r) / 2.0 + (v * scale / ext) * (plot_r - plot_l) / 2.0;
        // Axis labels above the rows.
        let fmt = |v: f64| {
            if acc {
                if v == 0.0 {
                    "0".to_string()
                } else {
                    format!("{:+.0} pts", v)
                }
            } else {
                format!("{v:+.3}")
            }
        };
        for k in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let v = k * ext;
            let x = (plot_l + plot_r) / 2.0 + k * (plot_r - plot_l) / 2.0;
            body += &format!(
                r#"<line class="{}" x1="{x:.1}" x2="{x:.1}" y1="{:.1}" y2="{bottom:.1}"/><text class="tick" x="{x:.1}" y="{:.1}" text-anchor="middle">{}</text>"#,
                if k == 0.0 { "zero" } else { "grid" },
                top - 6.0,
                top - 12.0,
                esc(&fmt(v))
            );
        }
        body += &format!(
            r#"<text class="tick" x="{:.1}" y="{:.1}" text-anchor="start">← worse</text><text class="tick" x="{:.1}" y="{:.1}" text-anchor="end">better →</text>"#,
            plot_l,
            top - 26.0,
            plot_r,
            top - 26.0
        );
        for (j, f) in rows.iter().enumerate() {
            let ry = top + j as f64 * row_h;
            let cy = ry + row_h / 2.0;
            if j % 2 == 1 {
                body += &format!(
                    r#"<rect class="band" x="16" y="{ry:.1}" width="{:.1}" height="{row_h:.1}"/>"#,
                    w - 32.0
                );
            }
            let cls = match f.verdict.as_str() {
                "better" => "better",
                "worse" => "worse",
                "unclear" => "unclear",
                _ => "once",
            };
            let idea = if f.versus == "unedited" || f.versus == "the starting version" {
                f.idea.clone()
            } else {
                format!("{} vs {}", f.idea, f.versus)
            };
            let tip = format!(
                "call {}: {} — {} vs {}: {}{}",
                f.call,
                f.how,
                f.idea,
                f.versus,
                fmt(f.change * scale),
                f.interval.map_or_else(String::new, |(lo, hi)| format!(
                    " [{}, {}]",
                    fmt(lo * scale),
                    fmt(hi * scale)
                ))
            );
            let mut marks = String::new();
            if let Some((lo, hi)) = f.interval {
                marks += &format!(
                    r#"<line class="d-{cls} ci" x1="{:.1}" x2="{:.1}" y1="{cy:.1}" y2="{cy:.1}"/>"#,
                    xx(lo).max(plot_l),
                    xx(hi).min(plot_r)
                );
            }
            let cx = xx(f.change).clamp(plot_l, plot_r);
            marks += &format!(
                r#"<circle class="ring" cx="{cx:.1}" cy="{cy:.1}" r="7"/><circle class="d-{cls} dot" cx="{cx:.1}" cy="{cy:.1}" r="5"/>"#
            );
            body += &format!(
                r#"<g><title>{}</title><text class="idea" x="{label_r:.1}" y="{:.1}" text-anchor="end">{}</text>{marks}<text class="how" x="{how_l:.1}" y="{:.1}">{}</text><text class="{cls}" x="{:.1}" y="{:.1}" text-anchor="end">{}</text><rect class="hit" x="16" y="{ry:.1}" width="{:.1}" height="{row_h:.1}"/></g>"#,
                esc(&tip),
                cy + 4.5,
                esc(&clip(&idea, 50)),
                cy + 4.5,
                esc(&clip(&f.how, 44)),
                w - 24.0,
                cy + 4.5,
                esc(&format!("{} {}", fmt(f.change * scale), f.verdict)),
                w - 32.0
            );
        }
        y = bottom + 10.0;
    }
    if fs.is_empty() {
        y += 20.0;
        body += &format!(
            r#"<text class="how" x="24" y="{y:.1}">No measurement compared two versions.</text>"#
        );
        y += 10.0;
    }

    // When each idea was live.
    let n = p.calls.max(1);
    let x = |c: usize| plot_l + (c as f64 - 0.5) / n as f64 * (w - 24.0 - plot_l);
    let slot = (w - 24.0 - plot_l) / n as f64;
    y += 30.0;
    body += &format!(
        r#"<text class="ptitle" x="24" y="{:.1}">When each idea was live</text><text class="how" x="24" y="{:.1}">by tool call; darker = still live at the end; ✕ = a call that failed (hover for why)</text>"#,
        y,
        y + 18.0
    );
    let lanes_top = y + 48.0;
    let lane_h = 22.0;
    for (j, (c, tool, tag)) in p.marks.iter().enumerate() {
        let cx = x(*c);
        body += &format!(
            r#"<g><title>call {c}: {tool} `{}`</title><line class="mark" x1="{cx:.1}" x2="{cx:.1}" y1="{:.1}" y2="{:.1}"/><text class="tick" x="{cx:.1}" y="{:.1}" text-anchor="middle">{}{}</text></g>"#,
            esc(tag),
            lanes_top - 6.0,
            lanes_top + (p.lanes.len() + 1) as f64 * lane_h,
            lanes_top - 10.0 - (j % 2) as f64 * 14.0,
            if tool == "revert" { "↺ " } else { "" },
            esc(&clip(tag, 14))
        );
    }
    for (j, (label, _kind, spans)) in p.lanes.iter().enumerate() {
        let ly = lanes_top + j as f64 * lane_h;
        let at_end = spans.last().is_some_and(|s| s.1 == n);
        body += &format!(
            r#"<text class="idea" x="{label_r:.1}" y="{:.1}" text-anchor="end"><title>{}</title>{}</text>"#,
            ly + 15.0,
            esc(label),
            esc(&clip(label, 50))
        );
        for (a, b) in spans {
            let x0 = x(*a) - slot / 2.0;
            let x1 = x(*b) + slot / 2.0;
            let cls = if at_end && *b == n {
                "lane-end"
            } else {
                "lane"
            };
            body += &format!(
                r#"<g><title>{}: live from call {a} to {b}</title><rect class="{cls}" x="{:.1}" y="{:.1}" width="{:.1}" height="12" rx="4"/></g>"#,
                esc(label),
                x0 + 1.0,
                ly + 5.0,
                (x1 - x0 - 2.0).max(4.0)
            );
        }
    }
    let fy = lanes_top + p.lanes.len() as f64 * lane_h;
    body += &format!(
        r#"<text class="idea" x="{label_r:.1}" y="{:.1}" text-anchor="end">failed calls ({})</text>"#,
        fy + 15.0,
        p.failed.len()
    );
    for (c, tool, e) in &p.failed {
        let cx = x(*c);
        let cy = fy + 11.0;
        body += &format!(
            r#"<g><title>call {c}: {} failed — {}</title><line class="fail" x1="{:.1}" x2="{:.1}" y1="{:.1}" y2="{:.1}"/><line class="fail" x1="{:.1}" x2="{:.1}" y1="{:.1}" y2="{:.1}"/><circle class="hit" cx="{cx:.1}" cy="{cy:.1}" r="10"/></g>"#,
            esc(tool),
            esc(&clip(e, 200)),
            cx - 4.0,
            cx + 4.0,
            cy - 4.0,
            cy + 4.0,
            cx - 4.0,
            cx + 4.0,
            cy + 4.0,
            cy - 4.0
        );
    }
    let bottom = fy + lane_h + 4.0;
    body += &format!(
        r#"<line class="axis" x1="{plot_l:.1}" x2="{:.1}" y1="{bottom:.1}" y2="{bottom:.1}"/>"#,
        w - 24.0
    );
    let step = ((n as f64 / 10.0).ceil() as usize).max(1);
    let mut c = step;
    while c <= n {
        body += &format!(
            r#"<text class="tick" x="{:.1}" y="{:.1}" text-anchor="middle">{c}</text>"#,
            x(c),
            bottom + 16.0
        );
        c += step;
    }
    body += &format!(
        r#"<text class="tick" x="{label_r:.1}" y="{:.1}" text-anchor="end">tool call</text>"#,
        bottom + 16.0
    );
    let height = bottom + 36.0;
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0}" height="{height:.0}" viewBox="0 0 {w:.0} {height:.0}" role="img" aria-labelledby="t"><title id="t">{}</title><style>{STYLE}</style><rect class="bg" width="100%" height="100%"/>{body}</svg>"#,
        esc(title)
    )
}

/// The trail of a run directory: its state's, else its transcript's.
pub fn steps_of_run(dir: &std::path::Path) -> Result<(Vec<Step>, &'static str), String> {
    let state = std::fs::read_to_string(dir.join("state.json")).ok();
    if let Some(st) = state
        .as_deref()
        .and_then(|t| serde_json::from_str::<Value>(t).ok())
        && let Some(t) = st["trail"].as_array()
        && !t.is_empty()
    {
        let steps: Vec<Step> =
            serde_json::from_value(Value::Array(t.clone())).map_err(|e| e.to_string())?;
        return Ok((steps, "recorded by the server (state trail)"));
    }
    let text = std::fs::read_to_string(dir.join("transcript.jsonl")).map_err(|e| {
        format!(
            "{}: no trail in state.json and no transcript ({e})",
            dir.display()
        )
    })?;
    let st = state
        .as_deref()
        .and_then(|t| serde_json::from_str::<Value>(t).ok());
    let steps = from_transcript(&text, st.as_ref());
    if steps.is_empty() {
        return Err(format!(
            "{}: the transcript has no tool calls this reader knows",
            dir.display()
        ));
    }
    Ok((
        steps,
        "reconstructed from the transcript (edits followed from the edit calls)",
    ))
}

/// Write `exploration.svg` and `exploration.json` (its table) in `dir`.
pub fn write_chart(dir: &std::path::Path, title: &str) -> Result<std::path::PathBuf, String> {
    let (steps, source) = steps_of_run(dir)?;
    let report = std::fs::read_to_string(dir.join("report.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok());
    let svg = chart_svg(&steps, report.as_ref(), title, source);
    let out = dir.join("exploration.svg");
    std::fs::write(&out, svg).map_err(|e| e.to_string())?;
    std::fs::write(
        dir.join("exploration.json"),
        serde_json::to_string_pretty(
            &json!({ "source": source, "findings": findings(&steps), "plot": plot(&steps) }),
        )
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(kind: &str, block: Value) -> String {
        json!({ "type": kind, "message": { "content": [block] } }).to_string()
    }

    #[test]
    fn a_transcript_gives_calls_results_and_live_edits() {
        let call = |id: &str, name: &str, input: Value| {
            line(
                "assistant",
                json!({"type": "tool_use", "id": id, "name": name, "input": input}),
            )
        };
        let result = |id: &str, text: &str| {
            line(
                "user",
                json!({"type": "tool_result", "tool_use_id": id,
                                 "content": [{"type": "text", "text": text}]}),
            )
        };
        let text = [
            call("1", "mcp__wsp__qa_eval", json!({"n": 64, "offset": 0})),
            result("1", r#"{"accuracy": 0.25}"#),
            call("2", "mcp__wsp__steer", json!({"node": "%1233", "vector": "ex", "scale": 0.3})),
            result("2", r#"{"rewrite": 0}"#),
            call("3", "mcp__wsp__qa_eval", json!({"n": 64, "offset": 0})),
            result("3", r#"{"accuracy": 0.39}"#),
            call("4", "mcp__wsp__trace", json!({"nodes": ["%1"]})),
            result("4", "no graph run had all of [\"%1\"]"),
            call("5", "mcp__wsp__compare", json!({"a": "base", "b": "current", "metric": "loglik"})),
            result("5", r#"{"metric": "loglik", "paired": {"mean_b_minus_a": 0.04, "ci95": [0.01, 0.07], "verdict": "b better"}}"#),
            call("6", "mcp__wsp__rewrites", json!({"remove": 0})),
            result("6", r#"{"removed": "steer"}"#),
            call("7", "mcp__wsp__revert", json!({"tag": "branch_start"})),
            result("7", r#"{"reverted": "branch_start"}"#),
        ]
        .join("\n");
        // The run branched from a version with one edited tensor.
        let state =
            json!({ "checkpoints": { "branch_start": { "blk.3.attn_q_norm.weight": {} } } });
        let steps = from_transcript(&text, Some(&state));
        assert_eq!(steps.len(), 7);
        assert!(!steps[3].ok, "the failed trace");
        let p = plot(&steps);
        assert_eq!(p.accuracy.len(), 2);
        assert_eq!(
            p.accuracy[0].kind, "weight",
            "started from the branch's version"
        );
        assert_eq!(
            p.accuracy[1].kind, "activation",
            "measured with the steer live"
        );
        assert_eq!(p.deltas.len(), 1);
        assert_eq!((p.deltas[0].lo, p.deltas[0].hi), (0.01, 0.07));
        assert_eq!(p.failed.len(), 1);
        let steer = p
            .lanes
            .iter()
            .find(|l| l.0.starts_with("steer %1233"))
            .unwrap();
        assert_eq!(steer.2, vec![(2, 5)], "live from the steer to its removal");
        let w = p.lanes.iter().find(|l| l.1 == "weight").unwrap();
        assert_eq!(w.2, vec![(1, 7)], "the branch's tensor, live throughout");
        let f = findings(&steps);
        assert_eq!(f.len(), 2, "{f:?}");
        assert_eq!(
            (f[0].idea.as_str(), f[0].versus.as_str()),
            ("steer %1233 by 0.3 x ex", "the starting version")
        );
        assert!(
            (f[0].change - 0.14).abs() < 1e-9 && f[0].verdict == "no interval",
            "{f:?}"
        );
        assert_eq!(f[1].verdict, "better");
        let svg = chart_svg(&steps, None, "a run", "from a transcript");
        assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
        assert!(svg.contains("steer %1233"));
    }

    #[test]
    fn tensors_of_one_kind_share_a_lane() {
        let l = weight_lanes(
            [
                "blk.3.attn_q_norm.weight",
                "blk.7.attn_q_norm.weight",
                "token_embd.weight",
            ]
            .into_iter(),
        );
        let labels: Vec<&str> = l.iter().map(|x| x.label.as_str()).collect();
        assert_eq!(
            labels,
            ["attn_q_norm.weight in 2 blocks", "token_embd.weight"]
        );
    }
}
