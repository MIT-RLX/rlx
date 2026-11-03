// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The agent's text: the system prompt, sent with every call, and one message
//! per decision. Facts and protocol only — no recommended settings.
//!
//! The system prompt is a template ([`Prompt`]): the neutral default, the
//! aggressive variant (a review of every group's hyperparameters at each
//! decision), or a file of your own, filled with the same `{fields}` in one
//! pass — only the known names, so braces in examples need no escaping and a
//! field's value is never itself searched for fields.

use std::path::PathBuf;

use serde_json::Value;

use super::Waiting;
use super::program::ParamInfo;
use crate::{ApiError, ApiResult};

/// Which system prompt a run uses.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Prompt {
    #[default]
    Neutral,
    /// The neutral prompt, except that each decision starts with a review of
    /// every group's hyperparameters and the rationale has four parts.
    Aggressive,
    /// A template file with the neutral prompt's `{fields}`.
    Template(PathBuf),
}

/// The fields a template may use.
pub const FIELDS: [&str; 12] = [
    "goal",
    "total",
    "decisions",
    "mode",
    "max_decisions",
    "every",
    "trial",
    "facts",
    "validation",
    "interface",
    "code",
    "table",
];

impl Prompt {
    /// `"neutral"`, `"aggressive"`, or the path of a template file.
    pub fn parse(s: &str) -> Self {
        match s {
            "neutral" => Self::Neutral,
            "aggressive" => Self::Aggressive,
            path => Self::Template(PathBuf::from(path)),
        }
    }

    pub fn is_aggressive(&self) -> bool {
        *self == Self::Aggressive
    }

    /// The template text.
    pub fn template(&self) -> ApiResult<String> {
        match self {
            Self::Neutral => Ok(NEUTRAL.to_string()),
            Self::Aggressive => Ok(NEUTRAL.replace(NEUTRAL_DECISIONS, AGGRESSIVE_DECISIONS)),
            Self::Template(p) => std::fs::read_to_string(p).map_err(|e| {
                ApiError::type_(format!(
                    "agent prompt: \"neutral\", \"aggressive\" or a template file; {}: {e}",
                    p.display()
                ))
            }),
        }
    }
}

/// Replace `{name}` for the known `fields` only, in one pass.
pub fn fill(template: &str, fields: &[(&str, String)]) -> String {
    let mut out = String::with_capacity(template.len() + 4096);
    let mut rest = template;
    while let Some(i) = rest.find('{') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let hit = fields
            .iter()
            .find(|(name, _)| after.starts_with(name) && after[name.len()..].starts_with('}'));
        match hit {
            Some((name, value)) => {
                out.push_str(value);
                rest = &after[name.len() + 1..];
            }
            None => {
                out.push('{');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

const NEUTRAL: &str = "You are a coding agent inside one RLX training run. While the model \
trains, you design and adjust its optimizer (the update rule of each group of parameters, the \
learning rates and their schedule, and the other hyperparameters) to reach the best result on the \
evaluation stated in the goal, which you are not shown.

TASK
The goal of this run, in the words of the person running it: {goal}
The run is planned for {total} updates.{facts}{validation}
{decisions} You are also asked when the training loss becomes non-finite. At most {max_decisions} \
decisions are asked in all. {mode} Between decisions your program runs by itself at every update, \
so whatever should change between decisions (a warmup, a decay, a guard against spikes) must be \
written into it.
The model, the data and their order, the loss, the batch size, the gradient clipping and the \
number of updates are fixed; the optimizer is yours.

PROTOCOL
- An update is one training step. A decision is a point where you read what happened and may \
edit your program; keeping it is a valid decision.
- The run checks every {every} updates (and at each scheduled decision) for a non-finite loss. \
Your analysis (below) decides what the reports hold and how often they are made; every decision's \
reports are in inbox/.
- When you end your turn, the run compares your program with the running one. A changed program \
is built and takes over; one that fails to build is not installed, the running one stays, and you \
see the error.
- A new program is on trial for its first {trial} updates. If it fails at its first update, that \
update is undone exactly and made by the previous program instead; if it fails later in the \
trial, training continues with the previous program and the failing update is not undone. After \
its trial, a failing program is replaced by the one before it, rebuilt from its files with its \
optimizer state starting from zero. You see every error at your next decision.
- Only one decision is in flight at a time; reasons that arise meanwhile are merged into the next. \
Each message may carry a note from the person running the training; take it into account.

YOUR PROGRAM
{interface}

WORKSPACE
This folder holds your program, inbox/ (the inputs of every decision so far, as JSON) and \
reference/ (the parameter table and the starting program{code}). You may keep notes in NOTES.md; \
they persist for the whole run. Your tools work on this folder only: you can read every file in it \
and write only your program's files and NOTES.md. Nothing you write runs except your program.

DECISIONS
Before changing anything, check what you predicted at your previous decision against what \
happened. End every decision with a short rationale:
1. Observed: what the evidence since your last decision shows, in numbers.
2. Changed or kept: what you edited and why, or which evidence you are waiting for.
3. Prediction: the effect you expect, by roughly when, and what result would show you were wrong.
Write schedules as functions of the fraction of training, never of when you wrote them.

THE RUN
Parameters (name, shape), in the order your program receives them:
{table}
";

const NEUTRAL_DECISIONS: &str = "Before changing anything, check what you predicted at your \
previous decision against what happened. End every decision with a short rationale:
1. Observed: what the evidence since your last decision shows, in numbers.
2. Changed or kept: what you edited and why, or which evidence you are waiting for.
3. Prediction: the effect you expect, by roughly when, and what result would show you were wrong.";

const AGGRESSIVE_DECISIONS: &str = "At each decision, first check what you predicted at your \
previous decision against what happened. Then review, for each group of parameters, whether its \
current settings (the update rule, the learning rate and its schedule, momentum or betas, \
epsilon, weight decay) suit the next stretch of training, and which adjustments, in which \
direction and by how much, would improve the result. Rules in your program can also make such \
adjustments between decisions, from signals they measure. End every decision with a short \
rationale:
1. Observed: what the evidence since your last decision shows, in numbers.
2. Review: for each group, whether its settings suit the next stretch, and which adjustments \
(direction and size) you expect to help.
3. Changed: what you edited and why.
4. Prediction: the effect you expect, by roughly when, and what result would show you were wrong.";

/// What the system prompt states about the run.
pub(crate) struct RunFacts<'a> {
    pub goal: &'a str,
    pub total: u64,
    /// `None` until an estimated schedule is set.
    pub schedule: Option<&'a [u64]>,
    pub max_decisions: usize,
    pub every: u64,
    pub trial: u64,
    pub waiting: Waiting,
    pub max_lag: u64,
    pub pause_on_nonfinite: bool,
    /// The run's facts as one sentence or more, starting with a space ("" for none).
    pub facts: String,
    /// The validation set's sentence, starting with a space ("" without one).
    pub validation: String,
    /// The shared code's mention, starting with ", " ("" for none).
    pub code: String,
    pub interface: String,
    pub params: &'a [ParamInfo],
}

fn shown(updates: &[u64]) -> String {
    let all: Vec<String> = updates.iter().map(|u| u.to_string()).collect();
    if all.len() <= 6 {
        all.join(", ")
    } else {
        format!("{}, …, {}", all[..4].join(", "), all[all.len() - 1])
    }
}

pub(crate) fn system(template: &str, f: &RunFacts<'_>) -> String {
    let decisions = match f.schedule {
        None => "The decision schedule is set once the first updates are timed; each message \
                 says how many decisions are left."
            .to_string(),
        Some([]) => "No decisions are scheduled.".to_string(),
        Some([u]) => format!("You are asked once, at update {u}."),
        Some(s) => format!(
            "You are asked at updates {} ({} scheduled decisions).",
            shown(s),
            s.len()
        ),
    };
    let nan_pause = if f.pause_on_nonfinite {
        " A non-finite training loss pauses training until you answer."
    } else {
        ""
    };
    let mode = match f.waiting {
        Waiting::Always => "Training pauses at each decision until you finish; your edits take \
                            effect from the update at which the decision was asked."
            .to_string(),
        Waiting::Never => format!(
            "Training keeps running while you work; your edits take effect at the first update \
             after you finish, and the next message says which update that was.{nan_pause}"
        ),
        Waiting::Auto => format!(
            "Training pauses for your answer when it would otherwise take effect more than {} \
             updates late (judged from the time per update and your recent answer times; always \
             at the first decision), and keeps running otherwise; each message says which \
             applies.{nan_pause}",
            f.max_lag
        ),
    };
    let width = f.params.iter().map(|p| p.name.len()).max().unwrap_or(0);
    let table: Vec<String> = f
        .params
        .iter()
        .map(|p| format!("  {:width$}  {:?}", p.name, p.shape))
        .collect();
    fill(
        template,
        &[
            ("goal", f.goal.to_string()),
            ("total", f.total.to_string()),
            ("decisions", decisions),
            ("mode", mode),
            ("max_decisions", f.max_decisions.to_string()),
            ("every", f.every.to_string()),
            ("trial", f.trial.to_string()),
            ("facts", f.facts.clone()),
            ("validation", f.validation.clone()),
            ("interface", f.interface.clone()),
            ("code", f.code.clone()),
            ("table", table.join("\n")),
        ],
    )
}

/// One decision's inputs, as its message states them.
pub(crate) struct DecisionFacts<'a> {
    pub k: usize,
    pub update: u64,
    pub total: u64,
    pub paused: bool,
    pub reasons: &'a [String],
    pub previous: Option<&'a str>,
    pub losses: &'a [f32],
    pub losses_from: u64,
    pub validation: Option<&'a [Value]>,
    pub shown: &'a [Value],
    pub skipped: &'a [u64],
    pub reports: &'a [Value],
    pub errors: &'a [Value],
    pub notes: &'a [String],
    pub program_ms: Option<f64>,
    pub start_ms: Option<f64>,
    pub decisions_left: usize,
    pub cost_so_far: f64,
    pub files: Vec<&'a str>,
    pub aggressive: bool,
}

/// At most `n` values: means of equal runs of `values` when there are more.
fn chunked(values: &[f32], n: usize) -> (Vec<f32>, usize) {
    if values.len() <= n {
        return (values.to_vec(), 1);
    }
    let size = values.len().div_ceil(n);
    let means = values
        .chunks(size)
        .map(|c| c.iter().sum::<f32>() / c.len() as f32)
        .collect();
    (means, size)
}

fn ms(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.3} ms"))
        .unwrap_or_else(|| "not measured yet".into())
}

/// Validation measurements shown in a message; all are in the inbox.
const VALIDATION_SHOWN: usize = 50;
/// Values the training script chose to show, per message.
const SHOWN_VALUES: usize = 20;

fn validation_line(k: usize, all: &[Value]) -> String {
    let head = "Validation set (held out from training; not the evaluation in the goal)";
    if all.is_empty() {
        return format!("{head}: no measurement yet.");
    }
    let start = all.len().saturating_sub(VALIDATION_SHOWN);
    let which = if start == 0 {
        format!("all {} measurements so far", all.len())
    } else {
        format!(
            "the most recent {} of {} (all in inbox/decision-{k:03}.json)",
            all.len() - start,
            all.len()
        )
    };
    let entries: Vec<String> = all[start..]
        .iter()
        .map(|v| {
            let mut parts = Vec::new();
            for key in ["loss", "accuracy"] {
                if let Some(x) = v[key].as_f64() {
                    parts.push(format!("{key} {x:.4}"));
                }
            }
            format!("{}: {}", v["update"], parts.join(", "))
        })
        .collect();
    format!("{head}, {which} (update: values): {}", entries.join("; "))
}

pub(crate) fn decision(d: &DecisionFacts<'_>) -> String {
    let pct = 100.0 * d.update as f64 / d.total.max(1) as f64;
    let mut lines = vec![format!(
        "Decision {}. Asked at update {} of {} ({pct:.1}% of training), before that update's \
         parameter change.",
        d.k, d.update, d.total
    )];
    lines.push(if d.paused {
        format!(
            "Training is paused until you finish; your edits take effect from update {}.",
            d.update
        )
    } else {
        "Training continues while you work; your edits take effect at the first update after you \
         finish."
            .into()
    });
    lines.push(format!("Why now: {}.", d.reasons.join("; ")));
    if let Some(p) = d.previous {
        lines.push(p.to_string());
    }
    let (loss, per) = chunked(d.losses, 100);
    if loss.is_empty() {
        lines.push("No training loss was recorded since your previous decision.".into());
    } else {
        let values: Vec<String> = loss.iter().map(|v| format!("{v:.4}")).collect();
        lines.push(format!(
            "Training loss since your previous decision (mean per {per} update{}, {} values, from \
             update {}): {}",
            if per > 1 { "s" } else { "" },
            loss.len(),
            d.losses_from,
            values.join(", ")
        ));
    }
    if let Some(v) = d.validation {
        lines.push(validation_line(d.k, v));
    }
    if !d.shown.is_empty() {
        let start = d.shown.len().saturating_sub(SHOWN_VALUES);
        let values: Vec<String> = d.shown[start..].iter().map(Value::to_string).collect();
        lines.push(format!(
            "Values the training script chose to show you (most recent {}): {}",
            values.len(),
            values.join(", ")
        ));
    }
    if !d.skipped.is_empty() {
        let mut s: Vec<String> = d.skipped.iter().take(20).map(u64::to_string).collect();
        if d.skipped.len() > 20 {
            s.push("…".into());
        }
        lines.push(format!(
            "Updates your program skipped since your previous decision: {} ({}).",
            d.skipped.len(),
            s.join(", ")
        ));
    }
    lines.push(format!(
        "Reports since your previous decision: {} (all in inbox/decision-{:03}.json).",
        d.reports.len(),
        d.k
    ));
    if let Some(last) = d.reports.last() {
        let mut s = last.to_string();
        s.truncate(6000);
        lines.push(format!("Most recent report: {s}"));
    }
    if d.errors.is_empty() {
        lines.push("No errors since your previous decision.".into());
    } else {
        let start = d.errors.len().saturating_sub(10);
        let shown: Vec<String> = d.errors[start..]
            .iter()
            .map(|e| {
                format!(
                    "- update {} in {}:\n{}",
                    e["update"],
                    e["where"].as_str().unwrap_or("?"),
                    e["error"].as_str().unwrap_or("?")
                )
            })
            .collect();
        lines.push(format!(
            "Errors since your previous decision:\n{}",
            shown.join("\n")
        ));
    }
    lines.push(format!(
        "Timing: your program {} per update (the starting program: {}).",
        ms(d.program_ms),
        ms(d.start_ms)
    ));
    for n in d.notes {
        lines.push(format!("Note from the person running this training: {n}"));
    }
    lines.push(format!(
        "Decisions so far: {}; decisions left after this one: {}; agent cost so far: ${:.2}.",
        d.k - 1,
        d.decisions_left,
        d.cost_so_far
    ));
    lines.push(if d.aggressive {
        format!(
            "Your program is in this folder ({}). Edit it in place. End with your rationale in the \
             four parts of DECISIONS: observed, review, changed, prediction.",
            d.files.join(", ")
        )
    } else {
        format!(
            "Your program is in this folder ({}). Edit it in place, or keep it (a valid decision). \
             End with your rationale in the three parts of DECISIONS: observed, changed or kept, \
             prediction.",
            d.files.join(", ")
        )
    });
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_loss_histories_are_averaged_to_a_hundred_values() {
        let v: Vec<f32> = (0..250).map(|i| i as f32).collect();
        let (m, per) = chunked(&v, 100);
        assert_eq!(per, 3);
        assert_eq!(m.len(), 84);
        assert_eq!(m[0], 1.0);
        assert_eq!(chunked(&v[..5], 100), (v[..5].to_vec(), 1));
    }

    #[test]
    fn templates_fill_known_fields_once_and_leave_other_braces() {
        let t = "{goal} {unknown} {\"lr\": 1} {goal}";
        let out = fill(t, &[("goal", "a {goal} b".to_string())]);
        assert_eq!(out, "a {goal} b {unknown} {\"lr\": 1} a {goal} b");
        assert!(
            NEUTRAL.contains(NEUTRAL_DECISIONS),
            "the aggressive edit applies"
        );
        let aggressive = Prompt::Aggressive.template().unwrap();
        assert!(aggressive.contains("2. Review:") && !aggressive.contains("2. Changed or kept"));
        for f in FIELDS {
            assert!(
                NEUTRAL.contains(&format!("{{{f}}}")),
                "{f} is used by the neutral prompt"
            );
        }
    }
}
