// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Measuring edits so that a decision is not noise.
//!
//! - **Metrics, per item:** `loglik` (teacher-forced mean log-probability
//!   of the reference — continuous, deterministic, far less noisy than
//!   accuracy), `accuracy` (greedy answer right or not) and `return`
//!   (control tasks: one episode).
//! - **Paired statistics:** two versions are scored on the *same* items and
//!   the per-item differences summarised — mean, a bootstrap 95% interval,
//!   a sign-flip test (exact McNemar for 0/1 scores) and a verdict.
//! - **Sensitivity:** how much the metric moves when one tensor is scaled
//!   by `1 + eps` — where an edit can matter, found with forward passes
//!   only.
//! - **Search:** successive halving over candidate edits (each a list of
//!   tool calls) on growing item sets, the winner then checked against the
//!   live network on items no round used (selecting the best of many on the
//!   same items overstates it).

use crate::env::EnvKind;
use crate::lm::QaItem;
use crate::space::{Variant, WeightSpace};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Loglik,
    Accuracy,
    Return,
}

impl Metric {
    pub fn name(self) -> &'static str {
        match self {
            Metric::Loglik => "loglik",
            Metric::Accuracy => "accuracy",
            Metric::Return => "return",
        }
    }

    /// `name`, else the task's default (`loglik` for QA, `return` for
    /// control tasks).
    pub fn parse(ws: &WeightSpace, name: Option<&str>) -> Result<Self, String> {
        let qa = ws.state.env == EnvKind::Qa;
        let control = !qa && !ws.state.env.is_classifier() && !ws.is_loaded();
        let m = match name {
            None if qa => Metric::Loglik,
            None => Metric::Return,
            Some("loglik") => Metric::Loglik,
            Some("accuracy") => Metric::Accuracy,
            Some("return") => Metric::Return,
            Some(o) => return Err(format!("metric `{o}`: use loglik | accuracy | return")),
        };
        match m {
            Metric::Loglik | Metric::Accuracy if !qa => {
                Err("loglik / accuracy are for question-answering tasks".into())
            }
            Metric::Return if !control => Err("return is for control tasks".into()),
            _ => Ok(m),
        }
    }
}

/// Practice questions `offset..offset + n` (1..=64; the dev split held
/// back).
pub fn practice(ws: &WeightSpace, n: usize, offset: usize) -> Result<Vec<QaItem>, String> {
    let q = ws
        .state
        .qa
        .as_ref()
        .ok_or("not a question-answering task")?;
    let items = crate::lm::practice_items(q)?;
    let pick: Vec<QaItem> = items
        .into_iter()
        .skip(offset)
        .take(n.clamp(1, 64))
        .collect();
    if pick.is_empty() {
        return Err(format!("no practice questions from {offset}"));
    }
    Ok(pick)
}

/// What a score is computed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Practice questions (QA) or episodes from a seed (control).
    Practice { n: usize, offset: usize },
    /// The held-back dev questions.
    Dev,
    /// The general texts (log-likelihood only).
    General { n: usize },
}

/// Per-item scores of the live network, not budgeted (harness and
/// trainers; tools go through [`score_items`]).
pub fn score_raw(
    ws: &mut WeightSpace,
    metric: Metric,
    mode: &str,
    source: Source,
    masks: &[crate::graphops::Unit],
) -> Result<Vec<f64>, String> {
    let items = match source {
        Source::General { n } => {
            let path = ws
                .state
                .qa
                .as_ref()
                .and_then(|q| q.general.clone())
                .ok_or("no general texts (make the space with --qa-general)")?;
            let texts: Vec<String> = crate::lm::general_texts(&path)?
                .into_iter()
                .take(n.clamp(1, 64))
                .collect();
            return Ok(ws
                .general_loglik(&texts, masks)?
                .into_iter()
                .map(|x| x.0)
                .collect());
        }
        Source::Dev => {
            let q = ws
                .state
                .qa
                .as_ref()
                .ok_or("not a question-answering task")?;
            crate::lm::dev_items(q)?
        }
        Source::Practice { n, offset } if metric != Metric::Return => practice(ws, n, offset)?,
        Source::Practice { n, offset } => {
            let r = ws.evaluate(n.clamp(1, 256), offset as u64);
            return Ok(r.episode_returns.iter().map(|&x| x as f64).collect());
        }
    };
    match metric {
        Metric::Loglik => Ok(ws
            .qa_loglik(&items, mode, masks)?
            .into_iter()
            .map(|x| x.0)
            .collect()),
        Metric::Accuracy => Ok(ws
            .qa_answer_masked(&items, masks)?
            .into_iter()
            .map(|a| f64::from(u8::from(a.2)))
            .collect()),
        Metric::Return => Err("return is scored on episodes".into()),
    }
}

/// Per-item scores of the live network: `n` practice questions from
/// `offset`, or `n` episodes from seed `offset`. Charges one forward pass.
pub fn score_items(
    ws: &mut WeightSpace,
    metric: Metric,
    mode: &str,
    n: usize,
    offset: usize,
    masks: &[crate::graphops::Unit],
    what: &str,
) -> Result<Vec<f64>, String> {
    ws.charge(what)?;
    score_raw(ws, metric, mode, Source::Practice { n, offset }, masks)
}

/// A small deterministic generator for resampling.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// Paired comparison of per-item scores `a` and `b` (same items, same
/// order): `b − a`.
pub fn paired(a: &[f64], b: &[f64], binary: bool) -> Value {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| y - x).collect();
    let n = d.len();
    let m = mean(&d);
    let sd = if n > 1 {
        (d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    } else {
        0.0
    };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut means: Vec<f64> = (0..4000)
        .map(|_| (0..n).map(|_| d[rng.below(n.max(1))]).sum::<f64>() / n.max(1) as f64)
        .collect();
    means.sort_by(f64::total_cmp);
    let (lo, hi) = (means[100], means[3899]);
    // Two-sided p: exact McNemar for 0/1 scores, else a sign-flip test.
    let p = if binary {
        let up = d.iter().filter(|x| **x > 0.0).count();
        let down = d.iter().filter(|x| **x < 0.0).count();
        binom_two_sided(up.min(down), up + down)
    } else {
        let hits = (0..4000)
            .filter(|_| {
                let s: f64 = d
                    .iter()
                    .map(|x| if rng.next() & 1 == 0 { *x } else { -x })
                    .sum();
                (s / n.max(1) as f64).abs() >= m.abs() - 1e-12
            })
            .count();
        (hits as f64 + 1.0) / 4001.0
    };
    // Both the interval and the test: on 0/1 scores a bootstrap interval
    // can clear 0 while the exact test does not (12 vs 4 of 50: p 0.077).
    let verdict = if lo > 0.0 && p < 0.05 {
        "b better"
    } else if hi < 0.0 && p < 0.05 {
        "b worse"
    } else {
        "no clear difference"
    };
    let mut v = json!({
        "items": n, "mean_b_minus_a": m, "sd": sd,
        "ci95": [lo, hi], "p": p, "verdict": verdict,
    });
    if binary {
        v["b_right_a_wrong"] = json!(d.iter().filter(|x| **x > 0.0).count());
        v["a_right_b_wrong"] = json!(d.iter().filter(|x| **x < 0.0).count());
    }
    v
}

/// `P(X ≤ k or X ≥ n − k)` for `X ~ Binomial(n, ½)`.
fn binom_two_sided(k: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut c = 1.0f64; // C(n, 0)
    let mut tail = 0.0;
    for i in 0..=k {
        if i > 0 {
            c = c * (n - i + 1) as f64 / i as f64;
        }
        tail += c;
    }
    (2.0 * tail / 2f64.powi(n as i32)).min(1.0)
}

/// Score versions `a` and `b` on the same items (two forward passes).
#[allow(clippy::too_many_arguments)]
pub fn compare_variants(
    ws: &mut WeightSpace,
    a: (Variant, Vec<crate::graphops::Unit>),
    b: (Variant, Vec<crate::graphops::Unit>),
    metric: Metric,
    mode: &str,
    n: usize,
    offset: usize,
    what: &str,
) -> Result<Value, String> {
    if ws.remaining() < 2 {
        return Err(format!("{what} needs 2 forward passes"));
    }
    let live = ws.current_variant();
    let t0 = std::time::Instant::now();
    ws.swap_variant(a.0);
    let ra = score_items(ws, metric, mode, n, offset, &a.1, &format!("{what} (a)"));
    ws.swap_variant(b.0);
    let rb = score_items(ws, metric, mode, n, offset, &b.1, &format!("{what} (b)"));
    ws.swap_variant(live);
    let (ra, rb) = (ra?, rb?);
    Ok(json!({
        "metric": metric.name(), "mode": mode, "offset": offset,
        "a_mean": mean(&ra), "b_mean": mean(&rb),
        "paired": paired(&ra, &rb, metric == Metric::Accuracy),
        "seconds": t0.elapsed().as_secs_f32(),
    }))
}

/// Whether the space has general texts to guard comparisons with.
pub fn has_guard(ws: &WeightSpace) -> bool {
    ws.state.qa.as_ref().is_some_and(|q| q.general.is_some())
}

/// The general-text log-likelihood of `a` and `b` (two forward passes): a
/// fall says the edit damages the model beyond the task.
pub fn guard_variants(
    ws: &mut WeightSpace,
    a: &Variant,
    b: &Variant,
    what: &str,
) -> Result<Value, String> {
    if ws.remaining() < 2 {
        return Err(format!("{what} needs 2 forward passes for the guard"));
    }
    let live = ws.current_variant();
    ws.swap_variant(a.clone());
    let ra = ws
        .charge(&format!("{what} guard (a)"))
        .and_then(|_| score_raw(ws, Metric::Loglik, "", Source::General { n: 16 }, &[]));
    ws.swap_variant(b.clone());
    let rb = ws
        .charge(&format!("{what} guard (b)"))
        .and_then(|_| score_raw(ws, Metric::Loglik, "", Source::General { n: 16 }, &[]));
    ws.swap_variant(live);
    let (ra, rb) = (ra?, rb?);
    let p = paired(&ra, &rb, false);
    let mut v = json!({ "metric": "general loglik", "a_mean": mean(&ra), "b_mean": mean(&rb), "paired": p });
    if v["paired"]["verdict"] == "b worse" {
        v["warning"] =
            json!("b models general text worse: the edit may damage the model beyond the task");
    }
    Ok(v)
}

/// Compare `a` and `b` on the held-back dev questions (two forward passes,
/// a limited number of times). When `b` is clearly better its version is
/// kept as checkpoint `verified` — what `--keep verified` scores.
pub fn dev_check(
    ws: &mut WeightSpace,
    a: &str,
    b: &str,
    metric: Metric,
    mode: &str,
) -> Result<Value, String> {
    if ws.state.dev_checks_left == 0 {
        return Err("no dev checks left".into());
    }
    if ws.remaining() < 2 {
        return Err("dev_check needs 2 forward passes".into());
    }
    let q = ws
        .state
        .qa
        .as_ref()
        .ok_or("dev_check is for question-answering tasks")?;
    if q.dev_n == 0 {
        return Err("this task has no dev split".into());
    }
    let (va, vb) = (ws.variant(a)?, ws.variant(b)?);
    ws.state.dev_checks_left -= 1;
    let live = ws.current_variant();
    let t0 = std::time::Instant::now();
    ws.swap_variant(va);
    let ra = ws
        .charge("dev_check (a)")
        .and_then(|_| score_raw(ws, metric, mode, Source::Dev, &[]));
    ws.swap_variant(vb.clone());
    let rb = ws
        .charge("dev_check (b)")
        .and_then(|_| score_raw(ws, metric, mode, Source::Dev, &[]));
    ws.swap_variant(live);
    let (ra, rb) = (ra?, rb?);
    let p = paired(&ra, &rb, metric == Metric::Accuracy);
    let verified = p["verdict"] == "b better";
    if verified {
        ws.state.checkpoints.insert("verified".into(), vb.params);
        ws.state
            .checkpoint_rewrites
            .insert("verified".into(), vb.rewrites);
        ws.state
            .log
            .push(format!("dev_check: `{b}` verified against `{a}`"));
    }
    Ok(json!({
        "metric": metric.name(), "questions": ra.len(),
        "a_mean": mean(&ra), "b_mean": mean(&rb), "paired": p,
        "stored_as_verified": verified,
        "dev_checks_left": ws.state.dev_checks_left,
        "seconds": t0.elapsed().as_secs_f32(),
    }))
}

/// Scale each tensor by `1 + eps` in turn and measure the change.
pub fn sensitivity(
    ws: &mut WeightSpace,
    tensors: &[String],
    metric: Metric,
    mode: &str,
    n: usize,
    offset: usize,
    eps: f32,
) -> Result<Value, String> {
    if tensors.is_empty() || tensors.len() > 16 {
        return Err("give 1..=16 tensors".into());
    }
    let cost = 1 + tensors.len();
    if ws.remaining() < cost {
        return Err(format!("sensitivity needs {cost} forward passes"));
    }
    let base = score_items(ws, metric, mode, n, offset, &[], "sensitivity: base")?;
    let mut rows = Vec::new();
    for t in tensors {
        let live = ws.current_variant();
        let edited = ws.edit(t, &format!("sensitivity probe {t} x{}", 1.0 + eps), |w| {
            for x in &mut w.data {
                *x *= 1.0 + eps;
            }
            Ok(())
        });
        let scored = match edited {
            Ok(()) => score_items(
                ws,
                metric,
                mode,
                n,
                offset,
                &[],
                &format!("sensitivity: {t}"),
            ),
            Err(e) => Err(e),
        };
        ws.swap_variant(live);
        match scored {
            Ok(s) => {
                let d = mean(&s) - mean(&base);
                rows.push(
                    json!({ "tensor": t, "change": d, "per_unit_scale": d / eps as f64,
                                  "paired": paired(&base, &s, metric == Metric::Accuracy) }),
                );
            }
            Err(e) => rows.push(json!({ "tensor": t, "error": e })),
        }
    }
    rows.sort_by(|a, b| {
        let f = |v: &Value| v["change"].as_f64().map_or(-1.0, f64::abs);
        f(b).total_cmp(&f(a))
    });
    Ok(json!({ "metric": metric.name(), "eps": eps, "base_mean": mean(&base), "tensors": rows }))
}

/// One candidate edit for `search`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub name: String,
    /// Tool calls (edits only), e.g. `{"tool": "scale", "name": …, "factor": 1.1}`.
    pub calls: Vec<Value>,
}

/// Tools a candidate may call: edits, no forward passes.
const SEARCH_TOOLS: &[&str] = &[
    "set",
    "set_row",
    "set_col",
    "write",
    "fill",
    "scale",
    "add_rank1",
    "project_out",
    "inject",
    "channel_bias",
    "token_bias",
    "steer",
    "repeat",
];

fn apply_calls(ws: &mut WeightSpace, calls: &[Value]) -> Result<(), String> {
    for c in calls {
        let name = c["tool"].as_str().unwrap_or("");
        if !SEARCH_TOOLS.contains(&name) {
            return Err(format!(
                "`{name}` is not an edit tool (search allows: {})",
                SEARCH_TOOLS.join(", ")
            ));
        }
        crate::tools::allowed(ws, c)?;
        let call: crate::tools::Call =
            serde_json::from_value(c.clone()).map_err(|e| format!("bad call: {e}"))?;
        crate::tools::dispatch(ws, call)?;
    }
    Ok(())
}

/// Forward passes a search takes.
pub fn search_cost(candidates: usize, rounds: usize, confirm: bool) -> usize {
    let mut alive = candidates + 1;
    let mut cost = 0;
    for _ in 0..rounds {
        cost += alive;
        alive = (alive.div_ceil(2) + 1).min(alive).max(2);
    }
    cost + if confirm { 2 } else { 0 }
}

/// Successive halving over `candidates` (and the live network, kept every
/// round as the reference), then the winner against the live network on
/// fresh items.
#[allow(clippy::too_many_arguments)]
pub fn search(
    ws: &mut WeightSpace,
    candidates: &[Candidate],
    metric: Metric,
    mode: &str,
    n: usize,
    rounds: usize,
    offset: usize,
    confirm: bool,
    apply_best: bool,
) -> Result<Value, String> {
    if candidates.is_empty() || candidates.len() > 16 {
        return Err("give 1..=16 candidates".into());
    }
    let rounds = rounds.clamp(1, 4);
    let cost = search_cost(candidates.len(), rounds, confirm);
    if ws.remaining() < cost {
        return Err(format!(
            "this search takes {cost} forward passes ({} left); fewer candidates or rounds",
            ws.remaining()
        ));
    }
    // Index 0 is the live network.
    let mut alive: Vec<usize> = (0..=candidates.len()).collect();
    let mut history: Vec<Vec<Option<f64>>> = vec![Vec::new(); candidates.len() + 1];
    let mut errors: Vec<Option<String>> = vec![None; candidates.len() + 1];
    let mut last_scores: Vec<Vec<f64>> = vec![Vec::new(); candidates.len() + 1];
    let mut used = 0;
    for r in 0..rounds {
        let items = (n << r).min(if metric == Metric::Return { 256 } else { 64 });
        used = used.max(items);
        for &c in &alive {
            let live = ws.current_variant();
            let applied = if c == 0 {
                Ok(())
            } else {
                apply_calls(ws, &candidates[c - 1].calls)
            };
            let scored = applied.and_then(|_| {
                score_items(
                    ws,
                    metric,
                    mode,
                    items,
                    offset,
                    &[],
                    &format!("search round {r}"),
                )
            });
            ws.swap_variant(live);
            match scored {
                Ok(s) => {
                    history[c].push(Some(mean(&s)));
                    last_scores[c] = s;
                }
                Err(e) => {
                    history[c].push(None);
                    errors[c] = Some(e);
                }
            }
        }
        let mut ranked: Vec<usize> = alive
            .iter()
            .copied()
            .filter(|&c| errors[c].is_none())
            .collect();
        ranked.sort_by(|&a, &b| {
            let s = |c: usize| history[c].last().copied().flatten().unwrap_or(f64::MIN);
            s(b).total_cmp(&s(a))
        });
        let keep = ranked.len().div_ceil(2).max(1);
        let mut next: Vec<usize> = ranked.into_iter().take(keep).collect();
        if !next.contains(&0) {
            next.push(0);
        }
        alive = next;
    }
    let best = alive.iter().copied().filter(|&c| c != 0).max_by(|&a, &b| {
        let s = |c: usize| history[c].last().copied().flatten().unwrap_or(f64::MIN);
        s(a).total_cmp(&s(b))
    });
    let name = |c: usize| {
        if c == 0 {
            "current".to_string()
        } else {
            candidates[c - 1].name.clone()
        }
    };
    let mut out = json!({
        "metric": metric.name(),
        "candidates": (0..=candidates.len()).map(|c| json!({
            "name": name(c), "round_means": history[c], "error": errors[c],
        })).collect::<Vec<_>>(),
        "best": best.map(name),
    });
    let Some(best) = best else {
        return Ok(out);
    };
    out["best_vs_current_selection_items"] = paired(
        &last_scores[0],
        &last_scores[best],
        metric == Metric::Accuracy,
    );
    let mut better = history[best].last().copied().flatten() > history[0].last().copied().flatten();
    if confirm {
        // Fresh items: questions after the ones the rounds used, or other
        // episode seeds.
        let fresh = offset + used;
        let live = ws.current_variant();
        apply_calls(ws, &candidates[best - 1].calls)?;
        let edited = ws.current_variant();
        ws.swap_variant(live.clone());
        let c = compare_variants(
            ws,
            (live, Vec::new()),
            (edited, Vec::new()),
            metric,
            mode,
            used,
            fresh,
            "search confirm",
        )?;
        better = c["paired"]["verdict"] == "b better";
        out["confirm_on_fresh_items"] = c;
    }
    if apply_best {
        if better {
            apply_calls(ws, &candidates[best - 1].calls)?;
            ws.state
                .log
                .push(format!("search: applied `{}`", name(best)));
            out["applied"] = json!(name(best));
        } else {
            out["applied"] = Value::Null;
            out["not_applied_because"] = json!(if confirm {
                "the winner was not clearly better than the live network on fresh items"
            } else {
                "the winner did not beat the live network"
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gain_needs_both_the_interval_and_the_exact_test() {
        // 12 questions right only with b, 4 only with a, 34 the same: the
        // bootstrap interval sits at 0 (by item order, a hair above or
        // on it), the exact McNemar test is 0.077.
        let mut a = vec![0.0; 50];
        let mut b = vec![0.0; 50];
        for x in &mut b[..12] {
            *x = 1.0;
        }
        for x in &mut a[12..16] {
            *x = 1.0;
        }
        let p = paired(&a, &b, true);
        assert!((p["p"].as_f64().unwrap() - 0.0768).abs() < 1e-3, "{p}");
        assert_eq!(p["verdict"], "no clear difference");
        // 16 vs 2: both agree.
        for x in &mut b[12..16] {
            *x = 1.0;
        }
        for x in &mut b[16..20] {
            *x = 1.0;
        }
        a[12..16].fill(0.0);
        a[20..22].fill(1.0);
        assert_eq!(paired(&a, &b, true)["verdict"], "b better");
    }

    #[test]
    fn paired_stats_see_a_consistent_gain_and_not_noise() {
        let a: Vec<f64> = (0..40).map(|i| (i % 7) as f64).collect();
        let b: Vec<f64> = a.iter().map(|x| x + 0.5).collect();
        let v = paired(&a, &b, false);
        assert_eq!(v["verdict"], "b better");
        assert!(v["p"].as_f64().unwrap() < 0.01);
        let c: Vec<f64> = a
            .iter()
            .enumerate()
            .map(|(i, x)| if i % 2 == 0 { x + 1.0 } else { x - 1.0 })
            .collect();
        assert_eq!(paired(&a, &c, false)["verdict"], "no clear difference");
        // 0/1: 9 gained, 1 lost of 40.
        let a01: Vec<f64> = (0..40).map(|i| f64::from(u8::from(i < 10))).collect();
        let mut b01 = a01.clone();
        for x in b01.iter_mut().skip(10).take(9) {
            *x = 1.0;
        }
        b01[0] = 0.0;
        let v = paired(&a01, &b01, true);
        assert_eq!(v["b_right_a_wrong"], 9);
        assert!(
            (v["p"].as_f64().unwrap() - 0.021_484_375).abs() < 1e-9,
            "{v}"
        );
    }

    #[test]
    fn search_cost_counts_the_reference_every_round() {
        // 4 candidates + current = 5; then 3 + current kept … ; + 2 confirm.
        assert_eq!(search_cost(4, 1, false), 5);
        assert_eq!(search_cost(4, 2, true), 5 + 4 + 2);
    }
}
