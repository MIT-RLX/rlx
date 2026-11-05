// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! LoRA baselines: low-rank adapters trained on the practice set, as a
//! reference for what training reaches where the agent writes weights.
//!
//! - **Forward-only** ([`train_mezo`], MeZO-style): the factors `B·A` of
//!   each target tensor are moved along a random direction `z` by `±eps`,
//!   the task metric measured both ways on the same practice items, and the
//!   factors stepped by `lr · (f₊ − f₋)/2eps · z`. Forward passes only, so it
//!   runs on any package (a decode-only language model included) and any
//!   task network — slowly: one step is two evaluations.
//! - **Backprop** ([`train_backprop`]): a package exported with a trainable
//!   full-sequence function (rlx-models: `rlx-export … --variant train`) is
//!   trained by `rlx_runtime::model::trainer` (LoRA rewrite, next-token
//!   cross-entropy masked to the reference answer, AdamW); the trained
//!   factors are then applied to the space's own package by tensor name.
//!
//! Either way the adapter ends up in the space as low-rank factors
//! (`W + Σ u vᵀ`), so `score`, `generate`, `export` and every tool see the
//! adapted model, and the adapter package is written like any overlay.
//! These are baselines, not agent tools: they train with the practice set,
//! the agent writes weights.

use crate::measure::{Metric, Source, score_raw};
use crate::space::{Variant, WeightSpace};
#[cfg(feature = "training")]
use rlx_runtime::model::Tensor as MTensor;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub struct LoraOptions {
    /// Tensor-name patterns (`*` matches any run of characters).
    pub targets: Vec<String>,
    pub rank: usize,
    pub alpha: f32,
    pub steps: usize,
    pub lr: f32,
    /// Items (questions, or episodes) per forward-only step.
    pub batch: usize,
    /// Forward-only perturbation size.
    pub eps: f32,
    /// QA scoring mode (`solution` | `answer`).
    pub mode: String,
    /// Steps between progress lines.
    pub eval_every: usize,
    pub seed: u64,
    /// Equal-compute cap: stop once this many tokens are processed (the
    /// unit the agent's token budget counts; backprop counts a token 3×,
    /// for its forward and backward).
    pub max_tokens: Option<usize>,
    /// Skip a batch whose loss exceeds `gate` × the median of recent ones
    /// (the batch is scored before the step).
    pub gate: Option<f32>,
    /// Keep the adapter every `snapshot_every` steps and, at the end, pick
    /// the final one or an average of the last 2 / 4 / 8 snapshots by
    /// their score on the dev split (needs one).
    pub average: bool,
    pub snapshot_every: usize,
    /// Train on these question / answer rows (GSM8K JSONL, e.g. from
    /// [`distill`]) instead of the practice set; the dev split and the
    /// held-out questions stay the task's.
    pub train_items: Option<String>,
}

impl Default for LoraOptions {
    fn default() -> Self {
        Self {
            targets: vec!["*attn_q*".into(), "*attn_v*".into()],
            rank: 8,
            alpha: 16.0,
            steps: 100,
            lr: 1e-4,
            batch: 4,
            eps: 1e-3,
            mode: "solution".into(),
            eval_every: 10,
            seed: 0,
            max_tokens: None,
            gate: None,
            average: false,
            snapshot_every: 20,
            train_items: None,
        }
    }
}

/// `*`-glob match.
pub fn glob(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == name;
    }
    let mut rest = name;
    for (i, p) in parts.iter().enumerate() {
        if i == 0 {
            match rest.strip_prefix(p) {
                Some(r) => rest = r,
                None => return false,
            }
        } else if i == parts.len() - 1 {
            return rest.ends_with(p);
        } else {
            match rest.find(p) {
                Some(k) => rest = &rest[k + p.len()..],
                None => return false,
            }
        }
    }
    true
}

/// The 2-D, editable tensors whose names match `patterns`.
fn targets_of(ws: &WeightSpace, patterns: &[String]) -> Result<Vec<String>, String> {
    let names: Vec<String> = match ws.store() {
        Some(st) => st
            .names()
            .filter(|n| st.editable(n) && st.entry(n).is_some_and(|(s, _)| s.len() == 2))
            .cloned()
            .collect(),
        None => ws
            .state
            .params
            .iter()
            .filter(|(_, t)| t.rows > 1)
            .map(|(n, _)| n.clone())
            .collect(),
    };
    let picked: Vec<String> = names
        .into_iter()
        .filter(|n| patterns.iter().any(|p| glob(p, n)))
        .collect();
    if picked.is_empty() {
        return Err(format!(
            "no 2-D editable tensor matches {patterns:?} (see list_params)"
        ));
    }
    Ok(picked)
}

/// A small generator: xorshift, Gaussian by Box–Muller.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gaussian(&mut self) -> f32 {
        let (a, b) = (self.uniform(), self.uniform());
        ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
    }
}

fn median(v: &[f64]) -> Option<f64> {
    if v.len() < 8 {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    Some(s[s.len() / 2])
}

/// Pick among `candidates` (name, install) the one scoring best on the dev
/// split (mean log-likelihood per token), starting each from `live`; leave
/// it installed. Returns the choice and every score.
fn select_on_dev(
    ws: &mut WeightSpace,
    live: &Variant,
    candidates: Vec<(
        String,
        Box<dyn Fn(&mut WeightSpace) -> Result<(), String> + '_>,
    )>,
    mode: &str,
) -> Result<Value, String> {
    if ws.state.qa.as_ref().is_none_or(|q| q.dev_n == 0) {
        return Err("averaging picks on the dev split: make the space with --dev-n".into());
    }
    let mut scores = Vec::new();
    for (name, install) in &candidates {
        ws.swap_variant(live.clone());
        install(ws)?;
        let s = score_raw(ws, Metric::Loglik, mode, Source::Dev, &[])?;
        scores.push((name.clone(), s.iter().sum::<f64>() / s.len().max(1) as f64));
    }
    let best = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.1.total_cmp(&b.1.1))
        .map(|(i, _)| i)
        .ok_or("no candidates")?;
    ws.swap_variant(live.clone());
    (candidates[best].1)(ws)?;
    Ok(
        json!({ "selected": scores[best].0, "dev_loglik": scores.iter().map(|(n, v)| json!({ "candidate": n, "mean_logprob": v })).collect::<Vec<_>>() }),
    )
}

/// One forward-only (SPSA) step on `theta`, minimizing `f`: the loss at
/// `theta ± eps·z` (same `z`, drawn from `seed`) and the step `lr ·
/// (f₊ − f₋)/(2 eps) · z`. Returns `(f₊, f₋)`.
pub fn spsa_step(
    theta: &mut [f32],
    eps: f32,
    lr: f32,
    seed: u64,
    mut f: impl FnMut(&[f32]) -> Result<f64, String>,
) -> Result<(f64, f64), String> {
    let mut rng = Rng::new(seed);
    let z: Vec<f32> = (0..theta.len()).map(|_| rng.gaussian()).collect();
    let plus: Vec<f32> = theta.iter().zip(&z).map(|(t, z)| t + eps * z).collect();
    let minus: Vec<f32> = theta.iter().zip(&z).map(|(t, z)| t - eps * z).collect();
    let (fp, fm) = (f(&plus)?, f(&minus)?);
    let g = ((fp - fm) / (2.0 * eps as f64)) as f32;
    for (t, z) in theta.iter_mut().zip(&z) {
        *t -= lr * g * z;
    }
    Ok((fp, fm))
}

/// One target's factors inside the flat parameter vector: `B` `[rows, r]`
/// then `A` `[r, cols]`.
struct Site {
    name: String,
    rows: usize,
    cols: usize,
    offset: usize,
}

/// The live network plus `scale · B·A` on each site, as low-rank factors.
fn install(
    ws: &mut WeightSpace,
    live: &Variant,
    sites: &[Site],
    theta: &[f32],
    rank: usize,
    scale: f32,
    note: &str,
) -> Result<(), String> {
    ws.swap_variant(live.clone());
    let log_len = ws.state.log.len();
    for s in sites {
        let b = &theta[s.offset..s.offset + s.rows * rank];
        let a = &theta[s.offset + s.rows * rank..s.offset + s.rows * rank + rank * s.cols];
        for k in 0..rank {
            let u: Vec<f32> = (0..s.rows).map(|i| scale * b[i * rank + k]).collect();
            if u.iter().all(|x| *x == 0.0) {
                continue;
            }
            let v = a[k * s.cols..(k + 1) * s.cols].to_vec();
            ws.edit_rank1(&s.name, u, v, note)?;
        }
    }
    ws.state.log.truncate(log_len);
    Ok(())
}

/// Forward-only LoRA training (see the module docs); not budgeted. Leaves
/// the trained adapter installed and returns the training record.
pub fn train_mezo(ws: &mut WeightSpace, o: &LoraOptions) -> Result<Value, String> {
    if o.train_items.is_some() {
        return Err("--train-items is for backprop LoRA (MeZO scores the practice set)".into());
    }
    let metric = Metric::parse(ws, None)?;
    let names = targets_of(ws, &o.targets)?;
    let mut sites = Vec::new();
    let mut len = 0;
    for n in names {
        let t = ws.param(&n)?;
        let (rows, cols) = (t.rows, t.cols);
        sites.push(Site {
            name: n,
            rows,
            cols,
            offset: len,
        });
        len += rows * o.rank + o.rank * cols;
    }
    // B starts at zero (the model unchanged), A small and random.
    let mut theta = vec![0f32; len];
    let mut rng = Rng::new(o.seed ^ 0xA11CE);
    for s in &sites {
        let a0 = s.offset + s.rows * o.rank;
        let sd = 1.0 / (s.cols as f32).sqrt();
        for x in &mut theta[a0..a0 + o.rank * s.cols] {
            *x = sd * rng.gaussian();
        }
    }
    let scale = o.alpha / o.rank as f32;
    let live = ws.current_variant();
    let practice_len = match metric {
        Metric::Return => usize::MAX,
        _ => crate::lm::practice_items(ws.state.qa.as_ref().expect("qa"))?.len(),
    };
    let batch = o.batch.max(1);
    let mut curve = Vec::new();
    let t0 = std::time::Instant::now();
    let tokens0 = ws.state.tokens_used;
    let budget_before = ws.state.token_budget;
    ws.state.token_budget = o.max_tokens.map(|m| tokens0 + m);
    let mut stopped = None;
    let mut done = 0;
    let mut snapshots: Vec<Vec<f32>> = Vec::new();
    let mut history: Vec<f64> = Vec::new();
    let mut gated = 0;
    for step in 0..o.steps {
        let offset = match metric {
            Metric::Return => 10_000 + step * batch,
            _ => (step * batch) % practice_len.saturating_sub(batch).max(1),
        };
        let mut f = |th: &[f32]| -> Result<f64, String> {
            install(ws, &live, &sites, th, o.rank, scale, "lora (mezo)")?;
            let s = score_raw(
                ws,
                metric,
                &o.mode,
                Source::Practice { n: batch, offset },
                &[],
            )?;
            // Minimize the negative metric.
            Ok(-s.iter().sum::<f64>() / s.len().max(1) as f64)
        };
        let theta_before = theta.clone();
        let stepped = spsa_step(&mut theta, o.eps, o.lr, o.seed ^ (step as u64 + 1), &mut f);
        let (fp, fm) = match stepped {
            Ok(x) => x,
            Err(e) if e.contains("token budget exhausted") => {
                stopped = Some(format!("token budget reached after {step} steps"));
                break;
            }
            Err(e) => {
                ws.state.token_budget = budget_before;
                return Err(e);
            }
        };
        done = step + 1;
        // The gate: a step whose losses spike against the recent median is
        // undone.
        if let (Some(g), Some(m)) = (o.gate, median(&history))
            && fp.max(fm) - m > (g as f64 - 1.0) * m.abs()
        {
            theta = theta_before;
            gated += 1;
        } else {
            history.push((fp + fm) / 2.0);
        }
        if o.average && (step + 1) % o.snapshot_every.max(1) == 0 {
            snapshots.push(theta.clone());
        }
        if step % o.eval_every.max(1) == 0 || step + 1 == o.steps {
            let line = json!({ "step": step, "metric_plus": -fp, "metric_minus": -fm,
                               "seconds": t0.elapsed().as_secs_f32() });
            eprintln!("lora mezo {line}");
            curve.push(line);
        }
    }
    ws.state.token_budget = budget_before;
    let mut selection = Value::Null;
    if o.average && !snapshots.is_empty() {
        let mut cands: Vec<(
            String,
            Box<dyn Fn(&mut WeightSpace) -> Result<(), String> + '_>,
        )> = Vec::new();
        let fin = theta.clone();
        let (sites_r, rank) = (&sites, o.rank);
        cands.push((
            "final".into(),
            Box::new(move |ws: &mut WeightSpace| {
                install(
                    ws,
                    &ws.current_variant(),
                    sites_r,
                    &fin,
                    rank,
                    scale,
                    "lora (mezo)",
                )
            }),
        ));
        for k in [2usize, 4, 8] {
            if snapshots.len() < k {
                break;
            }
            let tail = &snapshots[snapshots.len() - k..];
            let mut mean = vec![0f32; theta.len()];
            for t in tail {
                mean.iter_mut().zip(t).for_each(|(m, x)| *m += x / k as f32);
            }
            cands.push((
                format!("average of last {k}"),
                Box::new(move |ws: &mut WeightSpace| {
                    install(
                        ws,
                        &ws.current_variant(),
                        sites_r,
                        &mean,
                        rank,
                        scale,
                        "lora (mezo)",
                    )
                }),
            ));
        }
        selection = select_on_dev(ws, &live, cands, &o.mode)?;
    } else {
        install(ws, &live, &sites, &theta, o.rank, scale, "lora (mezo)")?;
    }
    ws.state.log.push(format!(
        "lora mezo: {done} steps on {} tensors, rank {}",
        sites.len(),
        o.rank
    ));
    Ok(json!({
        "method": "mezo (forward-only)", "metric": metric.name(),
        "targets": sites.iter().map(|s| &s.name).collect::<Vec<_>>(),
        "rank": o.rank, "alpha": o.alpha, "steps": done, "lr": o.lr, "eps": o.eps,
        "trained_values": len, "curve": curve,
        "evaluations": 2 * done, "tokens_processed": ws.state.tokens_used - tokens0,
        "stopped": stopped, "gated_steps": gated, "selection": selection,
        "seconds": t0.elapsed().as_secs_f32(),
    }))
}

/// Apply trainer LoRA updates (`W + scale·B·A`, each in its stored layout)
/// to the space's tensors of the same names, as low-rank factors.
pub fn install_updates(
    ws: &mut WeightSpace,
    updates: &[rlx_runtime::model::rlxp::LoraUpdate],
) -> Result<Vec<String>, String> {
    let mut done = Vec::new();
    for up in updates {
        let t = ws.param(&up.tensor).map_err(|e| {
            format!(
                "the adapted tensor `{}` is not in this space: {e}",
                up.tensor
            )
        })?;
        let (rows, cols) = (t.rows, t.cols);
        let r = up.rank;
        let (b_rows, a_cols) = (up.b.len() / r, up.a.len() / r);
        let direct = (b_rows, a_cols) == (rows, cols);
        if !direct && (a_cols, b_rows) != (rows, cols) {
            return Err(format!(
                "`{}`: the update is [{b_rows}, {a_cols}], the tensor here [{rows}, {cols}]",
                up.tensor
            ));
        }
        for k in 0..r {
            let bk: Vec<f32> = (0..b_rows).map(|i| up.scale * up.b[i * r + k]).collect();
            let ak = up.a[k * a_cols..(k + 1) * a_cols].to_vec();
            let (u, v) = if direct { (bk, ak) } else { (ak, bk) };
            ws.edit_rank1(&up.tensor, u, v, "lora (trained)")?;
        }
        done.push(up.tensor.clone());
    }
    Ok(done)
}

#[cfg(feature = "training")]
/// Train a LoRA on `train_pkg` (a package with a trainable function) with
/// `options` (`rlx_runtime::model::trainer` options: function, loss,
/// train, optimizer), feeding `batch(step)` each step; install the trained
/// factors into the space. Returns the losses.
pub fn train_backprop_with(
    ws: &mut WeightSpace,
    train_pkg: &str,
    options: &Value,
    o: &LoraOptions,
    mut batch: impl FnMut(
        &mut WeightSpace,
        usize,
        &[(String, Vec<usize>)],
    ) -> Result<Vec<MTensor>, String>,
) -> Result<Value, String> {
    let (steps, eval_every, max_tokens) = (o.steps, o.eval_every, o.max_tokens);
    let mut trainer = rlx_runtime::model::trainer::open_trainer(train_pkg, options)
        .map_err(|e| format!("{train_pkg}: {e:#}"))?;
    let live = ws.current_variant();
    let mut snapshots: Vec<Vec<rlx_runtime::model::rlxp::LoraUpdate>> = Vec::new();
    let mut accepted: Vec<f64> = Vec::new();
    let mut gated = 0;
    let inputs = trainer.inputs().to_vec();
    let mut losses = Vec::with_capacity(steps);
    let mut curve = Vec::new();
    let t0 = std::time::Instant::now();
    // Tokens a step processes: the batch's sequence, forward and backward.
    let per_step = 3 * inputs
        .iter()
        .find(|(n, _)| n == "input_ids")
        .map_or(1, |(_, d)| d.iter().product::<usize>());
    let mut stopped = None;
    for step in 0..steps {
        if max_tokens.is_some_and(|m| (step + 1) * per_step > m) {
            stopped = Some(format!("token budget reached after {step} steps"));
            break;
        }
        let b = batch(ws, step, &inputs)?;
        // The gate: score the batch first; skip it when its loss spikes.
        if let Some(g) = o.gate
            && let Some(m) = median(&accepted[accepted.len().saturating_sub(32)..])
        {
            let pre = trainer
                .loss(&b)
                .map_err(|e| format!("step {step}: {e:#}"))? as f64;
            if pre - m > (g as f64 - 1.0) * m.abs() {
                gated += 1;
                continue;
            }
        }
        let loss = trainer
            .step(&b)
            .map_err(|e| format!("step {step}: {e:#}"))?;
        losses.push(loss);
        accepted.push(loss as f64);
        if o.average && (step + 1) % o.snapshot_every.max(1) == 0 {
            snapshots.push(
                trainer
                    .lora_updates()
                    .map_err(|e| format!("reading the factors: {e:#}"))?,
            );
        }
        if step % eval_every.max(1) == 0 || step + 1 == steps {
            let k = eval_every.max(1).min(losses.len());
            let recent = losses[losses.len() - k..].iter().sum::<f32>() / k as f32;
            let line = json!({ "step": step, "loss": loss, "mean_recent": recent,
                               "seconds": t0.elapsed().as_secs_f32() });
            eprintln!("lora backprop {line}");
            curve.push(line);
        }
    }
    let updates = trainer
        .lora_updates()
        .map_err(|e| format!("reading the trained factors: {e:#}"))?;
    let mut selection = Value::Null;
    let installed = if o.average && !snapshots.is_empty() {
        let scaled = |us: &[rlx_runtime::model::rlxp::LoraUpdate], w: f32| {
            us.iter()
                .map(|u| rlx_runtime::model::rlxp::LoraUpdate {
                    tensor: u.tensor.clone(),
                    rank: u.rank,
                    a: u.a.clone(),
                    b: u.b.clone(),
                    scale: u.scale * w,
                })
                .collect::<Vec<_>>()
        };
        let mut cands: Vec<(
            String,
            Box<dyn Fn(&mut WeightSpace) -> Result<(), String> + '_>,
        )> = Vec::new();
        let fin = scaled(&updates, 1.0);
        cands.push((
            "final".into(),
            Box::new(move |ws: &mut WeightSpace| install_updates(ws, &fin).map(|_| ())),
        ));
        for k in [2usize, 4, 8] {
            if snapshots.len() < k {
                break;
            }
            // The mean of k adapters: each at 1/k (a rank-k·r update).
            let parts: Vec<_> = snapshots[snapshots.len() - k..]
                .iter()
                .map(|s| scaled(s, 1.0 / k as f32))
                .collect();
            cands.push((
                format!("average of last {k}"),
                Box::new(move |ws: &mut WeightSpace| {
                    for p in &parts {
                        install_updates(ws, p)?;
                    }
                    Ok(())
                }),
            ));
        }
        selection = select_on_dev(ws, &live, cands, &o.mode)?;
        updates.iter().map(|u| u.tensor.clone()).collect()
    } else {
        install_updates(ws, &updates)?
    };
    ws.state.log.push(format!(
        "lora backprop: {} steps, {} tensors adapted",
        losses.len(),
        installed.len()
    ));
    Ok(json!({
        "method": "backprop (rlx_runtime trainer)",
        "adapted": installed, "steps": losses.len(), "curve": curve,
        "tokens_processed": (losses.len() + gated) * per_step, "stopped": stopped,
        "gated_steps": gated, "selection": selection,
        "first_loss": losses.first(), "last_loss": losses.last(),
        "seconds": t0.elapsed().as_secs_f32(),
    }))
}

#[cfg(feature = "training")]
/// Backprop LoRA on a QA task: each step one practice question, the input
/// the prompt and reference solution, the loss next-token cross-entropy on
/// the reference answer's tokens only.
pub fn train_backprop(
    ws: &mut WeightSpace,
    train_pkg: &str,
    o: &LoraOptions,
) -> Result<Value, String> {
    let q = ws
        .state
        .qa
        .clone()
        .ok_or("backprop LoRA trains on a question-answering task's practice set")?;
    let items = match &o.train_items {
        Some(f) => crate::lm::load_items(f)?,
        None => crate::lm::practice_items(&q)?,
    };
    if items.is_empty() {
        return Err("no items to train on".into());
    }
    let options = qa_train_options(&o.targets, o.rank, o.alpha, o.lr);
    let mode = o.mode.clone();
    train_backprop_with(ws, train_pkg, &options, o, |ws, step, inputs| {
        let seq = seq_of(inputs)?;
        Ok(qa_batch(ws, &items[step % items.len()], &mode, seq)?
            .into_iter()
            .map(|(n, v)| MTensor::f32(n, vec![1, seq], &v))
            .collect())
    })
}

/// The trainer options for LoRA on a QA package: next-token cross-entropy
/// masked to the reference answer.
#[cfg(feature = "training")]
pub fn qa_train_options(targets: &[String], rank: usize, alpha: f32, lr: f32) -> Value {
    json!({
        "loss": { "cross_entropy": { "logits": "logits", "labels": "labels", "mask": "mask" } },
        "train": { "lora": { "targets": targets, "rank": rank, "alpha": alpha } },
        "optimizer": { "lr": lr },
    })
}

/// The trained function's sequence length (its `input_ids`).
#[cfg(feature = "training")]
pub fn seq_of(inputs: &[(String, Vec<usize>)]) -> Result<usize, String> {
    inputs
        .iter()
        .find(|(n, _)| n == "input_ids")
        .and_then(|(_, d)| d.last().copied())
        .ok_or_else(|| "the trained function has no `input_ids` input".to_string())
}

/// One question as a training example: `input_ids`, `labels` (the next
/// token) and `mask` (1 on the reference answer's tokens), `seq` long.
#[cfg(feature = "training")]
pub fn qa_batch(
    ws: &mut WeightSpace,
    it: &crate::lm::QaItem,
    mode: &str,
    seq: usize,
) -> Result<Vec<(String, Vec<f32>)>, String> {
    let p = ws.qa_prompt(&it.question)?;
    let (ctx, cont) = crate::lm::split_reference(it, mode, &p)?;
    let codec = ws.tokenizer().ok_or("the package carries no tokenizer")?;
    let c = codec
        .encode(&ctx)
        .map_err(|e| format!("tokenizing: {e:#}"))?;
    let r = codec
        .encode_plain(&cont)
        .map_err(|e| format!("tokenizing: {e:#}"))?;
    let ids: Vec<u32> = c.iter().chain(&r).copied().collect();
    let (mut x, mut y, mut m) = (vec![0f32; seq], vec![0f32; seq], vec![0f32; seq]);
    for i in 0..seq.min(ids.len().saturating_sub(1)) {
        x[i] = ids[i] as f32;
        y[i] = ids[i + 1] as f32;
        // Only the reference answer's tokens are learned.
        if i + 1 >= c.len() {
            m[i] = 1.0;
        }
    }
    Ok(vec![
        ("input_ids".into(), x),
        ("labels".into(), y),
        ("mask".into(), m),
    ])
}

/// Context distillation data: practice questions answered with `shots`
/// worked examples (the first practice items' reference solutions) in
/// front of the prompt, the correct answers kept as GSM8K rows
/// (`{question, answer}`, the answer ending in `#### n`). Training on them
/// with the plain prompt moves what the examples do into the weights.
/// Questions are practice items `shots + offset ..` (dev and near-duplicate
/// ones are never among them); each row is handed to `row` as it is made.
pub fn distill(
    ws: &mut WeightSpace,
    shots: usize,
    n: usize,
    offset: usize,
    row: &mut dyn FnMut(&Value) -> Result<(), String>,
) -> Result<Value, String> {
    let q = ws
        .state
        .qa
        .clone()
        .ok_or("distillation answers a question-answering task's practice questions")?;
    if q.prompt_style != "plain" {
        return Err(
            "distillation puts worked examples before plain prompts (--prompt-style plain)".into(),
        );
    }
    let items = crate::lm::practice_items(&q)?;
    if items.len() <= shots {
        return Err(format!(
            "{} practice items, {shots} of them examples",
            items.len()
        ));
    }
    let mut prefix = String::new();
    for it in &items[..shots] {
        let p = ws.qa_prompt(&it.question)?;
        let (ctx, cont) = crate::lm::split_reference(it, "solution", &p)?;
        prefix.push_str(&format!("{ctx}{cont}\n\n"));
    }
    let picked: Vec<&crate::lm::QaItem> = items[shots..].iter().skip(offset).take(n).collect();
    let (mut kept, mut done) = (0, 0);
    let t0 = std::time::Instant::now();
    for chunk in picked.chunks(16) {
        let prompts: Vec<String> = chunk
            .iter()
            .map(|it| Ok(format!("{prefix}{}", ws.qa_prompt(&it.question)?)))
            .collect::<Result<_, String>>()?;
        let max = q.max_tokens;
        let outs = ws.lm_generate_masked(&prompts, max, &[], true)?;
        for (it, o) in chunk.iter().zip(&outs) {
            let answer = crate::lm::cut_at_answer(o).trim();
            if crate::lm::final_answer(answer).as_deref() == Some(it.gold.as_str())
                && answer.contains("####")
            {
                row(&json!({ "question": it.question, "answer": answer }))?;
                kept += 1;
            }
        }
        done += chunk.len();
        eprintln!(
            "distill: {done}/{} answered, {kept} correct ({:.0}s)",
            picked.len(),
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(json!({
        "questions": done, "kept": kept,
        "accuracy_with_examples": kept as f64 / done.max(1) as f64,
        "shots": shots, "seconds": t0.elapsed().as_secs_f64(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs_match() {
        assert!(glob("*attn_q*", "blk.3.attn_qkv.weight"));
        assert!(glob("blk.*.ffn_down.weight", "blk.12.ffn_down.weight"));
        assert!(!glob("*attn_v*", "blk.3.ffn_up.weight"));
        assert!(glob("l1.w", "l1.w"));
    }

    #[test]
    fn spsa_descends_a_quadratic() {
        let target = [1.0f32, -2.0, 0.5, 3.0];
        let f = |t: &[f32]| -> Result<f64, String> {
            Ok(t.iter()
                .zip(&target)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum())
        };
        let mut theta = vec![0f32; 4];
        let start = f(&theta).unwrap();
        for s in 0..400 {
            spsa_step(&mut theta, 1e-3, 0.05, s, f).unwrap();
        }
        let end = f(&theta).unwrap();
        assert!(end < start * 0.01, "{start} -> {end} ({theta:?})");
    }
}
