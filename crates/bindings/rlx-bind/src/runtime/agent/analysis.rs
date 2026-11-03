// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The agent's analysis program: `analysis.json` says which statistics each
//! report holds, how often reports are made, and when the run should ask for
//! a decision on its own. It only reads; it never changes an update.
//!
//! Like `updater.json` it is data, not code, and the same for every program
//! host: the agent edits it in its workspace, and a change to it alone
//! replaces the analysis without rebuilding the updater or opening a trial.

use std::collections::HashMap;

use rlx_optim::OptItem;
use serde_json::{Map, Value, json};

use super::program::glob;
use crate::runtime::opts::Opts;
use crate::{ApiError, ApiResult};

pub const ANALYSIS: &str = "analysis.json";

/// Per-param tables list at most this many tensors (the largest gradients).
const MAX_SHOWN: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Stats {
    grad_norm: bool,
    weight_norm: bool,
    update_ratio: bool,
    grad_cosine: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WakeStat {
    LossMean,
    LossMax,
    GradNormTotal,
}

impl WakeStat {
    fn name(self) -> &'static str {
        match self {
            Self::LossMean => "loss_mean",
            Self::LossMax => "loss_max",
            Self::GradNormTotal => "grad_norm_total",
        }
    }
}

#[derive(Clone, Debug)]
struct Wake {
    stat: WakeStat,
    above: Option<f64>,
    below: Option<f64>,
    /// True while the condition holds, so it asks once per episode.
    firing: bool,
}

/// A parsed `analysis.json`, with what it remembers between reports.
#[derive(Clone, Debug)]
pub struct Analysis {
    /// Updates between reports.
    pub every: u64,
    stats: Stats,
    params: Vec<String>,
    /// Globs over the forward graph's further outputs to summarise.
    activations: Vec<String>,
    wake: Vec<Wake>,
    previous_grads: HashMap<String, Vec<f32>>,
}

/// Mean, standard deviation, RMS and largest magnitude of an output, and how
/// many of its values are not finite.
fn activation_stats(v: &[f32]) -> Value {
    let finite: Vec<f64> = v
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| *x as f64)
        .collect();
    let n = finite.len().max(1) as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let var = finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let rms = (finite.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
    let max_abs = finite.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    json!({"mean": round(mean), "std": round(var.sqrt()), "rms": round(rms),
           "max_abs": round(max_abs), "nonfinite": v.len() - finite.len()})
}

impl Analysis {
    /// The starting analysis: loss, gradient and weight norms every `every` updates.
    pub fn starting_text(every: u64) -> String {
        let v = json!({
            "every": every,
            "stats": ["grad_norm", "weight_norm"],
            "params": ["*"],
            "wake": [],
        });
        serde_json::to_string_pretty(&v).unwrap() + "\n"
    }

    pub fn parse(text: &str) -> ApiResult<Self> {
        let v: Value = serde_json::from_str(text)
            .map_err(|e| ApiError::type_(format!("{ANALYSIS} is not valid JSON: {e}")))?;
        let o = Opts::new(ANALYSIS, &v)?;
        let every = o.u64("every")?.unwrap_or(50);
        if every == 0 {
            return Err(ApiError::range(format!(
                "{ANALYSIS}: every must be at least 1"
            )));
        }
        let mut stats = Stats::default();
        for s in strings(o.get("stats"), "stats")? {
            match s.as_str() {
                "grad_norm" => stats.grad_norm = true,
                "weight_norm" => stats.weight_norm = true,
                "update_ratio" => stats.update_ratio = true,
                "grad_cosine" => stats.grad_cosine = true,
                other => {
                    return Err(ApiError::type_(format!(
                        "{ANALYSIS}: unknown stat {other:?} (grad_norm, weight_norm, update_ratio, grad_cosine)"
                    )));
                }
            }
        }
        let mut params = strings(o.get("params"), "params")?;
        if params.is_empty() {
            params.push("*".into());
        }
        let activations = strings(o.get("activations"), "activations")?;
        let wake = match o.get("wake") {
            None => Vec::new(),
            Some(Value::Array(rules)) => rules
                .iter()
                .enumerate()
                .map(|(i, r)| Wake::parse(i, r))
                .collect::<ApiResult<_>>()?,
            Some(other) => {
                return Err(ApiError::type_(format!(
                    "{ANALYSIS}: `wake` must be a list, got {other}"
                )));
            }
        };
        o.finish()?;
        Ok(Self {
            every,
            stats,
            params,
            activations,
            wake,
            previous_grads: HashMap::new(),
        })
    }

    /// Whether the report needs the params from before the update.
    pub fn wants_update_ratio(&self) -> bool {
        self.stats.update_ratio
    }

    /// The report of update `u` (made after it) and the reasons, if any, to
    /// ask for a decision now. `before` are the params before the update, in
    /// `items` order, when [`Self::wants_update_ratio`].
    pub fn report(
        &mut self,
        u: u64,
        losses: &[f32],
        items: &[OptItem<'_>],
        before: Option<&[Vec<f32>]>,
        outputs: &[(String, Vec<f32>)],
        program_ms: Option<f64>,
    ) -> (Value, Vec<String>) {
        let finite: Vec<f32> = losses.iter().copied().filter(|x| x.is_finite()).collect();
        let mean =
            (!finite.is_empty()).then(|| finite.iter().sum::<f32>() as f64 / finite.len() as f64);
        let max = finite.iter().copied().reduce(f32::max).map(f64::from);
        let min = finite.iter().copied().reduce(f32::min).map(f64::from);
        let grad_norms: Vec<f64> = items.iter().map(|it| norm(it.grad)).collect();
        let total = grad_norms.iter().map(|g| g * g).sum::<f64>().sqrt();

        let mut picked: Vec<usize> = (0..items.len())
            .filter(|&i| self.params.iter().any(|g| glob(g, items[i].name)))
            .collect();
        if picked.len() > MAX_SHOWN {
            picked.sort_by(|&a, &b| grad_norms[b].total_cmp(&grad_norms[a]));
            picked.truncate(MAX_SHOWN);
        }
        let table = |f: &dyn Fn(usize) -> Option<f64>| -> Value {
            let m: Map<String, Value> = picked
                .iter()
                .filter_map(|&i| f(i).map(|v| (items[i].name.to_string(), json!(round(v)))))
                .collect();
            Value::Object(m)
        };
        let mut r = json!({
            "update": u,
            "loss_mean": mean.map(round),
            "loss_min": min.map(round),
            "loss_max": max.map(round),
            "nonfinite_losses": losses.len() - finite.len(),
            "grad_norm_total": round(total),
            "program_ms_per_update": program_ms.map(round),
        });
        let captured: Map<String, Value> = outputs
            .iter()
            .filter(|(name, _)| self.activations.iter().any(|g| glob(g, name)))
            .map(|(name, v)| (name.clone(), activation_stats(v)))
            .collect();
        if !captured.is_empty() {
            r["activations"] = Value::Object(captured);
        }
        if self.stats.grad_norm {
            r["grad_norm"] = table(&|i| Some(grad_norms[i]));
        }
        if self.stats.weight_norm {
            r["weight_norm"] = table(&|i| Some(norm(items[i].param)));
        }
        if self.stats.update_ratio
            && let Some(before) = before
        {
            r["update_ratio"] = table(&|i| {
                let w = norm(&before[i]);
                let d = items[i]
                    .param
                    .iter()
                    .zip(&before[i])
                    .map(|(a, b)| ((a - b) as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (w > 0.0).then(|| d / w)
            });
        }
        if self.stats.grad_cosine {
            let prev = &self.previous_grads;
            r["grad_cosine"] = table(&|i| {
                let p = prev.get(items[i].name)?;
                let (a, b) = (norm(p), grad_norms[i]);
                (a > 0.0 && b > 0.0).then(|| {
                    p.iter()
                        .zip(items[i].grad)
                        .map(|(x, y)| *x as f64 * *y as f64)
                        .sum::<f64>()
                        / (a * b)
                })
            });
            self.previous_grads = picked
                .iter()
                .map(|&i| (items[i].name.to_string(), items[i].grad.to_vec()))
                .collect();
        }
        let mut reasons = Vec::new();
        for w in &mut self.wake {
            let value = match w.stat {
                WakeStat::LossMean => mean,
                WakeStat::LossMax => max,
                WakeStat::GradNormTotal => Some(total),
            };
            let Some(v) = value else { continue };
            let hit = w.above.is_some_and(|x| v > x) || w.below.is_some_and(|x| v < x);
            if hit && !w.firing {
                let bound = match (w.above, w.below) {
                    (Some(x), _) if v > x => format!("above {x}"),
                    (_, Some(x)) => format!("below {x}"),
                    _ => String::new(),
                };
                reasons.push(format!(
                    "your analysis: {} = {} is {bound} (update {u})",
                    w.stat.name(),
                    round(v)
                ));
            }
            w.firing = hit;
        }
        (r, reasons)
    }

    /// The prompt section on `analysis.json`.
    pub fn interface() -> String {
        format!(
            "Your analysis is {ANALYSIS}, in this folder; it only reads and never changes an update:\n\
             {{\"every\": <updates between reports>, \"stats\": [\"grad_norm\", \"weight_norm\", \
             \"update_ratio\", \"grad_cosine\"], \"params\": [\"<glob>\", …], \
             \"wake\": [{{\"stat\": \"loss_mean\" | \"loss_max\" | \"grad_norm_total\", \"above\": <x>, \
             \"below\": <x>}}, …]}}\n\
             - Every report has the update, the training loss since the previous report (mean, min, max, \
             non-finite count), the total gradient norm and your program's time per update. `stats` adds \
             per-param tables, for the params matching `params` (the {MAX_SHOWN} largest gradients when more \
             match): grad_norm, weight_norm, update_ratio (|change of this update| / |weights before it|) \
             and grad_cosine (cosine between this report's gradient and the previous report's).\n\
             - A `wake` rule asks for a decision as soon as its statistic crosses its bound, once each time \
             it does; wake-ups count toward the decision limit.\n\
             - \"activations\": [\"<glob>\", …] summarises the model's further outputs whose names match \
             (mean, std, rms, max_abs, non-finite count), when the training script's graph has any.\n\
             - Changing only {ANALYSIS} replaces the analysis without rebuilding your updater."
        )
    }
}

impl Wake {
    fn parse(i: usize, v: &Value) -> ApiResult<Self> {
        let what = format!("{ANALYSIS} wake rule {i}");
        let o = Opts::new(&what, v)?;
        let stat = match o.str("stat")? {
            Some("loss_mean") => WakeStat::LossMean,
            Some("loss_max") => WakeStat::LossMax,
            Some("grad_norm_total") => WakeStat::GradNormTotal,
            other => {
                return Err(ApiError::type_(format!(
                    "{what}: stat is loss_mean, loss_max or grad_norm_total, got {other:?}"
                )));
            }
        };
        let (above, below) = (o.f64("above")?, o.f64("below")?);
        o.finish()?;
        if above.is_none() && below.is_none() {
            return Err(ApiError::type_(format!("{what}: give `above` or `below`")));
        }
        Ok(Self {
            stat,
            above,
            below,
            firing: false,
        })
    }
}

fn strings(v: Option<&Value>, what: &str) -> ApiResult<Vec<String>> {
    match v {
        None => Ok(Vec::new()),
        Some(Value::Array(a)) => a
            .iter()
            .map(|s| {
                s.as_str().map(str::to_string).ok_or_else(|| {
                    ApiError::type_(format!("{ANALYSIS}: `{what}` holds strings, got {s}"))
                })
            })
            .collect(),
        Some(other) => Err(ApiError::type_(format!(
            "{ANALYSIS}: `{what}` must be a list, got {other}"
        ))),
    }
}

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt()
}

fn round(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items<'a>(p: &'a mut [f32], g: &'a [f32]) -> [OptItem<'a>; 1] {
        [OptItem {
            name: "w",
            shape: &[2],
            param: p,
            grad: g,
        }]
    }

    #[test]
    fn reports_hold_the_chosen_stats_and_wake_once_per_crossing() {
        let mut a = Analysis::parse(
            r#"{"every": 10, "stats": ["grad_norm", "update_ratio", "grad_cosine"],
                "wake": [{"stat": "loss_mean", "above": 2.0}]}"#,
        )
        .unwrap();
        let (mut p, g) = ([3.0f32, 4.0], [1.0f32, 0.0]);
        let before = vec![vec![3.0f32, 4.5]];
        let (r, why) = a.report(
            10,
            &[1.0, 3.0, f32::NAN],
            &items(&mut p, &g),
            Some(&before),
            &[("h".to_string(), vec![1.0, -3.0, f32::NAN])],
            None,
        );
        assert!(
            r.get("activations").is_none(),
            "no activation globs: none captured"
        );
        assert_eq!(r["loss_mean"], 2.0);
        assert_eq!(r["nonfinite_losses"], 1);
        assert_eq!(r["grad_norm"]["w"], 1.0);
        assert!((r["update_ratio"]["w"].as_f64().unwrap() - 0.5 / 4.5f64.hypot(3.0)).abs() < 1e-6);
        assert!(
            r["grad_cosine"].as_object().unwrap().is_empty(),
            "no previous gradient yet"
        );
        assert!(r.get("weight_norm").is_none());
        assert!(why.is_empty(), "2.0 is not above 2.0");

        let g2 = [0.0f32, 1.0];
        let (r, why) = a.report(20, &[3.0], &items(&mut p, &g2), None, &[], None);
        assert_eq!(r["grad_cosine"]["w"], 0.0);
        assert_eq!(
            why,
            vec!["your analysis: loss_mean = 3 is above 2 (update 20)".to_string()]
        );
        let (_, why) = a.report(30, &[4.0], &items(&mut p, &g2), None, &[], None);
        assert!(why.is_empty(), "still above: no second wake");
        let (_, _) = a.report(40, &[1.0], &items(&mut p, &g2), None, &[], None);
        let (_, why) = a.report(50, &[5.0], &items(&mut p, &g2), None, &[], None);
        assert_eq!(why.len(), 1, "crossed again");
    }

    #[test]
    fn bad_analyses_say_what_is_wrong() {
        let err = |t: &str| Analysis::parse(t).unwrap_err().message;
        assert!(err(r#"{"stats": ["hessian"]}"#).contains("unknown stat \"hessian\""));
        assert!(err(r#"{"wake": [{"stat": "loss_mean"}]}"#).contains("`above` or `below`"));
        assert!(err(r#"{"every": 0}"#).contains("at least 1"));
        assert!(err(r#"{"evry": 5}"#).contains("'evry'"));
        assert!(Analysis::parse(&Analysis::starting_text(25)).unwrap().every == 25);
    }
}
