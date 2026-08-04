// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optimization-opportunity miner. Fuses three signals the harness records into
//! one ranked list of *what to actually optimize*:
//!   1. **cost**   — FLOPs of the op (from the `write_site_costs` sidecar);
//!   2. **temporal recurrence** — does the op's input data *repeat over time*?
//!      Each step's input is fingerprinted (moments + histogram); `recur` = the
//!      fraction of steps whose fingerprint matches an earlier one (memoize hit
//!      rate), `drift` = the mean change between consecutive steps (delta-compute
//!      headroom);
//!   3. **decomposability** — matmul/conv are *linear*, so a repeated input can
//!      be memoized and a small-drift input delta-computed (`f(x+Δ)=f(x)+f(Δ)`).
//!
//! Expensive × repeats-over-time × decomposable = the top targets.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};

/// Per-(site,step) input fingerprint built from the recorded sketches.
#[derive(Default, Clone)]
struct Fp {
    mean: f32,
    sumsq: f32,
    l1: f32,
    nnz: f32,
    hist: Vec<f32>,
    /// Element-wise sample (gathered spread values) — the discriminative
    /// fingerprint: distinguishes an exact temporal repeat from mere
    /// distributional stationarity, and measures real per-step element drift.
    elemsig: Vec<f32>,
}

impl Fp {
    /// Fixed-order feature vector. Prefer the element-wise `elemsig` (values);
    /// fall back to moments + histogram (distribution) when it's absent.
    fn vec(&self) -> Vec<f32> {
        if !self.elemsig.is_empty() {
            return self.elemsig.clone();
        }
        let mut v = vec![self.mean, self.sumsq, self.l1, self.nnz];
        v.extend_from_slice(&self.hist);
        v
    }
}

/// Relative L2 distance between two fingerprints (0 = identical).
fn dist(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    let num: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();
    let den: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-9;
    num / den
}

/// A ranked optimization opportunity.
#[derive(Clone, Debug)]
pub struct Opportunity {
    pub site: String,
    pub gflops: f64,
    pub cost_share: f32,
    /// Fraction of steps whose input repeats an earlier step (memoize hit rate).
    pub recur: f32,
    /// Mean consecutive-step input change (delta-compute headroom = 1 - drift).
    pub drift: f32,
    pub recommendation: String,
    /// Cost-weighted expected saving (0..1 of this op's cost × its share).
    pub savings: f32,
}

/// A step counts as a "repeat" if its input matches an earlier step within this.
const MATCH_THRESH: f32 = 1e-3;

/// Mine opportunities from a multi-step sketch CSV + a `site,flops` sidecar.
pub fn mine_opportunities(csv: &str, sidecar: &str) -> std::io::Result<Vec<Opportunity>> {
    // (site, step) → fingerprint of the op's LHS (activation) input.
    let mut fps: HashMap<(String, u64), Fp> = HashMap::new();
    let f = std::fs::File::open(csv)?;
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if i == 0 || line.is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        if c.len() != 13 || c[9] != "lhs" {
            continue; // recurrence of the op = recurrence of its activation input
        }
        let step: u64 = c[1].parse().unwrap_or(0);
        let site = c[8].to_string();
        let idx: usize = c[11].parse().unwrap_or(0);
        let val: f32 = c[12].parse().unwrap_or(0.0);
        let e = fps.entry((site, step)).or_default();
        match c[10] {
            "mean" => e.mean = val,
            "sumsq" => e.sumsq = val,
            "l1" => e.l1 = val,
            "nnz" => e.nnz = val,
            "hist" => {
                if e.hist.len() <= idx {
                    e.hist.resize(idx + 1, 0.0);
                }
                e.hist[idx] = val;
            }
            "elemsig" => {
                if e.elemsig.len() <= idx {
                    e.elemsig.resize(idx + 1, 0.0);
                }
                e.elemsig[idx] = val;
            }
            _ => {}
        }
    }

    // FLOPs per site.
    let mut flops: HashMap<String, u64> = HashMap::new();
    if let Ok(sf) = std::fs::File::open(sidecar) {
        for (i, line) in BufReader::new(sf).lines().enumerate() {
            let line = line?;
            if i == 0 || line.is_empty() {
                continue;
            }
            let c: Vec<&str> = line.split(',').collect();
            if c.len() == 2 {
                flops.insert(c[0].to_string(), c[1].parse().unwrap_or(0));
            }
        }
    }
    let total_flops: u64 = flops.values().sum::<u64>().max(1);

    // Per-site time series of fingerprints.
    let mut by_site: HashMap<String, Vec<(u64, Vec<f32>)>> = HashMap::new();
    for ((site, step), fp) in &fps {
        by_site
            .entry(site.clone())
            .or_default()
            .push((*step, fp.vec()));
    }

    let mut out = Vec::new();
    for (site, mut series) in by_site {
        series.sort_by_key(|&(s, _)| s);
        if series.len() < 2 {
            continue;
        }
        // recur: fraction of steps matching an earlier step.
        let mut repeats = 0usize;
        for i in 1..series.len() {
            let hit = (0..i).any(|j| dist(&series[i].1, &series[j].1) < MATCH_THRESH);
            if hit {
                repeats += 1;
            }
        }
        let recur = repeats as f32 / (series.len() - 1) as f32;
        // drift: mean consecutive distance.
        let drift = (1..series.len())
            .map(|i| dist(&series[i].1, &series[i - 1].1))
            .sum::<f32>()
            / (series.len() - 1) as f32;

        let fl = *flops.get(&site).unwrap_or(&0);
        let cost_share = fl as f32 / total_flops as f32;
        let delta_headroom = (1.0 - drift).clamp(0.0, 1.0);
        let (rec, best) = if recur > 0.2 {
            (
                format!("memoize — cache the {:.0}% repeated calls", recur * 100.0),
                recur,
            )
        } else if drift < 0.3 {
            (
                format!("delta-compute — linear op, tiny drift {drift:.2}"),
                delta_headroom,
            )
        } else {
            ("(input varies — no temporal reuse)".into(), 0.0)
        };
        out.push(Opportunity {
            site,
            gflops: fl as f64 / 1e9,
            cost_share,
            recur,
            drift,
            recommendation: rec,
            savings: cost_share * best,
        });
    }

    out.sort_by(|a, b| b.savings.partial_cmp(&a.savings).unwrap());
    Ok(out)
}

/// Pretty-print the ranked opportunities.
pub fn report(opps: &[Opportunity]) {
    println!(
        "{:<14} {:>8} {:>7} {:>6} {:>6}   {:<8}  recommendation",
        "site", "GFLOP", "%cost", "recur", "drift", "save"
    );
    println!("{}", "-".repeat(100));
    for o in opps.iter().take(14) {
        println!(
            "{:<14} {:>8.3} {:>6.1}% {:>6.0}% {:>6.2}   {:>6.1}%   {}",
            o.site.chars().take(14).collect::<String>(),
            o.gflops,
            o.cost_share * 100.0,
            o.recur * 100.0,
            o.drift,
            o.savings * 100.0,
            o.recommendation
        );
    }
    let total: f32 = opps.iter().map(|o| o.savings).sum();
    println!(
        "\nAggregate recoverable compute (Σ cost-share × reuse): {:.1}%",
        total * 100.0
    );
}
