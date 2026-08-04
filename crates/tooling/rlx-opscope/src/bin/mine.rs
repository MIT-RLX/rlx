// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-mine` — read a tidy sketch CSV and report, per (distribution ×
//! tensor role), the exploitable structure the sketches reveal and which
//! specialized-kernel family it points to. Handles both the independent-run
//! sweep and the step-indexed decode sequence (adds a temporal section when
//! more than one `step` is present).
//!
//! Usage: `opscope-mine [in.csv]`  (default: `opscope.csv`)

use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, BufReader};

/// (dist, site, role, stat, run, step) → the sketch vector (idx-ordered).
type Vecs = HashMap<(String, String, String, String, u64, u64), Vec<f32>>;
/// (dist, site, role, run, step) → source-tensor element count.
type Numels = HashMap<(String, String, String, u64, u64), usize>;
/// Grouping key for reported signals: (dist, site, role).
type Key = (String, String, String);

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

fn cv(v: &[f32]) -> f32 {
    let m = mean(v);
    if m.abs() < 1e-12 || v.is_empty() {
        return 0.0;
    }
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32;
    var.sqrt() / m.abs()
}

fn agg(map: &HashMap<Key, Vec<f32>>, key: &Key) -> f32 {
    mean(map.get(key).map(|v| v.as_slice()).unwrap_or(&[]))
}

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "opscope.csv".into());
    let f = std::fs::File::open(&path)?;
    let mut vecs: Vecs = HashMap::new();
    let mut numels: Numels = HashMap::new();
    let mut steps_seen: BTreeSet<u64> = BTreeSet::new();

    for (li, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if li == 0 || line.is_empty() {
            continue; // header
        }
        // run_id,step,backend,dist,M,K,N,numel,site,role,stat,idx,value
        let c: Vec<&str> = line.split(',').collect();
        if c.len() != 13 {
            continue;
        }
        let run: u64 = c[0].parse().unwrap_or(0);
        let step: u64 = c[1].parse().unwrap_or(0);
        let dist = c[3].to_string();
        let numel: usize = c[7].parse().unwrap_or(0);
        let site = c[8].to_string();
        let role = c[9].to_string();
        let stat = c[10].to_string();
        let val: f32 = c[12].parse().unwrap_or(0.0);

        steps_seen.insert(step);
        numels.insert((dist.clone(), site.clone(), role.clone(), run, step), numel);
        vecs.entry((dist, site, role, stat, run, step))
            .or_default()
            .push(val);
    }

    // Per-(dist,site,role) signals: average of each per-(run,step) value.
    let mut density: HashMap<Key, Vec<f32>> = HashMap::new();
    let mut outlier: HashMap<Key, Vec<f32>> = HashMap::new();
    let mut hist_top: HashMap<Key, Vec<f32>> = HashMap::new();
    let mut pos_cv: HashMap<Key, Vec<f32>> = HashMap::new();
    let mut adj_ratio: HashMap<Key, Vec<f32>> = HashMap::new();
    // (dist,site,role) → l1 scalar per step, for cross-step stationarity.
    let mut l1_series: HashMap<Key, Vec<(u64, f32)>> = HashMap::new();
    let mut keys: BTreeSet<Key> = BTreeSet::new();

    for ((dist, site, role, stat, run, step), v) in &vecs {
        let key = (dist.clone(), site.clone(), role.clone());
        keys.insert(key.clone());
        match stat.as_str() {
            "nnz" => {
                if let Some(&ne) =
                    numels.get(&(dist.clone(), site.clone(), role.clone(), *run, *step))
                {
                    density
                        .entry(key)
                        .or_default()
                        .push(v[0] / ne.max(1) as f32);
                }
            }
            "chan_maxabs" => {
                let mx = v.iter().cloned().fold(0.0f32, f32::max);
                let mn = mean(v);
                if mn > 1e-9 {
                    outlier.entry(key).or_default().push(mx / mn);
                }
            }
            "hist" => {
                let sum: f32 = v.iter().sum();
                if sum > 0.0 {
                    let mx = v.iter().cloned().fold(0.0f32, f32::max);
                    hist_top.entry(key).or_default().push(mx / sum);
                }
            }
            "pos_sumsq" => {
                pos_cv.entry(key.clone()).or_default().push(cv(v));
                // Pair with adjacency (same site/run/step) for the coherence ratio.
                if let Some(adj) = vecs.get(&(
                    dist.clone(),
                    site.clone(),
                    role.clone(),
                    "adj_sumsq".into(),
                    *run,
                    *step,
                )) {
                    let pe = mean(v);
                    if pe > 1e-9 {
                        adj_ratio.entry(key).or_default().push(mean(adj) / pe);
                    }
                }
            }
            "l1" => l1_series.entry(key).or_default().push((*step, v[0])),
            _ => {}
        }
    }

    println!(
        "{:<9} {:<10} {:<5} {:>8} {:>8} {:>8} {:>7} {:>7}   suggested exploit",
        "dist", "site", "role", "density", "outlier", "histtop", "pos_cv", "adj"
    );
    println!("{}", "-".repeat(112));

    let mut ranked: Vec<(f32, String)> = Vec::new();
    for key @ (d, s, r) in &keys {
        if !density.contains_key(key) {
            continue;
        }
        let den = agg(&density, key);
        let out = agg(&outlier, key);
        let htp = agg(&hist_top, key);
        let pcv = agg(&pos_cv, key);
        let adj = agg(&adj_ratio, key);

        let (label, score) = if den < 0.5 {
            (
                format!("sparse-GEMM (skip {:.0}% zeros)", (1.0 - den) * 100.0),
                1.0 - den,
            )
        } else if out > 6.0 {
            (
                format!("per-channel int quant (outlier {out:.1}×)"),
                (out / 30.0).min(1.0),
            )
        } else if htp > 0.25 {
            (
                format!("quantize/LUT (spiky, {:.0}% in one bin)", htp * 100.0),
                htp,
            )
        } else if adj_ratio.contains_key(key) && adj < 0.5 {
            (
                format!("delta-compute (adjacent rows cohere, Δ/E={adj:.2})"),
                1.0 - adj,
            )
        } else if pcv > 0.75 {
            (
                format!("sequence/positional structure (cv {pcv:.2})"),
                (pcv / 3.0).min(1.0),
            )
        } else {
            (
                "dense — no cheap sketch exploit (low-rank not sketch-observable; deep-dump)"
                    .into(),
                0.0,
            )
        };

        let s_short: String = s.chars().take(10).collect();
        println!(
            "{:<9} {:<10} {:<5} {:>8.3} {:>8.2} {:>8.3} {:>7.2} {:>7.2}   {label}",
            d, s_short, r, den, out, htp, pcv, adj
        );
        if score > 0.0 {
            ranked.push((score, format!("{d}/{s}/{r}: {label}")));
        }
    }

    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\nRanked exploitable sites (by sketch signal strength):");
    if ranked.is_empty() {
        println!("  (none — all sites look dense/structureless to the current sketches)");
    }
    for (i, (score, what)) in ranked.iter().enumerate() {
        println!("  {:>2}. [{:.2}] {what}", i + 1, score);
    }

    // Temporal section: only when the CSV spans multiple decode steps.
    if steps_seen.len() > 1 {
        println!(
            "\nCross-step temporal analysis ({} steps):",
            steps_seen.len()
        );
        let mut tkeys: Vec<&Key> = l1_series.keys().collect();
        tkeys.sort();
        for key @ (d, s, r) in tkeys {
            let mut series = l1_series[key].clone();
            series.sort_by_key(|&(st, _)| st);
            let vals: Vec<f32> = series.iter().map(|&(_, v)| v).collect();
            let c = cv(&vals);
            let verdict = if c < 1e-4 {
                "STATIONARY across steps → precompute / prepack".to_string()
            } else {
                format!("drifting (l1 cv {c:.4} across steps → temporal coherence)")
            };
            println!("  {d}/{s}/{r:<4}: {verdict}");
        }
    }

    Ok(())
}
