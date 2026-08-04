// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-actions` — **data-driven transform recommender.** For each op/weight
//! it picks the *cheapest ACTION whose measured error stays under a precision
//! budget* — where the menu goes beyond quant to aggressive replacements that
//! only fire where the flowing data proves them safe:
//!
//! - **ternary** `α·{-1,0,+1}` — ~1.6-bit, adds-only matmul (no multiplies);
//! - **low-rank factor** `W≈U·V` — fewer FLOPs + bytes;
//! - **int8 / int4** — the safe compression tier;
//! - **skip layer** — drop a block whose residual delta is ~0 (layer minimization).
//!
//! It's fusion's cousin: fusion collapses ops structurally; this *replaces* an op
//! with a cheaper equivalent (or removes it) based on what its data actually does
//! — saving space, time, and bandwidth while holding precision.
//!
//!   opscope-actions              # synthetic diverse model demo
//!   opscope-actions <dir>        # recommend actions for every *.tensor in <dir>

use rlx_opscope::layers::{Decomp, LayerProfile, profile_layer};
use rlx_opscope::probe::{block_identity_gap, load_tensor};
use rlx_opscope::{Dist, sample};

const BUDGET: f32 = 0.02; // keep precision: ≤2% per-op reconstruction error

fn human(b: usize) -> String {
    if b >= 1 << 20 {
        format!("{:.1}MB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1}KB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{b}B")
    }
}

/// Is this action an aggressive *replacement* (vs plain quant)?
fn is_replace(d: &Decomp) -> bool {
    matches!(
        d,
        Decomp::Ternary
            | Decomp::LowRank(_)
            | Decomp::Monarch
            | Decomp::Tucker(..)
            | Decomp::Tt(..)
            | Decomp::Sparse24
    )
}

/// A hand-built genuinely-ternary weight (mostly 0, some ±α) so the ternary
/// action fires — most trained weights are NOT ternary (the honest common case).
fn ternary_weight(k: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..k * n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            match s % 5 {
                0 => 0.8,
                1 => -0.8,
                _ => 0.0,
            }
        })
        .collect()
}

fn recommend(profiles: &[LayerProfile]) {
    println!(
        "── per-op transform action (cheapest whose error ≤ {:.0}%) ──\n",
        BUDGET * 100.0
    );
    println!(
        "  {:<16} {:<18} {:>7}  {:>8} {:>6}  kind",
        "layer", "action", "err", "bytes", "save"
    );
    let (mut dense_total, mut chosen_total) = (0usize, 0usize);
    for p in profiles {
        let pick = p.best_within(BUDGET);
        dense_total += p.dense_bytes;
        chosen_total += pick.bytes;
        let save = p.dense_bytes as f32 / pick.bytes.max(1) as f32;
        let kind = if matches!(pick.decomp, Decomp::Dense) {
            "— keep (no safe win)"
        } else if pick.decomp == Decomp::Ternary {
            "REPLACE ⇒ ternary (adds-only)"
        } else if is_replace(&pick.decomp) {
            "REPLACE (structural)"
        } else {
            "quantize"
        };
        println!(
            "  {:<16} {:<18} {:>7.4}  {:>8} {:>5.1}×  {kind}",
            p.name,
            pick.decomp.label(),
            pick.rel_err,
            human(pick.bytes),
            save
        );
    }
    println!(
        "\n  ⇒ model {} → {}  ({:.2}× smaller / less bandwidth), every op within {:.0}% error.",
        human(dense_total),
        human(chosen_total),
        dense_total as f32 / chosen_total.max(1) as f32,
        BUDGET * 100.0
    );
    // Which ops the *data* said could be replaced (not just quantized).
    let ternary: Vec<&str> = profiles
        .iter()
        .filter(|p| p.best_within(BUDGET).decomp == Decomp::Ternary)
        .map(|p| p.name.as_str())
        .collect();
    let factored: Vec<&str> = profiles
        .iter()
        .filter(|p| matches!(p.best_within(BUDGET).decomp, Decomp::LowRank(_)))
        .map(|p| p.name.as_str())
        .collect();
    if !ternary.is_empty() {
        println!(
            "  ternary-viable (adds-only, ~1.6-bit): {}",
            ternary.join(", ")
        );
    }
    if !factored.is_empty() {
        println!(
            "  low-rank-replaceable (fewer FLOPs+bytes): {}",
            factored.join(", ")
        );
    }
}

/// Layer minimization from the DATA FLOW: a block whose output ≈ its input adds
/// ~nothing to the residual stream → skip it. (In a real run these are the tapped
/// residual-stream tensors before/after each block; here we synthesize two.)
fn layer_skip_demo() {
    println!("\n── layer minimization (skip near-identity blocks; from residual deltas) ──");
    let n = 4096usize;
    let resin = sample(Dist::Gaussian, 1, n, 40);
    // near-identity block: output = input + tiny perturbation (a redundant layer).
    let pert = sample(Dist::Gaussian, 1, n, 41);
    let near_id: Vec<f32> = resin
        .iter()
        .zip(&pert)
        .map(|(a, b)| a + 0.008 * b)
        .collect();
    // real block: output is a genuinely different transform.
    let real: Vec<f32> = sample(Dist::Gaussian, 1, n, 42);
    for (name, out) in [
        ("block.near_identity", &near_id),
        ("block.real_transform", &real),
    ] {
        let gap = block_identity_gap(&resin, out);
        let verdict = if gap < BUDGET {
            "SKIP — contributes <2%, drop the whole layer"
        } else {
            "keep — real transform"
        };
        println!("  {name:<22} residual Δ {gap:>6.3}   → {verdict}");
    }
    println!(
        "  (skipping a block saves its ENTIRE weight stream + compute — the biggest single win when the"
    );
    println!(
        "   flow data says a layer is redundant. Tap the real residual stream to find them per model.)"
    );
}

fn synthetic_model() -> Vec<LayerProfile> {
    vec![
        profile_layer("attn.q_proj", &sample(Dist::LowRank, 256, 256, 1), 256, 256),
        profile_layer(
            "attn.o_proj",
            &sample(Dist::Gaussian, 256, 256, 3),
            256,
            256,
        ),
        profile_layer("mlp.gate", &sample(Dist::Quantized, 256, 512, 4), 256, 512),
        profile_layer("mlp.down", &sample(Dist::LowRank, 512, 256, 6), 512, 256),
        // a genuinely ternary weight — the replace-with-ternary case:
        profile_layer("router.gate", &ternary_weight(256, 256, 7), 256, 256),
    ]
}

fn load_dir(dir: &str) -> Vec<LayerProfile> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "tensor").unwrap_or(false))
        .collect();
    entries.sort();
    entries
        .iter()
        .filter_map(|p| {
            load_tensor(p.to_str().unwrap()).ok().map(|(r, c, d)| {
                (
                    p.file_stem().unwrap().to_string_lossy().into_owned(),
                    r,
                    c,
                    d,
                )
            })
        })
        .filter(|(_, r, c, _)| *r >= 2 && *c >= 2)
        .map(|(name, r, c, d)| profile_layer(&name, &d, r, c))
        .collect()
}

fn main() {
    let profiles = match std::env::args().nth(1) {
        Some(dir) if std::path::Path::new(&dir).is_dir() => {
            println!("opscope-actions — real weights from {dir}\n");
            load_dir(&dir)
        }
        _ => {
            println!("opscope-actions — synthetic diverse model\n");
            synthetic_model()
        }
    };
    recommend(&profiles);
    layer_skip_demo();
}
