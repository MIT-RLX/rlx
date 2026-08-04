// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-layers` — **per-layer decomposition mining** + the **DADO** allocator.
//!
//! For every layer weight it probes each decomposition kernel (low-rank /
//! Monarch / Tucker / TT / int8 / int4 / 2:4) and measures the real weight
//! reconstruction error and byte cost, then allocates one decomposition per
//! layer under a global memory budget with DADO — the decomposition-aware
//! distributional optimizer — and cross-checks against a greedy Lagrangian
//! sweep.
//!
//! Usage:
//!   opscope-layers                 # synthetic diverse-structure model demo
//!   opscope-layers <dir>           # mine every *.tensor in <dir> (real weights)
//!   opscope-layers <dir> 0.35      # ...at a 35%-of-dense byte budget
//!
//! Feed it real weights with `probe::save_tensor` dumps — e.g. the qwen example
//! `opscope_qwen.rs --dump-tensors <dir>` writes one file per weight.

use rlx_opscope::layers::{
    Allocation, DadoConfig, LayerProfile, QuantStat, dado_allocate, greedy_allocate, profile_layer,
    quant_stat,
};
use rlx_opscope::probe::load_tensor;
use rlx_opscope::{Dist, sample};

fn human_bytes(b: usize) -> String {
    if b >= 1 << 30 {
        format!("{:.2}GB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1}MB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1}KB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{b}B")
    }
}

/// Print the per-layer probe table: every option's error + compression.
fn print_profiles(profiles: &[LayerProfile]) {
    println!("── per-layer decomposition probe ──\n");
    for p in profiles {
        println!(
            "{}  [{}×{}]  dense {}",
            p.name,
            p.rows,
            p.cols,
            human_bytes(p.dense_bytes)
        );
        let mut opts = p.options.clone();
        opts.sort_by_key(|o| o.bytes);
        for o in &opts {
            let comp = p.dense_bytes as f32 / o.bytes.max(1) as f32;
            let flag = if o.rel_err <= 0.02 {
                "✓ tight"
            } else if o.rel_err <= 0.08 {
                "~ ok"
            } else {
                ""
            };
            println!(
                "    {:<26} err {:>7.4}   {:>8}  {:>5.1}×  {}",
                o.decomp.label(),
                o.rel_err,
                human_bytes(o.bytes),
                comp,
                flag
            );
        }
        // The cheapest option within a 2% error budget = the layer's own pick.
        let pick = p.best_within(0.02);
        println!(
            "    → within 2% err: {} ({:.1}×)\n",
            pick.decomp.label(),
            p.dense_bytes as f32 / pick.bytes.max(1) as f32
        );
    }
}

fn print_allocation(tag: &str, a: &Allocation, profiles: &[LayerProfile]) {
    println!(
        "{tag}: {} → {}  ({:.2}× smaller)   Σerr {:.4}",
        human_bytes(a.dense_bytes),
        human_bytes(a.total_bytes),
        a.compression(),
        a.total_err
    );
    for (l, &c) in profiles.iter().zip(&a.choice) {
        if l.options[c].decomp != rlx_opscope::layers::Decomp::Dense {
            println!(
                "    {:<16} → {:<26} err {:.4}",
                l.name,
                l.options[c].decomp.label(),
                l.options[c].rel_err
            );
        }
    }
}

fn allocate_report(profiles: &[LayerProfile], budget: f32) {
    println!(
        "\n══ DADO allocation @ {:.0}% byte budget ══",
        budget * 100.0
    );
    let greedy = greedy_allocate(profiles, budget);
    let dado = dado_allocate(profiles, budget, DadoConfig::default());
    print_allocation("greedy (Lagrangian)", &greedy, profiles);
    println!();
    print_allocation("DADO   (distributional)", &dado, profiles);
    // Greedy is bytes-first (stops at the first feasible point); DADO minimizes
    // error *within* the budget, so it may spend leftover headroom for fidelity.
    let verdict = if dado.total_err <= greedy.total_err + 1e-4 {
        "DADO ≤ greedy error within budget (spends leftover headroom on fidelity)."
    } else {
        "greedy edged DADO on error — raise iters/pop."
    };
    println!("\n  {verdict}");
}

/// Synthetic model whose layers have *deliberately different* structure so each
/// decomposition wins somewhere — the allocator must then choose per layer.
fn synthetic_profiles() -> Vec<LayerProfile> {
    vec![
        // near-low-rank projections → factored/Tucker/TT should win
        profile_layer("attn.q_proj", &sample(Dist::LowRank, 256, 256, 1), 256, 256),
        profile_layer("attn.k_proj", &sample(Dist::LowRank, 256, 128, 2), 256, 128),
        // square perfect-square → Monarch is on the menu
        profile_layer(
            "attn.o_proj",
            &sample(Dist::Gaussian, 256, 256, 3),
            256,
            256,
        ),
        // quantizes cleanly (snapped values) → int8/int4
        profile_layer("mlp.gate", &sample(Dist::Quantized, 256, 512, 4), 256, 512),
        // dense gaussian — the hard case, only quant/sparse give modest wins
        profile_layer("mlp.up", &sample(Dist::Gaussian, 256, 512, 5), 256, 512),
        // low-rank down-projection
        profile_layer("mlp.down", &sample(Dist::LowRank, 512, 256, 6), 512, 256),
        // outlier channels → per-channel int shines
        profile_layer("lm_head", &sample(Dist::Outlier, 256, 400, 7), 256, 400),
    ]
}

/// Load every `*.tensor` in `dir` (written by `probe::save_tensor`) as a layer.
fn load_dir(dir: &str) -> Vec<LayerProfile> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "tensor").unwrap_or(false))
        .collect();
    entries.sort();
    let mut profiles = Vec::new();
    for path in entries {
        let (rows, cols, data) = match load_tensor(path.to_str().unwrap()) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skip {}: {e}", path.display());
                continue;
            }
        };
        // Only 2-D matmul weights are decomposable here; skip vectors/scalars.
        if rows < 2 || cols < 2 {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        println!("  probing {name} [{rows}×{cols}] …");
        profiles.push(profile_layer(&name, &data, rows, cols));
    }
    profiles
}

// ─────────────────── quant-sensitivity sweep (fast) ───────────────────

/// Mean of a field over a slice (0 if empty).
fn mean_of(stats: &[&QuantStat], f: impl Fn(&QuantStat) -> f32) -> f32 {
    if stats.is_empty() {
        return 0.0;
    }
    stats.iter().map(|s| f(s)).sum::<f32>() / stats.len() as f32
}

/// A short bar scaled to `max` (12 cells).
fn bar(v: f32, max: f32) -> String {
    let n = if max <= 0.0 {
        0
    } else {
        ((v / max) * 12.0).round() as usize
    };
    "█".repeat(n.min(12))
}

/// Depth-sensitivity + projection-type + mixed-precision report over the fast
/// per-weight quant metrics. `int4_budget` = per-channel rel-err ceiling under
/// which a weight is allowed to drop to int4 in the mixed-precision recipe.
fn quant_report(stats: &[QuantStat], int4_budget: f32) {
    let total_params: usize = stats.iter().map(|s| s.numel).sum();
    println!(
        "── quant-sensitivity sweep: {} weights, {:.1}M params ──\n",
        stats.len(),
        total_params as f64 / 1e6
    );

    // 1) Depth trend — mean per-channel int8/int4 error over each layer's weights.
    let max_layer = stats.iter().filter_map(|s| s.layer).max().unwrap_or(0);
    let max_i4 = stats.iter().map(|s| s.int4_pc).fold(0f32, f32::max);
    println!("depth trend (mean per-channel rel-err over each layer's weights):");
    println!("  layer   int8      int4    int4 bar");
    for l in 0..=max_layer {
        let row: Vec<&QuantStat> = stats.iter().filter(|s| s.layer == Some(l)).collect();
        if row.is_empty() {
            continue;
        }
        let (i8, i4) = (mean_of(&row, |s| s.int8_pc), mean_of(&row, |s| s.int4_pc));
        println!("  {l:>3}    {i8:>7.4}   {i4:>7.4}   {}", bar(i4, max_i4));
    }

    // Depth correlation of int4 error: compare first vs last third of layers.
    let third = (max_layer + 1) / 3;
    let early: Vec<&QuantStat> = stats
        .iter()
        .filter(|s| s.layer.map(|l| l < third).unwrap_or(false))
        .collect();
    let late: Vec<&QuantStat> = stats
        .iter()
        .filter(|s| s.layer.map(|l| l >= max_layer - third).unwrap_or(false))
        .collect();
    println!(
        "\n  early third int4 {:.4}  vs  late third int4 {:.4}  →  {}",
        mean_of(&early, |s| s.int4_pc),
        mean_of(&late, |s| s.int4_pc),
        if mean_of(&late, |s| s.int4_pc) > mean_of(&early, |s| s.int4_pc) * 1.05 {
            "deeper layers quantize WORSE"
        } else if mean_of(&early, |s| s.int4_pc) > mean_of(&late, |s| s.int4_pc) * 1.05 {
            "earlier layers quantize WORSE"
        } else {
            "roughly depth-invariant"
        }
    );

    // 2) By projection type — mass + how int4-friendly (biggest mass first, so
    //    the whole-model view shows where the parameters actually live).
    let mut kinds: Vec<String> = stats.iter().map(|s| s.kind.clone()).collect();
    kinds.sort();
    kinds.dedup();
    let mut by_kind: Vec<(String, f32, f32, f32, usize)> = kinds
        .iter()
        .map(|k| {
            let g: Vec<&QuantStat> = stats.iter().filter(|s| &s.kind == k).collect();
            let mass: usize = g.iter().map(|s| s.numel).sum();
            (
                k.clone(),
                mean_of(&g, |s| s.int8_pc),
                mean_of(&g, |s| s.int4_pc),
                mean_of(&g, |s| s.outlier),
                mass,
            )
        })
        .collect();
    by_kind.sort_by_key(|x| std::cmp::Reverse(x.4)); // biggest mass first
    println!("\nby weight type (biggest mass first):");
    println!(
        "  {:<14} params    %model   int8      int4    outlier",
        "kind"
    );
    for (k, i8, i4, ol, mass) in &by_kind {
        println!(
            "  {k:<14} {:>6.1}M   {:>5.1}%   {i8:>7.4}   {i4:>7.4}   {ol:>5.1}×",
            *mass as f64 / 1e6,
            *mass as f64 / total_params.max(1) as f64 * 100.0
        );
    }

    // 3) Outlier hotspots — top AWQ/SmoothQuant candidates.
    let mut ol: Vec<&QuantStat> = stats.iter().collect();
    ol.sort_by(|a, b| b.outlier.partial_cmp(&a.outlier).unwrap());
    println!("\noutlier hotspots (keep these channels high-precision):");
    for s in ol.iter().take(5) {
        println!("  {:<44} {:>6.1}× max/median channel", s.name, s.outlier);
    }

    // 4) Mixed-precision recipe — int4 where it fits the budget, else int8.
    let (mut i4_bytes, mut i8_bytes, mut all8_bytes, mut all4_bytes) =
        (0usize, 0usize, 0usize, 0usize);
    let (mut n_i4, mut sum_err) = (0usize, 0f32);
    let mut kind_forced_i8: std::collections::BTreeMap<String, usize> = Default::default();
    for s in stats {
        all8_bytes += s.numel; // 1 B/elem
        all4_bytes += s.numel / 2;
        if s.int4_pc <= int4_budget {
            i4_bytes += s.numel / 2;
            n_i4 += 1;
            sum_err += s.int4_pc;
        } else {
            i8_bytes += s.numel;
            sum_err += s.int8_pc;
            *kind_forced_i8.entry(s.kind.clone()).or_default() += 1;
        }
    }
    let mixed = i4_bytes + i8_bytes;
    println!(
        "\nmixed-precision recipe (int4 where per-chan err ≤ {int4_budget}):\n  \
         {}/{} weights → int4;  mixed {:.1}MB  vs  uniform-int8 {:.1}MB  vs  uniform-int4 {:.1}MB ({:.2}× under int8)\n  \
         Σ per-weight err: mixed {:.3}  (uniform-int4 would be {:.3})",
        n_i4,
        stats.len(),
        mixed as f64 / 1e6,
        all8_bytes as f64 / 1e6,
        all4_bytes as f64 / 1e6,
        all8_bytes as f64 / mixed.max(1) as f64,
        sum_err,
        stats.iter().map(|s| s.int4_pc).sum::<f32>(),
    );
    if !kind_forced_i8.is_empty() {
        let forced: Vec<String> = kind_forced_i8
            .iter()
            .map(|(k, n)| format!("{k}(×{n})"))
            .collect();
        println!("  forced to int8 (int4 over budget): {}", forced.join(", "));
    }
}

/// Load every `*.tensor` in `dir` as `(name, rows, cols, data)` — no full probe.
fn load_dir_raw(dir: &str) -> Vec<(String, usize, usize, Vec<f32>)> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "tensor").unwrap_or(false))
        .collect();
    entries.sort();
    let mut out = Vec::new();
    for path in entries {
        if let Ok((rows, cols, data)) = load_tensor(path.to_str().unwrap()) {
            if rows >= 2 && cols >= 2 {
                out.push((
                    path.file_stem().unwrap().to_string_lossy().into_owned(),
                    rows,
                    cols,
                    data,
                ));
            }
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // `opscope-layers <dir> --quant [int4_budget]` → fast quant-sensitivity sweep.
    if args.iter().any(|a| a == "--quant") {
        let dir = args
            .iter()
            .skip(1)
            .find(|a| std::path::Path::new(a).is_dir())
            .cloned()
            .expect("--quant needs a <dir>");
        let budget: f32 = args
            .iter()
            .filter_map(|a| a.parse::<f32>().ok())
            .find(|&b| b < 1.0)
            .unwrap_or(0.1);
        println!("opscope-layers --quant — real weights from {dir}\n");
        let raw = load_dir_raw(&dir);
        let stats: Vec<QuantStat> = raw
            .iter()
            .map(|(n, r, c, d)| quant_stat(n, d, *r, *c, 0.03))
            .collect();
        quant_report(&stats, budget);
        return;
    }

    let (profiles, source) = match args.get(1) {
        Some(dir) if std::path::Path::new(dir).is_dir() => {
            (load_dir(dir), format!("real weights from {dir}"))
        }
        _ => (
            synthetic_profiles(),
            "synthetic diverse-structure model".to_string(),
        ),
    };
    let budget: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.35);

    println!("opscope-layers — {source}\n");
    print_profiles(&profiles);

    let dense: usize = profiles.iter().map(|p| p.dense_bytes).sum();
    println!(
        "── model total: {} across {} layers ──",
        human_bytes(dense),
        profiles.len()
    );
    for b in [0.5f32, budget, 0.2] {
        allocate_report(&profiles, b);
    }
}
