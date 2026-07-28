// RLX — versatile ML compiler + runtime. MIT OR Apache-2.0.
//! Shard-aware checkpoint download — the streamlined replacement for hand-rolled
//! `curl`/`wget` loops. Each node fetches only the safetensors shards its
//! pipeline stage's layers touch (+ optionally embeddings / LM head), verified.
//!
//! Per-node download (run on each machine):
//!   cargo run -p rlx-hub --example shard_download -- \
//!     --repo mlx-community/DeepSeek-V4-Flash-2bit-DQ --dest ~/ckpt --layers 18:35
//!   # first stage adds --embed, last stage adds --head
//!
//! Just print the multi-node plan (no download):
//!   cargo run -p rlx-hub --example shard_download -- \
//!     --repo mlx-community/DeepSeek-V4-Flash-2bit-DQ --plan 0:18,18:35,35:43

use anyhow::{Context, Result, bail};
use rlx_hub::{HfRepo, download_files, fetch_index, fetch_sizes, plan_layer_stages};
use std::collections::HashMap;
use std::ops::Range;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}
fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}
fn parse_range(s: &str) -> Result<Range<usize>> {
    let (a, b) = s.split_once(':').with_context(|| format!("range `{s}` must be A:B"))?;
    Ok(a.trim().parse()?..b.trim().parse()?)
}
fn gb(b: u64) -> f64 {
    b as f64 / 1e9
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let repo_id = flag(&args, "--repo").context("--repo <hf-id> required")?;
    let repo = HfRepo::new(repo_id);
    let index = fetch_index(&repo).context("fetch index.json")?;
    let sizes = fetch_sizes(&repo).unwrap_or_default();
    let sz = |files: &[String]| -> u64 { files.iter().filter_map(|f| sizes.get(f)).sum() };

    // ── Plan mode: print the whole multi-node split, no download ──
    if let Some(plan) = flag(&args, "--plan") {
        let ranges: Vec<Range<usize>> = plan.split(',').map(parse_range).collect::<Result<_>>()?;
        // embed → first stage, lm_head/norm → last stage (typical pipeline).
        let last = ranges.len() - 1;
        let extra: Vec<Vec<&str>> = (0..ranges.len())
            .map(|i| {
                let mut e = vec![];
                if i == 0 { e.push("model.embed_tokens"); }
                if i == last { e.extend(["lm_head", "model.norm"]); }
                e
            })
            .collect();
        let stages = plan_layer_stages(&index, &ranges, &extra);
        let mut total = 0u64;
        let mut overlap: HashMap<String, usize> = HashMap::new();
        println!("{:<6}{:<12}{:>8}{:>8}  shards", "stage", "layers", "files", "GB");
        for s in &stages {
            let b = sz(&s.shards);
            total += b;
            for f in &s.shards {
                *overlap.entry(f.clone()).or_default() += 1;
            }
            let nums: Vec<String> = s.shards.iter().map(|f| shard_num(f)).collect();
            println!(
                "{:<6}{:<12}{:>8}{:>8.1}  [{}]",
                s.stage,
                format!("{}..{}", s.layers.start, s.layers.end),
                s.shards.len(),
                gb(b),
                nums.join(",")
            );
        }
        let mut dup: Vec<String> = overlap.iter().filter(|(_, c)| **c > 1).map(|(f, _)| shard_num(f)).collect();
        dup.sort_by_key(|s| s.parse::<u32>().unwrap_or(0));
        println!(
            "total download {:.1} GB (unique {:.1} GB; boundary shards on 2 nodes: [{}])",
            gb(total),
            gb(sizes.values().sum::<u64>().min(sz(&index.shards()))),
            dup.join(",")
        );
        return Ok(());
    }

    // ── Download mode: this node fetches its stage's shards ──
    let range = parse_range(&flag(&args, "--layers").context("--layers A:B (this node's stage) or --plan required")?)?;
    let dest = flag(&args, "--dest").unwrap_or_else(|| ".".into());
    let dest = std::path::Path::new(&dest);
    let mut extra = vec![];
    if has(&args, "--embed") { extra.push("model.embed_tokens"); }
    if has(&args, "--head") { extra.extend(["lm_head", "model.norm"]); }
    let stage = plan_layer_stages(&index, &[range.clone()], &[extra])
        .into_iter()
        .next()
        .unwrap();

    // Small files every node needs to load: config + index (+ tokenizer for the coordinator).
    let mut files = vec![
        "config.json".to_string(),
        "model.safetensors.index.json".to_string(),
    ];
    if has(&args, "--tokenizer") {
        files.extend(["tokenizer.json".to_string(), "tokenizer_config.json".to_string()]);
    }
    files.extend(stage.shards.iter().cloned());

    println!(
        "node stage layers {}..{}: {} shards, {:.1} GB → {}",
        range.start, range.end, stage.shards.len(), gb(sz(&stage.shards)), dest.display()
    );
    let report = download_files(&repo, &files, dest, &sizes, |f, note| {
        // Terse per-file line; the boundary/config files are tiny.
        if note != "cached" {
            println!("  [{note}] {f}");
        }
    });
    println!(
        "done: {} ok, {} failed, {:.1} GB",
        report.ok.len(),
        report.failed.len(),
        gb(report.bytes)
    );
    if !report.failed.is_empty() {
        for (f, e) in &report.failed {
            eprintln!("  FAILED {f}: {e}");
        }
        bail!("{} file(s) failed verification/download", report.failed.len());
    }
    Ok(())
}

/// `model-00008-of-00019.safetensors` → `8`.
fn shard_num(f: &str) -> String {
    f.split("-of-")
        .next()
        .and_then(|s| s.rsplit('-').next())
        .map(|s| s.trim_start_matches('0').to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| f.to_string())
}
