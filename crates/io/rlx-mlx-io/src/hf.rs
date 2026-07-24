// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! HuggingFace hub fetch for mlx-community weight dirs.

use anyhow::{Context, Result, bail};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default small 4-bit mlx-community checkpoint for optional e2e.
pub const DEFAULT_HF_MLX_REPO: &str = "mlx-community/SmolLM2-135M-Instruct-4bit";

/// Resolve cache dir: `$RLX_HF_CACHE` or `$HOME/.cache/rlx/hf`.
pub fn hf_cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("RLX_HF_CACHE") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".cache/rlx/hf")
}

fn hub_base(repo: &str) -> String {
    let rev = std::env::var("RLX_HF_REVISION").unwrap_or_else(|_| "main".into());
    format!("https://huggingface.co/{repo}/resolve/{rev}")
}

fn download_file(url: &str, dest: &Path) -> Result<()> {
    if dest.is_file() && dest.metadata()?.len() > 0 {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = dest.with_extension("partial");
    // Prefer curl (common on macOS/Linux); fall back to ureq if linked later.
    let status = Command::new("curl")
        .args([
            "-fsSL",
            "--retry",
            "3",
            "-L",
            "-o",
            tmp.to_str().context("dest utf8")?,
            url,
        ])
        .status()
        .with_context(|| format!("curl {url}"))?;
    if !status.success() {
        let _ = fs::remove_file(&tmp);
        bail!("curl failed for {url} (status {status})");
    }
    fs::rename(&tmp, dest).with_context(|| format!("rename {}", dest.display()))?;
    Ok(())
}

/// Fetch `config.json` + primary safetensors (or index shards) into `dest_dir`.
///
/// Skips files that already exist. Set `RLX_HF_MLX_REPO` to override the repo id.
pub fn fetch_mlx_community(repo: &str, dest_dir: impl AsRef<Path>) -> Result<PathBuf> {
    let dest_dir = dest_dir.as_ref();
    fs::create_dir_all(dest_dir)?;
    let base = hub_base(repo);

    download_file(
        &format!("{base}/config.json"),
        &dest_dir.join("config.json"),
    )?;

    // Prefer single-file model.safetensors; else follow index.
    let single = dest_dir.join("model.safetensors");
    let single_url = format!("{base}/model.safetensors");
    match download_file(&single_url, &single) {
        Ok(()) => return Ok(dest_dir.to_path_buf()),
        Err(_) => {
            let _ = fs::remove_file(&single);
        }
    }

    let index_path = dest_dir.join("model.safetensors.index.json");
    download_file(&format!("{base}/model.safetensors.index.json"), &index_path)?;
    let index: serde_json::Value = serde_json::from_slice(&fs::read(&index_path)?)?;
    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .context("weight_map missing")?;
    let mut files: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    files.sort();
    files.dedup();
    for f in files {
        download_file(&format!("{base}/{f}"), &dest_dir.join(&f))?;
    }
    Ok(dest_dir.to_path_buf())
}

/// Fetch [`DEFAULT_HF_MLX_REPO`] (or `$RLX_HF_MLX_REPO`) into the cache.
pub fn fetch_default_mlx_community() -> Result<PathBuf> {
    let repo = std::env::var("RLX_HF_MLX_REPO").unwrap_or_else(|_| DEFAULT_HF_MLX_REPO.into());
    let dest = hf_cache_dir().join(repo.replace('/', "--"));
    fetch_mlx_community(&repo, &dest)
}

/// Write a tiny marker so offline tests can detect a prior successful fetch.
pub fn write_fetch_ok(dir: &Path) -> Result<()> {
    let mut f = fs::File::create(dir.join(".rlx_fetch_ok"))?;
    writeln!(f, "ok")?;
    Ok(())
}

pub fn fetch_ok(dir: &Path) -> bool {
    dir.join(".rlx_fetch_ok").is_file() || dir.join("config.json").is_file()
}
