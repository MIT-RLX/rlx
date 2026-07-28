// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Resumable HuggingFace file download + integrity verification. Uses `curl`
//! (ubiquitous on macOS/Linux, robust resume via `-C -`); verification is pure
//! Rust — exact byte size (from the HF API) plus, for `.safetensors`, a
//! structural check that the header parses and the declared data length matches
//! the file (catches truncated / interrupted downloads without a full re-read).

use crate::index::SafetensorsIndex;
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A HuggingFace repo at a revision.
#[derive(Debug, Clone)]
pub struct HfRepo {
    pub id: String,
    pub revision: String,
}

impl HfRepo {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            revision: "main".to_string(),
        }
    }
    pub fn at(id: impl Into<String>, revision: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            revision: revision.into(),
        }
    }
    /// The `resolve` URL for a file (LFS-redirected to the CDN by HF).
    pub fn file_url(&self, file: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.id, self.revision, file
        )
    }
    pub fn api_url(&self) -> String {
        format!("https://huggingface.co/api/models/{}?blobs=true", self.id)
    }
}

/// `curl -fsSL <url>` → bytes (for small metadata files).
pub fn curl_bytes(url: &str) -> Result<Vec<u8>> {
    let out = Command::new("curl")
        .args(["-fsSL", "--retry", "3", url])
        .output()
        .with_context(|| format!("spawn curl for {url}"))?;
    if !out.status.success() {
        bail!(
            "curl {url} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(out.stdout)
}

/// Fetch + parse `model.safetensors.index.json`.
pub fn fetch_index(repo: &HfRepo) -> Result<SafetensorsIndex> {
    SafetensorsIndex::parse(&curl_bytes(&repo.file_url("model.safetensors.index.json"))?)
}

/// Fetch `{filename → byte size}` from the HF API (for verification + planning).
pub fn fetch_sizes(repo: &HfRepo) -> Result<HashMap<String, u64>> {
    let v: serde_json::Value = serde_json::from_slice(&curl_bytes(&repo.api_url())?)?;
    let mut m = HashMap::new();
    for s in v
        .get("siblings")
        .and_then(|x| x.as_array())
        .into_iter()
        .flatten()
    {
        if let (Some(n), Some(sz)) = (
            s.get("rfilename").and_then(|x| x.as_str()),
            s.get("size").and_then(|x| x.as_u64()),
        ) {
            m.insert(n.to_string(), sz);
        }
    }
    Ok(m)
}

/// Verify a downloaded file: exact size (when known) + `.safetensors` structural
/// integrity (header parses and data length matches the file).
pub fn verify_file(path: &Path, expected_size: Option<u64>) -> Result<()> {
    let len = fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    if let Some(sz) = expected_size {
        if sz != 0 && len != sz {
            bail!(
                "{}: size {len} != expected {sz} (incomplete)",
                path.display()
            );
        }
    }
    if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        verify_safetensors_structure(path, len)
            .with_context(|| format!("safetensors integrity {}", path.display()))?;
    }
    Ok(())
}

/// A `.safetensors` file is `u64 header_len | JSON header | tensor data`. The
/// max `data_offsets[1]` across tensors must equal `file_len - (8 + header_len)`.
fn verify_safetensors_structure(path: &Path, file_len: u64) -> Result<()> {
    if file_len < 8 {
        bail!("file too small ({file_len} B)");
    }
    let mut f = File::open(path)?;
    let mut lenb = [0u8; 8];
    f.read_exact(&mut lenb)?;
    let hlen = u64::from_le_bytes(lenb);
    if 8 + hlen > file_len {
        bail!("header length {hlen} exceeds file (truncated)");
    }
    let mut hdr = vec![0u8; hlen as usize];
    f.read_exact(&mut hdr)?;
    let v: serde_json::Value = serde_json::from_slice(&hdr).context("parse header json")?;
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("header not an object"))?;
    let mut max_end = 0u64;
    for (k, t) in obj {
        if k == "__metadata__" {
            continue;
        }
        if let Some(end) = t
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .and_then(|o| o.get(1))
            .and_then(|x| x.as_u64())
        {
            max_end = max_end.max(end);
        }
    }
    let expected = 8 + hlen + max_end;
    if expected != file_len {
        bail!("data ends at {expected} B but file is {file_len} B (truncated/corrupt)");
    }
    Ok(())
}

/// Download one file into `dest_dir` (resumable via `curl -C -`). Returns the
/// local path. Idempotent — a complete file resumes to "nothing to do".
pub fn download_file(repo: &HfRepo, file: &str, dest_dir: &Path) -> Result<PathBuf> {
    let dest = dest_dir.join(file);
    if let Some(p) = dest.parent() {
        fs::create_dir_all(p)?;
    }
    let url = repo.file_url(file);
    let status = Command::new("curl")
        .args([
            "-fSL",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "-C",
            "-",
            "-o",
        ])
        .arg(&dest)
        .arg(&url)
        .status()
        .with_context(|| format!("spawn curl for {file}"))?;
    if !status.success() {
        bail!("curl failed for {file} (exit {status})");
    }
    Ok(dest)
}

/// Outcome of a batch download.
#[derive(Debug, Default)]
pub struct DownloadReport {
    pub ok: Vec<String>,
    pub failed: Vec<(String, String)>,
    pub bytes: u64,
}

/// Download + verify a set of files into `dest_dir`. A file that already exists
/// and verifies is skipped. `sizes` (from [`fetch_sizes`]) drives size checks
/// (pass an empty map to rely on structural checks only). `on_event(file, note)`
/// reports progress.
pub fn download_files(
    repo: &HfRepo,
    files: &[String],
    dest_dir: &Path,
    sizes: &HashMap<String, u64>,
    mut on_event: impl FnMut(&str, &str),
) -> DownloadReport {
    fs::create_dir_all(dest_dir).ok();
    let mut report = DownloadReport::default();
    for file in files {
        let expected = sizes.get(file).copied();
        let dest = dest_dir.join(file);
        // Fast path: already present + verifies.
        if dest.is_file() && verify_file(&dest, expected).is_ok() {
            on_event(file, "cached");
            report.bytes +=
                expected.unwrap_or_else(|| fs::metadata(&dest).map(|m| m.len()).unwrap_or(0));
            report.ok.push(file.clone());
            continue;
        }
        on_event(file, "downloading");
        let res = download_file(repo, file, dest_dir).and_then(|p| {
            verify_file(&p, expected)?;
            Ok(p)
        });
        match res {
            Ok(p) => {
                on_event(file, "verified");
                report.bytes += fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                report.ok.push(file.clone());
            }
            Err(e) => {
                on_event(file, "FAILED");
                report.failed.push((file.clone(), e.to_string()));
            }
        }
    }
    report
}
