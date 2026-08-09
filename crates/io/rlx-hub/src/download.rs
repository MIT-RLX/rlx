// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Resumable HuggingFace file download + integrity verification. Uses `curl`
//! (ubiquitous on macOS/Linux, robust resume via `-C -`); verification is pure
//! Rust — exact byte size (from the HF API), for `.safetensors` a structural
//! check that the header parses and the declared data length matches the file,
//! and (when the API exposes it) a full content SHA-256. The first two catch
//! truncated / interrupted downloads without a re-read; the digest catches
//! silent corruption or a tampered mirror.

use crate::error::{HubError, Result};
use crate::index::SafetensorsIndex;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::Write as _;
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

/// Map a `curl` spawn failure — a missing binary is a distinct, actionable error.
fn spawn_err(e: std::io::Error) -> HubError {
    if e.kind() == std::io::ErrorKind::NotFound {
        HubError::MissingTool
    } else {
        HubError::Io(e)
    }
}

/// `curl -fsSL <url>` → bytes (for small metadata files).
pub fn curl_bytes(url: &str) -> Result<Vec<u8>> {
    let out = Command::new("curl")
        .args(["-fsSL", "--retry", "3", url])
        .output()
        .map_err(spawn_err)?;
    if !out.status.success() {
        return Err(HubError::Curl {
            url: url.to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        });
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

/// Fetch `{filename → content SHA-256 (hex)}` from the HF API (`?blobs=true`).
/// Only LFS files (`.safetensors`, `.gguf`, …) carry a content digest under
/// `siblings[].lfs.sha256` (some responses mirror it in `lfs.oid`). Small
/// non-LFS files expose only a git blob `oid`, which is a *sha1* of the git
/// object — **not** a content sha256 — so they're skipped (no `lfs`).
pub fn fetch_sha256s(repo: &HfRepo) -> Result<HashMap<String, String>> {
    let v: serde_json::Value = serde_json::from_slice(&curl_bytes(&repo.api_url())?)?;
    let mut m = HashMap::new();
    for s in v
        .get("siblings")
        .and_then(|x| x.as_array())
        .into_iter()
        .flatten()
    {
        let (Some(name), Some(lfs)) = (s.get("rfilename").and_then(|x| x.as_str()), s.get("lfs"))
        else {
            continue; // no `lfs` block ⇒ not an LFS file ⇒ no content sha256.
        };
        // Prefer `lfs.sha256`; fall back to `lfs.oid` only when it *is* a sha256.
        if let Some(sha) = lfs.get("sha256").and_then(|x| x.as_str()).or_else(|| {
            lfs.get("oid")
                .and_then(|x| x.as_str())
                .filter(|s| is_sha256_hex(s))
        }) {
            m.insert(name.to_string(), sha.to_string());
        }
    }
    Ok(m)
}

/// `true` for a 64-char lowercase-or-uppercase hex string (a sha256 digest).
fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Streaming SHA-256 of a file's contents → lowercase hex. Reads in 64 KiB
/// chunks so multi-GB shards never land in memory whole.
pub fn sha256_hex(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let mut hex = String::with_capacity(64);
    for b in hasher.finalize() {
        let _ = write!(hex, "{b:02x}");
    }
    Ok(hex)
}

/// Verify a downloaded file: exact size (when known) + `.safetensors` structural
/// integrity (header parses and data length matches the file). Content-agnostic
/// — see [`verify_file_hashed`] to also check a SHA-256.
pub fn verify_file(path: &Path, expected_size: Option<u64>) -> Result<()> {
    verify_file_hashed(path, expected_size, None)
}

/// Like [`verify_file`] but, when `expected_sha256` is `Some`, also streams the
/// file and compares its content digest (case-insensitive hex), bailing on a
/// mismatch. The size + structural checks are cheap; the digest is only worth
/// it once a file *looks* complete, so it runs last.
pub fn verify_file_hashed(
    path: &Path,
    expected_size: Option<u64>,
    expected_sha256: Option<&str>,
) -> Result<()> {
    let len = fs::metadata(path)?.len();
    if let Some(sz) = expected_size
        && sz != 0
        && len != sz
    {
        return Err(HubError::SizeMismatch {
            path: path.display().to_string(),
            got: len,
            want: sz,
        });
    }
    if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        verify_safetensors_structure(path, len)?;
    }
    if let Some(want) = expected_sha256 {
        let got = sha256_hex(path)?;
        if !got.eq_ignore_ascii_case(want) {
            return Err(HubError::Sha256Mismatch {
                path: path.display().to_string(),
                got,
                want: want.to_string(),
            });
        }
    }
    Ok(())
}

/// A `.safetensors` file is `u64 header_len | JSON header | tensor data`. The
/// max `data_offsets[1]` across tensors must equal `file_len - (8 + header_len)`.
fn verify_safetensors_structure(path: &Path, file_len: u64) -> Result<()> {
    let structural = |reason: String| HubError::Structural {
        path: path.display().to_string(),
        reason,
    };
    if file_len < 8 {
        return Err(structural(format!("file too small ({file_len} B)")));
    }
    let mut f = File::open(path)?;
    let mut lenb = [0u8; 8];
    f.read_exact(&mut lenb)?;
    let hlen = u64::from_le_bytes(lenb);
    // Subtract on the trusted side: `hlen` is untrusted, so `8 + hlen` could
    // overflow u64 and wrap past this guard, letting the alloc below balloon.
    // `file_len >= 8` is guaranteed above, so `file_len - 8` cannot underflow.
    if hlen > file_len - 8 {
        return Err(structural(format!(
            "header length {hlen} exceeds file (truncated)"
        )));
    }
    let mut hdr = vec![0u8; hlen as usize];
    f.read_exact(&mut hdr)?;
    // A malformed header is corruption, not a legit API JSON error → Structural.
    let v: serde_json::Value =
        serde_json::from_slice(&hdr).map_err(|e| structural(format!("parse header json: {e}")))?;
    let obj = v
        .as_object()
        .ok_or_else(|| structural("header not an object".to_string()))?;
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
        return Err(structural(format!(
            "data ends at {expected} B but file is {file_len} B (truncated/corrupt)"
        )));
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
        .map_err(spawn_err)?;
    if !status.success() {
        return Err(HubError::CommandFailed {
            file: file.to_string(),
            status: status.to_string(),
        });
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
/// (pass an empty map to rely on structural checks only); `sha256s` (from
/// [`fetch_sha256s`]) adds a content-digest check for the files it knows (pass
/// `None` to skip hashing entirely). `on_event(file, note)` reports progress.
pub fn download_files(
    repo: &HfRepo,
    files: &[String],
    dest_dir: &Path,
    sizes: &HashMap<String, u64>,
    sha256s: Option<&HashMap<String, String>>,
    mut on_event: impl FnMut(&str, &str),
) -> DownloadReport {
    fs::create_dir_all(dest_dir).ok();
    let mut report = DownloadReport::default();
    for file in files {
        let expected = sizes.get(file).copied();
        let sha = sha256s.and_then(|m| m.get(file)).map(String::as_str);
        let dest = dest_dir.join(file);
        // Fast path: already present + verifies (size + structure + sha256 when known).
        if dest.is_file() && verify_file_hashed(&dest, expected, sha).is_ok() {
            on_event(file, "cached");
            report.bytes +=
                expected.unwrap_or_else(|| fs::metadata(&dest).map(|m| m.len()).unwrap_or(0));
            report.ok.push(file.clone());
            continue;
        }
        on_event(file, "downloading");
        let res = download_file(repo, file, dest_dir).and_then(|p| {
            verify_file_hashed(&p, expected, sha)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Write `bytes` to a per-process temp file and return its path.
    fn tmp(name: &str, bytes: &[u8]) -> PathBuf {
        let p = std::env::temp_dir().join(format!("rlx-hub-test-{}-{name}", std::process::id()));
        fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn sha256_of_known_bytes() {
        let p = tmp("abc", b"abc");
        assert_eq!(
            sha256_hex(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        fs::remove_file(&p).ok();
    }

    #[test]
    fn is_sha256_hex_shape() {
        assert!(is_sha256_hex(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        ));
        assert!(!is_sha256_hex("deadbeef")); // too short (a git sha1 is 40)
        assert!(!is_sha256_hex(&"z".repeat(64))); // not hex
    }

    #[test]
    fn sha256_mismatch_bails() {
        let p = tmp("mismatch", b"hello");
        let err = verify_file_hashed(&p, None, Some(&"0".repeat(64))).unwrap_err();
        assert!(
            matches!(err, HubError::Sha256Mismatch { .. }),
            "got {err:?}"
        );
        // The real digest (any letter-case) verifies.
        let good = sha256_hex(&p).unwrap();
        assert!(verify_file_hashed(&p, None, Some(&good)).is_ok());
        assert!(verify_file_hashed(&p, None, Some(&good.to_uppercase())).is_ok());
        fs::remove_file(&p).ok();
    }

    #[test]
    fn size_mismatch_bails() {
        let p = tmp("size", b"12345");
        let err = verify_file_hashed(&p, Some(999), None).unwrap_err();
        assert!(
            matches!(
                err,
                HubError::SizeMismatch {
                    got: 5,
                    want: 999,
                    ..
                }
            ),
            "got {err:?}"
        );
        fs::remove_file(&p).ok();
    }
}
