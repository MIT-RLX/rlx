// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed error taxonomy for the crate's public surface. Downstream consumers
//! match on [`HubError`] variants (missing tool, size / sha256 / structural
//! mismatch, …) instead of destructuring opaque `anyhow` strings; internal
//! helpers stay pragmatic and lift into these via `?`.

use thiserror::Error;

/// A `rlx-hub` result. Aliases the single [`HubError`] type so signatures read
/// like `std::io::Result`.
pub(crate) type Result<T> = std::result::Result<T, HubError>;

/// Everything that can go wrong fetching / verifying a checkpoint file.
#[derive(Debug, Error)]
pub enum HubError {
    /// `curl` isn't on `PATH` — every download shells out to it.
    #[error("curl not found on PATH (needed for downloads); install curl and retry")]
    MissingTool,

    /// A metadata `curl` (captured output) exited non-zero.
    #[error("curl {url} failed: {stderr}")]
    Curl { url: String, stderr: String },

    /// A file-download `curl` (streamed to disk) exited non-zero.
    #[error("download of {file} failed (curl {status})")]
    CommandFailed { file: String, status: String },

    /// Downloaded size ≠ the size the HF API declared (incomplete transfer).
    #[error("{path}: size {got} != expected {want} (incomplete)")]
    SizeMismatch { path: String, got: u64, want: u64 },

    /// Content SHA-256 ≠ the digest the HF API declared (corrupt/altered).
    #[error("{path}: sha256 {got} != expected {want} (corrupt)")]
    Sha256Mismatch {
        path: String,
        got: String,
        want: String,
    },

    /// A `.safetensors` file failed its header / data-length structural check.
    #[error("{path}: {reason}")]
    Structural { path: String, reason: String },

    /// A `model.safetensors.index.json` couldn't be understood.
    #[error("index.json: {0}")]
    Index(String),

    /// Filesystem / process I/O.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// JSON coming off the HF API.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
