// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistence for the shared GPU dispatch table.
//!
//! `rlx_gpu_kernels::dispatch` owns the table and its (de)serialization as pure
//! string functions — that crate is a dependency-free source-and-decisions crate
//! and deliberately touches neither the filesystem nor the environment. This
//! module is the CUDA backend's side of that split: where the cache lives, when
//! it is read, and when it is written.
//!
//! The point of persisting at all is that the measurement is the expensive part.
//! A tuning sweep compiles and benchmarks several physical schedules per shape
//! bucket; that cost should be paid once per machine, not once per process. On a
//! cold cache every lookup falls back to the compile-time default, which is the
//! historical hand-written routing — so a missing or stale cache costs
//! performance, never correctness.

use std::path::PathBuf;
use std::sync::OnceLock;

/// Where the tuning cache lives.
///
/// `RLX_GPU_TUNING_CACHE` overrides; otherwise it sits beside the PTX cache
/// under `$XDG_CACHE_HOME` / `~/.cache`. `None` disables persistence (the table
/// still works, it just re-tunes every process).
pub fn tuning_cache_path() -> Option<PathBuf> {
    if let Some(p) = rlx_ir::env::var("RLX_GPU_TUNING_CACHE") {
        return Some(PathBuf::from(p));
    }
    let base = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cache"))
        })?;
    Some(base.join("rlx-cuda").join("dispatch-tuning.tsv"))
}

/// Read the persisted overrides into the process table. Runs at most once.
///
/// Failures are silent by design: an unreadable, truncated, or
/// newer-format cache must degrade to the compile-time defaults, not take the
/// process down. Set `RLX_VERBOSE` to see what was applied and skipped.
pub fn ensure_tuning_cache_loaded() {
    static LOADED: OnceLock<()> = OnceLock::new();
    LOADED.get_or_init(|| {
        let Some(path) = tuning_cache_path() else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return;
        };
        let report = rlx_gpu_kernels::dispatch::load_overrides(&text);
        if rlx_ir::env::flag("RLX_VERBOSE") {
            eprintln!(
                "rlx-cuda: dispatch tuning cache {}: {} applied, {} skipped",
                path.display(),
                report.applied,
                report.skipped
            );
        }
    });
}

/// Write the current override table to the tuning cache.
///
/// Returns the path written, or `None` when persistence is disabled or the write
/// failed — a tuner should report that rather than assume its work was saved.
pub fn save_tuning_cache() -> Option<PathBuf> {
    let path = tuning_cache_path()?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok()?;
    }
    std::fs::write(&path, rlx_gpu_kernels::dispatch::save_overrides()).ok()?;
    Some(path)
}

/// Where the append-only decision journal lives — beside the tuning cache.
///
/// Separate file from the cache on purpose: the cache is *current state*, read on
/// every process start and kept small; the journal only grows and is only ever
/// read by a human asking "when did this route change, and why".
pub fn decision_journal_path() -> Option<PathBuf> {
    tuning_cache_path().map(|p| p.with_extension("journal.tsv"))
}

/// Append tuning decisions to the journal, creating it with a header if new.
///
/// Returns the path written, or `None` if journaling is disabled or the write
/// failed. A failed journal write must never fail a tuning run — the cache is the
/// artifact that matters; the journal is provenance.
pub fn append_decisions(records: &[rlx_gpu_dispatch::dispatch::DecisionRecord]) -> Option<PathBuf> {
    use std::io::Write as _;
    if records.is_empty() {
        return None;
    }
    let path = decision_journal_path()?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok()?;
    }
    let fresh = !path.exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .ok()?;
    if fresh {
        f.write_all(rlx_gpu_dispatch::dispatch::decision_journal_header().as_bytes())
            .ok()?;
    }
    f.write_all(rlx_gpu_dispatch::dispatch::render_decisions(records).as_bytes())
        .ok()?;
    Some(path)
}

/// Seconds since the Unix epoch, for stamping journal records.
///
/// The dispatch crate deliberately does not read the clock (dependency-free and
/// deterministic under test), so the timestamp is supplied here.
pub fn now_unix_s() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An explicit path is honoured verbatim — the tuner and the runtime must
    /// agree on where the cache is when a harness points them at a temp dir.
    #[test]
    fn explicit_path_overrides_the_default_location() {
        // SAFETY: single-threaded test, restored immediately.
        let prev = rlx_ir::env::var("RLX_GPU_TUNING_CACHE");
        unsafe { std::env::set_var("RLX_GPU_TUNING_CACHE", "/tmp/rlx-tuning-test.tsv") };
        assert_eq!(
            tuning_cache_path(),
            Some(PathBuf::from("/tmp/rlx-tuning-test.tsv"))
        );
        unsafe {
            match prev {
                Some(v) => std::env::set_var("RLX_GPU_TUNING_CACHE", v),
                None => std::env::remove_var("RLX_GPU_TUNING_CACHE"),
            }
        }
    }
}
