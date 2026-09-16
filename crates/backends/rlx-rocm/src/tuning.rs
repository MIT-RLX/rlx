// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistence for the shared GPU dispatch table — ROCm side.
//!
//! Mirrors `rlx_cuda::tuning`. The table and its (de)serialization live in
//! `rlx_gpu_kernels::dispatch`, which touches neither the filesystem nor the
//! environment; each backend owns where its cache lives and when it is read.
//!
//! The cache file is shared in *format* but not in *content*: every record
//! carries its `arch` (`sm_86`, `gfx908`, …), so pointing both backends at one
//! file is safe — a CUDA row simply never matches a ROCm lookup. They default to
//! separate paths anyway, because a machine usually has one or the other.

use std::path::PathBuf;
use std::sync::OnceLock;

/// Where the tuning cache lives. `RLX_GPU_TUNING_CACHE` overrides; otherwise it
/// sits beside the `.hsaco` cache under `$XDG_CACHE_HOME` / `~/.cache`.
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
    Some(base.join("rlx-rocm").join("dispatch-tuning.tsv"))
}

/// Read the persisted overrides into the process table. Runs at most once.
///
/// Failures are silent by design: an unreadable, truncated, or newer-format
/// cache must degrade to the compile-time defaults, which are the historical
/// hand-written routing. Set `RLX_VERBOSE` to see what was applied and skipped.
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
                "rlx-rocm: dispatch tuning cache {}: {} applied, {} skipped",
                path.display(),
                report.applied,
                report.skipped
            );
        }
    });
}

/// Write the current override table to the tuning cache. `None` when
/// persistence is disabled or the write failed — a tuner should report that
/// rather than assume its work was saved.
pub fn save_tuning_cache() -> Option<PathBuf> {
    let path = tuning_cache_path()?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok()?;
    }
    std::fs::write(&path, rlx_gpu_kernels::dispatch::save_overrides()).ok()?;
    Some(path)
}
