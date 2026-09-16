// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistence + arch key for the shared GPU dispatch table — Metal side.
//!
//! Mirrors `rlx_cuda::tuning` / `rlx_rocm::tuning`. The table and its
//! (de)serialization live in `rlx_gpu_dispatch::dispatch`, which touches neither
//! the filesystem nor the environment; each backend owns where its cache lives
//! and when it is read.
//!
//! What Metal tunes is different from CUDA's. There is no scalar tiled `matmul`
//! kernel here to re-tile — dense f32 GEMM runs through MPS or one of seven
//! hand-written simdgroup kernels — so the tunable decision is **which variant**,
//! and the table's job is to override `cost::pick_sgemm`'s hand-written cascade
//! with a measured answer.

use std::path::PathBuf;
use std::sync::OnceLock;

/// This device's key in the shared dispatch table, with the persisted tuning
/// cache guaranteed loaded.
///
/// Keyed by Apple GPU family rather than the full device name: that is the
/// granularity Metal's own cost model already varies at (`MetalHwModel`), and a
/// finer key would fragment the cache without changing any decision.
pub fn gpu_arch() -> &'static rlx_gpu_dispatch::dispatch::GpuArch {
    static ARCH: OnceLock<rlx_gpu_dispatch::dispatch::GpuArch> = OnceLock::new();
    ARCH.get_or_init(|| {
        ensure_tuning_cache_loaded();
        let family = format!("{:?}", crate::cost::hw_model().gpu_family).to_lowercase();
        rlx_gpu_dispatch::dispatch::GpuArch::metal(&family)
    })
}

/// Where the tuning cache lives. `RLX_GPU_TUNING_CACHE` overrides; otherwise
/// under `$XDG_CACHE_HOME` / `~/.cache`.
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
    Some(base.join("rlx-metal").join("dispatch-tuning.tsv"))
}

/// Read the persisted overrides into the process table. Runs at most once.
///
/// Failures are silent by design: an unreadable or newer-format cache must
/// degrade to the compile-time defaults — here, `cost::pick_sgemm`'s existing
/// cascade — rather than take the process down. `RLX_VERBOSE` reports counts.
pub fn ensure_tuning_cache_loaded() {
    static LOADED: OnceLock<()> = OnceLock::new();
    LOADED.get_or_init(|| {
        let Some(path) = tuning_cache_path() else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return;
        };
        let report = rlx_gpu_dispatch::dispatch::load_overrides(&text);
        if rlx_ir::env::flag("RLX_VERBOSE") {
            eprintln!(
                "rlx-metal: dispatch tuning cache {}: {} applied, {} skipped",
                path.display(),
                report.applied,
                report.skipped
            );
        }
    });
}

/// Write the current override table to the tuning cache. `None` when
/// persistence is disabled or the write failed.
pub fn save_tuning_cache() -> Option<PathBuf> {
    let path = tuning_cache_path()?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok()?;
    }
    std::fs::write(&path, rlx_gpu_dispatch::dispatch::save_overrides()).ok()?;
    Some(path)
}
