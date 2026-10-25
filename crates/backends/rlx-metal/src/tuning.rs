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
/// Keyed by Apple GPU family **and tier** (`metal-m4-pro`), not the full
/// device name. Family alone lumped a 10-core M4 with a 40-core M4 Max, and the
/// sgemm choice is occupancy-sensitive (`Simd64` needs enough threadgroups to
/// fill the GPU), so a winner measured on one tier is not evidence for another
/// — which matters once rows ship (`rlx_gpu_dispatch::shipped`).
pub fn gpu_arch() -> &'static rlx_gpu_dispatch::dispatch::GpuArch {
    static ARCH: OnceLock<rlx_gpu_dispatch::dispatch::GpuArch> = OnceLock::new();
    ARCH.get_or_init(|| {
        ensure_tuning_cache_loaded();
        let hw = crate::cost::hw_model();
        rlx_gpu_dispatch::dispatch::GpuArch::metal(&arch_family_tier(
            &format!("{:?}", hw.gpu_family),
            &hw.gpu_name,
        ))
    })
}

/// `family-tier` from the cost model's family and `MTLDevice.name`
/// (`"M4"`, `"Apple M4 Pro"` → `m4-pro`). Tier is `base` unless the name says
/// pro / max / ultra.
fn arch_family_tier(family: &str, device_name: &str) -> String {
    let lower = device_name.to_lowercase();
    let tier = ["ultra", "max", "pro"]
        .into_iter()
        .find(|t| lower.split_whitespace().any(|w| w == *t))
        .unwrap_or("base");
    format!("{}-{tier}", family.to_lowercase())
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
        // Shipped per-arch rows first; the user cache below overrides them.
        if !rlx_ir::env::flag("RLX_GPU_NO_SHIPPED_TUNING") {
            rlx_gpu_dispatch::shipped::load();
        }
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

#[cfg(test)]
mod tests {
    use super::arch_family_tier;

    /// Tiers of one family must not share tuning rows.
    #[test]
    fn arch_key_separates_tiers() {
        assert_eq!(arch_family_tier("M4", "Apple M4 Pro"), "m4-pro");
        assert_eq!(arch_family_tier("M4", "Apple M4"), "m4-base");
        assert_eq!(arch_family_tier("M4", "Apple M4 Max"), "m4-max");
        assert_eq!(arch_family_tier("M1Pro", "Apple M1 Ultra"), "m1pro-ultra");
        assert_ne!(
            arch_family_tier("M4", "Apple M4"),
            arch_family_tier("M4", "Apple M4 Pro")
        );
    }
}
