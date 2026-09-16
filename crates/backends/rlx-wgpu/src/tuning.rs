// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistence + arch key for the shared GPU dispatch table — wgpu side.
//!
//! Mirrors `rlx_cuda::tuning` / `rlx_metal::tuning`. The table and its
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
/// Keyed by **backend and adapter**, not just the chip: wgpu runs over Metal,
/// Vulkan, DX12 and GL, and which matmul path wins differs per backend at least
/// as much as per GPU — the CoopF32 kernel that is opt-in-only on Metal (it
/// produced orthogonal garbage there) is the fast path on a discrete Vulkan
/// adapter. One key covering both would average two opposite answers.
pub fn gpu_arch() -> &'static rlx_gpu_dispatch::dispatch::GpuArch {
    static ARCH: OnceLock<rlx_gpu_dispatch::dispatch::GpuArch> = OnceLock::new();
    ARCH.get_or_init(|| {
        ensure_tuning_cache_loaded();
        match crate::device::wgpu_device() {
            Some(d) => {
                rlx_gpu_dispatch::dispatch::GpuArch::wgpu(&format!("{:?}", d.backend), &d.name)
            }
            None => rlx_gpu_dispatch::dispatch::GpuArch::unknown(),
        }
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
    Some(base.join("rlx-wgpu").join("dispatch-tuning.tsv"))
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
                "rlx-wgpu: dispatch tuning cache {}: {} applied, {} skipped",
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
    use rlx_gpu_dispatch::dispatch::{Choice, WgpuMatmul};

    /// The two enums must map onto each other exactly. A variant added on one
    /// side without the other would silently become a *different* kernel.
    #[test]
    fn dispatch_variant_mapping_round_trips() {
        use crate::backend::{matmul_compute_from_dispatch, matmul_compute_to_dispatch};
        for d in rlx_gpu_dispatch::dispatch::WGPU_MATMUL_VARIANTS {
            assert_eq!(
                matmul_compute_to_dispatch(matmul_compute_from_dispatch(*d)),
                *d,
                "round-trip {d:?}"
            );
        }
    }

    /// wgpu adapter names carry spaces and vendor punctuation, and the cache
    /// format is tab-separated — a tab in the arch key would split a record and
    /// silently drop the tuning it encodes.
    #[test]
    fn arch_key_survives_the_cache_format() {
        let arch = rlx_gpu_dispatch::dispatch::GpuArch::wgpu("Metal", "Apple M4 Pro");
        let w = rlx_gpu_dispatch::dispatch::Workload::Matmul {
            m: 1,
            k: 4096,
            n: 4096,
        };
        rlx_gpu_dispatch::dispatch::clear_overrides();
        rlx_gpu_dispatch::dispatch::set_override(
            w.key(&arch),
            Choice::WgpuMatmul(WgpuMatmul::CoopF32),
        )
        .expect("legal override");
        let text = rlx_gpu_dispatch::dispatch::save_overrides();
        rlx_gpu_dispatch::dispatch::clear_overrides();
        let report = rlx_gpu_dispatch::dispatch::load_overrides(&text);
        assert_eq!(report.applied, 1, "arch key did not survive a round-trip");
        assert_eq!(
            rlx_gpu_dispatch::dispatch::resolve(&arch, &w),
            Choice::WgpuMatmul(WgpuMatmul::CoopF32)
        );
        rlx_gpu_dispatch::dispatch::clear_overrides();
    }

    /// Two adapters on different wgpu backends must not share tuning — the
    /// CoopF32 path that is opt-in-only on Metal is the fast path on Vulkan.
    #[test]
    fn backend_is_part_of_the_key() {
        let metal = rlx_gpu_dispatch::dispatch::GpuArch::wgpu("Metal", "Apple M4 Pro");
        let vulkan = rlx_gpu_dispatch::dispatch::GpuArch::wgpu("Vulkan", "Apple M4 Pro");
        assert_ne!(metal, vulkan);
    }
}
