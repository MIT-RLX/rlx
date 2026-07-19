// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Curated catalog of the `RLX_*` options users hit most often.
//!
//! Hundreds of `RLX_*` names exist for bisect / escape hatches. This table is
//! the **discoverability** surface — not an exhaustive dump. Prefer
//! [`crate::CompileOptions`] when a setting affects compile semantics.
//!
//! Print with [`format_catalog`] or `just env-catalog`.

/// One documented environment variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvVarDoc {
    /// Full name including `RLX_` prefix.
    pub name: &'static str,
    /// Short group label (`device`, `metal`, `gguf`, …).
    pub group: &'static str,
    /// One-line purpose.
    pub summary: &'static str,
}

/// High-signal `RLX_*` variables, grouped for humans and agents.
pub const CATALOG: &[EnvVarDoc] = &[
    // ── Device selection ──────────────────────────────────────────
    EnvVarDoc {
        name: "RLX_DEVICE",
        group: "device",
        summary: "Default device hint for resolved runs (cpu, metal, mlx, cuda, gpu, …)",
    },
    EnvVarDoc {
        name: "RLX_DEVICE_CHAIN",
        group: "device",
        summary: "Fallback order when a preferred device fails (e.g. cuda,gpu,cpu)",
    },
    EnvVarDoc {
        name: "RLX_DEVICES",
        group: "device",
        summary: "Allow-list of devices for DevicePolicy::from_env",
    },
    EnvVarDoc {
        name: "RLX_BENCHMARK_PICK",
        group: "device",
        summary: "Micro-benchmark N runs to pick the fastest device (needs inputs)",
    },
    // ── Dispatch / debug ──────────────────────────────────────────
    EnvVarDoc {
        name: "RLX_DISPATCH_REPORT",
        group: "debug",
        summary: "Print legalize/dispatch report during compile (1 = on)",
    },
    EnvVarDoc {
        name: "RLX_DBG_CUSTOM",
        group: "debug",
        summary: "Log host custom-op staging (onnx.* dtype bridge) on GPU backends",
    },
    EnvVarDoc {
        name: "RLX_VERBOSE",
        group: "debug",
        summary: "Extra runtime logging",
    },
    EnvVarDoc {
        name: "RLX_ALLOW_THROTTLE",
        group: "debug",
        summary: "Skip thermal gate for one-off benches (prefer `just throttle`)",
    },
    // ── Metal ─────────────────────────────────────────────────────
    EnvVarDoc {
        name: "RLX_DISABLE_MPSGRAPH",
        group: "metal",
        summary: "Force Metal thunk path instead of MPSGraph regions",
    },
    EnvVarDoc {
        name: "RLX_METAL_DEQUANT_GPU_DISABLE",
        group: "metal",
        summary: "Disable Metal GPU GGUF dequant (host / legacy path)",
    },
    EnvVarDoc {
        name: "RLX_METAL_DEQUANT_MATMUL_LEGACY",
        group: "metal",
        summary: "Use pre-fused dequant+matmul path (materializes weights)",
    },
    // ── MLX ───────────────────────────────────────────────────────
    EnvVarDoc {
        name: "RLX_MLX_MODE",
        group: "mlx",
        summary: "eager | lazy | compiled execution mode",
    },
    EnvVarDoc {
        name: "RLX_MLX_GGUF_HOST_FALLBACK",
        group: "mlx",
        summary: "Force host GGUF dequant on MLX",
    },
    EnvVarDoc {
        name: "RLX_MLX_SDPA_REFERENCE",
        group: "mlx",
        summary: "Use reference SDPA composition for bisects",
    },
    // ── GGUF / quant ──────────────────────────────────────────────
    EnvVarDoc {
        name: "RLX_DISABLE_METAL_DEQUANT_GPU",
        group: "gguf",
        summary: "Alias family: opt out of Metal on-device GGUF dequant",
    },
    // ── wgpu / CUDA / ROCm ────────────────────────────────────────
    EnvVarDoc {
        name: "RLX_WGPU_GDN_HOST",
        group: "wgpu",
        summary: "Force GatedDeltaNet host fallback on wgpu (skip WGSL)",
    },
    EnvVarDoc {
        name: "RLX_ROCM_PINNED_IO",
        group: "rocm",
        summary: "Use pinned host I/O for ROCm graph exec (default on in graph mode)",
    },
    EnvVarDoc {
        name: "RLX_INDEXING_FULL_ARENA",
        group: "gpu",
        summary: "Force full-arena mirror for indexing host-fallback (bisect; slow on discrete GPUs)",
    },
    EnvVarDoc {
        name: "RLX_FFT_FAST",
        group: "gpu",
        summary: "Enable native on-chip GPU FFT when compiled (0 disables)",
    },
    EnvVarDoc {
        name: "RLX_CUDA_CONV_FWD_HOST",
        group: "cuda",
        summary: "Force CUDA Conv2d forward through CPU host-fallback (bisect)",
    },
    EnvVarDoc {
        name: "RLX_CUDA_CONV_BWD_CUDNN",
        group: "cuda",
        summary: "Allow cuDNN for grouped/degenerate Conv2d backward shapes",
    },
    EnvVarDoc {
        name: "RLX_FAST_CONV",
        group: "cpu",
        summary: "CPU Conv2d im2col+BLAS path (default on; set 0 for scalar nested loops)",
    },
    EnvVarDoc {
        name: "RLX_NO_IO_PEAKS_OUTPUT",
        group: "compile",
        summary: "Disable compile-time IO-gated peaks-only fusion",
    },
    // ── CoreML / ANE ──────────────────────────────────────────────
    EnvVarDoc {
        name: "RLX_COREML_HOST_DEQUANT",
        group: "coreml",
        summary: "Force CoreML hybrid host dequant segments",
    },
];

/// Entries whose `group` matches (case-sensitive), or all when `group` is empty.
pub fn catalog_for_group(group: &str) -> Vec<&'static EnvVarDoc> {
    if group.is_empty() {
        return CATALOG.iter().collect();
    }
    CATALOG.iter().filter(|e| e.group == group).collect()
}

/// Pretty-print the catalog (optionally filtered by group).
pub fn format_catalog(group: Option<&str>) -> String {
    let entries = catalog_for_group(group.unwrap_or(""));
    let mut out = String::from("# RLX environment catalog (curated)\n\n");
    let mut cur = "";
    for e in entries {
        if e.group != cur {
            cur = e.group;
            out.push_str(&format!("## {cur}\n\n"));
        }
        out.push_str(&format!("- `{}` — {}\n", e.name, e.summary));
    }
    out.push_str(
        "\nNot exhaustive — escape-hatch flags live in backend sources. \
         Prefer CompileOptions when a setting changes compile semantics.\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_nonempty_and_prefixed() {
        assert!(!CATALOG.is_empty());
        for e in CATALOG {
            assert!(e.name.starts_with("RLX_"), "{}", e.name);
            assert!(!e.group.is_empty());
            assert!(!e.summary.is_empty());
        }
    }

    #[test]
    fn format_includes_device_group() {
        let s = format_catalog(Some("device"));
        assert!(s.contains("RLX_DEVICE"));
        assert!(!s.contains("RLX_DISABLE_MPSGRAPH"));
    }
}
