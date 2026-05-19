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

//! Compile-time kernel-config tables (plan #14).
//!
//! Borrowed from MAX's `internal_utils/nvidia_configs.mojo` /
//! `amd_configs.mojo` pattern: tile sizes, kernel-selection
//! thresholds, etc. as compile-time data structures kernels query
//! instead of scattered match-arms.
//!
//! Today the values live as `const`s here and are surfaced through
//! [`kernel_config_for`]. The goal is one source of truth — when
//! we want to tune for a new arch (M5 Apple Silicon, x86 Zen5,
//! etc.) we add a row to the table, not a new match arm in 12
//! files.

/// Coarse target classification — refined as new SoCs emerge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuArch {
    AppleSilicon, // M-series; AMX + NEON
    AarchGeneric, // Other ARM (RPi, AWS Graviton)
    X86_64,
    Other,
}

impl CpuArch {
    /// Pick the best label for the running target.
    pub const fn current() -> Self {
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            Self::AppleSilicon
        }
        #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
        {
            Self::AarchGeneric
        }
        #[cfg(target_arch = "x86_64")]
        {
            Self::X86_64
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::Other
        }
    }
}

/// Op category that a kernel config is keyed against. Coarse —
/// refines as we learn which thresholds actually want per-shape
/// tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpClass {
    /// Matmul / linear-algebra hot path.
    Matmul,
    /// SDPA attention.
    Attention,
    /// Element-wise / activation / norm.
    Elementwise,
    /// View / shape ops (reshape, narrow, transpose).
    Shape,
}

/// Settings the dispatch logic asks for. All in elements unless
/// noted otherwise.
#[derive(Debug, Clone, Copy)]
pub struct KernelConfig {
    /// Below this batch*seq, prefer NEON over BLAS for matmul / SDPA.
    pub neon_seq_threshold: usize,
    /// par_for granularity for elementwise.
    pub par_grain: usize,
    /// Below this total element count, run sequentially (par_for
    /// has positive overhead even at 0 work).
    pub par_threshold: usize,
    /// FusedAttnBlock fires when batch*seq <= this.
    pub fuse_attn_threshold: usize,
}

const APPLE_SILICON: KernelConfig = KernelConfig {
    neon_seq_threshold: 32,
    par_grain: 64,
    par_threshold: 30_000,
    fuse_attn_threshold: 64,
};

const AARCH_GENERIC: KernelConfig = KernelConfig {
    neon_seq_threshold: 24,
    par_grain: 32,
    par_threshold: 20_000,
    fuse_attn_threshold: 48,
};

const X86_DEFAULT: KernelConfig = KernelConfig {
    neon_seq_threshold: 16, // AVX2 path; lower threshold reflects bigger vector unit
    par_grain: 32,
    par_threshold: 20_000,
    fuse_attn_threshold: 32,
};

const FALLBACK: KernelConfig = KernelConfig {
    neon_seq_threshold: 16,
    par_grain: 16,
    par_threshold: 10_000,
    fuse_attn_threshold: 16,
};

/// Look up the canonical kernel config for `(arch, op_class)`. The
/// table is `const`-evaluated so callers pay no lookup cost.
pub const fn kernel_config_for(arch: CpuArch, op: OpClass) -> KernelConfig {
    // Today the per-op variation is small — we return one row per
    // arch and let callers read the field they care about. As more
    // shape-specific tuning data lands, the OpClass dimension
    // becomes load-bearing.
    let _ = op;
    match arch {
        CpuArch::AppleSilicon => APPLE_SILICON,
        CpuArch::AarchGeneric => AARCH_GENERIC,
        CpuArch::X86_64 => X86_DEFAULT,
        CpuArch::Other => FALLBACK,
    }
}

/// Convenience: defaults for the running target.
pub const fn current_config(op: OpClass) -> KernelConfig {
    kernel_config_for(CpuArch::current(), op)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_resolves() {
        let cfg = current_config(OpClass::Matmul);
        // All targets at minimum produce a non-zero threshold.
        assert!(cfg.neon_seq_threshold > 0);
        assert!(cfg.par_threshold > 0);
    }

    #[test]
    fn apple_silicon_picks_higher_thresholds() {
        let m = kernel_config_for(CpuArch::AppleSilicon, OpClass::Matmul);
        let f = kernel_config_for(CpuArch::Other, OpClass::Matmul);
        assert!(m.neon_seq_threshold >= f.neon_seq_threshold);
        assert!(m.par_threshold >= f.par_threshold);
    }
}
