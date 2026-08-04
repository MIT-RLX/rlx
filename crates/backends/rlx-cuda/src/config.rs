// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed CUDA runtime options loaded once from the `RLX_*` registry.

use std::sync::{OnceLock, RwLock};

use crate::{CompileMode, ExecMode};

/// Which attention kernel variant to run (`RLX_CUDA_ATTENTION`). `Auto` is
/// shape-aware: the Tensor-Core (WMMA) kernel when it's eligible AND the
/// workload is big enough to amortize its per-block overhead, else the scalar
/// flash / row kernel. `Scalar`/`Wmma`/`Row` force one variant (with a scalar
/// fallback when a forced variant can't handle the shape). See
/// `backend::run` dispatch + `docs/attention-variants.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionVariant {
    #[default]
    Auto,
    Scalar,
    Wmma,
    Row,
}

/// CUDA backend options (hot-path flags loaded once per process by default).
#[derive(Debug, Clone)]
pub struct CudaRuntimeConfig {
    pub wmma: bool,
    /// Attention kernel variant policy (`RLX_CUDA_ATTENTION=auto|scalar|wmma|row`).
    pub attention: AttentionVariant,
    /// `Auto`-mode threshold: only pick the WMMA attention kernel when
    /// `batch * heads * seq_q >= this` (below it the scalar kernel is as fast
    /// or faster — the WMMA per-block overhead doesn't amortize). Tunable via
    /// `RLX_CUDA_ATTENTION_WMMA_MIN_WORK` (default 12288).
    pub attention_wmma_min_work: u64,
    /// Opt-in Hopper TMA/wgmma kernel variants (`RLX_CUDA_TMA`). Default off;
    /// only takes effect on sm_90 (see `backend::helpers::tma_arch`).
    pub tma: bool,
    pub no_tf32: bool,
    pub parity: bool,
    pub no_cublaslt: bool,
    pub conv_tf32: bool,
    pub nondet_conv: bool,
    /// Prefer stable IMPLICIT_GEMM for conv bwd (default on).
    pub conv_stable_bwd: bool,
    pub conv_fwd_host: bool,
    pub conv_fwd_cudnn: bool,
    pub conv_bwd_host: bool,
    pub conv_bwd_cudnn: bool,
    pub no_cudnn: bool,
    pub im2col_host: bool,
    /// `None` = unset; `Some(false)` = `=0`; `Some(true)` = explicitly on.
    pub pinned_io: Option<bool>,
    pub compile_mode: CompileMode,
    pub exec_mode: ExecMode,
    pub log_fallback: bool,
    pub log_conv_path: bool,
    pub dump_nodes: bool,
    pub dump_nodes_limit: usize,
    pub dump_intermediate: bool,
    pub path_trace: bool,
    pub gguf_host: bool,
    pub compile_timing: bool,
    pub arena_debug: bool,
    pub no_packed_bshd_attn: bool,
    pub gdn_host: bool,
    pub ptx_cache: Option<String>,
}

impl Default for CudaRuntimeConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl CudaRuntimeConfig {
    /// Load from process env / [`rlx_ir::env`] overrides via the registry.
    pub fn from_env() -> Self {
        use rlx_ir::env_registry as reg;
        let pinned = match reg::var("RLX_CUDA_PINNED_IO").as_deref() {
            None => None,
            Some(v) if v.eq_ignore_ascii_case("0") => Some(false),
            Some(_) => Some(true),
        };
        let compile_mode = match reg::var("RLX_CUDA_COMPILE_MODE").as_deref() {
            Some(mode) if mode.eq_ignore_ascii_case("aot") => CompileMode::Aot,
            _ => CompileMode::Jit,
        };
        let exec_mode = match reg::var("RLX_CUDA_EXEC_MODE").as_deref() {
            Some(mode) if mode.eq_ignore_ascii_case("graph") => ExecMode::Graph,
            Some(mode) => {
                let lower = mode.to_ascii_lowercase();
                if let Some(rest) = lower.strip_prefix("multistream") {
                    let n = rest.trim_start_matches([':', '=']).parse().unwrap_or(2);
                    ExecMode::MultiStream(n.max(1))
                } else {
                    ExecMode::Stream
                }
            }
            None => ExecMode::Stream,
        };
        let attention = match reg::var("RLX_CUDA_ATTENTION").as_deref() {
            Some(v) if v.eq_ignore_ascii_case("scalar") => AttentionVariant::Scalar,
            Some(v) if v.eq_ignore_ascii_case("wmma") => AttentionVariant::Wmma,
            Some(v) if v.eq_ignore_ascii_case("row") => AttentionVariant::Row,
            Some(v) if v.eq_ignore_ascii_case("auto") => AttentionVariant::Auto,
            // Back-compat: the old boolean opt-in forces the WMMA variant.
            _ if reg::flag("RLX_CUDA_ATTENTION_WMMA") => AttentionVariant::Wmma,
            _ => AttentionVariant::Auto,
        };
        let attention_wmma_min_work = reg::var("RLX_CUDA_ATTENTION_WMMA_MIN_WORK")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(12288);
        Self {
            wmma: reg::flag("RLX_CUDA_WMMA"),
            attention,
            attention_wmma_min_work,
            tma: reg::flag("RLX_CUDA_TMA"),
            no_tf32: reg::flag("RLX_CUDA_NO_TF32"),
            parity: reg::flag("RLX_CUDA_PARITY"),
            no_cublaslt: reg::flag("RLX_CUDA_NO_CUBLASLT"),
            conv_tf32: reg::flag("RLX_CUDA_CONV_TF32"),
            nondet_conv: reg::flag("RLX_CUDA_NONDET_CONV"),
            conv_stable_bwd: reg::flag_or("RLX_CUDA_CONV_STABLE_BWD", true),
            conv_fwd_host: reg::flag("RLX_CUDA_CONV_FWD_HOST"),
            conv_fwd_cudnn: reg::flag("RLX_CUDA_CONV_FWD_CUDNN"),
            conv_bwd_host: reg::flag("RLX_CUDA_CONV_BWD_HOST"),
            conv_bwd_cudnn: reg::flag("RLX_CUDA_CONV_BWD_CUDNN"),
            no_cudnn: reg::flag("RLX_CUDA_NO_CUDNN"),
            im2col_host: reg::var("RLX_CUDA_IM2COL_HOST").is_some(),
            pinned_io: pinned,
            compile_mode,
            exec_mode,
            log_fallback: reg::flag("RLX_CUDA_LOG_FALLBACK"),
            log_conv_path: reg::flag("RLX_CUDA_LOG_CONV_PATH"),
            dump_nodes: reg::flag("RLX_CUDA_DUMP_NODES"),
            dump_nodes_limit: reg::parse_or("RLX_CUDA_DUMP_NODES_LIMIT", 64usize),
            dump_intermediate: reg::flag("RLX_CUDA_DUMP_INTERMEDIATE"),
            path_trace: reg::flag("RLX_CUDA_PATH_TRACE"),
            gguf_host: reg::var("RLX_CUDA_GGUF_HOST").as_deref() == Some("1"),
            compile_timing: reg::flag("RLX_CUDA_COMPILE_TIMING"),
            arena_debug: reg::flag("RLX_CUDA_ARENA_DEBUG"),
            no_packed_bshd_attn: reg::flag("RLX_CUDA_NO_PACKED_BSHD_ATTN"),
            gdn_host: reg::flag("RLX_CUDA_GDN_HOST"),
            ptx_cache: reg::var("RLX_CUDA_PTX_CACHE"),
        }
    }

    pub fn matmul_parity_mode(&self) -> bool {
        self.no_tf32 || self.parity || self.no_cublaslt
    }

    pub fn conv_tf32_enabled(&self) -> bool {
        self.conv_tf32 && !(self.no_tf32 || self.parity)
    }

    pub fn deterministic_conv(&self) -> bool {
        !self.nondet_conv
    }

    pub fn pinned_host_io_disabled(&self) -> bool {
        self.pinned_io == Some(false)
    }

    pub fn pinned_output_staging_enabled(&self) -> bool {
        !self.pinned_host_io_disabled()
    }

    pub fn pinned_input_staging_enabled(&self, exec_mode: ExecMode) -> bool {
        if self.pinned_host_io_disabled() {
            return false;
        }
        matches!(exec_mode, ExecMode::Graph) || self.pinned_io == Some(true)
    }

    pub fn im2col_use_gpu(&self, n: u32, exec_mode: ExecMode) -> bool {
        if self.im2col_host {
            return false;
        }
        if matches!(exec_mode, ExecMode::Graph) {
            return n > 0;
        }
        n > 0
    }
}

static CONFIG: OnceLock<RwLock<CudaRuntimeConfig>> = OnceLock::new();

fn map() -> &'static RwLock<CudaRuntimeConfig> {
    CONFIG.get_or_init(|| RwLock::new(CudaRuntimeConfig::from_env()))
}

/// Process-wide CUDA config (loaded once; refresh with [`reload_runtime_config`]).
pub fn runtime_config() -> CudaRuntimeConfig {
    map().read().expect("cuda config").clone()
}

/// Re-read env into the process-wide config (tests / bisect).
pub fn reload_runtime_config() {
    *map().write().expect("cuda config") = CudaRuntimeConfig::from_env();
}

/// Install an explicit config (tests).
pub fn install_runtime_config(cfg: CudaRuntimeConfig) {
    *map().write().expect("cuda config") = cfg;
}
