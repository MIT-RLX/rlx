// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runtime tuning for the MLX backend.

use std::sync::{OnceLock, RwLock};

use crate::ffi;
use crate::lower::MlxMode;

/// Default `mlx::compile` multi-output slot limit when env / explicit override unset.
pub const DEFAULT_COMPILE_OUTPUT_CAP: usize = 1024;

/// Primary env var (also read by `rlx-runtime::compile_output_cap`).
pub const COMPILE_OUTPUT_CAP_ENV: &str = "RLX_COMPILE_OUTPUT_CAP";

/// Legacy alias env var.
pub const COMPILE_OUTPUT_CAP_ENV_MLX: &str = "RLX_MLX_COMPILE_OUTPUT_CAP";

/// Current compile output cap (explicit override, else env, else
/// [`DEFAULT_COMPILE_OUTPUT_CAP`]).
pub fn compile_output_cap() -> usize {
    unsafe { ffi::rlx_mlx_compile_output_cap() }
}

/// Override the compile output cap for this process. Pass `0` to clear the override
/// (same as [`reset_compile_output_cap`]).
pub fn set_compile_output_cap(cap: usize) {
    unsafe {
        ffi::rlx_mlx_set_compile_output_cap(cap);
    }
}

/// Clear an explicit cap override; subsequent reads use env / default again.
pub fn reset_compile_output_cap() {
    unsafe {
        ffi::rlx_mlx_reset_compile_output_cap();
    }
}

/// Default IR node count above which Compiled is skipped for Lazy.
pub const DEFAULT_COMPILE_MAX_NODES: usize = 1536;

/// Typed MLX runtime options loaded once from the `RLX_*` registry.
#[derive(Debug, Clone)]
pub struct MlxRuntimeConfig {
    pub mode: MlxMode,
    pub fuse_cap: usize,
    pub debug_eval: bool,
    pub gguf_host_fallback: bool,
    pub sdpa_reference: bool,
    pub q1_host: bool,
    pub q1_mv_disable: bool,
    pub dequant_cache_disable: bool,
    pub dequant_cache_bytes: Option<usize>,
    /// `None` = no limit (force Compiled); `Some(n)` = skip Compiled above n nodes.
    pub compile_max_nodes: Option<usize>,
    pub warn_lazy_all: bool,
    pub param_view: bool,
}

impl Default for MlxRuntimeConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl MlxRuntimeConfig {
    pub fn from_env() -> Self {
        use rlx_ir::env_registry as reg;
        let mode = match reg::var("RLX_MLX_MODE").as_deref() {
            Some(s) if s.eq_ignore_ascii_case("eager") => MlxMode::Eager,
            Some(s) if s.eq_ignore_ascii_case("lazy") => MlxMode::Lazy,
            Some(s) if s.eq_ignore_ascii_case("compiled") => MlxMode::Compiled,
            // Historical default when unset: Compiled (see `mode_from_env`).
            _ => MlxMode::Compiled,
        };
        let compile_max_nodes = match reg::var("RLX_MLX_COMPILE_MAX_NODES") {
            Some(s) => match s.parse::<usize>() {
                Ok(0) => None,
                Ok(n) => Some(n),
                Err(_) => Some(DEFAULT_COMPILE_MAX_NODES),
            },
            None => Some(DEFAULT_COMPILE_MAX_NODES),
        };
        Self {
            mode,
            fuse_cap: reg::parse_or("RLX_MLX_FUSE_CAP", 12usize),
            debug_eval: reg::var("RLX_MLX_DEBUG_EVAL").is_some(),
            gguf_host_fallback: reg::flag("RLX_MLX_GGUF_HOST_FALLBACK"),
            sdpa_reference: reg::flag("RLX_MLX_SDPA_REFERENCE"),
            q1_host: reg::var("RLX_MLX_Q1_HOST").as_deref() == Some("1"),
            q1_mv_disable: reg::var("RLX_MLX_Q1_MV_DISABLE").as_deref() == Some("1"),
            dequant_cache_disable: reg::var("RLX_MLX_DEQUANT_CACHE_DISABLE").as_deref()
                == Some("1"),
            dequant_cache_bytes: reg::var("RLX_MLX_DEQUANT_CACHE_BYTES")
                .and_then(|s| s.parse().ok()),
            compile_max_nodes,
            warn_lazy_all: reg::var("RLX_MLX_WARN_LAZY")
                .as_deref()
                .is_some_and(|v| v.eq_ignore_ascii_case("all")),
            param_view: reg::var("RLX_MLX_PARAM_VIEW").as_deref() == Some("1"),
        }
    }
}

static CONFIG: OnceLock<RwLock<MlxRuntimeConfig>> = OnceLock::new();

fn map() -> &'static RwLock<MlxRuntimeConfig> {
    CONFIG.get_or_init(|| RwLock::new(MlxRuntimeConfig::from_env()))
}

pub fn runtime_config() -> MlxRuntimeConfig {
    map().read().expect("mlx config").clone()
}

pub fn reload_runtime_config() {
    *map().write().expect("mlx config") = MlxRuntimeConfig::from_env();
}

pub fn install_runtime_config(cfg: MlxRuntimeConfig) {
    *map().write().expect("mlx config") = cfg;
}
