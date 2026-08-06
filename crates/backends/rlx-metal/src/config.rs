// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed Metal runtime options loaded once from the `RLX_*` registry.

use std::sync::{OnceLock, RwLock};

/// Metal backend options (MPSGraph / GGUF / host-fallback routing).
#[derive(Debug, Clone)]
pub struct MetalRuntimeConfig {
    pub disable_mpsgraph: bool,
    pub disable_mpsgraph_executable: bool,
    pub disable_mpsgraph_hybrid: bool,
    pub mpsgraph_force: bool,
    pub mpsgraph_min_flops: u64,
    pub mpsgraph_trace: bool,
    pub mpsgraph_param_const: bool,
    pub mps_fp16: bool,
    pub dequant_gpu_disable: bool,
    pub dequant_matmul_legacy: bool,
    pub metal_debug: bool,
    pub metal_trace: bool,
    pub host_fallback: bool,
    pub gdn_host_fallback: bool,
    pub fft_host_fallback: bool,
    pub lstm_cpu: bool,
    pub rnn_host_fallback: bool,
    pub ssm_cpu: bool,
    pub sample_host: bool,
    pub concat_host: bool,
    pub fuse_decode: bool,
    pub fuse_decode_log: bool,
    pub sdpa_flash_decode: Option<bool>,
    pub sdpa_flash_partitions: Option<u32>,
    pub sdpa_tune_cache_path: Option<String>,
    pub sdpa_tune_cache_load: bool,
    pub sdpa_tune_cache_persist: bool,
    pub sdpa_tune_cache_max_entries: usize,
    pub sdpa_tune_cache_eviction: crate::kernel_plan::TuneCacheEviction,
}

impl Default for MetalRuntimeConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl MetalRuntimeConfig {
    pub fn from_env() -> Self {
        use rlx_ir::env_registry as reg;
        Self {
            disable_mpsgraph: reg::flag("RLX_DISABLE_MPSGRAPH"),
            disable_mpsgraph_executable: reg::flag("RLX_DISABLE_MPSGRAPH_EXECUTABLE"),
            disable_mpsgraph_hybrid: !rlx_ir::env::is_unset("RLX_DISABLE_MPSGRAPH_HYBRID"),
            mpsgraph_force: reg::flag("RLX_MPSGRAPH_FORCE"),
            mpsgraph_min_flops: reg::parse_or("RLX_MPSGRAPH_MIN_FLOPS", 0u64),
            mpsgraph_trace: reg::flag("RLX_MPSGRAPH_TRACE"),
            mpsgraph_param_const: reg::flag("RLX_MPSGRAPH_PARAM_CONST"),
            mps_fp16: reg::flag("RLX_MPS_FP16"),
            // Alias-aware: also honors RLX_DISABLE_METAL_DEQUANT_GPU
            dequant_gpu_disable: reg::flag("RLX_METAL_DEQUANT_GPU_DISABLE"),
            dequant_matmul_legacy: reg::flag("RLX_METAL_DEQUANT_MATMUL_LEGACY"),
            metal_debug: reg::flag("RLX_METAL_DEBUG"),
            metal_trace: reg::flag("RLX_METAL_TRACE"),
            host_fallback: reg::flag("RLX_METAL_HOST_FALLBACK"),
            gdn_host_fallback: reg::flag("RLX_METAL_GDN_HOST_FALLBACK")
                || reg::flag("RLX_METAL_GDN_CPU"),
            fft_host_fallback: reg::flag("RLX_METAL_FFT_HOST_FALLBACK"),
            lstm_cpu: reg::flag("RLX_METAL_LSTM_HOST_FALLBACK") || reg::flag("RLX_METAL_LSTM_CPU"),
            rnn_host_fallback: reg::flag("RLX_METAL_RNN_HOST_FALLBACK"),
            ssm_cpu: reg::flag("RLX_METAL_SSM_HOST_FALLBACK") || reg::flag("RLX_METAL_SSM_CPU"),
            sample_host: reg::flag("RLX_METAL_SAMPLE_HOST"),
            concat_host: reg::flag("RLX_METAL_CONCAT_HOST"),
            fuse_decode: reg::var("RLX_METAL_FUSE_DECODE").as_deref() != Some("0"),
            fuse_decode_log: reg::flag("RLX_METAL_FUSE_DECODE_LOG"),
            sdpa_flash_decode: reg::var("RLX_METAL_SDPA_FLASH_DECODE").map(|v| v != "0"),
            sdpa_flash_partitions: reg::var("RLX_METAL_SDPA_FLASH_P")
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|&p| p > 0),
            sdpa_tune_cache_path: reg::var("RLX_METAL_SDPA_TUNE_CACHE"),
            sdpa_tune_cache_load: reg::var("RLX_METAL_SDPA_TUNE_CACHE_LOAD")
                .map(|v| v != "0")
                .unwrap_or(true),
            sdpa_tune_cache_persist: reg::var("RLX_METAL_SDPA_TUNE_CACHE_PERSIST")
                .map(|v| v != "0")
                .unwrap_or(true),
            sdpa_tune_cache_max_entries: reg::var("RLX_METAL_SDPA_TUNE_CACHE_MAX_ENTRIES")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(256)
                .max(1),
            sdpa_tune_cache_eviction: parse_sdpa_cache_eviction(reg::var(
                "RLX_METAL_SDPA_TUNE_CACHE_EVICTION",
            )),
        }
    }

    pub fn with_sdpa_flash(mut self, enabled: Option<bool>, partitions: Option<u32>) -> Self {
        self.sdpa_flash_decode = enabled;
        self.sdpa_flash_partitions = partitions.filter(|&p| p > 0);
        self
    }

    pub fn with_sdpa_tune_cache(
        mut self,
        path: Option<String>,
        load: bool,
        persist: bool,
        max_entries: usize,
    ) -> Self {
        self.sdpa_tune_cache_path = path;
        self.sdpa_tune_cache_load = load;
        self.sdpa_tune_cache_persist = persist;
        self.sdpa_tune_cache_max_entries = max_entries.max(1);
        self
    }

    pub fn with_sdpa_tune_cache_eviction(
        mut self,
        eviction: crate::kernel_plan::TuneCacheEviction,
    ) -> Self {
        self.sdpa_tune_cache_eviction = eviction;
        self
    }
}

fn parse_sdpa_cache_eviction(v: Option<String>) -> crate::kernel_plan::TuneCacheEviction {
    match v.as_deref() {
        Some("keep-high") | Some("keep_high") | Some("high") => {
            crate::kernel_plan::TuneCacheEviction::KeepHighBuckets
        }
        Some("lru") => crate::kernel_plan::TuneCacheEviction::Lru,
        _ => crate::kernel_plan::TuneCacheEviction::KeepLowBuckets,
    }
}

static CONFIG: OnceLock<RwLock<MetalRuntimeConfig>> = OnceLock::new();

fn map() -> &'static RwLock<MetalRuntimeConfig> {
    CONFIG.get_or_init(|| RwLock::new(MetalRuntimeConfig::from_env()))
}

pub fn runtime_config() -> MetalRuntimeConfig {
    map().read().expect("metal config").clone()
}

pub fn reload_runtime_config() {
    *map().write().expect("metal config") = MetalRuntimeConfig::from_env();
}

pub fn install_runtime_config(cfg: MetalRuntimeConfig) {
    *map().write().expect("metal config") = cfg;
}
