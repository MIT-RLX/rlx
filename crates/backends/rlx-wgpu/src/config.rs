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

//! Typed wgpu runtime options loaded once from the `RLX_*` registry.

use std::sync::{OnceLock, RwLock};

/// wgpu backend options (conv thresholds, dumps, coop-matrix).
#[derive(Debug, Clone)]
pub struct WgpuRuntimeConfig {
    pub gdn_host: bool,
    pub im2col_min_spatial: usize,
    pub im2col_min_k: usize,
    pub im2col_min_cout: usize,
    pub tiled_min_spatial: usize,
    pub dump_nodes: bool,
    pub dump_nodes_limit: usize,
    pub dump_tail: bool,
    pub dump_inputs: bool,
    pub schedule: bool,
    pub large_buffers: bool,
    pub print_limits: bool,
    pub matmul_f32_only: bool,
    pub f16_weights: bool,
    pub no_tiled_conv: bool,
    pub conv_im2col: bool,
    pub debug: bool,
}

impl Default for WgpuRuntimeConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl WgpuRuntimeConfig {
    pub fn from_env() -> Self {
        use rlx_ir::env_registry as reg;
        // Defaults match `backend/mod.rs` constants when unset.
        Self {
            gdn_host: reg::flag("RLX_WGPU_GDN_HOST"),
            im2col_min_spatial: reg::parse_or("RLX_WGPU_IM2COL_MIN_SPATIAL", 2048usize),
            im2col_min_k: reg::parse_or("RLX_WGPU_IM2COL_MIN_K", 256usize),
            im2col_min_cout: reg::parse_or("RLX_WGPU_IM2COL_MIN_COUT", 64usize),
            tiled_min_spatial: reg::parse_or("RLX_WGPU_TILED_MIN_SPATIAL", 256usize),
            dump_nodes: reg::flag("RLX_WGPU_DUMP_NODES"),
            dump_nodes_limit: reg::parse_or("RLX_WGPU_DUMP_NODES_LIMIT", 40usize),
            dump_tail: reg::flag("RLX_WGPU_DUMP_TAIL"),
            dump_inputs: reg::flag("RLX_WGPU_DUMP_INPUTS"),
            schedule: reg::flag("RLX_WGPU_SCHEDULE"),
            large_buffers: reg::flag("RLX_WGPU_LARGE_BUFFERS"),
            print_limits: reg::flag("RLX_WGPU_PRINT_LIMITS"),
            matmul_f32_only: reg::flag("RLX_WGPU_MATMUL_F32_ONLY"),
            f16_weights: reg::flag("RLX_WGPU_F16_WEIGHTS"),
            no_tiled_conv: reg::flag("RLX_WGPU_NO_TILED_CONV"),
            conv_im2col: reg::flag("RLX_WGPU_CONV_IM2COL"),
            debug: reg::flag("RLX_WGPU_DEBUG"),
        }
    }
}

static CONFIG: OnceLock<RwLock<WgpuRuntimeConfig>> = OnceLock::new();

fn map() -> &'static RwLock<WgpuRuntimeConfig> {
    CONFIG.get_or_init(|| RwLock::new(WgpuRuntimeConfig::from_env()))
}

pub fn runtime_config() -> WgpuRuntimeConfig {
    map().read().expect("wgpu config").clone()
}

pub fn reload_runtime_config() {
    *map().write().expect("wgpu config") = WgpuRuntimeConfig::from_env();
}

pub fn install_runtime_config(cfg: WgpuRuntimeConfig) {
    *map().write().expect("wgpu config") = cfg;
}
