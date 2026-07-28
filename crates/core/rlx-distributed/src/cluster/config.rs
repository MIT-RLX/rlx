// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Declarative cluster config** (TOML) — the DX surface. A coordinator loads
//! one file describing the model, the placement policy, and per-node device /
//! precision / RNG / KV-cache / RAM settings, then probes + plans + runs.
//!
//! ```toml
//! model = "mlx-community/DeepSeek-V4-Flash-2bit-DQ"   # HF id or local dir
//! seq = 6
//! rng_seed = 42            # global; a node may override
//! reserve_ram_gb = 6.0     # headroom left free on every node
//!
//! [placement]
//! policy = "ram_balanced"  # ram_balanced | throughput | manual
//!
//! [[node]]
//! addr = "127.0.0.1:9100"
//! ssh = "macmini"                    # for probe / launch / weight sync
//! ckpt_dir = "/Users/Shared/DeepSeek-V4-Flash-2bit-DQ"
//! device = "cpu"                     # cpu | metal | cuda | ane | vulkan | "metal+cpu"
//! precision = "bf16"                 # f32 | f16 | bf16 | mixed
//! kv_cache = "host"                  # none | host | device
//! max_ram_gb = 44                    # cap (e.g. leave room for other apps)
//! # layers = "0:12"                  # manual override (policy = "manual")
//! ```

use anyhow::{Context, Result};
use rlx_runtime::{Device, parse_device_list};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::path::Path;

/// Where a node keeps the attention KV cache across decode steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum KvPolicy {
    /// No persistent cache (prefill-only / recompute).
    #[default]
    None,
    /// Keep KV in host RAM (works for any device; PCIe copy on GPU).
    Host,
    /// Keep KV resident on the compute device (fastest; device-mem permitting).
    Device,
}

/// How the coordinator assigns layers to nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PlacementPolicy {
    /// Balance so each node's stage fills a similar fraction of its RAM budget
    /// (fits first; the default — a model too big for any one node still runs).
    #[default]
    RamBalanced,
    /// Balance per-stage compute time by node throughput (GFLOP/s) — minimizes
    /// the pipeline's critical path when everything already fits.
    Throughput,
    /// Use each node's explicit `layers` range verbatim.
    Manual,
}

/// Per-node settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// `host:port` the coordinator connects to (and the worker binds).
    pub addr: String,
    /// SSH alias/host for probing, launching, and syncing weights (None = local).
    #[serde(default)]
    pub ssh: Option<String>,
    /// Local checkpoint directory ON THAT NODE.
    pub ckpt_dir: String,
    /// Device string (`cpu`, `metal`, `cuda`, `ane`, `vulkan`, or `a+b` combo —
    /// first is primary, rest are CPU/host spill for weights that don't fit).
    #[serde(default = "default_device")]
    pub device: String,
    /// Numeric precision string. Float widths: `f32` | `f16` | `bf16` | `mixed`.
    /// Scaled-GEMM shorthands: `fp8` (e4m3) | `fp8e5m2` | `mxfp8` | `nvfp4` |
    /// `mxfp4`. Or ANY minifloat by name — `f8e4m3`, `f6e3m2`, `f4e2m1`,
    /// `f4e3m0`, … (the `fNeXmY` family). The model crate maps it to compile
    /// flags; exotic formats change numerics and want backend/tensor-core support.
    #[serde(default = "default_precision")]
    pub precision: String,
    #[serde(default)]
    pub kv_cache: KvPolicy,
    /// Per-node RNG seed (overrides the global). Keeps sampling reproducible and
    /// *distinct* per stage when desired.
    #[serde(default)]
    pub rng_seed: Option<u64>,
    /// Hard cap on this node's stage RAM (GB). The planner never exceeds it.
    #[serde(default)]
    pub max_ram_gb: Option<f64>,
    /// Manual `"a:b"` layer range (used when policy = manual, or to pin a node).
    #[serde(default)]
    pub layers: Option<String>,
}

fn default_device() -> String {
    "cpu".into()
}
fn default_precision() -> String {
    "bf16".into()
}

impl NodeConfig {
    /// Parsed device preference list (primary first). CPU spill devices follow.
    pub fn devices(&self) -> Result<Vec<Device>> {
        parse_device_list(&self.device.replace('+', ",")).map_err(|e| anyhow::anyhow!("node {}: bad device `{}`: {e:?}", self.addr, self.device))
    }
    /// Primary compute device (first in the list).
    pub fn primary_device(&self) -> Device {
        self.devices().ok().and_then(|d| d.into_iter().next()).unwrap_or(Device::Cpu)
    }
    /// True if the node lists a CPU/host spill target after its primary.
    pub fn cpu_offload(&self) -> bool {
        self.devices().map(|d| d.len() > 1 && d.iter().any(|x| *x == Device::Cpu)).unwrap_or(false)
    }
    /// Manual layer range if given.
    pub fn manual_range(&self) -> Option<Range<usize>> {
        let s = self.layers.as_ref()?;
        let (a, b) = s.split_once(':')?;
        Some(a.trim().parse().ok()?..b.trim().parse().ok()?)
    }
}

/// Whole-cluster config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// HF repo id or a local directory holding the checkpoint.
    pub model: String,
    /// Prompt length the pipeline graphs are built for.
    #[serde(default = "default_seq")]
    pub seq: usize,
    /// Global RNG seed (a node's `rng_seed` overrides it).
    #[serde(default)]
    pub rng_seed: Option<u64>,
    /// RAM (GB) to leave free on every node beyond its stage.
    #[serde(default = "default_reserve")]
    pub reserve_ram_gb: f64,
    #[serde(default)]
    pub placement: PlacementSection,
    /// The nodes, in pipeline order.
    #[serde(rename = "node")]
    pub nodes: Vec<NodeConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlacementSection {
    #[serde(default)]
    pub policy: PlacementPolicy,
}

fn default_seq() -> usize {
    8
}
fn default_reserve() -> f64 {
    6.0
}

impl ClusterConfig {
    pub fn from_toml_str(s: &str) -> Result<Self> {
        toml::from_str(s).context("parse cluster TOML")
    }
    pub fn from_path(p: impl AsRef<Path>) -> Result<Self> {
        let s = std::fs::read_to_string(p.as_ref()).with_context(|| format!("read {}", p.as_ref().display()))?;
        Self::from_toml_str(&s)
    }
    /// Effective RNG seed for node `i` (per-node override, else global, else 0).
    pub fn seed_for(&self, i: usize) -> u64 {
        self.nodes.get(i).and_then(|n| n.rng_seed).or(self.rng_seed).unwrap_or(0)
    }
}
