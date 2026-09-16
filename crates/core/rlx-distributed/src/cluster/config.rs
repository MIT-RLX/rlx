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
//! policy = "auto"          # auto | ram_balanced | throughput | manual
//!
//! # Nodes can be discovered instead of listed:
//! # [discover]
//! # ssh_hosts = ["node-a", "node-b"]
//! # ckpt_dir  = "/data/models/GLM-5.3-Flash"
//!
//! [[node]]
//! addr = "127.0.0.1:9100"
//! ssh = "node-a"                     # ~/.ssh/config alias — for probe / launch / weight sync
//! ckpt_dir = "/path/to/DeepSeek-V4-Flash-2bit-DQ"
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
    /// **Everything the probe knows.** Capacity is the better of "experts
    /// resident in RAM/VRAM" and "dense weights resident, expert banks on
    /// disk"; the share each node gets is proportional to how fast it can run
    /// one layer, counting compute *and* the disk time to stream in the experts
    /// that layer touches.
    ///
    /// This is the policy to use for a fine-grained MoE, where RAM alone is the
    /// wrong question: the banks are most of the model, only `top_k` of them are
    /// read per token, and whether a node is a good host depends on its NVMe as
    /// much as its GPU.
    Auto,
    /// Use each node's explicit `layers` range verbatim.
    Manual,
}

/// What a node is doing right now.
///
/// A cluster that runs for days needs machines to come and go without a
/// restart, so a node's *presence* and its *participation* are separate things.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum NodeRole {
    /// Carries a stage and serves traffic.
    #[default]
    Active,
    /// Loaded with the **same layers** as the node it backs, kept warm, serving
    /// nothing. Promoting it is a pointer swap rather than a cold weight load —
    /// which for a 90 GB stage is the difference between seconds and minutes.
    Standby,
    /// Being retired. Keeps its weights until traffic has moved off, then leaves
    /// the plan. `Cluster::retire` puts a node here.
    Draining,
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
    #[serde(default)]
    pub device: DeviceList,
    /// Numeric precision string. Float widths: `f32` | `f16` | `bf16` | `mixed`.
    /// Scaled-GEMM shorthands: `fp8` (e4m3) | `fp8e5m2` | `mxfp8` | `nvfp4` |
    /// `mxfp4`. Or ANY minifloat by name — `f8e4m3`, `f6e3m2`, `f4e2m1`,
    /// `f4e3m0`, … (the `fNeXmY` family). The model crate maps it to compile
    /// flags; exotic formats change numerics and want backend/tensor-core support.
    #[serde(default)]
    pub precision: Precision,
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
    /// Active / standby / draining. See [`NodeRole`].
    #[serde(default)]
    pub role: NodeRole,
    /// For a standby: the `addr` of the active node it mirrors. When unset, a
    /// standby backs whichever active stage is largest — the one whose loss
    /// would cost the most to re-plan.
    #[serde(default)]
    pub standby_for: Option<String>,
}

impl NodeConfig {
    /// Device preference list (primary first). CPU spill devices follow.
    ///
    /// Infallible: the string was parsed and validated when the config loaded.
    pub fn devices(&self) -> &[Device] {
        self.device.as_slice()
    }
    /// Primary compute device (first in the list).
    pub fn primary_device(&self) -> Device {
        self.device.primary()
    }
    /// True if the node lists a CPU/host spill target after its primary.
    pub fn cpu_offload(&self) -> bool {
        self.device.has_cpu_spill()
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
    /// Find machines instead of listing them. See
    /// [`super::discover::DiscoverSection`]; discovered nodes are appended to
    /// any written by hand.
    #[serde(default)]
    pub discover: super::discover::DiscoverSection,
    /// The nodes, in pipeline order. May be empty when `[discover]` is set.
    #[serde(rename = "node", default)]
    pub nodes: Vec<NodeConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlacementSection {
    #[serde(default)]
    pub policy: PlacementPolicy,
    #[serde(default)]
    pub objective: Objective,
}

/// What the planner is trying to make small.
///
/// Distinct from [`PlacementPolicy`], which says *how* to split. Two clusters
/// can both be RAM-balanced and want opposite plans: a chat endpoint wants the
/// fewest hops it can get away with, a batch job wants every stage the same
/// length so the pipeline never stalls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Optimize {
    /// **Time to first token.** Every extra stage is a network hop on the
    /// critical path, so this prefers to use as FEW nodes as will hold the
    /// model, packing each one full before reaching for the next.
    Latency,
    /// **Tokens per second.** With the pipeline full, throughput is set by the
    /// slowest stage, so spread work until stage times match — extra hops cost
    /// nothing once they overlap.
    #[default]
    Throughput,
    /// **Headroom.** Spread as thinly as possible across every node, leaving the
    /// most free RAM per machine. For sharing a cluster with other work.
    Memory,
}

/// Objective plus the constraints that shape it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objective {
    #[serde(default)]
    pub optimize: Optimize,
    /// Context length the KV cache must hold. The planner reserves KV bytes per
    /// node *before* fitting weights — a plan that fits at seq 8 and OOMs at 128 k
    /// is not a plan.
    #[serde(default = "default_context")]
    pub context: usize,
    /// Seconds a network hop adds to the critical path. Only [`Optimize::Latency`]
    /// uses it; the default is a LAN round trip for a hidden-state relay.
    #[serde(default = "default_hop_secs")]
    pub hop_secs: f64,
    /// Cache bytes per layer per token, when you know it and the checkpoint
    /// cannot be read for it. Overrides inference.
    #[serde(default)]
    pub kv_bytes_per_layer_token: Option<u64>,
    /// What to do when the cache size cannot be determined.
    #[serde(default)]
    pub on_unknown_kv: UnknownKv,
    /// Cache element dtype at plan time (`f32` default, `f16`/`bf16`, `f8`).
    ///
    /// Mirrors `CostOptions::kv_dtype`, which applies when the profile is
    /// *inferred*; this one applies when the planner has to synthesize a bound
    /// itself. Defaulting to `f32` keeps that bound the conservative one.
    #[serde(default)]
    pub kv_dtype: Option<String>,
    /// Precisions to try, widest first. The planner walks down until the model
    /// fits and reports what it settled on, instead of failing with "too big"
    /// and leaving you to guess the next step.
    #[serde(default)]
    pub precision_ladder: Vec<Precision>,
}

impl Default for Objective {
    fn default() -> Self {
        Self {
            optimize: Optimize::default(),
            context: default_context(),
            hop_secs: default_hop_secs(),
            kv_bytes_per_layer_token: None,
            on_unknown_kv: UnknownKv::default(),
            kv_dtype: None,
            precision_ladder: Vec::new(),
        }
    }
}

/// What the planner does when the KV cache size cannot be worked out.
///
/// The default is to stop. Reserving nothing is the tempting choice and the
/// wrong one: the plan looks fine, every stage fits, and the node dies when the
/// cache fills — at which point the failure is a long way from its cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum UnknownKv {
    /// Refuse to plan, and say how to resolve it.
    #[default]
    Fail,
    /// Reserve a deliberately pessimistic bound: the MHA worst case of
    /// `2 × hidden × elem` bytes per layer per token, as if every layer cached
    /// full keys and values at model width. Real caches are smaller — GQA by the
    /// head ratio, MLA by ~64× — so a plan that fits under this fits in practice.
    ///
    /// Requires the model's width. Without it there is no bound to be
    /// conservative *about*, so this errors rather than inventing one — see
    /// [`super::placement::resolve_kv`].
    Assume,
    /// Reserve nothing and carry on. Only when you have sized the cache
    /// yourself, or there is no cache at all.
    Ignore,
}

fn default_context() -> usize {
    4096
}
fn default_hop_secs() -> f64 {
    0.002
}

/// A numeric precision, validated at config load.
///
/// Kept as a newtype over the name rather than an enum because the set is open
/// by design: any `fNeXmY` minifloat is legal, and the model crate maps the name
/// to compile flags. What a type buys here is that a typo fails when the config
/// is *read*, naming the file, instead of surfacing as a puzzling compile flag
/// on node three — and that [`Self::bits`] lives with the thing it describes.
///
/// Serialized as the plain string, so existing configs are unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Precision(String);

impl Precision {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Bits per weight, for scaling a cost model between precisions.
    ///
    /// Understands the `fNeXmY` family as well as the named shorthands, so a
    /// precision ladder can step onto `f6e3m2` and have it actually cost
    /// something — the previous lookup table silently ignored anything it did
    /// not list by name.
    pub fn bits(&self) -> Option<f64> {
        Some(match self.0.as_str() {
            "f32" => 32.0,
            "f16" | "bf16" | "mixed" => 16.0,
            "fp8" | "fp8e5m2" | "mxfp8" => 8.0,
            // Int8 with per-block scales costs slightly more than 8 bits.
            "int8" | "q8_0" => 8.5,
            // 4-bit blocks likewise carry a shared exponent per group.
            "nvfp4" | "mxfp4" => 4.25,
            other => return minifloat_bits(other),
        })
    }
}

/// Width of an `fNeXmY` minifloat name (`f8e4m3` → 8), or a bare `fN`.
fn minifloat_bits(name: &str) -> Option<f64> {
    let rest = name.strip_prefix('f')?;
    let width: String = rest.chars().take_while(char::is_ascii_digit).collect();
    if width.is_empty() {
        return None;
    }
    let bits: u32 = width.parse().ok()?;
    let tail = &rest[width.len()..];
    if tail.is_empty() {
        // A bare `fN` is only a float width for the widths that exist.
        return matches!(bits, 8 | 16 | 32 | 64).then_some(bits as f64);
    }
    // `eXmY`: exponent and mantissa must both be present and numeric.
    let em = tail.strip_prefix('e')?;
    let (e, m) = em.split_once('m')?;
    if e.is_empty() || m.is_empty() || !e.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    m.chars().all(|c| c.is_ascii_digit()).then_some(bits as f64)
}

impl std::str::FromStr for Precision {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let p = Self(s.trim().to_string());
        if p.bits().is_none() {
            anyhow::bail!(
                "unknown precision `{s}`; expected f32 | f16 | bf16 | mixed | fp8 | \
                 mxfp8 | mxfp4 | nvfp4 | int8, or an fNeXmY minifloat such as f8e4m3"
            );
        }
        Ok(p)
    }
}

impl TryFrom<String> for Precision {
    type Error = anyhow::Error;
    fn try_from(s: String) -> Result<Self> {
        s.parse()
    }
}
impl From<Precision> for String {
    fn from(p: Precision) -> String {
        p.0
    }
}
impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl Default for Precision {
    fn default() -> Self {
        Self("bf16".into())
    }
}

/// A node's device preference, primary first, with CPU/host spill after.
///
/// Parsed once at config load, so `"cuda+cpu"` is validated there rather than
/// at every use — which also lets [`NodeConfig::primary_device`] be infallible
/// instead of silently defaulting to CPU when the string was malformed.
///
/// Serialized as the original string, so existing configs round-trip verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct DeviceList {
    raw: String,
    devices: Vec<Device>,
}

impl DeviceList {
    /// Primary compute device (first listed).
    pub fn primary(&self) -> Device {
        self.devices.first().copied().unwrap_or(Device::Cpu)
    }
    pub fn as_slice(&self) -> &[Device] {
        &self.devices
    }
    pub fn as_str(&self) -> &str {
        &self.raw
    }
    /// True if a CPU/host spill target follows the primary.
    pub fn has_cpu_spill(&self) -> bool {
        self.devices.len() > 1 && self.devices.contains(&Device::Cpu)
    }
}

impl std::str::FromStr for DeviceList {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let raw = s.trim().to_string();
        let devices = parse_device_list(&raw.replace('+', ","))
            .map_err(|e| anyhow::anyhow!("bad device `{raw}`: {e:?}"))?;
        if devices.is_empty() {
            anyhow::bail!("device list `{raw}` is empty");
        }
        Ok(Self { raw, devices })
    }
}

impl TryFrom<String> for DeviceList {
    type Error = anyhow::Error;
    fn try_from(s: String) -> Result<Self> {
        s.parse()
    }
}
impl From<DeviceList> for String {
    fn from(d: DeviceList) -> String {
        d.raw
    }
}
impl std::fmt::Display for DeviceList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.raw)
    }
}
impl Default for DeviceList {
    fn default() -> Self {
        Self {
            raw: "cpu".into(),
            devices: vec![Device::Cpu],
        }
    }
}

fn default_seq() -> usize {
    8
}
fn default_reserve() -> f64 {
    6.0
}

impl ClusterConfig {
    pub fn from_toml_str(s: &str) -> Result<Self> {
        // Inline the cause rather than `.context(..)`: a validation failure from
        // `DeviceList` / `Precision` is the whole reason those types exist, and
        // `to_string()` on a context-wrapped anyhow error shows only the outer
        // layer — leaving the caller with "parse cluster TOML" and no idea which
        // field was wrong.
        toml::from_str(s).map_err(|e| anyhow::anyhow!("parse cluster TOML: {e}"))
    }
    pub fn from_path(p: impl AsRef<Path>) -> Result<Self> {
        let s = std::fs::read_to_string(p.as_ref())
            .with_context(|| format!("read {}", p.as_ref().display()))?;
        Self::from_toml_str(&s)
    }
    /// Effective RNG seed for node `i` (per-node override, else global, else 0).
    pub fn seed_for(&self, i: usize) -> u64 {
        self.nodes
            .get(i)
            .and_then(|n| n.rng_seed)
            .or(self.rng_seed)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-listed and discovered nodes must coexist: pin the machines you care
    /// about, let the rest be found. Hand-written entries come first, so they
    /// keep the head of the pipeline (the embedding stage).
    #[test]
    fn discover_and_hand_listed_nodes_coexist() {
        let toml = r#"
model = "some/model"

[placement]
policy = "auto"

[discover]
ssh_hosts = ["node-b", "node-c"]
ckpt_dir  = "/data/model"
port      = 9100

[[node]]
addr = "10.0.0.5:9100"
ckpt_dir = "/pinned/model"
device = "cuda"
max_ram_gb = 40.0
"#;
        let cfg = ClusterConfig::from_toml_str(toml).unwrap();
        assert_eq!(cfg.placement.policy, PlacementPolicy::Auto);
        // The hand-written node survives verbatim...
        assert_eq!(cfg.nodes.len(), 1);
        assert_eq!(cfg.nodes[0].addr, "10.0.0.5:9100");
        assert_eq!(cfg.nodes[0].max_ram_gb, Some(40.0));
        // ...and discovery is configured alongside it.
        assert!(!cfg.discover.is_empty());
        assert_eq!(cfg.discover.ssh_hosts, ["node-b", "node-c"]);
        assert_eq!(cfg.discover.ckpt_dir.as_deref(), Some("/data/model"));
        assert_eq!(cfg.discover.port, 9100);
    }

    /// A config with ONLY `[discover]` is legal — `nodes` defaults to empty.
    #[test]
    fn discover_only_config_needs_no_node_blocks() {
        let cfg = ClusterConfig::from_toml_str(
            r#"
model = "m"
[discover]
ssh_hosts = ["a"]
ckpt_dir = "/m"
"#,
        )
        .unwrap();
        assert!(cfg.nodes.is_empty());
        assert!(!cfg.discover.is_empty());
    }

    /// And an all-manual config still parses with no `[discover]` at all, so
    /// existing cluster files keep working unchanged.
    #[test]
    fn hand_listed_only_config_is_unchanged() {
        let cfg = ClusterConfig::from_toml_str(
            r#"
model = "m"
[[node]]
addr = "a:1"
ckpt_dir = "/m"
"#,
        )
        .unwrap();
        assert_eq!(cfg.nodes.len(), 1);
        assert!(cfg.discover.is_empty());
        assert_eq!(cfg.placement.policy, PlacementPolicy::RamBalanced);
    }

    /// Existing configs must be byte-identical on the wire — the types are a
    /// load-time check, not a format change.
    #[test]
    fn device_and_precision_still_serialize_as_plain_strings() {
        let cfg = ClusterConfig::from_toml_str(
            r#"
model = "m"
[[node]]
addr = "a:1"
ckpt_dir = "/m"
device = "cuda+cpu"
precision = "f8e4m3"
"#,
        )
        .unwrap();
        assert_eq!(cfg.nodes[0].device.as_str(), "cuda+cpu");
        assert_eq!(cfg.nodes[0].precision.as_str(), "f8e4m3");
        // Round-trips verbatim, `+` separator and all.
        let back = toml::to_string(&cfg).unwrap();
        assert!(back.contains(r#"device = "cuda+cpu""#), "{back}");
        assert!(back.contains(r#"precision = "f8e4m3""#), "{back}");
    }

    /// The point of parsing at load: a typo names the config, not a compile flag
    /// on node three.
    #[test]
    fn bad_device_and_precision_fail_at_load() {
        let bad_dev = ClusterConfig::from_toml_str(
            r#"model="m"
[[node]]
addr="a:1"
ckpt_dir="/m"
device="quantum"
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(bad_dev.contains("quantum"), "{bad_dev}");

        let bad_prec = ClusterConfig::from_toml_str(
            r#"model="m"
[[node]]
addr="a:1"
ckpt_dir="/m"
precision="f7ish"
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(bad_prec.contains("unknown precision"), "{bad_prec}");
    }

    /// Device access is infallible after load, and `+` means spill.
    #[test]
    fn device_list_exposes_primary_and_spill() {
        let d: DeviceList = "cuda+cpu".parse().unwrap();
        assert_eq!(d.primary(), Device::Cuda);
        assert!(d.has_cpu_spill());
        let solo: DeviceList = "metal".parse().unwrap();
        assert_eq!(solo.primary(), Device::Metal);
        assert!(!solo.has_cpu_spill());
        assert_eq!(DeviceList::default().primary(), Device::Cpu);
    }

    /// The old lookup table silently ignored anything it did not list, so a
    /// ladder step onto a minifloat did nothing. The family is now parsed.
    #[test]
    fn minifloat_widths_are_parsed_not_looked_up() {
        for (name, bits) in [
            ("f8e4m3", 8.0),
            ("f8e5m2", 8.0),
            ("f6e3m2", 6.0),
            ("f4e2m1", 4.0),
            ("f4e3m0", 4.0),
        ] {
            let p: Precision = name.parse().unwrap();
            assert_eq!(p.bits(), Some(bits), "{name}");
        }
        // Named shorthands still work, block formats keep their overhead.
        assert_eq!(Precision::default().bits(), Some(16.0));
        assert_eq!("f32".parse::<Precision>().unwrap().bits(), Some(32.0));
        assert_eq!("mxfp4".parse::<Precision>().unwrap().bits(), Some(4.25));
        assert_eq!("int8".parse::<Precision>().unwrap().bits(), Some(8.5));
    }

    /// Near-misses must be rejected rather than read as a width.
    #[test]
    fn malformed_minifloats_are_rejected() {
        for bad in ["f8e4", "f8m3", "fe4m3", "f8e4mx", "f9", "bf", "f8e4m"] {
            assert!(
                bad.parse::<Precision>().is_err(),
                "`{bad}` should not parse"
            );
        }
    }

    #[test]
    fn policy_names_round_trip() {
        for (name, want) in [
            ("auto", PlacementPolicy::Auto),
            ("ram_balanced", PlacementPolicy::RamBalanced),
            ("throughput", PlacementPolicy::Throughput),
            ("manual", PlacementPolicy::Manual),
        ] {
            let cfg = ClusterConfig::from_toml_str(&format!(
                "model = \"m\"\n[placement]\npolicy = \"{name}\"\n[[node]]\naddr=\"a:1\"\nckpt_dir=\"/m\"\n"
            ))
            .unwrap();
            assert_eq!(cfg.placement.policy, want, "{name}");
        }
    }
}
