// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Finding the machines** — turn a short `[discover]` stanza into the
//! `[[node]]` list you would otherwise write by hand.
//!
//! Hand-written node blocks are the part of a cluster config that goes stale:
//! every host needs an `addr`, an `ssh` alias, a `ckpt_dir` and a `device`, and
//! the device is the one you are most likely to get wrong — you have to know
//! whether that box ended up with CUDA or just Vulkan. Discovery asks each
//! machine instead.
//!
//! Two ways in, both dependency-free:
//!
//! ```toml
//! [discover]
//! ssh_hosts = ["node-a", "node-b", "node-c"]   # ~/.ssh/config aliases
//! ckpt_dir  = "/data/models/GLM-5.3-Flash"     # same path on each host
//! port      = 9100
//! ```
//!
//! ```toml
//! [discover]
//! subnet = "10.0.0.0/24"    # TCP-scan for workers already listening
//! port   = 9100
//! ```
//!
//! `ssh_hosts` is the one to reach for: it can probe hardware (over
//! [`super::caps::probe_remote`]) and so fill in the device and RAM. A subnet
//! scan only learns that something is listening, so it yields nodes with default
//! settings that a later probe fills in.
//!
//! Discovery is additive: any hand-written `[[node]]` entries are kept, and a
//! discovered host that duplicates one is dropped rather than added twice — so
//! you can pin one machine and let the rest be found.

use super::caps::probe_remote;
use super::config::NodeConfig;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream};
use std::time::Duration;

/// The `[discover]` section.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscoverSection {
    /// SSH aliases/hosts to probe. Preferred: yields real hardware info.
    #[serde(default)]
    pub ssh_hosts: Vec<String>,
    /// CIDR to TCP-scan for already-running workers, e.g. `"10.0.0.0/24"`.
    #[serde(default)]
    pub subnet: Option<String>,
    /// Worker port (both modes).
    #[serde(default = "default_port")]
    pub port: u16,
    /// Checkpoint directory, as it appears **on the discovered hosts**.
    #[serde(default)]
    pub ckpt_dir: Option<String>,
    /// Force a device instead of picking the best each host reports.
    #[serde(default)]
    pub device: Option<super::config::DeviceList>,
    /// Precision for discovered nodes.
    #[serde(default)]
    pub precision: Option<super::config::Precision>,
    /// Per-host TCP connect timeout for the subnet scan, milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_port() -> u16 {
    9100
}
fn default_timeout_ms() -> u64 {
    300
}

impl DiscoverSection {
    pub fn is_empty(&self) -> bool {
        self.ssh_hosts.is_empty() && self.subnet.is_none()
    }
}

/// Pick the device a host should run, from what it reported.
///
/// Preference is accelerator-before-CPU, and among accelerators the one with the
/// most usable memory — with Apple's unified Metal treated as "as much as RAM",
/// since it has no separate ceiling. Ties fall back to the probe's own ordering,
/// which lists the primary accelerator first.
///
/// Returns the choice **and any device labels this binary did not recognise**.
/// Those are reported rather than dropped: a node running a different rlx build
/// can name a device this coordinator has never heard of, and silently planning
/// its GPU box as a CPU stage is the kind of 10× slowdown nobody thinks to look
/// for.
fn best_device(caps: &super::caps::NodeCaps) -> (super::config::DeviceList, Vec<String>) {
    use rlx_runtime::Device;
    let mut best: Option<(Device, u64)> = None;
    let mut skipped = Vec::new();
    for d in &caps.devices {
        let Some(kind) = d.parsed() else {
            skipped.push(format!("{} (unrecognised by this build)", d.device));
            continue;
        };
        // Present on the node but not compiled into the node's binary: choosing
        // it would hand the stage a device that panics on first compile.
        if !d.available {
            skipped.push(format!("{} (node's build cannot target it)", d.device));
            continue;
        }
        // The Neural Engine is not a general compute target for a pipeline stage.
        if matches!(kind, Device::Cpu | Device::Ane) {
            continue;
        }
        let mem = if d.mem_bytes > 0 {
            d.mem_bytes
        } else {
            caps.ram_total
        };
        if best.as_ref().is_none_or(|&(_, m)| mem > m) {
            best = Some((kind, mem));
        }
    }
    // Built from the parsed device, so there is no second parse to fail.
    let chosen = best
        .and_then(|(d, _)| rlx_runtime::device_label(d).parse().ok())
        .unwrap_or_default();
    (chosen, skipped)
}

/// Discovered nodes, plus the hosts that would not answer and why.
pub type Discovered = (Vec<NodeConfig>, Vec<(String, String)>);

/// Probe each SSH host and synthesize a [`NodeConfig`].
///
/// `remote_bin` is the worker binary on those hosts (its `--probe` mode
/// self-reports). A host that fails to probe is reported and skipped rather than
/// failing the whole discovery — one unreachable machine should not stop a
/// cluster from coming up on the rest.
pub fn discover_ssh(d: &DiscoverSection, remote_bin: &str) -> Result<Discovered> {
    let ckpt = d
        .ckpt_dir
        .clone()
        .context("[discover] ssh_hosts needs `ckpt_dir` (the path on those hosts)")?;
    let mut out = Vec::new();
    let mut failed = Vec::new();
    for host in &d.ssh_hosts {
        let addr = format!("{host}:{}", d.port);
        match probe_remote(host, remote_bin, &addr, &ckpt) {
            Ok(caps) => {
                let device = d.device.clone().unwrap_or_else(|| {
                    let (chosen, unknown) = best_device(&caps);
                    if !unknown.is_empty() {
                        eprintln!(
                            "discover: {host} has device(s) that cannot be used for a \
                             stage — {} — so they were skipped when choosing its \
                             device (a version mismatch or a missing Cargo feature \
                             on that host would explain it)",
                            unknown.join(", ")
                        );
                    }
                    chosen
                });
                out.push(NodeConfig {
                    addr,
                    ssh: Some(host.clone()),
                    ckpt_dir: ckpt.clone(),
                    device,
                    precision: d.precision.clone().unwrap_or_default(),
                    kv_cache: Default::default(),
                    rng_seed: None,
                    max_ram_gb: None,
                    layers: None,
                    role: Default::default(),
                    standby_for: None,
                });
            }
            Err(e) => failed.push((host.clone(), e.to_string())),
        }
    }
    Ok((out, failed))
}

/// Expand an IPv4 CIDR into its host addresses (network and broadcast excluded).
///
/// Capped at a /16; anything larger is a configuration mistake rather than a
/// scan worth doing serially.
fn cidr_hosts(cidr: &str) -> Result<Vec<Ipv4Addr>> {
    let (base, bits) = cidr
        .split_once('/')
        .context("subnet must be CIDR, e.g. 10.0.0.0/24")?;
    let base: Ipv4Addr = base.parse().context("subnet base address")?;
    let bits: u32 = bits.parse().context("subnet prefix length")?;
    if bits > 32 {
        bail!("subnet prefix /{bits} is not a valid IPv4 prefix");
    }
    if bits < 16 {
        bail!("subnet /{bits} is too large to scan serially; use /16 or smaller");
    }
    let host_bits = 32 - bits;
    let net = u32::from(base) & (!0u32).checked_shl(host_bits).unwrap_or(0);
    let count = 1u32 << host_bits;
    // Skip .0 (network) and the last (broadcast) for prefixes that have them.
    let (lo, hi) = if count > 2 {
        (1, count - 1)
    } else {
        (0, count)
    };
    Ok((lo..hi).map(|i| Ipv4Addr::from(net + i)).collect())
}

/// TCP-scan a subnet for workers already listening on `port`.
///
/// This only learns *that* something answers — it cannot ask what hardware it
/// has, so the resulting nodes carry defaults and want a follow-up
/// [`super::Cluster::probe`]. Prefer `ssh_hosts` when you can.
pub fn discover_subnet(d: &DiscoverSection) -> Result<Vec<NodeConfig>> {
    let Some(cidr) = d.subnet.as_deref() else {
        return Ok(Vec::new());
    };
    let ckpt = d.ckpt_dir.clone().unwrap_or_default();
    let timeout = Duration::from_millis(d.timeout_ms);
    let mut out = Vec::new();
    for ip in cidr_hosts(cidr)? {
        let sa = SocketAddr::new(IpAddr::V4(ip), d.port);
        if TcpStream::connect_timeout(&sa, timeout).is_ok() {
            out.push(NodeConfig {
                addr: sa.to_string(),
                ssh: None,
                ckpt_dir: ckpt.clone(),
                device: d.device.clone().unwrap_or_default(),
                precision: d.precision.clone().unwrap_or_default(),
                kv_cache: Default::default(),
                rng_seed: None,
                max_ram_gb: None,
                layers: None,
                role: Default::default(),
                standby_for: None,
            });
        }
    }
    Ok(out)
}

/// Merge discovered nodes into an existing list, keeping hand-written entries
/// and dropping duplicates by `addr`.
pub fn merge_nodes(existing: &mut Vec<NodeConfig>, found: Vec<NodeConfig>) -> usize {
    let mut added = 0;
    for n in found {
        if existing.iter().any(|e| e.addr == n.addr) {
            continue;
        }
        existing.push(n);
        added += 1;
    }
    added
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::caps::{DeviceInfo, NodeCaps};

    fn caps_with(devices: Vec<DeviceInfo>, ram: u64) -> NodeCaps {
        NodeCaps {
            addr: "x:1".into(),
            os: "linux".into(),
            cores: 8,
            ram_total: ram,
            ram_avail: ram,
            disk_free: 0,
            devices,
            gflops: 1.0,
            io_mbps: 0.0,
        }
    }
    fn dev(label: &str, mem: u64, unified: bool) -> DeviceInfo {
        DeviceInfo {
            device: label.into(),
            name: label.into(),
            mem_bytes: mem,
            unified,
            gflops: 0.0,
            available: true,
        }
    }

    /// A GPU the node's own build cannot target must not be chosen for its
    /// stage: `Session::new` on it panics, so planning around it hands that node
    /// a device it will die on. It has to fall back to CPU and say why.
    #[test]
    fn a_device_the_node_cannot_target_is_skipped() {
        let mut cuda = dev("cuda", 24_000_000_000, false);
        cuda.available = false;
        let c = caps_with(vec![dev("cpu", 0, false), cuda], 64_000_000_000);
        let (chosen, notes) = best_device(&c);
        assert_eq!(chosen.as_str(), "cpu");
        assert_eq!(notes.len(), 1, "the skip must be reported, not silent");
        assert!(notes[0].contains("cuda"), "{notes:?}");
    }

    #[test]
    fn cidr_24_expands_to_usable_hosts() {
        let h = cidr_hosts("10.0.0.0/24").unwrap();
        assert_eq!(h.len(), 254, "network and broadcast excluded");
        assert_eq!(h[0], Ipv4Addr::new(10, 0, 0, 1));
        assert_eq!(h[253], Ipv4Addr::new(10, 0, 0, 254));
    }

    #[test]
    fn cidr_respects_the_base_mask() {
        // 10.0.5.7/24 describes the 10.0.5.0 network, not a range starting at .7.
        let h = cidr_hosts("10.0.5.7/24").unwrap();
        assert_eq!(h[0], Ipv4Addr::new(10, 0, 5, 1));
    }

    #[test]
    fn oversized_and_malformed_subnets_are_rejected() {
        assert!(cidr_hosts("10.0.0.0/8").is_err(), "/8 is 16M hosts");
        assert!(cidr_hosts("10.0.0.0").is_err(), "not CIDR");
        assert!(cidr_hosts("10.0.0.0/33").is_err());
    }

    /// The device pick is the thing a hand-written config most often gets wrong,
    /// so it must prefer the biggest real accelerator — and never the ANE.
    #[test]
    fn best_device_prefers_the_largest_accelerator() {
        let c = caps_with(
            vec![
                dev("cpu", 0, false),
                dev("vulkan", 8_000_000_000, true),
                dev("cuda", 24_000_000_000, false),
            ],
            64_000_000_000,
        );
        assert_eq!(best_device(&c).0.as_str(), "cuda");
    }

    #[test]
    fn unified_metal_counts_as_host_ram_not_zero() {
        // Metal reports mem_bytes = 0 (no separate ceiling); it must still beat a
        // small discrete GPU rather than being read as "no memory".
        let c = caps_with(
            vec![
                dev("cpu", 0, false),
                dev("metal", 0, true),
                dev("vulkan", 2_000_000_000, true),
            ],
            128_000_000_000,
        );
        assert_eq!(best_device(&c).0.as_str(), "metal");
    }

    #[test]
    fn ane_is_never_chosen_and_cpu_is_the_floor() {
        let c = caps_with(
            vec![dev("cpu", 0, false), dev("ane", 0, true)],
            16_000_000_000,
        );
        assert_eq!(best_device(&c).0.as_str(), "cpu");
    }

    /// A device label this build does not know must be REPORTED, not folded
    /// into "cpu". `DeviceInfo::kind()` collapses it to `Cpu`, which would make
    /// the accelerator invisible — the planner skips it as the host CPU and
    /// ignores its memory, silently turning a GPU box into a CPU stage.
    #[test]
    fn unrecognized_devices_are_reported_not_silently_cpu() {
        let c = caps_with(
            vec![
                dev("cpu", 0, false),
                dev("quantum-tpu", 80_000_000_000, false),
            ],
            64_000_000_000,
        );
        let (chosen, unknown) = best_device(&c);
        assert_eq!(
            unknown.len(),
            1,
            "the unknown label must surface: {unknown:?}"
        );
        assert!(unknown[0].contains("quantum-tpu"), "{unknown:?}");
        // With nothing recognised, CPU is the honest answer — but it is now
        // accompanied by the reason.
        assert_eq!(chosen.as_str(), "cpu");
        // And the lossy accessor is what would have hidden it.
        let unknown_dev = &c.devices[1];
        assert!(!unknown_dev.is_recognized());
        assert_eq!(unknown_dev.kind(), rlx_runtime::Device::Cpu);
        assert_eq!(unknown_dev.parsed(), None);
    }

    /// A recognised accelerator alongside an unrecognised one is still chosen,
    /// and the unknown is still reported.
    #[test]
    fn unrecognized_device_does_not_block_a_known_accelerator() {
        let c = caps_with(
            vec![
                dev("cpu", 0, false),
                dev("weird-npu", 99_000_000_000, false),
                dev("cuda", 24_000_000_000, false),
            ],
            64_000_000_000,
        );
        let (chosen, unknown) = best_device(&c);
        assert_eq!(chosen.as_str(), "cuda");
        assert_eq!(unknown.len(), 1, "{unknown:?}");
        assert!(unknown[0].contains("weird-npu"), "{unknown:?}");
    }

    #[test]
    fn merge_keeps_hand_written_nodes_and_dedups() {
        let mut existing = vec![NodeConfig {
            addr: "a:9100".into(),
            ssh: None,
            ckpt_dir: "/pinned".into(),
            device: "cuda".parse().unwrap(),
            precision: "f16".parse().unwrap(),
            kv_cache: Default::default(),
            rng_seed: None,
            max_ram_gb: Some(40.0),
            layers: None,
            role: Default::default(),
            standby_for: None,
        }];
        let found = vec![
            NodeConfig {
                addr: "a:9100".into(),
                ssh: Some("a".into()),
                ckpt_dir: "/auto".into(),
                device: "cpu".parse().unwrap(),
                precision: "bf16".parse().unwrap(),
                kv_cache: Default::default(),
                rng_seed: None,
                max_ram_gb: None,
                layers: None,
                role: Default::default(),
                standby_for: None,
            },
            NodeConfig {
                addr: "b:9100".into(),
                ssh: Some("b".into()),
                ckpt_dir: "/auto".into(),
                device: "metal".parse().unwrap(),
                precision: "bf16".parse().unwrap(),
                kv_cache: Default::default(),
                rng_seed: None,
                max_ram_gb: None,
                layers: None,
                role: Default::default(),
                standby_for: None,
            },
        ];
        assert_eq!(merge_nodes(&mut existing, found), 1);
        assert_eq!(existing.len(), 2);
        // The pinned entry survives untouched.
        assert_eq!(existing[0].ckpt_dir, "/pinned");
        assert_eq!(existing[0].max_ram_gb, Some(40.0));
    }
}
