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

//! Backend allowlists, env-driven defaults, and selection introspection.

use rlx_driver::Device;
use rlx_ir::Graph;

use crate::cost::fastest_device_for_with_policy;
use crate::device_ext::{DEVICE_PRIORITY, is_available, supports_graph};
use crate::device_parse::{device_label, parse_device, parse_device_list};
use crate::registry::backend_for;

/// How [`GraphDevices::resolve_with_inputs`] picks a backend when no hint is set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePickStrategy {
    /// Rank via calibrated cost models + platform priority (default).
    #[default]
    CostModel,
    /// Run a short [`crate::benchmark_devices`] once and cache the winner.
    Benchmark { runs: usize },
}

/// Which backends a process may use — intersected with compile-time features
/// and runtime availability.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DevicePolicy {
    allow: Option<Vec<Device>>,
    deny: Vec<Device>,
    prefer: Vec<Device>,
    pick: DevicePickStrategy,
}

impl DevicePolicy {
    /// Allow every compiled-in backend (default).
    pub fn all() -> Self {
        Self::default()
    }

    /// Restrict to an explicit backend set the developer ships.
    pub fn only(devices: impl IntoIterator<Item = Device>) -> Self {
        Self {
            allow: Some(devices.into_iter().collect()),
            ..Self::default()
        }
    }

    /// Exclude specific backends from consideration.
    pub fn with_deny(mut self, devices: impl IntoIterator<Item = Device>) -> Self {
        self.deny.extend(devices);
        self
    }

    /// Prefer these backends when cost models tie or are unavailable.
    pub fn with_prefer(mut self, devices: impl IntoIterator<Item = Device>) -> Self {
        self.prefer.extend(devices);
        self
    }

    /// Pick the fastest backend via a one-time micro-benchmark (needs inputs at resolve time).
    pub fn with_benchmark_pick(mut self, runs: usize) -> Self {
        self.pick = DevicePickStrategy::Benchmark { runs: runs.max(1) };
        self
    }

    pub fn pick_strategy(&self) -> DevicePickStrategy {
        self.pick
    }

    /// Read policy from process env (see [`Self::from_env_key`]).
    pub fn from_env() -> Self {
        Self::from_env_key("RLX")
    }

    /// Read `PREFIX_DEVICES`, `PREFIX_DENY_DEVICES`, `PREFIX_PREFER_DEVICES`.
    pub fn from_env_key(prefix: &str) -> Self {
        let mut policy = Self::default();
        let devices_key = format!("{prefix}_DEVICES");
        let deny_key = format!("{prefix}_DENY_DEVICES");
        let prefer_key = format!("{prefix}_PREFER_DEVICES");

        if let Some(raw) = rlx_ir::env::var(&devices_key) {
            if let Ok(list) = parse_device_list(&raw) {
                policy.allow = Some(list);
            }
        }
        if let Some(raw) = rlx_ir::env::var(&deny_key) {
            if let Ok(list) = parse_device_list(&raw) {
                policy.deny = list;
            }
        }
        if let Some(raw) = rlx_ir::env::var(&prefer_key) {
            if let Ok(list) = parse_device_list(&raw) {
                policy.prefer = list;
            }
        }
        let bench_key = format!("{prefix}_BENCHMARK_PICK");
        if let Some(raw) = rlx_ir::env::var(&bench_key) {
            if let Ok(runs) = raw.trim().parse::<usize>() {
                policy.pick = DevicePickStrategy::Benchmark { runs: runs.max(1) };
            }
        }
        policy
    }

    /// Devices to show in reports when no allow-list is set.
    pub fn probe_set(&self) -> Vec<Device> {
        self.allow.clone().unwrap_or_else(|| Device::all().to_vec())
    }

    /// Filter and order `candidates` according to this policy.
    pub fn apply(&self, mut candidates: Vec<Device>) -> Vec<Device> {
        if let Some(allow) = &self.allow {
            candidates.retain(|d| allow.contains(d));
        }
        candidates.retain(|d| !self.deny.contains(d));
        candidates.sort_by_key(|d| self.rank_key(*d));
        candidates
    }

    fn rank_key(&self, device: Device) -> (u8, u8) {
        let prefer = self
            .prefer
            .iter()
            .position(|d| *d == device)
            .map(|i| i as u8)
            .unwrap_or(u8::MAX);
        let platform = DEVICE_PRIORITY
            .iter()
            .position(|d| *d == device)
            .map(|i| i as u8)
            .unwrap_or(u8::MAX);
        (prefer, platform)
    }
}

/// Default device hint from `RLX_DEVICE` (or `PREFIX_DEVICE` via [`device_from_env_key`]).
pub fn device_from_env() -> Option<Device> {
    device_from_env_key("RLX")
}

/// Read `PREFIX_DEVICE` env var.
pub fn device_from_env_key(prefix: &str) -> Option<Device> {
    let key = format!("{prefix}_DEVICE");
    rlx_ir::env::var(&key).and_then(|raw| parse_device(&raw).ok())
}

/// Backends on this host that can lower `graph`, filtered by `policy`.
pub fn devices_for_with_policy(graph: &Graph, policy: &DevicePolicy) -> Vec<Device> {
    policy.apply(
        crate::available_devices()
            .into_iter()
            .filter(|d| supports_graph(*d, graph))
            .collect(),
    )
}

/// One row of backend introspection for UIs and logs.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceCandidate {
    pub device: Device,
    pub label: &'static str,
    pub available: bool,
    pub registered: bool,
    pub supports_graph: bool,
    pub recommended: bool,
    pub blocker: Option<String>,
    /// Advisory feature names this device's executables typically expose
    /// (see [`crate::ExecutableCapabilities`]). Confirmed after compile via
    /// [`crate::ExecutableGraph::capabilities`].
    pub capabilities: Vec<&'static str>,
}

/// Advisory capability bits matching what each backend's
/// [`crate::ExecutableGraph::capabilities`] override usually reports.
pub fn advisory_capabilities(device: Device) -> crate::ExecutableCapabilities {
    use crate::ExecutableCapabilities as C;
    match device {
        Device::Cpu => C {
            clone: true,
            moe: true,
            typed_io: true,
            active_extent: true,
            ..C::NONE
        },
        Device::Metal => C {
            clone: true,
            gpu_handles: true,
            kv_resident: true,
            typed_io: true,
            async_pipeline: true,
            active_extent: true,
            ..C::NONE
        },
        Device::Mlx => C {
            clone: true,
            moe: true,
            persistent_handles: true,
            gpu_handles: true,
            kv_resident: true,
            typed_io: true,
            active_extent: true,
            ..C::NONE
        },
        Device::Ane => C {
            clone: true,
            typed_io: true,
            ..C::NONE
        },
        Device::Cuda => C {
            clone: true,
            gpu_handles: true,
            kv_resident: true,
            typed_io: true,
            active_extent: true,
            ..C::NONE
        },
        Device::Rocm => C {
            clone: true,
            moe: true,
            gpu_handles: true,
            kv_resident: true,
            ..C::NONE
        },
        Device::Gpu | Device::WebGpu => C {
            clone: true,
            gpu_handles: true,
            typed_io: true,
            active_extent: true,
            ..C::NONE
        },
        Device::Vulkan => C {
            clone: true,
            gpu_handles: true,
            kv_resident: true,
            typed_io: true,
            ..C::NONE
        },
        Device::OneApi => C {
            clone: true,
            typed_io: true,
            ..C::NONE
        },
        Device::Tpu => C {
            clone: true,
            typed_io: true,
            ..C::NONE
        },
        Device::OpenGl | Device::DirectX => C {
            typed_io: true,
            ..C::NONE
        },
        Device::Hexagon => C::NONE,
    }
}

/// Explain which backends are viable for `graph` under `policy`.
pub fn device_report(graph: &Graph, policy: &DevicePolicy) -> Vec<DeviceCandidate> {
    let recommended = fastest_device_for_with_policy(graph, policy);
    policy
        .probe_set()
        .into_iter()
        .map(|device| {
            let available = is_available(device);
            let registered = backend_for(device).is_some();
            let supports = available && supports_graph(device, graph);
            let blocker = if !available {
                Some("not available on this host or in this build".into())
            } else if !supports {
                crate::first_unsupported_op(device, graph)
                    .map(|(idx, op)| format!("unsupported op at node {idx}: {op:?}"))
            } else if policy.deny.contains(&device) {
                Some("denied by DevicePolicy".into())
            } else if policy
                .allow
                .as_ref()
                .is_some_and(|allow| !allow.contains(&device))
            {
                Some("not in DevicePolicy allow-list".into())
            } else {
                None
            };
            DeviceCandidate {
                device,
                label: device_label(device),
                available,
                registered,
                supports_graph: supports,
                recommended: device == recommended,
                blocker,
                capabilities: advisory_capabilities(device).enabled_names(),
            }
        })
        .collect()
}

/// Resolve the backend to use: explicit hint → env → fastest for `graph`.
pub fn resolve_device(
    graph: &Graph,
    hint: Option<Device>,
    policy: &DevicePolicy,
) -> Result<Device, String> {
    let candidates = devices_for_with_policy(graph, policy);
    if candidates.is_empty() {
        return Err(
            "no backend can lower this graph under the current policy — \
             widen DevicePolicy or enable additional Cargo features"
                .into(),
        );
    }

    if let Some(device) = hint {
        return pick_from_candidates(device, &candidates, "hint");
    }
    if let Some(device) = device_from_env() {
        if let Ok(device) = pick_from_candidates(device, &candidates, "RLX_DEVICE") {
            return Ok(device);
        }
    }
    Ok(fastest_device_for_with_policy(graph, policy))
}

fn pick_from_candidates(
    device: Device,
    candidates: &[Device],
    source: &str,
) -> Result<Device, String> {
    if candidates.contains(&device) {
        return Ok(device);
    }
    Err(format!(
        "{source} requested {device} but viable backends are [{}]",
        candidates
            .iter()
            .map(|d| device_label(*d))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Ordered fallback chain from `RLX_DEVICE_CHAIN` (`cuda,gpu,cpu`).
pub fn device_chain_from_env() -> Vec<Device> {
    device_chain_from_env_key("RLX")
}

/// Read `PREFIX_DEVICE_CHAIN`.
pub fn device_chain_from_env_key(prefix: &str) -> Vec<Device> {
    let key = format!("{prefix}_DEVICE_CHAIN");
    rlx_ir::env::var(&key)
        .and_then(|raw| parse_device_list(&raw).ok())
        .unwrap_or_default()
}

/// First device in `chain` that is viable under `policy` for `graph`.
pub fn resolve_device_chain(
    graph: &Graph,
    chain: &[Device],
    policy: &DevicePolicy,
) -> Result<Device, String> {
    let viable = devices_for_with_policy(graph, policy);
    for &device in chain {
        if viable.contains(&device) {
            return Ok(device);
        }
    }
    Err(format!(
        "no device in chain [{}] can run this graph — viable: [{}]",
        chain
            .iter()
            .map(|d| device_label(*d))
            .collect::<Vec<_>>()
            .join(", "),
        viable
            .iter()
            .map(|d| device_label(*d))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Errors collected when every backend in the chain fails at run time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFallbackError {
    pub attempts: Vec<(Device, String)>,
}

impl std::fmt::Display for DeviceFallbackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "all backends failed:")?;
        for (d, e) in &self.attempts {
            write!(f, "\n  {}: {e}", device_label(*d))?;
        }
        Ok(())
    }
}

impl std::error::Error for DeviceFallbackError {}

impl From<String> for DeviceFallbackError {
    fn from(msg: String) -> Self {
        Self {
            attempts: vec![(Device::Cpu, msg)],
        }
    }
}

/// Try `chain` in order; return the first successful result from `run`.
pub fn run_with_fallback<T, F>(
    graph: &Graph,
    policy: &DevicePolicy,
    chain: &[Device],
    mut run: F,
) -> Result<(Device, T), DeviceFallbackError>
where
    F: FnMut(Device) -> Result<T, String>,
{
    let viable = devices_for_with_policy(graph, policy);
    let mut attempts = Vec::new();
    for &device in chain {
        if !viable.contains(&device) {
            attempts.push((device, "not viable for this graph under policy".into()));
            continue;
        }
        match run(device) {
            Ok(value) => return Ok((device, value)),
            Err(err) => attempts.push((device, err)),
        }
    }
    if attempts.is_empty() {
        attempts.push((Device::Cpu, "empty fallback chain".into()));
    }
    Err(DeviceFallbackError { attempts })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    fn tiny_graph() -> Graph {
        let mut g = Graph::new("tiny");
        let x = g.input("x", Shape::new(&[2], DType::F32));
        g.set_outputs(vec![x]);
        g
    }

    #[test]
    fn only_policy_restricts_devices_for() {
        let g = tiny_graph();
        let all = devices_for_with_policy(&g, &DevicePolicy::default());
        let cpu_only = devices_for_with_policy(&g, &DevicePolicy::only([Device::Cpu]));
        assert_eq!(cpu_only, vec![Device::Cpu]);
        assert!(all.contains(&Device::Cpu));
    }

    #[test]
    fn resolve_honors_hint_then_env() {
        let g = tiny_graph();
        let policy = DevicePolicy::only([Device::Cpu]);
        assert_eq!(
            resolve_device(&g, Some(Device::Cpu), &policy).unwrap(),
            Device::Cpu
        );

        rlx_ir::env::set("RLX_DEVICE", "cpu");
        assert_eq!(resolve_device(&g, None, &policy).unwrap(), Device::Cpu);
        rlx_ir::env::unset("RLX_DEVICE");
    }

    #[test]
    fn device_report_marks_recommended() {
        let g = tiny_graph();
        let policy = DevicePolicy::only([Device::Cpu]);
        let rows = device_report(&g, &policy);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].recommended);
        assert!(rows[0].supports_graph);
        assert!(
            rows[0].capabilities.contains(&"clone"),
            "cpu advisory caps: {:?}",
            rows[0].capabilities
        );
    }
}
