// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the known-issue checks may look at, gathered once.
//!
//! Every file lookup goes through [`HostFacts::root`] and every env lookup
//! through [`HostFacts::env`], so a test builds a `HostFacts` over a temp tree
//! and fake env and asserts which issues fire — no NPU, GPU or distro needed
//! to cover a diagnosis.

use std::path::PathBuf;

use super::{DeviceStatus, DoctorReport};
use crate::Device;

/// One device as the checks see it.
#[derive(Debug, Clone)]
pub struct DeviceFact {
    pub device: Device,
    pub status: DeviceStatus,
    /// The backend's reason it is unavailable, plus any check-run error.
    pub reason: String,
}

/// The host as the known-issue checks see it.
#[derive(Debug, Clone, Default)]
pub struct HostFacts {
    /// Filesystem root for every probe (`/` on a real host).
    pub root: PathBuf,
    pub env: Vec<(String, String)>,
    pub devices: Vec<DeviceFact>,
    /// Detected external versions: `cuda_driver`, `nvrtc`, `mlir_aie`, …
    pub versions: Vec<(&'static str, (u32, u32, u32))>,
    /// wgpu is on a Vulkan/DX12 adapter, where elementwise and structure ops
    /// are host-routed by design.
    pub wgpu_host_routed: bool,
}

impl HostFacts {
    /// Facts for this process, from a finished report.
    pub fn from_report(r: &DoctorReport) -> Self {
        HostFacts {
            root: PathBuf::from("/"),
            env: std::env::vars().collect(),
            devices: r
                .devices
                .iter()
                .map(|d| DeviceFact {
                    device: d.device,
                    status: d.status,
                    reason: [
                        d.fix.clone(),
                        d.toolchain_warning.clone(),
                        d.check.as_ref().and_then(|c| c.error.clone()),
                    ]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("; "),
                })
                .collect(),
            versions: r.versions.clone(),
            wgpu_host_routed: r.wgpu_host_routed,
        }
    }

    pub fn env(&self, k: &str) -> Option<&str> {
        self.env
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v.as_str())
    }

    /// `abs` re-rooted under [`Self::root`].
    pub fn path(&self, abs: &str) -> PathBuf {
        self.root.join(abs.trim_start_matches('/'))
    }

    pub fn exists(&self, abs: &str) -> bool {
        self.path(abs).exists()
    }

    /// `abs` is a directory with at least one entry.
    pub fn dir_nonempty(&self, abs: &str) -> bool {
        std::fs::read_dir(self.path(abs)).is_ok_and(|mut rd| rd.next().is_some())
    }

    /// Some `<abs>/*/<file>` contains `needle` (e.g. a kfd topology property).
    pub fn any_child_file_contains(&self, abs: &str, file: &str, needle: &str) -> bool {
        std::fs::read_dir(self.path(abs)).is_ok_and(|rd| {
            rd.flatten().any(|e| {
                std::fs::read_to_string(e.path().join(file)).is_ok_and(|t| t.contains(needle))
            })
        })
    }

    /// Entries of a `:`-separated env var (absolute host paths).
    pub fn env_dirs(&self, k: &str) -> Vec<String> {
        self.env(k)
            .map(|v| {
                v.split(':')
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn device(&self, d: Device) -> Option<&DeviceFact> {
        self.devices.iter().find(|f| f.device == d)
    }

    /// The backend is built into this binary.
    pub fn compiled(&self, d: Device) -> bool {
        self.device(d)
            .is_some_and(|f| f.status != DeviceStatus::NotCompiled)
    }

    pub fn status(&self, d: Device) -> Option<DeviceStatus> {
        self.device(d).map(|f| f.status)
    }

    pub fn reason_contains(&self, d: Device, needle: &str) -> bool {
        self.device(d).is_some_and(|f| f.reason.contains(needle))
    }

    pub fn version(&self, k: &str) -> Option<(u32, u32, u32)> {
        self.versions.iter().find(|(n, _)| *n == k).map(|(_, v)| *v)
    }
}
