// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host hardware profiler for RLX.
//!
//! Probes the local machine for compute backends, GPUs, and VRAM so RLX can do
//! VRAM-aware device selection and feed a topology planner. All probes shell out
//! to standard vendor tools (`nvidia-smi`, `rocminfo`, `vulkaninfo`, `sysctl`,
//! `system_profiler`) via [`std::process::Command`] and degrade gracefully when a
//! tool is missing — a machine with no GPU tooling still returns a valid profile
//! that at minimum reports the CPU backend.
//!
//! The public types are `serde`-serializable so a caller can persist a profile or
//! ship it across a network to a topology planner. Nothing here depends on any
//! RLX backend crate or planner; a caller builds its own `NodeSpec { vram_bytes }`
//! from a [`GpuInfo`] or [`HostProfile::largest_gpu_vram_bytes`].
//!
//! # Example
//! ```no_run
//! let profile = rlx_hwprofile::detect();
//! println!("primary backend: {:?}", profile.primary_backend());
//! if let Some(vram) = profile.largest_gpu_vram_bytes() {
//!     println!("largest GPU has {} bytes of VRAM", vram);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::process::Command;

/// A compute backend RLX can target on this host.
///
/// This is intentionally a small, self-contained enum so this crate depends on no
/// RLX backend/runtime types. Map it to a concrete backend at the call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    /// The host CPU. Always available.
    Cpu,
    /// NVIDIA CUDA.
    Cuda,
    /// AMD ROCm / HIP.
    Rocm,
    /// Apple Metal (unified memory on Apple Silicon).
    Metal,
    /// Vulkan compute.
    Vulkan,
}

/// A single detected GPU.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    /// Human-readable device name (e.g. `"NVIDIA CUDA GPU 0"`).
    pub name: String,
    /// Usable video memory in bytes. `0` when the probe could not determine it.
    ///
    /// On unified-memory hosts (Apple Silicon) this is the total system RAM,
    /// since the GPU shares it with the CPU.
    pub vram_bytes: u64,
    /// The backend that drives this device.
    pub backend: BackendKind,
    /// Zero-based device ordinal within its backend.
    pub index: u32,
    /// Compute-architecture string when known: CUDA SM (e.g. `"89"`) or ROCm gfx
    /// (e.g. `"gfx1100"`). `None` when the probe could not determine it.
    pub compute: Option<String>,
}

/// A snapshot of the host's compute-relevant hardware.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostProfile {
    /// Operating system, from [`std::env::consts::OS`].
    pub os: String,
    /// CPU architecture, from [`std::env::consts::ARCH`].
    pub arch: String,
    /// Number of logical CPU cores.
    pub cpu_cores: usize,
    /// Total system RAM in bytes. `0` when it could not be determined.
    pub total_ram_bytes: u64,
    /// All detected GPUs, in probe order (NVIDIA first, then ROCm, then Vulkan).
    pub gpus: Vec<GpuInfo>,
    /// Backends usable on this host. Always contains [`BackendKind::Cpu`].
    pub available_backends: Vec<BackendKind>,
}

impl HostProfile {
    /// The VRAM of the largest GPU, or `None` when no GPU was detected.
    ///
    /// GPUs with unknown VRAM (`vram_bytes == 0`) do not contribute a value; if
    /// every detected GPU has unknown VRAM this returns `None`.
    pub fn largest_gpu_vram_bytes(&self) -> Option<u64> {
        self.gpus
            .iter()
            .map(|gpu| gpu.vram_bytes)
            .filter(|bytes| *bytes > 0)
            .max()
    }

    /// The preferred backend for this host.
    ///
    /// Picks the backend of the first detected GPU (probe order already ranks
    /// NVIDIA → ROCm → Vulkan), falling back to [`BackendKind::Cpu`] when no GPU
    /// is present.
    pub fn primary_backend(&self) -> BackendKind {
        self.gpus
            .first()
            .map(|gpu| gpu.backend)
            .unwrap_or(BackendKind::Cpu)
    }
}

/// Probe the local host and return a [`HostProfile`].
///
/// Never panics: a host with no GPU tooling installed still yields a valid
/// profile whose `available_backends` contains [`BackendKind::Cpu`].
pub fn detect() -> HostProfile {
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    let cpu_cores = detect_cpu_cores();

    let mut available_backends = vec![BackendKind::Cpu];

    #[cfg(target_os = "macos")]
    let (total_ram_bytes, gpus) = {
        let total_ram_bytes = macos_total_ram_bytes();
        // Apple Silicon / Metal: unified memory means the whole system RAM is the
        // GPU's usable "VRAM".
        let gpus = vec![macos_metal_gpu(total_ram_bytes)];
        available_backends.push(BackendKind::Metal);
        (total_ram_bytes, gpus)
    };

    #[cfg(not(target_os = "macos"))]
    let (total_ram_bytes, gpus) = {
        let total_ram_bytes = host_total_ram_bytes();
        let gpus = detect_pc_gpus();
        for backend in [BackendKind::Cuda, BackendKind::Rocm, BackendKind::Vulkan] {
            if gpus.iter().any(|gpu| gpu.backend == backend)
                && !available_backends.contains(&backend)
            {
                available_backends.push(backend);
            }
        }
        (total_ram_bytes, gpus)
    };

    HostProfile {
        os,
        arch,
        cpu_cores,
        total_ram_bytes,
        gpus,
        available_backends,
    }
}

fn detect_cpu_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

// ---------------------------------------------------------------------------
// macOS / Apple Metal
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn macos_total_ram_bytes() -> u64 {
    command_output("sysctl", &["-n", "hw.memsize"])
        .and_then(|out| out.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

#[cfg(target_os = "macos")]
fn macos_metal_gpu(unified_ram_bytes: u64) -> GpuInfo {
    GpuInfo {
        name: macos_gpu_name(),
        vram_bytes: unified_ram_bytes,
        backend: BackendKind::Metal,
        index: 0,
        compute: None,
    }
}

#[cfg(target_os = "macos")]
fn macos_gpu_name() -> String {
    // Prefer the discrete "Chipset Model" from system_profiler; fall back to the
    // CPU brand string (accurate for Apple Silicon SoCs), then a generic label.
    if let Some(name) = system_profiler_chipset_model() {
        return name;
    }
    if let Some(brand) = command_output("sysctl", &["-n", "machdep.cpu.brand_string"]) {
        let brand = brand.trim();
        if !brand.is_empty() {
            return brand.to_string();
        }
    }
    "Apple GPU".to_string()
}

#[cfg(target_os = "macos")]
fn system_profiler_chipset_model() -> Option<String> {
    let output = command_output("system_profiler", &["SPDisplaysDataType"])?;
    parse_chipset_model(&output)
}

/// Extract the first `Chipset Model:` value from `system_profiler SPDisplaysDataType`.
#[cfg(any(target_os = "macos", test))]
fn parse_chipset_model(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let (_, value) = line.trim().split_once("Chipset Model:")?;
        let value = value.trim();
        (!value.is_empty()).then(|| value.to_string())
    })
}

// ---------------------------------------------------------------------------
// Non-macOS host RAM (Linux /proc/meminfo, best-effort elsewhere)
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "macos"))]
fn host_total_ram_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            if let Some(bytes) = parse_meminfo_total_bytes(&meminfo) {
                return bytes;
            }
        }
    }
    0
}

/// Parse `MemTotal:` (reported in kB) from `/proc/meminfo` into bytes.
#[cfg(any(target_os = "linux", test))]
fn parse_meminfo_total_bytes(meminfo: &str) -> Option<u64> {
    for line in meminfo.lines() {
        let Some(rest) = line.strip_prefix("MemTotal:") else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let value = parts.next()?.parse::<u64>().ok()?;
        // The unit is conventionally "kB" (kibibytes) in /proc/meminfo.
        return Some(value.saturating_mul(1024));
    }
    None
}

// ---------------------------------------------------------------------------
// PC GPU detection (Linux / Windows): NVIDIA, ROCm, Vulkan
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "macos"))]
fn detect_pc_gpus() -> Vec<GpuInfo> {
    let mut gpus = detect_nvidia_gpus();
    gpus.extend(detect_rocm_gpus(gpus.len() as u32));
    gpus.extend(detect_vulkan_gpus(&gpus, gpus.len() as u32));
    gpus
}

// --- NVIDIA via nvidia-smi -------------------------------------------------

#[cfg(not(target_os = "macos"))]
fn detect_nvidia_gpus() -> Vec<GpuInfo> {
    // Name + VRAM per device (aligned by row/index).
    let query = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
    );
    // Compute capability per device index, e.g. `0, 8.9`.
    let compute_caps = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=index,compute_cap",
            "--format=csv,noheader,nounits",
        ],
    )
    .map(|out| nvidia_compute_caps_by_index(&out))
    .unwrap_or_default();

    let Some(query) = query else {
        return Vec::new();
    };
    parse_nvidia_query_gpu(&query, &compute_caps)
}

/// Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`.
/// `memory.total` is in MiB (nounits strips the unit); we convert to bytes.
#[cfg(any(not(target_os = "macos"), test))]
fn parse_nvidia_query_gpu(output: &str, compute_caps: &[(u32, String)]) -> Vec<GpuInfo> {
    output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .filter_map(|(idx, line)| {
            let (name, mem_mib) = line.split_once(',')?;
            let name = name.trim();
            if name.is_empty() {
                return None;
            }
            let vram_bytes = mem_mib
                .trim()
                .parse::<u64>()
                .ok()
                .map(|mib| mib.saturating_mul(1024 * 1024))
                .unwrap_or(0);
            let index = idx as u32;
            let compute = compute_caps
                .iter()
                .find(|(i, _)| *i == index)
                .map(|(_, sm)| sm.clone());
            Some(GpuInfo {
                name: name.to_string(),
                vram_bytes,
                backend: BackendKind::Cuda,
                index,
                compute,
            })
        })
        .collect()
}

/// Parse `index, compute_cap` rows (e.g. `0, 8.9`) into `(index, "89")` SM strings.
#[cfg(any(not(target_os = "macos"), test))]
fn nvidia_compute_caps_by_index(output: &str) -> Vec<(u32, String)> {
    output
        .lines()
        .filter_map(|line| {
            let (index, cap) = line.split_once(',')?;
            let index = index.trim().parse::<u32>().ok()?;
            let sm = cuda_sm_from_compute_cap(cap.trim())?;
            Some((index, sm))
        })
        .collect()
}

/// Convert a `major.minor` compute-cap string (`"8.9"`) to an SM string (`"89"`).
#[cfg(any(not(target_os = "macos"), test))]
fn cuda_sm_from_compute_cap(value: &str) -> Option<String> {
    let (major, minor) = value.split_once('.')?;
    let major = major.trim();
    let minor = minor.trim();
    if major.is_empty()
        || minor.is_empty()
        || !major.chars().all(|c| c.is_ascii_digit())
        || !minor.chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    Some(format!("{major}{minor}"))
}

// --- AMD / ROCm via rocminfo ----------------------------------------------

#[cfg(not(target_os = "macos"))]
fn detect_rocm_gpus(start_index: u32) -> Vec<GpuInfo> {
    let Some(output) = command_output("rocminfo", &[]) else {
        return Vec::new();
    };
    parse_rocminfo(&output, start_index)
}

/// Parse `rocminfo` output into GPU-agent [`GpuInfo`]s.
///
/// `rocminfo` prints one `Agent N` block per device. We keep only GPU agents
/// (`Device Type: GPU`), reading `Marketing Name` for the display name, `Name`
/// for the gfx architecture, and the coarse-grained pool `Size` (in KiB) for VRAM.
#[cfg(any(not(target_os = "macos"), test))]
fn parse_rocminfo(output: &str, start_index: u32) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    let mut next_index = start_index;

    // State for the agent block currently being parsed.
    let mut in_agent = false;
    let mut is_gpu = false;
    let mut marketing: Option<String> = None;
    let mut gfx: Option<String> = None;
    let mut vram_kib: u64 = 0;
    let mut in_pool = false;

    let flush = |gpus: &mut Vec<GpuInfo>,
                 next_index: &mut u32,
                 is_gpu: bool,
                 marketing: &Option<String>,
                 gfx: &Option<String>,
                 vram_kib: u64| {
        if !is_gpu {
            return;
        }
        let name = marketing
            .clone()
            .or_else(|| gfx.clone())
            .unwrap_or_else(|| "AMD GPU".to_string());
        gpus.push(GpuInfo {
            name,
            vram_bytes: vram_kib.saturating_mul(1024),
            backend: BackendKind::Rocm,
            index: *next_index,
            compute: gfx.clone(),
        });
        *next_index += 1;
    };

    for raw in output.lines() {
        let line = raw.trim();

        if line.starts_with("Agent ") {
            // Starting a new agent block: flush the previous one.
            if in_agent {
                flush(
                    &mut gpus,
                    &mut next_index,
                    is_gpu,
                    &marketing,
                    &gfx,
                    vram_kib,
                );
            }
            in_agent = true;
            is_gpu = false;
            marketing = None;
            gfx = None;
            vram_kib = 0;
            in_pool = false;
            continue;
        }

        if !in_agent {
            continue;
        }

        if let Some((_, value)) = line.split_once("Device Type:") {
            is_gpu = value.trim().eq_ignore_ascii_case("GPU");
        } else if let Some((_, value)) = line.split_once("Marketing Name:") {
            let value = value.trim();
            if !value.is_empty() {
                marketing = Some(value.to_string());
            }
        } else if let Some((_, value)) = line.split_once("Name:") {
            // `Name: gfx1100` gives the architecture; ignore non-gfx names.
            let value = value.trim();
            if value.starts_with("gfx") {
                gfx = Some(value.to_string());
            }
        } else if line.starts_with("Pool Info") {
            in_pool = true;
        } else if in_pool {
            if let Some((_, value)) = line.split_once("Size:") {
                // e.g. `Size: 25149440(0x17fc000) KB` — take the coarse-grained
                // (largest) pool size as an approximation of VRAM.
                if let Some(kib) = parse_rocm_pool_size_kib(value) {
                    vram_kib = vram_kib.max(kib);
                }
            }
        }
    }

    if in_agent {
        flush(
            &mut gpus,
            &mut next_index,
            is_gpu,
            &marketing,
            &gfx,
            vram_kib,
        );
    }

    gpus
}

/// Parse a rocminfo pool `Size:` value like `25149440(0x17fc000) KB` into KiB.
#[cfg(any(not(target_os = "macos"), test))]
fn parse_rocm_pool_size_kib(value: &str) -> Option<u64> {
    let value = value.trim();
    let digits: String = value.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<u64>().ok()
}

// --- Intel / other via vulkaninfo (best-effort) ---------------------------

#[cfg(not(target_os = "macos"))]
fn detect_vulkan_gpus(existing: &[GpuInfo], start_index: u32) -> Vec<GpuInfo> {
    let Some(output) = command_output("vulkaninfo", &["--summary"]) else {
        return Vec::new();
    };
    parse_vulkaninfo(&output, existing, start_index)
}

/// Parse `vulkaninfo --summary` `deviceName = ...` lines into [`GpuInfo`]s.
///
/// Skips software rasterizers (llvmpipe/lavapipe/swiftshader) and any device
/// whose name matches a GPU already reported by a more specific probe
/// (nvidia-smi / rocminfo), so an NVIDIA card is not double-counted as Vulkan.
/// Vulkan does not expose VRAM in the summary, so `vram_bytes` is `0`.
#[cfg(any(not(target_os = "macos"), test))]
fn parse_vulkaninfo(output: &str, existing: &[GpuInfo], start_index: u32) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    let mut next_index = start_index;
    for line in output.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("deviceName") else {
            continue;
        };
        let Some((_, name)) = rest.split_once('=') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() || looks_like_software_vulkan_adapter(name) {
            continue;
        }
        if existing
            .iter()
            .any(|gpu| names_refer_to_same_gpu(&gpu.name, name))
        {
            continue;
        }
        gpus.push(GpuInfo {
            name: name.to_string(),
            vram_bytes: 0,
            backend: BackendKind::Vulkan,
            index: next_index,
            compute: None,
        });
        next_index += 1;
    }
    gpus
}

#[cfg(any(not(target_os = "macos"), test))]
fn looks_like_software_vulkan_adapter(value: &str) -> bool {
    let label = value.to_ascii_lowercase();
    [
        "llvmpipe",
        "swiftshader",
        "lavapipe",
        "softpipe",
        "software rasterizer",
    ]
    .iter()
    .any(|marker| label.contains(marker))
}

/// Loose match: do two device names refer to the same physical GPU? Used to
/// dedupe a Vulkan device against an NVIDIA/ROCm device already detected.
#[cfg(any(not(target_os = "macos"), test))]
fn names_refer_to_same_gpu(a: &str, b: &str) -> bool {
    let a = a.to_ascii_lowercase();
    let b = b.to_ascii_lowercase();
    if a == b {
        return true;
    }
    // Share a distinctive model token (e.g. "4090", "w7900").
    let model_token = |s: &str| -> Option<String> {
        s.split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|t| t.len() >= 4 && t.chars().any(|c| c.is_ascii_digit()))
            .map(|t| t.to_string())
            .next()
    };
    match (model_token(&a), model_token(&b)) {
        (Some(ta), Some(tb)) => ta == tb,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Shared command runner
// ---------------------------------------------------------------------------

/// Run `program args…` and return stdout as a `String` on success. Returns `None`
/// when the binary is missing, the process fails, or stdout is not valid UTF-8 —
/// so every probe degrades gracefully rather than panicking.
fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).ok())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_a_valid_profile() {
        let profile = detect();
        assert!(
            profile.available_backends.contains(&BackendKind::Cpu),
            "CPU backend must always be available"
        );
        assert!(!profile.os.is_empty(), "os must be non-empty");
        assert!(!profile.arch.is_empty(), "arch must be non-empty");
        assert!(profile.cpu_cores >= 1, "must report at least one core");
    }

    #[test]
    fn primary_backend_falls_back_to_cpu_without_gpus() {
        let profile = HostProfile {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            cpu_cores: 8,
            total_ram_bytes: 0,
            gpus: Vec::new(),
            available_backends: vec![BackendKind::Cpu],
        };
        assert_eq!(profile.primary_backend(), BackendKind::Cpu);
        assert_eq!(profile.largest_gpu_vram_bytes(), None);
    }

    #[test]
    fn largest_gpu_vram_picks_the_max_and_ignores_unknown() {
        let profile = HostProfile {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            cpu_cores: 8,
            total_ram_bytes: 0,
            gpus: vec![
                GpuInfo {
                    name: "GPU A".to_string(),
                    vram_bytes: 8 * 1024 * 1024 * 1024,
                    backend: BackendKind::Cuda,
                    index: 0,
                    compute: Some("89".to_string()),
                },
                GpuInfo {
                    name: "GPU B".to_string(),
                    vram_bytes: 24 * 1024 * 1024 * 1024,
                    backend: BackendKind::Cuda,
                    index: 1,
                    compute: Some("89".to_string()),
                },
                GpuInfo {
                    name: "GPU C (unknown vram)".to_string(),
                    vram_bytes: 0,
                    backend: BackendKind::Vulkan,
                    index: 2,
                    compute: None,
                },
            ],
            available_backends: vec![BackendKind::Cpu, BackendKind::Cuda, BackendKind::Vulkan],
        };
        assert_eq!(
            profile.largest_gpu_vram_bytes(),
            Some(24 * 1024 * 1024 * 1024)
        );
        assert_eq!(profile.primary_backend(), BackendKind::Cuda);
    }

    #[test]
    fn parses_nvidia_query_gpu_with_vram_and_compute() {
        let output = "\
NVIDIA CUDA GPU 0, 24564
NVIDIA CUDA GPU 1, 10240
";
        let caps = vec![(0u32, "89".to_string()), (1u32, "86".to_string())];
        let gpus = parse_nvidia_query_gpu(output, &caps);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].name, "NVIDIA CUDA GPU 0");
        assert_eq!(gpus[0].vram_bytes, 24564 * 1024 * 1024);
        assert_eq!(gpus[0].backend, BackendKind::Cuda);
        assert_eq!(gpus[0].index, 0);
        assert_eq!(gpus[0].compute.as_deref(), Some("89"));
        assert_eq!(gpus[1].index, 1);
        assert_eq!(gpus[1].compute.as_deref(), Some("86"));
    }

    #[test]
    fn nvidia_compute_caps_parse_major_minor_into_sm() {
        let output = "0, 8.9\n1, 8.6\n";
        assert_eq!(
            nvidia_compute_caps_by_index(output),
            vec![(0, "89".to_string()), (1, "86".to_string())]
        );
    }

    #[test]
    fn parses_rocminfo_gpu_agent() {
        let output = "\
*******
Agent 1
*******
  Name:                    AMD Ryzen 9 7950X
  Marketing Name:          AMD Ryzen 9 7950X
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon PRO W7900
  Device Type:             GPU
  Pool Info:
    Pool 1
      Segment:             GLOBAL; FLAGS: COARSE GRAINED
      Size:                46137344(0x2c00000) KB
";
        let gpus = parse_rocminfo(output, 0);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "AMD Radeon PRO W7900");
        assert_eq!(gpus[0].backend, BackendKind::Rocm);
        assert_eq!(gpus[0].compute.as_deref(), Some("gfx1100"));
        assert_eq!(gpus[0].vram_bytes, 46137344u64 * 1024);
        assert_eq!(gpus[0].index, 0);
    }

    #[test]
    fn vulkaninfo_skips_software_and_deduped_devices() {
        let output = "\
GPU0:
	deviceName         = llvmpipe (LLVM 18.1.8, 256 bits)
GPU1:
	deviceName         = NVIDIA CUDA GPU 0
GPU2:
	deviceName         = Intel(R) Arc(tm) A770 Graphics
";
        let existing = vec![GpuInfo {
            name: "NVIDIA CUDA GPU 0".to_string(),
            vram_bytes: 24564 * 1024 * 1024,
            backend: BackendKind::Cuda,
            index: 0,
            compute: Some("89".to_string()),
        }];
        let gpus = parse_vulkaninfo(output, &existing, 1);
        // llvmpipe skipped, 4090 deduped against existing CUDA device.
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "Intel(R) Arc(tm) A770 Graphics");
        assert_eq!(gpus[0].backend, BackendKind::Vulkan);
        assert_eq!(gpus[0].index, 1);
        assert_eq!(gpus[0].vram_bytes, 0);
    }

    #[test]
    fn parses_meminfo_total_into_bytes() {
        let meminfo = "\
MemTotal:       65851234 kB
MemFree:         1234567 kB
";
        assert_eq!(parse_meminfo_total_bytes(meminfo), Some(65851234u64 * 1024));
    }

    #[test]
    fn parses_chipset_model_from_system_profiler() {
        let output = "\
Graphics/Displays:

    Apple M2 Max:

      Chipset Model: Apple M2 Max
      Type: GPU
      Bus: Built-In
";
        assert_eq!(parse_chipset_model(output).as_deref(), Some("Apple M2 Max"));
    }

    #[test]
    fn cuda_sm_rejects_non_numeric_compute_cap() {
        assert_eq!(cuda_sm_from_compute_cap("8.9").as_deref(), Some("89"));
        assert_eq!(cuda_sm_from_compute_cap("N/A"), None);
        assert_eq!(cuda_sm_from_compute_cap(""), None);
    }

    #[test]
    fn profile_round_trips_through_json() {
        let profile = HostProfile {
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
            cpu_cores: 12,
            total_ram_bytes: 64 * 1024 * 1024 * 1024,
            gpus: vec![GpuInfo {
                name: "Apple M2 Max".to_string(),
                vram_bytes: 64 * 1024 * 1024 * 1024,
                backend: BackendKind::Metal,
                index: 0,
                compute: None,
            }],
            available_backends: vec![BackendKind::Cpu, BackendKind::Metal],
        };
        let json = serde_json::to_string(&profile).expect("serialize");
        let back: HostProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(profile, back);
    }
}
