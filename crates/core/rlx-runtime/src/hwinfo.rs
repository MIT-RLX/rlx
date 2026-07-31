// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hardware introspection (plan #47).
//!
//! Exposes CPU / GPU details and current thermal state. The
//! autotuner / calibrator uses [`HwSnapshot::fingerprint`] to key
//! its cache so calibration data is invalidated when the user moves
//! a workspace between machines (or even when the OS re-classifies
//! cores after a SoC update).
//!
//! Three layers:
//! - **CPU** — [`HwSnapshot::collect`]: core topology, caches, and the
//!   Apple thermal state. Read-only; no shell-out beyond `pmset -g therm`
//!   (also invoked from `scripts/check-throttle.sh`).
//! - **GPU telemetry** — [`device_thermal`] / [`all_gpu_thermal`]: live
//!   temperature / power / clock / fan across the CUDA and ROCm backends
//!   via the `rlx_cuda::nvml` (NVML) and `rlx_rocm::rsmi` (ROCm-SMI)
//!   shims. Unprivileged; missing sensors read as `None`, never a fake 0.
//! - **GPU control** — [`set_power_cap`] / [`set_locked_clocks`] /
//!   [`set_fan_percent`] (and resets): root-only knobs returning a typed
//!   [`ThermalError`], range-validated so a caller can't exceed the
//!   device's safe envelope.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
#[cfg(target_os = "macos")]
use std::process::Command;

use rlx_driver::Device;

/// Coarse thermal state. Apple Silicon reports CPU speed limit and
/// scheduler limit via `pmset -g therm`; both are 100 when nominal,
/// less when throttling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThermalState {
    Nominal,
    Throttled { cpu_speed_pct: u32, sched_pct: u32 },
    Unknown,
}

#[derive(Debug, Clone)]
pub struct HwSnapshot {
    pub os: &'static str,
    pub arch: &'static str,
    pub cpu_brand: String,
    /// Total logical CPUs (including E-cores on Apple Silicon).
    pub total_cpus: usize,
    /// Performance cores. 0 if unknown / not asymmetric.
    pub perf_cores: usize,
    /// L1 data cache bytes per core (0 if unknown).
    pub l1d_bytes: usize,
    /// L2 cache bytes per cluster (0 if unknown).
    pub l2_bytes: usize,
    /// Cache line size from the OS.
    pub cache_line: usize,
    pub thermal: ThermalState,
}

impl HwSnapshot {
    /// Read all the queryable hardware details. Cheap (~ms); call
    /// per-process at startup or whenever you need a fresh thermal
    /// reading.
    pub fn collect() -> Self {
        let total_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);

        // Mutated only under per-OS cfg blocks below (currently macOS).
        #[allow(unused_mut)]
        let mut snap = Self {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            cpu_brand: String::new(),
            total_cpus,
            perf_cores: 0,
            l1d_bytes: 0,
            l2_bytes: 0,
            cache_line: 0,
            thermal: ThermalState::Unknown,
        };

        // sysctl is available on every Apple platform; the `hw.*` keys are
        // readable from the iOS sandbox too (failures degrade to 0 / "").
        #[cfg(target_vendor = "apple")]
        {
            snap.cpu_brand = sysctl_str("machdep.cpu.brand_string").unwrap_or_default();
            snap.perf_cores = sysctl_usize("hw.perflevel0.physicalcpu").unwrap_or(0);
            snap.l1d_bytes = sysctl_usize("hw.l1dcachesize").unwrap_or(0);
            snap.l2_bytes = sysctl_usize("hw.l2cachesize").unwrap_or(0);
            snap.cache_line = sysctl_usize("hw.cachelinesize").unwrap_or(0);
        }
        // `pmset` is a macOS CLI; the iOS sandbox forbids spawning processes,
        // so iOS leaves `thermal` at `Unknown`.
        #[cfg(target_os = "macos")]
        {
            snap.thermal = read_pmset_thermal().unwrap_or(ThermalState::Unknown);
        }

        snap
    }

    /// Stable hash of the *machine* fields (everything except
    /// thermal state). Suitable as a calibration cache key — same
    /// machine returns the same fingerprint across boots.
    pub fn fingerprint(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.os.hash(&mut h);
        self.arch.hash(&mut h);
        self.cpu_brand.hash(&mut h);
        self.total_cpus.hash(&mut h);
        self.perf_cores.hash(&mut h);
        self.l1d_bytes.hash(&mut h);
        self.l2_bytes.hash(&mut h);
        self.cache_line.hash(&mut h);
        h.finish()
    }

    /// Convenience: is the machine currently throttling?
    pub fn is_throttled(&self) -> bool {
        matches!(self.thermal, ThermalState::Throttled { .. })
    }
}

#[cfg(target_vendor = "apple")]
fn sysctl_usize(name: &str) -> Option<usize> {
    use std::ffi::CString;
    let cname = CString::new(name).ok()?;
    let mut val: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::os::raw::c_char,
            oldp: *mut std::os::raw::c_void,
            oldlenp: *mut usize,
            newp: *mut std::os::raw::c_void,
            newlen: usize,
        ) -> std::os::raw::c_int;
    }
    let rc = unsafe {
        sysctlbyname(
            cname.as_ptr(),
            &mut val as *mut u64 as *mut _,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 { Some(val as usize) } else { None }
}

#[cfg(target_vendor = "apple")]
fn sysctl_str(name: &str) -> Option<String> {
    use std::ffi::CString;
    let cname = CString::new(name).ok()?;
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::os::raw::c_char,
            oldp: *mut std::os::raw::c_void,
            oldlenp: *mut usize,
            newp: *mut std::os::raw::c_void,
            newlen: usize,
        ) -> std::os::raw::c_int;
    }
    // First call: query buffer length.
    let mut len: usize = 0;
    let rc = unsafe {
        sysctlbyname(
            cname.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len];
    let rc = unsafe {
        sysctlbyname(
            cname.as_ptr(),
            buf.as_mut_ptr() as *mut _,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    // Strip trailing NUL.
    if let Some(&0) = buf.last() {
        buf.pop();
    }
    String::from_utf8(buf).ok()
}

#[cfg(target_os = "macos")]
fn read_pmset_thermal() -> Option<ThermalState> {
    let out = Command::new("pmset").args(["-g", "therm"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut cpu_speed = 100u32;
    let mut sched = 100u32;
    for line in s.lines() {
        if let Some(rest) = line.split('=').nth(1) {
            let val = rest.trim().parse::<u32>().ok();
            if line.contains("CPU_Speed_Limit") {
                if let Some(v) = val {
                    cpu_speed = v;
                }
            } else if line.contains("CPU_Scheduler_Limit")
                && let Some(v) = val
            {
                sched = v;
            }
        }
    }
    Some(if cpu_speed < 100 || sched < 100 {
        ThermalState::Throttled {
            cpu_speed_pct: cpu_speed,
            sched_pct: sched,
        }
    } else {
        ThermalState::Nominal
    })
}

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
fn read_pmset_thermal() -> Option<ThermalState> {
    None
}

// ── Live GPU telemetry ───────────────────────────────────────────────
//
// Cross-backend, read-only, unprivileged GPU temperature / power / clock
// sampling. The concrete per-vendor readers live inside their backend
// crates as self-contained libloading shims (`rlx_cuda::nvml`,
// `rlx_rocm::rsmi`), exactly like `is_available` dispatches into each
// backend behind its Cargo feature. This keeps `Device` the only thing
// that knows the vendor mapping and adds no new cross-crate deps.
//
// Control (set power cap / lock clocks / fan) is deliberately absent: it
// needs root and is a separate follow-up.

/// A live, read-only GPU telemetry sample, normalised to friendly units.
///
/// Every reading is `Option` because boards differ in what they expose —
/// a discrete card (MI100, desktop RTX) reports a junction/hotspot sensor
/// and a settable power cap, while a laptop GPU or APU/iGPU omits the fan
/// reading, the junction sensor, or the cap. A `None` means "this board /
/// driver does not report it", never a fabricated zero.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuThermal {
    /// rlx backend family this sample came through (`Cuda` / `Rocm`).
    pub device: Device,
    /// Hardware ordinal within that backend (NVML index / rocm_smi dv_ind).
    pub index: u32,
    /// Board product name, if the driver reports one.
    pub name: Option<String>,
    /// Primary GPU-die / edge temperature, °C.
    pub temp_c: Option<f32>,
    /// Junction / hotspot temperature, °C (discrete AMD; `None` on NVIDIA/iGPU).
    pub hotspot_c: Option<f32>,
    /// VRAM temperature, °C (discrete cards only).
    pub mem_temp_c: Option<f32>,
    /// Instantaneous board power draw, W.
    pub power_w: Option<f32>,
    /// Currently enforced / configured power cap, W.
    pub power_cap_w: Option<f32>,
    /// Fan duty cycle, %.
    pub fan_percent: Option<f32>,
    /// Core/SM clock, MHz (NVIDIA SM clock; `None` on ROCm for now).
    pub clock_mhz: Option<u32>,
    /// GPU utilization, %.
    pub util_percent: Option<f32>,
}

impl GpuThermal {
    /// How close the board is to its power cap, in `[0, 1]` — a proxy for
    /// "am I power-limited right now". `None` when either reading is absent.
    pub fn power_headroom(&self) -> Option<f32> {
        match (self.power_w, self.power_cap_w) {
            (Some(p), Some(cap)) if cap > 0.0 => Some((p / cap).clamp(0.0, 1.0)),
            _ => None,
        }
    }
}

/// Number of monitorable GPUs for `device`'s backend. `0` when the
/// backend feature is disabled or its management library is absent.
pub fn device_thermal_count(device: Device) -> usize {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda => rlx_cuda::nvml::device_count(),
        #[cfg(feature = "rocm")]
        Device::Rocm => rlx_rocm::rsmi::device_count(),
        _ => {
            let _ = device;
            0
        }
    }
}

/// Read a one-shot telemetry sample for GPU `index` of `device`'s
/// backend. Returns `None` when the backend feature is off, the
/// management library is missing, or the index is out of range.
/// Read-only and unprivileged — safe to call from any thread at any time.
pub fn device_thermal(device: Device, index: u32) -> Option<GpuThermal> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let s = rlx_cuda::nvml::sample(index)?;
            Some(GpuThermal {
                device,
                index,
                name: s.name,
                temp_c: s.temp_c,
                hotspot_c: None,
                mem_temp_c: None,
                power_w: s.power_w,
                power_cap_w: s.power_cap_w,
                fan_percent: s.fan_percent,
                clock_mhz: s.sm_clock_mhz,
                util_percent: s.util_percent,
            })
        }
        #[cfg(feature = "rocm")]
        Device::Rocm => {
            let s = rlx_rocm::rsmi::sample(index)?;
            Some(GpuThermal {
                device,
                index,
                name: s.name,
                temp_c: s.temp_edge_c,
                hotspot_c: s.temp_hotspot_c,
                mem_temp_c: s.temp_mem_c,
                power_w: s.power_w,
                power_cap_w: s.power_cap_w,
                fan_percent: s.fan_percent,
                clock_mhz: None,
                util_percent: s.util_percent,
            })
        }
        _ => {
            let _ = (device, index);
            None
        }
    }
}

/// Sample every monitorable GPU across all compiled-in GPU backends
/// (CUDA + ROCm today). Empty when no GPU management library is present.
pub fn all_gpu_thermal() -> Vec<GpuThermal> {
    let mut out = Vec::new();
    for &device in &[Device::Cuda, Device::Rocm] {
        for index in 0..device_thermal_count(device) as u32 {
            if let Some(s) = device_thermal(device, index) {
                out.push(s);
            }
        }
    }
    out
}

// ── GPU control (privileged) ─────────────────────────────────────────
//
// Root-only knobs: set/reset power cap, locked clocks (NVIDIA), and fan.
// Reads above are unprivileged; these are not. The concrete drivers live
// in the same backend shims and return raw status codes; the runtime maps
// them to a typed `ThermalError` and pre-validates ranges so a caller
// can't drive a GPU outside its safe envelope.

/// Why a GPU control operation didn't take effect.
#[derive(Debug, Clone, PartialEq)]
pub enum ThermalError {
    /// Backend feature disabled, management library absent, or bad index.
    Unavailable,
    /// This board / driver doesn't expose the knob (e.g. laptop power-cap,
    /// ROCm clock-lock, a server GPU with no controllable fan).
    Unsupported,
    /// The operation needs elevated privileges — run as root / `sudo`.
    PermissionDenied,
    /// Requested value is outside the device's valid range.
    OutOfRange { requested: f32, min: f32, max: f32 },
    /// Driver returned a status we don't specifically classify.
    Driver(i32),
}

impl std::fmt::Display for ThermalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThermalError::Unavailable => write!(
                f,
                "device unavailable (backend feature off or management library absent)"
            ),
            ThermalError::Unsupported => write!(f, "not supported on this GPU/driver"),
            ThermalError::PermissionDenied => write!(f, "permission denied — run as root (sudo)"),
            ThermalError::OutOfRange {
                requested,
                min,
                max,
            } => write!(
                f,
                "requested {requested:.0} outside valid range {min:.0}..={max:.0}"
            ),
            ThermalError::Driver(n) => write!(f, "driver error (status {n})"),
        }
    }
}

impl std::error::Error for ThermalError {}

// Raw driver status → ThermalError. `-1`/`-2` are the shims' sentinels
// (unavailable / symbol-missing); the positive codes are vendor-specific.
#[cfg(feature = "cuda")]
fn map_nvml(rc: i32) -> ThermalError {
    match rc {
        -1 => ThermalError::Unavailable,
        -2 | 3 => ThermalError::Unsupported, // NVML_ERROR_NOT_SUPPORTED = 3
        4 => ThermalError::PermissionDenied, // NVML_ERROR_NO_PERMISSION = 4
        n => ThermalError::Driver(n),
    }
}

#[cfg(feature = "rocm")]
fn map_rsmi(rc: i32) -> ThermalError {
    match rc {
        -1 => ThermalError::Unavailable,
        // RSMI_STATUS_NOT_SUPPORTED = 2, NOT_YET_IMPLEMENTED = 9
        -2 | 2 | 9 => ThermalError::Unsupported,
        4 => ThermalError::PermissionDenied, // RSMI_STATUS_PERMISSION = 4
        n => ThermalError::Driver(n),
    }
}

/// Valid power-cap range `(min_w, max_w)` for `device`'s GPU `index`.
/// Read-only; `None` when unavailable. Use it to bound a [`set_power_cap`].
pub fn power_cap_range(device: Device, index: u32) -> Option<(f32, f32)> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda => rlx_cuda::nvml::power_cap_range(index),
        #[cfg(feature = "rocm")]
        Device::Rocm => rlx_rocm::rsmi::power_cap_range(index),
        _ => {
            let _ = (device, index);
            None
        }
    }
}

/// Set the sustained power cap (watts) for `device`'s GPU `index`.
///
/// Requires root. Rejected with [`ThermalError::OutOfRange`] if outside the
/// device's [`power_cap_range`], [`ThermalError::Unsupported`] on boards
/// that don't allow it (many laptop GPUs — use [`set_locked_clocks`]
/// there), or [`ThermalError::PermissionDenied`] without privileges.
pub fn set_power_cap(device: Device, index: u32, watts: f32) -> Result<(), ThermalError> {
    if let Some((min, max)) = power_cap_range(device, index) {
        // ±0.5 W tolerance so a boundary value isn't rejected on rounding.
        if watts + 0.5 < min || watts - 0.5 > max {
            return Err(ThermalError::OutOfRange {
                requested: watts,
                min,
                max,
            });
        }
    }
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda => rlx_cuda::nvml::set_power_cap(index, watts).map_err(map_nvml),
        #[cfg(feature = "rocm")]
        Device::Rocm => rlx_rocm::rsmi::set_power_cap(index, watts).map_err(map_rsmi),
        _ => {
            let _ = (device, index, watts);
            Err(ThermalError::Unsupported)
        }
    }
}

/// Pin `device`'s GPU `index` core clocks into `[min_mhz, max_mhz]`.
///
/// NVIDIA only (the effective lever on laptop parts that reject a power
/// cap). ROCm returns [`ThermalError::Unsupported`] — its clock control
/// needs perf-level=MANUAL + a frequency bitmask; use [`set_power_cap`]
/// on the MI100 instead. Requires root.
pub fn set_locked_clocks(
    device: Device,
    index: u32,
    min_mhz: u32,
    max_mhz: u32,
) -> Result<(), ThermalError> {
    let _ = (index, min_mhz, max_mhz);
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            rlx_cuda::nvml::set_locked_clocks(index, min_mhz, max_mhz).map_err(map_nvml)
        }
        _ => Err(ThermalError::Unsupported),
    }
}

/// Release a previous [`set_locked_clocks`] (back to automatic). NVIDIA
/// only; requires root.
pub fn reset_locked_clocks(device: Device, index: u32) -> Result<(), ThermalError> {
    let _ = index;
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda => rlx_cuda::nvml::reset_locked_clocks(index).map_err(map_nvml),
        _ => Err(ThermalError::Unsupported),
    }
}

/// Set fan duty (%) on `device`'s GPU `index`; enters manual fan mode
/// until [`reset_fan`]. Requires root; `Unsupported` on GPUs with no
/// controllable fan (many servers / laptops).
pub fn set_fan_percent(device: Device, index: u32, percent: f32) -> Result<(), ThermalError> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda => rlx_cuda::nvml::set_fan_percent(index, percent).map_err(map_nvml),
        #[cfg(feature = "rocm")]
        Device::Rocm => rlx_rocm::rsmi::set_fan_percent(index, percent).map_err(map_rsmi),
        _ => {
            let _ = (device, index, percent);
            Err(ThermalError::Unsupported)
        }
    }
}

/// Return `device`'s GPU `index` fan to automatic control. Requires root.
pub fn reset_fan(device: Device, index: u32) -> Result<(), ThermalError> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda => rlx_cuda::nvml::reset_fan(index).map_err(map_nvml),
        #[cfg(feature = "rocm")]
        Device::Rocm => rlx_rocm::rsmi::reset_fan(index).map_err(map_rsmi),
        _ => {
            let _ = (device, index);
            Err(ThermalError::Unsupported)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_thermal_never_panics() {
        // Must be graceful on hosts with no NVIDIA/AMD management lib
        // (Mac/CI): empty inventory, no reads, no panic.
        let all = all_gpu_thermal();
        for s in &all {
            // If we got a sample, the ordinal must be addressable again.
            assert!(device_thermal(s.device, s.index).is_some());
        }
        // Counts are non-negative by type; calling is enough to exercise
        // the dispatch under whatever features are enabled.
        let _ = device_thermal_count(Device::Cuda);
        let _ = device_thermal_count(Device::Rocm);
    }

    #[test]
    fn control_is_graceful_without_hardware() {
        // On a host without the management lib, every control call must
        // return a typed error (never panic). CPU/GPU-family devices with
        // no thermal backend → Unsupported; the specific vendor error only
        // shows up on real hardware.
        for &d in &[Device::Cpu, Device::Metal, Device::Cuda, Device::Rocm] {
            let _ = power_cap_range(d, 0);
            assert!(set_power_cap(d, 0, 100.0).is_err() || crate::is_available(d));
            assert!(set_fan_percent(d, 0, 50.0).is_err() || crate::is_available(d));
            let _ = reset_fan(d, 0);
            let _ = reset_locked_clocks(d, 0);
        }
        // ROCm clock-lock is intentionally unsupported everywhere.
        assert_eq!(
            set_locked_clocks(Device::Rocm, 0, 1000, 1000),
            Err(ThermalError::Unsupported)
        );
    }

    #[test]
    fn thermal_error_display_is_actionable() {
        assert!(ThermalError::PermissionDenied.to_string().contains("root"));
        let oor = ThermalError::OutOfRange {
            requested: 400.0,
            min: 100.0,
            max: 290.0,
        };
        let s = oor.to_string();
        assert!(s.contains("400") && s.contains("290"));
    }

    #[test]
    fn snapshot_doesnt_panic() {
        let snap = HwSnapshot::collect();
        // The OS / arch fields are always set.
        assert!(!snap.os.is_empty());
        assert!(!snap.arch.is_empty());
    }

    #[test]
    fn fingerprint_is_stable_across_collects() {
        // Two collects on the same machine must agree on fingerprint
        // (thermal state is excluded).
        let a = HwSnapshot::collect();
        let b = HwSnapshot::collect();
        assert_eq!(a.fingerprint(), b.fingerprint());
    }
}
