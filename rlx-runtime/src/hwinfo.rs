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

//! Hardware introspection (plan #47).
//!
//! Apple-side equivalent of MAX's `nvml.mojo` / `device_query.mojo`.
//! Exposes CPU / GPU details and current thermal state. The
//! autotuner / calibrator uses [`HwSnapshot::fingerprint`] to key
//! its cache so calibration data is invalidated when the user moves
//! a workspace between Macs (or even when the OS re-classifies
//! cores after a SoC update).
//!
//! Pure read-only queries; no shell-out beyond `pmset -g therm`
//! (which we already invoke from `scripts/check-throttle.sh`).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::process::Command;

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

        #[cfg(target_os = "macos")]
        {
            snap.cpu_brand = sysctl_str("machdep.cpu.brand_string").unwrap_or_default();
            snap.perf_cores = sysctl_usize("hw.perflevel0.physicalcpu").unwrap_or(0);
            snap.l1d_bytes = sysctl_usize("hw.l1dcachesize").unwrap_or(0);
            snap.l2_bytes = sysctl_usize("hw.l2cachesize").unwrap_or(0);
            snap.cache_line = sysctl_usize("hw.cachelinesize").unwrap_or(0);
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

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
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

#[cfg(test)]
mod tests {
    use super::*;

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
