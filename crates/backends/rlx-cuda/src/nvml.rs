// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NVML shim — NVIDIA Management Library, read-only GPU telemetry.
//!
//! Same shape as `roctx.rs` / the `preload_real_cudnn` dlopen in
//! `device.rs`: resolve `libnvidia-ml.so.1` at runtime via libloading
//! so the crate still compiles + tests on hosts without an NVIDIA
//! driver (Mac, CI). `NvmlRuntime::load()` returns `None` when the
//! library is absent, and each per-metric read returns `None` when the
//! driver reports a non-success status for *that* query — laptop GPUs
//! routinely lack a fan reading or a settable power cap, so we never
//! fabricate a value for a sensor the board doesn't expose.
//!
//! `cudarc` (our CUDA runtime) does **not** enable its `nvml` feature,
//! and NVML's init is refcounted, so this independent dlopen is safe
//! alongside the live cuBLAS/NVRTC context.
//!
//! Read-only + unprivileged by design. Control (set power limit / lock
//! clocks) needs root and is a deliberate follow-up behind its own
//! entry points.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::Arc;
use std::sync::OnceLock;

use libloading::Library;

/// Opaque `nvmlDevice_t` (really `struct nvmlDevice_st*`).
type NvmlDevice = *mut c_void;

// nvmlReturn_t is an int enum; 0 == NVML_SUCCESS. We treat any non-zero
// as "metric unavailable" and leave the field `None` — for read-only
// sampling the exact error code (NOT_SUPPORTED=3, NO_PERMISSION=4)
// doesn't change what we do.
type FnInit = unsafe extern "C" fn() -> c_int; // nvmlInit_v2
type FnCount = unsafe extern "C" fn(*mut c_uint) -> c_int; // nvmlDeviceGetCount_v2
type FnHandle = unsafe extern "C" fn(c_uint, *mut NvmlDevice) -> c_int; // nvmlDeviceGetHandleByIndex_v2
type FnName = unsafe extern "C" fn(NvmlDevice, *mut c_char, c_uint) -> c_int; // nvmlDeviceGetName
type FnTemp = unsafe extern "C" fn(NvmlDevice, c_uint, *mut c_uint) -> c_int; // nvmlDeviceGetTemperature
type FnPower = unsafe extern "C" fn(NvmlDevice, *mut c_uint) -> c_int; // nvmlDeviceGetPowerUsage
type FnPowerLimit = unsafe extern "C" fn(NvmlDevice, *mut c_uint) -> c_int; // nvmlDeviceGetEnforcedPowerLimit
type FnFan = unsafe extern "C" fn(NvmlDevice, *mut c_uint) -> c_int; // nvmlDeviceGetFanSpeed
type FnClock = unsafe extern "C" fn(NvmlDevice, c_uint, *mut c_uint) -> c_int; // nvmlDeviceGetClockInfo

#[repr(C)]
struct NvmlUtil {
    gpu: c_uint,
    memory: c_uint,
}
type FnUtil = unsafe extern "C" fn(NvmlDevice, *mut NvmlUtil) -> c_int; // nvmlDeviceGetUtilizationRates

// ── Control (privileged; require root) ───────────────────────────────
type FnPowerConstraints = unsafe extern "C" fn(NvmlDevice, *mut c_uint, *mut c_uint) -> c_int; // nvmlDeviceGetPowerManagementLimitConstraints
type FnSetPowerLimit = unsafe extern "C" fn(NvmlDevice, c_uint) -> c_int; // nvmlDeviceSetPowerManagementLimit
type FnSetLockedClocks = unsafe extern "C" fn(NvmlDevice, c_uint, c_uint) -> c_int; // nvmlDeviceSetGpuLockedClocks
type FnResetLockedClocks = unsafe extern "C" fn(NvmlDevice) -> c_int; // nvmlDeviceResetGpuLockedClocks
type FnSetFan = unsafe extern "C" fn(NvmlDevice, c_uint, c_uint) -> c_int; // nvmlDeviceSetFanSpeed_v2
type FnResetFan = unsafe extern "C" fn(NvmlDevice, c_uint) -> c_int; // nvmlDeviceSetDefaultFanSpeed_v2

// NVML enum constants we use.
const NVML_TEMPERATURE_GPU: c_uint = 0;
const NVML_CLOCK_SM: c_uint = 1;

// Negative sentinels the control fns return before any driver call is made
// (real nvmlReturn_t codes are non-negative), so the runtime can tell
// "library/device absent" and "symbol missing" apart from a driver error.
/// Runtime/device handle unavailable (lib absent or bad index).
pub const NVML_ERR_UNAVAILABLE: i32 = -1;
/// The requested control symbol isn't present in this NVML build.
pub const NVML_ERR_NO_SYMBOL: i32 = -2;

/// Resolved NVML entry points. Core reads (`count`/`handle`/`temp`/
/// `power`) are required; the rest are `Option` so a driver that ships
/// an older NVML without one symbol still gives us the primary
/// telemetry instead of failing the whole load.
pub struct NvmlRuntime {
    _lib: Library,
    count: FnCount,
    handle: FnHandle,
    temp: FnTemp,
    power: FnPower,
    name: Option<FnName>,
    power_limit: Option<FnPowerLimit>,
    fan: Option<FnFan>,
    clock: Option<FnClock>,
    util: Option<FnUtil>,
    power_constraints: Option<FnPowerConstraints>,
    set_power_limit: Option<FnSetPowerLimit>,
    set_locked_clocks: Option<FnSetLockedClocks>,
    reset_locked_clocks: Option<FnResetLockedClocks>,
    set_fan: Option<FnSetFan>,
    reset_fan: Option<FnResetFan>,
}

unsafe impl Send for NvmlRuntime {}
unsafe impl Sync for NvmlRuntime {}

impl NvmlRuntime {
    fn load() -> Option<Arc<Self>> {
        unsafe {
            let lib = Library::new("libnvidia-ml.so.1")
                .or_else(|_| Library::new("libnvidia-ml.so"))
                .or_else(|_| Library::new("nvml.dll"))
                .ok()?;
            macro_rules! req {
                ($name:literal, $ty:ty) => {{
                    let s: libloading::Symbol<$ty> = lib.get($name).ok()?;
                    *s.into_raw()
                }};
            }
            macro_rules! opt {
                ($name:literal, $ty:ty) => {
                    lib.get::<$ty>($name).ok().map(|s| *s.into_raw())
                };
            }
            // NVML must be initialised before any device query. Do it once
            // for the process; the OS reclaims on exit (no shutdown needed
            // for a monitor).
            let init: FnInit = req!(b"nvmlInit_v2", FnInit);
            if init() != 0 {
                return None;
            }
            let rt = NvmlRuntime {
                count: req!(b"nvmlDeviceGetCount_v2", FnCount),
                handle: req!(b"nvmlDeviceGetHandleByIndex_v2", FnHandle),
                temp: req!(b"nvmlDeviceGetTemperature", FnTemp),
                power: req!(b"nvmlDeviceGetPowerUsage", FnPower),
                name: opt!(b"nvmlDeviceGetName", FnName),
                power_limit: opt!(b"nvmlDeviceGetEnforcedPowerLimit", FnPowerLimit),
                fan: opt!(b"nvmlDeviceGetFanSpeed", FnFan),
                clock: opt!(b"nvmlDeviceGetClockInfo", FnClock),
                util: opt!(b"nvmlDeviceGetUtilizationRates", FnUtil),
                power_constraints: opt!(
                    b"nvmlDeviceGetPowerManagementLimitConstraints",
                    FnPowerConstraints
                ),
                set_power_limit: opt!(b"nvmlDeviceSetPowerManagementLimit", FnSetPowerLimit),
                set_locked_clocks: opt!(b"nvmlDeviceSetGpuLockedClocks", FnSetLockedClocks),
                reset_locked_clocks: opt!(b"nvmlDeviceResetGpuLockedClocks", FnResetLockedClocks),
                set_fan: opt!(b"nvmlDeviceSetFanSpeed_v2", FnSetFan),
                reset_fan: opt!(b"nvmlDeviceSetDefaultFanSpeed_v2", FnResetFan),
                _lib: lib,
            };
            Some(Arc::new(rt))
        }
    }
}

/// Process-wide NVML runtime. `None` when `libnvidia-ml` isn't loadable.
fn runtime() -> Option<Arc<NvmlRuntime>> {
    static RUNTIME: OnceLock<Option<Arc<NvmlRuntime>>> = OnceLock::new();
    RUNTIME.get_or_init(NvmlRuntime::load).clone()
}

/// One read-only telemetry reading in friendly units. Every field is
/// `None` when the driver doesn't expose that sensor for this GPU.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NvmlSample {
    pub name: Option<String>,
    /// GPU die temperature, °C.
    pub temp_c: Option<f32>,
    /// Instantaneous board power draw, W.
    pub power_w: Option<f32>,
    /// Currently enforced power cap, W (often absent on laptop GPUs).
    pub power_cap_w: Option<f32>,
    /// Fan duty cycle, %.
    pub fan_percent: Option<f32>,
    /// SM clock, MHz.
    pub sm_clock_mhz: Option<u32>,
    /// GPU utilization, %.
    pub util_percent: Option<f32>,
}

/// Number of NVIDIA GPUs NVML can see (0 when NVML is unavailable).
pub fn device_count() -> usize {
    let Some(rt) = runtime() else {
        return 0;
    };
    let mut n: c_uint = 0;
    unsafe {
        if (rt.count)(&mut n) == 0 {
            n as usize
        } else {
            0
        }
    }
}

/// Read a one-shot telemetry sample for GPU `index`. `None` when NVML
/// is unavailable or the index is out of range.
pub fn sample(index: u32) -> Option<NvmlSample> {
    let rt = runtime()?;
    unsafe {
        let mut dev: NvmlDevice = std::ptr::null_mut();
        if (rt.handle)(index as c_uint, &mut dev) != 0 {
            return None;
        }

        let mut s = NvmlSample::default();

        if let Some(name) = rt.name {
            let mut buf = [0u8; 96];
            if name(dev, buf.as_mut_ptr() as *mut c_char, buf.len() as c_uint) == 0 {
                if let Ok(cs) = CStr::from_ptr(buf.as_ptr() as *const c_char).to_str() {
                    s.name = Some(cs.to_string());
                }
            }
        }

        let mut t: c_uint = 0;
        if (rt.temp)(dev, NVML_TEMPERATURE_GPU, &mut t) == 0 {
            s.temp_c = Some(t as f32);
        }

        // nvmlDeviceGetPowerUsage reports milliwatts.
        let mut p: c_uint = 0;
        if (rt.power)(dev, &mut p) == 0 {
            s.power_w = Some(p as f32 / 1000.0);
        }

        if let Some(power_limit) = rt.power_limit {
            let mut pl: c_uint = 0;
            if power_limit(dev, &mut pl) == 0 {
                s.power_cap_w = Some(pl as f32 / 1000.0);
            }
        }

        if let Some(fan) = rt.fan {
            let mut f: c_uint = 0;
            if fan(dev, &mut f) == 0 {
                s.fan_percent = Some(f as f32);
            }
        }

        if let Some(clock) = rt.clock {
            let mut c: c_uint = 0;
            if clock(dev, NVML_CLOCK_SM, &mut c) == 0 {
                s.sm_clock_mhz = Some(c);
            }
        }

        if let Some(util) = rt.util {
            let mut u = NvmlUtil { gpu: 0, memory: 0 };
            if util(dev, &mut u) == 0 {
                s.util_percent = Some(u.gpu as f32);
            }
        }

        Some(s)
    }
}

// ── Control (privileged) ─────────────────────────────────────────────
//
// Each returns `Ok(())` on success, or `Err(code)` where `code` is a raw
// `nvmlReturn_t` (NVML_ERROR_NO_PERMISSION=4 when not root,
// NVML_ERROR_NOT_SUPPORTED=3 on GPUs that reject the knob — laptop parts
// commonly reject power-cap and fan control) or one of the negative
// sentinels above. The runtime maps these to a typed `ThermalError`.

/// Resolve the runtime + device handle for `index`, then run `f`.
fn with_device<F>(index: u32, f: F) -> Result<(), i32>
where
    F: FnOnce(&NvmlRuntime, NvmlDevice) -> Result<(), i32>,
{
    let rt = runtime().ok_or(NVML_ERR_UNAVAILABLE)?;
    unsafe {
        let mut dev: NvmlDevice = std::ptr::null_mut();
        if (rt.handle)(index as c_uint, &mut dev) != 0 {
            return Err(NVML_ERR_UNAVAILABLE);
        }
        f(&rt, dev)
    }
}

/// Allowed power-cap range for GPU `index`, `(min_w, max_w)`. `None` when
/// NVML / the constraints query is unavailable.
pub fn power_cap_range(index: u32) -> Option<(f32, f32)> {
    let rt = runtime()?;
    let constraints = rt.power_constraints?;
    unsafe {
        let mut dev: NvmlDevice = std::ptr::null_mut();
        if (rt.handle)(index as c_uint, &mut dev) != 0 {
            return None;
        }
        let (mut lo, mut hi): (c_uint, c_uint) = (0, 0);
        if constraints(dev, &mut lo, &mut hi) == 0 {
            Some((lo as f32 / 1000.0, hi as f32 / 1000.0))
        } else {
            None
        }
    }
}

/// Set the sustained power cap for GPU `index` (watts → mW).
pub fn set_power_cap(index: u32, watts: f32) -> Result<(), i32> {
    with_device(index, |rt, dev| {
        let set = rt.set_power_limit.ok_or(NVML_ERR_NO_SYMBOL)?;
        let mw = (watts * 1000.0).round().max(0.0) as c_uint;
        match unsafe { set(dev, mw) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    })
}

/// Pin GPU `index` core clocks into `[min_mhz, max_mhz]`. On laptop GPUs
/// that reject a power cap, this is the effective thermal/power lever.
pub fn set_locked_clocks(index: u32, min_mhz: u32, max_mhz: u32) -> Result<(), i32> {
    with_device(index, |rt, dev| {
        let set = rt.set_locked_clocks.ok_or(NVML_ERR_NO_SYMBOL)?;
        match unsafe { set(dev, min_mhz as c_uint, max_mhz as c_uint) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    })
}

/// Release a previous [`set_locked_clocks`] on GPU `index` (back to auto).
pub fn reset_locked_clocks(index: u32) -> Result<(), i32> {
    with_device(index, |rt, dev| {
        let reset = rt.reset_locked_clocks.ok_or(NVML_ERR_NO_SYMBOL)?;
        match unsafe { reset(dev) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    })
}

/// Set fan 0 of GPU `index` to `percent` duty (enters manual fan mode
/// until [`reset_fan`]).
pub fn set_fan_percent(index: u32, percent: f32) -> Result<(), i32> {
    with_device(index, |rt, dev| {
        let set = rt.set_fan.ok_or(NVML_ERR_NO_SYMBOL)?;
        let pct = percent.round().clamp(0.0, 100.0) as c_uint;
        match unsafe { set(dev, 0, pct) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    })
}

/// Return fan 0 of GPU `index` to driver (automatic) control.
pub fn reset_fan(index: u32) -> Result<(), i32> {
    with_device(index, |rt, dev| {
        let reset = rt.reset_fan.ok_or(NVML_ERR_NO_SYMBOL)?;
        match unsafe { reset(dev, 0) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loads cleanly (or cleanly reports absence) on every host —
    /// on Mac/CI `libnvidia-ml` is missing, so this must not panic.
    #[test]
    fn count_and_sample_never_panic() {
        let n = device_count();
        // Sample index 0 regardless — must be graceful when NVML is absent.
        let s = sample(0);
        if n == 0 {
            assert!(s.is_none());
        }
    }
}
