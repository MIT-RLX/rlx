// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ROCm-SMI shim — AMD GPU read-only telemetry.
//!
//! Sister to `roctx.rs`: dlopen `librocm_smi64.so` at runtime via
//! libloading so the crate still compiles + tests on hosts without
//! ROCm (Mac, CI). `RsmiRuntime::load()` returns `None` when the
//! library is absent; each per-metric read returns `None` when the
//! driver reports a non-success status for *that* query. AMD boards
//! differ widely in which sensors they expose — a discrete card
//! (e.g. MI100) reports junction/hotspot + average power + a settable
//! cap, while an APU/iGPU reports only edge temp + socket power — so
//! we never fabricate a value the board doesn't provide.
//!
//! `librocm_smi64` is index-based (`dv_ind`) and covers every AMD GPU
//! the driver enumerates, including a discrete + integrated pair on the
//! same host. `libamd_smi` is the forward-looking replacement; we start
//! on rocm_smi because its scalar API is simpler and avoids the
//! version-fragile `rsmi_frequencies_t` struct layout.
//!
//! Read-only + unprivileged. Control (set power cap / clocks / fan)
//! needs root and is a deliberate follow-up.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{CStr, c_char, c_int, c_uint};
use std::sync::Arc;
use std::sync::OnceLock;

use libloading::Library;

// rsmi_status_t: 0 == RSMI_STATUS_SUCCESS. Any non-zero → the metric is
// unavailable/unsupported for this device; we leave the field `None`.
type FnInit = unsafe extern "C" fn(u64) -> c_int; // rsmi_init
type FnNumDev = unsafe extern "C" fn(*mut c_uint) -> c_int; // rsmi_num_monitor_devices
type FnName = unsafe extern "C" fn(c_uint, *mut c_char, usize) -> c_int; // rsmi_dev_name_get
type FnTemp = unsafe extern "C" fn(c_uint, c_uint, c_uint, *mut i64) -> c_int; // rsmi_dev_temp_metric_get
type FnPowerAve = unsafe extern "C" fn(c_uint, c_uint, *mut u64) -> c_int; // rsmi_dev_power_ave_get
type FnSocketPower = unsafe extern "C" fn(c_uint, *mut u64) -> c_int; // rsmi_dev_current_socket_power_get
type FnPowerCap = unsafe extern "C" fn(c_uint, c_uint, *mut u64) -> c_int; // rsmi_dev_power_cap_get
type FnFan = unsafe extern "C" fn(c_uint, c_uint, *mut i64) -> c_int; // rsmi_dev_fan_speed_get
type FnBusy = unsafe extern "C" fn(c_uint, *mut c_uint) -> c_int; // rsmi_dev_busy_percent_get

// ── Control (privileged; require root) ───────────────────────────────
// NOTE: rsmi_dev_power_cap_range_get fills (max, min) in THAT order.
type FnPowerCapRange = unsafe extern "C" fn(c_uint, c_uint, *mut u64, *mut u64) -> c_int; // rsmi_dev_power_cap_range_get
type FnPowerCapSet = unsafe extern "C" fn(c_uint, c_uint, u64) -> c_int; // rsmi_dev_power_cap_set
type FnFanSet = unsafe extern "C" fn(c_uint, c_uint, u64) -> c_int; // rsmi_dev_fan_speed_set
type FnFanReset = unsafe extern "C" fn(c_uint, c_uint) -> c_int; // rsmi_dev_fan_reset

/// Runtime/device unavailable (lib absent or bad index) — negative so it
/// never collides with a real `rsmi_status_t` (all non-negative).
pub const RSMI_ERR_UNAVAILABLE: i32 = -1;
/// The requested control symbol isn't present in this rocm_smi build.
pub const RSMI_ERR_NO_SYMBOL: i32 = -2;

// rocm_smi enum constants we use.
const RSMI_TEMP_TYPE_EDGE: c_uint = 0;
const RSMI_TEMP_TYPE_JUNCTION: c_uint = 1;
const RSMI_TEMP_TYPE_MEMORY: c_uint = 2;
const RSMI_TEMP_CURRENT: c_uint = 0;
/// Raw fan-speed reads are 0..=255 (`RSMI_MAX_FAN_SPEED`).
const RSMI_MAX_FAN_SPEED: f32 = 255.0;

/// Resolved rocm_smi entry points. `temp`/`num_dev` are required; the
/// rest are `Option` so a device/driver missing one symbol still yields
/// the primary telemetry.
pub struct RsmiRuntime {
    _lib: Library,
    num_dev: FnNumDev,
    temp: FnTemp,
    name: Option<FnName>,
    power_ave: Option<FnPowerAve>,
    socket_power: Option<FnSocketPower>,
    power_cap: Option<FnPowerCap>,
    fan: Option<FnFan>,
    busy: Option<FnBusy>,
    power_cap_range: Option<FnPowerCapRange>,
    power_cap_set: Option<FnPowerCapSet>,
    fan_set: Option<FnFanSet>,
    fan_reset: Option<FnFanReset>,
}

unsafe impl Send for RsmiRuntime {}
unsafe impl Sync for RsmiRuntime {}

impl RsmiRuntime {
    fn load() -> Option<Arc<Self>> {
        unsafe {
            let lib = Library::new("librocm_smi64.so")
                .or_else(|_| Library::new("librocm_smi64.so.7"))
                .or_else(|_| Library::new("librocm_smi64.so.5"))
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
            // rsmi_init(0) must precede any query. Init once per process.
            let init: FnInit = req!(b"rsmi_init", FnInit);
            if init(0) != 0 {
                return None;
            }
            let rt = RsmiRuntime {
                num_dev: req!(b"rsmi_num_monitor_devices", FnNumDev),
                temp: req!(b"rsmi_dev_temp_metric_get", FnTemp),
                name: opt!(b"rsmi_dev_name_get", FnName),
                power_ave: opt!(b"rsmi_dev_power_ave_get", FnPowerAve),
                socket_power: opt!(b"rsmi_dev_current_socket_power_get", FnSocketPower),
                power_cap: opt!(b"rsmi_dev_power_cap_get", FnPowerCap),
                fan: opt!(b"rsmi_dev_fan_speed_get", FnFan),
                busy: opt!(b"rsmi_dev_busy_percent_get", FnBusy),
                power_cap_range: opt!(b"rsmi_dev_power_cap_range_get", FnPowerCapRange),
                power_cap_set: opt!(b"rsmi_dev_power_cap_set", FnPowerCapSet),
                fan_set: opt!(b"rsmi_dev_fan_speed_set", FnFanSet),
                fan_reset: opt!(b"rsmi_dev_fan_reset", FnFanReset),
                _lib: lib,
            };
            Some(Arc::new(rt))
        }
    }

    /// Read one temperature sensor (°C), `None` if that sensor is absent
    /// on this board (e.g. iGPUs have no junction/memory sensor).
    unsafe fn temp_c(&self, dv: c_uint, sensor: c_uint) -> Option<f32> {
        let mut milli: i64 = 0;
        let rc = unsafe { (self.temp)(dv, sensor, RSMI_TEMP_CURRENT, &mut milli) };
        if rc == 0 {
            Some(milli as f32 / 1000.0)
        } else {
            None
        }
    }
}

/// Process-wide rocm_smi runtime. `None` when `librocm_smi64` isn't
/// loadable.
fn runtime() -> Option<Arc<RsmiRuntime>> {
    static RUNTIME: OnceLock<Option<Arc<RsmiRuntime>>> = OnceLock::new();
    RUNTIME.get_or_init(RsmiRuntime::load).clone()
}

/// One read-only telemetry reading in friendly units.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RsmiSample {
    pub name: Option<String>,
    /// Edge temperature, °C (the sensor `rocm-smi --showtemp` calls "edge").
    pub temp_edge_c: Option<f32>,
    /// Junction / hotspot temperature, °C (discrete cards only).
    pub temp_hotspot_c: Option<f32>,
    /// VRAM temperature, °C (discrete cards only).
    pub temp_mem_c: Option<f32>,
    /// Board power draw, W (average package power, or socket power on APUs).
    pub power_w: Option<f32>,
    /// Configured power cap, W.
    pub power_cap_w: Option<f32>,
    /// Fan duty cycle, % (raw 0..255 normalised).
    pub fan_percent: Option<f32>,
    /// GPU utilization, %.
    pub util_percent: Option<f32>,
}

/// Number of AMD GPUs rocm_smi can monitor (0 when unavailable).
pub fn device_count() -> usize {
    let Some(rt) = runtime() else {
        return 0;
    };
    let mut n: c_uint = 0;
    unsafe {
        if (rt.num_dev)(&mut n) == 0 {
            n as usize
        } else {
            0
        }
    }
}

/// Read a one-shot telemetry sample for AMD GPU `index` (rocm_smi
/// `dv_ind`). `None` when rocm_smi is unavailable.
pub fn sample(index: u32) -> Option<RsmiSample> {
    let rt = runtime()?;
    let dv = index as c_uint;
    unsafe {
        if dv >= device_count() as c_uint {
            return None;
        }

        let mut s = RsmiSample::default();

        if let Some(name) = rt.name {
            let mut buf = [0u8; 96];
            if name(dv, buf.as_mut_ptr() as *mut c_char, buf.len()) == 0 {
                if let Ok(cs) = CStr::from_ptr(buf.as_ptr() as *const c_char).to_str() {
                    let cs = cs.trim();
                    if !cs.is_empty() {
                        s.name = Some(cs.to_string());
                    }
                }
            }
        }

        s.temp_edge_c = rt.temp_c(dv, RSMI_TEMP_TYPE_EDGE);
        s.temp_hotspot_c = rt.temp_c(dv, RSMI_TEMP_TYPE_JUNCTION);
        s.temp_mem_c = rt.temp_c(dv, RSMI_TEMP_TYPE_MEMORY);

        // Prefer average package power; fall back to current socket power
        // for APUs/iGPUs that don't implement the average sensor. Both
        // report microwatts.
        if let Some(power_ave) = rt.power_ave {
            let mut uw: u64 = 0;
            if power_ave(dv, 0, &mut uw) == 0 && uw > 0 {
                s.power_w = Some(uw as f32 / 1_000_000.0);
            }
        }
        if s.power_w.is_none() {
            if let Some(socket_power) = rt.socket_power {
                let mut uw: u64 = 0;
                if socket_power(dv, &mut uw) == 0 && uw > 0 {
                    s.power_w = Some(uw as f32 / 1_000_000.0);
                }
            }
        }

        if let Some(power_cap) = rt.power_cap {
            let mut uw: u64 = 0;
            if power_cap(dv, 0, &mut uw) == 0 && uw > 0 {
                s.power_cap_w = Some(uw as f32 / 1_000_000.0);
            }
        }

        if let Some(fan) = rt.fan {
            let mut raw: i64 = 0;
            if fan(dv, 0, &mut raw) == 0 {
                s.fan_percent = Some((raw as f32 / RSMI_MAX_FAN_SPEED * 100.0).clamp(0.0, 100.0));
            }
        }

        if let Some(busy) = rt.busy {
            let mut b: c_uint = 0;
            if busy(dv, &mut b) == 0 {
                s.util_percent = Some(b as f32);
            }
        }

        Some(s)
    }
}

// ── Control (privileged) ─────────────────────────────────────────────
//
// Each returns `Ok(())` on success, or `Err(code)` where `code` is a raw
// `rsmi_status_t` (RSMI_STATUS_PERMISSION=4 when not root,
// RSMI_STATUS_NOT_SUPPORTED=2 on boards without the knob) or a negative
// sentinel above. The runtime maps these to a typed `ThermalError`.
//
// Clock control is intentionally NOT here: on ROCm it needs
// perf-level=MANUAL plus a frequency-index bitmask via the
// version-fragile `rsmi_frequencies_t` struct. The MI100's lever is its
// power cap; the runtime reports ROCm clock-lock as `Unsupported`.

/// Allowed power-cap range for GPU `index`, `(min_w, max_w)`. `None` when
/// rocm_smi / the range query is unavailable.
pub fn power_cap_range(index: u32) -> Option<(f32, f32)> {
    let rt = runtime()?;
    let range = rt.power_cap_range?;
    if (index as usize) >= device_count() {
        return None;
    }
    // rocm_smi fills (max, min) — note the order.
    let (mut max_uw, mut min_uw): (u64, u64) = (0, 0);
    let rc = unsafe { range(index as c_uint, 0, &mut max_uw, &mut min_uw) };
    if rc == 0 {
        Some((min_uw as f32 / 1_000_000.0, max_uw as f32 / 1_000_000.0))
    } else {
        None
    }
}

/// Set the sustained power cap for GPU `index` (watts → µW).
pub fn set_power_cap(index: u32, watts: f32) -> Result<(), i32> {
    let rt = runtime().ok_or(RSMI_ERR_UNAVAILABLE)?;
    let set = rt.power_cap_set.ok_or(RSMI_ERR_NO_SYMBOL)?;
    if (index as usize) >= device_count() {
        return Err(RSMI_ERR_UNAVAILABLE);
    }
    let uw = (watts as f64 * 1_000_000.0).round().max(0.0) as u64;
    match unsafe { set(index as c_uint, 0, uw) } {
        0 => Ok(()),
        rc => Err(rc),
    }
}

/// Set fan 0 of GPU `index` to `percent` duty (mapped to the 0..255 raw
/// scale). Enters manual fan mode until [`reset_fan`].
pub fn set_fan_percent(index: u32, percent: f32) -> Result<(), i32> {
    let rt = runtime().ok_or(RSMI_ERR_UNAVAILABLE)?;
    let set = rt.fan_set.ok_or(RSMI_ERR_NO_SYMBOL)?;
    if (index as usize) >= device_count() {
        return Err(RSMI_ERR_UNAVAILABLE);
    }
    let raw = (percent.clamp(0.0, 100.0) / 100.0 * RSMI_MAX_FAN_SPEED).round() as u64;
    match unsafe { set(index as c_uint, 0, raw) } {
        0 => Ok(()),
        rc => Err(rc),
    }
}

/// Return fan 0 of GPU `index` to driver (automatic) control.
pub fn reset_fan(index: u32) -> Result<(), i32> {
    let rt = runtime().ok_or(RSMI_ERR_UNAVAILABLE)?;
    let reset = rt.fan_reset.ok_or(RSMI_ERR_NO_SYMBOL)?;
    if (index as usize) >= device_count() {
        return Err(RSMI_ERR_UNAVAILABLE);
    }
    match unsafe { reset(index as c_uint, 0) } {
        0 => Ok(()),
        rc => Err(rc),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Must not panic on any host — on Mac/CI `librocm_smi64` is absent.
    #[test]
    fn count_and_sample_never_panic() {
        let n = device_count();
        let s = sample(0);
        if n == 0 {
            assert!(s.is_none());
        }
    }
}
