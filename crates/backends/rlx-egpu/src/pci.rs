// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PCI discovery — the IOKit registry on macOS, `/sys/bus/pci` on Linux.
//!
//! PCIe tunnelling over USB4/Thunderbolt works on Apple Silicon: a device behind
//! the tunnel enumerates as a regular `IOPCIDevice` with `IOPCITunnelled = true`.
//! What macOS does not ship on arm64 is a driver for base class 0x03 (display),
//! so a GPU arrives on the bus, enumerates, and is left unclaimed.
//!
//! Linux enumerates the same facts under `/sys/bus/pci/devices/*`: `vendor`,
//! `device`, `class`, and two signals for "arrived from outside the box".
//! `untrusted` is the precise one — the kernel sets it for a device behind an
//! external-facing port — but it depends on firmware marking the port that way
//! (ACPI `ExternalFacingPort`), and it was measured **absent entirely** on two
//! different Linux hosts. Relying on it alone reports every device as internal,
//! so a hotplug-slot ancestor is checked as well; a Thunderbolt/USB4 tunnel is
//! always presented behind a hotplug-capable downstream port.
//!
//! Neither signal detects a **directly-cabled** external GPU — OCuLink and the
//! like present an ordinary PCIe root port, with no tunnel, no `untrusted`, and
//! no hotplug slot. Such a card is indistinguishable from one in a slot, and is
//! out of scope by design: on Linux the vendor stack drives it, and an Apple
//! Silicon host has no PCIe egress other than USB4/Thunderbolt, so there
//! external always means tunnelled.
//!
//! Keeping one discovery surface across both hosts is what lets the tunnel
//! logic be exercised on a Linux box with a real GPU on the bus, instead of only
//! on the machine that needs the workaround.
//!
//! Discovery reads identity attributes only, so it needs no entitlement, no
//! driver, and no device I/O. Reaching the device is a separate stage
//! (`crate::dext`).

/// One PCI function as the host reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PciDevice {
    /// PCI vendor ID.
    pub vendor_id: u16,
    /// PCI device ID.
    pub device_id: u16,
    /// PCI base class — the high byte of the class code.
    pub base_class: u32,
    /// `true` when the device sits behind a Thunderbolt/USB4 PCIe tunnel, i.e.
    /// it is external rather than on the host's own root complex.
    /// (`IOPCITunnelled` on macOS; `untrusted` or a hotplug ancestor on Linux.)
    pub tunnelled: bool,
    /// Kernel driver bound to the device, on hosts that report one. Always
    /// `None` on macOS, where nothing in-tree binds class 0x03 on arm64 — the
    /// dext claim is a registry-wide lookup instead ([`dext_service_present`]).
    pub driver: Option<String>,
}

impl PciDevice {
    /// Identity label for logs: `1002:7550`.
    pub fn id_string(&self) -> String {
        format!("{:04x}:{:04x}", self.vendor_id, self.device_id)
    }

    /// `true` when a userspace driver could take this device: nothing in the
    /// kernel owns it, or it has been handed to `vfio-pci` on purpose. On macOS
    /// a display-class device is unclaimed by construction, so this reports the
    /// registry-wide claim by the extension named in [`service_name`].
    pub fn reachable_from_userspace(&self) -> bool {
        self.reachable_via(&service_name())
    }

    /// [`Self::reachable_from_userspace`] against an explicitly named driver
    /// extension — pass `dext::DextConfig::service()` when one is configured.
    pub fn reachable_via(&self, service: &str) -> bool {
        match self.driver.as_deref() {
            None => {
                if cfg!(target_os = "macos") {
                    service_present(service)
                } else {
                    true
                }
            }
            Some("vfio-pci") => true,
            Some(_) => false,
        }
    }
}

/// Every PCI function the host reports, ordered deterministically.
pub fn enumerate() -> Vec<PciDevice> {
    #[allow(unused_mut)]
    let mut found: Vec<PciDevice> = {
        #[cfg(target_os = "macos")]
        {
            darwin::enumerate()
        }
        #[cfg(target_os = "linux")]
        {
            linux::enumerate()
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Vec::new()
        }
    };
    // Registry / directory iteration order is not contractual; sort so callers
    // that take the first match pick the same device every run.
    found.sort_by(|a, b| {
        (a.vendor_id, a.device_id, a.base_class).cmp(&(b.vendor_id, b.device_id, b.base_class))
    });
    found
}

/// Display-class functions (base class 0x03) reachable over a PCIe tunnel.
///
/// The tunnel filter is what separates an external GPU from a GPU on the host's
/// own root complex — an iGPU, or a discrete card in a slot, both of which are
/// reached through their vendor stack (`rlx-cuda`, `rlx-rocm`), not this crate.
pub fn external_gpus() -> Vec<PciDevice> {
    enumerate()
        .into_iter()
        .filter(|d| d.base_class == crate::ids::CLASS_DISPLAY && d.tunnelled)
        .collect()
}

/// Every display-class function, tunnelled or not. Inventory and diagnostics —
/// [`external_gpus`] is the set this crate can act on.
pub fn display_devices() -> Vec<PciDevice> {
    enumerate()
        .into_iter()
        .filter(|d| d.base_class == crate::ids::CLASS_DISPLAY)
        .collect()
}

/// IOService name published by the extension rlx is known to interoperate with.
/// An external component's own identifier, not one rlx chose — configure it
/// through `dext::DextConfig` or `RLX_EGPU_SERVICE` rather than assuming it.
pub(crate) const DEFAULT_SERVICE: &str = "tinygpu";

/// IOService name to look for, with `RLX_EGPU_SERVICE` layered over the default.
///
/// Callers holding a `dext::DextConfig` should pass its `service()` to
/// [`service_present`] instead; this is the fallback for the discovery path,
/// which has no configuration of its own.
pub fn service_name() -> String {
    rlx_ir::env::var("RLX_EGPU_SERVICE").unwrap_or_else(|| DEFAULT_SERVICE.to_string())
}

/// `true` when a driver extension publishing `name` has claimed a GPU — i.e.
/// the userspace transport in `crate::dext` has something to open. Looking up
/// a service name needs no entitlement. Always `false` off macOS, where there is
/// no driver-extension concept.
pub fn service_present(name: &str) -> bool {
    #[cfg(target_os = "macos")]
    {
        match std::ffi::CString::new(name) {
            Ok(name) => darwin::service_present(&name),
            Err(_) => false,
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = name;
        false
    }
}

/// [`service_present`] against [`service_name`].
pub fn dext_service_present() -> bool {
    service_present(&service_name())
}

#[cfg(target_os = "linux")]
mod linux {
    use super::PciDevice;
    use std::fs;
    use std::path::Path;

    /// Read a `0x`-prefixed sysfs hex attribute.
    fn read_hex(path: &Path) -> Option<u32> {
        let text = fs::read_to_string(path).ok()?;
        u32::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok()
    }

    /// `domain:bus:device` addresses of every PCI hotplug slot.
    ///
    /// The second of the two tunnel signals, and in practice the load-bearing
    /// one. `untrusted` needs the firmware to mark a port external-facing
    /// (ACPI `ExternalFacingPort`), which desktop boards frequently do not —
    /// measured absent entirely on two different Linux hosts — so relying on it
    /// alone reports every device as internal and an attached eGPU as missing.
    ///
    /// A Thunderbolt/USB4 PCIe tunnel is always presented behind a
    /// hotplug-capable downstream port, so a hotplug ancestor is the signal
    /// that survives when `untrusted` is not there. It is a heuristic: a
    /// chassis with hotplug drive bays would also match, which is why it is
    /// paired with the display-class filter rather than used alone.
    fn hotplug_slots() -> Vec<String> {
        let mut slots = Vec::new();
        let Ok(entries) = fs::read_dir("/sys/bus/pci/slots") else {
            return slots;
        };
        for entry in entries.flatten() {
            if let Ok(address) = fs::read_to_string(entry.path().join("address")) {
                slots.push(address.trim().to_string());
            }
        }
        slots
    }

    /// `true` when the device sits below a hotplug slot.
    ///
    /// The sysfs symlink resolves to the full topology
    /// (`pci0000:00/0000:00:02.4/0000:05:00.0/…`), so the ancestors are the
    /// path components before the device itself.
    fn below_hotplug_slot(dir: &Path, slots: &[String]) -> bool {
        let Ok(resolved) = fs::canonicalize(dir) else {
            return false;
        };
        let components: Vec<String> = resolved
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        path_is_below_slot(&components, slots)
    }

    /// The matching rule, split out from the filesystem so it can be tested
    /// without a hotplug device attached — which is otherwise only reachable by
    /// plugging one in.
    pub(super) fn path_is_below_slot(components: &[String], slots: &[String]) -> bool {
        // Skip the device itself: a hotplug slot names the port above it, and
        // a device is not plugged into its own slot.
        components.iter().rev().skip(1).any(|ancestor| {
            // Slot addresses are `domain:bus:device`, with no function.
            ancestor
                .rsplit_once('.')
                .is_some_and(|(without_function, _)| slots.iter().any(|s| s == without_function))
        })
    }

    pub(super) fn enumerate() -> Vec<PciDevice> {
        let mut found = Vec::new();
        let Ok(entries) = fs::read_dir("/sys/bus/pci/devices") else {
            return found;
        };
        let slots = hotplug_slots();
        for entry in entries.flatten() {
            let dir = entry.path();
            let (Some(vendor), Some(device), Some(class)) = (
                read_hex(&dir.join("vendor")),
                read_hex(&dir.join("device")),
                read_hex(&dir.join("class")),
            ) else {
                continue;
            };
            // Two signals, either of which means "arrived from outside the
            // box". `untrusted` is the precise one and is often absent; the
            // hotplug ancestor is the one that actually fires on hardware.
            let untrusted = fs::read_to_string(dir.join("untrusted"))
                .map(|v| v.trim() == "1")
                .unwrap_or(false);
            let tunnelled = untrusted || below_hotplug_slot(&dir, &slots);
            let driver = fs::read_link(dir.join("driver"))
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));
            found.push(PciDevice {
                vendor_id: vendor as u16,
                device_id: device as u16,
                base_class: class >> 16,
                tunnelled,
                driver,
            });
        }
        found
    }
}

#[cfg(target_os = "macos")]
mod darwin {
    use super::PciDevice;
    use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};

    type IoObject = c_uint;
    type CFTypeRef = *const c_void;
    type CFStringRef = *const c_void;
    type CFTypeId = usize;

    #[repr(C)]
    struct CFRange {
        location: isize,
        length: isize,
    }

    const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOServiceMatching(name: *const c_char) -> *mut c_void;
        fn IOServiceNameMatching(name: *const c_char) -> *mut c_void;
        fn IOServiceGetMatchingServices(
            main_port: c_uint,
            matching: *mut c_void,
            iterator: *mut IoObject,
        ) -> c_int;
        fn IOServiceGetMatchingService(main_port: c_uint, matching: *mut c_void) -> IoObject;
        fn IOIteratorNext(iterator: IoObject) -> IoObject;
        fn IOObjectRelease(object: IoObject) -> c_int;
        fn IORegistryEntryCreateCFProperty(
            entry: IoObject,
            key: CFStringRef,
            allocator: *const c_void,
            options: u32,
        ) -> CFTypeRef;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringCreateWithCString(
            allocator: *const c_void,
            cstr: *const c_char,
            encoding: u32,
        ) -> CFStringRef;
        fn CFRelease(cf: CFTypeRef);
        fn CFGetTypeID(cf: CFTypeRef) -> CFTypeId;
        fn CFDataGetTypeID() -> CFTypeId;
        fn CFDataGetLength(data: CFTypeRef) -> isize;
        fn CFDataGetBytes(data: CFTypeRef, range: CFRange, buffer: *mut u8);
        fn CFBooleanGetTypeID() -> CFTypeId;
        fn CFBooleanGetValue(boolean: CFTypeRef) -> u8;
    }

    /// Read a registry property that holds little-endian bytes (`vendor-id`,
    /// `class-code`, …) as a `u32`. `None` if absent or a different CF type.
    unsafe fn read_u32(service: IoObject, key: &str) -> Option<u32> {
        let key = CString::new(key).ok()?;
        unsafe {
            let cf_key = CFStringCreateWithCString(
                std::ptr::null(),
                key.as_ptr(),
                K_CF_STRING_ENCODING_UTF8,
            );
            if cf_key.is_null() {
                return None;
            }
            let raw = IORegistryEntryCreateCFProperty(service, cf_key, std::ptr::null(), 0);
            CFRelease(cf_key);
            if raw.is_null() {
                return None;
            }
            let value = (CFGetTypeID(raw) == CFDataGetTypeID()).then(|| {
                let len = CFDataGetLength(raw).clamp(0, 4);
                let mut bytes = [0u8; 4];
                CFDataGetBytes(
                    raw,
                    CFRange {
                        location: 0,
                        length: len,
                    },
                    bytes.as_mut_ptr(),
                );
                u32::from_le_bytes(bytes)
            });
            CFRelease(raw);
            value
        }
    }

    /// Read a `CFBoolean` registry property (`IOPCITunnelled`).
    unsafe fn read_bool(service: IoObject, key: &str) -> Option<bool> {
        let key = CString::new(key).ok()?;
        unsafe {
            let cf_key = CFStringCreateWithCString(
                std::ptr::null(),
                key.as_ptr(),
                K_CF_STRING_ENCODING_UTF8,
            );
            if cf_key.is_null() {
                return None;
            }
            let raw = IORegistryEntryCreateCFProperty(service, cf_key, std::ptr::null(), 0);
            CFRelease(cf_key);
            if raw.is_null() {
                return None;
            }
            let value =
                (CFGetTypeID(raw) == CFBooleanGetTypeID()).then(|| CFBooleanGetValue(raw) != 0);
            CFRelease(raw);
            value
        }
    }

    pub(super) fn enumerate() -> Vec<PciDevice> {
        let mut found = Vec::new();
        unsafe {
            let matching = IOServiceMatching(c"IOPCIDevice".as_ptr());
            if matching.is_null() {
                return found;
            }
            let mut iterator: IoObject = 0;
            // Takes ownership of `matching` — do not release it here.
            if IOServiceGetMatchingServices(0, matching, &mut iterator) != 0 {
                return found;
            }
            loop {
                let service = IOIteratorNext(iterator);
                if service == 0 {
                    break;
                }
                if let (Some(vendor), Some(device), Some(class)) = (
                    read_u32(service, "vendor-id"),
                    read_u32(service, "device-id"),
                    read_u32(service, "class-code"),
                ) {
                    found.push(PciDevice {
                        vendor_id: vendor as u16,
                        device_id: device as u16,
                        base_class: class >> 16,
                        tunnelled: read_bool(service, "IOPCITunnelled").unwrap_or(false),
                        // Nothing in-tree binds class 0x03 on arm64; the dext
                        // claim is a registry-wide lookup, not a per-device
                        // attribute. See `dext_service_present`.
                        driver: None,
                    });
                }
                IOObjectRelease(service);
            }
            IOObjectRelease(iterator);
        }
        found
    }

    pub(super) fn service_present(name: &CStr) -> bool {
        unsafe {
            let matching = IOServiceNameMatching(name.as_ptr());
            if matching.is_null() {
                return false;
            }
            // Takes ownership of `matching`.
            let service = IOServiceGetMatchingService(0, matching);
            if service == 0 {
                return false;
            }
            IOObjectRelease(service);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumeration_reports_a_consistent_bus() {
        let devices = enumerate();
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            // Both hosts expose at least their own root bridges.
            assert!(!devices.is_empty(), "PCI enumeration returned nothing");
        }
        // Discovery is read-only, so repeating it is stable — and the sort
        // makes the order stable too, which matters because `detect` acts on
        // the first match.
        assert_eq!(devices, enumerate());
    }

    #[test]
    fn the_tunnel_filter_is_what_selects_an_external_gpu() {
        // Every match must carry both properties the filter tests for; a host
        // GPU (iGPU, or a card in a slot) must never appear, because it belongs
        // to its vendor stack rather than to this crate.
        for device in &external_gpus() {
            assert_eq!(device.base_class, crate::ids::CLASS_DISPLAY);
            assert!(device.tunnelled);
        }
        assert!(external_gpus().len() <= display_devices().len());
        for device in &display_devices() {
            assert_eq!(device.base_class, crate::ids::CLASS_DISPLAY);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_device_below_a_hotplug_slot_reads_as_external() {
        // `untrusted` was measured absent on every Linux host available, so the
        // hotplug ancestor is the signal that actually fires. No machine here
        // has anything in a hotplug slot, so the rule is tested directly.
        let path: Vec<String> = [
            "/",
            "sys",
            "devices",
            "pci0000:00",
            "0000:00:02.4",
            "0000:05:00.0",
            "0000:07:00.0",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        // A slot naming an ancestor: the device arrived from outside.
        assert!(super::linux::path_is_below_slot(
            &path,
            &["0000:05:00".to_string()]
        ));
        assert!(super::linux::path_is_below_slot(
            &path,
            &["0000:00:02".to_string()]
        ));
        // A slot naming an unrelated port, or none at all.
        assert!(!super::linux::path_is_below_slot(
            &path,
            &["0000:08:00".to_string()]
        ));
        assert!(!super::linux::path_is_below_slot(&path, &[]));
        // The device's own address is not a slot it is plugged into — matching
        // it would mark every device in a hotplug-capable system as external.
        assert!(!super::linux::path_is_below_slot(
            &path,
            &["0000:07:00".to_string()]
        ));
    }

    #[test]
    fn a_kernel_bound_device_is_not_reachable_from_userspace() {
        // On Linux this is the real discriminator: `nvidia` or `amdgpu` holding
        // the device means no userspace driver can take it. Only an unbound
        // device — or one deliberately handed to vfio-pci — is reachable.
        for device in enumerate() {
            match device.driver.as_deref() {
                Some("vfio-pci") | None => {}
                Some(_) => assert!(
                    !device.reachable_from_userspace(),
                    "{} is bound to {:?} yet reported reachable",
                    device.id_string(),
                    device.driver
                ),
            }
        }
    }
}
