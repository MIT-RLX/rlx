// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! rlx-egpu — external GPU over USB4/Thunderbolt on Apple Silicon.
//!
//! ## What the host actually withholds
//!
//! PCIe tunnelling over Thunderbolt/USB4 works on Apple Silicon. A device behind
//! the tunnel enumerates as a normal `IOPCIDevice` and carries
//! `IOPCITunnelled = true`; an NVMe enclosure on a Mac mini's USB4 port is a
//! bridge plus a storage function in the IOKit registry, driven at full speed.
//! What macOS does not ship on arm64 is a driver for PCI base class 0x03, so a
//! discrete GPU enumerates and is then left unclaimed. The gap is a driver, not
//! a bus.
//!
//! That shapes this crate. [`pci`] reads the registry to find an unclaimed
//! display-class device on the tunnel — no entitlement, no driver, no device
//! I/O. `dext` claims one through a driver extension that holds the
//! restricted PCI entitlement, giving config space, BAR mappings, and DMA
//! buffers with physical addresses. Bringing that device up so it executes
//! kernels is a third stage, and it is not written.
//!
//! ## Stages
//!
//! | Stage | Feature | State |
//! |-------|---------|-------|
//! | PCI discovery over the tunnel | *(default)* | complete — [`detect`] |
//! | Config / BAR / DMA transport | `dext` | complete — `dext::PciTransport` |
//! | AMD RDNA3/4 device bring-up | `am` | not implemented |
//! | Ring submission + kernels | — | not started |
//!
//! [`is_available`] is gated on the bring-up stage, so it reports `false` today
//! even with a supported card attached and the transport working. A backend that
//! cannot execute a graph does not claim it can, and device selection will not
//! dispatch here.
//!
//! ## The two external dependencies, stated plainly
//!
//! - **The entitlement.** `com.apple.developer.driverkit.transport.pci` is
//!   granted by Apple per development team. rlx does not have it, so the
//!   transport talks to whichever signed extension is installed. Owning that
//!   end to end means applying for the entitlement and shipping a dext of our own.
//! - **The bring-up.** Loading signed firmware through the PSP, starting the
//!   memory controller, SMU, and GFX/SDMA rings, and building GPUVM page tables
//!   is several thousand lines of register sequencing per architecture. The
//!   reference implementation (tinygrad's `AM` driver) is the scale to plan for.
//!
//! ## Cost model, before anyone plans around it
//!
//! The link is a PCIe x4-class tunnel, and BAR access is a socket round-trip per
//! MMIO operation into the process that holds the driver connection. DMA memory
//! is shared by descriptor and mapped directly, so host-side access is free.
//! Workloads that keep weights resident and submit few large kernels survive
//! this; anything with per-kernel host traffic will be bound by the transport,
//! not the GPU.

#[cfg(feature = "am")]
pub mod am;
pub mod aot;
pub mod codeobj;
pub mod firmware;
pub mod ids;
pub mod pci;

#[cfg(feature = "dext")]
pub mod dext;

/// Ops this backend can lower. Empty: there is no execution path yet, and the
/// legalizer must not be told otherwise.
pub const SUPPORTED_OPS: &[rlx_ir::OpKind] = &[];

/// Why a stage could not proceed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EgpuError {
    /// The operation has no meaning on this host (not macOS, no tunnel).
    Unsupported(String),
    /// The helper app that owns the driver-extension connection is not present.
    AppMissing(String),
    /// No driver extension has claimed a GPU.
    DextAbsent(String),
    /// Socket, process, or mapping failure.
    Io(String),
    /// The helper reported a failure, or sent something unexpected.
    Protocol(String),
}

impl std::fmt::Display for EgpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EgpuError::Unsupported(m) => write!(f, "eGPU unsupported here: {m}"),
            EgpuError::AppMissing(m) => write!(f, "eGPU helper missing: {m}"),
            EgpuError::DextAbsent(m) => write!(f, "eGPU driver extension absent: {m}"),
            EgpuError::Io(m) => write!(f, "eGPU I/O: {m}"),
            EgpuError::Protocol(m) => write!(f, "eGPU transport: {m}"),
        }
    }
}

impl std::error::Error for EgpuError {}

/// What the PCIe tunnel is carrying right now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EgpuStatus {
    /// No display-class device on a PCIe tunnel (nothing plugged in, or the
    /// enclosure is not powered / not linked).
    Absent,
    /// A GPU is on the tunnel, but no userspace bring-up sequence covers it —
    /// pre-RDNA3 AMD, pre-Ampere NVIDIA, or another vendor entirely.
    Unsupported { vendor_id: u16, device_id: u16 },
    /// A supported GPU is on the tunnel and unclaimed: no driver extension has
    /// taken it, so nothing can reach its registers.
    Detected { vendor_id: u16, device_id: u16 },
    /// A supported GPU is on the tunnel and a driver extension has claimed it.
    /// The transport in `dext` can open it; execution still needs the
    /// bring-up stage.
    Claimed { vendor_id: u16, device_id: u16 },
}

impl EgpuStatus {
    /// A GPU is physically on the bus, whatever its state.
    pub fn hardware_present(&self) -> bool {
        !matches!(self, EgpuStatus::Absent)
    }

    /// The `vendor:device` pair, when there is one.
    pub fn identity(&self) -> Option<(u16, u16)> {
        match *self {
            EgpuStatus::Absent => None,
            EgpuStatus::Unsupported {
                vendor_id,
                device_id,
            }
            | EgpuStatus::Detected {
                vendor_id,
                device_id,
            }
            | EgpuStatus::Claimed {
                vendor_id,
                device_id,
            } => Some((vendor_id, device_id)),
        }
    }
}

/// Probe the PCIe tunnel for an external GPU.
///
/// Reads host identity attributes only — the IOKit registry on macOS,
/// `/sys/bus/pci` on Linux. A GPU on the host's own root complex is never
/// reported: it belongs to its vendor stack (`rlx-cuda`, `rlx-rocm`). Hosts
/// with neither discovery surface report [`EgpuStatus::Absent`].
pub fn detect() -> EgpuStatus {
    detect_via(&pci::service_name())
}

/// [`detect`] against an explicitly named driver extension — pass
/// `dext::DextConfig::service()` when one is configured, so discovery and the
/// transport agree on which extension they are talking about.
pub fn detect_via(service: &str) -> EgpuStatus {
    let Some(gpu) = pci::external_gpus().into_iter().next() else {
        return EgpuStatus::Absent;
    };
    let (vendor_id, device_id) = (gpu.vendor_id, gpu.device_id);
    if !ids::is_supported(vendor_id, device_id) {
        return EgpuStatus::Unsupported {
            vendor_id,
            device_id,
        };
    }
    if gpu.reachable_via(service) {
        EgpuStatus::Claimed {
            vendor_id,
            device_id,
        }
    } else {
        EgpuStatus::Detected {
            vendor_id,
            device_id,
        }
    }
}

/// `true` if an external GPU is on the bus at all. Use for inventory; use
/// [`is_available`] to gate execution.
pub fn hardware_present() -> bool {
    detect().hardware_present()
}

/// `true` only when rlx can execute a graph on the eGPU.
///
/// That needs a claimed device **and** a bring-up path that has been run on
/// hardware. The `am` module exists but has never been executed, so this stays
/// `false` — a written sequence is not a working one, and gating on
/// `cfg!(feature = "am")` alone would let an untested driver become a dispatch
/// target. Flip `am::VALIDATED_ON_HARDWARE` once a card has actually come up.
pub fn is_available() -> bool {
    #[cfg(feature = "am")]
    {
        am::VALIDATED_ON_HARDWARE && matches!(detect(), EgpuStatus::Claimed { .. })
    }
    #[cfg(not(feature = "am"))]
    {
        false
    }
}

/// `"1 GPU"` / `"2 GPUs"` for the diagnostic text.
fn pluralize_gpu(count: usize) -> String {
    if count == 1 {
        "1 GPU".to_string()
    } else {
        format!("{count} GPUs")
    }
}

/// One-line description of what was found and what is missing.
pub fn diagnostic() -> String {
    match detect() {
        EgpuStatus::Absent => {
            // Name what *was* on the bus. "Nothing found" is ambiguous between
            // an empty tunnel and a GPU that was found and then filtered out
            // for sitting on the root complex, and those need different fixes.
            let host_gpus = pci::display_devices();
            let elsewhere = if host_gpus.is_empty() {
                String::new()
            } else {
                let listed = host_gpus
                    .iter()
                    .map(|d| format!("{} {}", ids::vendor_name(d.vendor_id), d.id_string()))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    " {} present but not on a tunnel ({listed}) — reached through the vendor \
                     stack (rlx-cuda / rlx-rocm) instead.",
                    pluralize_gpu(host_gpus.len())
                )
            };
            let hint = if cfg!(target_os = "macos") {
                "macOS has no arm64 driver for this class, so an attached card would still \
                 enumerate unclaimed — check that the enclosure is powered and linked."
            } else {
                "this crate covers GPUs arriving over a Thunderbolt/USB4 PCIe tunnel."
            };
            format!("eGPU: no display-class device on a PCIe tunnel. {hint}{elsewhere}")
        }
        EgpuStatus::Unsupported {
            vendor_id,
            device_id,
        } => format!(
            "eGPU: {} {vendor_id:04x}:{device_id:04x} is on the PCIe tunnel, but no userspace \
             bring-up covers it (AMD needs RDNA3 or newer, NVIDIA needs Ampere or newer).",
            ids::vendor_name(vendor_id),
        ),
        EgpuStatus::Detected {
            vendor_id,
            device_id,
        } => format!(
            "eGPU: {} {vendor_id:04x}:{device_id:04x} is on the PCIe tunnel and unclaimed — no \
             driver extension has taken it, so its registers are unreachable. Install and \
             enable one under System Settings > General > Login Items & Extensions > Driver \
             Extensions.",
            ids::vendor_name(vendor_id),
        ),
        EgpuStatus::Claimed {
            vendor_id,
            device_id,
        } => format!(
            "eGPU: {} {vendor_id:04x}:{device_id:04x} is claimed and reachable over the \
             transport, but device bring-up (feature `am`) is not implemented, so no graph \
             can execute here yet.",
            ids::vendor_name(vendor_id),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_execution_path_is_claimed() {
        // The op list and `is_available` must agree: nothing runs here yet.
        assert!(SUPPORTED_OPS.is_empty());
        assert!(!is_available());
    }

    #[test]
    fn detection_and_diagnostic_agree() {
        let status = detect();
        let text = diagnostic();
        assert!(text.starts_with("eGPU:"));
        match status.identity() {
            Some((vendor_id, device_id)) => {
                assert!(text.contains(&format!("{vendor_id:04x}:{device_id:04x}")));
                assert!(status.hardware_present());
            }
            None => {
                assert_eq!(status, EgpuStatus::Absent);
                assert!(!status.hardware_present());
            }
        }
    }

    #[test]
    fn absent_names_the_gpus_it_filtered_out() {
        // "Nothing found" is ambiguous between an empty tunnel and a GPU that
        // was found and excluded for sitting on the root complex. On a host
        // with a GPU (a Linux workstation, an Intel iGPU), the diagnostic must
        // name it so the two cases are distinguishable.
        if detect() != EgpuStatus::Absent {
            return;
        }
        let text = diagnostic();
        for device in pci::display_devices() {
            assert!(
                text.contains(&device.id_string()),
                "diagnostic omitted the host GPU {}: {text}",
                device.id_string()
            );
        }
    }
}
