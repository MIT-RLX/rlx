// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PCI vendor / device identity for the eGPUs a userspace ring driver can drive.
//!
//! The membership tests below mirror the device sets a working userspace
//! bring-up covers today (tinygrad's `AM` / `NV` drivers). Being on this list
//! means the register interface and firmware sequence are known — not that rlx
//! can execute a graph on it (see [`crate::is_available`]).

/// PCI vendor ID — AMD/ATI.
pub const VENDOR_AMD: u16 = 0x1002;
/// PCI vendor ID — NVIDIA.
pub const VENDOR_NV: u16 = 0x10de;
/// PCI vendor ID — Intel. No bring-up path; named so diagnostics can say which
/// GPU they found rather than reporting an unknown vendor.
pub const VENDOR_INTEL: u16 = 0x8086;

/// PCI base class 0x03 — display controller. The class the host has no arm64
/// driver for, and therefore the class that arrives unclaimed on the bus.
pub const CLASS_DISPLAY: u32 = 0x03;

/// AMD parts with a known bring-up sequence: Navi 31 / Navi 33 (RDNA3),
/// Navi 44 / Navi 48 (RDNA4), and MI300X. Matched exactly.
pub const AMD_DEVICES: &[u16] = &[
    0x744c, // Navi 31 — RX 7900 XT / XTX
    0x7480, // Navi 33 — RX 7600
    0x74a1, // Aqua Vanjaram — MI300X
    0x7550, // Navi 48 — RX 9070 / 9070 XT
    0x7551, // Navi 48
    0x7590, // Navi 44 — RX 9060 XT
    0x75a0, // Navi 44
];

/// NVIDIA parts with a known bring-up sequence, matched on the high byte of the
/// device ID: GA10x (Ampere) through GB20x (Blackwell).
pub const NV_DEVICE_FAMILIES: &[u16] = &[
    0x2200, // GA102 — RTX 3080 / 3090
    0x2400, // GA103 / GA104
    0x2500, // GA106 / GA107
    0x2600, // AD102
    0x2700, // AD103 / AD104
    0x2800, // AD106 / AD107
    0x2b00, // GB202
    0x2c00, // GB203
    0x2d00, // GB205
    0x2f00, // GB20x
];

/// Vendor label for a PCI vendor ID.
pub fn vendor_name(vendor_id: u16) -> &'static str {
    match vendor_id {
        VENDOR_AMD => "AMD",
        VENDOR_NV => "NVIDIA",
        VENDOR_INTEL => "Intel",
        _ => "unknown vendor",
    }
}

/// `true` when a `(vendor, device)` pair has a known userspace bring-up path.
pub fn is_supported(vendor_id: u16, device_id: u16) -> bool {
    match vendor_id {
        VENDOR_AMD => AMD_DEVICES.contains(&device_id),
        VENDOR_NV => NV_DEVICE_FAMILIES.contains(&(device_id & 0xff00)),
        _ => false,
    }
}

/// Firmware families a card needs before it can be brought up.
///
/// The exact filenames come from the IP discovery table the card itself
/// reports, which cannot be read before the transport is open — but the
/// *families* follow from the part, so a card can be identified and its
/// firmware fetched ahead of ever claiming it.
///
/// Knowing what firmware a part needs is independent of having a bring-up
/// sequence for it, so this covers parts [`is_supported`] rejects — notably the
/// MI100, which is the development target for `am` on a Linux host. Empty for a
/// part whose firmware set is not recorded.
///
/// Names match the `family` column of `firmware/manifest.tsv` and the arguments
/// `scripts/pull_gpu_firmware.sh` accepts.
pub fn firmware_families(vendor_id: u16, device_id: u16) -> &'static [&'static str] {
    match (vendor_id, device_id) {
        // RDNA3 — Navi 31 / Navi 33.
        (VENDOR_AMD, 0x744c | 0x7480) => &["gc_11_0", "psp_13_0", "smu_13_0", "sdma_6_0"],
        // RDNA4 — Navi 48 / Navi 44.
        (VENDOR_AMD, 0x7550 | 0x7551 | 0x7590 | 0x75a0) => {
            &["gc_12_0", "psp_14_0", "smu_14_0", "sdma_7_0"]
        }
        // CDNA3 — MI300X.
        (VENDOR_AMD, 0x74a1) => &["gc_9_4", "psp_13_0", "smu_13_0", "sdma_4_4"],
        // CDNA1 — MI100. Firmware is ASIC-named rather than IP-versioned, so
        // one family carries the whole set. No bring-up covers this part; the
        // blobs exist so `am` can be developed against a real card.
        (VENDOR_AMD, 0x738c) => &["arcturus"],
        // NVIDIA boots through GSP. The per-chip families carry only the
        // booter / bootloader / FMC; the GSP image itself exists once, under
        // ga102, and covers every family (it bundles per-family microkernels
        // and seven signature sections). So every part needs `ga102` too —
        // verified by the ad102 and gb202 gsp paths 404ing upstream.
        (VENDOR_NV, d) => match d & 0xff00 {
            0x2200 | 0x2400 | 0x2500 => &["ga102"],
            0x2600 | 0x2700 | 0x2800 => &["ad102", "ga102"],
            0x2b00 | 0x2c00 | 0x2d00 | 0x2f00 => &["gb202", "ga102"],
            _ => &[],
        },
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amd_matches_exactly_and_nv_matches_by_family() {
        // RX 9070 (Navi 48) — exact match.
        assert!(is_supported(VENDOR_AMD, 0x7550));
        // Polaris (RX 580) — no RDNA bring-up, correctly rejected.
        assert!(!is_supported(VENDOR_AMD, 0x67df));
        // RTX 5090 (GB202, 0x2b85) — matched on the 0x2b00 family.
        assert!(is_supported(VENDOR_NV, 0x2b85));
        // Turing (RTX 2080, 0x1e87) — pre-Ampere, correctly rejected.
        assert!(!is_supported(VENDOR_NV, 0x1e87));
        // Intel Arc — named, but no bring-up path at all.
        assert!(!is_supported(VENDOR_INTEL, 0x56a0));
        assert_eq!(vendor_name(VENDOR_INTEL), "Intel");
    }

    #[test]
    fn every_supported_card_names_the_firmware_it_needs() {
        // A supported card with no firmware families would fetch nothing and
        // then stall partway through bring-up on a missing file.
        for &device in AMD_DEVICES {
            let families = firmware_families(VENDOR_AMD, device);
            assert!(!families.is_empty(), "no firmware for AMD {device:04x}");
            // Every AMD part needs all four IP blocks loaded.
            assert_eq!(families.len(), 4, "AMD {device:04x}: {families:?}");
        }
        for &family in NV_DEVICE_FAMILIES {
            let families = firmware_families(VENDOR_NV, family | 0x85);
            assert!(!families.is_empty(), "no firmware for NVIDIA {family:04x}");
        }
        // Unsupported parts name nothing rather than guessing.
        assert!(firmware_families(VENDOR_AMD, 0x67df).is_empty());
        assert!(firmware_families(VENDOR_INTEL, 0x56a0).is_empty());
    }

    #[test]
    fn every_nvidia_part_fetches_the_shared_gsp_image() {
        // There is one GSP image and it lives under ga102 — the ad102 and gb202
        // gsp paths 404 upstream. A part whose families omit `ga102` would pull
        // its booter and bootloader, then have nothing to boot: the 60 MB image
        // itself would be missing. Ada hit exactly that before this test.
        for &family in NV_DEVICE_FAMILIES {
            let families = firmware_families(VENDOR_NV, family | 0x85);
            assert!(
                families.contains(&"ga102"),
                "NVIDIA {family:04x} does not fetch the shared GSP image: {families:?}"
            );
        }
    }

    #[test]
    fn firmware_is_known_for_parts_no_bring_up_covers() {
        // The MI100 has no bring-up sequence, but its firmware set is recorded
        // so it can be fetched and verified — the two facts are independent,
        // and conflating them would make the development target unfetchable.
        const MI100: u16 = 0x738c;
        assert!(!is_supported(VENDOR_AMD, MI100));
        assert_eq!(firmware_families(VENDOR_AMD, MI100), &["arcturus"]);
    }
}
