// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The signed firmware a bring-up hands to the card.
//!
//! Two different signatures get conflated when people ask whether a driver can
//! be avoided, so to be exact about which one this is:
//!
//! - A **driver extension** is signed by Apple. That governs whether a process
//!   may claim the PCI device, and the card never sees it.
//! - **Firmware** is signed by AMD or NVIDIA. The PSP (AMD) and the GSP
//!   bootloader (NVIDIA) verify it against keys fused into the die and refuse
//!   anything else.
//!
//! rlx cannot produce the second kind and does not try. The blobs are
//! redistributable, carried by `linux-firmware`, and bring-up consists of
//! handing the card its own firmware and letting it boot itself. What this
//! module owns is knowing *which* blobs are needed and whether the local copies
//! are the ones the manifest pins — fetching is
//! `scripts/pull_gpu_firmware.sh`, so nothing here reaches the network.
//!
//! A hash mismatch is reported, never repaired in place: an unverified blob is
//! one the PSP will reject, and turning a clear checksum failure into an opaque
//! firmware-load hang helps no one.

use crate::EgpuError;
use std::path::{Path, PathBuf};

/// The manifest, compiled in so the pinned set travels with the binary.
const MANIFEST: &str = include_str!("../firmware/manifest.tsv");

/// One blob: where it lives upstream and what it must hash to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirmwareEntry {
    /// `amd` / `nvidia`.
    pub vendor: String,
    /// IP-block family (`gc_11_0`, `psp_14_0`) or GPU family (`ga102`).
    pub family: String,
    /// Directory within linux-firmware.
    pub path: String,
    /// File name.
    pub name: String,
    /// Expected SHA-256, lowercase hex.
    pub sha256: String,
}

impl FirmwareEntry {
    /// Location under the cache root.
    pub fn cache_path(&self, root: &Path) -> PathBuf {
        root.join(&self.path).join(&self.name)
    }
}

/// What the local cache holds for one entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FirmwareState {
    /// Present and hashes to the pinned value.
    Verified,
    /// Not downloaded yet.
    Missing,
    /// Present but hashes to something else — the file must not be used.
    Corrupt { found: String },
    /// Present but unreadable.
    Unreadable(String),
}

/// linux-firmware commit the manifest pins, read from its header.
pub fn pinned_commit() -> &'static str {
    MANIFEST
        .lines()
        .find_map(|l| l.strip_prefix("# commit"))
        .map(str::trim)
        .unwrap_or("")
}

/// Every entry in the compiled-in manifest.
pub fn manifest() -> Vec<FirmwareEntry> {
    MANIFEST
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .filter_map(|line| {
            let mut f = line.split('\t').map(str::trim);
            Some(FirmwareEntry {
                vendor: f.next()?.to_string(),
                family: f.next()?.to_string(),
                path: f.next()?.to_string(),
                name: f.next()?.to_string(),
                sha256: f.next()?.to_string(),
            })
        })
        .collect()
}

/// Entries whose vendor or family matches one of `filters`; all of them when
/// `filters` is empty. Mirrors the script's selection so both agree.
pub fn select(filters: &[String]) -> Vec<FirmwareEntry> {
    manifest()
        .into_iter()
        .filter(|e| filters.is_empty() || filters.iter().any(|f| *f == e.vendor || *f == e.family))
        .collect()
}

/// Where the puller writes: `RLX_FW_DIR`, else `XDG_CACHE_HOME/rlx/firmware`,
/// else `~/.cache/rlx/firmware`. Kept identical to the script.
pub fn cache_dir() -> PathBuf {
    if let Some(dir) = rlx_ir::env::var_os("RLX_FW_DIR") {
        return PathBuf::from(dir);
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("rlx").join("firmware")
}

/// SHA-256 of a file, lowercase hex.
#[cfg(feature = "firmware")]
fn hash_file(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Check one entry against the cache.
#[cfg(feature = "firmware")]
pub fn state_of(entry: &FirmwareEntry, root: &Path) -> FirmwareState {
    let path = entry.cache_path(root);
    if !path.is_file() {
        return FirmwareState::Missing;
    }
    match hash_file(&path) {
        Ok(found) if found == entry.sha256 => FirmwareState::Verified,
        Ok(found) => FirmwareState::Corrupt { found },
        Err(e) => FirmwareState::Unreadable(e.to_string()),
    }
}

/// Check every selected entry.
#[cfg(feature = "firmware")]
pub fn audit(filters: &[String]) -> Vec<(FirmwareEntry, FirmwareState)> {
    let root = cache_dir();
    select(filters)
        .into_iter()
        .map(|e| {
            let state = state_of(&e, &root);
            (e, state)
        })
        .collect()
}

/// Load a verified blob for handing to the card.
///
/// Refuses anything whose hash does not match the pin, so a corrupt or
/// substituted file fails here with a readable error instead of inside a
/// firmware-load handshake.
#[cfg(feature = "firmware")]
pub fn load(name: &str) -> Result<Vec<u8>, EgpuError> {
    let entry = manifest()
        .into_iter()
        .find(|e| e.name == name)
        .ok_or_else(|| EgpuError::Protocol(format!("{name} is not in the firmware manifest")))?;
    let root = cache_dir();
    let path = entry.cache_path(&root);
    match state_of(&entry, &root) {
        FirmwareState::Verified => {
            std::fs::read(&path).map_err(|e| EgpuError::Io(format!("{}: {e}", path.display())))
        }
        FirmwareState::Missing => Err(EgpuError::Io(format!(
            "{name} is not in {} — run scripts/pull_gpu_firmware.sh {}",
            root.display(),
            entry.family
        ))),
        FirmwareState::Corrupt { found } => Err(EgpuError::Protocol(format!(
            "{name} hashes to {found}, manifest pins {} — refusing to load it",
            entry.sha256
        ))),
        FirmwareState::Unreadable(e) => Err(EgpuError::Io(format!("{name}: {e}"))),
    }
}

/// Stub for builds without the `firmware` feature: the manifest is still
/// readable, but nothing can be verified or loaded without a hash.
#[cfg(not(feature = "firmware"))]
pub fn load(_name: &str) -> Result<Vec<u8>, EgpuError> {
    Err(EgpuError::Unsupported(
        "firmware loading needs the `firmware` feature (SHA-256 verification)".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_parses_and_pins_a_commit() {
        let entries = manifest();
        assert!(entries.len() > 50, "got {} entries", entries.len());
        assert_eq!(
            pinned_commit().len(),
            40,
            "linux-firmware pin must be a full commit hash"
        );
    }

    #[test]
    fn every_entry_carries_a_well_formed_hash_and_unique_identity() {
        let mut seen = std::collections::HashSet::new();
        for e in manifest() {
            assert_eq!(e.sha256.len(), 64, "{}: bad hash length", e.name);
            assert!(
                e.sha256
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
                "{}: hash must be lowercase hex",
                e.name
            );
            assert!(!e.vendor.is_empty() && !e.family.is_empty() && !e.path.is_empty());
            assert!(
                seen.insert((e.path.clone(), e.name.clone())),
                "{}/{} appears twice",
                e.path,
                e.name
            );
        }
    }

    #[test]
    fn selection_matches_on_vendor_or_family() {
        let all = manifest().len();
        assert_eq!(select(&[]).len(), all);

        let amd = select(&["amd".to_string()]);
        assert!(!amd.is_empty());
        assert!(amd.iter().all(|e| e.vendor == "amd"));

        let nvidia = select(&["nvidia".to_string()]);
        assert!(!nvidia.is_empty());
        assert_eq!(amd.len() + nvidia.len(), all, "every entry has a vendor");

        let family = select(&["psp_14_0".to_string()]);
        assert!(!family.is_empty());
        assert!(family.iter().all(|e| e.family == "psp_14_0"));

        assert!(select(&["nothing-matches".to_string()]).is_empty());
    }

    #[test]
    fn the_supported_cards_all_have_firmware_families_present() {
        // Each supported architecture needs its PSP, SMU, GFX and SDMA blobs in
        // the manifest, or a bring-up would get partway and stall on a missing
        // file. Check the families rather than exact filenames, which are
        // chosen at runtime from the card's IP discovery table.
        let families: std::collections::HashSet<String> =
            manifest().into_iter().map(|e| e.family).collect();
        for needed in [
            "gc_11_0", "psp_13_0", "smu_13_0", "sdma_6_0", // RDNA3 — Navi 31/33
            "gc_12_0", "psp_14_0", "smu_14_0", "sdma_7_0", // RDNA4 — Navi 44/48
            "gc_9_4", "sdma_4_4", // CDNA3 — MI300X
            "ga102", "ad102", "gb202", // NVIDIA GSP boot chain
        ] {
            assert!(families.contains(needed), "no firmware for {needed}");
        }
    }

    #[cfg(feature = "firmware")]
    #[test]
    fn a_missing_blob_names_the_command_that_fetches_it() {
        // Point the cache at an empty directory so this holds whether or not
        // the developer has pulled firmware.
        let empty = std::env::temp_dir().join("rlx-egpu-fw-absent");
        let entry = &manifest()[0];
        assert_eq!(state_of(entry, &empty), FirmwareState::Missing);
    }

    #[cfg(feature = "firmware")]
    #[test]
    fn a_corrupt_blob_is_refused_rather_than_used() {
        let dir = std::env::temp_dir().join("rlx-egpu-fw-corrupt");
        let entry = manifest().into_iter().next().unwrap();
        let path = entry.cache_path(&dir);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not the firmware").unwrap();

        match state_of(&entry, &dir) {
            FirmwareState::Corrupt { found } => {
                assert_ne!(found, entry.sha256);
                assert_eq!(found.len(), 64);
            }
            other => panic!("expected Corrupt, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
