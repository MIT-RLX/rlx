// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Audit the local cache of signed vendor firmware against the pinned manifest.
//!
//! ```sh
//! cargo run -p rlx-egpu --features firmware --example egpu_firmware
//! cargo run -p rlx-egpu --features firmware --example egpu_firmware -- psp_14_0 gc_12_0
//! ```
//!
//! Fetching is `scripts/pull_gpu_firmware.sh`; this only reports.

use rlx_egpu::firmware::{self, FirmwareState};

fn main() {
    let filters: Vec<String> = std::env::args().skip(1).collect();
    let entries = firmware::audit(&filters);
    if entries.is_empty() {
        eprintln!("no manifest entries matched {filters:?}");
        std::process::exit(2);
    }

    println!("linux-firmware pin: {}", firmware::pinned_commit());
    println!("cache: {}", firmware::cache_dir().display());
    println!();

    let (mut verified, mut missing, mut bad) = (0, 0, 0);
    for (entry, state) in &entries {
        match state {
            FirmwareState::Verified => verified += 1,
            FirmwareState::Missing => missing += 1,
            FirmwareState::Corrupt { found } => {
                bad += 1;
                println!("CORRUPT  {}/{}", entry.path, entry.name);
                println!("         pinned {}", entry.sha256);
                println!("         found  {found}");
            }
            FirmwareState::Unreadable(e) => {
                bad += 1;
                println!("UNREADABLE {}/{}: {e}", entry.path, entry.name);
            }
        }
    }

    println!(
        "{} of {} verified, {missing} missing, {bad} unusable",
        verified,
        entries.len()
    );
    if missing > 0 {
        let families: std::collections::BTreeSet<&str> = entries
            .iter()
            .filter(|(_, s)| *s == FirmwareState::Missing)
            .map(|(e, _)| e.family.as_str())
            .collect();
        println!(
            "fetch with: scripts/pull_gpu_firmware.sh {}",
            families.into_iter().collect::<Vec<_>>().join(" ")
        );
    }
    if bad > 0 {
        std::process::exit(1);
    }
}
