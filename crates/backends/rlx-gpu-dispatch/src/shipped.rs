// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Per-hardware tuning that ships with the crate.**
//!
//! [`crate::dispatch::default_choice`] is arch-independent by design (it
//! reproduces the historical routing bit-for-bit), and measured overrides
//! used to exist only in a per-machine cache that a fresh install never has.
//! So a winner measured on, say, an M4 Pro helped exactly one machine.
//!
//! `tuning/shipped.tsv` carries those measured winners in the ordinary cache
//! format, arch-tagged, so they apply on every machine of the same arch and
//! are inert everywhere else (the key includes the arch). Backends load it
//! *before* their user cache; the table is last-write-wins, so a local re-tune
//! of the same key overrides the shipped row. Each backend gates this on
//! `RLX_GPU_NO_SHIPPED_TUNING` (this crate reads no environment).
//!
//! Rows are admitted only with evidence: a measured speedup and at least one
//! confirming held-out shape, as produced by the backends' `tune_dispatch`
//! examples. The tests below enforce that, so a hand-pinned guess cannot ride
//! along looking like a measurement.

use crate::dispatch::{LoadReport, load_overrides};

/// The shipped rows, in the tuning-cache v2 format.
pub const SHIPPED_TSV: &str = include_str!("../tuning/shipped.tsv");

/// Install the shipped rows into the process table. Call before loading the
/// user cache so local measurements win.
pub fn load() -> LoadReport {
    load_overrides(SHIPPED_TSV)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> impl Iterator<Item = Vec<&'static str>> {
        SHIPPED_TSV
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.split('\t').collect())
    }

    #[test]
    fn every_shipped_row_parses() {
        crate::dispatch::clear_overrides();
        let r = load();
        assert_eq!(r.skipped, 0, "a shipped row failed to parse or was illegal");
        assert_eq!(r.applied, rows().count());
        crate::dispatch::clear_overrides();
    }

    #[test]
    fn every_shipped_row_carries_evidence() {
        for f in rows() {
            assert_eq!(f.len(), 6, "row needs speedup + holdouts: {f:?}");
            let sp: f32 = f[4].parse().expect("speedup must be measured, not '-'");
            let holdouts: u16 = f[5].parse().expect("holdout count");
            assert!(sp > 1.0, "a shipped row must be a win: {f:?}");
            assert!(
                holdouts >= 1,
                "a shipped row must be holdout-validated: {f:?}"
            );
        }
    }
}
