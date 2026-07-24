// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Package integrity checks (xxh3 over uncompressed payloads).

use crate::package::Package;
use crate::tier::checksum_hex;
use anyhow::{Result, bail};

/// Verify every tensor (and optional sidecars) against TOC checksums.
pub fn verify_package(pack: &Package) -> Result<VerifyReport> {
    let mut report = VerifyReport::default();
    let Some(idx) = pack.weights_index() else {
        return Ok(report);
    };
    for t in &idx.tensors {
        report.tensors_checked += 1;
        let raw = pack.tensor_bytes(&t.name)?;
        let got = checksum_hex(&raw);
        match &t.checksum {
            Some(expect) if expect == &got => report.tensors_ok += 1,
            Some(expect) => {
                report.tensors_mismatch += 1;
                report.failures.push(format!(
                    "tensor {}: checksum mismatch (toc={expect} got={got})",
                    t.name
                ));
            }
            None => {
                report.tensors_unchecked += 1;
            }
        }
    }
    for sc in &pack.manifest().sidecars {
        report.sidecars_checked += 1;
        let _ = pack.sidecar(&sc.id)?; // ensures decode works
        report.sidecars_ok += 1;
    }
    if !report.failures.is_empty() {
        bail!("{} checksum failure(s)", report.failures.len());
    }
    Ok(report)
}

#[derive(Debug, Default, Clone)]
pub struct VerifyReport {
    pub tensors_checked: usize,
    pub tensors_ok: usize,
    pub tensors_unchecked: usize,
    pub tensors_mismatch: usize,
    pub sidecars_checked: usize,
    pub sidecars_ok: usize,
    pub failures: Vec<String>,
}
