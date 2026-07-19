// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! JSON serialization for HIR / LIR modules (feature `serialize`).

#![cfg(feature = "serialize")]

use crate::hir::HirModule;
use crate::lir::LirModule;

/// Serialize a [`HirModule`] to a JSON string.
pub fn hir_to_json(hir: &HirModule) -> Result<String, serde_json::Error> {
    serde_json::to_string(hir)
}

/// Deserialize a [`HirModule`] from JSON.
pub fn hir_from_json(s: &str) -> Result<HirModule, serde_json::Error> {
    serde_json::from_str(s)
}

/// Serialize a [`LirModule`] to a JSON string (AOT cache format).
pub fn lir_to_json(lir: &LirModule) -> Result<String, serde_json::Error> {
    serde_json::to_string(lir)
}

/// Deserialize a [`LirModule`] from JSON.
pub fn lir_from_json(s: &str) -> Result<LirModule, serde_json::Error> {
    serde_json::from_str(s)
}

/// Magic + schema version for [`lir_to_bytes`] / [`lir_from_bytes`].
/// Bump [`LIR_BIN_VERSION`] whenever a serialized `Op` / LIR field layout changes
/// so stale AOT caches fail with a clear error instead of a serde variant panic.
const LIR_BIN_MAGIC: &[u8; 8] = b"RLXLIR01";
/// Monotonic schema id for the bincode body (not the magic string).
pub const LIR_BIN_VERSION: u32 = 2;

/// Serialize a [`LirModule`] to compact binary (the AOT-cache format). Far
/// smaller and much faster to load than JSON — the big baked constants (STFT
/// DFT matrices, mel filterbanks, resampler kernels) become raw f32 bytes
/// instead of ASCII numbers-as-text, so a 74 MB JSON LIR shrinks to ~30 MB and
/// deserialization drops from ~10 s to well under 1 s.
pub fn lir_to_bytes(lir: &LirModule) -> Result<Vec<u8>, bincode::Error> {
    let body = bincode::serialize(lir)?;
    let mut out = Vec::with_capacity(12 + body.len());
    out.extend_from_slice(LIR_BIN_MAGIC);
    out.extend_from_slice(&LIR_BIN_VERSION.to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Deserialize a [`LirModule`] from the compact binary form.
pub fn lir_from_bytes(b: &[u8]) -> Result<LirModule, bincode::Error> {
    if b.len() >= 12 && &b[..8] == LIR_BIN_MAGIC {
        let ver = u32::from_le_bytes(b[8..12].try_into().unwrap());
        if ver != LIR_BIN_VERSION {
            return Err(Box::new(bincode::ErrorKind::Custom(format!(
                "LIR binary schema version {ver} != current {LIR_BIN_VERSION} (stale AOT cache)"
            ))));
        }
        return bincode::deserialize(&b[12..]);
    }
    // Legacy unversioned blobs (pre-header): attempt decode, then callers treat
    // failure as a cache miss. Prefer failing closed on schema drift.
    bincode::deserialize(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use crate::Shape;
    use crate::hir::HirModule;

    #[test]
    fn hir_module_json_roundtrip() {
        let mut hir = HirModule::new("serde_test");
        let x = hir.input("x", Shape::new(&[2, 4], DType::F32));
        let w = hir.param("w", Shape::new(&[4, 3], DType::F32));
        let y = hir.linear(x, w, None, None, Shape::new(&[2, 3], DType::F32));
        hir.set_outputs(vec![y]);
        let json = hir_to_json(&hir).expect("serialize HIR");
        let back = hir_from_json(&json).expect("deserialize HIR");
        assert_eq!(back.name, hir.name);
        assert_eq!(back.len(), hir.len());
        assert_eq!(back.outputs, hir.outputs);
    }
}
