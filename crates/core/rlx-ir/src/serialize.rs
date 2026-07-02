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
