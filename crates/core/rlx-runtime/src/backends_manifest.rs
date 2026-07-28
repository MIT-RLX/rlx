// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compile-time backend manifest — which Cargo features were enabled.

use serde::Deserialize;

/// Parsed `backends_manifest.json` emitted by `build.rs`.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct BackendsManifest {
    #[serde(rename = "crate")]
    pub crate_name: String,
    pub version: String,
    pub backends: Vec<String>,
}

impl BackendsManifest {
    /// Manifest for this build (from `build.rs` via `RLX_BACKENDS_MANIFEST_PATH`).
    pub fn current() -> Self {
        let raw = include_str!(env!("RLX_BACKENDS_MANIFEST_PATH"));
        serde_json::from_str(raw).expect("parse backends_manifest.json")
    }

    /// Raw JSON string for deploy scripts / telemetry.
    pub fn json() -> &'static str {
        include_str!(env!("RLX_BACKENDS_MANIFEST_PATH"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_parses_and_includes_cpu() {
        let m = BackendsManifest::current();
        assert_eq!(m.crate_name, "rlx-runtime");
        assert!(m.backends.iter().any(|b| b == "cpu"));
    }
}
