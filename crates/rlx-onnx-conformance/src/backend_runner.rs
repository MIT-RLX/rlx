// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Bulk ONNX backend node test runner (opset-aware registry diff).

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_onnx_import::ops::{format_registry_dashboard, op_is_registered};

/// Scan a directory of ONNX backend test cases (`test_*` folders with `model.onnx`).
pub fn collect_backend_test_models(root: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut out = Vec::new();
    if !root.is_dir() {
        return Ok(out);
    }
    for entry in std::fs::read_dir(root).context("read backend test dir")? {
        let entry = entry?;
        let path = entry.path();
        if path.join("model.onnx").is_file() {
            out.push(path.join("model.onnx"));
        }
    }
    out.sort();
    Ok(out)
}

/// Return ONNX op types in a model that are not in the import registry.
pub fn unsupported_ops_in_model(model_path: &Path) -> Result<Vec<String>> {
    let (manifest, ..) = rlx_onnx_import::prepare_onnx_file(model_path)?;
    let mut missing = HashSet::new();
    for op in manifest.op_histogram.keys() {
        if !op_is_registered(op) {
            missing.insert(op.clone());
        }
    }
    let mut v: Vec<_> = missing.into_iter().collect();
    v.sort();
    Ok(v)
}

/// Coverage dashboard: registry size grouped by [`OpCategory`].
pub fn coverage_dashboard() -> String {
    format_registry_dashboard()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dashboard_groups_by_category() {
        let dash = coverage_dashboard();
        assert!(dash.contains("registered_ops="));
        assert!(dash.contains("[arithmetic]"));
        assert!(dash.contains("[control_flow]"));
    }
}
