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

//! Minimal synthetic ONNX graphs for registry ops.

use std::path::Path;

use anyhow::Result;
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};

/// Build HIR from an ONNX file using generic strict import (no quant-bundle rewrites).
pub fn import_onnx_strict(path: &Path) -> Result<rlx_ir::hir::HirModule> {
    let opts = ImportOptions {
        strict: true,
        use_quantized_kernels: false,
        ..ImportOptions::default()
    };
    let (hir, _params, _report, _manifest) = build_hir_from_onnx_file(path, opts)?;
    Ok(hir)
}
