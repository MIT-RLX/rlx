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

use std::path::{Path, PathBuf};

use anyhow::Result;
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};

/// Embedded ONNX `RandomNormalLike` (mean=0.1, scale=2, seed=7, template [2×3]).
pub fn random_normal_like_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rng_normal_like.onnx")
}

/// Embedded ONNX `RandomNormal` (mean=0.1, scale=2, seed=7, shape=[4]).
pub fn random_normal_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rng_normal.onnx")
}

/// Embedded ONNX `RandomUniformLike` (low=0, high=1, seed=7, template [2×3]).
pub fn random_uniform_like_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rng_uniform_like.onnx")
}

/// Embedded ONNX `RandomUniform` (low=0, high=1, seed=7, shape=[4]).
pub fn random_uniform_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rng_uniform.onnx")
}

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
