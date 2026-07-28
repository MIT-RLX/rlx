// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compile-check an ONNX RLX bundle (HIR lower + CPU compile).
//!
//! ```sh
//! RLX_ONNX_BUNDLE=/path/to/bundle \
//!   cargo run -p rlx-onnx-import --features runtime --bin bundle-compile --release
//! ```

use anyhow::Result;
use rlx_onnx_import::{
    ImportOptions, build_hir_from_bundle, load_bundle, onnx_bundle_dir, onnx_scatter,
};

fn main() -> Result<()> {
    onnx_scatter::register_onnx_scatter_elements_kernel();
    let bundle = load_bundle(&onnx_bundle_dir())?;
    let (_hir, _params, _typed, report) =
        build_hir_from_bundle(&bundle, ImportOptions::quant_bundle())?;
    println!(
        "compile-check ok: lowered={} skipped={} stubbed={}",
        report.lowered, report.skipped, report.stubbed
    );
    Ok(())
}
