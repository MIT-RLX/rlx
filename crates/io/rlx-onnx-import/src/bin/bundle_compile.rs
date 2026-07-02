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
