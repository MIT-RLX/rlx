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

//! Emit TinyConv-MNIST SystemVerilog for the checked-in demo tree.
//!
//! ```sh
//! cargo run -p rlx-fpga --example export_mnist --release
//! # writes examples/mnist_sv/
//! ```
//!
//! Or via the umbrella prelude (`rlx` feature `fpga`):
//!
//! ```ignore
//! use rlx::prelude::*;
//! ExportSession::fpga("out")
//!     .sideband(SidebandSpec::input("temp", 8))
//!     .export_model(&tinyconv_mnist_from_cortexm())?;
//! ```

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use rlx_fpga::codegen::emit_with_config;
use rlx_fpga::export_config::{
    FpgaExportConfig, HwTarget, IoConfig, OutputIface, OutputKind, SidebandSpec,
};
use rlx_fpga::model::tinyconv_mnist_from_cortexm;
use rlx_fpga::verilog::mem_hex_bytes;
use rlx_fpga::weights::TEST_IMAGE;

fn main() -> ExitCode {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/mnist_sv"));

    // ASIC / SoC soft ports + memory readout + demo scalar sidebands.
    let cfg = FpgaExportConfig::default()
        .with_hw_target(HwTarget::Generic)
        .with_output_kind(OutputKind::Argmax)
        .with_io(
            IoConfig::default()
                .with_output(OutputIface::ScalarAndMemory)
                .sideband(SidebandSpec::input("temp", 8))
                .sideband(SidebandSpec::input("batch_id", 16)),
        );

    eprintln!("emitting TinyConv-MNIST → {}", out.display());
    if let Err(e) = emit_with_config(&tinyconv_mnist_from_cortexm(), &cfg, &out) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    if let Err(e) = fs::write(out.join("tb_image.mem"), mem_hex_bytes(TEST_IMAGE)) {
        eprintln!("error writing tb_image.mem: {e}");
        return ExitCode::FAILURE;
    }
    eprintln!("done. Open examples/mnist_sv/top.sv — soft ports + temp/batch_id sidebands.");
    ExitCode::SUCCESS
}
