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

//! `cargo run -p rlx-fpga --bin rlx-fpga-emit [-- [<target>] [<out_dir>]]`
//!
//! `<target>` is one of `latency | size | energy | precision | bandwidth`,
//! defaulting to `precision` (bit-exact with the reference). `<out_dir>`
//! defaults to `rlx-fpga/hw/<model_name>__<target>/`.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use rlx_fpga::codegen::emit_model_tuned;
use rlx_fpga::estimate::estimate;
use rlx_fpga::model::tinyconv_mnist_from_cortexm;
use rlx_fpga::passes::{optimize, summary};
use rlx_fpga::tune::{OptTarget, Tune};
use rlx_fpga::verilog::mem_hex_bytes;
use rlx_fpga::weights::TEST_IMAGE;

fn parse_target(s: &str) -> Option<OptTarget> {
    Some(match s.to_ascii_lowercase().as_str() {
        "latency" => OptTarget::Latency,
        "size" => OptTarget::Size,
        "energy" => OptTarget::Energy,
        "precision" => OptTarget::Precision,
        "bandwidth" => OptTarget::Bandwidth,
        _ => return None,
    })
}

fn main() -> ExitCode {
    let model = tinyconv_mnist_from_cortexm();

    let mut args = std::env::args().skip(1);
    let arg1 = args.next();
    let arg2 = args.next();

    let (target, out_dir): (OptTarget, PathBuf) = match (arg1.as_deref(), arg2.as_deref()) {
        (Some(s), Some(p)) => match parse_target(s) {
            Some(t) => (t, PathBuf::from(p)),
            None => {
                eprintln!("error: unknown target {s:?}");
                return ExitCode::FAILURE;
            }
        },
        (Some(s), None) => match parse_target(s) {
            Some(t) => (t, default_out(&model.name, t)),
            None => (OptTarget::Precision, PathBuf::from(s)),
        },
        (None, _) => (
            OptTarget::Precision,
            default_out(&model.name, OptTarget::Precision),
        ),
    };

    let tune = Tune::for_target(target);
    eprintln!(
        "emitting {} (target={:?}) → {}",
        model.name,
        target,
        out_dir.display()
    );
    eprintln!("  {}", tune);

    let opt = optimize(&model, &tune);
    eprintln!("  {}", summary(&opt));
    let est = estimate(&opt);
    eprintln!("  {}", est.summary());

    if let Err(e) = emit_model_tuned(&model, &tune, &out_dir) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }

    let mem_path = out_dir.join("tb_image.mem");
    if let Err(e) = fs::write(&mem_path, mem_hex_bytes(TEST_IMAGE)) {
        eprintln!("error writing {}: {e}", mem_path.display());
        return ExitCode::FAILURE;
    }

    eprintln!("done.");
    ExitCode::SUCCESS
}

fn default_out(model_name: &str, target: OptTarget) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("hw");
    let tag = format!("{model_name}__{:?}", target).to_lowercase();
    p.push(tag);
    p
}
