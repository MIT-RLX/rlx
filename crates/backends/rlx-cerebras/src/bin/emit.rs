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

//! `rlx-cerebras-emit` — write the CSL artifact set for a single matmul.
//!
//! ```text
//!   rlx-cerebras-emit <M> <K> <N> [out_dir]
//! ```
//!
//! Emits `layout.csl`, `pe_program.csl`, `run.py`, `commands_wse{2,3}.sh`
//! into `out_dir` (default `./cerebras-out`). Compile + run on a Linux host
//! with the Cerebras SDK container:  `bash commands_wse2.sh`.

use std::path::PathBuf;
use std::process::ExitCode;

use rlx_cerebras::{Model, emit_model};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: {} <M> <K> <N> [out_dir]", args[0]);
        return ExitCode::FAILURE;
    }
    let parse = |s: &str, name: &str| -> Result<usize, String> {
        s.parse::<usize>()
            .map_err(|_| format!("{name} must be a positive integer, got {s:?}"))
    };
    let (m, k, n) = match (
        parse(&args[1], "M"),
        parse(&args[2], "K"),
        parse(&args[3], "N"),
    ) {
        (Ok(m), Ok(k), Ok(n)) => (m, k, n),
        (e1, e2, e3) => {
            for e in [e1, e2, e3].into_iter().filter_map(Result::err) {
                eprintln!("error: {e}");
            }
            return ExitCode::FAILURE;
        }
    };
    let out_dir = args
        .get(4)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("cerebras-out"));

    let model = Model::single_matmul(format!("matmul_{m}x{k}x{n}"), m, k, n);
    match emit_model(&model, &out_dir) {
        Ok(()) => {
            println!(
                "wrote CSL artifacts for {m}x{k}x{n} matmul to {}",
                out_dir.display()
            );
            println!(
                "next (Linux + Cerebras SDK container): cd {} && bash commands_wse2.sh",
                out_dir.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
