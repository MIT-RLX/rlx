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

//! `rlx-qnn-emit` — write the QNN model artifact set for a single matmul.
//!
//! ```text
//!   rlx-qnn-emit <M> <K> <N> [out_dir]
//! ```
//!
//! Emits `qnn_model.cpp`, `verify.py`, and `run_qnn.sh` into `out_dir`
//! (default `./qnn-out`). Build + run on a Linux host with the QNN SDK:
//! `cd qnn-out && bash run_qnn.sh`.

use std::path::PathBuf;
use std::process::ExitCode;

use rlx_qnn::{Model, emit_model};

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
        .unwrap_or_else(|| PathBuf::from("qnn-out"));

    let model = Model::single_matmul(format!("matmul_{m}x{k}x{n}"), m, k, n);
    match emit_model(&model, &out_dir) {
        Ok(()) => {
            println!(
                "wrote QNN model artifacts for {m}x{k}x{n} matmul to {}",
                out_dir.display()
            );
            println!(
                "next (Linux + QNN SDK): cd {} && bash run_qnn.sh",
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
