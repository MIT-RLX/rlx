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

//! `rlx-qnn-emit` — write the QNN model artifact set for a single matmul,
//! linear, linear+relu, matmul+softmax, two-layer MLP, or static-weight
//! linear graph.
//!
//! ```text
//!   rlx-qnn-emit <M> <K> <N> [out_dir]
//!   rlx-qnn-emit --linear <M> <K> <N> [out_dir]
//!   rlx-qnn-emit --linear-relu <M> <K> <N> [out_dir]
//!   rlx-qnn-emit --matmul-softmax <M> <K> <N> [out_dir]
//!   rlx-qnn-emit --linear-static <M> <K> <N> [out_dir]
//!   rlx-qnn-emit --mlp2 <M> <K> <H> <N> [out_dir]
//! ```
//!
//! Emits `qnn_model.cpp`, `verify.py`, and `run_qnn.sh` into `out_dir`
//! (default `./qnn-out`). Build + run on a Linux host with the QNN SDK:
//! `cd qnn-out && bash run_qnn.sh`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rlx_qnn::{Model, emit_model};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerKind {
    MatMul,
    Linear,
    LinearRelu,
    MatMulSoftmax,
    LinearStatic,
    Mlp2,
}

fn usage(argv0: &str) {
    eprintln!(
        "usage: {argv0} [--linear | --linear-relu | --matmul-softmax | --linear-static] <M> <K> <N> [out_dir]"
    );
    eprintln!("       {argv0} --mlp2 <M> <K> <H> <N> [out_dir]");
    eprintln!("       {argv0} <M> <K> <N> [out_dir]  (default: matmul)");
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        usage(&args[0]);
        return ExitCode::FAILURE;
    }

    let (kind, rest): (LayerKind, &[String]) = match args[1].as_str() {
        "--linear" | "linear" => (LayerKind::Linear, &args[2..]),
        "--linear-relu" | "linear-relu" => (LayerKind::LinearRelu, &args[2..]),
        "--matmul-softmax" | "matmul-softmax" => (LayerKind::MatMulSoftmax, &args[2..]),
        "--linear-static" | "linear-static" => (LayerKind::LinearStatic, &args[2..]),
        "--mlp2" | "mlp2" => (LayerKind::Mlp2, &args[2..]),
        _ => (LayerKind::MatMul, &args[1..]),
    };

    let parse = |s: &str, name: &str| -> Result<usize, String> {
        s.parse::<usize>()
            .map_err(|_| format!("{name} must be a positive integer, got {s:?}"))
    };

    if kind == LayerKind::Mlp2 {
        if rest.len() < 4 {
            usage(&args[0]);
            return ExitCode::FAILURE;
        }
        let (m, k, h, n) = match (
            parse(&rest[0], "M"),
            parse(&rest[1], "K"),
            parse(&rest[2], "H"),
            parse(&rest[3], "N"),
        ) {
            (Ok(m), Ok(k), Ok(h), Ok(n)) => (m, k, h, n),
            (e1, e2, e3, e4) => {
                for e in [e1, e2, e3, e4].into_iter().filter_map(Result::err) {
                    eprintln!("error: {e}");
                }
                return ExitCode::FAILURE;
            }
        };
        let out_dir = rest
            .get(4)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("qnn-out"));
        let model = Model::mlp2(format!("mlp2_{m}x{k}x{h}x{n}"), m, k, h, n);
        return emit_and_report(&model, &out_dir, &format!("{m}x{k}x{h}x{n} mlp2"));
    }

    if rest.len() < 3 {
        usage(&args[0]);
        return ExitCode::FAILURE;
    }

    let (m, k, n) = match (
        parse(&rest[0], "M"),
        parse(&rest[1], "K"),
        parse(&rest[2], "N"),
    ) {
        (Ok(m), Ok(k), Ok(n)) => (m, k, n),
        (e1, e2, e3) => {
            for e in [e1, e2, e3].into_iter().filter_map(Result::err) {
                eprintln!("error: {e}");
            }
            return ExitCode::FAILURE;
        }
    };
    let out_dir = rest
        .get(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("qnn-out"));

    let (model, kind_label) = match kind {
        LayerKind::MatMul => (
            Model::single_matmul(format!("matmul_{m}x{k}x{n}"), m, k, n),
            "matmul",
        ),
        LayerKind::Linear => (
            Model::linear(format!("linear_{m}x{k}x{n}"), m, k, n),
            "linear",
        ),
        LayerKind::LinearRelu => (
            Model::linear_relu(format!("linear_relu_{m}x{k}x{n}"), m, k, n),
            "linear+relu",
        ),
        LayerKind::MatMulSoftmax => (
            Model::matmul_softmax(format!("matmul_softmax_{m}x{k}x{n}"), m, k, n),
            "matmul+softmax",
        ),
        LayerKind::LinearStatic => (
            Model::linear_static(format!("linear_static_{m}x{k}x{n}"), m, k, n),
            "linear-static",
        ),
        LayerKind::Mlp2 => unreachable!(),
    };

    emit_and_report(&model, &out_dir, &format!("{m}x{k}x{n} {kind_label}"))
}

fn emit_and_report(model: &Model, out_dir: &Path, label: &str) -> ExitCode {
    match emit_model(model, out_dir) {
        Ok(()) => {
            println!(
                "wrote QNN model artifacts for {label} to {}",
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
