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

// Several helpers (per-layer-bits search, bias correction, alternate
// emit paths) are kept around as scaffolding for future training-flow
// experiments. They're called from CLI flags that only some training
// runs exercise, so the unused-warning is expected.
#![allow(dead_code)]

//! Native fp32 trainer for the rlx-cortexm TinyConv-MNIST demo.
//!
//! Replaces the PyTorch-based `tools/train_mnist.py`: builds the same
//! TinyConv architecture as an `rlx_ir::Graph`, derives a gradient
//! graph via `rlx_opt::autodiff::grad_with_loss`, runs SGD through the
//! `rlx_cpu` executor, then quantizes the trained weights to INT8 and
//! emits `src/model_weights.rs` in the same format the firmware
//! consumes.
//!
//! Usage:
//! ```text
//! cargo run -p rlx-cortexm-trainer --release -- \
//!     --epochs 2 --batch 128 \
//!     --data ~/.cache/torchvision-mnist/MNIST/raw \
//!     --out  rlx-cortexm/src/model_weights.rs
//! ```
//!
//! Data layout: this trainer reads the standard MNIST IDX files
//! (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`,
//! `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`) — the same
//! files torchvision downloads under
//! `~/.cache/torchvision-mnist/MNIST/raw/`. Run the existing Python
//! script once with `python3 tools/train_mnist.py --epochs 0` (or
//! `python3 -c 'from torchvision import datasets; datasets.MNIST(...)'`)
//! if the cache isn't there yet.

mod bias_correct;
mod blob;
mod emit;
mod graph;
mod mnist;
mod quant;
mod train;

use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug)]
struct Args {
    epochs: usize,
    batch: usize,
    learning_rate: f32,
    momentum: f32,
    data_dir: PathBuf,
    out_path: PathBuf,
    seed: u64,
    /// Number of training images per epoch (0 = full 60_000).
    train_limit: usize,
    /// Number of test images for accuracy report (0 = full 10_000).
    eval_limit: usize,
    /// Number of test images to write to `tests/data/test_set.bin`
    /// for the firmware integration test.
    val_set_size: usize,
    /// Bits per weight: 8 (raw i8), 4 (nibble-packed), 2 (ternary, crumb-packed).
    /// Activations are always i8.
    weight_bits: u8,
    /// Quantization-aware training. When on, the graph builder wraps
    /// each conv/FC weight param in `Op::FakeQuantize { bits }` so the
    /// SGD optimizer sees the deployment-time rounding during training.
    /// Default: on for `weight_bits ∈ {2, 4}` (where PTQ accuracy
    /// degrades), off for `weight_bits = 8` (PTQ is fine at i8).
    qat: Option<bool>,
}

impl Default for Args {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_default();
        Self {
            epochs: 2,
            batch: 128,
            learning_rate: 0.05,
            momentum: 0.9,
            data_dir: PathBuf::from(format!("{home}/.cache/torchvision-mnist/MNIST/raw")),
            out_path: PathBuf::from("rlx-cortexm/src/model_weights.rs"),
            seed: 0,
            train_limit: 0,
            eval_limit: 0,
            val_set_size: 500,
            weight_bits: 8,
            qat: None,
        }
    }
}

impl Args {
    /// Resolve the `qat: Option<bool>` to a concrete on/off, applying
    /// the auto-rule when the user didn't pass `--qat`.
    pub fn qat_enabled(&self) -> bool {
        self.qat.unwrap_or(self.weight_bits < 8)
    }
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{arg} requires a value"));
        match arg.as_str() {
            "--epochs" => a.epochs = val()?.parse().map_err(|e| format!("--epochs: {e}"))?,
            "--batch" => a.batch = val()?.parse().map_err(|e| format!("--batch: {e}"))?,
            "--lr" => a.learning_rate = val()?.parse().map_err(|e| format!("--lr: {e}"))?,
            "--momentum" => a.momentum = val()?.parse().map_err(|e| format!("--momentum: {e}"))?,
            "--data" => a.data_dir = PathBuf::from(val()?),
            "--out" => a.out_path = PathBuf::from(val()?),
            "--seed" => a.seed = val()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--train-limit" => {
                a.train_limit = val()?.parse().map_err(|e| format!("--train-limit: {e}"))?
            }
            "--eval-limit" => {
                a.eval_limit = val()?.parse().map_err(|e| format!("--eval-limit: {e}"))?
            }
            "--val-set" => {
                a.val_set_size = val()?.parse().map_err(|e| format!("--val-set: {e}"))?
            }
            "--weight-bits" => {
                a.weight_bits = val()?.parse().map_err(|e| format!("--weight-bits: {e}"))?;
                if !matches!(a.weight_bits, 8 | 4 | 2) {
                    return Err(format!(
                        "--weight-bits must be 8, 4, or 2 (got {})",
                        a.weight_bits
                    ));
                }
            }
            "--qat" => {
                a.qat = Some(match val()?.as_str() {
                    "on" | "true" | "1" => true,
                    "off" | "false" | "0" => false,
                    "auto" => {
                        a.qat = None;
                        continue;
                    }
                    other => return Err(format!("--qat must be on/off/auto (got {other})")),
                });
            }
            "--help" | "-h" => {
                println!("{}", USAGE);
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
        }
    }
    Ok(a)
}

const USAGE: &str = "\
Usage: train-mnist [OPTIONS]

  --epochs N            Number of training epochs (default: 2)
  --batch N             Mini-batch size (default: 128)
  --lr F                Learning rate (default: 0.05)
  --momentum F          SGD momentum (default: 0.9)
  --data PATH           Directory containing MNIST IDX files
                        (default: ~/.cache/torchvision-mnist/MNIST/raw)
  --out PATH            Output path for model_weights.rs
                        (default: rlx-cortexm/src/model_weights.rs)
  --seed N              RNG seed for weight init + shuffling
                        (default: 0)
  --train-limit N       Use only the first N training images per epoch
                        (default: 0 = use all 60,000)
  --eval-limit N        Evaluate on N test images (default: 0 = all 10,000)
  --val-set N           Write N test images to tests/data/test_set.bin
                        for the firmware bulk-validation test (default: 500)
  --weight-bits N       Bits per weight: 8 (default), 4 (nibble-packed),
                        or 2 (ternary, crumb-packed). Activations are
                        always i8. Lower bits → smaller flash footprint
                        and accuracy cost in 0.5–2 pp range; see README.
  --qat MODE            Quantization-aware training: on/off/auto
                        (default: auto = on when --weight-bits < 8).
                        Wraps weight params in `Op::FakeQuantize` so
                        SGD sees deployment-time rounding. Required
                        for usable accuracy at i2 / i4.
  -h, --help            Show this help text";

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    eprintln!("{:#?}", args);

    if let Err(e) = run(&args) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(args: &Args) -> Result<(), String> {
    let dataset = mnist::load(&args.data_dir).map_err(|e| {
        format!(
            "loading MNIST from {}: {e}\n\n\
            If you don't have the IDX files yet, the easiest way is to run \
            the existing Python tool once (it will download via torchvision):\n  \
            python3 rlx-cortexm/tools/train_mnist.py --epochs 0\n\n\
            Then re-run this trainer.",
            args.data_dir.display()
        )
    })?;
    eprintln!(
        "MNIST: {} train, {} test images",
        dataset.train.len(),
        dataset.test.len()
    );

    let trained = train::run(&dataset, args)?;
    let calibrated = quant::calibrate_and_quantize(&trained, &dataset, args)?;
    emit::write_model_weights(&calibrated, &args.out_path)?;
    emit::write_test_set(&dataset, args)?;
    Ok(())
}
