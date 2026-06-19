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

//! CLI: load an ONNX file and run one inference (zero-filled inputs by default).

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rlx_onnx::{OnnxCompileLevel, OnnxExecBackend, OnnxModel};
use rlx_runtime::{Device, parse_device};

fn usage() -> &'static str {
    "usage: rlx-onnx-run <model.onnx> [--device cpu|cuda|metal|...] [--level 0-3] [--exec native|ort] [--list-io] [--seq-len N] [--warmup N] [--iters N]"
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let model = args
        .next()
        .map(PathBuf::from)
        .filter(|p| p.extension().is_some_and(|e| e == "onnx"))
        .context(usage())?;

    let mut device = Device::Cpu;
    let mut list_io = false;
    let mut warmup = 1usize;
    let mut iters = 1usize;
    let mut seq_len = 128usize;
    let mut level = OnnxCompileLevel::Level3;
    let mut backend = OnnxExecBackend::Native;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--device" => {
                let s = args.next().context("--device requires a value")?;
                device = parse_device(&s)?;
            }
            "--level" => {
                let n: u8 = args
                    .next()
                    .context("--level 0-3")?
                    .parse()
                    .context("--level 0-3")?;
                level = OnnxCompileLevel::from_u8(n);
            }
            "--exec" => {
                let s = args.next().context("--exec native|ort")?;
                backend = match s.as_str() {
                    "native" | "rlx" => OnnxExecBackend::Native,
                    "ort" => {
                        #[cfg(feature = "ort")]
                        {
                            OnnxExecBackend::Ort
                        }
                        #[cfg(not(feature = "ort"))]
                        {
                            bail!(
                                "rlx-onnx-run: rebuild with feature `ort-fallback` for --exec ort"
                            )
                        }
                    }
                    other => bail!("unknown --exec {other} (use native or ort)"),
                };
            }
            "--list-io" => list_io = true,
            "--warmup" => {
                warmup = args
                    .next()
                    .context("--warmup N")?
                    .parse()
                    .context("--warmup N")?;
            }
            "--iters" => {
                iters = args
                    .next()
                    .context("--iters N")?
                    .parse()
                    .context("--iters N")?;
            }
            "--seq-len" => {
                seq_len = args
                    .next()
                    .context("--seq-len N")?
                    .parse()
                    .context("--seq-len N")?;
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    let mut model = OnnxModel::load_with(&model, device, backend, level, seq_len)?;
    if list_io {
        model.print_io();
        return Ok(());
    }

    let inputs = model.zero_inputs_sized(seq_len as i64)?;
    for _ in 0..warmup {
        let _ = model.run(&inputs)?;
    }

    let t0 = Instant::now();
    let mut last = None;
    for _ in 0..iters {
        last = Some(model.run(&inputs)?);
    }
    let elapsed = t0.elapsed();

    let outs = last.context("no iterations")?;
    let backend_s = format!("{:?}", model.backend);
    let ep = model
        .ort_ep
        .as_deref()
        .map(|e| format!(", ort_ep={e}"))
        .unwrap_or_default();
    println!(
        "ok: {} output(s), {} iter(s), {:.2} ms total ({:.2} ms/iter), backend={backend_s}{ep}, level={:?}",
        outs.len(),
        iters,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / iters as f64,
        model.compile_level,
    );
    for (desc, tensor) in model.outputs.iter().zip(outs.iter()) {
        let n = match tensor {
            rlx_onnx::OnnxTensor::F32(v) => v.len(),
            rlx_onnx::OnnxTensor::I64(v) => v.len(),
            rlx_onnx::OnnxTensor::I32(v) => v.len(),
        };
        println!("  output {}: {} elements", desc.name, n);
    }
    Ok(())
}
