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

//! Convert a `.safetensors` file to GGUF with a chosen quant scheme.
//!
//! ```text
//! cargo run --example convert -p rlx-gguf-convert -- \
//!     model.safetensors model.q4_k.gguf Q4_K
//! ```
//!
//! Optional fourth arg is the architecture string written to
//! `general.architecture` (defaults to `"unknown"`).

use std::env;

use anyhow::{Context, Result, bail};

use rlx_gguf_convert::{Converter, Scheme};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        bail!(
            "usage: {} <input.safetensors> <output.gguf> <scheme> [arch]",
            args.first().map(|s| s.as_str()).unwrap_or("convert")
        );
    }
    let input = &args[1];
    let output = &args[2];
    let scheme = Scheme::parse(&args[3])?;
    let arch = args.get(4).cloned().unwrap_or_else(|| "unknown".into());

    let report = Converter::from_safetensors(input)
        .context("opening safetensors")?
        .default_scheme(scheme)
        // Keep biases / norms / 1-D tensors at native precision.
        .skip_quant_for(|name, shape| {
            shape.len() < 2
                || name.contains("norm")
                || name.contains("bias")
                || name.ends_with(".b")
        })
        .architecture(arch)
        .write_gguf(output)?;

    eprintln!(
        "Wrote {} tensors → {} ({:.1}% of source, {:.2}× smaller)",
        report.tensors,
        report.output_path.display(),
        report.output_bytes as f64 / report.input_bytes as f64 * 100.0,
        report.compression_ratio(),
    );
    // Per-scheme histogram.
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for (_, s) in &report.schemes {
        *counts.entry(format!("{s:?}")).or_default() += 1;
    }
    for (s, n) in &counts {
        eprintln!("  {s:>8}: {n} tensors");
    }
    Ok(())
}
