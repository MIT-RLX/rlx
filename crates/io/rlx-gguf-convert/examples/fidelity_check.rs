// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fidelity report for a (safetensors, GGUF) pair: for every tensor
//! present in both files, compute cosine similarity, max abs error,
//! and RMS error between the safetensors values and the dequantized
//! GGUF values.
//!
//! ```text
//! cargo run --release --example fidelity_check -p rlx-gguf-convert -- \
//!     model.safetensors model.q4_k.gguf
//! ```

use std::env;

use anyhow::{Context, Result, bail};
use safetensors::{SafeTensors, tensor::Dtype as StDtype};

fn dtype_to_f32(dtype: StDtype, bytes: &[u8]) -> Vec<f32> {
    match dtype {
        StDtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        StDtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        StDtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        _ => Vec::new(),
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        bail!("usage: {} <safetensors> <gguf>", args[0]);
    }
    let st_bytes = std::fs::read(&args[1]).context("reading safetensors")?;
    let st = SafeTensors::deserialize(&st_bytes).context("parsing safetensors")?;
    let gguf = rlx_gguf::GgufFile::from_path(&args[2])?;

    let mut covered = 0usize;
    let mut by_dtype: std::collections::BTreeMap<String, (usize, f64, f64, f64)> =
        std::collections::BTreeMap::new();
    println!(
        "{:<60} {:>6} {:>8} {:>10} {:>10} {:>10}",
        "tensor", "dtype", "n", "cos", "max_err", "rms_err"
    );
    println!("{}", "─".repeat(110));
    for name in st.names() {
        let st_view = st.tensor(name)?;
        let original = dtype_to_f32(st_view.dtype(), st_view.data());
        if original.is_empty() {
            continue;
        }
        let gguf_t = match gguf.get(name) {
            Some(t) => t,
            None => continue,
        };
        let (decoded, _shape) = match gguf.dequant_f32(name) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if decoded.len() != original.len() {
            eprintln!(
                "WARN {name}: length mismatch ({} vs {})",
                decoded.len(),
                original.len()
            );
            continue;
        }
        let n = original.len();
        let dot: f64 = original
            .iter()
            .zip(&decoded)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let na: f64 = original
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = decoded
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cos = if na == 0.0 || nb == 0.0 {
            1.0
        } else {
            dot / (na * nb)
        };
        let max_err = original
            .iter()
            .zip(&decoded)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let rms = {
            let sq: f64 = original
                .iter()
                .zip(&decoded)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            (sq / n as f64).sqrt()
        };
        let dtype_s = format!("{:?}", gguf_t.dtype);
        println!(
            "{:<60} {:>6} {:>8} {:>10.6} {:>10.4e} {:>10.4e}",
            name, dtype_s, n, cos, max_err, rms
        );
        covered += 1;
        let entry = by_dtype.entry(dtype_s).or_insert((0, 0.0, 0.0, 0.0));
        entry.0 += 1;
        entry.1 += cos;
        entry.2 = entry.2.max(max_err as f64);
        entry.3 += rms;
    }
    println!("{}", "─".repeat(110));
    println!("covered {covered} tensors");
    for (dtype, (count, cos_sum, max_err, rms_sum)) in &by_dtype {
        println!(
            "  {dtype:>6}: n={count:<4}  mean_cos={:.6}  max_err={:.4e}  mean_rms={:.4e}",
            cos_sum / *count as f64,
            max_err,
            rms_sum / *count as f64
        );
    }
    Ok(())
}
