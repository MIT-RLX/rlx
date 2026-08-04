// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-probe` — Tier 2 host probes that close the gaps the CSV sketches
//! can't see. Runs on a deep-dumped tensor file (`opscope-probe file.bin R C`)
//! or, with no args, on synthetic matrices with known structure so you can see
//! each probe discriminate.
//!
//! effective rank → factored matmul · quant error → int kernel · 2:4 error →
//! sparse tensor cores · cardinality → LUT / palettized weights.

use rlx_opscope::probe::{
    best_bitwidth, cardinality, load_tensor, nm_sparsity_error, outlier_channels,
    per_channel_quant_error, quant_error, stable_rank,
};
use rlx_opscope::{Dist, sample};

fn report(name: &str, x: &[f32], rows: usize, cols: usize) {
    let sr = stable_rank(x, rows, cols);
    let q4t = quant_error(x, 4); // per-tensor int4
    let q4c = per_channel_quant_error(x, rows, cols, 4); // per-channel int4
    let nm = nm_sparsity_error(x, rows, cols, 2, 4);
    let card = cardinality(x, 1000.0);
    let best = best_bitwidth(x, rows, cols, 0.02); // ≤2% per-channel error
    let (n_out, out_ratio) = outlier_channels(x, rows, cols, 6.0);

    // Pick the standout exploit — most-specific structural signal first, and
    // prefer per-channel / outlier-aware quant over per-tensor.
    let hint = if nm < 0.10 {
        "2:4 near-lossless → sparse tensor cores".to_string()
    } else if card <= 32 {
        format!("{card} distinct → LUT/palettized")
    } else if sr < 0.10 * cols as f32 {
        format!("low rank ≈{sr:.0} → factored matmul")
    } else if n_out > 0 && out_ratio > 8.0 {
        format!("{n_out} outlier ch ({out_ratio:.0}×) → AWQ/SmoothQuant mixed int4")
    } else if q4c < 0.05 && q4t >= 0.05 {
        "per-CHANNEL int4 (per-tensor too lossy) → grouped int4".into()
    } else {
        match best {
            Some(b) if b <= 4 => format!("int{b} per-channel"),
            Some(b) => format!("int{b}"),
            None => "dense fp — keep f16/bf16".into(),
        }
    };
    println!(
        "{name:<10} rank~{sr:>5.1}/{cols}  q4/tensor {q4t:>5.3}  q4/chan {q4c:>5.3}  best {}  out {n_out}({out_ratio:>3.0}×)  2:4 {nm:>4.2}   {hint}",
        best.map(|b| format!("int{b}"))
            .unwrap_or_else(|| "fp".into()),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 {
        // Probe a real deep-dumped tensor.
        let (r, c, x) = load_tensor(&args[1]).expect("load tensor");
        println!("dumped tensor {r}×{c}");
        report(&args[1], &x, r, c);
        return;
    }

    // Synthetic matrices with known structure — watch the probes discriminate.
    let (rows, cols) = (128usize, 128usize);
    println!(
        "{:<10} {:<24} {:<12} {:<12} {:<12}   exploit",
        "dist", "effective-rank", "quant", "2:4", "distinct"
    );
    println!("{}", "-".repeat(110));
    for d in [
        Dist::Gaussian,
        Dist::LowRank,
        Dist::Quantized,
        Dist::Sparse90,
    ] {
        let x = sample(d, rows, cols, 42);
        report(d.name(), &x, rows, cols);
    }
}
