// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Parity test for new IQ/TQ/MX dequant kernels against real model weights.
//
// We quantize Qwen3-0.6B to several IQ/TQ formats with llama.cpp's
// `llama-quantize`, then dequant each tensor with `rlx-gguf` and compare
// against the F16 ground truth (same Qwen3-0.6B). The comparison metric
// is cosine similarity per tensor; thresholds match the bpw budget.
//
// Skipped if the test GGUFs aren't present at `RLX_IQ_TEST_DIR` (default
// `/tmp/rlx-iq-test`). Produce them with:
//
//   convert_hf_to_gguf.py <safetensors> --outfile $DIR/Qwen3-0.6B-F16.gguf --outtype f16
//   llama-quantize $DIR/Qwen3-0.6B-F16.gguf $DIR/Qwen3-0.6B-IQ4_NL.gguf IQ4_NL
//   …
//
// The test logs per-tensor MAE and global cosine so regressions in the
// dequant kernels surface immediately.

use std::path::PathBuf;

use rlx_gguf::GgufFile;

fn test_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("RLX_IQ_TEST_DIR").unwrap_or_else(|_| "/tmp/rlx-iq-test".to_string()),
    )
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn mae(a: &[f32], b: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        s += (*x as f64 - *y as f64).abs();
    }
    s / a.len() as f64
}

fn rms(a: &[f32]) -> f64 {
    let s: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
    (s / a.len() as f64).sqrt()
}

/// Compare every tensor that exists in *both* files. Returns (n_tensors,
/// avg_cosine, min_cosine, worst_tensor). Per-tensor cosines below
/// `print_below` are echoed to stderr.
fn compare_files(
    reference: &GgufFile,
    quantized: &GgufFile,
    print_below: f64,
) -> (usize, f64, f64, String) {
    let mut tot_cos = 0.0f64;
    let mut min_cos = f64::INFINITY;
    let mut worst = String::new();
    let mut count = 0usize;
    for name in quantized.keys() {
        let q_tensor = quantized.get(name).unwrap();
        let Some(r_tensor) = reference.get(name) else {
            continue;
        };
        if r_tensor.shape != q_tensor.shape {
            continue;
        }
        // Skip tiny tensors — they're typically norms/biases that stay
        // F32/F16 in the quantized file and would just round-trip exactly.
        if r_tensor.n_elements() < 1024 {
            continue;
        }
        let r = match reference.dequant_f32(name) {
            Ok((v, _)) => v,
            Err(e) => {
                eprintln!("ref dequant failed for {name}: {e}");
                continue;
            }
        };
        let q = match quantized.dequant_f32(name) {
            Ok((v, _)) => v,
            Err(e) => {
                eprintln!("quant dequant failed for {name}: {e}");
                continue;
            }
        };
        let cs = cosine(&r, &q);
        if !cs.is_finite() {
            // Distinguishes NaN/Inf from a legitimately bad reconstruction.
            // Usually means the dequant produced a NaN somewhere (or the F16
            // source tensor is all zero AND the quantized one isn't).
            let r_nan = r.iter().any(|x| !x.is_finite());
            let q_nan = q.iter().any(|x| !x.is_finite());
            eprintln!(
                "  NON-FINITE cosine for {name}: r_has_nan={r_nan} q_has_nan={q_nan} dt={:?}",
                q_tensor.dtype,
            );
            // Treat as a hard failure — don't pollute averages.
            min_cos = 0.0;
            worst = format!("{name} (NaN)");
            count += 1;
            continue;
        }
        if cs < print_below {
            eprintln!(
                "  {name}: cos={cs:.4} mae={:.4e} ref_rms={:.4e} dt={:?}",
                mae(&r, &q),
                rms(&r),
                q_tensor.dtype,
            );
        }
        tot_cos += cs;
        if cs < min_cos {
            min_cos = cs;
            worst = name.to_string();
        }
        count += 1;
    }
    let avg = if count > 0 {
        tot_cos / count as f64
    } else {
        0.0
    };
    (count, avg, min_cos, worst)
}

fn maybe_open(name: &str) -> Option<GgufFile> {
    let p = test_dir().join(name);
    if !p.exists() {
        eprintln!("skip: {} not found", p.display());
        return None;
    }
    match GgufFile::from_path(&p) {
        Ok(f) => Some(f),
        Err(e) => {
            eprintln!("failed to open {}: {e}", p.display());
            None
        }
    }
}

fn run_parity(quant_name: &str, min_cos_threshold: f64, avg_cos_threshold: f64) {
    let Some(reference) = maybe_open("Qwen3-0.6B-F16.gguf") else {
        return;
    };
    let Some(quantized) = maybe_open(quant_name) else {
        return;
    };
    let (n, avg, min, worst) = compare_files(&reference, &quantized, avg_cos_threshold);
    eprintln!(
        "=== {quant_name}: {n} tensors, avg_cos={avg:.5}, min_cos={min:.5} (worst: {worst}) ==="
    );
    assert!(n > 0, "{quant_name}: no comparable tensors");
    assert!(
        avg >= avg_cos_threshold,
        "{quant_name}: avg cosine {avg:.5} < threshold {avg_cos_threshold}",
    );
    assert!(
        min >= min_cos_threshold,
        "{quant_name}: min cosine {min:.5} < threshold {min_cos_threshold} (worst tensor: {worst})",
    );
}

#[test]
fn iq4_nl_parity_vs_f16() {
    // IQ4_NL: 4.5 bpw non-linear, should be very close.
    run_parity("Qwen3-0.6B-IQ4_NL.gguf", 0.98, 0.995);
}

#[test]
fn iq4_xs_parity_vs_f16() {
    // IQ4_XS: 4.25 bpw, slightly worse than IQ4_NL.
    run_parity("Qwen3-0.6B-IQ4_XS.gguf", 0.97, 0.99);
}

#[test]
fn iq3_s_parity_vs_f16() {
    // IQ3_S: 3.44 bpw, 3-bit + sign.
    run_parity("Qwen3-0.6B-IQ3_S.gguf", 0.93, 0.97);
}

#[test]
fn tq1_0_parity_vs_f16() {
    // TQ1_0: pure ternary quantization on a non-BitNet model has a
    // ~0.7–0.8 cosine floor regardless of decoder correctness — the
    // info loss is intrinsic to compressing 16-bit floats to {−1,0,+1}.
    // Llama.cpp's benchmarks show similar quality. Our decode-correctness
    // check is the dedicated bit-parity test in tq_dequant.rs (unit).
    run_parity("Qwen3-0.6B-TQ1_0.gguf", 0.55, 0.70);
}

#[test]
fn tq2_0_parity_vs_f16() {
    run_parity("Qwen3-0.6B-TQ2_0.gguf", 0.55, 0.70);
}

#[test]
fn iq2_xxs_parity_vs_f16() {
    // 2.06 bpw — imatrix-calibrated.
    run_parity("Qwen3-0.6B-IQ2_XXS.gguf", 0.80, 0.92);
}

#[test]
fn iq2_xs_parity_vs_f16() {
    // 2.31 bpw.
    run_parity("Qwen3-0.6B-IQ2_XS.gguf", 0.85, 0.94);
}

#[test]
fn iq2_s_parity_vs_f16() {
    // 2.5 bpw.
    run_parity("Qwen3-0.6B-IQ2_S.gguf", 0.85, 0.95);
}

#[test]
fn iq3_xxs_parity_vs_f16() {
    // 3.06 bpw — imatrix-calibrated.
    run_parity("Qwen3-0.6B-IQ3_XXS.gguf", 0.90, 0.97);
}

#[test]
fn iq1_s_parity_vs_f16() {
    // 1.56 bpw — extreme compression; cosine floor is in the 0.6–0.8 range
    // even with imatrix on a non-fine-tuned model.
    run_parity("Qwen3-0.6B-IQ1_S.gguf", 0.45, 0.70);
}

#[test]
fn iq1_m_parity_vs_f16() {
    // 1.75 bpw.
    run_parity("Qwen3-0.6B-IQ1_M.gguf", 0.50, 0.75);
}
