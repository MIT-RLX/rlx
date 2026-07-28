// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Numeric parity check: run the imported model on CPU and compare against the
//! golden `reference.safetensors` PyTorch captured for the same inputs.

use crate::call::Lowered;
use crate::run::{assemble_params, load_safetensors_f32, run_cpu, run_dynamic};
use anyhow::{Result, anyhow};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize)]
pub struct OutputParity {
    pub index: usize,
    pub cosine: f64,
    pub max_abs_err: f64,
    pub rel_err: f64,
    pub numel: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParityReport {
    pub passed: bool,
    pub outputs: Vec<OutputParity>,
    pub cosine_threshold: f64,
    pub max_abs_threshold: f64,
}

/// Compare produced vs expected outputs (cosine + max abs err), shared by the
/// torch-IR and bundle verify paths.
pub fn compare_outputs(
    got: &[Vec<f32>],
    expected: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> (bool, Vec<OutputParity>) {
    let cosine_threshold = 0.999;
    let max_abs_threshold = 1e-2;
    let mut outputs = Vec::new();
    let mut passed = true;
    for (i, out) in got.iter().enumerate() {
        let Some((exp, _)) = expected.get(&format!("out::{i}")) else {
            continue;
        };
        if exp.len() != out.len() {
            passed = false;
            outputs.push(OutputParity {
                index: i,
                cosine: 0.0,
                max_abs_err: f64::INFINITY,
                rel_err: f64::INFINITY,
                numel: out.len(),
            });
            continue;
        }
        let cos = cosine(out, exp);
        let mut max_abs = 0.0f64;
        let mut denom = 0.0f64;
        for (&g, &e) in out.iter().zip(exp) {
            max_abs = max_abs.max((g as f64 - e as f64).abs());
            denom = denom.max((e as f64).abs());
        }
        let rel = if denom > 0.0 {
            max_abs / denom
        } else {
            max_abs
        };
        if cos < cosine_threshold || max_abs > max_abs_threshold {
            passed = false;
        }
        outputs.push(OutputParity {
            index: i,
            cosine: cos,
            max_abs_err: max_abs,
            rel_err: rel,
            numel: out.len(),
        });
    }
    (passed, outputs)
}

/// Run a bundle (any front-end) against a golden `reference.safetensors` and
/// report parity.
#[cfg(feature = "runtime")]
pub fn verify_bundle(bundle_dir: &Path, reference: &Path) -> Result<ParityReport> {
    let meta: crate::BundleMeta =
        serde_json::from_str(&std::fs::read_to_string(bundle_dir.join("meta.json"))?)?;
    let refm = load_safetensors_f32(reference)?;
    let inputs: Vec<(String, Vec<f32>)> = meta
        .inputs
        .iter()
        .map(|i| {
            let key = format!("in::{}", i.name);
            let (data, _) = refm
                .get(&key)
                .ok_or_else(|| anyhow!("reference input {key:?} missing"))?;
            Ok((i.name.clone(), data.clone()))
        })
        .collect::<Result<Vec<_>>>()?;
    let got = crate::run_bundle(bundle_dir, &inputs)?;
    let (passed, outputs) = compare_outputs(&got, &refm);
    Ok(ParityReport {
        passed,
        outputs,
        cosine_threshold: 0.999,
        max_abs_threshold: 1e-2,
    })
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Run the imported model on `device` against golden inputs/outputs in `dir`.
#[cfg(feature = "runtime")]
pub fn verify(
    dir: &Path,
    lo: &Lowered,
    hir: rlx_ir::HirModule,
    device: rlx_runtime::Device,
) -> Result<ParityReport> {
    let weights_path = dir.join("weights.safetensors");
    let weights = load_safetensors_f32(&weights_path)?;
    let params = assemble_params(lo, &weights)?;

    let ref_path = dir.join("reference.safetensors");
    let reference = load_safetensors_f32(&ref_path)
        .map_err(|e| anyhow!("reference.safetensors required for --verify: {e}"))?;

    // Inputs are always fed as f32 (integer inputs become an f32 graph input +
    // a cast in `hir_build`), so this works on every f32-arena backend incl. CUDA.
    let inputs: Vec<(String, Vec<f32>)> = lo
        .inputs
        .iter()
        .map(|i| {
            let key = format!("in::{}", i.name);
            let (data, _) = reference
                .get(&key)
                .ok_or_else(|| anyhow!("reference input {key:?} missing"))?;
            Ok((i.name.clone(), data.clone()))
        })
        .collect::<Result<Vec<_>>>()?;

    // Dynamic-shape models: bind each symbolic dim to the reference input's
    // actual extent, then run the specialized graph. (The graph can be re-run at
    // any other binding — verify just checks the reference shape.)
    let got = if lo.inputs.iter().any(|i| i.is_dynamic()) {
        let mut pairs: Vec<(u32, usize)> = Vec::new();
        for i in lo.inputs.iter().filter(|i| i.is_dynamic()) {
            let (_, shape) = reference
                .get(&format!("in::{}", i.name))
                .ok_or_else(|| anyhow!("reference input for {:?} missing", i.name))?;
            for (ax, dd) in i.dyn_dims.iter().enumerate() {
                if let (Some(sym), Some(&sz)) = (dd, shape.get(ax)) {
                    pairs.push((*sym, sz));
                }
            }
        }
        run_dynamic(
            hir,
            &params,
            &inputs,
            device,
            &rlx_ir::DimBinding::from_pairs(&pairs),
        )?
    } else {
        run_cpu(hir, &params, &[], &inputs, device)?
    };

    let cosine_threshold = 0.999;
    let max_abs_threshold = 1e-2;
    let mut outputs = Vec::new();
    let mut passed = true;
    for (i, out) in got.iter().enumerate() {
        let key = format!("out::{i}");
        let (expected, _) = reference
            .get(&key)
            .ok_or_else(|| anyhow!("reference output {key:?} missing"))?;
        if expected.len() != out.len() {
            passed = false;
            outputs.push(OutputParity {
                index: i,
                cosine: 0.0,
                max_abs_err: f64::INFINITY,
                rel_err: f64::INFINITY,
                numel: out.len(),
            });
            continue;
        }
        let cos = cosine(out, expected);
        let mut max_abs = 0.0f64;
        let mut denom = 0.0f64;
        for (&g, &e) in out.iter().zip(expected) {
            max_abs = max_abs.max((g as f64 - e as f64).abs());
            denom = denom.max((e as f64).abs());
        }
        let rel = if denom > 0.0 {
            max_abs / denom
        } else {
            max_abs
        };
        if cos < cosine_threshold || max_abs > max_abs_threshold {
            passed = false;
        }
        outputs.push(OutputParity {
            index: i,
            cosine: cos,
            max_abs_err: max_abs,
            rel_err: rel,
            numel: out.len(),
        });
    }
    Ok(ParityReport {
        passed,
        outputs,
        cosine_threshold,
        max_abs_threshold,
    })
}
