// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! ONNX front-end — reuse `rlx-onnx-import`'s opset-versioned ONNX→HIR importer
//! (~88 ops) rather than re-porting the ONNX operator list into the aten
//! registry. An `.onnx` file lands on the **same** runnable-bundle format the
//! PyTorch path produces, so downstream (`run`, parity, generated crate) is
//! shared.

#![cfg(feature = "onnx")]

use crate::{BundleIo, BundleMeta};
use anyhow::{Context, Result};
use rlx_ir::hir::HirModule;
use std::collections::HashMap;
use std::path::Path;

/// Import an `.onnx` file to (HIR, f32 params, manifest). `strict = false` so a
/// model that uses an op not yet lowered surfaces as a stub rather than aborting
/// the whole import.
pub fn import_onnx(
    path: &Path,
) -> Result<(
    HirModule,
    HashMap<String, Vec<f32>>,
    rlx_onnx_import::BundleManifest,
)> {
    let opts = rlx_onnx_import::ImportOptions {
        strict: false,
        ..Default::default()
    };
    let (hir, params, _report, manifest) = rlx_onnx_import::build_hir_from_onnx_file(path, opts)
        .with_context(|| format!("importing ONNX {}", path.display()))?;
    Ok((hir, params, manifest))
}

/// Write an f32 param map to a safetensors file (each tensor stored 1-D — the
/// graph carries the real shapes, and the runtime binds params by name).
fn write_safetensors_f32(path: &Path, params: &HashMap<String, Vec<f32>>) -> Result<()> {
    use safetensors::tensor::{Dtype, TensorView};
    let bufs: Vec<(String, Vec<u8>)> = params
        .iter()
        .map(|(k, d)| {
            (
                k.clone(),
                d.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>(),
            )
        })
        .collect();
    let mut views: Vec<(String, TensorView)> = Vec::with_capacity(bufs.len());
    for (k, b) in &bufs {
        views.push((
            k.clone(),
            TensorView::new(Dtype::F32, vec![b.len() / 4], b)?,
        ));
    }
    safetensors::serialize_to_file(views, None, path)?;
    Ok(())
}

fn dims_of(meta: &rlx_onnx_import::IoMeta) -> Vec<usize> {
    meta.meta
        .shape
        .iter()
        .map(|v| v.as_i64().unwrap_or(1).max(0) as usize)
        .collect()
}

/// Import `onnx_path` and write a runnable bundle (HIR + weights + meta) into
/// `bundle_dir`, mirroring the PyTorch path's bundle so `run_bundle` works.
pub fn emit_onnx_bundle(onnx_path: &Path, bundle_dir: &Path) -> Result<BundleMeta> {
    std::fs::create_dir_all(bundle_dir)?;
    let (hir, params, manifest) = import_onnx(onnx_path)?;

    let json = rlx_ir::hir_to_json(&hir).context("serializing HIR")?;
    std::fs::write(bundle_dir.join("model.hir.json"), json)?;
    write_safetensors_f32(&bundle_dir.join("weights.safetensors"), &params)?;

    let meta = BundleMeta {
        name: onnx_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("onnx_model")
            .to_string(),
        inputs: manifest
            .inputs
            .iter()
            .map(|io| BundleIo {
                name: io.name.clone(),
                shape: dims_of(io),
                dtype: io.meta.dtype.to_lowercase(),
            })
            .collect(),
        output_count: manifest.outputs.len(),
        zero_params: Vec::new(),
    };
    std::fs::write(
        bundle_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;
    Ok(meta)
}
