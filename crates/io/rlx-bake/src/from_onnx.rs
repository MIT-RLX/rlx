// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Optional ONNX → `.rlxp` (`onnx` cargo feature).
//!
//! Lowers `ModelProto` → MIR, specializes initializer params into named
//! Constants, and packs them like bake. Graph embedding is **optional**
//! (`OnnxImportOptions::include_graph`, default true) — use `--no-graph` on
//! the CLI for a weight-only archive.
//!
//! ```bash
//! cargo run -p rlx-bake --features onnx -- import-onnx model.onnx -o model.rlxp
//! ```

use crate::specialize_named;
use anyhow::{Context, Result, bail};
use rlx_ir::{Dim, Op, hir_to_graph};
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};
use rlx_pkg::{
    BakeWeight, ContainerKind, WriteOptions, package_from_bake,
};
use std::path::Path;

/// Options for [`onnx_to_rlxp`].
#[derive(Debug, Clone)]
pub struct OnnxImportOptions {
    pub container: ContainerKind,
    /// Embed the lowered MIR graph (default **true**). Optional — set false for weights only.
    pub include_graph: bool,
    pub compress_sidecars: bool,
    pub import: ImportOptions,
}

impl Default for OnnxImportOptions {
    fn default() -> Self {
        Self {
            container: ContainerKind::Flat,
            include_graph: true,
            compress_sidecars: true,
            import: ImportOptions::default(),
        }
    }
}

/// Convert `model.onnx` → `.rlxp` (optional executable graph + initializer weights).
pub fn onnx_to_rlxp(
    onnx_path: impl AsRef<Path>,
    out: impl AsRef<Path>,
    opts: &OnnxImportOptions,
) -> Result<()> {
    let onnx_path = onnx_path.as_ref();
    let (hir, params, report, manifest) = build_hir_from_onnx_file(onnx_path, opts.import.clone())
        .with_context(|| format!("ONNX import {}", onnx_path.display()))?;
    if report.skipped > 0 {
        eprintln!(
            "rlx-bake: ONNX import skipped {} node(s); unsupported: {:?}",
            report.skipped, report.unsupported
        );
    }

    let graph = hir_to_graph(hir).map_err(|e| anyhow::anyhow!("HIR→MIR: {e}"))?;
    let graph = specialize_named(&graph, &params);

    let weights = harvest_f32_constants(&graph);
    if weights.is_empty() {
        bail!("ONNX model produced no packable Constant weights");
    }

    let name = onnx_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("onnx")
        .to_string();

    let mut wopts = WriteOptions {
        name,
        producer: Some("rlx-bake/onnx_import".into()),
        container: opts.container,
        compress_sidecars: opts.compress_sidecars,
        include_graph: opts.include_graph,
        strip_graph_weights: true,
        write_checksums: true,
        ..WriteOptions::default()
    };
    if !wopts.features.iter().any(|f| f == "onnx_import") {
        wopts.features.push("onnx_import".into());
    }
    if opts.include_graph && !wopts.features.iter().any(|f| f == "executable_graph") {
        wopts.features.push("executable_graph".into());
    }

    let io = serde_json::json!({
        "inputs": manifest.inputs.iter().map(|i| serde_json::json!({
            "name": i.name,
            "dtype": i.meta.dtype,
            "shape": i.meta.shape,
        })).collect::<Vec<_>>(),
        "outputs": manifest.outputs.iter().map(|o| serde_json::json!({
            "name": o.name,
            "dtype": o.meta.dtype,
            "shape": o.meta.shape,
        })).collect::<Vec<_>>(),
    });
    wopts.sidecars.push((
        "io".into(),
        "application/json".into(),
        serde_json::to_vec_pretty(&io)?,
    ));

    package_from_bake(out, &graph, &weights, wopts)
}

fn harvest_f32_constants(graph: &rlx_ir::Graph) -> Vec<BakeWeight> {
    let mut out = Vec::new();
    for n in graph.nodes() {
        let Some(name) = n.name.as_deref() else {
            continue;
        };
        let Op::Constant { data } = &n.op else {
            continue;
        };
        if data.is_empty() {
            continue;
        }
        let shape: Vec<usize> = n
            .shape
            .dims()
            .iter()
            .map(|d| match d {
                Dim::Static(x) => *x,
                Dim::Dynamic(_) => 0,
            })
            .collect();
        out.push(BakeWeight {
            name: name.into(),
            shape,
            encoding: "f32".into(),
            data: data.clone(),
        });
    }
    out
}
