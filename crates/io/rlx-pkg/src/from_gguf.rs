// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Import a GGUF weight file into an `.rlxp` package.
//!
//! Defaults to **weight-only** (`include_graph: false`). Pass
//! `include_graph: true` for a stub graph; for a real executable graph from
//! ONNX, use `rlx-bake --features onnx -- import-onnx`.

use crate::tier::StorageTier;
use crate::write::{ContainerKind, PackedWeight, WriteOptions, write_package};
use anyhow::{Context, Result};
use rlx_gguf::{GgmlType, GgufFile, MetaValue};
use rlx_ir::{DType, Graph, Shape};
use std::path::Path;

/// Options for [`gguf_to_rlxp`].
#[derive(Debug, Clone)]
pub struct GgufImportOptions {
    pub container: ContainerKind,
    pub include_graph: bool,
    pub compress_sidecars: bool,
    pub auto_tier: bool,
}

impl Default for GgufImportOptions {
    fn default() -> Self {
        Self {
            container: ContainerKind::Flat,
            include_graph: false,
            compress_sidecars: true,
            auto_tier: true,
        }
    }
}

fn scheme_for(dtype: GgmlType) -> String {
    match dtype {
        GgmlType::F32 => "f32".into(),
        GgmlType::F16 => "f16".into(),
        GgmlType::BF16 => "bf16".into(),
        GgmlType::Q4K => "gguf_q4_k".into(),
        GgmlType::Q5K => "gguf_q5_k".into(),
        GgmlType::Q6K => "gguf_q6_k".into(),
        GgmlType::Q8K => "gguf_q8_k".into(),
        GgmlType::Q8_0 => "gguf_q8_0".into(),
        GgmlType::Q4_0 => "gguf_q4_0".into(),
        GgmlType::Q4_1 => "gguf_q4_1".into(),
        GgmlType::Q2K => "gguf_q2_k".into(),
        GgmlType::Q3K => "gguf_q3_k".into(),
        GgmlType::TQ1_0 => "gguf_tq1_0".into(),
        GgmlType::TQ2_0 => "gguf_tq2_0".into(),
        other => format!("gguf_{}", format!("{other:?}").to_ascii_lowercase()),
    }
}

/// Convert `model.gguf` → `.rlxp` (weights + optional stub graph).
pub fn gguf_to_rlxp(
    gguf_path: impl AsRef<Path>,
    out: impl AsRef<Path>,
    opts: &GgufImportOptions,
) -> Result<()> {
    let gguf_path = gguf_path.as_ref();
    let f = GgufFile::from_path_mmap(gguf_path)
        .with_context(|| format!("open gguf {}", gguf_path.display()))?;

    let mut weights = Vec::new();
    let names: Vec<String> = f.keys().map(|s| s.to_string()).collect();
    for name in names {
        let t = f.get(&name).with_context(|| format!("tensor {name}"))?;
        let bytes = f.tensor_bytes(t)?.to_vec();
        weights.push(PackedWeight {
            name: name.clone(),
            shape: t.shape.clone(),
            scheme: scheme_for(t.dtype),
            layout: "bt_nk".into(),
            data: bytes,
            rank: None,
            tier: StorageTier::Hot,
        });
    }

    let mut wopts = WriteOptions {
        name: gguf_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("gguf")
            .into(),
        producer: Some("rlx-pkg/gguf_import".into()),
        container: opts.container,
        compress_sidecars: opts.compress_sidecars,
        include_graph: opts.include_graph,
        write_checksums: true,
        ..WriteOptions::default()
    };

    if opts.auto_tier {
        crate::auto_tier::apply_auto_tier(&mut weights, &Default::default());
    }

    if let Some(MetaValue::String(arch)) = f.metadata.get("general.architecture") {
        wopts.sidecars.push((
            "gguf_architecture".into(),
            "text/plain".into(),
            arch.as_bytes().to_vec(),
        ));
    }

    let graph = if opts.include_graph {
        let mut g = Graph::new(&wopts.name);
        let s = Shape::new(&[1], DType::F32);
        let x = g.input("x", s);
        g.set_outputs(vec![x]);
        g
    } else {
        Graph::new(&wopts.name)
    };

    write_package(out, &graph, &weights, &wopts)
}
