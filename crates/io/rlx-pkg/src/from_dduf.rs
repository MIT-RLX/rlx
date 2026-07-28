// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Import HuggingFace DDUF (`.dduf`) into an `.rlxp` package.

use crate::tier::StorageTier;
use crate::write::{ContainerKind, PackedWeight, WriteOptions, write_package};
use anyhow::{Context, Result};
use rlx_dduf::visit_f32_tensors;
use rlx_ir::{DType, Graph, Shape};
use std::path::Path;

/// Options for [`dduf_to_rlxp`].
#[derive(Debug, Clone)]
pub struct DdufImportOptions {
    pub container: ContainerKind,
    pub include_graph: bool,
    pub compress_sidecars: bool,
    pub auto_tier: bool,
}

impl Default for DdufImportOptions {
    fn default() -> Self {
        Self {
            container: ContainerKind::Flat,
            include_graph: false,
            compress_sidecars: true,
            auto_tier: true,
        }
    }
}

/// Convert `model.dduf` → `.rlxp` (weights + optional stub graph).
///
/// Streams one safetensors ZIP member at a time so peak memory is roughly one
/// member + the growing `PackedWeight` list (not the full ZIP decoded at once).
pub fn dduf_to_rlxp(
    dduf_path: impl AsRef<Path>,
    out: impl AsRef<Path>,
    opts: &DdufImportOptions,
) -> Result<()> {
    let dduf_path = dduf_path.as_ref();
    let mut weights = Vec::new();
    let meta = visit_f32_tensors(dduf_path, |t| {
        let mut bytes = Vec::with_capacity(t.data.len() * 4);
        for x in &t.data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        weights.push(PackedWeight {
            name: t.name,
            shape: t.shape,
            scheme: "f32".into(),
            layout: "row_major".into(),
            data: bytes,
            rank: None,
            tier: StorageTier::Hot,
        });
        Ok(())
    })
    .with_context(|| format!("open dduf {}", dduf_path.display()))?;

    let mut sidecars = Vec::new();
    if let Some(idx) = &meta.model_index {
        sidecars.push((
            "model_index.json".into(),
            "application/json".into(),
            serde_json::to_vec_pretty(idx)?,
        ));
    }
    for (comp, cfg) in &meta.configs {
        sidecars.push((
            format!("{comp}/config.json"),
            "application/json".into(),
            serde_json::to_vec_pretty(cfg)?,
        ));
    }

    let mut wopts = WriteOptions {
        name: dduf_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("dduf")
            .into(),
        producer: Some("rlx-pkg/dduf_import".into()),
        container: opts.container,
        compress_sidecars: opts.compress_sidecars,
        include_graph: opts.include_graph,
        write_checksums: true,
        ..WriteOptions::default()
    };
    wopts.sidecars.extend(sidecars);

    if opts.auto_tier {
        crate::auto_tier::apply_auto_tier(&mut weights, &Default::default());
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

    write_package(out.as_ref(), &graph, &weights, &wopts)
        .with_context(|| format!("write {}", out.as_ref().display()))?;
    Ok(())
}
