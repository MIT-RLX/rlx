// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Import NVIDIA NeMo (`.nemo`) into an `.rlxp` package.

use crate::tier::StorageTier;
use crate::write::{ContainerKind, PackedWeight, WriteOptions, write_package};
use anyhow::{Context, Result};
use rlx_ir::{DType, Graph, Shape};
use rlx_nemo::{NemoModel, build_nemo_probe_graph};
use std::path::Path;

/// Options for [`nemo_to_rlxp`].
#[derive(Debug, Clone)]
pub struct NemoImportOptions {
    pub container: ContainerKind,
    pub include_graph: bool,
    pub compress_sidecars: bool,
    pub auto_tier: bool,
}

impl Default for NemoImportOptions {
    fn default() -> Self {
        Self {
            container: ContainerKind::Flat,
            include_graph: false,
            compress_sidecars: true,
            auto_tier: true,
        }
    }
}

/// Convert `model.nemo` → `.rlxp` (f32 weights + config / tokenizer sidecars).
pub fn nemo_to_rlxp(
    nemo_path: impl AsRef<Path>,
    out: impl AsRef<Path>,
    opts: &NemoImportOptions,
) -> Result<()> {
    let nemo_path = nemo_path.as_ref();
    let m =
        NemoModel::open(nemo_path).with_context(|| format!("open nemo {}", nemo_path.display()))?;

    let mut sidecars = Vec::new();
    let cfg_yaml =
        serde_yaml::to_string(m.config().root()).context("serialize model_config.yaml")?;
    sidecars.push((
        "model_config.yaml".into(),
        "application/yaml".into(),
        cfg_yaml.into_bytes(),
    ));
    for tok in m.tokenizers() {
        sidecars.push((
            tok.name.clone(),
            "application/octet-stream".into(),
            tok.bytes.clone(),
        ));
    }

    let mut weights = Vec::new();
    for name in m.names() {
        let t = m.tensor(&name)?;
        let mut bytes = Vec::with_capacity(t.data.len() * 4);
        for x in &t.data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        weights.push(PackedWeight {
            name,
            shape: t.shape,
            scheme: "f32".into(),
            layout: "row_major".into(),
            data: bytes,
            rank: None,
            tier: StorageTier::Hot,
        });
    }

    let mut wopts = WriteOptions {
        name: nemo_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("nemo")
            .into(),
        producer: Some("rlx-pkg/nemo_import".into()),
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
        if let Some(g) = build_nemo_probe_graph(&m, &wopts.name)? {
            g
        } else {
            let mut g = Graph::new(&wopts.name);
            let s = Shape::new(&[1], DType::F32);
            let x = g.input("x", s);
            g.set_outputs(vec![x]);
            g
        }
    } else {
        Graph::new(&wopts.name)
    };

    write_package(out.as_ref(), &graph, &weights, &wopts)
        .with_context(|| format!("write {}", out.as_ref().display()))?;
    Ok(())
}
