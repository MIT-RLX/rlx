// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Export baked `RlxFile` artifacts to `.rlxp` packages.

use crate::format::RlxFile;
use anyhow::Result;
use rlx_pkg::{BakeWeight, ContainerKind, WriteOptions, infer_container, package_from_bake};
use std::path::Path;

/// Write `file` as an `.rlxp` package (flat by default for `.rlxp`).
///
/// Weight bytes are stored once in the data region; graph Constants are
/// stripped on disk and filled on load (no duplex copy).
pub fn write_rlxp(
    path: impl AsRef<Path>,
    file: &RlxFile,
    container: Option<ContainerKind>,
) -> Result<()> {
    let path = path.as_ref();
    let container = infer_container(path, container)?;
    // Prefer table bytes; do not materialize into graph before pack (avoids duplex).
    let mut file = file.clone();
    if file.weights.iter().all(|w| w.data.is_empty()) && file.needs_materialize() {
        // Table empty but graph has bytes (runtime memory mode) — pull into table path
        // by materializing then we'll strip again on write.
        file.materialize_weights()?;
    }
    let weights: Vec<BakeWeight> = file
        .weights
        .iter()
        .filter(|w| !w.data.is_empty())
        .map(|w| BakeWeight {
            name: w.name.clone(),
            shape: w.shape.clone(),
            encoding: w.encoding.clone(),
            data: w.data.clone(),
        })
        .collect();
    // If weights table was empty after compact invert, harvest from graph Constants.
    let weights = if weights.is_empty() {
        harvest_constant_weights(&file)
    } else {
        weights
    };
    let opts = WriteOptions {
        name: file.meta.name.clone(),
        producer: Some("rlx-bake".into()),
        container,
        strip_graph_weights: true,
        ..WriteOptions::default()
    };
    package_from_bake(path, &file.graph, &weights, opts)
}

fn harvest_constant_weights(file: &RlxFile) -> Vec<BakeWeight> {
    use rlx_ir::Op;
    let mut out = Vec::new();
    for n in file.graph.nodes() {
        let Some(name) = n.name.as_deref() else {
            continue;
        };
        if let Op::Constant { data } = &n.op {
            if data.is_empty() {
                continue;
            }
            let shape: Vec<usize> = n
                .shape
                .dims()
                .iter()
                .map(|d| match d {
                    rlx_ir::Dim::Static(x) => *x,
                    rlx_ir::Dim::Dynamic(_) => 0,
                })
                .collect();
            out.push(BakeWeight {
                name: name.to_string(),
                shape,
                encoding: "f32".into(),
                data: data.clone(),
            });
        }
    }
    out
}

/// Convert a plaintext `*.rlx` (`RLXBAKE1`) into `.rlxp`.
pub fn convert_rlx_to_rlxp(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    container: Option<ContainerKind>,
) -> Result<()> {
    let file = crate::format::read_rlx(input)?;
    write_rlxp(output, &file, container)
}
