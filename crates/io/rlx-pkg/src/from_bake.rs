// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Build an `.rlxp` from bake-style graph + weight table (no bake crate dep).

use crate::write::{PackedWeight, WriteOptions, write_package};
use anyhow::Result;
use rlx_ir::Graph;
use std::path::Path;

/// Minimal weight row matching bake’s table (avoids depending on `rlx-bake`).
#[derive(Debug, Clone)]
pub struct BakeWeight {
    pub name: String,
    pub shape: Vec<usize>,
    /// Bake encoding string: `f32`, `gguf_tq2_0`, `gguf_q8_0`, …
    pub encoding: String,
    pub data: Vec<u8>,
}

impl BakeWeight {
    fn layout_for_encoding(encoding: &str) -> &'static str {
        if encoding.starts_with("gguf_") {
            "bt_nk"
        } else {
            "row_major"
        }
    }
}

/// Write a package from a baked graph and weight table.
pub fn package_from_bake(
    out: impl AsRef<Path>,
    graph: &Graph,
    weights: &[BakeWeight],
    mut opts: WriteOptions,
) -> Result<()> {
    if opts.name == "model" {
        if !graph.name.is_empty() {
            opts.name = graph.name.clone();
        }
    }
    if opts.producer.is_none() {
        opts.producer = Some("rlx-bake".into());
    }
    let packed: Vec<PackedWeight> = weights
        .iter()
        .map(|w| PackedWeight {
            name: w.name.clone(),
            shape: w.shape.clone(),
            scheme: w.encoding.clone(),
            layout: BakeWeight::layout_for_encoding(&w.encoding).into(),
            data: w.data.clone(),
            rank: None,
            tier: crate::tier::StorageTier::Hot,
        })
        .collect();
    write_package(out, graph, &packed, &opts)
}
