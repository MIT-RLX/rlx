// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Load a MIR `Graph`, HIR module JSON, or torch-import bundle directory.

use anyhow::{Context, Result, bail};
use rlx_ir::{Graph, HirModule};
use std::path::{Path, PathBuf};

/// Graph loaded from disk, plus an optional default weights path (bundle).
#[derive(Debug)]
pub struct LoadedGraph {
    pub graph: Graph,
    /// Bundle dirs expose `weights.safetensors` here when present.
    pub default_weights: Option<PathBuf>,
    pub source: String,
}

/// Load a MIR Graph JSON, HIR `model.hir.json`, or a bundle directory.
pub fn load_graph(path: &Path) -> Result<LoadedGraph> {
    if path.is_dir() {
        return load_bundle(path);
    }
    if !path.is_file() {
        bail!("{} is not a file or directory", path.display());
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let prefer_hir = file_name == "model.hir.json"
        || file_name.ends_with(".hir.json")
        || text.contains("\"fusion_policy\"");

    if prefer_hir {
        let graph = hir_json_to_graph(&text)
            .with_context(|| format!("parsing HIR from {}", path.display()))?;
        return Ok(LoadedGraph {
            graph,
            default_weights: None,
            source: format!("hir:{}", path.display()),
        });
    }

    match serde_json::from_str::<Graph>(&text) {
        Ok(graph) => Ok(LoadedGraph {
            graph,
            default_weights: None,
            source: format!("mir:{}", path.display()),
        }),
        Err(mir_err) => {
            let graph = hir_json_to_graph(&text).with_context(|| {
                format!(
                    "not MIR Graph ({mir_err}) and failed as HIR from {}",
                    path.display()
                )
            })?;
            Ok(LoadedGraph {
                graph,
                default_weights: None,
                source: format!("hir:{}", path.display()),
            })
        }
    }
}

fn load_bundle(dir: &Path) -> Result<LoadedGraph> {
    let hir_path = dir.join("model.hir.json");
    if !hir_path.is_file() {
        bail!("bundle dir {} missing model.hir.json", dir.display());
    }
    let text = std::fs::read_to_string(&hir_path)
        .with_context(|| format!("reading {}", hir_path.display()))?;
    let graph = hir_json_to_graph(&text)
        .with_context(|| format!("parsing HIR from {}", hir_path.display()))?;
    let weights = dir.join("weights.safetensors");
    let default_weights = if weights.is_file() {
        Some(weights)
    } else {
        None
    };
    Ok(LoadedGraph {
        graph,
        default_weights,
        source: format!("bundle:{}", dir.display()),
    })
}

fn hir_json_to_graph(text: &str) -> Result<Graph> {
    let hir: HirModule = serde_json::from_str(text).context("deserializing HirModule")?;
    let mir = hir.lower_to_mir().context("lowering HIR → MIR")?;
    Ok(mir.into_graph())
}
