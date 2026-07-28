// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Load `.rlxp` packages into a [`Session`](crate::Session).
//!
//! ```ignore
//! let compiled = rlx_runtime::pkg::compile_rlxp(&session, "model.rlxp")?;
//! ```
//!
//! Also: [`open_rlxp`], [`load_rlxp_graph`], placement helpers, and
//! `rlxp://path#tensor` URI resolution in the weight cache.

use crate::CompileOptions;
use crate::compiled::CompiledGraph;
use crate::session::Session;
use anyhow::{Context, Result};
use rlx_ir::Graph;
use rlx_pkg::{Package, Placement};
use std::path::Path;

/// Open a package and return its MIR graph (hot-tier materialize).
///
/// Fails if the pack is weight-only (`encoding = "none"`).
pub fn load_rlxp_graph(path: impl AsRef<Path>) -> Result<Graph> {
    let pack = Package::open(path.as_ref())
        .with_context(|| format!("open package {}", path.as_ref().display()))?;
    pack.graph()
}

/// Open a flat `.rlxp`, ZIP, or package directory.
pub fn open_rlxp(path: impl AsRef<Path>) -> Result<Package> {
    Package::open(path.as_ref())
        .with_context(|| format!("open package {}", path.as_ref().display()))
}

/// Compile the embedded MIR graph of an `.rlxp` package.
///
/// Requires `include_graph` / `encoding = bincode_graph_v1` at write time.
pub fn compile_rlxp(session: &Session, path: impl AsRef<Path>) -> Result<CompiledGraph> {
    let graph = load_rlxp_graph(path)?;
    Ok(session.compile(graph))
}

/// Compile an `.rlxp` and bind every catalog tensor onto matching Params
/// (and leave Constants to [`Package::graph`] materialize).
///
/// Used for MLX `--keep-packed` packs where DequantMatMul weights are Params
/// named `{base}.weight/.scales/.biases`.
pub fn compile_rlxp_bind_params(
    session: &Session,
    path: impl AsRef<Path>,
) -> Result<CompiledGraph> {
    let pack = open_rlxp(path.as_ref())?;
    let graph = pack
        .graph_with(rlx_pkg::MaterializeMode::HotOnly)
        .with_context(|| format!("graph {}", path.as_ref().display()))?;
    let mut compiled = session.compile(graph);
    for (name, bytes, dtype) in pack.typed_weight_bindings()? {
        compiled.set_param_typed(&name, &bytes, dtype);
    }
    Ok(compiled)
}

/// Compile with explicit [`CompileOptions`].
pub fn compile_rlxp_with(
    session: &Session,
    path: impl AsRef<Path>,
    options: &CompileOptions,
) -> Result<CompiledGraph> {
    let graph = load_rlxp_graph(path)?;
    Ok(session.compile_with(graph, options))
}

/// Placement metadata from a package, if present.
pub fn load_rlxp_placement(path: impl AsRef<Path>) -> Result<Option<Placement>> {
    let pack = open_rlxp(path)?;
    Ok(pack.placement().cloned())
}

/// Tensor names this rank should load for a placement (metadata-only packs).
///
/// When `placement` has no entry for a tensor, it is treated as replicated
/// (all ranks load it). When an entry exists, only listed ranks load it.
pub fn tensors_for_rank(placement: &Placement, rank: u32, all_names: &[&str]) -> Vec<String> {
    all_names
        .iter()
        .filter_map(|name| match placement.ranks_for(name) {
            Some(ranks) if ranks.contains(&rank) => Some((*name).to_string()),
            Some(_) => None,
            None => Some((*name).to_string()),
        })
        .collect()
}

/// Prefer a physical `dist/rank-{rank}/` tree when present; otherwise filter
/// the global weight index by [`tensors_for_rank`].
pub fn weight_names_for_rank(pack: &Package, rank: u32) -> Vec<String> {
    if pack.rank_root(rank).is_some() {
        return pack
            .tensors_for_rank(rank)
            .into_iter()
            .filter(|e| e.rank == Some(rank))
            .map(|e| e.name.clone())
            .collect();
    }
    let Some(idx) = pack.weights_index() else {
        return Vec::new();
    };
    let names: Vec<&str> = idx.names().collect();
    match pack.placement() {
        Some(pl) => tensors_for_rank(pl, rank, &names),
        None => names.into_iter().map(str::to_string).collect(),
    }
}
