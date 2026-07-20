// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Offline weight bake: merge graph + weights, optimize, emit `*.rlx`.
//!
//! The artifact is a **single file** containing the optimized MIR graph and an
//! explicit weight table (CoreML / ONNX-initializer style). Weight-aware passes
//! can skip zero matmuls, pack ternary weights (TQ2_0), and optionally quantize
//! (Q8_0).
//!
//! Choose a named [`BakeProfile`] (`merge` / `fold` / `exact` / `size`) or set
//! individual [`BakeOptions`] fields. Use [`MemoryMode`] to avoid storing weight
//! bytes twice (`compact` / `runtime`). Full-file encryption is behind the
//! `encrypt` feature (`write_rlx_encrypted` / `read_rlx_with_password`).

#[cfg(feature = "encrypt")]
pub mod crypto;
pub mod format;
pub mod load;
pub mod memory;
pub mod optimize;
pub mod profile;
pub mod weights;

#[cfg(feature = "encrypt")]
pub use crypto::{
    DEFAULT_M_KIB, DEFAULT_P_COST, DEFAULT_T_COST, RLX_ENC_MAGIC, RLX_ENC_VERSION, decrypt_bytes,
    encrypt_bytes, encrypt_bytes_with_params, is_encrypted,
};
#[cfg(feature = "encrypt")]
pub use format::{read_rlx_with_password, write_rlx_encrypted};
#[cfg(feature = "mmap")]
pub use format::read_rlx_mmap;
pub use format::{RlxFile, RlxIo, RlxMeta, RlxWeight, read_rlx, write_rlx};
pub use load::{LoadedGraph, load_graph};
pub use memory::{
    MemoryMode, MemoryStats, dedupe_identical_constants, ensure_runtime_ready,
};
pub use optimize::{OptimizeStats, WeightEncoding, WeightRewrite, is_ternary_f32};
pub use profile::BakeProfile;
pub use weights::load_safetensors_f32;

use optimize::optimize_weights;
use rlx_compile::{AlgebraicSimplify, ConstantFolding, DeadCodeElimination};
use rlx_fusion::pass::Pass;
use rlx_ir::{Dim, Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};

/// Toggles for the bake pipeline.
///
/// Prefer [`BakeOptions::from_profile`] / [`BakeProfile`] for common setups;
/// then flip individual fields if needed.
#[derive(Debug, Clone)]
pub struct BakeOptions {
    /// Named profile this was built from (informational; overrides win).
    pub profile: BakeProfile,
    pub constant_folding: bool,
    pub dce: bool,
    pub algebraic_simplify: bool,
    /// Replace `MatMul(x, 0)` with a zero constant.
    pub skip_zero: bool,
    /// Pack exact `{−1,0,+1}` MatMul weights as GGUF TQ2_0 + `DequantMatMul`.
    pub ternary: bool,
    /// Pack remaining F32 MatMul weights as GGUF Q8_0 + `DequantMatMul`.
    pub quant: bool,
    /// Unfold weight Constants into the artifact weight table.
    pub unfold: bool,
    /// Where weight bytes live after bake (disk / load RAM).
    pub memory: MemoryMode,
    /// Merge identical `Op::Constant` payloads into one node.
    pub dedupe_constants: bool,
    /// Keep folded-away source bindings in the weight table (larger file).
    pub keep_folded_bindings: bool,
}

impl Default for BakeOptions {
    fn default() -> Self {
        Self::from_profile(BakeProfile::Exact)
    }
}

impl BakeOptions {
    /// Build options for a named profile (see [`BakeProfile`]).
    pub fn from_profile(profile: BakeProfile) -> Self {
        match profile {
            BakeProfile::Merge => Self {
                profile,
                constant_folding: false,
                dce: false,
                algebraic_simplify: false,
                skip_zero: false,
                ternary: false,
                quant: false,
                unfold: true,
                memory: MemoryMode::Duplex,
                dedupe_constants: false,
                keep_folded_bindings: true,
            },
            BakeProfile::Fold => Self {
                profile,
                constant_folding: true,
                dce: true,
                algebraic_simplify: true,
                skip_zero: false,
                ternary: false,
                quant: false,
                unfold: true,
                memory: MemoryMode::Compact,
                dedupe_constants: true,
                keep_folded_bindings: false,
            },
            BakeProfile::Exact => Self {
                profile,
                constant_folding: true,
                dce: true,
                algebraic_simplify: true,
                skip_zero: true,
                ternary: true,
                quant: false,
                unfold: true,
                memory: MemoryMode::Compact,
                dedupe_constants: true,
                keep_folded_bindings: false,
            },
            BakeProfile::Size => Self {
                profile,
                constant_folding: true,
                dce: true,
                algebraic_simplify: true,
                skip_zero: true,
                ternary: true,
                quant: true,
                unfold: true,
                memory: MemoryMode::Compact,
                dedupe_constants: true,
                keep_folded_bindings: false,
            },
        }
    }
}

/// Summary of what bake did to the graph / weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BakeReport {
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub params_baked: usize,
    pub params_remaining: Vec<String>,
    pub constant_bytes: usize,
    pub weight_count: usize,
    pub weight_bytes: usize,
    pub optimize: OptimizeStats,
    pub memory: MemoryStats,
}

/// Specialize params, run weight-aware opts, fold, and build a merged `*.rlx` file.
pub fn bake(
    graph: &Graph,
    bindings: &HashMap<String, Vec<f32>>,
    opts: &BakeOptions,
) -> (RlxFile, BakeReport) {
    let nodes_before = graph.len();
    let param_names_before = param_names(graph);

    let out = specialize_named(graph, bindings);

    let (mut out, mut rewrites, opt_stats) = optimize_weights(
        out,
        opts.skip_zero,
        opts.ternary,
        opts.quant,
        opts.unfold,
    );

    // Fold / simplify passes often rebuild nodes without copying `.name`.
    // Snapshot named Constants so we can re-stamp after cleanup.
    let name_snap = snapshot_named_constant_bytes(&out);

    if opts.algebraic_simplify {
        out = AlgebraicSimplify.run(out);
    }
    if opts.dce {
        out = DeadCodeElimination.run(out);
    }
    if opts.constant_folding {
        out = ConstantFolding.run(out);
    }

    let mut mem_stats = MemoryStats {
        mode: opts.memory.as_str().to_string(),
        ..Default::default()
    };
    if opts.dedupe_constants {
        let (ng, n) = dedupe_identical_constants(&out);
        out = ng;
        mem_stats.constants_deduped = n;
    }
    restore_constant_names(&mut out, &name_snap);

    // Re-unfold after fold/DCE so the table matches surviving weight Constants.
    // Skip when we already recorded packed rewrites for those names.
    if opts.unfold {
        let again = optimize::unfold_weights(&out);
        let known: HashSet<String> = rewrites.iter().map(|r| r.name.clone()).collect();
        let known_bytes: HashSet<Vec<u8>> = rewrites.iter().map(|r| r.data.clone()).collect();
        for u in again {
            if known.contains(&u.name) || known_bytes.contains(&u.data) {
                continue;
            }
            rewrites.push(u);
        }
    }

    // Optionally keep every bound param in the weight table (even if folded away).
    // With keep_folded_bindings=false, still record bindings that remain as live
    // named Constants (e.g. elementwise Mul weights that unfold does not catalog).
    let present: HashSet<String> = rewrites.iter().map(|r| r.name.clone()).collect();
    let live_named: HashSet<String> = out
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Constant { .. } => n.name.clone(),
            _ => None,
        })
        .collect();
    for (name, values) in bindings {
        if present.contains(name) {
            continue;
        }
        if !param_names_before.iter().any(|p| p == name) {
            continue;
        }
        let still_live = live_named.contains(name);
        if !opts.keep_folded_bindings && !still_live {
            mem_stats.folded_bindings_dropped += 1;
            continue;
        }
        let shape = param_shape(graph, name).unwrap_or_else(|| vec![values.len()]);
        let data = if still_live {
            out.nodes()
                .iter()
                .find(|n| n.name.as_deref() == Some(name.as_str()))
                .and_then(|n| match &n.op {
                    Op::Constant { data } => Some(data.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| encode_f32_slice(values))
        } else {
            encode_f32_slice(values)
        };
        let encoding = infer_encoding(&out, name).unwrap_or(WeightEncoding::F32);
        rewrites.push(WeightRewrite {
            name: name.clone(),
            shape,
            encoding,
            data,
            note: if still_live {
                "baked from live constant".into()
            } else {
                "baked f32 (source binding)".into()
            },
        });
    }

    let remaining = param_names(&out);
    let params_baked = param_names_before
        .iter()
        .filter(|n| bindings.contains_key(n.as_str()))
        .count();

    let constant_bytes = out
        .nodes()
        .iter()
        .map(|n| match &n.op {
            Op::Constant { data } => data.len(),
            _ => 0,
        })
        .sum();

    let weights = dedupe_weights(rewrites);
    let weight_bytes: usize = weights.iter().map(|w| w.data.len()).sum();
    let weight_count = weights.len();

    let mut report = BakeReport {
        nodes_before,
        nodes_after: out.len(),
        params_baked,
        params_remaining: remaining,
        constant_bytes,
        weight_count,
        weight_bytes,
        optimize: OptimizeStats {
            skipped_zero_matmuls: opt_stats.skipped_zero_matmuls,
            ternary_packed: opt_stats.ternary_packed,
            quant_packed: opt_stats.quant_packed,
            weights_unfolded: weight_count,
        },
        memory: mem_stats,
    };

    let mut file = RlxFile::from_baked(out, &report, weights, &report.optimize);
    let layout = file.apply_memory_mode(opts.memory);
    report.memory.graph_bytes_stripped = layout.graph_bytes_stripped;
    report.memory.table_bytes_stripped = layout.table_bytes_stripped;
    report.constant_bytes = file.meta.constant_bytes;
    report.weight_bytes = file.meta.weight_bytes;
    report.nodes_after = file.graph.len();

    (file, report)
}

fn encode_f32_slice(values: &[f32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * 4);
    for &v in values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    data
}

/// Map constant payload → name (first wins) so fold passes can be name-restored.
fn snapshot_named_constant_bytes(graph: &Graph) -> HashMap<Vec<u8>, String> {
    let mut snap = HashMap::new();
    for n in graph.nodes() {
        let Some(name) = &n.name else {
            continue;
        };
        if let Op::Constant { data } = &n.op {
            snap.entry(data.clone()).or_insert_with(|| name.clone());
        }
    }
    snap
}

fn restore_constant_names(graph: &mut Graph, snap: &HashMap<Vec<u8>, String>) {
    for n in graph.nodes_mut() {
        if n.name.is_some() {
            continue;
        }
        let Op::Constant { data } = &n.op else {
            continue;
        };
        if let Some(name) = snap.get(data) {
            n.name = Some(name.clone());
        }
    }
}

fn encoding_rank(enc: WeightEncoding) -> u8 {
    match enc {
        WeightEncoding::GgufTQ2_0 => 3,
        WeightEncoding::GgufQ8_0 => 2,
        WeightEncoding::F32 => 1,
    }
}

fn infer_encoding(graph: &Graph, weight_name: &str) -> Option<WeightEncoding> {
    let wid = graph.nodes().iter().find_map(|n| {
        if n.name.as_deref() == Some(weight_name) {
            Some(n.id)
        } else {
            None
        }
    })?;
    for n in graph.nodes() {
        if !n.inputs.iter().any(|i| *i == wid) {
            continue;
        }
        if let Op::DequantMatMul { scheme } = &n.op {
            return Some(match scheme {
                rlx_ir::quant::QuantScheme::GgufTQ2_0 => WeightEncoding::GgufTQ2_0,
                rlx_ir::quant::QuantScheme::GgufQ8_0 => WeightEncoding::GgufQ8_0,
                _ => WeightEncoding::F32,
            });
        }
    }
    Some(WeightEncoding::F32)
}

/// One entry per name; prefer packed encodings over plain f32.
fn dedupe_weights(rewrites: Vec<WeightRewrite>) -> Vec<RlxWeight> {
    let mut best: HashMap<String, WeightRewrite> = HashMap::new();
    for r in rewrites {
        match best.get(&r.name) {
            Some(prev) if encoding_rank(prev.encoding) >= encoding_rank(r.encoding) => {}
            _ => {
                best.insert(r.name.clone(), r);
            }
        }
    }
    let mut names: Vec<String> = best.keys().cloned().collect();
    names.sort();
    names
        .into_iter()
        .map(|n| RlxWeight::from_rewrite(&best[&n]))
        .collect()
}

fn param_names(graph: &Graph) -> Vec<String> {
    let mut names: Vec<String> = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some(name.clone()),
            _ => None,
        })
        .collect();
    names.sort();
    names.dedup();
    names
}

fn param_shape(graph: &Graph, name: &str) -> Option<Vec<usize>> {
    for n in graph.nodes() {
        if let Op::Param { name: p } = &n.op {
            if p == name {
                return Some(
                    n.shape
                        .dims()
                        .iter()
                        .map(|d| match d {
                            Dim::Static(k) => *k,
                            Dim::Dynamic(_) => 0,
                        })
                        .collect(),
                );
            }
        }
    }
    None
}

/// Like `specialize_params`, but stamps the param name onto the Constant node.
fn specialize_named(graph: &Graph, bindings: &HashMap<String, Vec<f32>>) -> Graph {
    if bindings.is_empty() {
        return graph.clone();
    }
    let mut out = Graph::new(graph.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let new_id = match &node.op {
            Op::Param { name } => {
                if let Some(values) = bindings.get(name) {
                    let expected = node.shape.num_elements().unwrap_or(values.len());
                    assert_eq!(
                        values.len(),
                        expected,
                        "param '{name}' binding len {} != shape elements {expected}",
                        values.len()
                    );
                    let mut data = Vec::with_capacity(values.len() * 4);
                    for &v in values {
                        data.extend_from_slice(&v.to_le_bytes());
                    }
                    let id = out.add_node(Op::Constant { data }, vec![], node.shape.clone());
                    out.node_mut(id).name = Some(name.clone());
                    id
                } else {
                    out.add_node(node.op.clone(), vec![], node.shape.clone())
                }
            }
            _ => {
                let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                out.add_node(node.op.clone(), new_inputs, node.shape.clone())
            }
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(graph.outputs.iter().map(|o| id_map[o]).collect());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::BinaryOp;
    use rlx_ir::{DType, Shape};

    #[test]
    fn bake_includes_weights_table() {
        let s = Shape::new(&[2], DType::F32);
        let mut g = Graph::new("t");
        let x = g.input("x", s.clone());
        let w = g.param("w", s.clone());
        let y = g.binary(BinaryOp::Mul, x, w, s);
        g.set_outputs(vec![y]);

        let mut bindings = HashMap::new();
        bindings.insert("w".into(), vec![2.0, 3.0]);
        let (file, report) = bake(&g, &bindings, &BakeOptions::default());
        assert_eq!(report.params_remaining, Vec::<String>::new());
        assert!(report.params_baked >= 1);
        assert!(
            !file.weights.is_empty(),
            "merged *.rlx must include weights"
        );
        assert!(file.weights.iter().any(|w| w.name == "w"));
        assert!(!file
            .graph
            .nodes()
            .iter()
            .any(|n| matches!(&n.op, Op::Param { name } if name == "w")));
    }
}
