// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Build RLX graphs from packed MLX Linear triples.
//!
//! Param names match the mlx-community / RLXP weight catalog:
//! `{base}.weight`, `{base}.scales`, `{base}.biases` — so
//! `Package` tensors bind 1:1 via [`param_bindings_for`].

use anyhow::{Result, bail};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::load::MlxPackedLinear;

/// One packed Linear ready to bind as `Op::DequantMatMul` params.
#[derive(Debug, Clone)]
pub struct PackedLinearBinding {
    /// Logical base name (e.g. `model.layers.0.mlp.up_proj`).
    pub name: String,
    pub packed: MlxPackedLinear,
}

/// Drain every quantized Linear into packed bindings (affine + mxfp).
pub fn collect_packed_linears(
    weights: &mut crate::load::MlxWeights,
) -> Result<Vec<PackedLinearBinding>> {
    let keys: Vec<String> = weights
        .logical_keys()
        .into_iter()
        .filter(|k| weights.is_quantized_layer(k))
        .collect();
    let mut out = Vec::new();
    for k in keys {
        if let Some(packed) = weights.take_packed_linear(&k)? {
            let name = k.trim_end_matches(".weight").to_string();
            out.push(PackedLinearBinding { name, packed });
        }
    }
    Ok(out)
}

fn input_name(base: &str) -> String {
    let safe: String = base
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!("{safe}_x")
}

/// Parallel graph: one `x_{i}` → `DequantMatMul` → `y_{i}` per packed Linear.
///
/// Params use catalog names `{base}.weight/.scales/.biases`.
pub fn build_parallel_dequant_graph(
    graph_name: &str,
    linears: &[PackedLinearBinding],
    batch: usize,
) -> Result<Graph> {
    if linears.is_empty() {
        bail!("build_parallel_dequant_graph: no packed linears");
    }
    let mut g = Graph::new(graph_name);
    let mut outs = Vec::with_capacity(linears.len());
    for b in linears {
        let (n, k) = (b.packed.out_shape[0], b.packed.out_shape[1]);
        let n_groups = b.packed.n_groups().max(1);
        let x = g.input(input_name(&b.name), Shape::new(&[batch, k], DType::F32));
        let w = g.param(
            format!("{}.weight", b.name),
            Shape::new(&[b.packed.w_q.len()], DType::U8),
        );
        let s = g.param(
            format!("{}.scales", b.name),
            Shape::new(&[n, n_groups], b.packed.scale_dtype()),
        );
        let z = g.param(
            format!("{}.biases", b.name),
            Shape::new(&[n, n_groups], b.packed.bias_dtype()),
        );
        let y = g.add_node(
            Op::DequantMatMul {
                scheme: b.packed.scheme,
            },
            vec![x, w, s, z],
            Shape::new(&[batch, n], DType::F32),
        );
        outs.push(y);
    }
    g.set_outputs(outs);
    Ok(g)
}

/// Chain linears when consecutive `out_features` → next `in_features` match.
pub fn build_mlp_chain_graph(
    graph_name: &str,
    linears: &[PackedLinearBinding],
    batch: usize,
) -> Result<Graph> {
    if linears.is_empty() {
        bail!("build_mlp_chain_graph: no packed linears");
    }
    for w in linears.windows(2) {
        let out0 = w[0].packed.out_shape[0];
        let in1 = w[1].packed.out_shape[1];
        if out0 != in1 {
            bail!(
                "mlp chain dim mismatch: {} out={out0} vs {} in={in1}",
                w[0].name,
                w[1].name
            );
        }
    }
    let mut g = Graph::new(graph_name);
    let k0 = linears[0].packed.out_shape[1];
    let mut cur: NodeId = g.input("x", Shape::new(&[batch, k0], DType::F32));
    for b in linears {
        let (n, _k) = (b.packed.out_shape[0], b.packed.out_shape[1]);
        let n_groups = b.packed.n_groups().max(1);
        let w = g.param(
            format!("{}.weight", b.name),
            Shape::new(&[b.packed.w_q.len()], DType::U8),
        );
        let s = g.param(
            format!("{}.scales", b.name),
            Shape::new(&[n, n_groups], b.packed.scale_dtype()),
        );
        let z = g.param(
            format!("{}.biases", b.name),
            Shape::new(&[n, n_groups], b.packed.bias_dtype()),
        );
        cur = g.add_node(
            Op::DequantMatMul {
                scheme: b.packed.scheme,
            },
            vec![cur, w, s, z],
            Shape::new(&[batch, n], DType::F32),
        );
    }
    g.set_outputs(vec![cur]);
    Ok(g)
}

/// Bind one packed Linear (catalog param names).
pub fn param_bindings_for(binding: &PackedLinearBinding) -> Vec<(String, Vec<u8>, DType)> {
    let zp = if binding.packed.biases.is_empty() {
        vec![0u8; 4]
    } else {
        binding.packed.biases.clone()
    };
    vec![
        (
            format!("{}.weight", binding.name),
            binding.packed.w_q.clone(),
            DType::U8,
        ),
        (
            format!("{}.scales", binding.name),
            binding.packed.scales.clone(),
            binding.packed.scale_dtype(),
        ),
        (
            format!("{}.biases", binding.name),
            zp,
            binding.packed.bias_dtype(),
        ),
    ]
}
