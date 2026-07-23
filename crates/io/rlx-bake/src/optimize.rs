// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Weight-aware graph rewrites: skip zeros, pack ternary / quants.

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, Graph, NodeId, Op, Shape};
use std::collections::HashMap;

/// How a weight tensor is stored in the `*.rlx` weight table / graph Constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightEncoding {
    F32,
    /// GGUF TQ2_0 ternary (256 elems / 66 bytes).
    GgufTQ2_0,
    /// GGUF Q8_0 (32 elems / 34 bytes).
    GgufQ8_0,
}

impl WeightEncoding {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::GgufTQ2_0 => "gguf_tq2_0",
            Self::GgufQ8_0 => "gguf_q8_0",
        }
    }
}

/// One weight rewrite applied during bake (feeds the artifact weight table).
#[derive(Debug, Clone)]
pub struct WeightRewrite {
    pub name: String,
    pub shape: Vec<usize>,
    pub encoding: WeightEncoding,
    pub data: Vec<u8>,
    pub note: String,
}

/// Counters for weight-aware optimize passes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OptimizeStats {
    pub skipped_zero_matmuls: usize,
    pub ternary_packed: usize,
    pub quant_packed: usize,
    pub weights_unfolded: usize,
}

fn decode_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn encode_f32(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn static_dims(shape: &Shape) -> Option<Vec<usize>> {
    shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Some(*n),
            _ => None,
        })
        .collect()
}

fn is_all_zero(v: &[f32]) -> bool {
    v.iter().all(|&x| x == 0.0)
}

/// Exact ternary values `{−1, 0, +1}` (BitNet / TQ style).
pub fn is_ternary_f32(v: &[f32]) -> bool {
    v.iter().all(|&x| x == -1.0 || x == 0.0 || x == 1.0)
}

fn constant_f32(graph: &Graph, id: NodeId) -> Option<Vec<f32>> {
    match &graph.node(id).op {
        Op::Constant { data } if graph.node(id).shape.dtype() == DType::F32 => {
            Some(decode_f32(data))
        }
        _ => None,
    }
}

fn zeros_constant(graph: &mut Graph, shape: &Shape) -> NodeId {
    let n = shape.num_elements().unwrap_or(1);
    let zeros = vec![0.0f32; n];
    graph.add_node(
        Op::Constant {
            data: encode_f32(&zeros),
        },
        vec![],
        shape.clone(),
    )
}

fn copy_node(graph: &mut Graph, node: &rlx_ir::Node, inputs: Vec<NodeId>) -> NodeId {
    let id = graph.add_node(node.op.clone(), inputs, node.shape.clone());
    if let Some(name) = &node.name {
        graph.node_mut(id).name = Some(name.clone());
    }
    id
}

/// Replace `MatMul(x, 0)` with a zero constant of the matmul output shape.
pub fn skip_zero_matmul(graph: &Graph) -> (Graph, usize) {
    let mut out = Graph::new(graph.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut skipped = 0usize;

    for node in graph.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = if matches!(node.op, Op::MatMul) && new_inputs.len() == 2 {
            if let Some(w) = constant_f32(&out, new_inputs[1]) {
                if is_all_zero(&w) {
                    skipped += 1;
                    zeros_constant(&mut out, &node.shape)
                } else {
                    copy_node(&mut out, node, new_inputs)
                }
            } else {
                copy_node(&mut out, node, new_inputs)
            }
        } else {
            copy_node(&mut out, node, new_inputs)
        };
        id_map.insert(node.id, new_id);
    }

    let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|o| id_map[o]).collect();
    out.set_outputs(new_outputs);
    (out, skipped)
}

fn transpose_kn(vals: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; k * n];
    for i in 0..k {
        for j in 0..n {
            out[j * k + i] = vals[i * n + j];
        }
    }
    out
}

fn pack_matmul_weight(
    graph: &Graph,
    scheme: QuantScheme,
    encoding: WeightEncoding,
    quantize: fn(&[f32]) -> anyhow::Result<Vec<u8>>,
) -> (Graph, Vec<WeightRewrite>, usize) {
    let mut out = Graph::new(graph.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut rewrites = Vec::new();
    let mut packed = 0usize;

    for node in graph.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = if matches!(node.op, Op::MatMul) && new_inputs.len() == 2 {
            let w_id = new_inputs[1];
            let w_name = out.node(w_id).name.clone();
            let try_pack = {
                let w_node = out.node(w_id);
                match &w_node.op {
                    Op::Constant { data } if w_node.shape.dtype() == DType::F32 => {
                        let vals = decode_f32(data);
                        let dims = static_dims(&w_node.shape);
                        let ok_ternary =
                            encoding == WeightEncoding::GgufTQ2_0 && is_ternary_f32(&vals);
                        let ok_quant = encoding == WeightEncoding::GgufQ8_0;
                        if let Some(dims) = dims.filter(|_| ok_ternary || ok_quant) {
                            // DequantMatMul / GGUF path is BT: weight stored as [N, K].
                            let to_pack = if dims.len() == 2 {
                                let (k, n) = (dims[0], dims[1]);
                                transpose_kn(&vals, k, n)
                            } else {
                                vals.clone()
                            };
                            match quantize(&to_pack) {
                                Ok(bytes) => Some((vals.len(), dims, bytes)),
                                Err(_) => None,
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            };

            if let Some((numel, dims, bytes)) = try_pack {
                packed += 1;
                let name = w_name.unwrap_or_else(|| format!("w{}", w_id.0));
                let packed_id = out.add_node(
                    Op::Constant {
                        data: bytes.clone(),
                    },
                    vec![],
                    Shape::new(&[bytes.len()], DType::U8),
                );
                out.node_mut(packed_id).name = Some(name.clone());
                rewrites.push(WeightRewrite {
                    name: name.clone(),
                    // Logical MatMul weight shape [K, N]; bytes are BT-packed [N, K].
                    shape: dims,
                    encoding,
                    data: bytes,
                    note: format!(
                        "packed {numel} f32 → {} (BT [N,K] for DequantMatMul)",
                        encoding.as_str()
                    ),
                });
                let mm = out.add_node(
                    Op::DequantMatMul { scheme },
                    vec![new_inputs[0], packed_id],
                    node.shape.clone(),
                );
                if let Some(n) = &node.name {
                    out.node_mut(mm).name = Some(n.clone());
                }
                mm
            } else {
                copy_node(&mut out, node, new_inputs)
            }
        } else {
            copy_node(&mut out, node, new_inputs)
        };
        id_map.insert(node.id, new_id);
    }

    let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|o| id_map[o]).collect();
    out.set_outputs(new_outputs);
    (out, rewrites, packed)
}

/// Pack exact-ternary MatMul weights as GGUF TQ2_0 + `DequantMatMul`.
pub fn pack_ternary_matmul(graph: &Graph) -> (Graph, Vec<WeightRewrite>, usize) {
    pack_matmul_weight(
        graph,
        QuantScheme::GgufTQ2_0,
        WeightEncoding::GgufTQ2_0,
        |v| rlx_gguf::quantize(v, rlx_gguf::GgmlType::TQ2_0),
    )
}

/// Pack F32 MatMul weights as GGUF Q8_0 + `DequantMatMul` (numel % 32 == 0).
pub fn pack_quant_matmul(graph: &Graph) -> (Graph, Vec<WeightRewrite>, usize) {
    pack_matmul_weight(
        graph,
        QuantScheme::GgufQ8_0,
        WeightEncoding::GgufQ8_0,
        |v| rlx_gguf::quantize(v, rlx_gguf::GgmlType::Q8_0),
    )
}

fn encoding_for_weight(consumer: &Op, _weight_dtype: DType) -> WeightEncoding {
    if let Op::DequantMatMul { scheme } = consumer {
        return match scheme {
            QuantScheme::GgufTQ2_0 => WeightEncoding::GgufTQ2_0,
            QuantScheme::GgufQ8_0 => WeightEncoding::GgufQ8_0,
            _ => WeightEncoding::F32,
        };
    }
    WeightEncoding::F32
}

/// Unfold MatMul / DequantMatMul / Conv weight Constants into named table entries.
pub fn unfold_weights(graph: &Graph) -> Vec<WeightRewrite> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for node in graph.nodes() {
        let weight_inputs: &[usize] = match &node.op {
            Op::MatMul | Op::DequantMatMul { .. } => &[1],
            Op::Conv { .. } => &[1],
            Op::FusedMatMulBiasAct { .. } => &[1],
            Op::FusedConvBiasAct { .. } => &[1],
            _ => &[],
        };
        for &idx in weight_inputs {
            if idx >= node.inputs.len() {
                continue;
            }
            let wid = node.inputs[idx];
            if !seen.insert(wid.0) {
                continue;
            }
            let wn = graph.node(wid);
            let Op::Constant { data } = &wn.op else {
                continue;
            };
            let dims = static_dims(&wn.shape).unwrap_or_else(|| vec![data.len()]);
            let encoding = encoding_for_weight(&node.op, wn.shape.dtype());
            let name = wn.name.clone().unwrap_or_else(|| format!("w{}", wid.0));
            out.push(WeightRewrite {
                name,
                shape: dims,
                encoding,
                data: data.clone(),
                note: format!("unfolded {}", encoding.as_str()),
            });
        }
    }
    out
}

/// Run weight-aware optimize passes according to flags.
pub fn optimize_weights(
    graph: Graph,
    skip: bool,
    ternary: bool,
    quant: bool,
    unfold: bool,
) -> (Graph, Vec<WeightRewrite>, OptimizeStats) {
    let mut stats = OptimizeStats::default();
    let mut rewrites = Vec::new();
    let mut g = graph;

    if skip {
        let (ng, n) = skip_zero_matmul(&g);
        g = ng;
        stats.skipped_zero_matmuls = n;
    }
    if ternary {
        let (ng, rw, n) = pack_ternary_matmul(&g);
        g = ng;
        stats.ternary_packed = n;
        rewrites.extend(rw);
    }
    if quant {
        let (ng, rw, n) = pack_quant_matmul(&g);
        g = ng;
        stats.quant_packed = n;
        rewrites.extend(rw);
    }
    if unfold {
        let unfolded = unfold_weights(&g);
        stats.weights_unfolded = unfolded.len();
        // Prefer already-packed rewrites; add unfolded entries whose names are new.
        let known: std::collections::HashSet<String> =
            rewrites.iter().map(|r| r.name.clone()).collect();
        for u in unfolded {
            if !known.contains(&u.name) {
                rewrites.push(u);
            }
        }
    }
    (g, rewrites, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::BinaryOp;

    #[test]
    fn skip_zero_removes_matmul() {
        let s_x = Shape::new(&[2, 4], DType::F32);
        let s_w = Shape::new(&[4, 3], DType::F32);
        let s_y = Shape::new(&[2, 3], DType::F32);
        let mut g = Graph::new("z");
        let x = g.input("x", s_x);
        let w = g.add_node(
            Op::Constant {
                data: encode_f32(&[0.0; 12]),
            },
            vec![],
            s_w,
        );
        let y = g.add_node(Op::MatMul, vec![x, w], s_y.clone());
        g.set_outputs(vec![y]);
        let (out, n) = skip_zero_matmul(&g);
        assert_eq!(n, 1);
        assert!(matches!(out.node(out.outputs[0]).op, Op::Constant { .. }));
    }

    #[test]
    fn ternary_detect() {
        assert!(is_ternary_f32(&[-1.0, 0.0, 1.0, 0.0]));
        assert!(!is_ternary_f32(&[0.5, 0.0]));
    }

    #[test]
    fn mul_not_confused() {
        let s = Shape::new(&[2], DType::F32);
        let mut g = Graph::new("m");
        let x = g.input("x", s.clone());
        let w = g.param("w", s.clone());
        let y = g.binary(BinaryOp::Mul, x, w, s);
        g.set_outputs(vec![y]);
        let (out, n) = skip_zero_matmul(&g);
        assert_eq!(n, 0);
        assert_eq!(out.len(), g.len());
    }
}
