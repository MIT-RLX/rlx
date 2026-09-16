// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Static **bytes-moved** accounting — a memory-traffic ledger read off the IR.
//!
//! Most of the perf work in this repo has converged on the same finding from
//! different directions: the bottleneck was bytes, not math. Metal decode was
//! weight-*bytes*-bound (24 → 89 tps once f16 weights cut the traffic); qwen3.5
//! decode was re-uploading ~91 MB per token; shared params over CUDA VMM removed
//! 3.0 GB of HtoD. Each of those was found with a profiler, after the fact.
//!
//! A graph already contains enough information to predict them. This module
//! computes, per node, the bytes that *must* move and the FLOPs that *must*
//! happen, then reports arithmetic intensity — so "this graph moves 3× more than
//! the math needs" is answerable before anything is run, on any backend, with no
//! device present.
//!
//! # This is a lower bound, and it is diagnostic only
//!
//! [`NodeCost::bytes`] assumes each operand is read exactly once and the output
//! written exactly once — perfect reuse within an op, no reuse across ops. A
//! real kernel can only do worse: a naive matmul re-reads `B` once per output
//! row, an unfused chain round-trips through DRAM between every op. So the
//! number is a floor. "Measured traffic ≫ this" means there is headroom;
//! "measured ≈ this" means the op is already at its information-theoretic
//! minimum and the win has to come from somewhere else.
//!
//! **Do not wire this into a rewrite decision.** The fusion-pipeline work
//! already produced three "obvious" optimizations that measured as regressions,
//! and the region interpreter is 4–8× *slower* than not fusing at all despite
//! moving strictly fewer bytes. A static model cannot see launch overhead,
//! occupancy, or cache behaviour. Report it; let a benchmark decide.

use rlx_ir::op::OpKind;
use rlx_ir::{Graph, NodeId, Shape};

/// Traffic and work attributable to one node.
#[derive(Debug, Clone)]
pub struct NodeCost {
    /// Node this describes.
    pub id: NodeId,
    /// Op discriminant.
    pub kind: OpKind,
    /// Optional node name, when the graph carries one.
    pub name: Option<String>,
    /// Bytes read from operands, assuming each is streamed once.
    pub bytes_read: u64,
    /// Bytes written for the output tensor.
    pub bytes_written: u64,
    /// Of `bytes_read`, the share coming from [`OpKind::Param`] /
    /// [`OpKind::Constant`] operands.
    ///
    /// Broken out because it is the traffic that is *re-paid every step* while
    /// being invariant across steps — the qwen3.5 per-token weight re-upload
    /// and the CUDA shared-param work both live in this column.
    pub bytes_weights: u64,
    /// Floating-point operations, where a meaningful count exists.
    pub flops: u64,
}

impl NodeCost {
    /// Total traffic — read plus written.
    pub fn bytes(&self) -> u64 {
        self.bytes_read + self.bytes_written
    }

    /// FLOPs per byte moved. The x-axis of a roofline plot.
    ///
    /// `None` when the node moves no bytes (a shape-only op), which is different
    /// from an intensity of zero.
    pub fn intensity(&self) -> Option<f64> {
        let b = self.bytes();
        (b > 0).then(|| self.flops as f64 / b as f64)
    }
}

/// Whole-graph ledger.
#[derive(Debug, Clone, Default)]
pub struct GraphCost {
    /// Per-node costs, in graph order.
    pub nodes: Vec<NodeCost>,
}

impl GraphCost {
    /// Sum of all node traffic.
    pub fn total_bytes(&self) -> u64 {
        self.nodes.iter().map(|n| n.bytes()).sum()
    }
    /// Sum of all FLOPs.
    pub fn total_flops(&self) -> u64 {
        self.nodes.iter().map(|n| n.flops).sum()
    }
    /// Sum of weight/constant-attributable reads.
    pub fn total_weight_bytes(&self) -> u64 {
        self.nodes.iter().map(|n| n.bytes_weights).sum()
    }
    /// Whole-graph arithmetic intensity.
    pub fn intensity(&self) -> f64 {
        let b = self.total_bytes();
        if b == 0 {
            return 0.0;
        }
        self.total_flops() as f64 / b as f64
    }

    /// The `n` heaviest nodes by traffic, descending.
    pub fn top_by_bytes(&self, n: usize) -> Vec<&NodeCost> {
        let mut v: Vec<&NodeCost> = self.nodes.iter().filter(|c| c.bytes() > 0).collect();
        v.sort_by_key(|c| std::cmp::Reverse(c.bytes()));
        v.truncate(n);
        v
    }

    /// Traffic grouped by op kind, descending — where the bytes actually go.
    pub fn by_kind(&self) -> Vec<(OpKind, u64, u64)> {
        let mut acc: Vec<(OpKind, u64, u64)> = Vec::new();
        for c in &self.nodes {
            match acc.iter_mut().find(|(k, _, _)| *k == c.kind) {
                Some(e) => {
                    e.1 += c.bytes();
                    e.2 += c.flops;
                }
                None => acc.push((c.kind, c.bytes(), c.flops)),
            }
        }
        acc.sort_by_key(|(_, b, _)| std::cmp::Reverse(*b));
        acc
    }

    /// Human-readable summary: totals, top nodes, and the per-kind breakdown.
    pub fn report(&self, top: usize) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let tb = self.total_bytes();
        let tw = self.total_weight_bytes();
        let _ = writeln!(
            s,
            "bytes moved (lower bound): {}   flops: {}   intensity: {:.2} flop/byte",
            human_bytes(tb),
            human_count(self.total_flops()),
            self.intensity()
        );
        if tb > 0 {
            let _ = writeln!(
                s,
                "  of which weights/constants: {} ({:.1}%)",
                human_bytes(tw),
                100.0 * tw as f64 / tb as f64
            );
        }
        let _ = writeln!(s, "\ntop {top} nodes by traffic:");
        for c in self.top_by_bytes(top) {
            let _ = writeln!(
                s,
                "  {:>10}  {:>8.2} f/B  {:?}{}",
                human_bytes(c.bytes()),
                c.intensity().unwrap_or(0.0),
                c.kind,
                c.name
                    .as_deref()
                    .map(|n| format!(" [{n}]"))
                    .unwrap_or_default()
            );
        }
        let _ = writeln!(s, "\nby op kind:");
        for (k, b, f) in self.by_kind().into_iter().take(top) {
            if b == 0 {
                continue;
            }
            let _ = writeln!(
                s,
                "  {:>10}  {:>8.2} f/B  {k:?}",
                human_bytes(b),
                f as f64 / b as f64
            );
        }
        s
    }

    /// Split nodes into memory-bound and compute-bound against a machine
    /// balance (peak FLOP/s ÷ peak byte/s — the roofline ridge point).
    ///
    /// A node below the ridge cannot be helped by a faster ALU; one above it
    /// cannot be helped by faster memory. Returns `(memory_bound, compute_bound)`
    /// traffic totals.
    pub fn roofline_split(&self, ridge_flops_per_byte: f64) -> (u64, u64) {
        let mut mem = 0u64;
        let mut comp = 0u64;
        for c in &self.nodes {
            match c.intensity() {
                Some(i) if i >= ridge_flops_per_byte => comp += c.bytes(),
                Some(_) => mem += c.bytes(),
                None => {}
            }
        }
        (mem, comp)
    }
}

/// Byte count of a tensor, or 0 when the shape is dynamic.
///
/// Dynamic dims are skipped rather than guessed — a made-up batch size would
/// silently dominate the ledger.
fn tensor_bytes(shape: &Shape) -> u64 {
    match shape.num_elements() {
        Some(n) => (n * shape.dtype().size_bytes()) as u64,
        None => 0,
    }
}

/// Build the ledger for `graph`.
pub fn analyze(graph: &Graph) -> GraphCost {
    let nodes = graph
        .nodes()
        .iter()
        .map(|node| {
            let mut bytes_read = 0u64;
            let mut bytes_weights = 0u64;
            for &input in &node.inputs {
                let src = graph.node(input);
                let b = tensor_bytes(&src.shape);
                bytes_read += b;
                if matches!(src.op.kind(), OpKind::Param | OpKind::Constant) {
                    bytes_weights += b;
                }
            }
            // Leaves are storage, not movement: counting an Input's own tensor
            // as "written" would double-count it against every consumer's read.
            let is_leaf = matches!(
                node.op.kind(),
                OpKind::Input | OpKind::Param | OpKind::Constant
            );
            let bytes_written = if is_leaf {
                0
            } else {
                tensor_bytes(&node.shape)
            };
            NodeCost {
                id: node.id,
                kind: node.op.kind(),
                name: node.name.clone(),
                bytes_read: if is_leaf { 0 } else { bytes_read },
                bytes_written,
                bytes_weights: if is_leaf { 0 } else { bytes_weights },
                flops: flops(graph, node),
            }
        })
        .collect();
    GraphCost { nodes }
}

/// FLOPs for one node.
///
/// Only ops with an unambiguous count contribute. An op whose cost is not
/// modelled returns 0 rather than a guess — an inflated intensity would point
/// optimization effort at the wrong node, which is worse than reporting none.
fn flops(graph: &Graph, node: &rlx_ir::graph::Node) -> u64 {
    let out = node.shape.num_elements().unwrap_or(0) as u64;
    let operand_elems = |i: usize| -> u64 {
        node.inputs
            .get(i)
            .and_then(|id| graph.node(*id).shape.num_elements())
            .unwrap_or(0) as u64
    };
    // Contracted dimension for a matmul: the last dim of the LHS.
    let contract = || -> u64 {
        node.inputs
            .first()
            .map(|id| &graph.node(*id).shape)
            .and_then(|s| s.dims().last().copied())
            .and_then(|d| match d {
                rlx_ir::Dim::Static(n) => Some(n as u64),
                _ => None,
            })
            .unwrap_or(0)
    };
    match node.op.kind() {
        // 2 flops (multiply + add) per output element per contracted step.
        OpKind::MatMul
        | OpKind::DequantMatMul
        | OpKind::ScaledMatMul
        | OpKind::ScaledGroupedMatMul
        | OpKind::SynthMatMul => 2 * out * contract(),
        // One pass over the larger operand.
        OpKind::Binary | OpKind::Compare => out.max(operand_elems(0)),
        OpKind::Activation | OpKind::Cast => out,
        // Reductions and scans touch every input element once.
        OpKind::Reduce | OpKind::Cumsum | OpKind::CumProd | OpKind::CumMax => operand_elems(0),
        // Normalizations: two passes (statistics, then rescale) plus the affine.
        OpKind::LayerNorm | OpKind::RmsNorm => 3 * operand_elems(0),
        OpKind::Softmax => 3 * operand_elems(0),
        // Pure data movement — real cost is the bytes, already counted.
        OpKind::Reshape
        | OpKind::Transpose
        | OpKind::Slice
        | OpKind::Concat
        | OpKind::Pad
        | OpKind::Expand
        | OpKind::Gather
        | OpKind::ScatterAdd
        | OpKind::Input
        | OpKind::Param
        | OpKind::Constant => 0,
        _ => 0,
    }
}

/// `1.5 GB` etc. Base-1024, three significant figures.
pub fn human_bytes(b: u64) -> String {
    const U: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < U.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{b} B")
    } else {
        format!("{v:.2} {}", U[i])
    }
}

/// `1.50 G` etc. Base-1000, for FLOP counts.
pub fn human_count(n: u64) -> String {
    const U: [&str; 5] = ["", "K", "M", "G", "T"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1000.0 && i + 1 < U.len() {
        v /= 1000.0;
        i += 1;
    }
    if i == 0 {
        format!("{n}")
    } else {
        format!("{v:.2} {}", U[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Op, Shape};

    /// A matmul's traffic and FLOPs must match the closed form.
    #[test]
    fn matmul_cost_matches_closed_form() {
        let (m, k, n) = (64usize, 128usize, 32usize);
        let mut g = Graph::new("mm");
        let a = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            Shape::new(&[m, k], DType::F32),
        );
        let b = g.add_node(
            Op::Param { name: "w".into() },
            vec![],
            Shape::new(&[k, n], DType::F32),
        );
        let c = g.add_node(Op::MatMul, vec![a, b], Shape::new(&[m, n], DType::F32));

        let cost = analyze(&g);
        let mm = cost.nodes.iter().find(|x| x.id == c).unwrap();
        assert_eq!(mm.flops, (2 * m * n * k) as u64);
        assert_eq!(mm.bytes_read, ((m * k + k * n) * 4) as u64);
        assert_eq!(mm.bytes_written, (m * n * 4) as u64);
        // The B operand is a Param, so it lands in the weight column.
        assert_eq!(mm.bytes_weights, (k * n * 4) as u64);
    }

    /// Leaves must not contribute traffic of their own — otherwise every
    /// weight is counted twice, once as storage and once as a consumer's read.
    #[test]
    fn leaves_do_not_double_count() {
        let mut g = Graph::new("leaf");
        let a = g.add_node(
            Op::Param { name: "w".into() },
            vec![],
            Shape::new(&[1024], DType::F32),
        );
        let _ = g.add_node(
            Op::Activation(rlx_ir::op::Activation::Neg),
            vec![a],
            Shape::new(&[1024], DType::F32),
        );
        let cost = analyze(&g);
        let leaf = cost.nodes.iter().find(|c| c.id == a).unwrap();
        assert_eq!(leaf.bytes(), 0, "a leaf moves no bytes by itself");
        // Read once by the consumer, written once.
        assert_eq!(cost.total_bytes(), 1024 * 4 * 2);
    }

    /// Halving the weight dtype must halve the weight traffic — the property
    /// the Metal f16-weights work exploited, checkable without a device.
    #[test]
    fn narrower_weights_halve_weight_traffic() {
        let build = |dt: DType| {
            let mut g = Graph::new("w");
            let x = g.add_node(
                Op::Input { name: "x".into() },
                vec![],
                Shape::new(&[1, 4096], DType::F32),
            );
            let w = g.add_node(
                Op::Param { name: "w".into() },
                vec![],
                Shape::new(&[4096, 4096], dt),
            );
            g.add_node(Op::MatMul, vec![x, w], Shape::new(&[1, 4096], DType::F32));
            analyze(&g)
        };
        let f32_cost = build(DType::F32);
        let f16_cost = build(DType::F16);
        assert_eq!(
            f32_cost.total_weight_bytes(),
            2 * f16_cost.total_weight_bytes()
        );
        // And a decode-shaped matmul is overwhelmingly weight traffic.
        let share = f32_cost.total_weight_bytes() as f64 / f32_cost.total_bytes() as f64;
        assert!(
            share > 0.99,
            "decode GEMV should be ~all weight bytes: {share}"
        );
    }

    /// A GEMV is memory-bound and a large GEMM is compute-bound at any
    /// realistic ridge point. If this ever inverts, the FLOP model is wrong.
    #[test]
    fn roofline_separates_gemv_from_gemm() {
        let mk = |m: usize| {
            let mut g = Graph::new("r");
            let x = g.add_node(
                Op::Input { name: "x".into() },
                vec![],
                Shape::new(&[m, 4096], DType::F32),
            );
            let w = g.add_node(
                Op::Param { name: "w".into() },
                vec![],
                Shape::new(&[4096, 4096], DType::F32),
            );
            let y = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[m, 4096], DType::F32));
            let c = analyze(&g);
            c.nodes
                .iter()
                .find(|n| n.id == y)
                .unwrap()
                .intensity()
                .unwrap()
        };
        let gemv = mk(1);
        let gemm = mk(4096);
        assert!(gemv < 10.0, "GEMV intensity should be low: {gemv}");
        assert!(gemm > 100.0, "large GEMM should be compute-bound: {gemm}");
        assert!(gemm > gemv * 50.0);
    }

    /// Dynamic shapes contribute nothing rather than a fabricated number.
    #[test]
    fn dynamic_shapes_are_skipped_not_guessed() {
        let mut g = Graph::new("dyn");
        let s = Shape::from_dims(
            &[rlx_ir::Dim::Dynamic(0), rlx_ir::Dim::Static(16)],
            DType::F32,
        );
        let a = g.add_node(Op::Input { name: "x".into() }, vec![], s.clone());
        let _ = g.add_node(Op::Activation(rlx_ir::op::Activation::Neg), vec![a], s);
        assert_eq!(analyze(&g).total_bytes(), 0);
    }

    /// The report renders without panicking on an empty graph.
    #[test]
    fn report_handles_empty_graph() {
        let g = Graph::new("empty");
        let r = analyze(&g).report(5);
        assert!(r.contains("bytes moved"));
    }
}
