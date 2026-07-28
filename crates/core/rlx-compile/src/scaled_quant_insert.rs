// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native low-precision GEMM insertion (AMP-FP8/FP4).
//!
//! Rewrites each 2-D compute `Op::MatMul` into a native `Op::ScaledMatMul`:
//! both operands are dynamically quantized (`ScaledQuantScale` +
//! `ScaledQuantize`) and fed straight into the tensor-core GEMM with f32
//! accumulation — the path that wins on Hopper/Ada/Blackwell and CDNA3/CDNA4.
//!
//! `ScaledMatMul` is TN (`out = lhs · rhsᵀ`), so the rhs is transposed to
//! K-last; on a `Param` weight that `Transpose` const-folds at compile time, so
//! only activations pay a runtime quantize. Run this **before** `ConstantFolding`.
//!
//! Unlike [`crate::precision::AutoMixedPrecision`] (a dtype-relabel + `Cast`
//! engine), this is op-replacement + scale plumbing — a different transform, so
//! it lives in its own pass. It changes numerics, so it is **opt-in**: the
//! caller enables it per graph; nothing turns it on by default.

use rlx_ir::*;
use std::collections::HashMap;

/// Which element formats + scale layout to quantize matmul operands to.
#[derive(Debug, Clone, Copy)]
pub struct ScaledQuantConfig {
    pub lhs_format: ScaledFormat,
    pub rhs_format: ScaledFormat,
    pub scale_layout: ScaleLayout,
}

impl ScaledQuantConfig {
    /// Per-tensor FP8 E4M3 for both operands (Hopper / Ada default).
    pub fn fp8_e4m3() -> Self {
        Self {
            lhs_format: ScaledFormat::F8E4M3,
            rhs_format: ScaledFormat::F8E4M3,
            scale_layout: ScaleLayout::PerTensor,
        }
    }
    /// OCP microscaling MXFP8 (E4M3 elements, E8M0 block scales).
    pub fn mxfp8_e4m3() -> Self {
        Self {
            lhs_format: ScaledFormat::F8E4M3,
            rhs_format: ScaledFormat::F8E4M3,
            scale_layout: ScaleLayout::mx(),
        }
    }
}

/// Rewrite every 2-D `Op::MatMul` in `graph` into a native `Op::ScaledMatMul`.
/// Batched (rank > 2) matmuls and all other ops are copied unchanged.
pub fn insert_scaled_matmul(graph: Graph, cfg: ScaledQuantConfig) -> Graph {
    let mut out = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    // Quantize-once: a tensor feeding two matmuls is quantized a single time.
    let mut qcache: HashMap<(NodeId, ScaledFormat), (NodeId, NodeId)> = HashMap::new();
    // Transpose-once for shared rhs weights.
    let mut tcache: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();

        let new_id = match &node.op {
            Op::MatMul if node.shape.rank() == 2 => {
                let lhs = new_inputs[0];
                let rhs = new_inputs[1];
                let lhs_shape = out.node(lhs).shape.clone();
                let rhs_shape = out.node(rhs).shape.clone();
                let m = lhs_shape.dim(0).unwrap_static();
                let k = lhs_shape.dim(1).unwrap_static();
                let n = rhs_shape.dim(1).unwrap_static();

                // rhs [k,n] → [n,k] so ScaledMatMul's TN layout sees K-last.
                let rhs_t = *tcache.entry(rhs).or_insert_with(|| {
                    out.add_node(
                        Op::Transpose { perm: vec![1, 0] },
                        vec![rhs],
                        Shape::new(&[n, k], rhs_shape.dtype()),
                    )
                });

                let (lq, ls) = quantize_operand(
                    &mut out,
                    &mut qcache,
                    lhs,
                    m,
                    k,
                    cfg.lhs_format,
                    cfg.scale_layout,
                );
                let (rq, rs) = quantize_operand(
                    &mut out,
                    &mut qcache,
                    rhs_t,
                    n,
                    k,
                    cfg.rhs_format,
                    cfg.scale_layout,
                );

                out.add_node(
                    Op::ScaledMatMul {
                        lhs_format: cfg.lhs_format,
                        rhs_format: cfg.rhs_format,
                        scale_layout: cfg.scale_layout,
                        has_bias: false,
                    },
                    vec![lq, rq, ls, rs],
                    Shape::new(&[m, n], DType::F32),
                )
            }
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }

    let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|&id| id_map[&id]).collect();
    out.set_outputs(new_outputs);
    out
}

/// Insert (cached) `ScaledQuantScale` + `ScaledQuantize` for one operand of
/// logical shape `[rows, cols]` (blocks along cols = contraction). Returns
/// `(codes_id, scale_id)`.
fn quantize_operand(
    out: &mut Graph,
    qcache: &mut HashMap<(NodeId, ScaledFormat), (NodeId, NodeId)>,
    operand: NodeId,
    rows: usize,
    cols: usize,
    fmt: ScaledFormat,
    layout: ScaleLayout,
) -> (NodeId, NodeId) {
    if let Some(&hit) = qcache.get(&(operand, fmt)) {
        return hit;
    }
    let scale_shape = match layout {
        ScaleLayout::PerTensor => Shape::new(&[1], layout.scale_dtype()),
        _ => Shape::new(
            &[rows, cols.div_ceil(layout.block() as usize)],
            layout.scale_dtype(),
        ),
    };
    let scale = out.add_node(
        Op::ScaledQuantScale {
            format: fmt,
            scale_layout: layout,
        },
        vec![operand],
        scale_shape,
    );
    let codes = out.add_node(
        Op::ScaledQuantize {
            format: fmt,
            scale_layout: layout,
        },
        vec![operand, scale],
        Shape::new(&[rows, cols], DType::U8),
    );
    qcache.insert((operand, fmt), (codes, scale));
    (codes, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_kind(g: &Graph, k: OpKind) -> usize {
        g.nodes().iter().filter(|n| n.op.kind() == k).count()
    }

    #[test]
    fn rewrites_matmul_to_scaled() {
        let mut g = Graph::new("mm");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 3], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
        g.set_outputs(vec![mm]);

        let out = insert_scaled_matmul(g, ScaledQuantConfig::fp8_e4m3());

        assert_eq!(count_kind(&out, OpKind::MatMul), 0, "plain matmul remains");
        assert_eq!(count_kind(&out, OpKind::ScaledMatMul), 1);
        // one ScaledQuantize + one ScaledQuantScale per operand (2 operands)
        assert_eq!(count_kind(&out, OpKind::ScaledQuantize), 2);
        assert_eq!(count_kind(&out, OpKind::ScaledQuantScale), 2);
        // rhs transposed to K-last
        assert_eq!(count_kind(&out, OpKind::Transpose), 1);

        // Output is the ScaledMatMul, shape [2,3] f32.
        let out_node = out.node(out.outputs[0]);
        assert!(matches!(out_node.op, Op::ScaledMatMul { .. }));
        assert_eq!(out_node.shape.dim(0).unwrap_static(), 2);
        assert_eq!(out_node.shape.dim(1).unwrap_static(), 3);
        assert_eq!(out_node.shape.dtype(), DType::F32);
    }

    #[test]
    fn quantize_once_for_shared_activation() {
        // x feeds two matmuls — its quantization must be shared (cached).
        let mut g = Graph::new("shared");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w1 = g.param("w1", Shape::new(&[4, 3], DType::F32));
        let w2 = g.param("w2", Shape::new(&[4, 5], DType::F32));
        let mm1 = g.matmul(x, w1, Shape::new(&[2, 3], DType::F32));
        let mm2 = g.matmul(x, w2, Shape::new(&[2, 5], DType::F32));
        g.set_outputs(vec![mm1, mm2]);

        let out = insert_scaled_matmul(g, ScaledQuantConfig::fp8_e4m3());

        assert_eq!(count_kind(&out, OpKind::ScaledMatMul), 2);
        // 2 weights + 1 shared activation = 3 quantizations, not 4.
        assert_eq!(count_kind(&out, OpKind::ScaledQuantize), 3);
        assert_eq!(count_kind(&out, OpKind::ScaledQuantScale), 3);
    }

    #[test]
    fn coexists_with_f16_amp() {
        // fp8-ify the matmul, then run the f16 AutoMixedPrecision pass on top.
        // The f16 pass must leave the fp8 subgraph intact (no U8→f16 casts, no
        // dtype relabel of ScaledMatMul) while still lowering the surrounding
        // elementwise op.
        use crate::precision::{AutoMixedPrecision, PrecisionPolicy};
        use rlx_fusion::pass::Pass;

        let mut g = Graph::new("mix");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 3], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
        let relu = g.activation(
            rlx_ir::op::Activation::Relu,
            mm,
            Shape::new(&[2, 3], DType::F32),
        );
        g.set_outputs(vec![relu]);

        let g = insert_scaled_matmul(g, ScaledQuantConfig::fp8_e4m3());
        let g = AutoMixedPrecision::new(PrecisionPolicy::AutoMixed).run(g);

        let sm = g
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::ScaledMatMul { .. }))
            .expect("ScaledMatMul survived the f16 AMP pass");
        assert_eq!(
            sm.shape.dtype(),
            DType::F32,
            "ScaledMatMul output stayed f32"
        );
        for &i in &sm.inputs {
            let d = g.node(i).shape.dtype();
            assert!(
                d == DType::U8 || d == DType::F32,
                "ScaledMatMul input wrongly relabeled to {d}"
            );
        }
        // The graph must still pass shape verification end-to-end.
        let errs = rlx_ir::verify::verify(&g);
        assert!(errs.is_empty(), "post-AMP graph verifies: {errs:?}");
    }

    #[test]
    fn leaves_batched_matmul_alone() {
        let mut g = Graph::new("batched");
        let a = g.input("a", Shape::new(&[8, 2, 4], DType::F32));
        let b = g.param("b", Shape::new(&[8, 4, 3], DType::F32));
        let mm = g.matmul(a, b, Shape::new(&[8, 2, 3], DType::F32));
        g.set_outputs(vec![mm]);

        let out = insert_scaled_matmul(g, ScaledQuantConfig::fp8_e4m3());
        assert_eq!(
            count_kind(&out, OpKind::MatMul),
            1,
            "batched matmul untouched"
        );
        assert_eq!(count_kind(&out, OpKind::ScaledMatMul), 0);
    }
}
