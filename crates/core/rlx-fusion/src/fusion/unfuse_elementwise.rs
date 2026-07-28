// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `unfuse_elementwise` — extracted from the `fusion` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

// ── Helper: graph rewriter ──────────────────────────────────────────────

use crate::graph_rewrite::Rewriter;

// ── Pass 1: MatMul + Bias + Activation → FusedMatMulBiasAct ─────────────

use super::*;

pub struct UnfuseElementwiseRegions {
    /// When false, `ElementwiseRegion` nodes with an FKL prologue are kept
    /// for native GPU region kernels; when true (CPU), they decompose too.
    pub unfuse_prologue: bool,
}

impl UnfuseElementwiseRegions {
    /// GPU / Metal / CUDA / wgpu: unfuse plain regions, keep resize prologue.
    pub const FOR_GPU: UnfuseElementwiseRegions = UnfuseElementwiseRegions {
        unfuse_prologue: false,
    };
    /// CPU: decompose every region (no native region executor).
    pub const FOR_CPU: UnfuseElementwiseRegions = UnfuseElementwiseRegions {
        unfuse_prologue: true,
    };
}

impl Pass for UnfuseElementwiseRegions {
    fn name(&self) -> &str {
        "unfuse_elementwise_regions"
    }

    fn run(&self, graph: Graph) -> Graph {
        let any_region = graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::ElementwiseRegion { .. }));
        if !any_region {
            return graph;
        }

        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            if let Op::ElementwiseRegion {
                chain,
                num_inputs: _,
                scalar_input_mask: _,
                input_modulus: _,
                prologue,
                prologue_input: _,
            } = &node.op
            {
                if *prologue != RegionPrologue::None && !self.unfuse_prologue {
                    rw.copy_node(node);
                    continue;
                }
                let mut region_inputs: Vec<NodeId> =
                    node.inputs.iter().map(|id| rw.map(*id)).collect();
                if *prologue == RegionPrologue::ResizeNearest2x {
                    let in_shape = rw.new_graph.node(region_inputs[0]).shape.clone();
                    let out_shape = if in_shape.rank() == 4 {
                        Shape::new(
                            &[
                                in_shape.dim(0).unwrap_static(),
                                in_shape.dim(1).unwrap_static(),
                                in_shape.dim(2).unwrap_static() * 2,
                                in_shape.dim(3).unwrap_static() * 2,
                            ],
                            in_shape.dtype(),
                        )
                    } else {
                        node.shape.clone()
                    };
                    region_inputs[0] = rw.new_graph.add_node(
                        Op::ResizeNearest2x,
                        vec![region_inputs[0]],
                        out_shape,
                    );
                }
                let mut step_ids: Vec<NodeId> = Vec::with_capacity(chain.len());
                let region_shape = node.shape.clone();
                let region_dims: Vec<_> = region_shape.dims().to_vec();
                // Per-step result dtype, indexed by step position.
                // The chain may pass through Cast steps that change the
                // dtype mid-chain; using `region_shape.dtype()` blindly
                // would mis-tag intermediate Activation/Binary/Where
                // shapes. Track the dtype propagated through each step.
                let mut step_dtypes: Vec<rlx_ir::DType> = Vec::with_capacity(chain.len());
                let region_dtype = region_shape.dtype();
                let dtype_of = |op: &ChainOperand,
                                ins: &[NodeId],
                                step_dt: &[rlx_ir::DType],
                                rw: &Rewriter|
                 -> rlx_ir::DType {
                    match *op {
                        ChainOperand::Input(i) => rw.new_graph.node(ins[i as usize]).shape.dtype(),
                        ChainOperand::Step(i) => step_dt[i as usize],
                    }
                };
                // Shape of an operand in the rewritten graph. Critical
                // for broadcast inputs: a region whose final shape is
                // `[8, 1]` can still have a scalar operand at some
                // step; tagging that step with region_dims would lie
                // about its element count and trip the binary/activation
                // kernels (which size their reads/writes off the IR
                // shape, not the broadcast-aware semantics the L2
                // region kernel would have used). Use the actual node
                // shape so the unfused pipeline matches what each op
                // semantically produces.
                let shape_of = |op: &ChainOperand,
                                ins: &[NodeId],
                                step_ids: &[NodeId],
                                rw: &Rewriter|
                 -> Shape {
                    match *op {
                        ChainOperand::Input(i) => rw.new_graph.node(ins[i as usize]).shape.clone(),
                        ChainOperand::Step(i) => {
                            rw.new_graph.node(step_ids[i as usize]).shape.clone()
                        }
                    }
                };
                for step in chain {
                    let resolve = |op: &ChainOperand| -> NodeId {
                        match *op {
                            ChainOperand::Input(i) => region_inputs[i as usize],
                            ChainOperand::Step(i) => step_ids[i as usize],
                        }
                    };
                    let (new_id, dt) = match step {
                        ChainStep::Activation(a, src) => {
                            let s = resolve(src);
                            let dt = dtype_of(src, &region_inputs, &step_dtypes, &rw);
                            // Activation is element-wise: output shape
                            // == input shape (preserve broadcast-source
                            // shapes; do NOT promote to region_dims).
                            let src_shape = shape_of(src, &region_inputs, &step_ids, &rw);
                            let dims: Vec<_> = src_shape.dims().to_vec();
                            let shape = Shape::from_dims(&dims, dt);
                            (
                                rw.new_graph.add_node(Op::Activation(*a), vec![s], shape),
                                dt,
                            )
                        }
                        ChainStep::Cast(to, src) => {
                            let s = resolve(src);
                            let src_shape = shape_of(src, &region_inputs, &step_ids, &rw);
                            let dims: Vec<_> = src_shape.dims().to_vec();
                            let shape = Shape::from_dims(&dims, *to);
                            (
                                rw.new_graph.add_node(Op::Cast { to: *to }, vec![s], shape),
                                *to,
                            )
                        }
                        ChainStep::Binary(op, lhs, rhs) => {
                            let l = resolve(lhs);
                            let r = resolve(rhs);
                            let dt = dtype_of(lhs, &region_inputs, &step_dtypes, &rw);
                            // Binary: NumPy-style broadcast of operands.
                            let lhs_shape = shape_of(lhs, &region_inputs, &step_ids, &rw);
                            let rhs_shape = shape_of(rhs, &region_inputs, &step_ids, &rw);
                            let bcast = rlx_ir::shape::broadcast(&lhs_shape, &rhs_shape)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "unfuse_elementwise_regions: cannot broadcast \
                                         {lhs_shape:?} ⊗ {rhs_shape:?} for Binary({op:?}): {e}"
                                    )
                                });
                            let dims: Vec<_> = bcast.dims().to_vec();
                            let shape = Shape::from_dims(&dims, dt);
                            (
                                rw.new_graph.add_node(Op::Binary(*op), vec![l, r], shape),
                                dt,
                            )
                        }
                        ChainStep::Compare(op, lhs, rhs) => {
                            let l = resolve(lhs);
                            let r = resolve(rhs);
                            let lhs_shape = shape_of(lhs, &region_inputs, &step_ids, &rw);
                            let rhs_shape = shape_of(rhs, &region_inputs, &step_ids, &rw);
                            let bcast = rlx_ir::shape::broadcast(&lhs_shape, &rhs_shape)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "unfuse_elementwise_regions: cannot broadcast \
                                         {lhs_shape:?} ⊗ {rhs_shape:?} for Compare({op:?}): {e}"
                                    )
                                });
                            let dims: Vec<_> = bcast.dims().to_vec();
                            let shape = Shape::from_dims(&dims, rlx_ir::DType::Bool);
                            (
                                rw.new_graph.add_node(Op::Compare(*op), vec![l, r], shape),
                                rlx_ir::DType::Bool,
                            )
                        }
                        ChainStep::Where(c, x, y) => {
                            let cn = resolve(c);
                            let xn = resolve(x);
                            let yn = resolve(y);
                            let dt = dtype_of(x, &region_inputs, &step_dtypes, &rw);
                            // Where: broadcast across (cond, then, else).
                            let c_shape = shape_of(c, &region_inputs, &step_ids, &rw);
                            let x_shape = shape_of(x, &region_inputs, &step_ids, &rw);
                            let y_shape = shape_of(y, &region_inputs, &step_ids, &rw);
                            let bcast_xy = rlx_ir::shape::broadcast(&x_shape, &y_shape)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "unfuse_elementwise_regions: cannot broadcast \
                                         then/else {x_shape:?} ⊗ {y_shape:?} for Where: {e}"
                                    )
                                });
                            let bcast = rlx_ir::shape::broadcast(&c_shape, &bcast_xy)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "unfuse_elementwise_regions: cannot broadcast cond \
                                         {c_shape:?} ⊗ {bcast_xy:?} for Where: {e}"
                                    )
                                });
                            let dims: Vec<_> = bcast.dims().to_vec();
                            let shape = Shape::from_dims(&dims, dt);
                            (
                                rw.new_graph.add_node(Op::Where, vec![cn, xn, yn], shape),
                                dt,
                            )
                        }
                    };
                    step_ids.push(new_id);
                    step_dtypes.push(dt);
                }
                let _ = region_dtype;
                let _ = region_dims;
                // The region's "output" (= last step) replaces the original
                // ElementwiseRegion node id.
                let last = *step_ids.last().expect("chain non-empty per pass invariant");
                rw.replace(node.id, last);
                continue;
            }
            rw.copy_node(node);
        }
        rw.finish(&graph.outputs)
    }
}
