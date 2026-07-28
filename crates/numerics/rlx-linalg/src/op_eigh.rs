// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `eigh` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Node, NodeId, OpExtension, Shape, VjpContext};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct EighExt;

impl OpExtension for EighExt {
    fn name(&self) -> &str {
        LINALG_EIGH
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "eigh: A must be F64");
        assert_eq!(a.rank(), 2, "eigh: A must be 2D");
        let n = a.num_elements().expect("eigh: A must be statically shaped");
        let n_dim = (n as f64).sqrt() as usize;
        assert_eq!(n_dim * n_dim, n, "eigh: A must be square");
        // Packed: [eigenvalues (n), eigenvectors (n²)] flat 1D.
        Shape::new(&[n_dim + n], DType::F64)
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Forward output is packed [λ (n), V (n²)]. Upstream has the
        // same layout (built from the user's downstream Narrow + ops).
        // Unpack both, call eigh_backward(λ, V, dL/dλ, dL/dV).
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let a_shape = ctx.bwd.node(a_bwd).shape.clone();
        let n = match a_shape.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Vec::new(),
        };

        let packed_fwd = ctx.fwd_map[&node.id];
        let lambda_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: n,
            },
            vec![packed_fwd],
            Shape::new(&[n], DType::F64),
        );
        let v_flat_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: n,
                len: n * n,
            },
            vec![packed_fwd],
            Shape::new(&[n * n], DType::F64),
        );

        let dl_dlambda = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: n,
            },
            vec![ctx.upstream],
            Shape::new(&[n], DType::F64),
        );
        let dl_dv_flat = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: n,
                len: n * n,
            },
            vec![ctx.upstream],
            Shape::new(&[n * n], DType::F64),
        );

        // eigh_backward kernel reads V and dL/dV as flat n²; reshape
        // not strictly necessary because the kernel computes its
        // own row/col indexing — but we wrap them so shape inference
        // for the backward op sees consistent metadata.
        let g_a = ctx.bwd.custom_op(
            LINALG_EIGH_BACKWARD,
            Vec::new(),
            vec![lambda_fwd, v_flat_fwd, dl_dlambda, dl_dv_flat],
        );
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Forward Frechet via the eigh_jvp kernel. Inputs to the kernel:
        // (λ, V_flat, dA). Output: packed [t_λ, t_V_flat] of length n+n².
        let t_a = ctx.tangents[0]?;
        let a = ctx.fwd_map[&node.inputs[0]];
        let n = match ctx.bwd.shape(a).dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return None,
        };
        // Unpack λ and V from forward output (stored in JVP graph at fwd_map[&node.id]).
        let packed_fwd = ctx.fwd_map[&node.id];
        let lambda = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: n,
            },
            vec![packed_fwd],
            Shape::new(&[n], DType::F64),
        );
        let v_flat = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: n,
                len: n * n,
            },
            vec![packed_fwd],
            Shape::new(&[n * n], DType::F64),
        );
        // dA might be 2D [n,n] but the kernel expects flat n²; reshape.
        let da_flat = ctx.bwd.add_node(
            rlx_ir::Op::Reshape {
                new_shape: vec![(n * n) as i64],
            },
            vec![t_a],
            Shape::new(&[n * n], DType::F64),
        );
        Some(
            ctx.bwd
                .custom_op(LINALG_EIGH_JVP, Vec::new(), vec![lambda, v_flat, da_flat]),
        )
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct EighCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for EighCpu {
    fn name(&self) -> &str {
        LINALG_EIGH
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("eigh A")?;
        let out = output.expect_f64_mut("eigh out")?;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("eigh: A length {n_sq} not n²"));
        }
        algos::eigh(a, n, out)
    }
}

// ── QR ───────────────────────────────────────────────────────────
