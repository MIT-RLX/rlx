// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `qr` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::{DType, Node, NodeId, OpExtension, Shape, VjpContext};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct QrExt;

impl OpExtension for QrExt {
    fn name(&self) -> &str {
        LINALG_QR
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], attrs: &[u8]) -> Shape {
        // Shapes need both m and n. The infer_shape input is the matrix
        // A which carries them. We encode no special attrs (yet).
        let _ = attrs;
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "qr: A must be F64");
        assert_eq!(a.rank(), 2, "qr: A must be 2D");
        let m = match a.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("qr: dynamic dim"),
        };
        let n = match a.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("qr: dynamic dim"),
        };
        let k = m.min(n);
        Shape::new(&[m * k + k * n], DType::F64)
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Walter–Lehmann 2010 closed form via the qr_backward kernel.
        // Forward output is packed [Q (m·k), R (k·n)]; upstream has
        // the same layout. Unpack and call qr_backward(Q, R, dQ, dR).
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let a_shape = ctx.bwd.node(a_bwd).shape.clone();
        let m = match a_shape.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Vec::new(),
        };
        let n = match a_shape.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Vec::new(),
        };
        let k = m.min(n);

        let packed_fwd = ctx.fwd_map[&node.id];
        let q_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: m * k,
            },
            vec![packed_fwd],
            Shape::new(&[m * k], DType::F64),
        );
        let r_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k,
                len: k * n,
            },
            vec![packed_fwd],
            Shape::new(&[k * n], DType::F64),
        );
        let dq = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: m * k,
            },
            vec![ctx.upstream],
            Shape::new(&[m * k], DType::F64),
        );
        let dr = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k,
                len: k * n,
            },
            vec![ctx.upstream],
            Shape::new(&[k * n], DType::F64),
        );

        let g_a = ctx
            .bwd
            .custom_op(LINALG_QR_BACKWARD, Vec::new(), vec![q_fwd, r_fwd, dq, dr]);
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Walter-Lehmann forward via qr_jvp kernel. Inputs: (Q_flat,
        // R_flat, dA_flat); output packed [dQ_flat, dR_flat].
        let t_a = ctx.tangents[0]?;
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let a_shape = ctx.bwd.node(a_bwd).shape.clone();
        let m = match a_shape.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return None,
        };
        let n = match a_shape.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => return None,
        };
        let k = m.min(n);
        let packed_fwd = ctx.fwd_map[&node.id];
        let q_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: m * k,
            },
            vec![packed_fwd],
            Shape::new(&[m * k], DType::F64),
        );
        let r_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k,
                len: k * n,
            },
            vec![packed_fwd],
            Shape::new(&[k * n], DType::F64),
        );
        let da_flat = ctx.bwd.add_node(
            rlx_ir::Op::Reshape {
                new_shape: vec![(m * n) as i64],
            },
            vec![t_a],
            Shape::new(&[m * n], DType::F64),
        );
        Some(
            ctx.bwd
                .custom_op(LINALG_QR_JVP, Vec::new(), vec![q_fwd, r_fwd, da_flat]),
        )
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct QrCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for QrCpu {
    fn name(&self) -> &str {
        LINALG_QR
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("qr A")?;
        let a_shape = inputs[0].shape();
        let m = match a_shape.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Err("qr: dynamic dim 0".into()),
        };
        let n = match a_shape.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Err("qr: dynamic dim 1".into()),
        };
        let out = output.expect_f64_mut("qr out")?;
        algos::qr(a, m, n, out)
    }
}

// ── SVD ──────────────────────────────────────────────────────────
