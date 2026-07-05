// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! `svd` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::{DType, Node, NodeId, OpExtension, Shape, VjpContext};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct SvdExt;

impl OpExtension for SvdExt {
    fn name(&self) -> &str {
        LINALG_SVD
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "svd: A must be F64");
        assert_eq!(a.rank(), 2, "svd: A must be 2D");
        let m = match a.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("svd: dynamic dim"),
        };
        let n = match a.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("svd: dynamic dim"),
        };
        let k = m.min(n);
        // U (m·k) + S (k) + V^T (k·n)
        Shape::new(&[m * k + k + k * n], DType::F64)
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Forward output is packed [U (m·k), s (k), V^T (k·n)];
        // upstream has the same layout. Unpack and call svd_backward.
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
        let u_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: m * k,
            },
            vec![packed_fwd],
            Shape::new(&[m * k], DType::F64),
        );
        let s_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k,
                len: k,
            },
            vec![packed_fwd],
            Shape::new(&[k], DType::F64),
        );
        let vt_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k + k,
                len: k * n,
            },
            vec![packed_fwd],
            Shape::new(&[k * n], DType::F64),
        );
        let du = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: m * k,
            },
            vec![ctx.upstream],
            Shape::new(&[m * k], DType::F64),
        );
        let ds = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k,
                len: k,
            },
            vec![ctx.upstream],
            Shape::new(&[k], DType::F64),
        );
        let dvt = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k + k,
                len: k * n,
            },
            vec![ctx.upstream],
            Shape::new(&[k * n], DType::F64),
        );

        let g_a = ctx.bwd.custom_op(
            LINALG_SVD_BACKWARD,
            Vec::new(),
            vec![u_fwd, s_fwd, vt_fwd, du, ds, dvt],
        );
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Townsend forward via svd_jvp kernel. Inputs: (U_flat, s,
        // Vt_flat, dA_flat); output packed [dU, ds, dVt].
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
        let u_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: m * k,
            },
            vec![packed_fwd],
            Shape::new(&[m * k], DType::F64),
        );
        let s_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k,
                len: k,
            },
            vec![packed_fwd],
            Shape::new(&[k], DType::F64),
        );
        let vt_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k + k,
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
        Some(ctx.bwd.custom_op(
            LINALG_SVD_JVP,
            Vec::new(),
            vec![u_fwd, s_fwd, vt_fwd, da_flat],
        ))
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct SvdCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SvdCpu {
    fn name(&self) -> &str {
        LINALG_SVD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("svd A")?;
        let a_shape = inputs[0].shape();
        let m = match a_shape.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Err("svd: dynamic dim 0".into()),
        };
        let n = match a_shape.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Err("svd: dynamic dim 1".into()),
        };
        let out = output.expect_f64_mut("svd out")?;
        algos::svd(a, m, n, out)
    }
}

// ── LogDet ────────────────────────────────────────────────────────
