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

//! `pinv` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]


#![allow(unused_imports)]

use rlx_ir::{DType, Node, NodeId, OpExtension, Shape, VjpContext};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct PinvExt;


impl OpExtension for PinvExt {
    fn name(&self) -> &str {
        LINALG_PINV
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "pinv: A must be F64");
        assert_eq!(a.rank(), 2, "pinv: A must be 2D");
        let m = match a.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("pinv: dynamic dim"),
        };
        let n = match a.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("pinv: dynamic dim"),
        };
        Shape::new(&[n, m], DType::F64)
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Y = pinv(A); pinv_backward needs (A, Y, dL/dY) and the m attr.
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => return vec![],
        };
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let y_bwd = ctx.fwd_map[&node.id];
        let g_a = ctx.bwd.custom_op(
            LINALG_PINV_BACKWARD,
            attrs,
            vec![a_bwd, y_bwd, ctx.upstream],
        );
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Forward Frechet via pinv_jvp kernel (does its own internal SVD).
        let t_a = ctx.tangents[0]?;
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => return None,
        };
        let a = ctx.fwd_map[&node.inputs[0]];
        Some(ctx.bwd.custom_op(LINALG_PINV_JVP, attrs, vec![a, t_a]))
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct PinvCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for PinvCpu {
    fn name(&self) -> &str {
        LINALG_PINV
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("pinv A")?;
        let out = output.expect_f64_mut("pinv out")?;
        // Output shape is [n, m]; recover m, n via known total = m·n
        // and the fact that sqrt is needed here is ambiguous. Use the
        // attrs-free convention: square root only works for square A.
        // Better: forward pass takes m, n from attrs… but we don't have
        // them. Recover from output length and input length:
        //   a.len() == m·n,  out.len() == n·m  (same number).
        // We need m and n separately. Pull from input shape via
        // OpExtension::infer_shape having already validated it; here
        // we re-derive from sizes: lacking shape access, encode in
        // attrs. v1 simplification: assume row-major contiguous and
        // recover m from output's leading dim later. For now, re-derive
        // by requiring the kernel be called only when the executor has
        // already wired inputs sized m·n. We use the approach: factor
        // out the ambiguity by passing m as the first 4 bytes of attrs.
        // (Done below via attrs.)
        // FALLBACK: scan factors of a.len() to find best (m,n) such
        // that m·n = a.len(); ambiguous. We instead require attrs.
        let mn = a.len();
        // Attrs encode m as little-endian u32 (n derived as mn/m).
        // Builder always sets this.
        let attrs = _attrs;
        if attrs.len() < 4 {
            return Err("pinv: attrs must encode m (u32 LE)".into());
        }
        let m = u32::from_le_bytes(attrs[..4].try_into().unwrap()) as usize;
        if m == 0 || mn % m != 0 {
            return Err(format!("pinv: bad attrs m={m} for input len {mn}"));
        }
        let n = mn / m;
        algos::pinv(a, m, n, out)
    }
}

