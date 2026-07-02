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

//! Sparse linear algebra for RLX — CSR LU, mat-vec, Conjugate Gradient.
//!
//! Downstream package modeled on `jax.experimental.sparse`. Registers
//! against rlx's custom-op scaffold without requiring any edits to the
//! framework crates. Three ops + a `SparseTensor` boundary abstraction.
//!
//! ## Usage
//!
//! ```ignore
//! // At application startup, once.
//! rlx_sparse::register();
//!
//! // Build graph as usual.
//! let mut g = Graph::new("photonics");
//! let v  = g.input("values",  Shape::new(&[nnz], DType::F64));
//! let ci = ...; // I32 col_idx (Op::Constant or Op::Input)
//! let rp = ...; // I32 row_ptr
//! let b  = g.input("b", Shape::new(&[n], DType::F64));
//!
//! let a = rlx_sparse::SparseTensor::from_csr(v, ci, rp, n, n);
//! let x = a.solve(&mut g, b);                 // direct LU
//! let y = a.mat_vec(&mut g, x);               // sparse matvec
//! let z = a.cg_solve(&mut g, b, 200, 1e-12);  // iterative CG
//! ```
//!
//! ## What's registered
//!
//! - `rlx_sparse.lu_solve` — direct LU via host LAPACK dgesv.
//!   v1 densifies CSR before solving; performance not yet sparse-fast,
//!   semantics are correct. Swapping for SuiteSparse UMFPACK or KLU is
//!   a kernel-body change with zero IR diff.
//! - `rlx_sparse.mat_vec` — `y = A·x` over CSR.
//! - `rlx_sparse.cg_solve` — Conjugate Gradient for SPD systems with
//!   `max_iter` + `tol` baked into the op's `attrs` blob.
//!
//! ## Adjoint convention (v1)
//!
//! All three ops assume `A` is symmetric. The closed-form adjoint
//! `dL/db = solve(Aᵀ, dL/dx)` reuses the same CSR triplet as the
//! forward call. Non-symmetric `A` requires an explicit transpose
//! triplet — sketch in the `vjp` body of each op. `dL/dvalues` is
//! non-differentiable in v1; it's `gather(-(dL/db) ⊗ x)` and slots
//! in as a separate gather op.
//!
//! ## Backend support
//!
//! | Backend | Status |
//! |---|---|
//! | CPU    | Full forward + autodiff. Real LAPACK. |
//! | Metal  | Trait surface only — full executor dispatch is a follow-up. |
//! | MLX    | Trait surface only — full executor dispatch is a follow-up. |
//! | Others | `Op::Custom` rejected at legalize; pin graph to `Device::Cpu`. |

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]

#![allow(unused_imports)]

use std::sync::Arc;

use rlx_ir::{DType, Graph, Node, NodeId, Op, OpExtension, Shape, VjpContext, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

// ── Op names (stable strings; downstream callers use these to look
//    up the registered op or build `Op::Custom` directly) ─────────

use super::*;

pub(crate) struct SparsePcgExt;


impl OpExtension for SparsePcgExt {
    fn name(&self) -> &str {
        SPARSE_PCG_SOLVE
    }
    fn num_inputs(&self) -> usize {
        4
    } // values, col_idx, row_ptr, b

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        inputs[3].clone()
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Closed-form adjoint identical to CG (same SPD assumption).
        // Recurse into pcg_solve with the same attrs.
        let vals_b = ctx.fwd_map[&node.inputs[0]];
        let cidx_b = ctx.fwd_map[&node.inputs[1]];
        let rptr_b = ctx.fwd_map[&node.inputs[2]];
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => Vec::new(),
        };
        let g_b = ctx.bwd.custom_op(
            SPARSE_PCG_SOLVE,
            attrs,
            vec![vals_b, cidx_b, rptr_b, ctx.upstream],
        );
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad = ctx.bwd.custom_op(
            SPARSE_VALUES_GRAD,
            Vec::new(),
            vec![cidx_b, rptr_b, g_b, y_fwd],
        );
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);
        vec![(0, g_vals), (3, g_b)]
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct SparsePcgCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparsePcgCpu {
    fn name(&self) -> &str {
        SPARSE_PCG_SOLVE
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("pcg values")?;
        let col_idx = inputs[1].expect_i32("pcg col_idx")?;
        let row_ptr = inputs[2].expect_i32("pcg row_ptr")?;
        let b = inputs[3].expect_f64("pcg b")?;
        let out = output.expect_f64_mut("pcg x")?;
        let (max_iter, tol) = decode_cg_attrs(attrs)?;
        algos::pcg_solve(values, col_idx, row_ptr, b, out, max_iter, tol)
    }
}

// ── BiCGSTAB (non-symmetric) ──────────────────────────────────────

