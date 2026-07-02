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

pub(crate) struct SparseLuGeneralExt;


impl OpExtension for SparseLuGeneralExt {
    fn name(&self) -> &str {
        SPARSE_LU_SOLVE_GENERAL
    }
    fn num_inputs(&self) -> usize {
        7
    }
    // values_A, col_idx_A, row_ptr_A, b, values_AT, col_idx_AT, row_ptr_AT
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        let b = inputs[3];
        b.clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // dL/db = solve(Aᵀ, dL/dy) — emit ANOTHER lu_solve_general
        // call with the roles swapped: the adjoint's forward matrix
        // is Aᵀ (already provided in inputs 4..7), so the adjoint
        // call's transpose triplet is back to A (inputs 0..3). The
        // outer sparse-LU-general's adjoint is itself a sparse-LU-
        // general — same recursion as in CG.
        let vals_a = ctx.fwd_map[&node.inputs[0]];
        let cidx_a = ctx.fwd_map[&node.inputs[1]];
        let rptr_a = ctx.fwd_map[&node.inputs[2]];
        let vals_at = ctx.fwd_map[&node.inputs[4]];
        let cidx_at = ctx.fwd_map[&node.inputs[5]];
        let rptr_at = ctx.fwd_map[&node.inputs[6]];

        let g_b = ctx.bwd.custom_op(
            SPARSE_LU_SOLVE_GENERAL,
            Vec::new(),
            // forward A is now Aᵀ; transpose for *this* adjoint solve is A.
            vec![
                vals_at,
                cidx_at,
                rptr_at,
                ctx.upstream,
                vals_a,
                cidx_a,
                rptr_a,
            ],
        );
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad = ctx.bwd.custom_op(
            SPARSE_VALUES_GRAD,
            Vec::new(),
            vec![cidx_a, rptr_a, g_b, y_fwd],
        );
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);
        vec![(0, g_vals), (3, g_b)]
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct SparseLuGeneralCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparseLuGeneralCpu {
    fn name(&self) -> &str {
        SPARSE_LU_SOLVE_GENERAL
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        // Forward only reads A; the AT triplet rides along for the
        // VJP and is unused here. Same algos::lu_solve as the
        // symmetric version.
        let values = inputs[0].expect_f64("lu_solve_general values")?;
        let col_idx = inputs[1].expect_i32("lu_solve_general col_idx")?;
        let row_ptr = inputs[2].expect_i32("lu_solve_general row_ptr")?;
        let b = inputs[3].expect_f64("lu_solve_general b")?;
        let out = output.expect_f64_mut("lu_solve_general out")?;
        algos::lu_solve(values, col_idx, row_ptr, b, out)
    }
}

// ── GMRES Solve (non-symmetric iterative) ────────────────────────
//
// Iterative analog of CG for non-symmetric A. Same 7-input shape as
// `sparse_lu_solve_general`. Attrs encode `(max_iter, tol)` exactly
// like CG.

