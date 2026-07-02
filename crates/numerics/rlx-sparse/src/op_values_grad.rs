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

pub(crate) struct SparseValuesGradExt;


impl OpExtension for SparseValuesGradExt {
    fn name(&self) -> &str {
        SPARSE_VALUES_GRAD
    }
    fn num_inputs(&self) -> usize {
        4
    } // col_idx, row_ptr, u, v

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        // Output shape == col_idx shape (length nnz, F64).
        let col_idx = inputs[0];
        assert_eq!(
            col_idx.dtype(),
            DType::I32,
            "values_grad: col_idx must be I32"
        );
        let nnz = col_idx
            .num_elements()
            .expect("values_grad: col_idx must have static shape");
        Shape::new(&[nnz], DType::F64)
    }
    // Non-differentiable (it's itself a gradient kernel; second-order
    // derivatives are out of v1 scope).
}


#[cfg(feature = "cpu")]
pub(crate) struct SparseValuesGradCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparseValuesGradCpu {
    fn name(&self) -> &str {
        SPARSE_VALUES_GRAD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let col_idx = inputs[0].expect_i32("values_grad col_idx")?;
        let row_ptr = inputs[1].expect_i32("values_grad row_ptr")?;
        let u = inputs[2].expect_f64("values_grad u")?;
        let v = inputs[3].expect_f64("values_grad v")?;
        let out = output.expect_f64_mut("values_grad out")?;
        algos::values_grad(col_idx, row_ptr, u, v, out)
    }
}

// ── Sparse LU Solve (general / non-symmetric) ─────────────────────
//
// 7-input variant of `sparse_lu_solve`. Forward identity to the
// symmetric version (kernel only reads inputs 0..=3). The last 3
// inputs are the transpose CSR triplet used by the VJP for the
// adjoint solve `dL/db = solve(Aᵀ, dL/dx)`. Use this for non-
// symmetric matrices where reusing the forward triplet for the
// adjoint would be wrong.

