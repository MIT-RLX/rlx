// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `lsqr` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use std::sync::Arc;

use rlx_ir::{DType, Graph, Node, NodeId, Op, OpExtension, Shape, VjpContext, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

// ── Op names (stable strings; downstream callers use these to look
//    up the registered op or build `Op::Custom` directly) ─────────

use super::*;

pub(super) fn decode_lsqr_attrs(attrs: &[u8]) -> Result<(u32, f64, u32), String> {
    if attrs.len() < 16 {
        return Err(format!("lsqr: attrs len {} < 16", attrs.len()));
    }
    let max_iter = u32::from_le_bytes(attrs[0..4].try_into().unwrap());
    let tol = f64::from_le_bytes(attrs[4..12].try_into().unwrap());
    let n_cols = u32::from_le_bytes(attrs[12..16].try_into().unwrap());
    Ok((max_iter, tol, n_cols))
}

pub(crate) struct SparseLsqrExt;

impl OpExtension for SparseLsqrExt {
    fn name(&self) -> &str {
        SPARSE_LSQR_SOLVE
    }
    fn num_inputs(&self) -> usize {
        4
    } // values, col_idx, row_ptr, b
    fn infer_shape(&self, inputs: &[&Shape], attrs: &[u8]) -> Shape {
        let (_, _, n_cols) =
            decode_lsqr_attrs(attrs).expect("lsqr: attrs must encode (max_iter, tol, n_cols)");
        Shape::new(&[n_cols as usize], inputs[3].dtype())
    }
    // VJP deferred — see SPARSE_LSQR_SOLVE doc.
}

#[cfg(feature = "cpu")]
pub(crate) struct SparseLsqrCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparseLsqrCpu {
    fn name(&self) -> &str {
        SPARSE_LSQR_SOLVE
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("lsqr values")?;
        let col_idx = inputs[1].expect_i32("lsqr col_idx")?;
        let row_ptr = inputs[2].expect_i32("lsqr row_ptr")?;
        let b = inputs[3].expect_f64("lsqr b")?;
        let out = output.expect_f64_mut("lsqr x")?;
        let (max_iter, tol, n_cols) = decode_lsqr_attrs(attrs)?;
        algos::lsqr_solve(
            values,
            col_idx,
            row_ptr,
            b,
            out,
            max_iter,
            tol,
            n_cols as usize,
        )
    }
}

// ── Pure-Rust SpGEMM (CSR × CSR → CSR) ────────────────────────────
