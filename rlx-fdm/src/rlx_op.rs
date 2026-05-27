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

//! RLX graph integration via `rlx_sparse` + FDM custom ops (`feature rlx-sparse`).

use std::sync::Arc;

use rlx_ir::{DType, Graph, Node, NodeId, Op, OpExtension, Shape, VjpContext, register_op};
use rlx_sparse::SparseTensor;

use crate::csr_spec::CsrAssemblySpec;

#[cfg(feature = "rlx-sparse")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

/// `values = assemble_csr(q)` for a fixed topology (attrs = [`CsrAssemblySpec::encode`]).
pub const FDM_ASSEMBLE_CSR: &str = "fdm.assemble_csr_values";

/// `dq = vjp(d_values)` for [`FDM_ASSEMBLE_CSR`].
pub const FDM_ASSEMBLE_CSR_VJP: &str = "fdm.assemble_csr_vjp";

/// Register `rlx_sparse` + FDM CSR assembly ops — call once before compiling FDM graphs.
pub fn register_rlx_sparse() {
    rlx_sparse::register();
    register_fdm_ops();
}

/// Register FDM custom ops (IR VJP + CPU kernels).
pub fn register_fdm_ops() {
    register_op(Arc::new(AssembleCsrExt));
    register_op(Arc::new(AssembleCsrVjpExt));
    register_cpu_kernel(Arc::new(AssembleCsrCpu));
    register_cpu_kernel(Arc::new(AssembleCsrVjpCpu));
}

/// CSR buffers for `rlx_sparse.pcg_solve`.
pub struct FdmCsr {
    pub values: Vec<f64>,
    pub col_idx: Vec<i32>,
    pub row_ptr: Vec<i32>,
    pub n: usize,
}

/// `x = K⁻¹ b` with Jacobi PCG (`rlx_sparse.pcg_solve`).
pub fn pcg_solve_graph(g: &mut Graph, csr: &FdmCsr, b: NodeId, max_iter: u32, tol: f64) -> NodeId {
    let v = const_f64(g, &csr.values);
    let ci = const_i32(g, &csr.col_idx);
    let rp = const_i32(g, &csr.row_ptr);
    let a = SparseTensor::from_csr(v, ci, rp, csr.n, csr.n);
    a.pcg_solve(g, b, max_iter, tol)
}

/// `csr_values = assemble(q)` for a fixed topology encoded in `spec`.
pub fn assemble_csr_values_graph(g: &mut Graph, q: NodeId, spec: &CsrAssemblySpec) -> NodeId {
    g.custom_op(FDM_ASSEMBLE_CSR, spec.encode(), vec![q])
}

struct AssembleCsrExt;

impl OpExtension for AssembleCsrExt {
    fn name(&self) -> &str {
        FDM_ASSEMBLE_CSR
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, _inputs: &[&Shape], attrs: &[u8]) -> Shape {
        let spec = CsrAssemblySpec::decode(attrs).expect("fdm.assemble_csr_values attrs");
        Shape::new(&[spec.nnz], DType::F64)
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        let attrs = match &node.op {
            Op::Custom { attrs, .. } => attrs.clone(),
            _ => Vec::new(),
        };
        let dq = ctx
            .bwd
            .custom_op(FDM_ASSEMBLE_CSR_VJP, attrs, vec![ctx.upstream]);
        vec![(0, dq)]
    }
}

struct AssembleCsrVjpExt;

impl OpExtension for AssembleCsrVjpExt {
    fn name(&self) -> &str {
        FDM_ASSEMBLE_CSR_VJP
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, _inputs: &[&Shape], attrs: &[u8]) -> Shape {
        let spec = CsrAssemblySpec::decode(attrs).expect("fdm.assemble_csr_vjp attrs");
        Shape::new(&[spec.num_edges], DType::F64)
    }
}

struct AssembleCsrCpu;

impl CpuKernel for AssembleCsrCpu {
    fn name(&self) -> &str {
        FDM_ASSEMBLE_CSR
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let q = inputs[0].expect_f64("assemble_csr q")?;
        let out = output.expect_f64_mut("assemble_csr values")?;
        let spec = CsrAssemblySpec::decode(attrs)?;
        let values = spec.assemble(q);
        if out.len() != values.len() {
            return Err(format!(
                "assemble_csr: out len {} != nnz {}",
                out.len(),
                values.len()
            ));
        }
        out.copy_from_slice(&values);
        Ok(())
    }
}

struct AssembleCsrVjpCpu;

impl CpuKernel for AssembleCsrVjpCpu {
    fn name(&self) -> &str {
        FDM_ASSEMBLE_CSR_VJP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let d_values = inputs[0].expect_f64("assemble_csr_vjp d_values")?;
        let out = output.expect_f64_mut("assemble_csr_vjp dq")?;
        let spec = CsrAssemblySpec::decode(attrs)?;
        let dq = spec.vjp(d_values, &[]);
        if out.len() != dq.len() {
            return Err(format!(
                "assemble_csr_vjp: out len {} != ne {}",
                out.len(),
                dq.len()
            ));
        }
        out.copy_from_slice(&dq);
        Ok(())
    }
}

fn const_f64(g: &mut Graph, xs: &[f64]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 8);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[xs.len()], DType::F64),
    )
}

fn const_i32(g: &mut Graph, xs: &[i32]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[xs.len()], DType::I32),
    )
}
