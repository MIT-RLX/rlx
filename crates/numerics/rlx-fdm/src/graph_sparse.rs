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

//! MIR builders using `rlx_sparse.pcg_solve` (`feature rlx-sparse`).

use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use rlx_sparse::SparseTensor;

use crate::csr_spec::CsrAssemblySpec;
use crate::network::Network;
use crate::rlx_op::assemble_csr_values_graph;
use crate::sparse::{export_csr, pattern_fast};
use crate::structure::Structure;

/// CSR stiffness + three PCG solves for `K x_c = P_c` (`c ∈ {x,y,z}`).
#[derive(Clone, Debug)]
pub struct FdmSparsePcgGraph {
    /// Force densities (design variable).
    pub q: NodeId,
    /// Assembled CSR values from `q` (`fdm.assemble_csr_values`).
    pub csr_values: NodeId,
    pub p: NodeId,
    pub xyz_free: NodeId,
    /// Fixed assembly pattern (for param upload / tests).
    pub csr_spec: CsrAssemblySpec,
}

/// PCG controls embedded in `rlx_sparse.pcg_solve` attrs.
#[derive(Clone, Copy, Debug)]
pub struct PcgGraphConfig {
    pub max_iter: u32,
    pub tol: f64,
}

impl Default for PcgGraphConfig {
    fn default() -> Self {
        Self {
            max_iter: 4000,
            tol: 1e-10,
        }
    }
}

/// Build `xyz_free = K(q)⁻¹ P` with Jacobi PCG on a fixed CSR pattern.
pub fn fdm_sparse_pcg_graph(
    g: &mut Graph,
    network: &Network,
    pcg: PcgGraphConfig,
) -> Result<FdmSparsePcgGraph, String> {
    network.validate()?;
    let structure = Structure::from_network(network);
    let nf = structure.num_free();
    let ne = structure.num_edges;
    if nf == 0 {
        return Err("no free nodes".into());
    }

    let csr_spec = CsrAssemblySpec::from_structure(&structure);
    let pat = pattern_fast(&structure);
    let (_values, col_idx, row_ptr, n) = export_csr(&pat, &network.q);
    if n != nf {
        return Err(format!("csr n={n} != num_free={nf}"));
    }

    let q = g.param("fdm_q", Shape::new(&[ne], DType::F64));
    let csr_values = assemble_csr_values_graph(g, q, &csr_spec);

    let ci_const = csr_i32_const(g, &col_idx);
    let rp_const = csr_i32_const(g, &row_ptr);
    let a = SparseTensor::from_csr(csr_values, ci_const, rp_const, nf, nf);

    let p_shape = Shape::new(&[nf, 3], DType::F64);
    let p = g.param("fdm_P", p_shape);

    let mut cols = Vec::with_capacity(3);
    for c in 0..3 {
        let col = g.narrow_(p, 1, c, 1);
        let rhs = g.reshape_(col, vec![nf as i64]);
        let x = a.pcg_solve(g, rhs, pcg.max_iter, pcg.tol);
        cols.push(g.reshape_(x, vec![nf as i64, 1]));
    }
    let xyz_free = g.concat_(cols, 1);

    Ok(FdmSparsePcgGraph {
        q,
        csr_values,
        p,
        xyz_free,
        csr_spec,
    })
}

/// Assemble CSR numeric values for current `q` (host helper / tests).
pub fn pack_csr_values(network: &Network) -> Result<Vec<f64>, String> {
    network.validate()?;
    let s = Structure::from_network(network);
    Ok(CsrAssemblySpec::from_structure(&s).assemble(&network.q))
}

/// Whether sparse PCG MIR is recommended for this network.
pub fn use_sparse_pcg_graph(network: &Network, min_free: usize) -> bool {
    Structure::from_network(network).num_free() >= min_free
}

fn csr_i32_const(g: &mut Graph, xs: &[i32]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        rlx_ir::Op::Constant { data: bytes },
        vec![],
        Shape::new(&[xs.len()], DType::I32),
    )
}
