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

//! MIR builders for differentiable FDM (jax_fdm + RLX `Op::DenseSolve`).

use rlx_ir::{DType, Graph, NodeId, Shape};

use crate::equilibrium::EquilibriumModel;
use crate::network::Network;
use crate::structure::Structure;

/// Pack `K` and `P` as graph parameters and emit `xyz_free = solve(K, P)`.
///
/// Design variables `q` can be wired later as param nodes feeding `K`/`P` rebuilds;
/// autodiff through `Op::DenseSolve` uses the implicit-function rule in `rlx-autodiff`.
#[derive(Clone, Debug)]
pub struct FdmDenseGraph {
    pub k: NodeId,
    pub p: NodeId,
    pub xyz_free: NodeId,
}

/// Build a static FDM graph for fixed `q` and anchor geometry (f64).
pub fn fdm_dense_graph(g: &mut Graph, network: &Network) -> Result<FdmDenseGraph, String> {
    network.validate()?;
    let structure = Structure::from_network(network);
    let nf = structure.num_free();
    if nf == 0 {
        return Err("no free nodes".into());
    }

    let k_mat = EquilibriumModel::stiffness_matrix(&network.q, &structure);
    let na = structure.num_fixed();
    let mut xyz_fixed = vec![0.0_f32; na * 3];
    for (j, &node) in structure.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xyz_fixed[j * 3 + c] = network.xyz[node * 3 + c] as f32;
        }
    }
    let p_mat = EquilibriumModel::load_matrix(
        &network.q,
        &xyz_fixed
            .iter()
            .map(|&x| x as f64)
            .collect::<Vec<_>>(),
        &network.loads,
        &structure,
    );

    let k_shape = Shape::new(&[nf, nf], DType::F64);
    let p_shape = Shape::new(&[nf, 3], DType::F64);
    let out_shape = Shape::new(&[nf, 3], DType::F64);

    let k = g.param(
        "fdm_K",
        k_shape,
    );
    let p = g.param("fdm_P", p_shape);
    let xyz_free = g.dense_solve(k, p, out_shape);

    // Caller must `set_param` with `k_mat` / `p_mat` flattened row-major.
    let _ = (k_mat, p_mat);

    Ok(FdmDenseGraph { k, p, xyz_free })
}

/// Static shape helper for `xyz_free` output.
pub fn xyz_free_shape(network: &Network) -> Result<Shape, String> {
    network.validate()?;
    let s = Structure::from_network(network);
    Ok(Shape::new(&[s.num_free(), 3], DType::F64))
}

/// Flatten equilibrium inputs for `Session::set_param`.
pub fn pack_stiffness(network: &Network) -> Result<Vec<f64>, String> {
    network.validate()?;
    let s = Structure::from_network(network);
    Ok(EquilibriumModel::stiffness_matrix(&network.q, &s))
}

pub fn pack_load_rhs(network: &Network) -> Result<Vec<f64>, String> {
    network.validate()?;
    let s = Structure::from_network(network);
    let na = s.num_fixed();
    let mut xyz_fixed = vec![0.0; na * 3];
    for (j, &node) in s.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xyz_fixed[j * 3 + c] = network.xyz[node * 3 + c];
        }
    }
    Ok(EquilibriumModel::load_matrix(
        &network.q,
        &xyz_fixed,
        &network.loads,
        &s,
    ))
}
