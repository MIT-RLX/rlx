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

//! End-to-end form-finding drivers (jax_fdm `equilibrium.fdm`).

use crate::equilibrium::{EquilibriumModel, FdmError};
use crate::iterative::IterativeConfig;
use crate::network::Network;
use crate::state::EquilibriumState;
use crate::structure::Structure;

/// Options mirroring jax_fdm `fdm(..., sparse=, tmax=, eta=)`.
#[derive(Clone, Debug)]
pub struct FdmOptions {
    /// Use CSR + PCG when `num_free ≥ 32`.
    pub sparse: bool,
    /// Iteration settings (`tmax=1` → single linear step).
    pub iterative: IterativeConfig,
}

impl Default for FdmOptions {
    fn default() -> Self {
        Self {
            sparse: false,
            iterative: IterativeConfig::linear(),
        }
    }
}

impl FdmOptions {
    /// jax_fdm-style nonlinear iteration with shape-dependent edge loads.
    pub fn nonlinear(tmax: u32, eta: f64, sparse: bool) -> Self {
        Self {
            sparse,
            iterative: IterativeConfig {
                tmax,
                eta,
                use_sparse: sparse,
                ..IterativeConfig::default()
            },
        }
    }
}

/// Compute static equilibrium with the force density method.
pub fn fdm(network: &Network) -> Result<EquilibriumState, FdmError> {
    fdm_with_options(network, &FdmOptions::default())
}

/// `fdm` with sparse / nonlinear controls.
pub fn fdm_with_options(
    network: &Network,
    options: &FdmOptions,
) -> Result<EquilibriumState, FdmError> {
    network.validate().map_err(FdmError::Validation)?;
    let structure = Structure::from_network(network);
    fdm_with_structure(network, &structure, options)
}

/// `fdm` reusing a pre-built [`Structure`] (topology fixed during optimization).
pub fn fdm_with_structure(
    network: &Network,
    structure: &Structure,
    options: &FdmOptions,
) -> Result<EquilibriumState, FdmError> {
    let load_state = network.load_state();
    let mut iterative = options.iterative.clone();
    iterative.use_sparse = options.sparse;
    let mesh = network.mesh_structure();
    EquilibriumModel::equilibrium_with_config(
        &network.q,
        &network.xyz,
        &load_state,
        structure,
        &network.edges,
        &iterative,
        mesh.as_ref(),
    )
}

/// Update a network's node coordinates from an equilibrium state.
pub fn apply_equilibrium(network: &mut Network, state: &EquilibriumState) {
    network.xyz.clone_from(&state.xyz);
}
