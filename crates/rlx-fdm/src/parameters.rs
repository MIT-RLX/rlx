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

//! Design parameters for constrained form-finding (jax_fdm `parameters`).

use crate::equilibrium::FdmError;
use crate::implicit::{
    AdjointSolveConfig, grad_loss_wrt_q, grad_loss_wrt_xyz_fixed_linear, solve_adjoint_columns,
};
use crate::loads::LoadState;
use crate::mesh::MeshStructure;
use crate::network::Network;
use crate::objective::{Goal, goals_grad_xyz_free};
use crate::state::EquilibriumState;
use crate::structure::Structure;

/// One optimizable degree of freedom (jax_fdm `EdgeForceDensityParameter`, `NodeSupportZ`, …).
#[derive(Clone, Debug)]
pub enum DesignParam {
    /// All edge force densities `q[e]`.
    AllEdgeQ { low: f64, up: f64 },
    /// Single edge `q[edge]`.
    EdgeQ { edge: usize, low: f64, up: f64 },
    /// Support node coordinate (`node` must be fixed).
    SupportCoord { node: usize, axis: usize },
    /// Nodal load on a free node (`px`/`py`/`pz`).
    FreeLoad { node: usize, axis: usize },
}

impl DesignParam {
    pub fn all_edge_q(low: f64, up: f64) -> Self {
        Self::AllEdgeQ { low, up }
    }
}

#[derive(Clone, Debug)]
enum Slot {
    EdgeQ {
        edge: usize,
        #[allow(dead_code)]
        low: f64,
        #[allow(dead_code)]
        up: f64,
    },
    SupportCoord {
        node: usize,
        axis: usize,
    },
    FreeLoad {
        node: usize,
        axis: usize,
    },
}

/// Packed design vector `x` with box bounds.
#[derive(Clone, Debug)]
pub struct DesignVector {
    pub x: Vec<f64>,
    pub low: Vec<f64>,
    pub up: Vec<f64>,
    slots: Vec<Slot>,
}

impl DesignVector {
    pub fn from_network_q(network: &Network) -> Self {
        Self::from_network(
            network,
            &[DesignParam::all_edge_q(f64::NEG_INFINITY, f64::INFINITY)],
        )
    }

    pub fn from_network(network: &Network, params: &[DesignParam]) -> Self {
        let mut x = Vec::new();
        let mut low = Vec::new();
        let mut up = Vec::new();
        let mut slots = Vec::new();
        for p in params {
            match p {
                DesignParam::AllEdgeQ { low: lo, up: hi } => {
                    for (e, &qi) in network.q.iter().enumerate() {
                        x.push(qi);
                        low.push(*lo);
                        up.push(*hi);
                        slots.push(Slot::EdgeQ {
                            edge: e,
                            low: *lo,
                            up: *hi,
                        });
                    }
                }
                DesignParam::EdgeQ {
                    edge,
                    low: lo,
                    up: hi,
                } => {
                    if *edge < network.q.len() {
                        x.push(network.q[*edge]);
                        low.push(*lo);
                        up.push(*hi);
                        slots.push(Slot::EdgeQ {
                            edge: *edge,
                            low: *lo,
                            up: *hi,
                        });
                    }
                }
                DesignParam::SupportCoord { node, axis } => {
                    if *node < network.xyz.len() / 3 && network.is_support[*node] && *axis < 3 {
                        x.push(network.xyz[*node * 3 + *axis]);
                        low.push(f64::NEG_INFINITY);
                        up.push(f64::INFINITY);
                        slots.push(Slot::SupportCoord {
                            node: *node,
                            axis: *axis,
                        });
                    }
                }
                DesignParam::FreeLoad { node, axis } => {
                    if *node < network.loads.len() / 3 && !network.is_support[*node] && *axis < 3 {
                        x.push(network.loads[*node * 3 + *axis]);
                        low.push(f64::NEG_INFINITY);
                        up.push(f64::INFINITY);
                        slots.push(Slot::FreeLoad {
                            node: *node,
                            axis: *axis,
                        });
                    }
                }
            }
        }
        Self { x, low, up, slots }
    }

    /// Apply trial vector `x` without mutating the stored design state.
    pub fn apply_x_to_network(&self, x: &[f64], network: &mut Network) {
        for (xi, slot) in x.iter().zip(self.slots.iter()) {
            match slot {
                Slot::EdgeQ { edge, .. } => network.q[*edge] = *xi,
                Slot::SupportCoord { node, axis } => {
                    network.xyz[*node * 3 + *axis] = *xi;
                }
                Slot::FreeLoad { node, axis } => {
                    network.loads[*node * 3 + *axis] = *xi;
                }
            }
        }
    }

    pub fn apply_to_network(&self, network: &mut Network) {
        self.apply_x_to_network(&self.x, network);
    }

    pub fn project(&mut self) {
        for (xi, (lo, hi)) in self.x.iter_mut().zip(self.low.iter().zip(self.up.iter())) {
            *xi = xi.clamp(*lo, *hi);
        }
    }

    /// Refresh packed `x` from a network after external projection (e.g. on full `q`).
    pub fn sync_from_network(&mut self, network: &Network) {
        for (xi, slot) in self.x.iter_mut().zip(self.slots.iter()) {
            match slot {
                Slot::EdgeQ { edge, .. } => *xi = network.q[*edge],
                Slot::SupportCoord { node, axis } => *xi = network.xyz[*node * 3 + *axis],
                Slot::FreeLoad { node, axis } => *xi = network.loads[*node * 3 + *axis],
            }
        }
    }

    pub fn add_q_l2_grad(&self, network: &Network, gx: &mut [f64], weight: f64) {
        if weight <= 0.0 {
            return;
        }
        for (gi, slot) in gx.iter_mut().zip(self.slots.iter()) {
            if let Slot::EdgeQ { edge, .. } = slot {
                *gi += weight * network.q[*edge];
            }
        }
    }

    pub fn gradient(
        &self,
        network: &Network,
        eq: &EquilibriumState,
        loss_grad_xyz_free: &[f64],
        structure: &Structure,
        load_state: &LoadState,
        xyz_fixed: &[f64],
        iter_config: &crate::iterative::IterativeConfig,
        mesh: Option<&MeshStructure>,
        fd_eps: f64,
        adjoint: &AdjointSolveConfig,
    ) -> Result<Vec<f64>, FdmError> {
        let xyz_free = crate::equilibrium::EquilibriumModel::pack_xyz_free(&eq.xyz, structure);
        let gq = grad_loss_wrt_q(
            &network.q,
            xyz_fixed,
            load_state,
            structure,
            &network.edges,
            &network.xyz,
            iter_config,
            mesh,
            &xyz_free,
            loss_grad_xyz_free,
            fd_eps,
        )?;
        let gxf = grad_loss_wrt_xyz_fixed_linear(
            &network.q,
            structure,
            &xyz_free,
            loss_grad_xyz_free,
            adjoint,
        )?;
        let lambda = solve_adjoint_columns(&network.q, structure, loss_grad_xyz_free, adjoint)?;

        Ok(self
            .slots
            .iter()
            .map(|slot| match slot {
                Slot::EdgeQ { edge, .. } => gq.dq.get(*edge).copied().unwrap_or(0.0),
                Slot::SupportCoord { node, axis } => {
                    support_packed_grad(&gxf, structure, *node, *axis)
                }
                Slot::FreeLoad { node, axis } => free_load_grad(&lambda, structure, *node, *axis),
            })
            .collect())
    }
}

fn support_packed_grad(
    gxf: &crate::implicit::XFixedGradient,
    structure: &Structure,
    node: usize,
    axis: usize,
) -> f64 {
    for (j, &n) in structure.indices_fixed.iter().enumerate() {
        if n == node {
            return gxf.dxf[j * 3 + axis];
        }
    }
    0.0
}

fn free_load_grad(lambda: &[f64], structure: &Structure, node: usize, axis: usize) -> f64 {
    for (a, &n) in structure.indices_free.iter().enumerate() {
        if n == node {
            return lambda[a * 3 + axis];
        }
    }
    0.0
}

pub fn loss_grad_xyz_free(
    goals: &[Goal],
    eq: &EquilibriumState,
    structure: &Structure,
    edges: &[(usize, usize)],
    is_support: &[bool],
    mesh: Option<&MeshStructure>,
) -> Vec<f64> {
    goals_grad_xyz_free(goals, eq, structure, edges, is_support, mesh)
}
