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

//! MIR / Session hooks for FDM equilibrium + `dL/dq` (jax_fdm + RLX optimizer path).

use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::equilibrium::{EquilibriumModel, FdmError};
use crate::graph::{FdmDenseGraph, fdm_dense_graph, pack_load_rhs, pack_stiffness};
use crate::implicit::{
    AdjointSolveConfig, QGradient, grad_loss_wrt_q, grad_loss_wrt_q_fixedpoint,
    grad_loss_wrt_q_linear_with_solver,
};
use crate::iterative::IterativeConfig;
use crate::network::Network;
use crate::objective::{Goal, goals_grad_xyz_free};
use crate::optimize::{OptimizeConfig, OptimizeResult, constrained_fdm};
use crate::reference::FdmOptions;
use crate::structure::Structure;

#[cfg(all(feature = "ir", feature = "rlx-sparse"))]
use crate::graph_sparse::{
    FdmSparsePcgGraph, PcgGraphConfig, fdm_sparse_pcg_graph, use_sparse_pcg_graph,
};

/// MIR equilibrium graph: dense `Op::DenseSolve` or CSR PCG (`rlx-sparse`).
#[derive(Clone, Debug)]
pub enum FdmEquilibriumGraph {
    Dense(FdmDenseGraph),
    #[cfg(all(feature = "ir", feature = "rlx-sparse"))]
    SparsePcg(FdmSparsePcgGraph),
}

impl FdmEquilibriumGraph {
    pub fn xyz_free(&self) -> NodeId {
        match self {
            Self::Dense(g) => g.xyz_free,
            #[cfg(all(feature = "ir", feature = "rlx-sparse"))]
            Self::SparsePcg(g) => g.xyz_free,
        }
    }
}

/// How [`FdmMirOptimizer`] computes `dL/dq`.
#[derive(Clone, Debug)]
pub enum FdmGradMode {
    /// Linear implicit adjoint (dense LU or PCG per [`FdmOptions::sparse`]).
    Linear,
    /// Nonlinear fixed-point adjoint for shape-dependent loads.
    FixedPoint(IterativeConfig),
    /// Central differences through [`grad_loss_wrt_q`].
    Auto { fd_eps: f64 },
}

impl Default for FdmGradMode {
    fn default() -> Self {
        Self::Auto { fd_eps: 1e-7 }
    }
}

/// Host-side optimizer that can drive MIR equilibrium solves + analytic gradients.
#[derive(Clone, Debug)]
pub struct FdmMirOptimizer {
    pub grad_mode: FdmGradMode,
    pub fdm: FdmOptions,
    /// Use [`graph_sparse::fdm_sparse_pcg_graph`] when `num_free ≥ sparse_graph_min_free`.
    pub sparse_graph_min_free: usize,
}

impl Default for FdmMirOptimizer {
    fn default() -> Self {
        Self {
            grad_mode: FdmGradMode::default(),
            fdm: FdmOptions::default(),
            sparse_graph_min_free: 32,
        }
    }
}

impl FdmMirOptimizer {
    /// Dense or sparse PCG equilibrium graph for the current topology.
    pub fn build_equilibrium_graph(
        &self,
        g: &mut Graph,
        network: &Network,
    ) -> Result<FdmEquilibriumGraph, String> {
        #[cfg(all(feature = "ir", feature = "rlx-sparse"))]
        {
            if self.fdm.sparse && use_sparse_pcg_graph(network, self.sparse_graph_min_free) {
                let mut pcg = PcgGraphConfig::default();
                pcg.max_iter = self.fdm.iterative.pcg_max_iter;
                pcg.tol = self.fdm.iterative.pcg_tol;
                return Ok(FdmEquilibriumGraph::SparsePcg(fdm_sparse_pcg_graph(
                    g, network, pcg,
                )?));
            }
        }
        Ok(FdmEquilibriumGraph::Dense(fdm_dense_graph(g, network)?))
    }

    /// Dense `K X = P` graph (always LU path).
    pub fn build_dense_equilibrium_graph(
        g: &mut Graph,
        network: &Network,
    ) -> Result<FdmDenseGraph, String> {
        fdm_dense_graph(g, network)
    }

    /// Pack host buffers for equilibrium graph params.
    pub fn pack_equilibrium_params(
        &self,
        network: &Network,
        graph: &FdmEquilibriumGraph,
    ) -> Result<(), String> {
        match graph {
            FdmEquilibriumGraph::Dense(_) => {
                let k = pack_stiffness(network)?;
                let p = pack_load_rhs(network)?;
                let _ = (k, p);
            }
            #[cfg(all(feature = "ir", feature = "rlx-sparse"))]
            FdmEquilibriumGraph::SparsePcg(_) => {
                let _ = pack_load_rhs(network)?;
            }
        }
        Ok(())
    }

    /// Set session params after [`Self::pack_equilibrium_params`] was used to build buffers.
    pub fn set_equilibrium_params(
        &self,
        compiled: &mut rlx_runtime::CompiledGraph,
        network: &Network,
        graph: &FdmEquilibriumGraph,
    ) -> Result<(), String> {
        match graph {
            FdmEquilibriumGraph::Dense(_) => {
                compiled.set_param_typed(
                    "fdm_K",
                    &f64_bytes(&pack_stiffness(network)?),
                    DType::F64,
                );
                compiled.set_param_typed("fdm_P", &f64_bytes(&pack_load_rhs(network)?), DType::F64);
            }
            #[cfg(all(feature = "ir", feature = "rlx-sparse"))]
            FdmEquilibriumGraph::SparsePcg(_) => {
                compiled.set_param_typed("fdm_q", &f64_bytes(&network.q), DType::F64);
                compiled.set_param_typed("fdm_P", &f64_bytes(&pack_load_rhs(network)?), DType::F64);
            }
        }
        Ok(())
    }

    fn adjoint_config(&self) -> AdjointSolveConfig {
        AdjointSolveConfig {
            use_sparse: self.fdm.sparse,
            pcg_max_iter: self.fdm.iterative.pcg_max_iter,
            pcg_tol: self.fdm.iterative.pcg_tol,
            sparse_min_free: self.sparse_graph_min_free,
        }
    }

    /// `dL/dq` given equilibrium positions and `∂L/∂x_f`.
    pub fn grad_loss_wrt_q(
        &self,
        network: &Network,
        loss_grad_xyz_free: &[f64],
        xyz_free: &[f64],
    ) -> Result<QGradient, FdmError> {
        let structure = Structure::from_network(network);
        let load_state = network.load_state();
        let xf = fixed_coords(network, &structure);
        let mesh = network.mesh_structure();
        let mut iter = self.fdm.iterative.clone();
        iter.use_sparse = self.fdm.sparse;

        match &self.grad_mode {
            FdmGradMode::Linear => grad_loss_wrt_q_linear_with_solver(
                &network.q,
                &xf,
                &structure,
                xyz_free,
                loss_grad_xyz_free,
                &self.adjoint_config(),
            ),
            FdmGradMode::FixedPoint(cfg) => grad_loss_wrt_q_fixedpoint(
                &network.q,
                &xf,
                &load_state,
                &structure,
                &network.edges,
                &network.xyz,
                cfg,
                mesh.as_ref(),
                loss_grad_xyz_free,
            ),
            FdmGradMode::Auto { fd_eps } => grad_loss_wrt_q(
                &network.q,
                &xf,
                &load_state,
                &structure,
                &network.edges,
                &network.xyz,
                &iter,
                mesh.as_ref(),
                xyz_free,
                loss_grad_xyz_free,
                *fd_eps,
            ),
        }
    }

    /// One constrained form-finding step using this optimizer's FDM + grad settings.
    pub fn constrained_fdm(
        &self,
        network: &Network,
        goals: &[Goal],
        constraints: &[crate::constraints::Constraint],
        mut config: OptimizeConfig,
    ) -> Result<OptimizeResult, FdmError> {
        config.fdm = self.fdm.clone();
        constrained_fdm(network, goals, constraints, &config)
    }
}

/// Signature metadata for a host-implemented `dL/dq` [`Op::CustomFn`].
#[derive(Clone, Debug)]
pub struct FdmGradQSignature {
    pub num_edges: usize,
    pub num_free: usize,
}

/// Build a placeholder [`Op::CustomFn`] for host-provided `dL/dq` (MIR optimizer hook).
///
/// Runtime should evaluate `dq` via [`FdmMirOptimizer::grad_loss_wrt_q`] and inject results;
/// the embedded graphs document the expected signature.
pub fn build_grad_q_custom_fn(
    g: &mut Graph,
    q: NodeId,
    loss_grad: NodeId,
    sig: &FdmGradQSignature,
) -> NodeId {
    let ne = sig.num_edges;
    let nf = sig.num_free;
    let loss_dim = nf * 3;

    let mut fwd = Graph::new("fdm_grad_q_fwd");
    let _q_in = fwd.input("q", Shape::new(&[ne], DType::F64));
    let _loss_grad_in = fwd.input("loss_grad", Shape::new(&[loss_dim], DType::F64));
    // Host fills `fdm_dq` via [`FdmMirOptimizer::set_grad_q_param`] before `Session::run`.
    let dq = fwd.param("fdm_dq", Shape::new(&[ne], DType::F64));
    fwd.set_outputs(vec![dq]);

    let mut vjp = Graph::new("fdm_grad_q_vjp");
    let _q_in = vjp.input("q", Shape::new(&[ne], DType::F64));
    let _loss_grad_in = vjp.input("loss_grad", Shape::new(&[loss_dim], DType::F64));
    let _primal_out = vjp.input("primal_output", Shape::new(&[ne], DType::F64));
    let _d_out = vjp.input("d_output", Shape::new(&[ne], DType::F64));
    let zero_q = vjp.add_node(
        Op::Constant {
            data: 0.0f64.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[ne], DType::F64),
    );
    let zero_l = vjp.add_node(
        Op::Constant {
            data: vec![0.0f64; loss_dim]
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        },
        vec![],
        Shape::new(&[loss_dim], DType::F64),
    );
    vjp.set_outputs(vec![zero_q, zero_l]);

    g.custom_fn(vec![q, loss_grad], fwd, Some(vjp), None)
}

/// Inject host-computed `dL/dq` into a graph built with [`build_grad_q_custom_fn`].
pub fn set_grad_q_param(
    optimizer: &FdmMirOptimizer,
    session: &mut rlx_runtime::CompiledGraph,
    network: &Network,
    loss_grad_xyz_free: &[f64],
    xyz_free: &[f64],
) -> Result<(), FdmError> {
    let gq = optimizer.grad_loss_wrt_q(network, loss_grad_xyz_free, xyz_free)?;
    let bytes: Vec<u8> = gq.dq.iter().flat_map(|v| v.to_le_bytes()).collect();
    session.set_param_typed("fdm_dq", &bytes, DType::F64);
    Ok(())
}

/// Goal gradient w.r.t. `q` in one call (equilibrium must be in `network` / `eq` coords).
pub fn goals_grad_wrt_q(
    optimizer: &FdmMirOptimizer,
    network: &Network,
    goals: &[Goal],
    loss_grad_xyz_free: Option<Vec<f64>>,
) -> Result<QGradient, FdmError> {
    let structure = Structure::from_network(network);
    let eq = crate::reference::fdm_with_options(network, &optimizer.fdm)?;
    let loss_grad = loss_grad_xyz_free.unwrap_or_else(|| {
        goals_grad_xyz_free(
            goals,
            &eq,
            &structure,
            &network.edges,
            &network.is_support,
            network.mesh_structure().as_ref(),
        )
    });
    let xyz_free = EquilibriumModel::pack_xyz_free(&eq.xyz, &structure);
    optimizer.grad_loss_wrt_q(network, &loss_grad, &xyz_free)
}

fn f64_bytes(xs: &[f64]) -> Vec<u8> {
    xs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn fixed_coords(network: &Network, structure: &Structure) -> Vec<f64> {
    let na = structure.num_fixed();
    let mut xf = vec![0.0; na * 3];
    for (j, &node) in structure.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xf[j * 3 + c] = network.xyz[node * 3 + c];
        }
    }
    xf
}
