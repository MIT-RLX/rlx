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

//! Force density method (FDM) for pin-jointed structures.
//!
//! Ported from [jax_fdm](https://github.com/arpastrana/jax_fdm): connectivity
//! incidence, stiffness `K = C_fᵀ diag(q) C_f`, load `P = L_f − C_fᵀ diag(q) C_a X_a`,
//! solve `K X_f = P`, then post-process lengths, forces, and residuals.
//!
//! ## Modules
//!
//! - [`network`] — topology, supports, force densities, nodal loads
//! - [`structure`] — incidence matrix and free/fixed indexing
//! - [`equilibrium`] — FDM linear algebra (jax_fdm `EquilibriumModel`)
//! - [`sparse`] — CSR stiffness + PCG (`EquilibriumModelSparse`)
//! - [`iterative`] — fixed-point nonlinear iteration (`solver_forward`)
//! - [`loads`] — shape-dependent edge / nodal loads
//! - [`reference`] — end-to-end `fdm()` / `fdm_with_options`
//! - [`goals`] — scalar objectives for inverse form-finding
//! - [`objective`] — weighted goals (`NetworkLoadpath`, `EdgeLength`)
//! - [`constraints`] — soft penalties + `q` projection
//! - [`optimize`] — [`constrained_fdm`] (GD / L-BFGS)
//! - [`lbfgs`] — limited-memory BFGS line search
//! - [`implicit`] — adjoint `dL/dq` (dense or PCG), `dL/dX_fixed`
//! - [`io`] — JSON interchange (jax_fdm `data/json`)

pub mod constraints;
pub mod equilibrium;
pub mod geometry;
pub mod goals;
pub mod implicit;
pub mod iterative;
pub mod lbfgs;
pub mod loads;
pub mod losses;
pub mod mesh;
pub mod network;
pub mod objective;
pub mod optimize;
pub mod parameters;
pub mod slsqp;
pub mod reference;
pub mod solve;
pub mod sparse;
pub mod sparse_fast;
pub mod state;
pub mod structure;

#[cfg(feature = "rlx-sparse")]
pub mod csr_spec;
#[cfg(feature = "rlx-sparse")]
pub mod rlx_op;

#[cfg(feature = "io")]
pub mod io;

#[cfg(feature = "ir")]
pub mod graph;
#[cfg(all(feature = "ir", feature = "rlx-sparse"))]
pub mod graph_sparse;
#[cfg(feature = "ir")]
pub mod mir_opt;
#[cfg(feature = "ir")]
pub mod fuse;

pub use equilibrium::{EquilibriumModel, FdmError};
pub use implicit::{
    accumulate_dq_from_lambda, accumulate_dxf_from_lambda, grad_loss_wrt_q, grad_loss_wrt_q_fd,
    grad_loss_wrt_q_fixedpoint, grad_loss_wrt_q_linear, grad_loss_wrt_q_linear_with_solver,
    grad_loss_wrt_xyz_fixed_linear, AdjointSolveConfig, QGradient, XFixedGradient,
};
pub use loads::{
    transpose_edge_loads_jacobian, transpose_face_loads_jacobian,
    transpose_face_loads_jacobian_fd,
};
pub use constraints::{
    constraints_grad_xyz_free, constraints_have_nonlinear, constraints_penalty,
    nonlinear_ineq_values, Constraint,
};
pub use losses::{losses_total, ErrorKind, Loss};
pub use parameters::{loss_grad_xyz_free, DesignParam, DesignVector};
pub use slsqp::Slsqp;
pub use goals::{
    edge_length_error, grad_edge_length_error_wrt_xyz_free, grad_mean_edge_length_wrt_xyz_free,
    grad_residual_wrt_xyz_free, mean_edge_length, network_loadpath, total_loadpath_proxy,
};
pub use objective::{
    goals_grad_xyz_free, goals_loss, goals_loss_with_structure, goals_report, CoordAxis, Goal,
    GoalReport,
};
pub use lbfgs::Lbfgs;
pub use optimize::{
    constrained_fdm, OptWorkspace, OptimizeConfig, OptimizeResult, OptimizerKind,
};
#[cfg(feature = "ir")]
pub use fuse::{
    can_fuse_equilibrium, try_constrained_fdm_fused, FusedEquilibriumRunner, FusedMirLoss,
};
#[cfg(feature = "fuse")]
pub use fuse::FusedAutodiffFormFinding;
pub use iterative::{
    config_for_implicit_adjoint, equilibrium_iterative, equilibrium_iterative_trajectory,
    IterativeConfig,
};
pub use loads::LoadState;
pub use mesh::{MeshStructure, edges_from_faces};
pub use network::Network;
pub use reference::{apply_equilibrium, fdm, fdm_with_options, FdmOptions};
pub use sparse::{SparseStiffness, pattern_fast};
pub use sparse_fast::SparseStiffnessFast;
pub use state::EquilibriumState;
pub use structure::Structure;

#[cfg(feature = "rlx-sparse")]
pub use csr_spec::CsrAssemblySpec;
#[cfg(feature = "rlx-sparse")]
pub use rlx_op::{
    assemble_csr_values_graph, register_fdm_ops, register_rlx_sparse, FdmCsr, pcg_solve_graph,
};
#[cfg(all(feature = "ir", feature = "rlx-sparse"))]
pub use graph_sparse::{
    fdm_sparse_pcg_graph, pack_csr_values, use_sparse_pcg_graph, FdmSparsePcgGraph,
    PcgGraphConfig,
};

#[cfg(feature = "io")]
pub use io::{
    from_json_path, from_json_str, merge_mesh, mesh_from_json_path, mesh_from_json_str,
    to_json_path, to_json_str, MeshDocument,
};

#[cfg(feature = "ir")]
pub use mir_opt::{
    build_grad_q_custom_fn, goals_grad_wrt_q, set_grad_q_param, FdmEquilibriumGraph, FdmGradMode,
    FdmGradQSignature,
    FdmMirOptimizer,
};
