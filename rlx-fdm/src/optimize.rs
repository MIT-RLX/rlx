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

//! Constrained form-finding (jax_fdm `constrained_fdm` + GD / L-BFGS).

use crate::constraints::{
    constraints_grad_xyz_free, constraints_have_nonlinear, constraints_penalty, Constraint,
};
use crate::equilibrium::{EquilibriumModel, FdmError};
use crate::implicit::AdjointSolveConfig;
use crate::lbfgs::Lbfgs;
use crate::loads::LoadState;
use crate::losses::{Loss, losses_total};
use crate::mesh::MeshStructure;
use crate::network::Network;
use crate::objective::{goals_report, Goal, GoalReport};
use crate::parameters::{loss_grad_xyz_free, DesignParam, DesignVector};
use crate::reference::{apply_equilibrium, fdm_with_structure, FdmOptions};
use crate::slsqp::Slsqp;
use crate::state::EquilibriumState;
use crate::structure::Structure;

/// Optimizer for [`constrained_fdm`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerKind {
    /// Projected gradient descent with fixed learning rate.
    Gd,
    /// Limited-memory BFGS with backtracking line search (box projection on `q`).
    Lbfgs,
    /// Penalty + L-BFGS for nonlinear inequalities (jax_fdm `SLSQP` subset).
    Slsqp,
    /// Alias for [`Self::Slsqp`] (external IPOPT not linked; use penalty SLSQP).
    Ipopt,
}

/// Gradient-descent / L-BFGS settings for [`constrained_fdm`].
#[derive(Clone, Debug)]
pub struct OptimizeConfig {
    pub optimizer: OptimizerKind,
    pub learning_rate: f64,
    pub max_iter: usize,
    pub loss_tol: f64,
    pub grad_tol: f64,
    pub fd_eps: f64,
    /// L2 penalty `½ · weight · ‖q‖²` on force densities (jax_fdm regularizer subset).
    pub q_l2_weight: f64,
    pub lbfgs_history: usize,
    pub fdm: FdmOptions,
    pub verbose: bool,
    /// Reuse a compiled MIR equilibrium graph each iteration (`feature ir`).
    pub fuse_mir: bool,
    /// Design variables (default: all edge `q`).
    pub parameters: Vec<DesignParam>,
    /// jax_fdm-style loss collections (default: goals as squared error).
    pub losses: Vec<Loss>,
    /// Penalty weight for nonlinear constraints in [`OptimizerKind::Slsqp`].
    pub slsqp_penalty_weight: f64,
    /// Match jax_fdm: use dense solver when constraints are present.
    pub force_dense_with_constraints: bool,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            optimizer: OptimizerKind::Gd,
            learning_rate: 0.05,
            max_iter: 200,
            loss_tol: 1e-6,
            grad_tol: 1e-5,
            fd_eps: 1e-7,
            q_l2_weight: 0.0,
            lbfgs_history: 10,
            fdm: FdmOptions::default(),
            verbose: false,
            fuse_mir: false,
            parameters: Vec::new(),
            losses: Vec::new(),
            slsqp_penalty_weight: 50.0,
            force_dense_with_constraints: true,
        }
    }
}

impl OptimizeConfig {
    /// Arch / loadpath form-finding: L-BFGS, linear equilibrium, CSR+PCG when large enough.
    pub fn arch_form_finding() -> Self {
        Self {
            optimizer: OptimizerKind::Lbfgs,
            learning_rate: 1.0,
            max_iter: 120,
            loss_tol: 1e-6,
            grad_tol: 1e-5,
            lbfgs_history: 12,
            fdm: FdmOptions {
                sparse: true,
                iterative: crate::iterative::IterativeConfig::linear(),
            },
            verbose: false,
            fuse_mir: true,
            ..Default::default()
        }
    }
}

/// Result of [`constrained_fdm`].
#[derive(Clone, Debug)]
pub struct OptimizeResult {
    pub network: Network,
    pub equilibrium: EquilibriumState,
    pub loss: f64,
    pub goal_loss: f64,
    pub penalty: f64,
    pub iterations: usize,
    pub loss_history: Vec<f64>,
    pub goal_reports: Vec<GoalReport>,
}

/// Cached topology and load state for repeated equilibrium + adjoint solves.
pub struct OptWorkspace {
    pub structure: Structure,
    pub load_state: LoadState,
    pub mesh: Option<MeshStructure>,
    pub xyz_fixed: Vec<f64>,
    pub iter_config: crate::iterative::IterativeConfig,
}

impl OptWorkspace {
    pub fn from_network(network: &Network, fdm: &FdmOptions) -> Self {
        let structure = Structure::from_network(network);
        let load_state = network.load_state();
        let mesh = network.mesh_structure();
        let na = structure.num_fixed();
        let mut xyz_fixed = vec![0.0; na * 3];
        for (j, &node) in structure.indices_fixed.iter().enumerate() {
            for c in 0..3 {
                xyz_fixed[j * 3 + c] = network.xyz[node * 3 + c];
            }
        }
        let mut iter_config = fdm.iterative.clone();
        iter_config.use_sparse = fdm.sparse;
        Self {
            structure,
            load_state,
            mesh,
            xyz_fixed,
            iter_config,
        }
    }

    pub fn solve(&self, network: &Network, fdm: &FdmOptions) -> Result<EquilibriumState, FdmError> {
        fdm_with_structure(network, &self.structure, fdm)
    }

    /// Equilibrium at trial `q` with fixed supports / anchor geometry from `network`.
    pub fn solve_q(&self, q: &[f64], network: &Network, fdm: &FdmOptions) -> Result<EquilibriumState, FdmError> {
        let mut iterative = fdm.iterative.clone();
        iterative.use_sparse = fdm.sparse;
        EquilibriumModel::equilibrium_with_config(
            q,
            &network.xyz,
            &self.load_state,
            &self.structure,
            &network.edges,
            &iterative,
            self.mesh.as_ref(),
        )
    }

    pub fn pack_xyz_free(&self, eq: &EquilibriumState) -> Vec<f64> {
        EquilibriumModel::pack_xyz_free(&eq.xyz, &self.structure)
    }


}

/// Constrained inverse form-finding: minimize goals subject to soft/hard constraints on `q`.
///
/// Design variables are edge force densities `q` (jax_fdm `EdgeForceDensityParameter`).
/// With `feature ir` and [`OptimizeConfig::fuse_mir`], reuses a compiled equilibrium graph
/// (see [`crate::fuse::FusedEquilibriumRunner`]).
pub fn constrained_fdm(
    network: &Network,
    goals: &[Goal],
    constraints: &[Constraint],
    config: &OptimizeConfig,
) -> Result<OptimizeResult, FdmError> {
    #[cfg(feature = "ir")]
    if config.fuse_mir {
        if let Some(result) = crate::fuse::try_constrained_fdm_fused(network, goals, constraints, config)? {
            return Ok(result);
        }
    }
    #[cfg(feature = "ir")]
    {
        constrained_fdm_host(network, goals, constraints, config, None)
    }
    #[cfg(not(feature = "ir"))]
    {
        constrained_fdm_host(network, goals, constraints, config, None)
    }
}

/// Host optimization loop; optional fused MIR equilibrium runner (`feature ir`).
pub(crate) fn constrained_fdm_host(
    network: &Network,
    goals: &[Goal],
    constraints: &[Constraint],
    config: &OptimizeConfig,
    #[cfg(feature = "ir")] mut fused_runner: Option<crate::fuse::FusedEquilibriumRunner>,
    #[cfg(not(feature = "ir"))] mut fused_runner: Option<()>,
) -> Result<OptimizeResult, FdmError> {
    network.validate().map_err(FdmError::Validation)?;
    let mut net = network.clone();
    let fdm = effective_fdm_options(config, constraints);
    let ws = OptWorkspace::from_network(&net, &fdm);
    let mesh = ws.mesh.as_ref();
    let loss_terms = effective_losses(config, goals);
    let params = effective_parameters(config);
    let mut design = DesignVector::from_network(&net, &params);
    let adjoint = AdjointSolveConfig::from(&fdm.iterative).with_sparse(fdm.sparse);

    let mut loss_history = Vec::new();
    if config.verbose {
        eprintln!(
            "constrained_fdm: {} loss terms, {} constraints, opt={:?} n_x={} sparse={}",
            loss_terms.len(),
            constraints.len(),
            config.optimizer,
            design.x.len(),
            fdm.sparse
        );
        if config.force_dense_with_constraints && constraints_have_nonlinear(constraints) && config.fdm.sparse {
            eprintln!("  note: jax_fdm uses dense FDM when constraints are active");
        }
    }

    let mut eq = solve_equilibrium(&ws, &net, config, &fdm, &mut fused_runner)?;
    let mut total = eval_total(&loss_terms, constraints, &eq, &ws.structure, &net, mesh, config);
    loss_history.push(total);

    let mut lbfgs = Lbfgs::default();
    lbfgs.history = config.lbfgs_history;
    let mut slsqp = Slsqp {
        penalty_weight: config.slsqp_penalty_weight,
        fd_eps: config.fd_eps,
        lbfgs: lbfgs.clone(),
    };

    for iter in 0..config.max_iter {
        let goal_loss = losses_total(&loss_terms, &eq, &ws.structure, &net.is_support, mesh);
        let penalty = constraints_penalty(constraints, &eq, &net.q);
        total = goal_loss + penalty + q_l2_loss(&net.q, config.q_l2_weight);

        let mut loss_grad = loss_grad_xyz_free(&[], &eq, &ws.structure, &net.edges, &net.is_support, mesh);
        for loss in &loss_terms {
            let lg = loss_grad_xyz_free(
                &loss.goals,
                &eq,
                &ws.structure,
                &net.edges,
                &net.is_support,
                mesh,
            );
            for (a, b) in loss_grad.iter_mut().zip(lg.iter()) {
                *a += *b;
            }
        }
        let cg = constraints_grad_xyz_free(constraints, &eq, &ws.structure, &net.edges);
        for (a, b) in loss_grad.iter_mut().zip(cg.iter()) {
            *a += *b;
        }

        let mut gx = design.gradient(
            &net,
            &eq,
            &loss_grad,
            &ws.structure,
            &ws.load_state,
            &ws.xyz_fixed,
            &ws.iter_config,
            mesh,
            config.fd_eps,
            &adjoint,
        )?;
        design.add_q_l2_grad(&net, &mut gx, config.q_l2_weight);
        let grad_norm: f64 = gx.iter().map(|x| x * x).sum::<f64>().sqrt();
        if config.verbose && (iter % 10 == 0 || iter + 1 == config.max_iter) {
            eprintln!(
                "  iter {iter:4}  loss={total:.6}  goals={goal_loss:.6}  pen={penalty:.6}  |grad|={grad_norm:.4}"
            );
        }
        if total < config.loss_tol || grad_norm < config.grad_tol {
            return Ok(finish(
                net,
                eq,
                total,
                goal_loss,
                penalty,
                iter + 1,
                loss_history,
                &loss_terms,
                &ws.structure,
                mesh,
            ));
        }

        let x_prev = design.x.clone();
        let g_prev = gx.clone();
        let opt = config.optimizer;
        match opt {
            OptimizerKind::Gd => {
                for (xi, gi) in design.x.iter_mut().zip(gx.iter()) {
                    *xi -= config.learning_rate * gi;
                }
            }
            OptimizerKind::Lbfgs => {
                let dir = lbfgs.direction(&gx);
                let (x_new, _) = lbfgs.line_search(&design.x, &dir, total, &gx, |x| {
                    eval_objective_at(
                        x,
                        &design,
                        &ws,
                        &net,
                        &loss_terms,
                        constraints,
                        config,
                        &fdm,
                        mesh,
                    )
                });
                design.x.copy_from_slice(&x_new);
            }
            OptimizerKind::Slsqp | OptimizerKind::Ipopt => {
                let low = design.low.clone();
                let up = design.up.clone();
                let mut x_step = design.x.clone();
                let nonlinear = |x: &[f64]| {
                    let mut trial = net.clone();
                    apply_design(&design, x, &mut trial);
                    let teq = ws.solve(&trial, &fdm).expect("trial eq");
                    crate::constraints::nonlinear_ineq_values(constraints, &teq, mesh, &trial.edges)
                };
                slsqp.step(
                    &mut x_step,
                    &low,
                    &up,
                    |x| {
                        eval_objective_at(
                            x,
                            &design,
                            &ws,
                            &net,
                            &loss_terms,
                            constraints,
                            config,
                            &fdm,
                            mesh,
                        )
                    },
                    |x, g| {
                        let mut trial = net.clone();
                        apply_design(&design, x, &mut trial);
                        let teq = ws.solve(&trial, &fdm).expect("eq grad");
                        let lg = combined_loss_grad(
                            &loss_terms,
                            constraints,
                            &teq,
                            &ws,
                            &trial,
                            mesh,
                        );
                        let sub = design
                            .gradient(
                                &trial,
                                &teq,
                                &lg,
                                &ws.structure,
                                &ws.load_state,
                                &ws.xyz_fixed,
                                &ws.iter_config,
                                mesh,
                                config.fd_eps,
                                &adjoint,
                            )
                            .expect("grad");
                        g.copy_from_slice(&sub);
                        design.add_q_l2_grad(&trial, g, config.q_l2_weight);
                    },
                    nonlinear,
                );
                design.x.copy_from_slice(&x_step);
            }
        }
        design.project();
        project_q_constraints(constraints, &mut design, &mut net);

        eq = solve_equilibrium(&ws, &net, config, &fdm, &mut fused_runner)?;
        total = eval_total(&loss_terms, constraints, &eq, &ws.structure, &net, mesh, config);
        loss_history.push(total);

        if matches!(opt, OptimizerKind::Lbfgs) {
            let lg = combined_loss_grad(&loss_terms, constraints, &eq, &ws, &net, mesh);
            if let Ok(mut g_new) = design.gradient(
                &net,
                &eq,
                &lg,
                &ws.structure,
                &ws.load_state,
                &ws.xyz_fixed,
                &ws.iter_config,
                mesh,
                config.fd_eps,
                &adjoint,
            ) {
                design.add_q_l2_grad(&net, &mut g_new, config.q_l2_weight);
                lbfgs.update(&x_prev, &g_prev, &design.x, &g_new);
            }
        }
    }

    let goal_loss = losses_total(&loss_terms, &eq, &ws.structure, &net.is_support, mesh);
    let penalty = constraints_penalty(constraints, &eq, &net.q);
    let l2 = q_l2_loss(&net.q, config.q_l2_weight);
    Ok(finish(
        net,
        eq,
        goal_loss + penalty + l2,
        goal_loss,
        penalty,
        config.max_iter,
        loss_history,
        &loss_terms,
        &ws.structure,
        mesh,
    ))
}

fn q_l2_loss(q: &[f64], weight: f64) -> f64 {
    if weight <= 0.0 {
        return 0.0;
    }
    0.5 * weight * q.iter().map(|&x| x * x).sum::<f64>()
}

fn project_q(q: &mut [f64], constraints: &[Constraint]) {
    for c in constraints {
        c.project_q(q);
    }
    for qi in q.iter_mut() {
        if qi.abs() < 1e-4 {
            *qi = qi.signum().max(-1.0).min(1.0) * 1e-4;
        }
    }
}

fn solve_equilibrium(
    ws: &OptWorkspace,
    net: &Network,
    _config: &OptimizeConfig,
    fdm: &FdmOptions,
    #[cfg(feature = "ir")] fused_runner: &mut Option<crate::fuse::FusedEquilibriumRunner>,
    #[cfg(not(feature = "ir"))] _fused_runner: &mut Option<()>,
) -> Result<EquilibriumState, FdmError> {
    #[cfg(feature = "ir")]
    if let Some(runner) = fused_runner.as_mut() {
        return runner.solve(net);
    }
    ws.solve(net, fdm)
}

fn effective_fdm_options(config: &OptimizeConfig, constraints: &[Constraint]) -> FdmOptions {
    let mut fdm = config.fdm.clone();
    if config.force_dense_with_constraints && !constraints.is_empty() {
        fdm.sparse = false;
        fdm.iterative.use_sparse = false;
    }
    fdm
}

fn effective_parameters(config: &OptimizeConfig) -> Vec<DesignParam> {
    if config.parameters.is_empty() {
        vec![DesignParam::all_edge_q(f64::NEG_INFINITY, f64::INFINITY)]
    } else {
        config.parameters.clone()
    }
}

fn effective_losses(config: &OptimizeConfig, goals: &[Goal]) -> Vec<Loss> {
    if !config.losses.is_empty() {
        config.losses.clone()
    } else if goals.is_empty() {
        Vec::new()
    } else {
        vec![Loss::new(goals.to_vec())]
    }
}

fn eval_total(
    losses: &[Loss],
    constraints: &[Constraint],
    eq: &EquilibriumState,
    structure: &Structure,
    net: &Network,
    mesh: Option<&MeshStructure>,
    config: &OptimizeConfig,
) -> f64 {
    losses_total(losses, eq, structure, &net.is_support, mesh)
        + constraints_penalty(constraints, eq, &net.q)
        + q_l2_loss(&net.q, config.q_l2_weight)
}

fn eval_objective_at(
    x: &[f64],
    design: &DesignVector,
    ws: &OptWorkspace,
    net: &Network,
    losses: &[Loss],
    constraints: &[Constraint],
    config: &OptimizeConfig,
    fdm: &FdmOptions,
    mesh: Option<&MeshStructure>,
) -> f64 {
    let mut trial = net.clone();
    design.apply_x_to_network(x, &mut trial);
    let teq = ws.solve(&trial, fdm).expect("trial equilibrium");
    eval_total(losses, constraints, &teq, &ws.structure, &trial, mesh, config)
}

fn apply_design(design: &DesignVector, x: &[f64], net: &mut Network) {
    design.apply_x_to_network(x, net);
}

fn combined_loss_grad(
    losses: &[Loss],
    constraints: &[Constraint],
    eq: &EquilibriumState,
    ws: &OptWorkspace,
    net: &Network,
    mesh: Option<&MeshStructure>,
) -> Vec<f64> {
    let mut loss_grad =
        loss_grad_xyz_free(&[], eq, &ws.structure, &net.edges, &net.is_support, mesh);
    for loss in losses {
        let lg = loss_grad_xyz_free(
            &loss.goals,
            eq,
            &ws.structure,
            &net.edges,
            &net.is_support,
            mesh,
        );
        for (a, b) in loss_grad.iter_mut().zip(lg.iter()) {
            *a += *b;
        }
    }
    let cg = constraints_grad_xyz_free(constraints, eq, &ws.structure, &net.edges);
    for (a, b) in loss_grad.iter_mut().zip(cg.iter()) {
        *a += *b;
    }
    loss_grad
}

fn project_q_constraints(
    constraints: &[Constraint],
    design: &mut DesignVector,
    net: &mut Network,
) {
    design.apply_to_network(net);
    project_q(&mut net.q, constraints);
    design.sync_from_network(net);
}

fn finish(
    mut network: Network,
    eq: EquilibriumState,
    loss: f64,
    goal_loss: f64,
    penalty: f64,
    iterations: usize,
    loss_history: Vec<f64>,
    losses: &[Loss],
    structure: &Structure,
    mesh: Option<&MeshStructure>,
) -> OptimizeResult {
    apply_equilibrium(&mut network, &eq);
    let goals: Vec<Goal> = losses.iter().flat_map(|l| l.goals.clone()).collect();
    OptimizeResult {
        goal_reports: goals_report(&goals, &eq, structure, &network.is_support, mesh),
        network,
        equilibrium: eq,
        loss,
        goal_loss,
        penalty,
        iterations,
        loss_history,
    }
}
