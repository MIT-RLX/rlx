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

//! Fused end-to-end form-finding: one compiled MIR equilibrium graph per topology,
//! host goals + implicit `dL/dq`. With `feature fuse`, MIR autodiff for simple losses.

use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::equilibrium::{EquilibriumModel, FdmError};
use crate::loads::nodes_load_at_mesh;
use crate::mir_opt::{FdmEquilibriumGraph, FdmMirOptimizer};
use crate::network::Network;
use crate::objective::Goal;
use crate::optimize::{constrained_fdm_host, OptimizeConfig, OptimizeResult};
use crate::state::EquilibriumState;
use crate::structure::Structure;

#[cfg(feature = "rlx-sparse")]
use crate::rlx_op::register_rlx_sparse;

/// Scalar loss wired into a fused MIR graph (for autodiff `dL/dq`, `feature fuse`).
#[derive(Clone, Debug)]
pub enum FusedMirLoss {
    /// `weight · (Σ z_free − target)²`.
    SumFreeZ { target: f64, weight: f64 },
}

/// Compiled equilibrium `q → x_f` reused across optimization steps.
pub struct FusedEquilibriumRunner {
    optimizer: FdmMirOptimizer,
    built: FdmEquilibriumGraph,
    session: CompiledGraph,
    structure: Structure,
    xyz_fixed: Vec<f64>,
}

impl FusedEquilibriumRunner {
    /// Build a fused forward graph when linear FDM + MIR are applicable.
    pub fn try_new(optimizer: &FdmMirOptimizer, network: &Network) -> Result<Option<Self>, FdmError> {
        if !can_fuse_equilibrium(optimizer, network) {
            return Ok(None);
        }
        #[cfg(feature = "rlx-sparse")]
        if optimizer.fdm.sparse {
            register_rlx_sparse();
        }
        let structure = Structure::from_network(network);
        let na = structure.num_fixed();
        let mut xyz_fixed = vec![0.0; na * 3];
        for (j, &node) in structure.indices_fixed.iter().enumerate() {
            for c in 0..3 {
                xyz_fixed[j * 3 + c] = network.xyz[node * 3 + c];
            }
        }
        let mut g = Graph::new("fdm_fused_equilibrium");
        let built = optimizer
            .build_equilibrium_graph(&mut g, network)
            .map_err(FdmError::Validation)?;
        g.set_outputs(vec![built.xyz_free()]);
        let session = Session::new(Device::Cpu).compile(g);
        Ok(Some(Self {
            optimizer: optimizer.clone(),
            built,
            session,
            structure,
            xyz_fixed,
        }))
    }

    pub fn solve(&mut self, network: &Network) -> Result<EquilibriumState, FdmError> {
        self.optimizer
            .set_equilibrium_params(&mut self.session, network, &self.built)
            .map_err(FdmError::Validation)?;
        let packed = &self.session.run_typed(&[])[0].0;
        let xyz_free = bytes_to_f64(packed);
        let xyz = EquilibriumModel::nodes_positions(&xyz_free, &self.xyz_fixed, &self.structure);
        let load_state = network.load_state();
        let loads = nodes_load_at_mesh(
            &xyz,
            &load_state,
            &self.structure,
            &network.edges,
            network.mesh_structure().as_ref(),
        );
        Ok(EquilibriumModel::equilibrium_state(
            &network.q,
            &xyz,
            &loads,
            &self.structure,
            &network.edges,
        ))
    }
}

/// Fused `q → loss` with MIR autodiff (`feature fuse`, sparse `fdm_q` graphs).
#[cfg(feature = "fuse")]
pub struct FusedAutodiffFormFinding {
    optimizer: FdmMirOptimizer,
    built: FdmEquilibriumGraph,
    fwd: CompiledGraph,
    bwd: CompiledGraph,
}

#[cfg(feature = "fuse")]
impl FusedAutodiffFormFinding {
    pub fn try_new(
        optimizer: &FdmMirOptimizer,
        network: &Network,
        loss: &FusedMirLoss,
    ) -> Result<Option<Self>, FdmError> {
        if !optimizer.fdm.sparse || !can_fuse_equilibrium(optimizer, network) {
            return Ok(None);
        }
        register_rlx_sparse();
        let s = Structure::from_network(network);
        let nf = s.num_free();
        let mut fwd_g = Graph::new("fdm_fused_opt_fwd");
        let built = optimizer
            .build_equilibrium_graph(&mut fwd_g, network)
            .map_err(FdmError::Validation)?;
        let xyz = built.xyz_free();
        let loss_node = emit_fused_loss(&mut fwd_g, xyz, nf, loss);
        fwd_g.set_outputs(vec![loss_node]);

        let q_node = find_param(&fwd_g, "fdm_q").ok_or_else(|| {
            FdmError::Validation("fused autodiff requires sparse graph with fdm_q".into())
        })?;

        let bwd_g = rlx_autodiff::grad_with_loss(&fwd_g, &[q_node]);
        let fwd = Session::new(Device::Cpu).compile(fwd_g);
        let bwd = Session::new(Device::Cpu).compile(bwd_g);

        Ok(Some(Self {
            optimizer: optimizer.clone(),
            built,
            fwd,
            bwd,
        }))
    }

    pub fn loss_and_grad_q(&mut self, network: &Network) -> Result<(f64, Vec<f64>), FdmError> {
        self.optimizer
            .set_equilibrium_params(&mut self.fwd, network, &self.built)
            .map_err(FdmError::Validation)?;
        self.optimizer
            .set_equilibrium_params(&mut self.bwd, network, &self.built)
            .map_err(FdmError::Validation)?;
        let loss = f64::from_le_bytes(self.fwd.run_typed(&[])[0].0[0..8].try_into().unwrap());
        let d_out = 1.0f64.to_le_bytes().to_vec();
        let outs = self.bwd.run_typed(&[("d_output", &d_out, DType::F64)]);
        let gq = bytes_to_f64(&outs[1].0);
        Ok((loss, gq))
    }
}

/// Run [`crate::optimize::constrained_fdm_host`] with a fused MIR equilibrium runner.
pub fn try_constrained_fdm_fused(
    network: &Network,
    goals: &[Goal],
    constraints: &[crate::constraints::Constraint],
    config: &OptimizeConfig,
) -> Result<Option<OptimizeResult>, FdmError> {
    let mir = FdmMirOptimizer {
        fdm: config.fdm.clone(),
        grad_mode: crate::mir_opt::FdmGradMode::Linear,
        sparse_graph_min_free: 8,
    };
    let Some(mut runner) = FusedEquilibriumRunner::try_new(&mir, network)? else {
        return Ok(None);
    };
    Ok(Some(constrained_fdm_host(
        network,
        goals,
        constraints,
        config,
        Some(runner),
    )?))
}

pub fn can_fuse_equilibrium(optimizer: &FdmMirOptimizer, network: &Network) -> bool {
    if optimizer.fdm.iterative.tmax > 1 {
        return false;
    }
    if network.load_state().has_shape_dependent() {
        return false;
    }
    true
}

fn emit_fused_loss(g: &mut Graph, xyz_free: NodeId, nf: usize, loss: &FusedMirLoss) -> NodeId {
    use rlx_ir::op::ReduceOp;
    match loss {
        FusedMirLoss::SumFreeZ { target, weight } => {
            let z_col = g.narrow_(xyz_free, 1, 2, 1);
            let z_vec = g.reshape_(z_col, vec![nf as i64]);
            let sum_z = g.reduce(z_vec, ReduceOp::Sum, vec![0], false, Shape::scalar(DType::F64));
            let target_c = f64_scalar(g, *target);
            let diff = g.sub(sum_z, target_c);
            let sq = g.mul(diff, diff);
            let w = f64_scalar(g, *weight);
            g.mul(sq, w)
        }
    }
}

fn f64_scalar(g: &mut Graph, v: f64) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::scalar(DType::F64),
    )
}

fn find_param(g: &Graph, name: &str) -> Option<NodeId> {
    for node in g.nodes() {
        if let Op::Param { name: n } = &node.op {
            if n == name {
                return Some(node.id);
            }
        }
    }
    None
}

fn bytes_to_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
