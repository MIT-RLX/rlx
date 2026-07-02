#![cfg(feature = "ir")]
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

use rlx_fdm::mir_opt::{
    FdmEquilibriumGraph, FdmGradMode, FdmGradQSignature, FdmMirOptimizer, build_grad_q_custom_fn,
    goals_grad_wrt_q, set_grad_q_param,
};
use rlx_fdm::{Goal, Network, fdm, goals_grad_xyz_free};
use rlx_runtime::{Device, Session};

#[test]
fn mir_optimizer_grad_nonzero() {
    let net = Network::arch_chain(4.0, 6, -1.0, -0.15);
    let goals = vec![Goal::min_free_z(-0.05, 1.0)];
    let opt = FdmMirOptimizer {
        grad_mode: FdmGradMode::Linear,
        ..Default::default()
    };
    let gq = goals_grad_wrt_q(&opt, &net, &goals, None).expect("grad");
    assert!(gq.dq.iter().any(|x| x.abs() > 1e-12));
    let _ = fdm(&net).expect("eq");
}

#[test]
fn custom_fn_grad_q_builds() {
    let net = Network::arch_chain(3.0, 4, -1.0, -0.1);
    let s = rlx_fdm::Structure::from_network(&net);
    let opt = FdmMirOptimizer::default();
    let mut g = rlx_ir::Graph::new("fdm_mir");
    let built = opt.build_equilibrium_graph(&mut g, &net).expect("graph");
    let sig = FdmGradQSignature {
        num_edges: net.num_edges(),
        num_free: s.num_free(),
    };
    let loss_grad = g.param(
        "loss_grad",
        rlx_ir::Shape::new(&[s.num_free() * 3], rlx_ir::DType::F64),
    );
    let FdmEquilibriumGraph::Dense(dense) = &built else {
        panic!("small arch should use dense graph");
    };
    let k_node = dense.k;
    let dq = build_grad_q_custom_fn(&mut g, k_node, loss_grad, &sig);
    g.set_outputs(vec![built.xyz_free(), dq]);

    let mut session = Session::new(Device::Cpu).compile(g);
    opt.set_equilibrium_params(&mut session, &net, &built)
        .expect("params");
    let eq = fdm(&net).expect("eq");
    let loss_grad = goals_grad_xyz_free(
        &[Goal::min_free_z(-0.1, 1.0)],
        &eq,
        &s,
        &net.edges,
        &net.is_support,
        None,
    );
    session.set_param_typed(
        "loss_grad",
        bytemuck::cast_slice(&loss_grad),
        rlx_ir::DType::F64,
    );
    let xyz_free = rlx_fdm::EquilibriumModel::pack_xyz_free(&eq.xyz, &s);
    set_grad_q_param(&opt, &mut session, &net, &loss_grad, &xyz_free).expect("dq param");
    let outs = session.run_typed(&[]);
    assert_eq!(outs.len(), 2);
    let dq: Vec<f64> = outs[1]
        .0
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(
        dq.iter().any(|x| x.abs() > 1e-12),
        "host fdm_dq param should be nonzero"
    );
}
