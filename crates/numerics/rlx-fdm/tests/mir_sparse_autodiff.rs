#![cfg(all(feature = "ir", feature = "rlx-sparse"))]
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

//! End-to-end `dL/dq` through sparse MIR: `q → assemble_csr → pcg_solve → loss`.

use rlx_autodiff::grad_with_loss;
use rlx_fdm::mir_opt::FdmMirOptimizer;
use rlx_fdm::{
    CsrAssemblySpec, Network, Structure, assemble_csr_values_graph, register_rlx_sparse,
};
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, GraphExt, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn assemble_csr_ad_matches_fd() {
    register_rlx_sparse();
    let net = Network::arch_chain(3.0, 8, -1.0, -0.1);
    let s = Structure::from_network(&net);
    let spec = CsrAssemblySpec::from_structure(&s);
    let ne = s.num_edges;

    let mut fwd = rlx_ir::Graph::new("assemble_csr");
    let q = fwd.param("fdm_q", Shape::new(&[ne], DType::F64));
    let values = assemble_csr_values_graph(&mut fwd, q, &spec);
    let loss = fwd.reduce(
        values,
        ReduceOp::Sum,
        vec![0],
        false,
        Shape::scalar(DType::F64),
    );
    fwd.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&fwd, &[q]);
    let mut session = Session::new(Device::Cpu).compile(fwd);
    session.set_param_typed("fdm_q", &f64_bytes(&net.q), DType::F64);

    let eps = 1e-7;
    let mut fd = vec![0.0; ne];
    for e in 0..ne {
        let mut qp = net.q.clone();
        qp[e] += eps;
        let lp = spec.assemble(&qp).iter().sum::<f64>();
        let mut qm = net.q.clone();
        qm[e] -= eps;
        let lm = spec.assemble(&qm).iter().sum::<f64>();
        fd[e] = (lp - lm) / (2.0 * eps);
    }

    let mut bwd_sess = Session::new(Device::Cpu).compile(bwd);
    bwd_sess.set_param_typed("fdm_q", &f64_bytes(&net.q), DType::F64);
    let gq =
        bytes_to_f64(&bwd_sess.run_typed(&[("d_output", &f64_bytes(&[1.0]), DType::F64)])[1].0);
    let mut max_err = 0.0f64;
    for (a, b) in gq.iter().zip(fd.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(max_err < 1e-5, "assemble_csr AD vs FD max_err={max_err}");
}

#[test]
fn sparse_mir_dq_matches_fd_on_z_sum() {
    register_rlx_sparse();
    let net = Network::arch_chain(4.0, 12, -1.0, -0.15);
    let s = Structure::from_network(&net);
    let nf = s.num_free();
    let opt = FdmMirOptimizer {
        fdm: rlx_fdm::FdmOptions {
            sparse: true,
            iterative: rlx_fdm::IterativeConfig {
                use_sparse: true,
                pcg_max_iter: 8000,
                pcg_tol: 1e-10,
                ..Default::default()
            },
        },
        sparse_graph_min_free: 8,
        ..Default::default()
    };

    let mut fwd = rlx_ir::Graph::new("fdm_sparse_ad_fwd");
    let built = opt
        .build_equilibrium_graph(&mut fwd, &net)
        .expect("equilibrium graph");
    let z_col = fwd.narrow_(built.xyz_free(), 1, 2, 1);
    let z_vec = fwd.reshape_(z_col, vec![nf as i64]);
    let loss = fwd.reduce(
        z_vec,
        ReduceOp::Sum,
        vec![0],
        false,
        Shape::scalar(DType::F64),
    );
    fwd.set_outputs(vec![loss]);

    let q = find_param(&fwd, "fdm_q");
    let bwd = grad_with_loss(&fwd, &[q]);

    let mut session = Session::new(Device::Cpu).compile(fwd);
    opt.set_equilibrium_params(&mut session, &net, &built)
        .expect("params");

    let mut run_loss = |network: &Network| -> f64 {
        opt.set_equilibrium_params(&mut session, network, &built)
            .expect("params");
        f64::from_le_bytes(session.run_typed(&[])[0].0[0..8].try_into().unwrap())
    };

    let eps = 1e-6;
    let mut fd_q = vec![0.0; net.num_edges()];
    for e in 0..net.num_edges() {
        let mut n_plus = net.clone();
        n_plus.q[e] += eps;
        let mut n_minus = net.clone();
        n_minus.q[e] -= eps;
        fd_q[e] = (run_loss(&n_plus) - run_loss(&n_minus)) / (2.0 * eps);
    }

    let mut bwd_session = Session::new(Device::Cpu).compile(bwd);
    opt.set_equilibrium_params(&mut bwd_session, &net, &built)
        .expect("params");
    let d_out = 1.0f64.to_le_bytes().to_vec();
    let outs = bwd_session.run_typed(&[("d_output", &d_out, DType::F64)]);
    let g_q = bytes_to_f64(&outs[1].0);

    assert!(fd_q.iter().any(|x| x.abs() > 1e-10), "FD dL/dq nonzero");
    assert!(
        g_q.iter().any(|x| x.is_finite() && x.abs() > 1e-12),
        "MIR autodiff dL/dq nonzero"
    );

    let mut max_err = 0.0f64;
    for (a, b) in g_q.iter().zip(fd_q.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(
        max_err < 0.12,
        "full MIR dL/dq vs FD max_err={max_err} (ad[0]={} fd[0]={})",
        g_q[0],
        fd_q[0]
    );
}

fn f64_bytes(xs: &[f64]) -> Vec<u8> {
    xs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn find_param(g: &rlx_ir::Graph, name: &str) -> rlx_ir::NodeId {
    for node in g.nodes() {
        if let rlx_ir::Op::Param { name: n } = &node.op {
            if n == name {
                return node.id;
            }
        }
    }
    panic!("param {name} not found");
}

fn bytes_to_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
