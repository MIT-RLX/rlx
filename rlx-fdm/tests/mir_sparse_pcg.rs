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

use rlx_fdm::mir_opt::{FdmEquilibriumGraph, FdmMirOptimizer};
use rlx_fdm::{Network, register_rlx_sparse};
use rlx_runtime::{Device, Session};

#[test]
fn mir_sparse_pcg_graph_matches_reference() {
    register_rlx_sparse();
    let net = Network::arch_chain(6.0, 40, -1.0, -0.2);
    let ref_eq = fdm_with_sparse(&net).expect("ref");

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
        sparse_graph_min_free: 32,
        ..Default::default()
    };

    let mut g = rlx_ir::Graph::new("fdm_sparse_mir");
    let built = opt.build_equilibrium_graph(&mut g, &net).expect("graph");
    let xyz = built.xyz_free();
    g.set_outputs(vec![xyz]);

    let mut session = Session::new(Device::Cpu).compile(g);
    opt.set_equilibrium_params(&mut session, &net, &built)
        .expect("params");

    let outs = session.run_typed(&[]);
    let packed = &outs[0].0;
    let s = rlx_fdm::Structure::from_network(&net);
    let nf = s.num_free();
    assert_eq!(packed.len(), nf * 3 * 8);
    let mut max_diff = 0.0f64;
    for (a, &node) in s.indices_free.iter().enumerate() {
        for c in 0..3 {
            let off = (a * 3 + c) * 8;
            let ir = f64::from_le_bytes(packed[off..off + 8].try_into().unwrap());
            let reference = ref_eq.xyz[node * 3 + c];
            max_diff = max_diff.max((reference - ir).abs());
        }
    }
    assert!(max_diff < 1e-4, "packed xyz_free max |ref-ir| = {max_diff}");

    match built {
        FdmEquilibriumGraph::SparsePcg(_) => {}
        FdmEquilibriumGraph::Dense(_) => panic!("expected sparse PCG graph for nf=40"),
    }
}

fn fdm_with_sparse(net: &Network) -> Result<rlx_fdm::EquilibriumState, rlx_fdm::FdmError> {
    let mut opts = rlx_fdm::FdmOptions::default();
    opts.sparse = true;
    opts.iterative.use_sparse = true;
    opts.iterative.pcg_max_iter = 8000;
    opts.iterative.pcg_tol = 1e-10;
    rlx_fdm::fdm_with_options(net, &opts)
}
