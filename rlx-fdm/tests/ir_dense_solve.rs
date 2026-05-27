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

use rlx_fdm::graph::{fdm_dense_graph, pack_load_rhs, pack_stiffness};
use rlx_fdm::{Network, fdm};
use rlx_ir::GraphExt;
use rlx_runtime::{Device, Session};

#[test]
fn dense_solve_graph_matches_reference() {
    let mut net =
        Network::from_polyline(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], -1.0);
    net.anchor_nodes(&[0, 2]);
    net.loads_on_free([0.0, 0.0, -1.0]);
    let ref_eq = fdm(&net).expect("ref");

    let mut g = rlx_ir::Graph::new("fdm");
    let built = fdm_dense_graph(&mut g, &net).expect("graph");
    g.set_outputs(vec![built.xyz_free]);

    let k = pack_stiffness(&net).expect("K");
    let p = pack_load_rhs(&net).expect("P");

    let mut session = Session::new(Device::Cpu).compile(g);
    session.set_param_typed("fdm_K", bytemuck::cast_slice(&k), rlx_ir::DType::F64);
    session.set_param_typed("fdm_P", bytemuck::cast_slice(&p), rlx_ir::DType::F64);

    let outs = session.run_typed(&[]);
    let z_ir = f64::from_le_bytes(outs[0].0[16..24].try_into().unwrap());
    let z_ref = ref_eq.xyz[3 * 1 + 2];
    assert!((z_ref - z_ir).abs() < 1e-4, "ref z={z_ref} ir z={z_ir}");
}

#[test]
fn f64_narrow_column_matches_row_major_rhs() {
    let net = Network::arch_chain(3.0, 6, -1.0, -0.1);
    let p = pack_load_rhs(&net).expect("P");
    let nf = rlx_fdm::Structure::from_network(&net).num_free();

    let mut g = rlx_ir::Graph::new("narrow_p");
    let p_node = g.param("fdm_P", rlx_ir::Shape::new(&[nf, 3], rlx_ir::DType::F64));
    let pz = g.narrow_(p_node, 1, 2, 1);
    let out = g.reshape_(pz, vec![nf as i64]);
    g.set_outputs(vec![out]);

    let mut session = Session::new(Device::Cpu).compile(g);
    session.set_param_typed("fdm_P", bytemuck::cast_slice(&p), rlx_ir::DType::F64);
    let got = session.run_typed(&[]);
    let z_col: Vec<f64> = got[0]
        .0
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for a in 0..nf {
        let ref_z = p[a * 3 + 2];
        assert!(
            (ref_z - z_col[a]).abs() < 1e-12,
            "row {a}: ref={ref_z} narrow={}",
            z_col[a]
        );
    }
}
