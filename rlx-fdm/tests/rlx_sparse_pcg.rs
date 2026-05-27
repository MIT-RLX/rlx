#![cfg(feature = "rlx-sparse")]
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

use rlx_fdm::{EquilibriumModel, FdmCsr, Network, Structure, pattern_fast, register_rlx_sparse};
use rlx_runtime::{Device, Session};

#[test]
fn rlx_sparse_pcg_matches_reference() {
    register_rlx_sparse();
    let net = Network::arch_chain(3.0, 8, -1.0, -0.2);
    let s = Structure::from_network(&net);
    let pat = pattern_fast(&s);
    let (values, col_idx, row_ptr, n) = rlx_fdm::sparse::export_csr(&pat, &net.q);
    let na = s.num_fixed();
    let mut xf = vec![0.0; na * 3];
    for (j, &node) in s.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xf[j * 3 + c] = net.xyz[node * 3 + c];
        }
    }
    let p = EquilibriumModel::load_matrix(&net.q, &xf, &net.loads, &s);
    let xyz_ref = EquilibriumModel::nodes_free_positions(&net.q, &xf, &net.loads, &s).expect("ref");
    let ref_z = xyz_ref[2];
    let mut bz = vec![0.0; n];
    for a in 0..n {
        bz[a] = p[a * 3 + 2];
    }

    let mut g = rlx_ir::Graph::new("fdm_pcg");
    let b = g.input("b", rlx_ir::Shape::new(&[n], rlx_ir::DType::F64));
    let x = rlx_fdm::pcg_solve_graph(
        &mut g,
        &FdmCsr {
            values,
            col_idx,
            row_ptr,
            n,
        },
        b,
        8000,
        1e-10,
    );
    g.set_outputs(vec![x]);

    let mut session = Session::new(Device::Cpu).compile(g);
    let out = session.run_typed(&[("b", &f64_slice_to_bytes(&bz), rlx_ir::DType::F64)]);
    let z = f64::from_le_bytes(out[0].0[0..8].try_into().unwrap());
    assert!((ref_z - z).abs() < 1e-5, "ref={ref_z} pcg={z}");
}

fn f64_slice_to_bytes(s: &[f64]) -> Vec<u8> {
    s.iter().flat_map(|v| v.to_le_bytes()).collect()
}
