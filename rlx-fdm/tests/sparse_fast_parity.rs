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
use rlx_fdm::{sparse, Structure, Network};

#[test]
fn fast_assembly_matches_naive_sparse() {
    let net = Network::arch_chain(5.0, 10, -1.0, -0.2);
    let s = Structure::from_network(&net);
    let na = s.num_fixed();
    let mut xf = vec![0.0; na * 3];
    for (j, &node) in s.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xf[j * 3 + c] = net.xyz[node * 3 + c];
        }
    }
    let diff = sparse::max_abs_diff_dense_sparse(&net.q, &xf, &net.loads, &s).expect("diff");
    assert!(diff < 1e-8, "naive sparse diff {diff}");

    let fast = rlx_fdm::SparseStiffnessFast::pattern(&s);
    let dense = rlx_fdm::EquilibriumModel::nodes_free_positions(&net.q, &xf, &net.loads, &s).expect("d");
    let p = rlx_fdm::EquilibriumModel::load_matrix(&net.q, &xf, &net.loads, &s);
    let sf = fast.solve_xyz(&net.q, &p, 8000, 1e-10).expect("f");
    let mut m = 0.0f64;
    for (a, b) in dense.iter().zip(sf.iter()) {
        m = m.max((a - b).abs());
    }
    assert!(m < 1e-8, "fast vs dense {m}");
}
