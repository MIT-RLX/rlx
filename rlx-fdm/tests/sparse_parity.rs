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
use rlx_fdm::{fdm_with_options, sparse, Network, FdmOptions, IterativeConfig};

#[test]
fn sparse_matches_dense_on_arch() {
    let net = Network::arch_chain(5.0, 10, -1.0, -0.2);
    let s = rlx_fdm::Structure::from_network(&net);
    let na = s.num_fixed();
    let mut xf = vec![0.0; na * 3];
    for (j, &node) in s.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xf[j * 3 + c] = net.xyz[node * 3 + c];
        }
    }
    let diff = sparse::max_abs_diff_dense_sparse(&net.q, &xf, &net.loads, &s).expect("diff");
    assert!(diff < 1e-8, "dense/sparse diff {diff}");

    let dense = fdm_with_options(&net, &FdmOptions::default()).expect("dense");
    let mut opt = FdmOptions::default();
    opt.sparse = true;
    let sparse_eq = fdm_with_options(&net, &opt).expect("sparse");
    let dz = (dense.xyz[3 * 5 + 2] - sparse_eq.xyz[3 * 5 + 2]).abs();
    assert!(dz < 1e-8, "mid z dense={} sparse={}", dense.xyz[3 * 5 + 2], sparse_eq.xyz[3 * 5 + 2]);
}
