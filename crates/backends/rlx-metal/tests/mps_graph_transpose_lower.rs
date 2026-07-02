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

//! MPSGraph lowering for 4D transpose (Brain-JEPA BHSD attention layout).

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_metal::mps_graph::mps_graph_supported;
use rlx_metal::mps_graph_lower::try_lower;

#[test]
fn mps_graph_lowers_bhsd_transpose_perm() {
    if !mps_graph_supported() {
        eprintln!("skip: MPSGraph not supported on this host");
        return;
    }

    let b = 1usize;
    let n = 64usize;
    let nh = 12usize;
    let dh = 64usize;

    let mut g = Graph::new("transpose_bhsd");
    let x = g.input("x", Shape::new(&[b, n, nh, dh], DType::F32));
    let y = g.transpose_(x, vec![0, 2, 1, 3]);
    g.set_outputs(vec![y]);

    let plan = try_lower(&g).expect("4D transpose [0,2,1,3] should lower to MPSGraph");
    assert_eq!(plan.inputs.len(), 1);
    assert_eq!(plan.outputs.len(), 1);
}
