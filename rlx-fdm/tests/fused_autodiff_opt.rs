#![cfg(feature = "fuse")]
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

//! Fully fused MIR autodiff GD on `Σ z` (equilibrium + loss + `dL/dq` in one compile).

use rlx_fdm::fuse::{FusedAutodiffFormFinding, FusedMirLoss};
use rlx_fdm::{FdmMirOptimizer, Network};

#[test]
fn fused_autodiff_gd_steps_on_sum_z() {
    let net = Network::arch_chain(4.0, 12, -1.0, -0.15);
    let mut mir = FdmMirOptimizer::default();
    mir.fdm.sparse = true;
    mir.sparse_graph_min_free = 8;
    let loss = FusedMirLoss::SumFreeZ {
        target: -0.5,
        weight: 1.0,
    };
    let mut fused =
        FusedAutodiffFormFinding::try_new(&mir, &net, &loss).expect("build").expect("sparse");

    let (loss0, gq) = fused.loss_and_grad_q(&net).expect("eval");
    assert!(loss0.is_finite() && loss0 > 0.0);
    assert!(
        gq.iter().any(|g| g.is_finite() && g.abs() > 1e-10),
        "MIR autodiff dL/dq should be nonzero"
    );
}
