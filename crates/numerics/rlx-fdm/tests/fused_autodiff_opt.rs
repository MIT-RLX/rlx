#![cfg(feature = "fuse")]
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
    let mut fused = FusedAutodiffFormFinding::try_new(&mir, &net, &loss)
        .expect("build")
        .expect("sparse");

    let (loss0, gq) = fused.loss_and_grad_q(&net).expect("eval");
    assert!(loss0.is_finite() && loss0 > 0.0);
    assert!(
        gq.iter().any(|g| g.is_finite() && g.abs() > 1e-10),
        "MIR autodiff dL/dq should be nonzero"
    );
}
