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

//! Verifies the pipelined-MAC FSM rewrite:
//! * Generated Verilog references S_PIPE / prev_valid / done_issuing
//!   (the new pipelined inner loop) — and *not* the old S_READ /
//!   S_WAIT / S_MAC sequence.
//! * S_REQ_ADDR / S_REQ_WAIT / S_REQ_DO are gone (epilogue collapses
//!   to a single S_WRITE).
//! * Estimator cycles drop by ~2.7× across every TinyConv preset.
//! * Reference output is unchanged (pipelining is a hardware-only
//!   change; Rust math is identical).

use rlx_fpga::codegen::collect_artifacts_opt;
use rlx_fpga::estimate::estimate;
use rlx_fpga::model::tinyconv_mnist_from_cortexm;
use rlx_fpga::passes::optimize;
use rlx_fpga::reference::run_with_precision;
use rlx_fpga::tune::{OptTarget, RequantPrecision, Tune};
use rlx_fpga::weights::TEST_IMAGE;

#[test]
fn conv_dense_kernels_use_pipelined_fsm() {
    let m = tinyconv_mnist_from_cortexm();
    let opt = optimize(&m, &Tune::default());
    let arts = collect_artifacts_opt(&opt);

    for layer_sv in ["layers/conv1.sv", "layers/conv2.sv", "layers/fc.sv"] {
        let sv = arts
            .iter()
            .find(|a| a.rel_path == layer_sv)
            .unwrap_or_else(|| panic!("missing {layer_sv}"));
        // New states present
        assert!(
            sv.content.contains("S_PIPE"),
            "{layer_sv} missing pipelined S_PIPE state"
        );
        assert!(
            sv.content.contains("prev_valid"),
            "{layer_sv} missing pipeline prev_valid register"
        );
        assert!(
            sv.content.contains("done_issuing"),
            "{layer_sv} missing pipeline done_issuing register"
        );
        // Old states gone. S_BIAS_WAIT remains (different state), so
        // we use unique-to-the-old-FSM markers: S_READ and S_REQ_ADDR.
        assert!(
            !sv.content.contains("S_READ"),
            "{layer_sv} still references the old S_READ state"
        );
        assert!(
            !sv.content.contains("S_REQ_ADDR"),
            "{layer_sv} still has the old S_REQ_ADDR (epilogue should collapse)"
        );
    }
}

#[test]
fn pipelined_kernels_run_about_3x_faster_in_estimator() {
    // Compare the current (pipelined) cycles against the lower-bound
    // we'd expect from the old 3-cycle-per-MAC FSM. With ~250k MACs
    // total, the old design was ~750k cycles + overhead; the new one
    // should be well under 350k for the Precision preset.
    let m = tinyconv_mnist_from_cortexm();
    let est = estimate(&optimize(&m, &Tune::for_target(OptTarget::Precision)));
    assert!(
        est.cycles < 350_000,
        "Precision cycles {} should be under 350k after pipelining",
        est.cycles
    );

    // Latency preset (P=4 + pipelined) should be even better.
    let lat = estimate(&optimize(&m, &Tune::for_target(OptTarget::Latency)));
    assert!(
        lat.cycles < 120_000,
        "Latency cycles {} should be under 120k after pipelining + P=4",
        lat.cycles
    );
    assert!(
        lat.cycles < est.cycles,
        "Latency ({}) should still beat Precision ({})",
        lat.cycles,
        est.cycles
    );
}

#[test]
fn reference_unchanged_after_pipelining() {
    // Pipelining only changes WHEN MACs run on hardware. The Rust
    // reference is purely math, unaffected. Check that prediction
    // matches the Q0.31 baseline for every preset that uses Q0.31.
    let m = tinyconv_mnist_from_cortexm();
    let baseline = run_with_precision(&m, TEST_IMAGE, RequantPrecision::Q0_31).0;
    for target in [OptTarget::Latency, OptTarget::Precision] {
        let tune = Tune::for_target(target);
        let p = run_with_precision(&m, TEST_IMAGE, tune.requant_precision).0;
        assert_eq!(
            p, baseline,
            "{target:?} reference diverged: got {p}, baseline {baseline}"
        );
    }
}
