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

//! End-to-end checks for the IR-driven optimizer pipeline.
//!
//! The contract:
//! * `Model` → `rlx_ir::Graph` cleanly, and the verifier accepts it.
//! * `fuse_conv_relu` fires on the two TinyConv `Conv2d → Relu` pairs.
//! * Codegen elides the relu kernel modules + their entries in `top.sv`.
//! * `arena_plan` collapses N+1 dedicated BRAMs to 2 ping-pong scratch
//!   BRAMs, materialized as `u_ar0` / `u_ar1` in `top.sv`.
//! * Estimator reflects both savings.

use rlx_fpga::codegen::collect_artifacts_opt;
use rlx_fpga::estimate::estimate;
use rlx_fpga::ir::to_graph;
use rlx_fpga::model::tinyconv_mnist_from_cortexm;
use rlx_fpga::passes::{optimize, summary};
use rlx_fpga::reference::run_with_precision;
use rlx_fpga::tune::{OptTarget, RequantPrecision, Tune};
use rlx_fpga::weights::TEST_IMAGE;

#[test]
fn ir_graph_passes_verifier_on_tinyconv() {
    let m = tinyconv_mnist_from_cortexm();
    let ir = to_graph(&m);
    let errors = rlx_ir::verify::verify(&ir.graph);
    assert!(
        errors.is_empty(),
        "IR verifier reported errors: {:?}",
        errors.iter().map(|e| e.to_string()).collect::<Vec<_>>()
    );
}

#[test]
fn fuse_conv_relu_fires_on_both_pairs() {
    let m = tinyconv_mnist_from_cortexm();
    let opt = optimize(&m, &Tune::default());
    let s = summary(&opt);
    assert!(
        s.contains("fuse_conv_relu=2 (elided=2)"),
        "summary was: {s}"
    );
}

#[test]
fn fused_top_sv_has_no_relu_kernel_or_instance() {
    let m = tinyconv_mnist_from_cortexm();
    let opt = optimize(&m, &Tune::default());
    let arts = collect_artifacts_opt(&opt);

    // Relu .sv files are not emitted at all.
    for elided in ["layers/relu1.sv", "layers/relu2.sv"] {
        assert!(
            !arts.iter().any(|a| a.rel_path == elided),
            "{elided} should not be emitted under default Tune"
        );
    }

    // top.sv references neither kernel module nor instance.
    let top = arts.iter().find(|a| a.rel_path == "top.sv").unwrap();
    for s in ["relu1_kernel", "relu2_kernel", "u_relu1", "u_relu2"] {
        assert!(!top.content.contains(s), "top.sv should not reference {s}");
    }

    // The conv kernels are tagged with the fuses_relu hint.
    let conv1 = arts
        .iter()
        .find(|a| a.rel_path == "layers/conv1.sv")
        .unwrap();
    assert!(
        conv1.content.contains("fuses_relu"),
        "conv1.sv should advertise the fuses_relu hint"
    );
    // And the requant epilogue clamps at OUT_ZP.
    assert!(
        conv1.content.contains("(q_raw < OUT_ZP[7:0])"),
        "conv1.sv should clamp q_raw at OUT_ZP for the fused relu"
    );
}

#[test]
fn arena_plan_emits_two_ping_pong_brams() {
    let m = tinyconv_mnist_from_cortexm();
    let opt = optimize(&m, &Tune::default());
    let arts = collect_artifacts_opt(&opt);
    let top = arts.iter().find(|a| a.rel_path == "top.sv").unwrap();

    // Exactly two arena BRAMs, named u_ar0 / u_ar1.
    assert!(top.content.contains("u_ar0"), "top.sv should declare u_ar0");
    assert!(top.content.contains("u_ar1"), "top.sv should declare u_ar1");
    assert!(
        !top.content.contains("u_ar2"),
        "arena should fit in 2 slots"
    );

    // No legacy per-stage BRAMs (a0_, a1_, …).
    assert!(
        !top.content.contains("u_a0 ("),
        "arena layout should not emit legacy u_a0"
    );
}

#[test]
fn arena_off_keeps_legacy_per_stage_brams() {
    let m = tinyconv_mnist_from_cortexm();
    let tune = Tune {
        arena_plan: false,
        fuse_conv_relu: false,
        ..Tune::default()
    };
    let opt = optimize(&m, &tune);
    let arts = collect_artifacts_opt(&opt);
    let top = arts.iter().find(|a| a.rel_path == "top.sv").unwrap();
    assert!(
        top.content.contains("u_a0"),
        "legacy mode should emit per-stage u_a0..u_aN"
    );
    assert!(
        !top.content.contains("u_ar0"),
        "legacy mode should not emit arena slots"
    );
}

#[test]
fn estimator_arena_uses_only_two_activation_brams() {
    let m = tinyconv_mnist_from_cortexm();

    // Compare arena-on vs arena-off by toggling that one knob.
    let with_arena = estimate(&optimize(
        &m,
        &Tune {
            arena_plan: true,
            ..Tune::default()
        },
    ));
    let without_arena = estimate(&optimize(
        &m,
        &Tune {
            arena_plan: false,
            ..Tune::default()
        },
    ));

    // Arena collapses N+1 activation BRAMs to 2 — strictly less BRAM total.
    assert!(
        with_arena.bram_bytes < without_arena.bram_bytes,
        "arena BRAM {} should be < no-arena BRAM {}",
        with_arena.bram_bytes,
        without_arena.bram_bytes
    );
}

#[test]
fn estimator_fusion_drops_cycles() {
    let m = tinyconv_mnist_from_cortexm();
    let with_fuse = estimate(&optimize(
        &m,
        &Tune {
            fuse_conv_relu: true,
            ..Tune::default()
        },
    ));
    let without_fuse = estimate(&optimize(
        &m,
        &Tune {
            fuse_conv_relu: false,
            ..Tune::default()
        },
    ));
    assert!(
        with_fuse.cycles < without_fuse.cycles,
        "fused cycles {} should be < unfused cycles {}",
        with_fuse.cycles,
        without_fuse.cycles
    );
}

#[test]
fn reference_unchanged_by_passes() {
    // The reference forward pass doesn't see Hints or Tune (other than
    // requant_precision); fusion / arena / parallelism are pure
    // hardware-level concerns. The reference output must therefore
    // be identical regardless of which preset we're targeting.
    let m = tinyconv_mnist_from_cortexm();
    let baseline = run_with_precision(&m, TEST_IMAGE, RequantPrecision::Q0_31).0;
    for target in [
        OptTarget::Latency,
        OptTarget::Size,
        OptTarget::Energy,
        OptTarget::Precision,
        OptTarget::Bandwidth,
    ] {
        let _opt = optimize(&m, &Tune::for_target(target));
        // (`optimize` runs the verifier internally — if it would have
        // panicked, the test would have failed already.)
        let p = run_with_precision(&m, TEST_IMAGE, Tune::for_target(target).requant_precision).0;
        // Q0.31 vs Q0.15 may differ by ≤1 logit-ulp but the argmax
        // converges on the same digit on real models. Keep the
        // assertion as "matches Q0.31 baseline" only for the Q0.31
        // presets; tolerate Q0.15 drift.
        if matches!(
            Tune::for_target(target).requant_precision,
            RequantPrecision::Q0_31
        ) {
            assert_eq!(
                p, baseline,
                "{target:?}: Q0.31 reference should match baseline"
            );
        }
    }
}
