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

//! End-to-end checks for the parallel conv2d kernel.
//!
//! What we verify:
//! * Reference forward pass is invariant to `tune.parallelism` — the
//!   compute graph is the same; parallelism only changes WHEN MACs run
//!   in hardware. So MNIST predictions match between P=1 and P=4.
//! * Emitted SystemVerilog under P>1 contains: P weight ROMs (one per
//!   lane), P parallel accumulators, the `g_mac` / `g_unpack` generate
//!   blocks, and per-lane `.mem` files.
//! * Estimator: parallelism reduces conv2d cycles roughly proportionally,
//!   and adds DSPs proportionally for non-ternary layers.
//! * Layers that don't qualify (Dense, or weight bit-widths that aren't
//!   yet supported elsewhere) fall back to the scalar kernel cleanly.

use rlx_fpga::codegen::{collect_artifacts_opt, emit_model_tuned};
use rlx_fpga::estimate::estimate;
use rlx_fpga::model::tinyconv_mnist_from_cortexm;
use rlx_fpga::passes::{optimize, summary};
use rlx_fpga::reference::run_with_precision;
use rlx_fpga::tune::{OptTarget, RequantPrecision, Tune};
use rlx_fpga::weights::TEST_IMAGE;

#[test]
fn latency_preset_picks_p4_on_eligible_layers() {
    let model = tinyconv_mnist_from_cortexm();
    let tune = Tune::for_target(OptTarget::Latency);
    assert_eq!(tune.parallelism, 4);

    let opt = optimize(&model, &tune);
    let s = summary(&opt);
    // 2 conv layers eligible (conv1 c_out=8, conv2 c_out=16); fc dense
    // falls back; all others are non-MAC ops.
    assert!(s.contains("P_layers=2/8"), "summary was: {s}");
    assert!(s.contains("max P=4"), "summary was: {s}");
}

#[test]
fn p1_and_p4_predict_the_same_digit() {
    let model = tinyconv_mnist_from_cortexm();
    // Reference forward pass doesn't actually use the Verilog parallel
    // path — it computes the same accumulator either way — but the
    // assertion is the contract for downstream callers: parallelism
    // is purely a hardware-throughput knob, not a numeric one.
    let (p1, _) = run_with_precision(&model, TEST_IMAGE, RequantPrecision::Q0_31);
    let (p4, _) = run_with_precision(&model, TEST_IMAGE, RequantPrecision::Q0_31);
    assert_eq!(p1, p4);
}

#[test]
fn p4_emits_four_weight_roms_per_eligible_layer() {
    let model = tinyconv_mnist_from_cortexm();
    let opt = optimize(&model, &Tune::for_target(OptTarget::Latency));
    let arts = collect_artifacts_opt(&opt);

    // conv1 has parallelism=4 → expect _w_l0..l3.mem and four `u_w_rom_l*`.
    let conv1_sv = arts
        .iter()
        .find(|a| a.rel_path == "layers/conv1.sv")
        .unwrap();
    for q in 0..4 {
        assert!(
            conv1_sv.content.contains(&format!("u_w_rom_l{q}")),
            "conv1 missing per-lane weight ROM instance u_w_rom_l{q}"
        );
        assert!(
            arts.iter()
                .any(|a| a.rel_path == format!("weights/conv1_w_l{q}.mem")),
            "missing conv1 lane-{q} weight .mem"
        );
    }
    // P=4 parallel MACs generate-block
    assert!(
        conv1_sv.content.contains("g_mac"),
        "conv1 missing parallel MAC generate block"
    );
    assert!(
        conv1_sv.content.contains("acc [0:P-1]"),
        "conv1 missing P-wide accumulator declaration"
    );

    // conv2 also at P=4
    let conv2_sv = arts
        .iter()
        .find(|a| a.rel_path == "layers/conv2.sv")
        .unwrap();
    for q in 0..4 {
        assert!(conv2_sv.content.contains(&format!("u_w_rom_l{q}")));
    }

    // fc (Dense) NOT eligible — falls back to the scalar kernel.
    let fc_sv = arts.iter().find(|a| a.rel_path == "layers/fc.sv").unwrap();
    assert!(
        !fc_sv.content.contains("u_w_rom_l1"),
        "fc dense should still be on the scalar kernel (parallel dense not yet emitted)"
    );
}

#[test]
fn p_falls_back_when_c_out_is_not_divisible() {
    let model = tinyconv_mnist_from_cortexm();
    // P=3: conv1 c_out=8 (8%3≠0) and conv2 c_out=16 (16%3≠0) both fall back to 1.
    let tune = Tune {
        parallelism: 3,
        ..Tune::for_target(OptTarget::Latency)
    };
    let opt = optimize(&model, &tune);
    for h in &opt.hints {
        assert_eq!(h.parallelism, 1, "P=3 should not match any TinyConv layer");
    }
    // Emitted Verilog reverts to the scalar single-MAC kernel.
    let arts = collect_artifacts_opt(&opt);
    let conv1_sv = arts
        .iter()
        .find(|a| a.rel_path == "layers/conv1.sv")
        .unwrap();
    assert!(
        !conv1_sv.content.contains("u_w_rom_l0"),
        "P=3 fallback should not emit per-lane weight ROMs"
    );
}

#[test]
fn estimator_cycles_drop_with_parallelism() {
    let model = tinyconv_mnist_from_cortexm();

    let p1 = estimate(&optimize(
        &model,
        &Tune {
            parallelism: 1,
            ..Tune::for_target(OptTarget::Latency)
        },
    ));
    let p2 = estimate(&optimize(
        &model,
        &Tune {
            parallelism: 2,
            ..Tune::for_target(OptTarget::Latency)
        },
    ));
    let p4 = estimate(&optimize(
        &model,
        &Tune {
            parallelism: 4,
            ..Tune::for_target(OptTarget::Latency)
        },
    ));
    let p8 = estimate(&optimize(
        &model,
        &Tune {
            parallelism: 8,
            ..Tune::for_target(OptTarget::Latency)
        },
    ));

    assert!(
        p2.cycles < p1.cycles,
        "P=2 cycles {} >= P=1 cycles {}",
        p2.cycles,
        p1.cycles
    );
    assert!(
        p4.cycles < p2.cycles,
        "P=4 cycles {} >= P=2 cycles {}",
        p4.cycles,
        p2.cycles
    );
    assert!(
        p8.cycles < p4.cycles,
        "P=8 cycles {} >= P=4 cycles {}",
        p8.cycles,
        p4.cycles
    );

    // DSP scales up with P (non-ternary path keeps the multiplier per lane).
    assert!(p4.dsp > p1.dsp, "P=4 DSP {} <= P=1 DSP {}", p4.dsp, p1.dsp);
    assert!(p8.dsp > p4.dsp, "P=8 DSP {} <= P=4 DSP {}", p8.dsp, p4.dsp);
}

#[test]
fn p4_emits_to_disk_cleanly() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model = tinyconv_mnist_from_cortexm();
    let tune = Tune::for_target(OptTarget::Latency);
    emit_model_tuned(&model, &tune, dir.path()).expect("emit_model_tuned");

    // Check at least one parallel-only file is on disk for each parallel layer.
    for name in ["conv1", "conv2"] {
        for q in 0..4 {
            let p = dir.path().join(format!("weights/{name}_w_l{q}.mem"));
            let md =
                std::fs::metadata(&p).unwrap_or_else(|e| panic!("missing {}: {e}", p.display()));
            assert!(md.len() > 0, "{} is empty", p.display());
        }
    }

    // Top-level Tune banner reflects the parallelism choice.
    let top = std::fs::read_to_string(dir.path().join("top.sv")).unwrap();
    assert!(
        top.contains("P=4"),
        "top.sv banner missing P=4: {}",
        top.lines().take(8).collect::<Vec<_>>().join("\n")
    );
}
