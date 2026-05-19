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

//! End-to-end checks for each `OptTarget` preset.
//!
//! What we verify:
//! * Reference forward pass picks the same MNIST digit under every
//!   preset (Q0.31 is bit-exact; Q0.15 is within ≤1 ulp per layer but
//!   the argmax is robust).
//! * The emitted SystemVerilog reflects each tuning knob:
//!   - `Latency` / `Precision` → keeps the multiply, full Q0.31, no shared requant
//!   - `Size` / `Energy`       → ternary fast path on ternary layers, Q0.15, shared requant when uniform
//! * Resource estimator moves resources in the expected directions
//!   across presets.

use rlx_fpga::codegen::{collect_artifacts_opt, emit_model_tuned};
use rlx_fpga::estimate::estimate;
use rlx_fpga::model::{Layer, Model, tinyconv_mnist_from_cortexm};
use rlx_fpga::pack::pack;
use rlx_fpga::passes::{optimize, summary};
use rlx_fpga::quant::quantize_multiplier;
use rlx_fpga::reference::run_with_precision;
use rlx_fpga::tune::{OptTarget, RequantPrecision, Tune};
use rlx_fpga::weights::TEST_IMAGE;

fn ternary_dense_model() -> Model {
    let logical: Vec<i8> = vec![1, -1, 0, 1];
    let packed = pack(&logical, 2);
    let (m0, sh) = quantize_multiplier(0.5);
    Model {
        name: "ternary_smoke".into(),
        input_len: 4,
        layers: vec![Layer::Dense {
            name: "fc",
            in_features: 4,
            out_features: 1,
            x_zp: 0,
            w_zp: 0,
            out_zp: 0,
            weight_bits: 2,
            requant: vec![(m0, sh)],
            weights: packed,
            bias: None,
        }],
    }
}

#[test]
fn every_preset_agrees_on_the_prediction() {
    // We don't assert the *correct* label (the embedded TEST_IMAGE
    // fixture is stale relative to the trainer revision); we only
    // assert that every preset agrees with the Q0.31 baseline. That
    // catches regressions where a knob silently changes the math.
    let model = tinyconv_mnist_from_cortexm();
    let baseline = run_with_precision(&model, TEST_IMAGE, RequantPrecision::Q0_31).0;
    for target in [
        OptTarget::Latency,
        OptTarget::Size,
        OptTarget::Energy,
        OptTarget::Precision,
        OptTarget::Bandwidth,
    ] {
        let tune = Tune::for_target(target);
        let (pred, _) = run_with_precision(&model, TEST_IMAGE, tune.requant_precision);
        assert_eq!(
            pred, baseline,
            "{:?}: predicted {} but baseline says {}",
            target, pred, baseline
        );
    }
}

#[test]
fn fold_zero_zp_drops_the_x_zp_subtraction_when_eligible() {
    // TinyConv-MNIST has every conv/dense layer at zp=0 → fold_zero_zp
    // should activate on every one.
    let model = tinyconv_mnist_from_cortexm();
    let opt = optimize(
        &model,
        &Tune {
            fold_zero_zp: true,
            ..Tune::default()
        },
    );
    let arts = collect_artifacts_opt(&opt);
    let conv1 = arts
        .iter()
        .find(|a| a.rel_path == "layers/conv1.sv")
        .unwrap();
    assert!(
        conv1.content.contains("hints: fast_mac"),
        "conv1 should be tagged with the fast_mac hint"
    );
    // Fast-mac MAC line uses raw $signed(...) — no `- X_ZP`
    assert!(
        !conv1.content.contains("- X_ZP"),
        "fast_mac kernel should not emit `- X_ZP`"
    );
}

#[test]
fn precision_preset_keeps_full_zp_subs_off_only_when_zps_are_zero() {
    // Even Precision preset enables fold_zero_zp because it's lossless.
    // (The optimizer only activates it when zps are actually zero.)
    let model = tinyconv_mnist_from_cortexm();
    let tune = Tune::for_target(OptTarget::Precision);
    let opt = optimize(&model, &tune);
    let s = summary(&opt);
    // 3 conv/dense layers, all qualifying.
    assert!(s.contains("fast_mac=3/8"), "summary was: {s}");
    // No shared_requant in Precision.
    assert!(s.contains("shared_requant=0/8"), "summary was: {s}");
}

#[test]
fn energy_preset_activates_ternary_fast_path_on_ternary_layers() {
    let model = ternary_dense_model();
    let tune = Tune::for_target(OptTarget::Energy);
    let opt = optimize(&model, &tune);
    let arts = collect_artifacts_opt(&opt);
    let fc_sv = arts.iter().find(|a| a.rel_path == "layers/fc.sv").unwrap();
    assert!(
        fc_sv.content.contains("ternary_fast_path"),
        "Energy preset should tag the ternary layer"
    );
    assert!(
        fc_sv.content.contains("Ternary fast path — direct crumb"),
        "ternary fast path should emit the case-tree comment"
    );
    // Crumb-mux replaces the multiplier — no `* w_val` MAC line.
    assert!(
        !fc_sv.content.contains("xv_corrected * w_val"),
        "ternary path must not multiply by w_val"
    );
    assert!(
        !fc_sv.content.contains("xv_corrected * (w_val"),
        "ternary path must not multiply at all"
    );
    // Energy uses Q0.15
    assert!(
        fc_sv.content.contains("requant_q15"),
        "Energy preset emits Q0.15 epilogue"
    );
}

#[test]
fn precision_preset_keeps_multiplier_even_on_ternary() {
    let model = ternary_dense_model();
    let tune = Tune::for_target(OptTarget::Precision);
    let opt = optimize(&model, &tune);
    let arts = collect_artifacts_opt(&opt);
    let fc_sv = arts.iter().find(|a| a.rel_path == "layers/fc.sv").unwrap();
    assert!(
        fc_sv.content.contains("xv_corrected * w_val"),
        "Precision preset must keep the integer multiply for parity"
    );
    assert!(
        fc_sv.content.contains("requant_q31"),
        "Precision preset uses Q0.31"
    );
    // Ternary tag must NOT appear
    assert!(
        !fc_sv.content.contains("ternary_fast_path"),
        "Precision preset must not enable ternary fast path"
    );
}

#[test]
fn shared_requant_collapses_uniform_table_to_localparam() {
    // ternary_dense_model has out_features=1, so its requant table is
    // length 1 — trivially uniform.
    let model = ternary_dense_model();
    let tune = Tune::for_target(OptTarget::Size);
    let opt = optimize(&model, &tune);
    let arts = collect_artifacts_opt(&opt);
    let fc_sv = arts.iter().find(|a| a.rel_path == "layers/fc.sv").unwrap();
    assert!(
        fc_sv.content.contains("M0_VAL"),
        "Size preset should emit the M0_VAL localparam"
    );
    assert!(
        fc_sv.content.contains("SHIFT_VAL"),
        "Size preset should emit the SHIFT_VAL localparam"
    );
    // No M0/shift .mem files emitted when shared
    assert!(
        !arts.iter().any(|a| a.rel_path == "weights/fc_m0.mem"),
        "shared_requant should skip the M0 .mem file"
    );
    assert!(
        !arts.iter().any(|a| a.rel_path == "weights/fc_sh.mem"),
        "shared_requant should skip the shift .mem file"
    );
}

#[test]
fn estimator_size_preset_smaller_than_precision() {
    let model = tinyconv_mnist_from_cortexm();
    let prec = estimate(&optimize(&model, &Tune::for_target(OptTarget::Precision)));
    let sz = estimate(&optimize(&model, &Tune::for_target(OptTarget::Size)));
    assert!(
        sz.lut <= prec.lut,
        "Size LUT ({}) should be ≤ Precision LUT ({})",
        sz.lut,
        prec.lut
    );
    assert!(
        sz.bram_bytes <= prec.bram_bytes,
        "Size BRAM ({}) should be ≤ Precision BRAM ({})",
        sz.bram_bytes,
        prec.bram_bytes
    );
}

#[test]
fn estimator_energy_drops_dsp_for_ternary() {
    let model = ternary_dense_model();
    let prec = estimate(&optimize(&model, &Tune::for_target(OptTarget::Precision)));
    let en = estimate(&optimize(&model, &Tune::for_target(OptTarget::Energy)));
    assert!(
        en.dsp < prec.dsp,
        "Energy DSP ({}) should be < Precision DSP ({}) on ternary",
        en.dsp,
        prec.dsp
    );
}

#[test]
fn q15_preset_emits_q15_module_in_top_tree() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model = tinyconv_mnist_from_cortexm();
    let tune = Tune::for_target(OptTarget::Energy);
    emit_model_tuned(&model, &tune, dir.path()).expect("emit_model_tuned");

    let q15_path = dir.path().join("primitives/requant_q15.sv");
    let q15 = std::fs::read_to_string(q15_path).expect("missing requant_q15.sv");
    assert!(q15.contains("module requant_q15"));

    let conv1 = std::fs::read_to_string(dir.path().join("layers/conv1.sv")).unwrap();
    assert!(
        conv1.contains("requant_q15"),
        "Energy preset should wire conv1 to requant_q15"
    );
}

#[test]
fn reference_q15_argmax_matches_q31_on_tinyconv_mnist() {
    let model = tinyconv_mnist_from_cortexm();
    let (p31, _) = run_with_precision(&model, TEST_IMAGE, RequantPrecision::Q0_31);
    let (p15, _) = run_with_precision(&model, TEST_IMAGE, RequantPrecision::Q0_15);
    assert_eq!(p31, p15, "Q0.15 must agree with Q0.31 on the test image");
}
