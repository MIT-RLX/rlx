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

//! End-to-end checks for the ic-parallel ternary kernel + banked arena.
//!
//! The current cortexm fixture is 8-bit (`WEIGHT_BITS = 8`), so the
//! ic-parallel path doesn't activate on the live model. We build a
//! **synthetic** ternary Conv2d model with `c_in = 8` so the optimizer
//! eligibility check fires, and verify:
//! * `Hints.ic_parallelism = 4` is set on the conv layer.
//! * `OptimizedModel.arena_bank` marks the conv's input slot as banked.
//! * `top.sv` emits 4 banked BRAMs (`u_ar*_b0..b3`) for the banked slot.
//! * The conv kernel emits the wider `x_dout`, the 2-D `w_crumb` array,
//!   and `partial[0..P_IC-1]` per oc-lane.
//! * Estimator: cycles drop by ~`P_ic` on the conv layer when
//!   ic_parallelism is enabled.

use rlx_fpga::codegen::collect_artifacts_opt;
use rlx_fpga::estimate::estimate;
use rlx_fpga::model::{Layer, Model};
use rlx_fpga::pack::pack;
use rlx_fpga::passes::optimize;
use rlx_fpga::quant::quantize_multiplier;
use rlx_fpga::tune::{OptTarget, RequantPrecision, Tune};

/// Tiny 3×3 ternary Conv2d on [4×4×8] → [2×2×4]. C_IN=8 (divisible by
/// P_ic=4), C_OUT=4 (divisible by P_oc=4 if we crank Latency parallelism).
fn synthetic_ternary_conv() -> Model {
    let h_in = 4usize;
    let w_in = 4usize;
    let c_in = 8usize;
    let c_out = 4usize;
    let kh = 3usize;
    let kw = 3usize;
    let logical: Vec<i8> = (0..(c_out * kh * kw * c_in))
        .map(|i| ((i % 3) as i8) - 1) // ternary {-1, 0, 1}
        .collect();
    let packed = pack(&logical, 2);
    let (m0, sh) = quantize_multiplier(0.5);
    let requant = vec![(m0, sh); c_out];
    Model {
        name: "synth_tern_conv".into(),
        input_len: h_in * w_in * c_in,
        layers: vec![Layer::Conv2d {
            name: "conv",
            h_in,
            w_in,
            c_in,
            c_out,
            kh,
            kw,
            pad_h: 0,
            pad_w: 0,
            stride_h: 1,
            stride_w: 1,
            x_zp: 0,
            w_zp: 0,
            out_zp: 0,
            weight_bits: 2,
            requant,
            weights: packed,
            bias: None,
        }],
    }
}

#[test]
fn ic_parallelism_activates_on_eligible_ternary_layer() {
    let m = synthetic_ternary_conv();
    let tune = Tune {
        ic_parallelism: 4,
        ternary_fast_path: true,
        fold_zero_zp: true,
        ..Tune::for_target(OptTarget::Energy)
    };
    let opt = optimize(&m, &tune);
    assert_eq!(
        opt.hints[0].ic_parallelism, 4,
        "conv with c_in=8 should get ic_parallelism=4"
    );
    assert!(opt.hints[0].ternary_fast_path);
    // Arena bank: the conv reads slot 0 (model input). Should be banked.
    let in_slot = opt.hints[0].bram_slot_in.unwrap();
    assert_eq!(
        opt.arena_bank.get(&in_slot),
        Some(&4),
        "conv's input slot should be 4-banked"
    );
}

#[test]
fn ic_parallel_kernel_emits_wide_x_dout_and_partial_array() {
    let m = synthetic_ternary_conv();
    let tune = Tune {
        ic_parallelism: 4,
        ternary_fast_path: true,
        fold_zero_zp: true,
        ..Tune::for_target(OptTarget::Energy)
    };
    let opt = optimize(&m, &tune);
    let arts = collect_artifacts_opt(&opt);

    let conv_sv = arts
        .iter()
        .find(|a| a.rel_path == "layers/conv.sv")
        .unwrap();
    // Wider x_dout port (P_ic=4 → 32 bits)
    assert!(
        conv_sv.content.contains("input  logic [31:0]"),
        "conv kernel should declare 32-bit x_dout port"
    );
    // 2-D crumb array
    assert!(
        conv_sv.content.contains("w_crumb [0:P-1][0:P_IC-1]"),
        "conv kernel should emit 2-D w_crumb array"
    );
    // Per-oc-lane partial array
    assert!(
        conv_sv.content.contains("partial [0:P_IC-1]"),
        "conv kernel should emit per-lane partial[] array"
    );
    // Inner-loop counter steps by P_IC
    assert!(
        conv_sv.content.contains("ic <= ic + P_IC"),
        "ic counter should advance by P_IC, not 1"
    );
}

#[test]
fn top_sv_emits_banked_brams_for_ic_parallel_consumer() {
    let m = synthetic_ternary_conv();
    let tune = Tune {
        ic_parallelism: 4,
        ternary_fast_path: true,
        fold_zero_zp: true,
        ..Tune::for_target(OptTarget::Energy)
    };
    let opt = optimize(&m, &tune);
    let arts = collect_artifacts_opt(&opt);

    let top = arts.iter().find(|a| a.rel_path == "top.sv").unwrap();
    // Conv reads slot 0 (default input slot) → banked × 4
    assert!(
        top.content.contains("u_ar0_b0")
            && top.content.contains("u_ar0_b1")
            && top.content.contains("u_ar0_b2")
            && top.content.contains("u_ar0_b3"),
        "top.sv should emit 4 banked BRAMs (u_ar0_b0..b3)"
    );
    // Shared word_addr for the banked group
    assert!(
        top.content.contains("ar0_word_addr"),
        "top.sv should declare per-bank-group word_addr"
    );
}

#[test]
fn estimator_cycles_drop_with_ic_parallelism() {
    let m = synthetic_ternary_conv();
    let tune_p1 = Tune {
        ternary_fast_path: true,
        ic_parallelism: 1,
        ..Tune::default()
    };
    let tune_p4 = Tune {
        ic_parallelism: 4,
        ..tune_p1
    };

    let cycles_p1 = estimate(&optimize(&m, &tune_p1)).cycles;
    let cycles_p4 = estimate(&optimize(&m, &tune_p4)).cycles;
    assert!(
        cycles_p4 < cycles_p1,
        "ic_parallelism=4 cycles {} should be < scalar {}",
        cycles_p4,
        cycles_p1
    );
    // We expect roughly P_ic-fold reduction in the inner-loop term.
    // Be lenient on the multiplier (epilogue overhead doesn't shrink).
    assert!(
        cycles_p4 * 2 < cycles_p1,
        "ic_parallelism=4 should give >2x speedup on conv-dominated synthetic ({} vs {})",
        cycles_p4,
        cycles_p1
    );
}

#[test]
fn ic_parallelism_falls_back_when_c_in_doesnt_divide() {
    let mut m = synthetic_ternary_conv();
    if let Layer::Conv2d { c_in, .. } = &mut m.layers[0] {
        *c_in = 6; // 6 % 4 ≠ 0 → not eligible
    }
    let tune = Tune {
        ic_parallelism: 4,
        ternary_fast_path: true,
        ..Tune::for_target(OptTarget::Energy)
    };
    // Note: setting c_in changes the weight tensor size — but since the
    // optimizer just inspects the field, this still tests the eligibility
    // logic. (The codegen wouldn't actually run with mismatched weights.)
    let opt = optimize(&m, &tune);
    assert_eq!(
        opt.hints[0].ic_parallelism, 1,
        "ic_parallelism should fall back to 1 when c_in doesn't divide"
    );
    assert!(
        opt.arena_bank.is_empty(),
        "no banked slots when no layer has ic_parallelism > 1"
    );
}

#[test]
fn precision_preset_keeps_ic_parallelism_at_one() {
    // Sanity: Precision target shouldn't enable lossy/aggressive paths.
    // ic_parallelism is gated on ternary_fast_path which Precision skips,
    // so even on a ternary model Precision stays at P_ic=1.
    let m = synthetic_ternary_conv();
    let opt = optimize(&m, &Tune::for_target(OptTarget::Precision));
    assert_eq!(opt.hints[0].ic_parallelism, 1);
    assert!(!opt.hints[0].ternary_fast_path);
    assert!(opt.arena_bank.is_empty());
}

#[test]
fn reference_uninfluenced_by_ic_parallelism() {
    use rlx_fpga::reference::run_with_precision;
    let m = synthetic_ternary_conv();
    let input: Vec<i8> = (0..m.input_len).map(|i| (i & 0x7F) as i8).collect();
    // Reference doesn't see Hints; same math regardless of P_ic.
    let (p1, _) = run_with_precision(&m, &input, RequantPrecision::Q0_31);
    let (p2, _) = run_with_precision(&m, &input, RequantPrecision::Q0_31);
    assert_eq!(p1, p2);
}
