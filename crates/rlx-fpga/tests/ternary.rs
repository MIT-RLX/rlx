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

//! End-to-end check that the FPGA reference forward pass + Verilog
//! emitter both handle ternary (2-bit) and i4 (4-bit) packed weights.
//!
//! These are *synthetic* layers — the production cortexm weight blob is
//! 8-bit today, so we hand-construct a tiny `Model` directly. Reference
//! values are computed by hand below; the FPGA reference must match
//! exactly, and the emitted Verilog must reference the right ROM depths
//! and `weight_unpack` parameters.

use rlx_fpga::codegen::{collect_artifacts, emit_model};
use rlx_fpga::model::{Layer, Model};
use rlx_fpga::pack::{pack, packed_byte_len};
use rlx_fpga::quant::quantize_multiplier;
use rlx_fpga::reference::run;

/// Tiny ternary dense layer:
///   in=4, out=1, weights = [1, -1, 0, 1] (packed into one byte),
///   input = [10, 20, 30, 40], no bias, M_real = 0.5.
///
/// Hand check:
///   acc       = 1·10 + (-1)·20 + 0·30 + 1·40           = 30
///   srdhm(30, 2^30) = trunc((30·2^30 + 2^30)/2^31)     = 15
///   rdpot(15, 0)                                       = 15
///   sat_i8(15 + 0)                                     = 15
fn ternary_dense_model() -> Model {
    let logical: Vec<i8> = vec![1, -1, 0, 1];
    let packed = pack(&logical, 2);
    assert_eq!(packed.len(), packed_byte_len(logical.len(), 2));

    let (m0, shift) = quantize_multiplier(0.5);
    let dense = Layer::Dense {
        name: "fc_t",
        in_features: 4,
        out_features: 1,
        x_zp: 0,
        w_zp: 0,
        out_zp: 0,
        weight_bits: 2,
        requant: vec![(m0, shift)],
        weights: packed,
        bias: None,
    };
    Model {
        name: "ternary_dense_check".into(),
        input_len: 4,
        layers: vec![dense],
    }
}

#[test]
fn ternary_dense_reference_matches_hand_compute() {
    let model = ternary_dense_model();
    let input: Vec<i8> = vec![10, 20, 30, 40];
    let (_pred, intermediates) = run(&model, &input);
    assert_eq!(
        intermediates[0],
        vec![15],
        "ternary dense reference should produce 15, got {:?}",
        intermediates[0]
    );
}

#[test]
fn ternary_dense_emits_2bit_unpack() {
    let model = ternary_dense_model();
    let arts = collect_artifacts(&model);

    // Find the dense layer's .sv file
    let dense_sv = arts
        .iter()
        .find(|a| a.rel_path == "layers/fc_t.sv")
        .expect("missing layers/fc_t.sv");
    assert!(
        dense_sv.content.contains("W_BITS=2"),
        "ternary dense should bake W_BITS=2 into localparams"
    );
    assert!(
        dense_sv.content.contains("weight_unpack #(.BITS(W_BITS))"),
        "ternary dense should instantiate weight_unpack"
    );

    // weights/<name>_w.mem has byte_count lines (1 byte for 4 ternary weights).
    let mem = arts
        .iter()
        .find(|a| a.rel_path == "weights/fc_t_w.mem")
        .expect("missing weights/fc_t_w.mem");
    assert_eq!(
        mem.content.lines().count(),
        1,
        "ternary 4-element weight tensor packs into 1 byte"
    );

    // Weight ROM depth in the SV must match the byte count, not the logical count.
    assert!(
        dense_sv.content.contains("W_BYTE_LEN=1;"),
        "weight ROM depth should be the byte length (1), not logical (4)"
    );

    // Sanity: weight_unpack primitive itself is generated.
    assert!(
        arts.iter()
            .any(|a| a.rel_path == "primitives/weight_unpack.sv"),
        "missing primitives/weight_unpack.sv"
    );
}

/// Tiny 4-bit nibble-packed dense layer:
///   in=4, out=1, weights = [3, -2, 5, -1], input = [2, 4, 6, 8], M=0.5.
///   acc = 3·2 + (-2)·4 + 5·6 + (-1)·8 = 6 - 8 + 30 - 8 = 20
///   srdhm(20, 2^30) = 10; rdpot(10, 0) = 10; sat_i8(10) = 10.
#[test]
fn nibble_dense_reference_matches_hand_compute() {
    let logical: Vec<i8> = vec![3, -2, 5, -1];
    let packed = pack(&logical, 4);
    let (m0, shift) = quantize_multiplier(0.5);

    let model = Model {
        name: "i4_dense_check".into(),
        input_len: 4,
        layers: vec![Layer::Dense {
            name: "fc_i4",
            in_features: 4,
            out_features: 1,
            x_zp: 0,
            w_zp: 0,
            out_zp: 0,
            weight_bits: 4,
            requant: vec![(m0, shift)],
            weights: packed,
            bias: None,
        }],
    };
    let (_pred, ints) = run(&model, &[2, 4, 6, 8]);
    assert_eq!(ints[0], vec![10]);
}

/// Same model, end-to-end: write to a tempdir and confirm the emitted
/// SV has the right shape.
#[test]
fn ternary_dense_emit_to_tempdir() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model = ternary_dense_model();
    emit_model(&model, dir.path()).expect("emit_model");

    let dense_sv = std::fs::read_to_string(dir.path().join("layers/fc_t.sv")).unwrap();
    assert!(dense_sv.contains("W_BITS=2"));
    assert!(dense_sv.contains("weight_unpack"));

    let weight_unpack_sv =
        std::fs::read_to_string(dir.path().join("primitives/weight_unpack.sv")).unwrap();
    assert!(
        weight_unpack_sv.contains("g_b2"),
        "weight_unpack must contain the BITS=2 generate branch"
    );
    assert!(
        weight_unpack_sv.contains("g_b4"),
        "weight_unpack must contain the BITS=4 generate branch"
    );
    assert!(
        weight_unpack_sv.contains("g_b8"),
        "weight_unpack must contain the BITS=8 generate branch"
    );
}
