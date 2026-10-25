// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end check that the FPGA reference forward pass + Verilog
//! emitter both handle ternary (2-bit) and i4 (4-bit) packed weights.
//!
//! These are *synthetic* layers — the production cortexm weight blob is
//! 8-bit today, so we hand-construct a tiny `Model` directly. Reference
//! values are computed by hand below; the FPGA reference must match
//! exactly, and the emitted Verilog must reference the right ROM depths
//! and `weight_unpack` parameters.

// Emits files / goes through `rlx-ir`: both need the `std` feature.
#![cfg(feature = "std")]

use rlx_fpga::codegen::{collect_artifacts, emit_model};
use rlx_fpga::model::{Layer, Model};
use rlx_fpga::pack::{pack, packed_byte_len};
use rlx_fpga::quant::quantize_multiplier;
use rlx_fpga::reference::run;
use rlx_fpga::{ExportQuantMode, FpgaExportConfig, LegalizeOptions, export_graph, prepare_model};
use rlx_ir::ternary::{TernaryConfig, ternarize};
use rlx_ir::{DType, Graph, Shape};

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
        weight_encoding: rlx_fpga::WeightEncoding::SignedInt,
        requant: vec![(m0, shift)],
        weights: packed,
        bias: None,
    };
    Model {
        name: "ternary_dense_check".into(),
        input_len: 4,
        layers: vec![dense],
        extra_outputs: vec![],
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
        dense_sv.content.contains("weight_unpack #(.BITS(W_BITS)"),
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
            weight_encoding: rlx_fpga::WeightEncoding::SignedInt,
            requant: vec![(m0, shift)],
            weights: packed,
            bias: None,
        }],
        extra_outputs: vec![],
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

// ── f32 graph → ExportQuantMode::Ternary (BitNet absmean PTQ) ──────────

const IN: usize = 8;
const HID: usize = 6;
const OUT: usize = 3;

fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

fn f32_mlp() -> Graph {
    let mut g = Graph::new("ternary_mlp");
    let x = g.input("x", Shape::new(&[1, IN], DType::F32));
    let w1 = g.param("fc1_w", Shape::new(&[IN, HID], DType::F32));
    let w2 = g.param("fc2_w", Shape::new(&[HID, OUT], DType::F32));
    let h = g.matmul(x, w1, Shape::new(&[1, HID], DType::F32));
    let h = g.activation(
        rlx_ir::op::Activation::Relu,
        h,
        Shape::new(&[1, HID], DType::F32),
    );
    let y = g.matmul(h, w2, Shape::new(&[1, OUT], DType::F32));
    g.set_outputs(vec![y]);
    g
}

/// `[in, out]` → `[out, in]` (FPGA Dense rows = output channels).
fn to_oi(w: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
    let mut oi = vec![0f32; w.len()];
    for i in 0..in_f {
        for o in 0..out_f {
            oi[o * in_f + i] = w[i * out_f + o];
        }
    }
    oi
}

/// Host f32 forward with BitNet-ternary weights; returns `(h, y)`.
fn host_forward(x: &[f32], w1: &[f32], w2: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let cfg = TernaryConfig::default();
    let t1 = ternarize(&to_oi(w1, IN, HID), HID, IN, &cfg).dequantize();
    let t2 = ternarize(&to_oi(w2, HID, OUT), OUT, HID, &cfg).dequantize();
    let h: Vec<f32> = (0..HID)
        .map(|o| (0..IN).map(|i| x[i] * t1[o * IN + i]).sum::<f32>().max(0.0))
        .collect();
    let y = (0..OUT)
        .map(|o| (0..HID).map(|i| h[i] * t2[o * HID + i]).sum())
        .collect();
    (h, y)
}

fn amax(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6)
}

#[test]
fn f32_mlp_exports_bitnet_ternary() {
    let mut seed = 7u32;
    let w1: Vec<f32> = (0..IN * HID).map(|_| lcg(&mut seed)).collect();
    let w2: Vec<f32> = (0..HID * OUT).map(|_| lcg(&mut seed)).collect();
    let x: Vec<f32> = (0..IN).map(|_| lcg(&mut seed)).collect();

    // Calibrate output scales on the ternary forward (what a user's
    // calibration batch would do).
    let x_scale = amax(&x) / 127.0;
    let xq: Vec<i8> = x
        .iter()
        .map(|&v| (v / x_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    let xd: Vec<f32> = xq.iter().map(|&q| f32::from(q) * x_scale).collect();
    let (h, y) = host_forward(&xd, &w1, &w2);
    let (h_scale, y_scale) = (amax(&h) / 127.0, amax(&y) / 127.0);

    let mut opts = LegalizeOptions::default();
    opts.weights_f32.insert("fc1_w".into(), w1.clone());
    opts.weights_f32.insert("fc2_w".into(), w2.clone());
    opts.act_scales.insert("input".into(), x_scale);
    opts.act_scales.insert("fc1_w".into(), h_scale);
    opts.act_scales.insert("fc2_w".into(), y_scale);

    let cfg = FpgaExportConfig::default().with_quant_mode(ExportQuantMode::Ternary);
    let model = prepare_model(&f32_mlp(), ExportQuantMode::Ternary, &opts, &cfg.from_graph)
        .expect("ternary legalize");

    // Packed codes are exactly the shared BitNet definition.
    let dense: Vec<_> = model
        .layers
        .iter()
        .filter_map(|l| match l {
            Layer::Dense {
                weight_bits,
                weights,
                in_features,
                out_features,
                ..
            } => Some((*weight_bits, weights.clone(), *in_features, *out_features)),
            _ => None,
        })
        .collect();
    assert_eq!(dense.len(), 2);
    for ((bits, packed, in_f, out_f), w) in dense.iter().zip([&w1, &w2]) {
        assert_eq!(*bits, 2);
        let want = ternarize(
            &to_oi(w, *in_f, *out_f),
            *out_f,
            *in_f,
            &TernaryConfig::default(),
        );
        let got: Vec<i8> = (0..in_f * out_f)
            .map(|i| rlx_cortexm::quant::read_weight(packed, i, 2) as i8)
            .collect();
        assert_eq!(got, want.codes);
        assert!(want.sparsity() > 0.0, "absmean should zero some weights");
    }

    // Integer reference ≈ f32 ternary forward, within requant error.
    let (_pred, ints) = run(&model, &xq);
    let y_int = &ints[2];
    for (o, &want) in y.iter().enumerate() {
        let got = f32::from(y_int[o]) * y_scale;
        assert!(
            (got - want).abs() <= 3.0 * y_scale,
            "y[{o}]: fpga {got} vs host {want} (lsb {y_scale})"
        );
    }

    // Ternary export takes the multiplier-free MAC even on the default
    // (Precision) tune, so it needs fewer DSPs than INT8.
    let dir = tempfile::tempdir().unwrap();
    let mut t_cfg = FpgaExportConfig::default().with_quant_mode(ExportQuantMode::Ternary);
    t_cfg.legalize = opts.clone();
    let t = export_graph(&f32_mlp(), &t_cfg, dir.path()).expect("ternary export");
    let dir8 = tempfile::tempdir().unwrap();
    let mut i_cfg = FpgaExportConfig::default().with_quant_mode(ExportQuantMode::Int8);
    i_cfg.legalize = opts;
    let i8x = export_graph(&f32_mlp(), &i_cfg, dir8.path()).expect("int8 export");
    assert!(
        t.estimate.dsp < i8x.estimate.dsp,
        "ternary dsp {} !< int8 dsp {}",
        t.estimate.dsp,
        i8x.estimate.dsp
    );
    let sv = std::fs::read_to_string(dir.path().join("layers/fc1.sv")).unwrap();
    assert!(sv.contains("W_BITS=2"), "fc1 should be 2-bit");
}
