// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! ORT vs RLX parity for ONNX random ops (CPU, Ort RNG backend).

use rlx_ir::RngOptions;
use rlx_onnx_conformance::{compare_tensors, harness::OrtSession, synthetic};
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};
use rlx_runtime::{CompileOptions, Device, Session};

const ORT_REF_NORMAL_LIKE: [f32; 6] = [
    -1.396_447_4,
    -1.232_599_5,
    2.326_51,
    -1.290_481_8,
    1.068_746,
    2.743_482_6,
];

const ORT_REF_NORMAL: [f32; 4] = [-1.396_447_4, -1.232_599_5, 2.326_51, -1.290_481_8];

const ORT_REF_UNIFORM_LIKE: [f32; 6] = [
    0.000_157_345_09,
    0.595_227_1,
    0.209_468_23,
    0.241_278_95,
    0.775_689_9,
    0.327_828_4,
];

const ORT_REF_UNIFORM: [f32; 4] = [0.000_157_345_09, 0.595_227_1, 0.209_468_23, 0.241_278_95];

#[test]
#[cfg(not(any(target_os = "ios", target_os = "android")))]
fn random_normal_like_ort_parity() {
    let path = synthetic::random_normal_like_fixture();
    let opts = ImportOptions {
        strict: false,
        ..ImportOptions::default()
    };
    let (hir, _params, _, _) =
        build_hir_from_onnx_file(&path, opts).expect("import RandomNormalLike fixture");
    let compile_opts = CompileOptions::new().rng(RngOptions::ort(7));
    let mut rlx = Session::new(Device::Cpu)
        .compile_hir_with(hir, &compile_opts)
        .expect("compile RandomNormalLike");
    let template = vec![0f32; 6];
    let got = rlx.run(&[("shape", &template)]).remove(0);

    let mut ort = OrtSession::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    let ref_out = ort
        .run_one_f32_input("shape", &template, &[2, 3], 0)
        .unwrap();

    let (max_diff, passed) = compare_tensors(&got, &ref_out, 1e-5);
    assert!(passed, "RandomNormalLike max diff {max_diff}");
    assert_eq!(got.len(), ORT_REF_NORMAL_LIKE.len());
    for (i, (&a, &b)) in got.iter().zip(ORT_REF_NORMAL_LIKE.iter()).enumerate() {
        assert!((a - b).abs() <= 1e-5, "elem {i}: {a} vs {b}");
    }
}

#[test]
#[cfg(not(any(target_os = "ios", target_os = "android")))]
fn random_normal_ort_parity() {
    let path = synthetic::random_normal_fixture();
    let opts = ImportOptions {
        strict: false,
        ..ImportOptions::default()
    };
    let (hir, _params, _, _) =
        build_hir_from_onnx_file(&path, opts).expect("import RandomNormal fixture");
    let compile_opts = CompileOptions::new().rng(RngOptions::ort(7));
    let mut rlx = Session::new(Device::Cpu)
        .compile_hir_with(hir, &compile_opts)
        .expect("compile RandomNormal");
    let got = rlx.run(&[]).remove(0);

    let mut ort = OrtSession::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    let ref_out = ort.run_no_inputs(0).unwrap();

    let (max_diff, passed) = compare_tensors(&got, &ref_out, 1e-5);
    assert!(passed, "RandomNormal max diff {max_diff}");
    for (i, (&a, &b)) in got.iter().zip(ORT_REF_NORMAL.iter()).enumerate() {
        assert!((a - b).abs() <= 1e-5, "elem {i}: {a} vs {b}");
    }
}

#[test]
#[cfg(not(any(target_os = "ios", target_os = "android")))]
fn random_uniform_like_ort_parity() {
    let path = synthetic::random_uniform_like_fixture();
    let opts = ImportOptions {
        strict: false,
        ..ImportOptions::default()
    };
    let (hir, _params, _, _) =
        build_hir_from_onnx_file(&path, opts).expect("import RandomUniformLike fixture");
    let compile_opts = CompileOptions::new().rng(RngOptions::ort(7));
    let mut rlx = Session::new(Device::Cpu)
        .compile_hir_with(hir, &compile_opts)
        .expect("compile RandomUniformLike");
    let template = vec![0f32; 6];
    let got = rlx.run(&[("shape", &template)]).remove(0);

    let mut ort = OrtSession::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    let ref_out = ort
        .run_one_f32_input("shape", &template, &[2, 3], 0)
        .unwrap();

    let (max_diff, passed) = compare_tensors(&got, &ref_out, 1e-5);
    assert!(passed, "RandomUniformLike max diff {max_diff}");
    for (i, (&a, &b)) in got.iter().zip(ORT_REF_UNIFORM_LIKE.iter()).enumerate() {
        assert!((a - b).abs() <= 1e-5, "elem {i}: {a} vs {b}");
    }
}

#[test]
#[cfg(not(any(target_os = "ios", target_os = "android")))]
fn random_uniform_ort_parity() {
    let path = synthetic::random_uniform_fixture();
    let opts = ImportOptions {
        strict: false,
        ..ImportOptions::default()
    };
    let (hir, _params, _, _) =
        build_hir_from_onnx_file(&path, opts).expect("import RandomUniform fixture");
    let compile_opts = CompileOptions::new().rng(RngOptions::ort(7));
    let mut rlx = Session::new(Device::Cpu)
        .compile_hir_with(hir, &compile_opts)
        .expect("compile RandomUniform");
    let got = rlx.run(&[]).remove(0);

    let mut ort = OrtSession::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    let ref_out = ort.run_no_inputs(0).unwrap();

    let (max_diff, passed) = compare_tensors(&got, &ref_out, 1e-5);
    assert!(passed, "RandomUniform max diff {max_diff}");
    for (i, (&a, &b)) in got.iter().zip(ORT_REF_UNIFORM.iter()).enumerate() {
        assert!((a - b).abs() <= 1e-5, "elem {i}: {a} vs {b}");
    }
}

#[test]
fn random_normal_like_import_lowers_native_rng() {
    let path = synthetic::random_normal_like_fixture();
    let opts = ImportOptions {
        strict: false,
        ..ImportOptions::default()
    };
    let (hir, _, report, _) = build_hir_from_onnx_file(&path, opts).expect("import");
    assert!(report.lowered >= 1, "expected lowered nodes");
    let graph = rlx_ir::hir_to_graph(hir).expect("hir to graph");
    assert!(
        graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, rlx_ir::Op::RngNormal { .. })),
        "expected Op::RngNormal in lowered graph"
    );
}

#[test]
fn random_normal_import_lowers_native_rng() {
    let path = synthetic::random_normal_fixture();
    let opts = ImportOptions {
        strict: false,
        ..ImportOptions::default()
    };
    let (hir, _, _, _) = build_hir_from_onnx_file(&path, opts).expect("import");
    let graph = rlx_ir::hir_to_graph(hir).expect("hir to graph");
    assert!(
        graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, rlx_ir::Op::RngNormal { .. })),
        "expected Op::RngNormal in lowered graph"
    );
}

#[test]
fn random_uniform_like_import_lowers_native_rng() {
    let path = synthetic::random_uniform_like_fixture();
    let opts = ImportOptions {
        strict: false,
        ..ImportOptions::default()
    };
    let (hir, _, report, _) = build_hir_from_onnx_file(&path, opts).expect("import");
    assert!(report.lowered >= 1, "expected lowered nodes");
    let graph = rlx_ir::hir_to_graph(hir).expect("hir to graph");
    assert!(
        graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, rlx_ir::Op::RngUniform { .. })),
        "expected Op::RngUniform in lowered graph"
    );
}

#[test]
fn random_uniform_import_lowers_native_rng() {
    let path = synthetic::random_uniform_fixture();
    let opts = ImportOptions {
        strict: false,
        ..ImportOptions::default()
    };
    let (hir, _, _, _) = build_hir_from_onnx_file(&path, opts).expect("import");
    let graph = rlx_ir::hir_to_graph(hir).expect("hir to graph");
    assert!(
        graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, rlx_ir::Op::RngUniform { .. })),
        "expected Op::RngUniform in lowered graph"
    );
}
