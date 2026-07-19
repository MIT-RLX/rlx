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

//! ONNX adaLN fixture (affine-free LayerNorm + Expand + Mul/Add modulation)
//! imports strictly and fuses to `AdaLayerNorm` via `FuseAdaLayerNorm`.

use rlx_compile::specialize_params;
use rlx_ir::op::{AdaNormKind, Op};
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};
use rlx_opt::FuseAdaLayerNorm;
use rlx_opt::pass::Pass as _;
use rlx_runtime::{Device, Session};

const B: usize = 2;
const S: usize = 4;
const D: usize = 8;

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| seed * (0.11 * (i as f32) - 0.07 * ((i % 3) as f32)))
        .collect()
}

#[test]
fn dit_adaln_onnx_import_fuses_to_ada_layer_norm() {
    let path = rlx_onnx_conformance::synthetic::dit_adaln_fixture();

    let opts = ImportOptions {
        strict: true,
        use_quantized_kernels: false,
        ..ImportOptions::default()
    };
    let (hir, params, report, _manifest) =
        build_hir_from_onnx_file(&path, opts).expect("import dit adaLN fixture (strict)");
    assert_eq!(report.stubbed, 0, "no node should be stubbed");
    assert!(
        report.unsupported.is_empty(),
        "unexpected unsupported ops: {:?}",
        report.unsupported
    );

    let graph = rlx_ir::hir_to_graph(hir).expect("hir_to_graph");
    let graph = specialize_params(&graph, &params);
    let fused = FuseAdaLayerNorm.run(graph);
    assert!(
        fused.nodes().iter().any(|n| {
            matches!(
                n.op,
                Op::AdaLayerNorm {
                    norm: AdaNormKind::LayerNorm,
                    ..
                }
            )
        }),
        "expected FuseAdaLayerNorm on ONNX adaLN fixture: {:?}",
        fused
            .nodes()
            .iter()
            .map(|n| format!("{:?}", n.op.kind()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn dit_adaln_onnx_session_cpu_run_finite() {
    let path = rlx_onnx_conformance::synthetic::dit_adaln_fixture();
    let opts = ImportOptions {
        strict: true,
        use_quantized_kernels: false,
        ..ImportOptions::default()
    };
    let (hir, params, _report, _manifest) =
        build_hir_from_onnx_file(&path, opts).expect("strict import");
    let graph = rlx_ir::hir_to_graph(hir).expect("hir_to_graph");
    let graph = specialize_params(&graph, &params);
    let fused = FuseAdaLayerNorm.run(graph);

    let x = fill(B * S * D, 1.0);
    let scale = fill(B * D, 0.2);
    let shift = fill(B * D, -0.1);

    let mut c = Session::new(Device::Cpu).compile(fused);
    let out = c.run(&[("x", &x), ("scale", &scale), ("shift", &shift)])[0].clone();
    assert_eq!(out.len(), B * S * D);
    assert!(out.iter().all(|v| v.is_finite()));
}
