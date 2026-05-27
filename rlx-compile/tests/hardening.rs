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
// Integration tests — compiler hardening (pipeline snapshots + rewrite).

use rlx_compile::{CompilePipeline, FusionTarget, inspect_pipeline, rewrite_for_backend};
use rlx_ir::{DType, Graph, GraphModule, Op, OpKind, Shape, verify::verify_all};

fn f32_shape(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

#[test]
fn pipeline_inspect_covers_hir_mir_lir() {
    let module = GraphModule::define("layer", |m| {
        let x = m.input("x", f32_shape(&[2, 64]));
        let w = m.param("w", f32_shape(&[64, 64]));
        m.linear(x, w, None, None, f32_shape(&[2, 64]))
    });
    let hir = module.into_hir().expect("hir");
    let pipe = CompilePipeline::new(FusionTarget::Cpu);
    let dump = inspect_pipeline(&pipe, hir).expect("inspect");

    assert!(dump.hir.contains("hir @layer"));
    assert!(dump.hir.contains("linear"));
    assert!(dump.mir_lowered.contains("mir @layer"));
    assert!(dump.mir_optimized.contains("graph @layer"));
    assert!(dump.lir.contains("lir @layer"));
    assert!(dump.lir.contains("arena:"));
    assert!(dump.lir.contains("fingerprint:"));
    assert!(dump.fusion.contains("fusion"));
}

#[test]
fn compile_module_passes_verifier() {
    let module = GraphModule::define("ffn", |m| {
        let x = m.input("x", f32_shape(&[4, 128]));
        let up = m.param("up", f32_shape(&[128, 256]));
        let gate = m.param("gate", f32_shape(&[128, 256]));
        let down = m.param("down", f32_shape(&[256, 128]));
        m.swiglu_ffn(x, up, gate, down, f32_shape(&[4, 128]))
    });
    let pipe = CompilePipeline::new(FusionTarget::Cpu);
    let result = pipe.compile_module(module).expect("compile");
    assert!(verify_all(result.lir.as_graph()).is_empty());
}

#[test]
fn fusion_clean_hir_linear() {
    let module = GraphModule::define("layer", |m| {
        let x = m.input("x", f32_shape(&[4, 32]));
        let w = m.param("w", f32_shape(&[32, 32]));
        let b = m.param("b", f32_shape(&[32]));
        m.linear_fused(x, w, b, None, f32_shape(&[4, 32]))
    });
    let pipe = CompilePipeline::new(FusionTarget::Cpu).with_assert_fusion_clean(true);
    pipe.compile_module(module)
        .expect("linear_fused should be fusion-clean");
}

#[test]
fn rewrite_unfuses_for_strict_backend() {
    let f = DType::F32;
    let mut g = Graph::new("f");
    let x = g.input("x", Shape::new(&[2, 8], f));
    let w = g.param("w", Shape::new(&[8, 4], f));
    let b = g.param("b", Shape::new(&[4], f));
    let out = g.fused_matmul_bias_act(x, w, b, None, Shape::new(&[2, 4], f));
    g.set_outputs(vec![out]);

    let strict = &[
        OpKind::Input,
        OpKind::Param,
        OpKind::MatMul,
        OpKind::Binary,
        OpKind::Expand,
        OpKind::Activation,
    ];
    let rewritten = rewrite_for_backend(g, strict);
    assert!(verify_all(&rewritten).is_empty());
    assert!(
        !rewritten
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::FusedMatMulBiasAct { .. }))
    );
}
