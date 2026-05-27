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
//! MLX `compile_lir` must not re-run fusion (HIR → LIR → backend path).

#![cfg(all(feature = "mlx", target_os = "macos"))]

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Op, OpKind, Shape};
use rlx_runtime::backend_for;
use rlx_runtime::stages::compile_graph_stages_for_backend;
use rlx_runtime::{CompileOptions, Device};

fn count_op_kind(graph: &Graph, kind: OpKind) -> usize {
    graph.nodes().iter().filter(|n| n.op.kind() == kind).count()
}

#[test]
fn mlx_compile_lir_preserves_fused_ops() {
    let mut g = Graph::new("fmm");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[3, 2], DType::F32));
    let b = g.param("b", Shape::new(&[2, 2], DType::F32));
    let y = g.add_node(
        Op::FusedMatMulBiasAct {
            activation: Some(Activation::Relu),
        },
        vec![x, w, b],
        Shape::new(&[2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);

    let opts = CompileOptions::new();
    let backend = backend_for(Device::Mlx).expect("mlx backend");
    let stages = compile_graph_stages_for_backend(Device::Mlx, g, &opts, backend.supported_ops());
    let lir = stages.lir;
    let fused_in_lir = count_op_kind(lir.as_graph(), OpKind::FusedMatMulBiasAct);
    assert_eq!(
        fused_in_lir, 1,
        "fusion pipeline should emit FusedMatMulBiasAct before compile_lir"
    );

    let mut exe = backend.compile_lir(lir, &opts);
    exe.set_param("w", &[1.0, 0.0, 0.0, -1.0, 2.0, 1.0]);
    exe.set_param("b", &[0.0, 0.0, 0.0, 0.0]);
    let outs = exe.run(&[("x", &[1.0, 0.5, -0.5, 0.0, 1.0, 1.0])]);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].len(), 4);
    assert!(outs[0].iter().all(|v| v.is_finite()));
}
