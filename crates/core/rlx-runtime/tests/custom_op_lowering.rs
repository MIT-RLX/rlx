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

//! A downstream custom op that provides an `OpExtension::lower` rule — but NO
//! per-backend kernel — still compiles and runs, because the compile pipeline
//! decomposes it to primitives before dispatch. This is the "middle tier":
//! fuses and runs on every backend with no kernel and no core `Op` edit.

use std::sync::Arc;

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, LowerContext, Node, NodeId, Op, OpExtension, Shape, register_op};
use rlx_runtime::{Device, Session};

/// `y = 3·x`, expressed as primitives (`x + x + x`). No `CpuKernel` is
/// registered for it — execution relies entirely on the lowering.
struct Triple;

impl OpExtension for Triple {
    fn name(&self) -> &str {
        "test_triple_e2e"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
    fn lower(&self, _node: &Node, ctx: &mut LowerContext) -> Option<NodeId> {
        let x = ctx.inputs[0];
        let shape = ctx.out.node(x).shape.clone();
        let two_x = ctx
            .out
            .add_node(Op::Binary(BinaryOp::Add), vec![x, x], shape.clone());
        Some(
            ctx.out
                .add_node(Op::Binary(BinaryOp::Add), vec![two_x, x], shape),
        )
    }
}

#[test]
fn custom_op_with_lowering_runs_on_cpu_without_a_kernel() {
    register_op(Arc::new(Triple));

    let f = DType::F32;
    let mut g = Graph::new("triple");
    let x = g.input("x", Shape::new(&[4], f));
    let y = g.custom_op("test_triple_e2e", vec![], vec![x]);
    g.set_outputs(vec![y]);

    // Compile through the real CPU pipeline. `precompile_cleanup` decomposes the
    // custom op to primitives, so no `CpuKernel` lookup is ever attempted.
    let mut c = Session::new(Device::Cpu).compile(g);
    let x0 = vec![1.0f32, 2.0, 3.0, 4.0];
    let outs = c.run(&[("x", &x0)]);
    let y = &outs[0];

    let expect: Vec<f32> = x0.iter().map(|v| 3.0 * v).collect();
    assert_eq!(y.len(), expect.len(), "y={y:?}");
    for (a, b) in y.iter().zip(expect.iter()) {
        assert!((a - b).abs() < 1e-5, "y={y:?} expect={expect:?}");
    }
}
