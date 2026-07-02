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

//! MPSGraph lowering for `ElementwiseRegion` with resize prologue.

#![cfg(target_os = "macos")]

use rlx_ir::op::{Activation, ChainOperand, ChainStep, RegionPrologue};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_metal::mps_graph::mps_graph_supported;
use rlx_metal::mps_graph_lower::try_lower;

#[test]
fn mps_graph_lowers_resize_prologue_chain_region() {
    if !mps_graph_supported() {
        eprintln!("skip: MPSGraph not supported on this host");
        return;
    }

    let mut g = Graph::new("mps_prologue");
    let x = g.input("x", Shape::new(&[1, 3, 8, 8], DType::F32));
    let a = g.input("a", Shape::new(&[1, 3, 16, 16], DType::F32));
    let chain = vec![
        ChainStep::Activation(Activation::Relu, ChainOperand::Input(0)),
        ChainStep::Binary(
            rlx_ir::op::BinaryOp::Add,
            ChainOperand::Step(0),
            ChainOperand::Input(1),
        ),
        ChainStep::Binary(
            rlx_ir::op::BinaryOp::Mul,
            ChainOperand::Step(1),
            ChainOperand::Input(1),
        ),
    ];
    let out = g.add_node(
        Op::ElementwiseRegion {
            chain,
            num_inputs: 2,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue: RegionPrologue::ResizeNearest2x,
            prologue_input: 0,
        },
        vec![x, a],
        Shape::new(&[1, 3, 16, 16], DType::F32),
    );
    g.set_outputs(vec![out]);

    let plan = try_lower(&g).expect("prologue region should lower via MPSGraph resize + chain");
    assert_eq!(plan.inputs.len(), 2);
    assert_eq!(plan.outputs.len(), 1);
}
