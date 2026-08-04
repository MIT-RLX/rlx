// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! MPSGraph lowering for `BatchElementwiseRegion`.
//!
//! Exercises `mps_graph_lower` (an `rlx_metal_host`/Apple-only module), so the
//! whole test is macOS-gated — it compiles to an empty test binary elsewhere.
#![cfg(target_os = "macos")]

use rlx_ir::op::{Activation, ChainOperand, ChainStep};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_metal::mps_graph_lower::try_lower;

fn nchw(n: usize, c: usize, h: usize, w: usize) -> Shape {
    Shape::new(&[n, c, h, w], DType::F32)
}

#[test]
fn mps_graph_lowers_batch_elementwise_region() {
    if !rlx_metal::mps_graph::mps_graph_supported() {
        eprintln!("skip mps_graph_lowers_batch_elementwise_region (MPSGraph unavailable)");
        return;
    }
    let mut g = Graph::new("mps_batch");
    let batch = g.input("batch", nchw(2, 3, 8, 8));
    let n0 = g.add_node(
        Op::Narrow {
            axis: 0,
            start: 0,
            len: 1,
        },
        vec![batch],
        nchw(1, 3, 8, 8),
    );
    let n1 = g.add_node(
        Op::Narrow {
            axis: 0,
            start: 1,
            len: 1,
        },
        vec![batch],
        nchw(1, 3, 8, 8),
    );
    let chain = vec![ChainStep::Activation(
        Activation::Relu,
        ChainOperand::Input(0),
    )];
    let out = g.add_node(
        Op::BatchElementwiseRegion {
            chain,
            num_batch_inputs: 2,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue: rlx_ir::RegionPrologue::None,
            prologue_input: 0,
        },
        vec![n0, n1],
        nchw(2, 3, 8, 8),
    );
    g.set_outputs(vec![out]);

    let plan = try_lower(&g).expect("batch region should lower via MPSGraph");
    assert_eq!(plan.outputs.len(), 1);
}
