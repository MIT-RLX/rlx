// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared FKL-style benchmark / test graphs.

use rlx_ir::op::{Activation, ChainOperand, ChainStep, RegionPrologue};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};

pub fn nchw(n: usize, c: usize, h: usize, w: usize) -> Shape {
    Shape::new(&[n, c, h, w], DType::F32)
}

/// `resize ? relu` primitive chain (unfused baseline).
pub fn resize_relu_graph(name: &str, n: usize, c: usize, h: usize, w: usize) -> Graph {
    let mut g = Graph::new(name);
    let x = g.input("x", nchw(n, c, h, w));
    let up = g.add_node(Op::ResizeNearest2x, vec![x], nchw(n, c, h * 2, w * 2));
    let out = g.relu(up);
    g.set_outputs(vec![out]);
    g
}

/// Hand-built `ElementwiseRegion` with resize prologue.
pub fn resize_relu_region_graph(name: &str, n: usize, c: usize, h: usize, w: usize) -> Graph {
    let mut g = Graph::new(name);
    let x = g.input("x", nchw(n, c, h, w));
    let chain = vec![ChainStep::Activation(
        Activation::Relu,
        ChainOperand::Input(0),
    )];
    let out = g.add_node(
        Op::ElementwiseRegion {
            chain,
            num_inputs: 1,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue: RegionPrologue::ResizeNearest2x,
            prologue_input: 0,
        },
        vec![x],
        nchw(n, c, h * 2, w * 2),
    );
    g.set_outputs(vec![out]);
    g
}

/// Per-slice `ElementwiseRegion` + `Concat` (batch fusion input).
pub fn batch_narrow_relu_regions_graph(
    name: &str,
    batch_n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Graph {
    let mut g = Graph::new(name);
    let batch = g.input("batch", nchw(batch_n, c, h, w));
    let chain = vec![ChainStep::Activation(
        Activation::Relu,
        ChainOperand::Input(0),
    )];
    let mut slices = Vec::with_capacity(batch_n);
    for i in 0..batch_n {
        let sl = g.add_node(
            Op::Narrow {
                axis: 0,
                start: i,
                len: 1,
            },
            vec![batch],
            nchw(1, c, h, w),
        );
        slices.push(g.add_node(
            Op::ElementwiseRegion {
                chain: chain.clone(),
                num_inputs: 1,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
                prologue: RegionPrologue::None,
                prologue_input: 0,
            },
            vec![sl],
            nchw(1, c, h, w),
        ));
    }
    let out = g.add_node(Op::Concat { axis: 0 }, slices, nchw(batch_n, c, h, w));
    g.set_outputs(vec![out]);
    g
}

/// Primitive `narrow ? relu ? concat` (needs `MarkBatchSliceRegions`).
pub fn batch_narrow_relu_primitive_graph(
    name: &str,
    batch_n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Graph {
    let mut g = Graph::new(name);
    let batch = g.input("batch", nchw(batch_n, c, h, w));
    let mut slices = Vec::with_capacity(batch_n);
    for i in 0..batch_n {
        let sl = g.add_node(
            Op::Narrow {
                axis: 0,
                start: i,
                len: 1,
            },
            vec![batch],
            nchw(1, c, h, w),
        );
        slices.push(g.relu(sl));
    }
    let out = g.add_node(Op::Concat { axis: 0 }, slices, nchw(batch_n, c, h, w));
    g.set_outputs(vec![out]);
    g
}
