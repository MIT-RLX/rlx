// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Transpose` swapping only the last two axes takes a batched-2-D fast
//! path. This checks it against the definition, for the shapes that reach it.
//!
//! Why it exists: the general path is an N-D index walk that measured
//! **103 MB/s** on an M4 Pro (a `[8, 2048, 1024]` f32 tensor took 653 ms,
//! against ~20 GB/s of memory bandwidth). It is a hot shape, not a corner:
//! `GroupedMatMul` wants `[E, in, out]` while GGUF stores expert banks as
//! `[E, out, in]`, so every dense MoE layer transposes all three of its banks,
//! and attention permutes `[B, H, S, D]` the same way. The fast path is ~10x.
//!
//! A fast path that is only *usually* right is worse than none, so the cases
//! below deliberately include ranks 3/4/5, non-square planes, odd extents that
//! do not divide the 32-wide tile, and a batch of one.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// `out[.., i, o] = in[.., o, i]`, straight from the definition.
fn reference(input: &[f32], dims: &[usize]) -> Vec<f32> {
    let rank = dims.len();
    let (d_o, d_i) = (dims[rank - 2], dims[rank - 1]);
    let plane = d_o * d_i;
    let batches: usize = dims[..rank - 2].iter().product();
    let mut out = vec![0f32; input.len()];
    for b in 0..batches {
        for o in 0..d_o {
            for i in 0..d_i {
                out[b * plane + i * d_o + o] = input[b * plane + o * d_i + i];
            }
        }
    }
    out
}

fn run_transpose(dims: &[usize], data: &[f32]) -> Vec<f32> {
    let rank = dims.len();
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(rank - 2, rank - 1);
    let out_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();

    let mut g = Graph::new("batched_transpose");
    let x = g.input("x", Shape::new(dims, DType::F32));
    let y = g.add_node(
        Op::Transpose { perm },
        vec![x],
        Shape::new(&out_dims, DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    c.run(&[("x", data)]).into_iter().next().expect("output")
}

fn ramp(n: usize) -> Vec<f32> {
    // Distinct values, so any misplaced element shows up rather than aliasing
    // onto a neighbour that happens to hold the same number.
    (0..n).map(|i| i as f32 * 0.5 - 3.0).collect()
}

#[test]
fn a_last_two_axis_swap_matches_the_definition() {
    let cases: &[&[usize]] = &[
        // The MoE expert-bank shape that motivated this, shrunk.
        &[8, 64, 32],
        // Square planes, and a batch of one (the parallel path is skipped).
        &[1, 33, 33],
        &[3, 16, 16],
        // Extents that do not divide the 32-wide tile, in both directions.
        &[2, 7, 45],
        &[2, 45, 7],
        // Rank 4 and 5: leading axes must be carried through untouched.
        &[2, 3, 17, 9],
        &[2, 2, 2, 5, 6],
        // Degenerate extents.
        &[4, 1, 10],
        &[4, 10, 1],
    ];
    for dims in cases {
        let n: usize = dims.iter().product();
        let data = ramp(n);
        let got = run_transpose(dims, &data);
        let want = reference(&data, dims);
        assert_eq!(
            got.len(),
            want.len(),
            "{dims:?}: produced {} elements, expected {}",
            got.len(),
            want.len()
        );
        assert_eq!(got, want, "{dims:?}: batched transpose disagrees");
    }
}

/// Transposing twice is the identity. Cheap, and it catches a fast path that is
/// self-consistently wrong in a way the reference above might share.
#[test]
fn transposing_twice_is_the_identity() {
    for dims in [vec![5, 12, 7], vec![2, 3, 40, 9]] {
        let n: usize = dims.iter().product();
        let data = ramp(n);
        let rank = dims.len();
        let once = run_transpose(&dims, &data);
        let mut swapped = dims.clone();
        swapped.swap(rank - 2, rank - 1);
        let twice = run_transpose(&swapped, &once);
        assert_eq!(twice, data, "{dims:?}: transpose is not an involution");
    }
}

/// Permutations that do NOT reduce to a last-two swap must not take the fast
/// path. They go through the general walk, and getting there matters: the fast
/// path assumes contiguous, untouched leading axes.
#[test]
fn other_permutations_are_still_correct() {
    let dims = [2usize, 3, 4];
    let n: usize = dims.iter().product();
    let data = ramp(n);

    for perm in [vec![1, 0, 2], vec![2, 0, 1], vec![1, 2, 0], vec![2, 1, 0]] {
        let out_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
        let mut g = Graph::new("perm");
        let x = g.input("x", Shape::new(&dims, DType::F32));
        let y = g.add_node(
            Op::Transpose { perm: perm.clone() },
            vec![x],
            Shape::new(&out_dims, DType::F32),
        );
        g.set_outputs(vec![y]);
        let mut c = Session::new(Device::Cpu).compile(g);
        let got = c.run(&[("x", data.as_slice())]).into_iter().next().unwrap();

        // Reference: walk the output index space and pull from the input.
        let in_strides = [dims[1] * dims[2], dims[2], 1];
        let mut want = vec![0f32; n];
        let mut oi = 0;
        for a in 0..out_dims[0] {
            for b in 0..out_dims[1] {
                for c2 in 0..out_dims[2] {
                    let coords = [a, b, c2];
                    let src: usize = (0..3).map(|d| coords[d] * in_strides[perm[d]]).sum();
                    want[oi] = data[src];
                    oi += 1;
                }
            }
        }
        assert_eq!(got, want, "perm {perm:?} is wrong");
    }
}
