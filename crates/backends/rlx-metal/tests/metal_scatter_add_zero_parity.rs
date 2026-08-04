// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parity guard for `Op::ScatterAdd` on Metal: the output slot must be
//! zero-initialized before the scatter-accumulate, so rows that no index
//! touches read 0 (not stale/garbage arena memory). This is the bug behind
//! `rlx-tiny` NaN-ing on the first optimizer step on Metal (SynthMatMul VJP
//! `d_codebook` + embedding-table gather backward both scatter into a small
//! table where most rows are untouched).
//!
//! The tricky case is when the destination arena slot already holds NaN from a
//! previous run: the compiled Session reuses its arena across `run()` calls, so
//! we prime the slot with NaN (run 1), then scatter into only a few rows
//! (run 2) and assert the untouched rows come back as 0 (== the CPU oracle),
//! not the primed NaN.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// CPU oracle: `out[idx[i]] += src[i]`, accumulate into zeros.
fn scatter_add_ref(updates: &[f32], indices: &[f32], out_dim: usize, trailing: usize) -> Vec<f32> {
    let mut out = vec![0f32; out_dim * trailing];
    let num_updates = indices.len();
    for i in 0..num_updates {
        let row = indices[i] as usize;
        for j in 0..trailing {
            out[row * trailing + j] += updates[i * trailing + j];
        }
    }
    out
}

fn scatter_graph(num_updates: usize, out_dim: usize, trailing: usize) -> Graph {
    let mut g = Graph::new("scatter_add_zero");
    let updates = g.input("updates", Shape::new(&[num_updates, trailing], DType::F32));
    let indices = g.input("indices", Shape::new(&[num_updates], DType::F32));
    let out = g.add_node(
        Op::ScatterAdd,
        vec![updates, indices],
        Shape::new(&[out_dim, trailing], DType::F32),
    );
    g.set_outputs(vec![out]);
    g
}

/// Untouched output rows must be zero even when the arena slot held stale NaN
/// from a previous run of the same compiled Session.
#[test]
fn metal_scatter_add_zeroes_untouched_rows_over_reuse() {
    const NUM_UPDATES: usize = 9216;
    const OUT_DIM: usize = 256;
    const TRAILING: usize = 4;

    let g = scatter_graph(NUM_UPDATES, OUT_DIM, TRAILING);
    let mut m = Session::new(Device::Metal).compile(g);

    // Run 1 — prime the destination arena slot with NaN across ALL rows.
    // Every index maps 1:1 onto a row, updates carry NaN.
    let prime_updates = vec![f32::NAN; NUM_UPDATES * TRAILING];
    let prime_indices: Vec<f32> = (0..NUM_UPDATES).map(|i| (i % OUT_DIM) as f32).collect();
    let primed = m
        .run(&[
            ("updates", prime_updates.as_slice()),
            ("indices", prime_indices.as_slice()),
        ])
        .remove(0);
    assert!(
        primed.iter().any(|v| v.is_nan()),
        "run 1 should have written NaN into the destination slot"
    );

    // Run 2 — scatter finite updates into only rows 0..4. Rows 4..256 are
    // untouched and MUST come back as 0 (zeroed), not the primed NaN.
    let mut updates = vec![0f32; NUM_UPDATES * TRAILING];
    let mut indices = vec![0f32; NUM_UPDATES];
    for i in 0..NUM_UPDATES {
        let row = i % 4; // only rows 0..4 receive updates
        indices[i] = row as f32;
        for j in 0..TRAILING {
            updates[i * TRAILING + j] = ((i + j) % 7) as f32 * 0.5 - 1.0;
        }
    }
    let out = m
        .run(&[
            ("updates", updates.as_slice()),
            ("indices", indices.as_slice()),
        ])
        .remove(0);

    let reference = scatter_add_ref(&updates, &indices, OUT_DIM, TRAILING);

    assert!(
        out.iter().all(|v| v.is_finite()),
        "Metal ScatterAdd output must be finite; found non-finite (stale NaN leaked through untouched rows)"
    );
    let mut max_err = 0f32;
    for (a, b) in out.iter().zip(reference.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(
        max_err < 1e-4,
        "Metal ScatterAdd != CPU oracle (max_err {max_err:.3e}); untouched rows likely not zeroed"
    );

    // Explicitly assert some untouched rows are exactly 0.
    for row in [4usize, 100, 255] {
        for j in 0..TRAILING {
            assert_eq!(
                out[row * TRAILING + j],
                0.0,
                "untouched row {row} col {j} should be 0, got {}",
                out[row * TRAILING + j]
            );
        }
    }
}
