// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The lane-mask reset seam on a real batched workload.
//!
//! `crates/core/rlx-runtime/tests/lane_reset.rs` checks the mechanism on a
//! counter. This checks it where it is meant to be used: several independent
//! D2Q9 worlds stepped in one graph, with an arbitrary subset reset each step —
//! the parallel-environment shape that motivated the seam.
//!
//! Two properties matter and neither is about `Op::Where` selecting correctly:
//!
//! 1. **World isolation.** A world that is not reset must evolve *bit-identically*
//!    to the same world run alone. Streaming rolls only the spatial axes, so a
//!    lane-axis leak (rolling axis 0, or a mask broadcast off by one) shows up
//!    here and nowhere else.
//! 2. **One artifact.** Every reset pattern goes through a single compiled
//!    graph. That is what makes a captured schedule survive an episode boundary.

#![cfg(feature = "ir")]

use rlx_ir::infer::GraphExt;
use rlx_ir::lanes::LaneExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_lbm::graph::{FIELD_NAMES, stream_step_graph_batched};
use rlx_lbm::sim::taylor_green;
use rlx_runtime::{Device, Session};

const NW: usize = 3;
const N: usize = 8;
const CELLS: usize = N * N;

/// Rest-state value of each of the six fields (`S = u ⊗ u = 0` at rest).
const REST: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// Batched stream step followed by a lane-masked reset to the rest state.
fn build(nworld: usize) -> Graph {
    let (mut g, outs) = stream_step_graph_batched(nworld, N, N);
    let mask = g.input("reset", Shape::new(&[nworld], DType::F32));
    let rests: Vec<_> = REST
        .iter()
        .map(|&v| g.full(&[nworld, N, N], v, DType::F32))
        .collect();
    let reset = g.reset_lanes_many_(&outs, &rests, mask);
    g.set_outputs(reset.clone());
    g
}

/// Fill `nworld` copies of a Taylor–Green field, each with its own amplitude so
/// the worlds are distinguishable.
fn initial_state(nworld: usize) -> Vec<Vec<f32>> {
    let mut fields = vec![Vec::with_capacity(nworld * CELLS); 6];
    for w in 0..nworld {
        let f = taylor_green(N, 0.02 + 0.01 * w as f64, 0.01);
        for y in 0..N {
            for x in 0..N {
                let m = f.at(x, y);
                for (k, v) in [m.rho, m.u[0], m.u[1], m.s[0], m.s[1], m.s[2]]
                    .iter()
                    .enumerate()
                {
                    fields[k].push(*v as f32);
                }
            }
        }
    }
    fields
}

fn run(
    compiled: &mut rlx_runtime::CompiledGraph,
    state: &[Vec<f32>],
    mask: &[f32],
) -> Vec<Vec<f32>> {
    let mut feeds: Vec<(&str, &[f32])> = FIELD_NAMES
        .iter()
        .zip(state.iter())
        .map(|(n, v)| (*n, v.as_slice()))
        .collect();
    feeds.push(("reset", mask));
    compiled.run(&feeds)
}

/// Worlds that are not reset must evolve exactly as if they were alone.
///
/// The reference is a separate `nworld = 1` graph with no reset at all, so a
/// disagreement means either the lane axis is being rolled or the mask is
/// reaching the wrong world.
#[test]
fn unmasked_worlds_evolve_as_if_alone() {
    let mut batched = Session::new(Device::Cpu).compile(build(NW));
    let mut solo = Session::new(Device::Cpu).compile(build(1));

    let mut state = initial_state(NW);
    // World 1 resets on every step; worlds 0 and 2 never do.
    let mask = vec![0.0f32, 1.0, 0.0];

    // Solo references for worlds 0 and 2, never reset.
    let all = initial_state(NW);
    let mut solo_state: Vec<Vec<Vec<f32>>> = [0usize, 2]
        .iter()
        .map(|&w| {
            all.iter()
                .map(|f| f[w * CELLS..(w + 1) * CELLS].to_vec())
                .collect()
        })
        .collect();

    for step in 0..4 {
        state = run(&mut batched, &state, &mask);
        for (si, &w) in [0usize, 2].iter().enumerate() {
            solo_state[si] = run(&mut solo, &solo_state[si], &[0.0]);
            for f in 0..6 {
                for c in 0..CELLS {
                    let got = state[f][w * CELLS + c];
                    let want = solo_state[si][f][c];
                    assert!(
                        (got - want).abs() <= 1e-6 * (1.0 + want.abs()),
                        "step {step}, world {w}, field {} cell {c}: batched {got} vs solo {want}",
                        FIELD_NAMES[f]
                    );
                }
            }
        }
    }
}

/// A masked world returns to rest exactly; the unmasked ones keep moving.
///
/// Each target starts from a **fresh** state rather than accumulating, because
/// a uniform rest field is a fixed point of the LBM step — every cell is
/// identical, so streaming returns it unchanged. An accumulating version would
/// leave previously-reset worlds at rest forever, and the "others are still
/// moving" guard below would fail for a reason that has nothing to do with the
/// mask.
#[test]
fn masked_worlds_return_to_rest() {
    let mut compiled = Session::new(Device::Cpu).compile(build(NW));

    for target in 0..NW {
        let mask: Vec<f32> = (0..NW)
            .map(|w| if w == target { 1.0 } else { 0.0 })
            .collect();
        let state = run(&mut compiled, &initial_state(NW), &mask);

        for f in 0..6 {
            for c in 0..CELLS {
                let v = state[f][target * CELLS + c];
                assert!(
                    (v - REST[f]).abs() < 1e-6,
                    "reset world {target}: field {} cell {c} = {v}, want {}",
                    FIELD_NAMES[f],
                    REST[f]
                );
            }
        }

        // Every other world must still be moving, or the mask is too broad and
        // the assertion above would pass vacuously.
        for w in (0..NW).filter(|w| *w != target) {
            let moving = (0..CELLS).any(|c| state[1][w * CELLS + c].abs() > 1e-4);
            assert!(
                moving,
                "resetting world {target} also flattened world {w} — mask is too broad"
            );
        }
    }
}

/// Rest really is a fixed point, which the test above depends on. Pinned here
/// so that if the step ever stops preserving a uniform field, the reason the
/// other test is structured the way it is stays discoverable.
#[test]
fn uniform_rest_is_a_fixed_point() {
    let mut compiled = Session::new(Device::Cpu).compile(build(1));
    let mut state: Vec<Vec<f32>> = REST.iter().map(|&v| vec![v; CELLS]).collect();
    for step in 0..3 {
        state = run(&mut compiled, &state, &[0.0]);
        for f in 0..6 {
            for c in 0..CELLS {
                assert!(
                    (state[f][c] - REST[f]).abs() < 1e-6,
                    "step {step}: rest state drifted in field {} cell {c}: {}",
                    FIELD_NAMES[f],
                    state[f][c]
                );
            }
        }
    }
}

/// All 2³ reset patterns through one compiled artifact, with no recompile and
/// no rebind — the property that makes this capture-safe.
#[test]
fn every_reset_pattern_uses_one_compiled_graph() {
    let mut compiled = Session::new(Device::Cpu).compile(build(NW));
    let mut state = initial_state(NW);

    for bits in 0..(1u32 << NW) {
        let mask: Vec<f32> = (0..NW)
            .map(|w| if bits >> w & 1 == 1 { 1.0 } else { 0.0 })
            .collect();
        state = run(&mut compiled, &state, &mask);

        for w in 0..NW {
            let reset = mask[w] != 0.0;
            let rho = state[0][w * CELLS];
            assert!(rho.is_finite(), "pattern {bits:03b}: world {w} diverged");
            if reset {
                assert!(
                    (rho - 1.0).abs() < 1e-6,
                    "pattern {bits:03b}: world {w} should be at rest, rho = {rho}"
                );
            }
        }
    }
}
