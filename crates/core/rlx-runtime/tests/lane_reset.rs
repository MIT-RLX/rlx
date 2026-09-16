// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-lane reset as data: one compiled graph, many different reset patterns.
//!
//! The claim this file exists to check is not "`Op::Where` selects correctly" —
//! it is that an *arbitrary sequence* of resets runs through a **single**
//! compiled artifact, with nothing rebound and no shape change. That is what
//! makes the pattern capture-safe: a schedule captured on step 1 is still valid
//! on step 500 no matter which lanes finished in between.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::lanes::LaneExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const LANES: usize = 4;
const WIDTH: usize = 3;

/// `state' = where(mask, 0, state + 1)` — a trivial per-lane step counter, so a
/// lane's value is exactly the number of steps since it was last reset.
fn build() -> Graph {
    let mut g = Graph::new("lane_reset");
    let state = g.input("state", Shape::new(&[LANES, WIDTH], DType::F32));
    let mask = g.input("reset", Shape::new(&[LANES], DType::F32));
    let one = g.full(&[LANES, WIDTH], 1.0, DType::F32);
    let stepped = g.add(state, one);
    let out = g.reset_lanes_to_(stepped, 0.0, mask);
    g.set_outputs(vec![out]);
    g
}

#[test]
fn one_compiled_graph_serves_every_reset_pattern() {
    let mut compiled = Session::new(Device::Cpu).compile(build());

    let mut state = vec![0.0f32; LANES * WIDTH];
    // Independent bookkeeping: what each lane's counter should be.
    let mut expect = [0.0f32; LANES];

    // Every one of the 16 subsets of 4 lanes, twice over — so each lane is reset
    // and left alone in every combination, all through the same artifact.
    for round in 0..32u32 {
        let bits = round % 16;
        let mask: Vec<f32> = (0..LANES)
            .map(|l| if bits >> l & 1 == 1 { 1.0 } else { 0.0 })
            .collect();

        let out = compiled
            .run(&[("state", &state), ("reset", &mask)])
            .pop()
            .unwrap();

        for l in 0..LANES {
            expect[l] = if mask[l] != 0.0 { 0.0 } else { expect[l] + 1.0 };
            for w in 0..WIDTH {
                assert_eq!(
                    out[l * WIDTH + w],
                    expect[l],
                    "round {round}: lane {l} col {w} (mask {mask:?})"
                );
            }
        }
        state = out;
    }
}

/// Lanes must not leak into each other: resetting lane 1 leaves lanes 0, 2, 3
/// exactly as they were, bit for bit.
#[test]
fn reset_is_confined_to_masked_lanes() {
    let mut compiled = Session::new(Device::Cpu).compile(build());
    // Distinct values per lane so a cross-lane write is visible.
    let state: Vec<f32> = (0..LANES * WIDTH).map(|i| 10.0 + i as f32).collect();

    for target in 0..LANES {
        let mask: Vec<f32> = (0..LANES)
            .map(|l| if l == target { 1.0 } else { 0.0 })
            .collect();
        let out = compiled
            .run(&[("state", &state), ("reset", &mask)])
            .pop()
            .unwrap();
        for l in 0..LANES {
            for w in 0..WIDTH {
                let got = out[l * WIDTH + w];
                let want = if l == target {
                    0.0
                } else {
                    state[l * WIDTH + w] + 1.0
                };
                assert_eq!(got, want, "resetting lane {target} disturbed lane {l}");
            }
        }
    }
}

/// An all-zero mask is a no-op and an all-ones mask resets everything — the two
/// degenerate patterns a masked implementation is most likely to special-case
/// wrongly.
#[test]
fn empty_and_full_masks_behave() {
    let mut compiled = Session::new(Device::Cpu).compile(build());
    let state: Vec<f32> = (0..LANES * WIDTH).map(|i| i as f32).collect();

    let none = vec![0.0f32; LANES];
    let out = compiled
        .run(&[("state", &state), ("reset", &none)])
        .pop()
        .unwrap();
    for i in 0..LANES * WIDTH {
        assert_eq!(out[i], state[i] + 1.0, "empty mask should reset nothing");
    }

    let all = vec![1.0f32; LANES];
    let out = compiled
        .run(&[("state", &state), ("reset", &all)])
        .pop()
        .unwrap();
    assert!(out.iter().all(|v| *v == 0.0), "full mask should reset all");
}

/// Any nonzero counts as "reset" — the mask is a predicate, not a scale factor.
/// A `mul`-based implementation would silently produce `0.5 * state` here.
#[test]
fn mask_is_a_predicate_not_a_multiplier() {
    let mut compiled = Session::new(Device::Cpu).compile(build());
    let state = vec![5.0f32; LANES * WIDTH];
    let mask = vec![0.5f32, -1.0, 0.0, 2.0];
    let out = compiled
        .run(&[("state", &state), ("reset", &mask)])
        .pop()
        .unwrap();
    let want = [0.0f32, 0.0, 6.0, 0.0];
    for l in 0..LANES {
        for w in 0..WIDTH {
            assert_eq!(
                out[l * WIDTH + w],
                want[l],
                "lane {l} with mask {}",
                mask[l]
            );
        }
    }
}

/// The multi-field form: one mask driving several state tensors, which is the
/// real shape of the problem (six moment fields, or a K and a V cache).
#[test]
fn multi_field_reset_shares_one_mask() {
    let mut g = Graph::new("multi");
    let shape = Shape::new(&[LANES, WIDTH], DType::F32);
    let a = g.input("a", shape.clone());
    let b = g.input("b", shape.clone());
    let mask = g.input("reset", Shape::new(&[LANES], DType::F32));
    let za = g.zeros(&[LANES, WIDTH], DType::F32);
    let zb = g.full(&[LANES, WIDTH], -1.0, DType::F32);
    let outs = g.reset_lanes_many_(&[a, b], &[za, zb], mask);
    g.set_outputs(outs);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let va: Vec<f32> = (0..LANES * WIDTH).map(|i| i as f32 + 1.0).collect();
    let vb: Vec<f32> = (0..LANES * WIDTH).map(|i| 100.0 + i as f32).collect();
    let mask = vec![0.0f32, 1.0, 0.0, 1.0];

    let out = compiled.run(&[("a", &va), ("b", &vb), ("reset", &mask)]);
    assert_eq!(out.len(), 2);
    for l in 0..LANES {
        for w in 0..WIDTH {
            let i = l * WIDTH + w;
            if mask[l] != 0.0 {
                assert_eq!(out[0][i], 0.0, "a lane {l}");
                assert_eq!(out[1][i], -1.0, "b lane {l}");
            } else {
                assert_eq!(out[0][i], va[i], "a lane {l}");
                assert_eq!(out[1][i], vb[i], "b lane {l}");
            }
        }
    }
}
