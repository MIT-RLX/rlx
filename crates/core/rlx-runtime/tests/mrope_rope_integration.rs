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

//! MRoPE feeds the existing per-token `Op::Rope` with no new op. This runs
//! `Op::Rope` end-to-end with a baked MRoPE table and checks (a) text positions
//! reduce to standard RoPE bit-for-bit, and (b) distinct 3-D vision positions
//! actually change the rotation.

#![cfg(feature = "cpu")]

use rlx_flow::{build_default_tables, build_mrope_tables};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::RopeStyle;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

/// Rope a `[1, seq, hidden]` input with an explicit `[seq, head_dim/2]` table.
fn rope_with_table(
    x: &[f32],
    seq: usize,
    hidden: usize,
    head_dim: usize,
    n_rot: usize,
    cos: &[f32],
    sin: &[f32],
) -> Vec<f32> {
    let half = head_dim / 2;
    let mut g = Graph::new("rope");
    let xin = g.input("x", Shape::new(&[1, seq, hidden], DType::F32));
    let c = g.input("cos", Shape::new(&[seq, half], DType::F32));
    let s = g.input("sin", Shape::new(&[seq, half], DType::F32));
    let y = g.rope_n_styled(xin, c, s, head_dim, n_rot, RopeStyle::NeoX);
    g.set_outputs(vec![y]);
    Session::new(Device::Cpu)
        .compile(g)
        .run(&[("x", x), ("cos", cos), ("sin", sin)])
        .pop()
        .unwrap()
}

#[test]
fn mrope_text_positions_match_standard_rope_through_op() {
    let (seq, head_dim, n_rot, hidden) = (4usize, 8usize, 8usize, 8usize);
    let x: Vec<f32> = (0..seq * hidden).map(|i| (i as f32) * 0.1 - 1.0).collect();

    // Text MRoPE: every modality shares the running scalar position; a single
    // section owns all pairs ⇒ identical to standard RoPE.
    let sections = [head_dim / 2, 0, 0, 0];
    let positions: Vec<[usize; 4]> = (0..seq).map(|p| [p, p, p, p]).collect();
    let (mc, ms) = build_mrope_tables(10_000.0, head_dim, n_rot, sections, &positions, false);
    let (dc, ds) = build_default_tables(10_000.0, head_dim, seq);

    let via_mrope = rope_with_table(&x, seq, hidden, head_dim, n_rot, &mc, &ms);
    let via_std = rope_with_table(&x, seq, hidden, head_dim, n_rot, &dc, &ds);
    assert_eq!(via_mrope.len(), via_std.len());
    for i in 0..via_mrope.len() {
        assert!(
            (via_mrope[i] - via_std[i]).abs() < 1e-5,
            "out[{i}]: mrope {} vs std {}",
            via_mrope[i],
            via_std[i]
        );
    }
}

#[test]
fn mrope_vision_positions_change_rotation() {
    let (seq, head_dim, n_rot, hidden) = (3usize, 8usize, 8usize, 8usize);
    let x: Vec<f32> = (0..seq * hidden).map(|i| (i as f32) * 0.07 + 0.3).collect();
    let sections = [1, 1, 2, 0]; // T,H,W over 4 pairs

    // Text-style monotonic positions vs a vision grid with distinct (t,h,w).
    let text: Vec<[usize; 4]> = (0..seq).map(|p| [p, p, p, p]).collect();
    let vision = [[0, 0, 0, 0], [0, 1, 2, 0], [0, 2, 1, 0]];
    let (tc, ts) = build_mrope_tables(10_000.0, head_dim, n_rot, sections, &text, false);
    let (vc, vs) = build_mrope_tables(10_000.0, head_dim, n_rot, sections, &vision, false);

    let text_out = rope_with_table(&x, seq, hidden, head_dim, n_rot, &tc, &ts);
    let vision_out = rope_with_table(&x, seq, hidden, head_dim, n_rot, &vc, &vs);
    // Token 0 shares position (0,0,0) in both ⇒ identical; later tokens differ.
    let hd = hidden;
    assert!(
        (0..hd).all(|i| (text_out[i] - vision_out[i]).abs() < 1e-6),
        "token 0 should match"
    );
    let diff: f32 = (hd..seq * hidden)
        .map(|i| (text_out[i] - vision_out[i]).abs())
        .sum();
    assert!(
        diff > 1e-3,
        "vision tokens should rotate differently, diff={diff}"
    );
}
