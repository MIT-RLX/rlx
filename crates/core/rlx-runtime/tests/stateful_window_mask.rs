// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A stateful step's masks, run: `mask` admits keys `0..=position`, and a
//! windowed one (`StatefulDecode::windows`) only the last `window` of them —
//! a sliding-window layer's. Both read one `iota`: the rewrite makes one
//! param node per name (two would each need binding, and a backend binds a
//! name once — the plain mask then compared zeros and admitted every key).

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, GraphExt, Shape};

/// `mask + 2 · mask_local` at `position`, its cache `cap` rows.
fn masks(position: f32, window: usize, cap: usize) -> Vec<f32> {
    let build = |cap: usize| {
        let mut g = Graph::new("step");
        let past = g.input("past_k_0", Shape::new(&[1, cap - 1, 1], DType::F32));
        let one = g.constant(1.0, DType::F32);
        let row = g.reshape_(one, vec![1, 1, 1]);
        let cache = g.concat(vec![past, row], 1, Shape::new(&[1, cap, 1], DType::F32));
        let mask = g.input("mask", Shape::new(&[1, cap], DType::F32));
        let local = g.input("mask_local", Shape::new(&[1, cap], DType::F32));
        let two = g.constant(2.0, DType::F32);
        let local = g.mul(local, two);
        let y = g.add(mask, local);
        g.set_outputs(vec![y, cache]);
        g
    };
    let symbolic =
        rlx_ir::dynamic::generalize(&build(37), &build(53), &[(0, 37, 53), (1, 36, 52)]).unwrap();
    let spec = rlx_ir::stateful::StatefulDecode {
        position: "position".into(),
        tables: Vec::new(),
        mask: Some(("mask".into(), "iota".into(), 64)),
        cache_symbol: 0,
        tokens: 1,
        starts: None,
        rope_offset: None,
        windows: vec![("mask_local".into(), window)],
    };
    let step = rlx_ir::stateful::stateful_decode(&symbolic, &spec).unwrap();
    let mut at = rlx_ir::DimBinding::new();
    at.set(0, cap);
    let step = rlx_ir::dynamic::specialize(&step, &at);
    let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(step);
    let iota: Vec<f32> = (0..64).map(|i| i as f32).collect();
    c.set_param("iota", &iota);
    c.run(&[("position", &[position])]).remove(0)
}

#[test]
fn a_windowed_mask_admits_the_last_window_of_the_step_mask() {
    // Position 5, window 3: keys 0..=5, the last three of them local.
    assert_eq!(masks(5.0, 3, 8), [1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 0.0, 0.0]);
    // A window past the start: the two masks agree.
    assert_eq!(masks(2.0, 4, 8), [3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}
