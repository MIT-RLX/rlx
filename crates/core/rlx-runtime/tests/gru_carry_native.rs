// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Native GPU GRU with CARRY (h0 initial state) parity vs CPU.
//!
//! Single-layer unidirectional GRU with `carry=true` used to take the host
//! round-trip (`Step::GruHost`, D2H→CPU→H2D) on CUDA/ROCm — a pure
//! dispatch-latency bubble for streaming/stateful GRU. The native `gru` kernel
//! now seeds its hidden state from `h0` (Tier-1). This pins the native carry
//! path against the CPU reference. Runs on the msi rig (`RLX_PARITY_DEVICE=cuda`,
//! default) and the amd rig (`RLX_PARITY_DEVICE=rocm`); no-ops without the GPU.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session, is_available};

fn mk(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 0.6 - 0.3
        })
        .collect()
}

fn target() -> Device {
    match std::env::var("RLX_PARITY_DEVICE") {
        Ok(s) => rlx_runtime::parse_device(&s).unwrap_or(Device::Cuda),
        Err(_) => Device::Cuda,
    }
}

#[test]
fn gru_carry_native_matches_cpu() {
    let dev = target();
    if !is_available(dev) {
        eprintln!("skip gru_carry_native ({dev:?} unavailable)");
        return;
    }
    // Single-layer, unidirectional, hidden ≤ 1024 → the NATIVE kernel path.
    let (b, s, inp, h) = (2usize, 6usize, 5usize, 16usize);
    let ex = rlx_cpu::thunk::rnn_expected_lens(3, b, s, inp, h, 1, false);
    let x = mk(ex.x, 1);
    let wih = mk(ex.w_ih, 2);
    let whh = mk(ex.w_hh, 3);
    let bih = mk(ex.bias, 4);
    let bhh = mk(ex.bias, 5);
    let h0 = mk(b * h, 6);

    let build = || {
        let f = DType::F32;
        let mut g = Graph::new("gru_carry");
        let x = g.input("x", Shape::new(&[b, s, inp], f));
        let a = g.input("w_ih", Shape::new(&[ex.w_ih], f));
        let c = g.input("w_hh", Shape::new(&[ex.w_hh], f));
        let d = g.input("b_ih", Shape::new(&[ex.bias], f));
        let e = g.input("b_hh", Shape::new(&[ex.bias], f));
        let hh = g.input("h0", Shape::new(&[b, h], f));
        let y = g.add_node(
            Op::Gru {
                hidden_size: h,
                num_layers: 1,
                bidirectional: false,
                carry: true,
            },
            vec![x, a, c, d, e, hh],
            Shape::new(&[b, s, h], f),
        );
        g.set_outputs(vec![y]);
        g
    };
    let feed: [(&str, &[f32]); 6] = [
        ("x", &x),
        ("w_ih", &wih),
        ("w_hh", &whh),
        ("b_ih", &bih),
        ("b_hh", &bhh),
        ("h0", &h0),
    ];

    let cpu = Session::new(Device::Cpu)
        .compile(build())
        .run(&feed)
        .remove(0);
    let gpu = Session::new(dev).compile(build()).run(&feed).remove(0);

    assert_eq!(cpu.len(), gpu.len(), "output len mismatch");
    let maxd = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[gru-carry] max|{dev:?}-CPU| = {maxd:.3e}");
    assert!(
        maxd < 1e-3,
        "native carry GRU diverged from CPU: {maxd:.3e}"
    );
}

/// Multi-layer / bidirectional (± carry) native GRU parity vs CPU.
///
/// Stacked and reversed-time GRU used to fall to `Step::GruHost` on CUDA/ROCm
/// (a full D2H→CPU→H2D bubble per op). The unified `gru` kernel now loops over
/// (layer, direction) on-device, ping-ponging intermediate layer outputs
/// through a scratch buffer. This pins every native geometry against CPU.
#[test]
fn gru_multilayer_bidir_native_matches_cpu() {
    let dev = target();
    if !is_available(dev) {
        eprintln!("skip gru_multilayer_bidir_native ({dev:?} unavailable)");
        return;
    }
    let (b, s, inp, h) = (2usize, 6usize, 5usize, 16usize);
    // (num_layers, bidirectional, carry)
    let cases: [(usize, bool, bool); 5] = [
        (2, false, false), // stacked, unidir
        (1, true, false),  // single, bidir (reversed dir)
        (2, true, false),  // stacked + bidir
        (2, false, true),  // stacked + carry
        (3, true, true),   // deep + bidir + carry (full geometry)
    ];

    for (nl, bidir, carry) in cases {
        let dirs = if bidir { 2 } else { 1 };
        let ex = rlx_cpu::thunk::rnn_expected_lens(3, b, s, inp, h, nl, bidir);
        let x = mk(ex.x, 1);
        let wih = mk(ex.w_ih, 2);
        let whh = mk(ex.w_hh, 3);
        let bih = mk(ex.bias, 4);
        let bhh = mk(ex.bias, 5);
        let h0 = mk(ex.state, 6); // num_layers·dirs·batch·hidden

        let build = || {
            let f = DType::F32;
            let mut g = Graph::new("gru_mlbd");
            let xn = g.input("x", Shape::new(&[b, s, inp], f));
            let a = g.input("w_ih", Shape::new(&[ex.w_ih], f));
            let c = g.input("w_hh", Shape::new(&[ex.w_hh], f));
            let d = g.input("b_ih", Shape::new(&[ex.bias], f));
            let e = g.input("b_hh", Shape::new(&[ex.bias], f));
            let mut inputs = vec![xn, a, c, d, e];
            if carry {
                let hh = g.input("h0", Shape::new(&[nl * dirs, b, h], f));
                inputs.push(hh);
            }
            let y = g.add_node(
                Op::Gru {
                    hidden_size: h,
                    num_layers: nl,
                    bidirectional: bidir,
                    carry,
                },
                inputs,
                Shape::new(&[b, s, dirs * h], f),
            );
            g.set_outputs(vec![y]);
            g
        };

        let base: [(&str, &[f32]); 5] = [
            ("x", &x),
            ("w_ih", &wih),
            ("w_hh", &whh),
            ("b_ih", &bih),
            ("b_hh", &bhh),
        ];
        let mut feed: Vec<(&str, &[f32])> = base.to_vec();
        if carry {
            feed.push(("h0", &h0));
        }

        let cpu = Session::new(Device::Cpu)
            .compile(build())
            .run(&feed)
            .remove(0);
        let gpu = Session::new(dev).compile(build()).run(&feed).remove(0);

        assert_eq!(
            cpu.len(),
            gpu.len(),
            "output len mismatch (L={nl}, bidir={bidir}, carry={carry})"
        );
        let maxd = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[gru-mlbd L={nl} bidir={bidir} carry={carry}] max|{dev:?}-CPU| = {maxd:.3e}");
        assert!(
            maxd < 1e-3,
            "native GRU diverged from CPU (L={nl}, bidir={bidir}, carry={carry}): {maxd:.3e}"
        );
    }
}
