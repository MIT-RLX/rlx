// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Native GPU LSTM — multi-layer / bidirectional / carry parity vs CPU.
//!
//! LSTM (gate order i,f,g,o; single merged bias; h0+c0 carry) across the native
//! GPU paths: CUDA/ROCm (`lstm_dir` cuBLAS), Metal (single-layer MSL; multi-layer
//! host), and wgpu (native WGSL, one dispatch per (layer,dir), hidden ≤ 256 —
//! previously EVERY LSTM fell to host). Pins every geometry against the CPU
//! reference. `RLX_PARITY_DEVICE` selects the device (default cuda).

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
fn lstm_multilayer_bidir_native_matches_cpu() {
    let dev = target();
    if !is_available(dev) {
        eprintln!("skip lstm_multilayer_bidir_native ({dev:?} unavailable)");
        return;
    }
    let (b, s, inp, h) = (2usize, 6usize, 5usize, 16usize);
    // (num_layers, bidirectional, carry)
    let cases: [(usize, bool, bool); 6] = [
        (1, false, false), // single, unidir — the common case (was host on wgpu)
        (2, false, false), // stacked, unidir
        (1, true, false),  // single, bidir (reversed dir)
        (2, true, false),  // stacked + bidir
        (2, false, true),  // stacked + carry (h0 + c0)
        (3, true, true),   // deep + bidir + carry (full geometry)
    ];

    for (nl, bidir, carry) in cases {
        let dirs = if bidir { 2 } else { 1 };
        // 4 gates (LSTM); single merged bias.
        let ex = rlx_cpu::thunk::rnn_expected_lens(4, b, s, inp, h, nl, bidir);
        let x = mk(ex.x, 1);
        let wih = mk(ex.w_ih, 2);
        let whh = mk(ex.w_hh, 3);
        let bias = mk(ex.bias, 4);
        let h0 = mk(ex.state, 5); // num_layers·dirs·batch·hidden
        let c0 = mk(ex.state, 6);

        let build = || {
            let f = DType::F32;
            let mut g = Graph::new("lstm_mlbd");
            let xn = g.input("x", Shape::new(&[b, s, inp], f));
            let a = g.input("w_ih", Shape::new(&[ex.w_ih], f));
            let c = g.input("w_hh", Shape::new(&[ex.w_hh], f));
            let d = g.input("bias", Shape::new(&[ex.bias], f));
            let mut inputs = vec![xn, a, c, d];
            if carry {
                let hh = g.input("h0", Shape::new(&[nl * dirs, b, h], f));
                let cc = g.input("c0", Shape::new(&[nl * dirs, b, h], f));
                inputs.push(hh);
                inputs.push(cc);
            }
            let y = g.add_node(
                Op::Lstm {
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

        let base: [(&str, &[f32]); 4] =
            [("x", &x), ("w_ih", &wih), ("w_hh", &whh), ("bias", &bias)];
        let mut feed: Vec<(&str, &[f32])> = base.to_vec();
        if carry {
            feed.push(("h0", &h0));
            feed.push(("c0", &c0));
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
        eprintln!("[lstm-mlbd L={nl} bidir={bidir} carry={carry}] max|{dev:?}-CPU| = {maxd:.3e}");
        assert!(
            maxd < 1e-3,
            "native LSTM diverged from CPU (L={nl}, bidir={bidir}, carry={carry}): {maxd:.3e}"
        );
    }
}
