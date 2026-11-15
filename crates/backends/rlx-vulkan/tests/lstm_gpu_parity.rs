// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Lstm` on the GPU path (`lstm_proj` + `lstm_rec` per layer, scratch
//! after the arena) vs CPU.
//!
//! The cases are the ones that break it: a bidirectional layer the size of
//! PyLaia's over a 320-frame line (the reverse direction walks t backwards),
//! two bidirectional layers with a batch (layer 1 reads layer 0 from
//! scratch, `W_ih` blocks of two widths), three unidirectional layers with a
//! hidden size not a multiple of 16 (the layer buffers alternate), and the
//! 512-unit ceiling. `RLX_VULKAN_LSTM=host` runs them on the host fallback.

use rlx_ir::{DType, Graph, Shape};
use rlx_vulkan::backend::VulkanExecutable;

mod common;

fn skip() -> bool {
    rlx_ir::env::skip_unless_device("vulkan", true, rlx_vulkan::is_available())
}

fn data(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i * 2654435761 + seed * 40503) % 2001) as f32 / 1000.0) - 1.0) * scale)
        .collect()
}

fn check(batch: usize, seq: usize, input: usize, hidden: usize, layers: usize, bidir: bool) {
    let f = DType::F32;
    let dirs = if bidir { 2 } else { 1 };
    let four_h = 4 * hidden;
    let wih_n: usize = (0..layers)
        .map(|l| dirs * four_h * if l == 0 { input } else { dirs * hidden })
        .sum();
    let whh_n = layers * dirs * four_h * hidden;
    let bias_n = layers * dirs * four_h;
    let build = || {
        let mut g = Graph::new("lstm");
        let x = g.input("x", Shape::new(&[batch, seq, input], f));
        let w_ih = g.input("w_ih", Shape::new(&[wih_n], f));
        let w_hh = g.input("w_hh", Shape::new(&[whh_n], f));
        let b = g.input("bias", Shape::new(&[bias_n], f));
        let y = g.lstm(
            x,
            w_ih,
            w_hh,
            b,
            hidden,
            layers,
            bidir,
            Shape::new(&[batch, seq, dirs * hidden], f),
        );
        g.set_outputs(vec![y]);
        g
    };
    // Weights ~ 1/√h so the gates stay out of saturation.
    let ws = 1.0 / (hidden as f32).sqrt();
    let x = data(batch * seq * input, 1, 1.0);
    let (wih, whh, bias) = (data(wih_n, 2, ws), data(whh_n, 3, ws), data(bias_n, 4, 0.1));
    let inputs: Vec<(&str, &[f32])> =
        vec![("x", &x), ("w_ih", &wih), ("w_hh", &whh), ("bias", &bias)];
    let want = {
        use rlx::prelude::*;
        Session::new(Device::Cpu)
            .compile(build())
            .run(&inputs)
            .remove(0)
    };
    let got = VulkanExecutable::compile(build()).run(&inputs).remove(0);
    assert_eq!(got.len(), want.len());
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-4,
        "lstm b={batch} s={seq} in={input} h={hidden} L={layers} bidir={bidir}: \
         max |vk − cpu| = {worst:e}"
    );
}

#[test]
fn bidirectional_line_sized() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(1, 320, 512, 256, 1, true);
}

#[test]
fn two_bidirectional_layers_batched() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(3, 17, 20, 48, 2, true);
}

#[test]
fn three_unidirectional_layers_odd_hidden() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(2, 11, 7, 100, 3, false);
}

#[test]
fn widest_hidden() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(1, 9, 64, 512, 1, true);
}
