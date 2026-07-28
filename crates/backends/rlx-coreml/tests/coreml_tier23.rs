// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// ANE parity for native host RNN ops, unfused Gru (now native), and Sample.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn ane_gru_native_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: ANE unavailable");
        return;
    }
    let mut g = Graph::new("gru");
    let x = g.input("x", Shape::new(&[1, 4, 8], DType::F32));
    let w_ih = g.param("w_ih", Shape::new(&[24, 8], DType::F32));
    let w_hh = g.param("w_hh", Shape::new(&[24, 8], DType::F32));
    let b_ih = g.param("b_ih", Shape::new(&[24], DType::F32));
    let b_hh = g.param("b_hh", Shape::new(&[24], DType::F32));
    let y = g.gru(
        x,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        None,
        8,
        1,
        false,
        Shape::new(&[1, 4, 8], DType::F32),
    );
    g.set_outputs(vec![y]);

    let x_data = vec![0.1f32; 32];
    let w_ih_d = vec![0.01f32; 24 * 8];
    let w_hh_d = vec![0.02f32; 24 * 8];
    let b_d = vec![0.0f32; 24];

    let mut cpu = Session::new(Device::Cpu).compile(g.clone());
    cpu.set_param("w_ih", &w_ih_d);
    cpu.set_param("w_hh", &w_hh_d);
    cpu.set_param("b_ih", &b_d);
    cpu.set_param("b_hh", &b_d);
    let cpu_out = cpu.run(&[("x", &x_data)]).remove(0);

    let mut ane = Session::new(Device::Ane).compile(g);
    ane.set_param("w_ih", &w_ih_d);
    ane.set_param("w_hh", &w_hh_d);
    ane.set_param("b_ih", &b_d);
    ane.set_param("b_hh", &b_d);
    let ane_out = ane.run(&[("x", &x_data)]).remove(0);

    assert_eq!(cpu_out.len(), ane_out.len());
    for (a, b) in cpu_out.iter().zip(ane_out.iter()) {
        assert!((a - b).abs() < 1e-3, "cpu {a} vs ane {b}");
    }
}

#[test]
fn ane_sample_suffix_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: ANE unavailable");
        return;
    }
    let mut g = Graph::new("sample");
    let logits = g.input("logits", Shape::new(&[2, 16], DType::F32));
    let y = g.sample(logits, 4, 0.9, 1.0, 42, Shape::new(&[2], DType::F32));
    g.set_outputs(vec![y]);
    let logits_data: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();

    let mut cpu = Session::new(Device::Cpu).compile(g.clone());
    let cpu_out = cpu.run(&[("logits", &logits_data)]).remove(0);
    let mut ane = Session::new(Device::Ane).compile(g);
    let ane_out = ane.run(&[("logits", &logits_data)]).remove(0);
    assert_eq!(cpu_out, ane_out);
}

#[test]
fn ane_lstm_native_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: ANE unavailable");
        return;
    }
    let mut g = Graph::new("lstm");
    let x = g.input("x", Shape::new(&[1, 4, 8], DType::F32));
    let w_ih = g.param("w_ih", Shape::new(&[32, 8], DType::F32));
    let w_hh = g.param("w_hh", Shape::new(&[32, 8], DType::F32));
    let bias = g.param("bias", Shape::new(&[32], DType::F32));
    let y = g.append_node(
        rlx_ir::Op::Lstm {
            hidden_size: 8,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        vec![x, w_ih, w_hh, bias],
        Shape::new(&[1, 4, 8], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);

    let x_data = vec![0.1f32; 32];
    let w_ih_d = vec![0.01f32; 32 * 8];
    let w_hh_d = vec![0.02f32; 32 * 8];
    let b_d = vec![0.0f32; 32];

    let mut cpu = Session::new(Device::Cpu).compile(g.clone());
    cpu.set_param("w_ih", &w_ih_d);
    cpu.set_param("w_hh", &w_hh_d);
    cpu.set_param("bias", &b_d);
    let cpu_out = cpu.run(&[("x", &x_data)]).remove(0);

    let mut ane = Session::new(Device::Ane).compile(g);
    ane.set_param("w_ih", &w_ih_d);
    ane.set_param("w_hh", &w_hh_d);
    ane.set_param("bias", &b_d);
    let ane_out = ane.run(&[("x", &x_data)]).remove(0);

    assert_eq!(cpu_out.len(), ane_out.len());
    for (a, b) in cpu_out.iter().zip(ane_out.iter()) {
        assert!((a - b).abs() < 1e-3, "cpu {a} vs ane {b}");
    }
}

#[test]
fn ane_rnn_native_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: ANE unavailable");
        return;
    }
    let mut g = Graph::new("rnn");
    let x = g.input("x", Shape::new(&[1, 4, 8], DType::F32));
    let w_ih = g.param("w_ih", Shape::new(&[8, 8], DType::F32));
    let w_hh = g.param("w_hh", Shape::new(&[8, 8], DType::F32));
    let bias = g.param("bias", Shape::new(&[8], DType::F32));
    let y = g.append_node(
        rlx_ir::Op::Rnn {
            hidden_size: 8,
            num_layers: 1,
            bidirectional: false,
            carry: false,
            relu: false,
        },
        vec![x, w_ih, w_hh, bias],
        Shape::new(&[1, 4, 8], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);

    let x_data = vec![0.1f32; 32];
    let w_ih_d = vec![0.01f32; 64];
    let w_hh_d = vec![0.02f32; 64];
    let b_d = vec![0.0f32; 8];

    let mut cpu = Session::new(Device::Cpu).compile(g.clone());
    cpu.set_param("w_ih", &w_ih_d);
    cpu.set_param("w_hh", &w_hh_d);
    cpu.set_param("bias", &b_d);
    let cpu_out = cpu.run(&[("x", &x_data)]).remove(0);

    let mut ane = Session::new(Device::Ane).compile(g);
    ane.set_param("w_ih", &w_ih_d);
    ane.set_param("w_hh", &w_hh_d);
    ane.set_param("bias", &b_d);
    let ane_out = ane.run(&[("x", &x_data)]).remove(0);

    assert_eq!(cpu_out.len(), ane_out.len());
    for (a, b) in cpu_out.iter().zip(ane_out.iter()) {
        assert!((a - b).abs() < 1e-3, "cpu {a} vs ane {b}");
    }
}

#[test]
fn ane_mamba2_native_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: ANE unavailable");
        return;
    }
    let (b, s, h, p, n) = (1usize, 4, 3, 8, 16);
    let mut g = Graph::new("mamba2");
    let x = g.input("x", Shape::new(&[b, s, h, p], DType::F32));
    let dt = g.input("dt", Shape::new(&[b, s, h], DType::F32));
    let a = g.input("a", Shape::new(&[h], DType::F32));
    let bb = g.input("b", Shape::new(&[b, s, h, n], DType::F32));
    let cc = g.input("c", Shape::new(&[b, s, h, n], DType::F32));
    let y = g.mamba2(
        x,
        dt,
        a,
        bb,
        cc,
        p,
        n,
        Shape::new(&[b, s, h, p], DType::F32),
    );
    g.set_outputs(vec![y]);

    let x_data = vec![0.1f32; b * s * h * p];
    let dt_data = vec![0.2f32; b * s * h];
    let a_data: Vec<f32> = (0..h).map(|i| -0.5 - 0.1 * i as f32).collect();
    let b_data = vec![0.05f32; b * s * h * n];
    let c_data = vec![0.03f32; b * s * h * n];
    let inputs = [
        ("x", x_data.as_slice()),
        ("dt", dt_data.as_slice()),
        ("a", a_data.as_slice()),
        ("b", b_data.as_slice()),
        ("c", c_data.as_slice()),
    ];

    let mut cpu = Session::new(Device::Cpu).compile(g.clone());
    let cpu_out = cpu.run(&inputs).remove(0);

    let mut ane = Session::new(Device::Ane).compile(g);
    let ane_out = ane.run(&inputs).remove(0);

    assert_eq!(cpu_out.len(), ane_out.len());
    for (a, b) in cpu_out.iter().zip(ane_out.iter()) {
        assert!((a - b).abs() < 1e-3, "cpu {a} vs ane {b}");
    }
}
