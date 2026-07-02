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
//! `Op::Lstm` correctness: (1) the native CPU kernel vs an independent
//! reference recurrence, and (2) the `unfuse` decomposition (the path
//! MLX / CoreML / TPU and autodiff take) vs the native kernel.

use rlx_ir::*;
use rlx_runtime::{Device, Session};

const B: usize = 2;
const S: usize = 4;
const IN: usize = 3;
const H: usize = 5;

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Independent host reference (gate order i, f, g, o; h0 = c0 = 0).
fn reference_lstm(x: &[f32], w_ih: &[f32], w_hh: &[f32], bias: &[f32]) -> Vec<f32> {
    let four_h = 4 * H;
    let mut y = vec![0f32; B * S * H];
    for bi in 0..B {
        let mut hs = [0f32; H];
        let mut cs = [0f32; H];
        for t in 0..S {
            let xt = &x[(bi * S + t) * IN..(bi * S + t + 1) * IN];
            let mut z = vec![0f32; four_h];
            for (r, zr) in z.iter_mut().enumerate() {
                let mut acc = bias[r];
                for (j, &xj) in xt.iter().enumerate() {
                    acc += w_ih[r * IN + j] * xj;
                }
                for (j, &hj) in hs.iter().enumerate() {
                    acc += w_hh[r * H + j] * hj;
                }
                *zr = acc;
            }
            for k in 0..H {
                let i_g = sigmoid(z[k]);
                let f_g = sigmoid(z[H + k]);
                let g_g = z[2 * H + k].tanh();
                let o_g = sigmoid(z[3 * H + k]);
                let c_new = f_g * cs[k] + i_g * g_g;
                cs[k] = c_new;
                let h_new = o_g * c_new.tanh();
                hs[k] = h_new;
                y[(bi * S + t) * H + k] = h_new;
            }
        }
    }
    y
}

fn build_lstm_graph() -> Graph {
    let four_h = 4 * H;
    let mut g = Graph::new("lstm");
    let x = g.input("x", Shape::new(&[B, S, IN], DType::F32));
    let w_ih = g.input("w_ih", Shape::new(&[four_h, IN], DType::F32));
    let w_hh = g.input("w_hh", Shape::new(&[four_h, H], DType::F32));
    let bias = g.input("bias", Shape::new(&[four_h], DType::F32));
    let y = g.add_node(
        Op::Lstm {
            hidden_size: H,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        vec![x, w_ih, w_hh, bias],
        Shape::new(&[B, S, H], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let four_h = 4 * H;
    let x: Vec<f32> = (0..B * S * IN)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1)
        .collect();
    let w_ih: Vec<f32> = (0..four_h * IN)
        .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.05)
        .collect();
    let w_hh: Vec<f32> = (0..four_h * H)
        .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.05)
        .collect();
    let bias: Vec<f32> = (0..four_h).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
    (x, w_ih, w_hh, bias)
}

#[test]
fn lstm_cpu_native_matches_reference() {
    let (x, w_ih, w_hh, bias) = inputs();
    let expected = reference_lstm(&x, &w_ih, &w_hh, &bias);

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(build_lstm_graph());
    let actual = compiled
        .run(&[
            ("x", x.as_slice()),
            ("w_ih", w_ih.as_slice()),
            ("w_hh", w_hh.as_slice()),
            ("bias", bias.as_slice()),
        ])
        .pop()
        .unwrap();

    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        assert!(
            (actual[i] - expected[i]).abs() < 1e-5,
            "native LSTM mismatch at {i}: {} vs {}",
            actual[i],
            expected[i]
        );
    }
}

fn run_device_case(device: Device) {
    let (x, w_ih, w_hh, bias) = inputs();
    let expected = reference_lstm(&x, &w_ih, &w_hh, &bias);
    let session = Session::new(device);
    let mut compiled = session.compile(build_lstm_graph());
    let actual = compiled
        .run(&[
            ("x", x.as_slice()),
            ("w_ih", w_ih.as_slice()),
            ("w_hh", w_hh.as_slice()),
            ("bias", bias.as_slice()),
        ])
        .pop()
        .unwrap();
    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        assert!(
            (actual[i] - expected[i]).abs() < 1e-4,
            "{device:?} LSTM mismatch at {i}: {} vs {}",
            actual[i],
            expected[i]
        );
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn lstm_metal_matches_reference() {
    run_device_case(Device::Metal);
}

// Larger hidden size stresses the native MSL kernel's threadgroup barriers
// (one thread per hidden unit, shared h_prev) well beyond the H=5 case.
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn lstm_metal_large_hidden_matches_cpu() {
    let (bb, ss, inn, hh) = (3usize, 6usize, 10usize, 100usize);
    let four_h = 4 * hh;
    let x: Vec<f32> = (0..bb * ss * inn)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.03)
        .collect();
    let w_ih: Vec<f32> = (0..four_h * inn)
        .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.02)
        .collect();
    let w_hh: Vec<f32> = (0..four_h * hh)
        .map(|i| ((i * 3 % 13) as f32 - 6.0) * 0.02)
        .collect();
    let bias: Vec<f32> = (0..four_h).map(|i| ((i % 7) as f32 - 3.0) * 0.02).collect();

    let mk_graph = || {
        let mut g = Graph::new("lstm_big");
        let xi = g.input("x", Shape::new(&[bb, ss, inn], DType::F32));
        let wih = g.input("w_ih", Shape::new(&[four_h, inn], DType::F32));
        let whh = g.input("w_hh", Shape::new(&[four_h, hh], DType::F32));
        let bs = g.input("bias", Shape::new(&[four_h], DType::F32));
        let y = g.add_node(
            Op::Lstm {
                hidden_size: hh,
                num_layers: 1,
                bidirectional: false,
                carry: false,
            },
            vec![xi, wih, whh, bs],
            Shape::new(&[bb, ss, hh], DType::F32),
        );
        g.set_outputs(vec![y]);
        g
    };
    let run = |dev: Device| {
        let mut c = Session::new(dev).compile(mk_graph());
        c.run(&[
            ("x", x.as_slice()),
            ("w_ih", w_ih.as_slice()),
            ("w_hh", w_hh.as_slice()),
            ("bias", bias.as_slice()),
        ])
        .pop()
        .unwrap()
    };
    let cpu = run(Device::Cpu);
    let metal = run(Device::Metal);
    assert_eq!(cpu.len(), metal.len());
    for i in 0..cpu.len() {
        assert!(
            (cpu[i] - metal[i]).abs() < 1e-4,
            "metal native LSTM vs cpu mismatch at {i}: {} vs {}",
            metal[i],
            cpu[i]
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn lstm_wgpu_matches_reference() {
    run_device_case(Device::Gpu);
}

#[test]
fn lstm_unfuse_decomposition_matches_native() {
    // The decomposed graph is what MLX / CoreML / TPU (no native LSTM)
    // and the autodiff pass run. It must match the fused kernel exactly.
    let (x, w_ih, w_hh, bias) = inputs();
    let expected = reference_lstm(&x, &w_ih, &w_hh, &bias);

    let decomposed = rlx_fusion::unfuse::unfuse_fused_for_autodiff(build_lstm_graph());
    assert!(
        !decomposed
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Lstm { .. })),
        "Op::Lstm should be decomposed away by unfuse"
    );

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(decomposed);
    let actual = compiled
        .run(&[
            ("x", x.as_slice()),
            ("w_ih", w_ih.as_slice()),
            ("w_hh", w_hh.as_slice()),
            ("bias", bias.as_slice()),
        ])
        .pop()
        .unwrap();

    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        assert!(
            (actual[i] - expected[i]).abs() < 1e-4,
            "decomposed LSTM mismatch at {i}: {} vs {}",
            actual[i],
            expected[i]
        );
    }
}
