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

//! basic test: build a simple Linear+GELU graph, compile for Metal,
//! run it, and compare output against CPU.

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::*;
    use rlx_runtime::{Device, Session};

    // y = gelu(x @ W + b)
    let build = || {
        let mut g = Graph::new("basic");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 3], DType::F32));
        let b = g.param("b", Shape::new(&[3], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
        let bias = g.binary(op::BinaryOp::Add, mm, b, Shape::new(&[2, 3], DType::F32));
        let out = g.activation(op::Activation::Gelu, bias, Shape::new(&[2, 3], DType::F32));
        g.set_outputs(vec![out]);
        g
    };

    let w_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let b_data = vec![0.5, -0.5, 0.0];
    let x_data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    // CPU reference
    let cpu_session = Session::new(Device::Cpu);
    let mut cpu = cpu_session.compile(build());
    cpu.set_param("w", &w_data);
    cpu.set_param("b", &b_data);
    let cpu_out = cpu.run(&[("x", &x_data)]);

    // Metal compile + run
    let metal_session = Session::new(Device::Metal);
    let mut metal = metal_session.compile(build());
    metal.set_param("w", &w_data);
    metal.set_param("b", &b_data);
    let metal_out = metal.run(&[("x", &x_data)]);

    println!("CPU   : {:?}", cpu_out[0]);
    println!("Metal : {:?}", metal_out[0]);

    let max_err: f32 = cpu_out[0]
        .iter()
        .zip(metal_out[0].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("max_err: {:.2e}", max_err);
    if max_err < 1e-4 {
        println!("PASS — Metal backend matches CPU within fp32 tolerance");
    } else {
        eprintln!("FAIL — Metal output differs from CPU");
        std::process::exit(1);
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("This example requires --features metal on macOS");
}
