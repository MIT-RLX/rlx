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

//! Phase-A canary: build the Nomic-style SwiGLU FFN sub-pattern
//! (matmul with concatenated weights → narrow×2 → silu → mul), compile
//! for both CPU and Metal, and assert outputs match within fp32 tolerance.
//!
//! Exercises the `FuseSwiGLU` pass + the new Metal `fused_swiglu` kernel.

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::op::{Activation, BinaryOp};
    use rlx_ir::*;
    use rlx_runtime::{Device, Session};

    // Builds: cat = mm(x, W) ; up = narrow(cat, 0, n) ; gate = narrow(cat, n, n)
    //         ; out = up * silu(gate)
    // Where W is [k, 2n] — modeling a `concat(fc11_w, fc12_w)` weight matrix.
    let m = 6usize;
    let k = 8usize;
    let n = 4usize;

    // Build via TWO separate matmuls sharing input — exercises the full
    // FuseSharedInputMatMul → Concat → FuseSwiGLU pipeline that fires on
    // real models like Nomic. Concatenated weights are split into halves.
    let build = || {
        let mut g = Graph::new("swiglu_parity");
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let w_up = g.param("w_up", Shape::new(&[k, n], DType::F32));
        let w_gate = g.param("w_gate", Shape::new(&[k, n], DType::F32));
        let up_mm = g.matmul(x, w_up, Shape::new(&[m, n], DType::F32));
        let gate_mm = g.matmul(x, w_gate, Shape::new(&[m, n], DType::F32));
        let gate = g.activation(Activation::Silu, gate_mm, Shape::new(&[m, n], DType::F32));
        let out = g.binary(BinaryOp::Mul, up_mm, gate, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![out]);
        g
    };

    // Deterministic data — small offsets so silu's exp doesn't explode.
    let x_data: Vec<f32> = (0..m * k)
        .map(|i| ((i * 13 + 7) % 23) as f32 / 23.0 - 0.5)
        .collect();
    let w_up_data: Vec<f32> = (0..k * n)
        .map(|i| ((i * 17 + 3) % 31) as f32 / 31.0 - 0.5)
        .collect();
    let w_gate_data: Vec<f32> = (0..k * n)
        .map(|i| ((i * 19 + 11) % 29) as f32 / 29.0 - 0.5)
        .collect();

    let cpu_session = Session::new(Device::Cpu);
    let mut cpu = cpu_session.compile(build());
    cpu.set_param("w_up", &w_up_data);
    cpu.set_param("w_gate", &w_gate_data);
    let cpu_out = cpu.run(&[("x", &x_data)]);

    let metal_session = Session::new(Device::Metal);
    let mut metal = metal_session.compile(build());
    metal.set_param("w_up", &w_up_data);
    metal.set_param("w_gate", &w_gate_data);
    let metal_out = metal.run(&[("x", &x_data)]);

    println!("CPU    : {:?}", cpu_out[0]);
    println!("Metal  : {:?}", metal_out[0]);

    assert_eq!(cpu_out[0].len(), m * n, "CPU output shape");
    assert_eq!(metal_out[0].len(), m * n, "Metal output shape");

    let max_err: f32 = cpu_out[0]
        .iter()
        .zip(metal_out[0].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("max_err: {:.2e}", max_err);
    if max_err < 1e-4 {
        println!("PASS — FusedSwiGLU lowering on Metal matches CPU within fp32 tolerance");
    } else {
        eprintln!("FAIL — output differs (max_err={max_err:.6e})");
        std::process::exit(1);
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("swiglu_parity requires --features metal on macOS");
}
