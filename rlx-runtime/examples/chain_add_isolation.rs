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

//! Minimal repro: a + b + c where each is a fresh input. No gather, no
//! matmul, no LN. Just three inputs and two chained adds.
//!
//! cargo run --example chain_add_isolation --release \
//!   --features "cpu,blas-accelerate" -p rlx-runtime

#[cfg(feature = "cpu")]
fn main() {
    for &n in &[1, 2, 4, 8, 16, 32, 100] {
        run_size(n);
    }
}

#[cfg(feature = "cpu")]
fn run_size(n: usize) {
    use rlx_ir::*;
    use rlx_runtime::{Device, Session};

    let build = || {
        let mut g = Graph::new("chain");
        let a = g.input("a", Shape::new(&[n], DType::F32));
        let b = g.input("b", Shape::new(&[n], DType::F32));
        let c = g.input("c", Shape::new(&[n], DType::F32));
        let s1 = g.binary(op::BinaryOp::Add, a, b, Shape::new(&[n], DType::F32));
        let s2 = g.binary(op::BinaryOp::Add, s1, c, Shape::new(&[n], DType::F32));
        g.set_outputs(vec![s2]);
        g
    };

    let a_data: Vec<f32> = vec![1.0; n];
    let b_data: Vec<f32> = vec![10.0; n];
    let c_data: Vec<f32> = vec![100.0; n];

    use rlx_runtime::CompileOptions;
    let session = Session::new(Device::Cpu);
    let opts = CompileOptions::new()
        .with_dce(false)
        .with_constant_folding(false);
    let mut compiled = session.compile_with(build(), &opts);
    let outs = compiled.run(&[("a", &a_data), ("b", &b_data), ("c", &c_data)]);
    let out = &outs[0];
    let all_111 = out.iter().all(|&v| (v - 111.0).abs() < 0.01);
    let status = if all_111 { "✓" } else { "✗" };
    println!(
        "n={n:3}  out[0]={:>5.1}  out[..3]={:?}  {status} (no dce/cf)",
        out[0],
        &out[..3.min(out.len())]
    );
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("requires cpu");
}
