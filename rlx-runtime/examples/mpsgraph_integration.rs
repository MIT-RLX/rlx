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

//! End-to-end MPSGraph integration: compile a small RLX graph with
//! `RLX_USE_MPSGRAPH=1` and verify it runs via the MPSGraph fast path.
//!
//! Graph: `out = gelu((x @ w) + b)` — a single FFN-style block.
//!
//! cargo run --example mpsgraph_integration --release \
//!   --features "cpu,metal,blas-accelerate"

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::*;
    use rlx_runtime::{Device, Session};

    let build = || {
        let mut g = Graph::new("ffn");
        let x = g.input("x", Shape::new(&[4, 8], DType::F32));
        let w = g.param("w", Shape::new(&[8, 16], DType::F32));
        let b = g.param("b", Shape::new(&[16], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[4, 16], DType::F32));
        let bias = g.binary(op::BinaryOp::Add, mm, b, Shape::new(&[4, 16], DType::F32));
        let out = g.activation(op::Activation::Gelu, bias, Shape::new(&[4, 16], DType::F32));
        g.set_outputs(vec![out]);
        g
    };

    let w_data: Vec<f32> = (0..8 * 16)
        .map(|i| {
            let row = i / 16;
            let col = i % 16;
            if col == row { 1.0 } else { 0.0 }
        })
        .collect();
    let b_data = vec![0.5f32; 16];
    let x_data: Vec<f32> = (0..4 * 8).map(|i| (i as f32) * 0.1).collect();

    let run_with = |use_mpsgraph: bool| -> Vec<f32> {
        if use_mpsgraph {
            rlx_ir::env::set("RLX_USE_MPSGRAPH", "1");
        } else {
            unsafe {
                rlx_ir::env::unset("RLX_USE_MPSGRAPH");
            }
        }
        let session = Session::new(Device::Metal);
        let mut compiled = session.compile(build());
        compiled.set_param("w", &w_data);
        compiled.set_param("b", &b_data);
        let outs = compiled.run(&[("x", &x_data)]);
        outs.into_iter().next().unwrap_or_default()
    };

    println!("══ Thunk path ══");
    let thunk_out = run_with(false);
    println!("first 8 values: {:?}", &thunk_out[..8]);

    println!("\n══ MPSGraph path ══");
    let mpsg_out = run_with(true);
    println!("first 8 values: {:?}", &mpsg_out[..8]);

    let max_err = thunk_out
        .iter()
        .zip(mpsg_out.iter())
        .map(|(t, g)| (t - g).abs())
        .fold(0f32, f32::max);
    println!("\nmax_err thunk vs MPSGraph: {:.2e}", max_err);
    // 1e-3 tolerance: independent matmul + GELU implementations differ by
    // ~ULP-scale per-element. We just want to confirm we're on the same
    // mathematical answer, not bit-identical.
    if max_err < 1e-3 {
        println!("✓ Both paths produce matching output (within 1e-3 tolerance)");
    } else {
        eprintln!("✗ paths diverge");
        std::process::exit(1);
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("requires --features metal on macOS");
}
