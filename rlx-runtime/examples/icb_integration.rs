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

//! End-to-end ICB integration: build an element-wise graph (no matmul),
//! compile through Session on Metal with `RLX_USE_ICB=1`, verify the
//! output matches the regular thunk path.
//!
//! cargo run --example icb_integration --release \
//!   --features "cpu,metal,blas-accelerate" -p rlx-runtime

#[cfg(all(feature = "cpu", feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::*;
    use rlx_runtime::{Device, Session};

    let n = 256;
    let build = || {
        let mut g = Graph::new("icb_test");
        let a = g.input("a", Shape::new(&[n], DType::F32));
        let b = g.input("b", Shape::new(&[n], DType::F32));
        let c = g.input("c", Shape::new(&[n], DType::F32));
        let g_a = g.activation(op::Activation::Gelu, a, Shape::new(&[n], DType::F32));
        let mul = g.binary(op::BinaryOp::Mul, g_a, b, Shape::new(&[n], DType::F32));
        let out = g.binary(op::BinaryOp::Add, mul, c, Shape::new(&[n], DType::F32));
        g.set_outputs(vec![out]);
        g
    };

    // Dump graph for inspection.
    {
        let g = build();
        println!("Graph has {} nodes:", g.len());
        for n in g.nodes() {
            println!("  [{}] {} {:?}", n.id, n.op, n.inputs);
        }
        println!("  outputs: {:?}", g.outputs);
        println!();
    }

    let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let b_data: Vec<f32> = vec![2.0; n];
    let c_data: Vec<f32> = vec![1.0; n];

    let run = |use_icb: bool, dev: Device| -> Vec<f32> {
        if use_icb {
            rlx_ir::env::set("RLX_USE_ICB", "1");
        } else {
            unsafe {
                rlx_ir::env::unset("RLX_USE_ICB");
            }
        }
        let session = Session::new(dev);
        let mut compiled = session.compile(build());
        let outs = compiled.run(&[("a", &a_data), ("b", &b_data), ("c", &c_data)]);
        outs.into_iter().next().unwrap_or_default()
    };

    // Index 100: a=1.0 → expect gelu(1.0)*2+1 ≈ 2.68
    // Index 200: a=2.0 → expect gelu(2.0)*2+1 ≈ 4.90
    let print_v = |label: &str, v: &[f32]| {
        println!(
            "{}: a[100]={:.4}  out[100]={:.4}  a[200]={:.4}  out[200]={:.4}",
            label, 1.0, v[100], 2.0, v[200]
        );
    };

    println!("══ CPU thunk (ground truth) ══");
    let cpu_out = run(false, Device::Cpu);
    print_v("CPU", &cpu_out);

    println!("\n══ Metal thunk path ══");
    let thunk_out = run(false, Device::Metal);
    print_v("Metal-thunk", &thunk_out);

    println!("\n══ Metal ICB path (RLX_USE_ICB=1) ══");
    let icb_out = run(true, Device::Metal);
    print_v("Metal-ICB", &icb_out);

    let diff = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    println!(
        "\nmax_err CPU vs Metal-thunk: {:.3e}",
        diff(&cpu_out, &thunk_out)
    );
    println!(
        "max_err CPU vs Metal-ICB:   {:.3e}",
        diff(&cpu_out, &icb_out)
    );
    println!(
        "max_err Metal-thunk vs ICB: {:.3e}",
        diff(&thunk_out, &icb_out)
    );

    if diff(&cpu_out, &icb_out) < 1e-4 {
        println!("\n✓ ICB integration matches CPU reference");
    } else {
        eprintln!("\n✗ ICB diverges from CPU reference");
        std::process::exit(1);
    }
}

#[cfg(not(all(feature = "cpu", feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("requires cpu + metal on macOS");
}
