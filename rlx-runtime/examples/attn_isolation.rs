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

//! Build a graph with ONLY Op::Attention. Compile on CPU and on Metal+MPSGraph,
//! compare outputs. Tiny shapes so we can read the values.
//!
//! cargo run --example attn_isolation --release \
//!   --features "cpu,metal,blas-accelerate" -p rlx-runtime

#[cfg(all(feature = "cpu", feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::*;
    use rlx_runtime::{Device, Session};

    // BERT-like shapes: B=4, S=8, NH=12, DH=32 → hidden=384 (MiniLM-L6).
    let b = 4;
    let s = 8;
    let nh = 12;
    let dh = 32;
    let h = nh * dh;

    let build = || {
        let mut g = Graph::new("attn");
        let q = g.input("q", Shape::new(&[b, s, h], DType::F32));
        let k = g.input("k", Shape::new(&[b, s, h], DType::F32));
        let v = g.input("v", Shape::new(&[b, s, h], DType::F32));
        let mask = g.input("mask", Shape::new(&[b, s], DType::F32));
        let out = g.attention_(q, k, v, mask, nh, dh);
        g.set_outputs(vec![out]);
        g
    };

    // Synthetic data: simple ramps so any reshape error is visible.
    let q_data: Vec<f32> = (0..b * s * h)
        .map(|i| ((i as f32) * 0.01).sin() * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..b * s * h)
        .map(|i| ((i as f32) * 0.02).cos() * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..b * s * h)
        .map(|i| ((i as f32) * 0.03).sin() * 0.1)
        .collect();
    // BERT-style padding pattern: each of 4 batches has different pad lengths.
    let mut mask_data = Vec::with_capacity(b * s);
    for bi in 0..b {
        let valid = match bi {
            0 => 6,
            1 => 5,
            2 => 8,
            3 => 4,
            _ => s,
        };
        for j in 0..s {
            mask_data.push(if j < valid { 1.0 } else { 0.0 });
        }
    }

    let run_with = |use_mpsgraph: bool, dev: Device| -> Vec<f32> {
        if use_mpsgraph {
            rlx_ir::env::set("RLX_USE_MPSGRAPH", "1");
            rlx_ir::env::set("RLX_MPSGRAPH_ATTENTION", "1");
        } else {
            unsafe {
                rlx_ir::env::unset("RLX_USE_MPSGRAPH");
            }
            unsafe {
                rlx_ir::env::unset("RLX_MPSGRAPH_ATTENTION");
            }
        }
        let session = Session::new(dev);
        let mut compiled = session.compile(build());
        let outs = compiled.run(&[
            ("q", &q_data),
            ("k", &k_data),
            ("v", &v_data),
            ("mask", &mask_data),
        ]);
        outs.into_iter().next().unwrap_or_default()
    };

    let cpu = run_with(false, Device::Cpu);
    let metal_thunk = run_with(false, Device::Metal);
    let metal_mpsg = run_with(true, Device::Metal);

    println!("CPU thunk:    {:?}", cpu);
    println!("Metal thunk:  {:?}", metal_thunk);
    println!("Metal MPSG:   {:?}", metal_mpsg);

    let diff = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    println!(
        "\nmax_err CPU vs Metal-thunk: {:.3e}",
        diff(&cpu, &metal_thunk)
    );
    println!(
        "max_err CPU vs Metal-MPSG:  {:.3e}",
        diff(&cpu, &metal_mpsg)
    );
    println!(
        "max_err Metal-thunk vs MPSG:{:.3e}",
        diff(&metal_thunk, &metal_mpsg)
    );
}

#[cfg(not(all(feature = "cpu", feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("requires cpu + metal features on macOS");
}
