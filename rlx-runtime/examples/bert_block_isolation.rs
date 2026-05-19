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

//! One BERT-style transformer block end-to-end via MPSGraph vs thunks.
//!
//! Mirrors rlx-models's BERT layer construction:
//!   QKV proj (matmul + add) → narrow ×3 → attention → out proj → +residual → LN
//!   FFN1 (matmul + add + GELU) → FFN2 (matmul + add) → +residual → LN
//!
//! cargo run --example bert_block_isolation --release \
//!   --features "cpu,metal,blas-accelerate" -p rlx-runtime

#[cfg(all(feature = "cpu", feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::*;
    use rlx_runtime::{Device, Session};

    let b = 4;
    let s = 8;
    let nh = 12;
    let dh = 32;
    let h = nh * dh;
    let int_dim = 4 * h;

    let build = || {
        let mut g = Graph::new("block");
        let x = g.input("x", Shape::new(&[b, s, h], DType::F32));
        let mask = g.input("mask", Shape::new(&[b, s], DType::F32));

        let qkv_w = g.param("qkv_w", Shape::new(&[h, 3 * h], DType::F32));
        let qkv_b = g.param("qkv_b", Shape::new(&[3 * h], DType::F32));
        let out_w = g.param("out_w", Shape::new(&[h, h], DType::F32));
        let out_b = g.param("out_b", Shape::new(&[h], DType::F32));
        let ln1_g = g.param("ln1_g", Shape::new(&[h], DType::F32));
        let ln1_b = g.param("ln1_b", Shape::new(&[h], DType::F32));
        let fc1_w = g.param("fc1_w", Shape::new(&[h, int_dim], DType::F32));
        let fc1_b = g.param("fc1_b", Shape::new(&[int_dim], DType::F32));
        let fc2_w = g.param("fc2_w", Shape::new(&[int_dim, h], DType::F32));
        let fc2_b = g.param("fc2_b", Shape::new(&[h], DType::F32));
        let ln2_g = g.param("ln2_g", Shape::new(&[h], DType::F32));
        let ln2_b = g.param("ln2_b", Shape::new(&[h], DType::F32));

        let qkv_mm = g.mm(x, qkv_w);
        let qkv = g.add(qkv_mm, qkv_b);
        let q = g.narrow_(qkv, 2, 0, h);
        let k = g.narrow_(qkv, 2, h, h);
        let v = g.narrow_(qkv, 2, 2 * h, h);

        let attn = g.attention_(q, k, v, mask, nh, dh);
        let attn_o = g.mm(attn, out_w);
        let attn_o = g.add(attn_o, out_b);
        let res1 = g.add(x, attn_o);
        let h1 = g.layer_norm(
            res1,
            ln1_g,
            ln1_b,
            -1,
            1e-12,
            Shape::new(&[b, s, h], DType::F32),
        );

        let fc1 = g.mm(h1, fc1_w);
        let fc1 = g.add(fc1, fc1_b);
        let fc1 = g.activation(
            op::Activation::Gelu,
            fc1,
            Shape::new(&[b, s, int_dim], DType::F32),
        );
        let fc2 = g.mm(fc1, fc2_w);
        let fc2 = g.add(fc2, fc2_b);
        let res2 = g.add(h1, fc2);
        let out = g.layer_norm(
            res2,
            ln2_g,
            ln2_b,
            -1,
            1e-12,
            Shape::new(&[b, s, h], DType::F32),
        );
        g.set_outputs(vec![out]);
        g
    };

    // Reproducible param data.
    let fill = |n: usize, seed: f32| -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.01).sin() * 0.05)
            .collect()
    };
    let x_data = fill(b * s * h, 0.0);
    let mask_data: Vec<f32> = (0..b * s)
        .map(|i| if (i % s) < 6 { 1.0 } else { 0.0 })
        .collect();
    let qkv_w_data = fill(h * 3 * h, 1.0);
    let qkv_b_data = vec![0.0f32; 3 * h];
    let out_w_data = fill(h * h, 2.0);
    let out_b_data = vec![0.0f32; h];
    let ln1_g_data = vec![1.0f32; h];
    let ln1_b_data = vec![0.0f32; h];
    let fc1_w_data = fill(h * int_dim, 3.0);
    let fc1_b_data = vec![0.0f32; int_dim];
    let fc2_w_data = fill(int_dim * h, 4.0);
    let fc2_b_data = vec![0.0f32; h];
    let ln2_g_data = vec![1.0f32; h];
    let ln2_b_data = vec![0.0f32; h];

    let run_with = |use_mpsg: bool, dev: Device| -> Vec<f32> {
        if use_mpsg {
            unsafe {
                std::env::set_var("RLX_USE_MPSGRAPH", "1");
            }
            unsafe {
                std::env::set_var("RLX_MPSGRAPH_ATTENTION", "1");
            }
        } else {
            unsafe {
                std::env::remove_var("RLX_USE_MPSGRAPH");
            }
            unsafe {
                std::env::remove_var("RLX_MPSGRAPH_ATTENTION");
            }
        }
        let session = Session::new(dev);
        let mut compiled = session.compile(build());
        for (n, d) in [
            ("qkv_w", &qkv_w_data),
            ("qkv_b", &qkv_b_data),
            ("out_w", &out_w_data),
            ("out_b", &out_b_data),
            ("ln1_g", &ln1_g_data),
            ("ln1_b", &ln1_b_data),
            ("fc1_w", &fc1_w_data),
            ("fc1_b", &fc1_b_data),
            ("fc2_w", &fc2_w_data),
            ("fc2_b", &fc2_b_data),
            ("ln2_g", &ln2_g_data),
            ("ln2_b", &ln2_b_data),
        ] {
            compiled.set_param(n, d);
        }
        let outs = compiled.run(&[("x", &x_data), ("mask", &mask_data)]);
        outs.into_iter().next().unwrap_or_default()
    };

    let cpu = run_with(false, Device::Cpu);
    let metal_thunk = run_with(false, Device::Metal);
    let metal_mpsg = run_with(true, Device::Metal);

    println!("CPU thunk[..6]:    {:?}", &cpu[..6]);
    println!("Metal thunk[..6]:  {:?}", &metal_thunk[..6]);
    println!("Metal MPSG[..6]:   {:?}", &metal_mpsg[..6]);

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
    eprintln!("requires cpu + metal on macOS");
}
