// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Wall time of `Op::Fft` per algorithm on one device:
//!
//! ```sh
//! cargo run --release -p rlx-runtime --example fft_algo_bench --features "cpu metal" -- metal
//! ```
use rlx_ir::{DType, FftAlgo, FftNorm, Graph, Shape};
use rlx_runtime::{Device, Session};

fn main() {
    let dev = std::env::args()
        .nth(1)
        .map(|s| rlx_runtime::parse_device(&s).unwrap())
        .unwrap_or(Device::Cpu);
    for (rows, n) in [
        (1usize, 4096usize),
        (64, 1024),
        (16, 1000),
        (1, 4099),
        (256, 360),
    ] {
        let x: Vec<u8> = (0..rows * 2 * n)
            .flat_map(|k| ((k as f32 * 0.37).sin()).to_le_bytes())
            .collect();
        let mut line = format!("rows={rows:<4} n={n:<5}");
        for (name, algo) in [("native", FftAlgo::Native), ("pocketfft", FftAlgo::SCIPY)] {
            let mut g = Graph::new("b");
            let i = g.input("x", Shape::new(&[rows, 2 * n], DType::F32));
            let y = g.fft_algo(i, false, FftNorm::Backward, algo);
            g.set_outputs(vec![y]);
            let mut c = Session::new(dev).compile(g);
            for _ in 0..5 {
                c.run_typed(&[("x", &x, DType::F32)]);
            }
            let reps = 200;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                c.run_typed(&[("x", &x, DType::F32)]);
            }
            line += &format!(
                "  {name}: {:>8.1} µs",
                t.elapsed().as_secs_f64() * 1e6 / reps as f64
            );
        }
        println!("{line}");
    }
}
