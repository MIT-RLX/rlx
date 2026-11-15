// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Cumsum` over a long row against an f64 prefix sum, on every backend
//! built in.
//!
//! The row is a vocoder's phase increments (f0 / sample rate, 80k samples).
//! One running f32 sum adds those small steps to a total in the hundreds and
//! drifts by ~2.5e-2 (0.16 rad of phase); every backend now stays within
//! 1e-3. Tolerances are per backend: CPU and CUDA/ROCm accumulate in f64,
//! the others scan in chunks.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session, is_available};

#[allow(clippy::vec_init_then_push)]
fn devices() -> Vec<Device> {
    let mut v = vec![Device::Cpu];
    #[cfg(all(feature = "metal", target_os = "macos"))]
    v.push(Device::Metal);
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    v.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    v.push(Device::Gpu);
    #[cfg(feature = "cuda")]
    v.push(Device::Cuda);
    #[cfg(feature = "rocm")]
    v.push(Device::Rocm);
    #[cfg(feature = "vulkan")]
    v.push(Device::Vulkan);
    v.retain(|&d| is_available(d));
    v
}

#[test]
fn long_row_cumsum_matches_f64() {
    let (rows, n) = (2usize, 81_600usize);
    let x: Vec<f32> = (0..rows * n)
        .map(|i| {
            let t = (i % n) as f64;
            ((150.0 + 150.0 * (t * 1e-4 + (i / n) as f64).sin().abs()) / 24_000.0) as f32
        })
        .collect();
    for exclusive in [false, true] {
        let mut want = vec![0f64; rows * n];
        for r in 0..rows {
            let mut acc = 0f64;
            for c in 0..n {
                let v = x[r * n + c] as f64;
                if exclusive {
                    want[r * n + c] = acc;
                    acc += v;
                } else {
                    acc += v;
                    want[r * n + c] = acc;
                }
            }
        }
        for dev in devices() {
            let mut g = Graph::new("cumsum_long_row");
            let i = g.input("x", Shape::new(&[rows, n], DType::F32));
            let y = g.add_node(
                Op::Cumsum { axis: 1, exclusive },
                vec![i],
                Shape::new(&[rows, n], DType::F32),
            );
            g.set_outputs(vec![y]);
            let mut c = Session::new(dev).compile(g);
            let got = c.run(&[("x", &x)]).swap_remove(0);
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0f64, f64::max);
            eprintln!("{dev:?} exclusive={exclusive}: max |d| {err:.2e}");
            assert!(
                err < 1e-3,
                "{dev:?} exclusive={exclusive}: max |d| {err:.2e}"
            );
        }
    }
}
