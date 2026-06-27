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

//! `Op::ArgMax`/`Op::ArgMin` cross-backend parity (f32-encoded indices). CPU is
//! the reference; argmin uses argmax-of-negated on MLX (first-hit tie-break).

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn run(
    device: Device,
    dims: &[usize],
    axis: usize,
    keep: bool,
    is_max: bool,
    x: &[f32],
) -> Vec<f32> {
    let mut g = Graph::new("argreduce");
    let inp = g.input("x", Shape::new(dims, DType::F32));
    // Output drops `axis` (or keeps as size 1 with keep_dim).
    let mut out_dims: Vec<usize> = dims.to_vec();
    if keep {
        out_dims[axis] = 1;
    } else {
        out_dims.remove(axis);
    }
    let out_shape = Shape::new(&out_dims, DType::F32);
    let y = if is_max {
        g.argmax(inp, axis, keep, out_shape)
    } else {
        g.argmin(inp, axis, keep, out_shape)
    };
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

fn cases() -> Vec<(&'static str, Vec<usize>, usize)> {
    vec![
        ("rows", vec![4, 8], 1),
        ("cols", vec![4, 8], 0),
        ("mid", vec![2, 5, 3], 1),
        ("last", vec![2, 3, 7], 2),
    ]
}

#[test]
fn argreduce_cpu_runs() {
    for (name, dims, axis) in cases() {
        let n: usize = dims.iter().product();
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.1)
            .collect();
        for is_max in [true, false] {
            let out = run(Device::Cpu, &dims, axis, false, is_max, &x);
            let bound = dims[axis] as f32;
            assert!(
                out.iter().all(|&v| v >= 0.0 && v < bound),
                "{name}: idx out of range"
            );
        }
    }
}

fn dataset(n: usize) -> Vec<f32> {
    // Distinct-ish values so argmax/argmin are unambiguous (avoids tie-break
    // divergence between backends).
    (0..n).map(|i| ((i * 13 + 1) % 97) as f32 * 0.5).collect()
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn argreduce_mlx_matches_cpu() {
    for (name, dims, axis) in cases() {
        let x = dataset(dims.iter().product());
        for is_max in [true, false] {
            for keep in [false, true] {
                let m = run(Device::Mlx, &dims, axis, keep, is_max, &x);
                let c = run(Device::Cpu, &dims, axis, keep, is_max, &x);
                assert_eq!(m, c, "mlx {name} max={is_max} keep={keep}");
            }
        }
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn argreduce_metal_matches_cpu() {
    for (name, dims, axis) in cases() {
        let x = dataset(dims.iter().product());
        for is_max in [true, false] {
            for keep in [false, true] {
                assert_eq!(
                    run(Device::Metal, &dims, axis, keep, is_max, &x),
                    run(Device::Cpu, &dims, axis, keep, is_max, &x),
                    "metal {name} max={is_max} keep={keep}"
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn argreduce_wgpu_matches_cpu() {
    for (name, dims, axis) in cases() {
        let x = dataset(dims.iter().product());
        for is_max in [true, false] {
            for keep in [false, true] {
                assert_eq!(
                    run(Device::Gpu, &dims, axis, keep, is_max, &x),
                    run(Device::Cpu, &dims, axis, keep, is_max, &x),
                    "wgpu {name} max={is_max} keep={keep}"
                );
            }
        }
    }
}
