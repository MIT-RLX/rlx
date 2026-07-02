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

//! `Op::Reverse` — batch-general sequence flip. Replaces the batch=1
//! reversed-index-gather workaround; the key property is that reversing the
//! seq axis flips each batch row *independently* (batch axis untouched).

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn run(device: Device, dims: &[usize], axes: Vec<usize>, x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("reverse");
    let inp = g.input("x", Shape::new(dims, DType::F32));
    let y = g.reverse(inp, axes);
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

/// Reference: reverse `x` (row-major `dims`) along `axes`.
fn reference(dims: &[usize], axes: &[usize], x: &[f32]) -> Vec<f32> {
    let rank = dims.len();
    let total: usize = dims.iter().product();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    let rev: Vec<bool> = (0..rank).map(|a| axes.contains(&a)).collect();
    let mut out = vec![0f32; total];
    for o in 0..total {
        let mut rem = o;
        let mut inf = 0usize;
        for ax in 0..rank {
            let idx = rem / strides[ax];
            rem %= strides[ax];
            let ii = if rev[ax] { dims[ax] - 1 - idx } else { idx };
            inf += ii * strides[ax];
        }
        out[o] = x[inf];
    }
    out
}

#[test]
fn reverse_batch_general_seq_axis() {
    // [batch=3, seq=4, feat=2] reversed on the seq axis: each batch row flips
    // independently. The old batch=1 gather would corrupt rows 1..3.
    let dims = [3usize, 4, 2];
    let x: Vec<f32> = (0..3 * 4 * 2).map(|i| i as f32).collect();
    let got = run(Device::Cpu, &dims, vec![1], &x);
    let want = reference(&dims, &[1], &x);
    assert_eq!(got, want, "seq-axis reverse must be batch-general");
    // Spot-check: batch 0 row order seq 0..3 → 3..0; batch 1 must use its own data.
    // x[b=1, s=0, :] = [8,9]; after reverse it lands at s=3 → got[1,3,:] == [8,9].
    let idx = |b: usize, s: usize, f: usize| (b * 4 + s) * 2 + f;
    assert_eq!(got[idx(1, 3, 0)], 8.0);
    assert_eq!(got[idx(1, 3, 1)], 9.0);
    assert_eq!(got[idx(2, 0, 0)], x[idx(2, 3, 0)]);
}

#[test]
fn reverse_multi_axis_and_identity() {
    let dims = [2usize, 3, 4];
    let x: Vec<f32> = (0..2 * 3 * 4).map(|i| (i * 3 % 7) as f32).collect();
    // Flip last two axes.
    assert_eq!(
        run(Device::Cpu, &dims, vec![1, 2], &x),
        reference(&dims, &[1, 2], &x)
    );
    // Empty axes = identity.
    assert_eq!(run(Device::Cpu, &dims, vec![], &x), x);
    // Flip all axes.
    assert_eq!(
        run(Device::Cpu, &dims, vec![0, 1, 2], &x),
        reference(&dims, &[0, 1, 2], &x)
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn reverse_mlx_matches_cpu() {
    let dims = [3usize, 4, 5];
    let x: Vec<f32> = (0..3 * 4 * 5).map(|i| (i % 11) as f32 * 0.5).collect();
    for axes in [vec![1], vec![0, 2], vec![2]] {
        assert_eq!(
            run(Device::Mlx, &dims, axes.clone(), &x),
            run(Device::Cpu, &dims, axes.clone(), &x),
            "mlx reverse axes={axes:?}"
        );
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn reverse_metal_matches_cpu() {
    let dims = [3usize, 4, 5];
    let x: Vec<f32> = (0..3 * 4 * 5).map(|i| (i % 11) as f32 * 0.5).collect();
    for axes in [vec![1], vec![0, 2], vec![2]] {
        assert_eq!(
            run(Device::Metal, &dims, axes.clone(), &x),
            run(Device::Cpu, &dims, axes.clone(), &x),
            "metal reverse axes={axes:?}"
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn reverse_wgpu_matches_cpu() {
    let dims = [3usize, 4, 5];
    let x: Vec<f32> = (0..3 * 4 * 5).map(|i| (i % 11) as f32 * 0.5).collect();
    for axes in [vec![1], vec![0, 2], vec![2]] {
        assert_eq!(
            run(Device::Gpu, &dims, axes.clone(), &x),
            run(Device::Cpu, &dims, axes.clone(), &x),
            "wgpu reverse axes={axes:?}"
        );
    }
}
