// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `Transpose` that a `MatMul` reads, also read back as a graph output — the
//! shape a per-node tap (`taps`) makes of every interior node — must come back
//! as the permuted values on every backend, not the un-permuted input (a view
//! the matmul folded into its strides).

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

fn run(device: Device, perm: &[usize], dims: &[usize], x: &[f32], w: &[f32]) -> Vec<Vec<f32>> {
    let f = DType::F32;
    let t_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
    let last = *t_dims.last().unwrap();
    let mut g = Graph::new("tapped_transpose");
    let xi = g.input("x", Shape::new(dims, f));
    let wi = g.input("w", Shape::new(&[last, 3], f));
    let a = g.mul(xi, xi);
    let t = g.transpose_(a, perm.to_vec());
    let m = g.mm(t, wi);
    let r = g.relu(m);
    // the model's output first, the tapped interior after it
    g.set_outputs(vec![r, a, t, m]);
    let mut c = Session::new(device).compile(g);
    c.finalize_params();
    c.run(&[("x", x), ("w", w)])
}

#[test]
fn tapped_transpose_output_matches_cpu() {
    #[cfg_attr(not(any(feature = "gpu", feature = "metal")), allow(unused_mut))]
    let mut devices: Vec<Device> = vec![];
    #[cfg(feature = "gpu")]
    if rlx_runtime::is_available(Device::Gpu) {
        devices.push(Device::Gpu);
    }
    #[cfg(feature = "metal")]
    if rlx_runtime::is_available(Device::Metal) {
        devices.push(Device::Metal);
    }
    if devices.is_empty() {
        eprintln!("no GPU backend — skipping");
        return;
    }
    let cases: [(&[usize], &[usize]); 3] = [
        (&[33, 17], &[1, 0]),
        (&[3, 9, 7], &[0, 2, 1]),
        (&[2, 5, 4, 6], &[0, 2, 1, 3]),
    ];
    for (dims, perm) in cases {
        let n: usize = dims.iter().product();
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.731).sin()).collect();
        let last = dims[*perm.last().unwrap()];
        let w: Vec<f32> = (0..last * 3)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.5)
            .collect();
        let want = run(Device::Cpu, perm, dims, &x, &w);
        for &d in &devices {
            let got = run(d, perm, dims, &x, &w);
            for (k, name) in ["relu", "x*x", "transpose", "matmul"].iter().enumerate() {
                let bad = got[k]
                    .iter()
                    .zip(&want[k])
                    .position(|(a, b)| (a - b).abs() > 1e-4 * (1.0 + b.abs()));
                assert!(
                    bad.is_none(),
                    "{d:?} {dims:?} {perm:?} {name}: first mismatch at {bad:?}"
                );
            }
        }
    }
}
