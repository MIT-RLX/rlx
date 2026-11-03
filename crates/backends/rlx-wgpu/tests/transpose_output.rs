// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `Transpose` read back as a graph output must come back materialized —
//! the permuted values in row-major order — also when a `MatMul` reads the
//! same transpose (where the backend may fold it into the matmul's strides
//! and never write it out).

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

fn permute(x: &[f32], dims: &[usize], perm: &[usize]) -> Vec<f32> {
    let out_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
    let stride = |d: &[usize]| {
        let mut s = vec![1usize; d.len()];
        for i in (0..d.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * d[i + 1];
        }
        s
    };
    let (is, os) = (stride(dims), stride(&out_dims));
    (0..x.len())
        .map(|o| {
            let src: usize = (0..perm.len())
                .map(|a| (o / os[a]) % out_dims[a] * is[perm[a]])
                .sum();
            x[src]
        })
        .collect()
}

#[test]
fn transpose_output_is_materialized() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // (input dims, perm): a matrix transpose, a batched one, a head split.
    let cases: [(&[usize], &[usize]); 3] = [
        (&[33, 17], &[1, 0]),
        (&[3, 9, 7], &[0, 2, 1]),
        (&[2, 5, 4, 6], &[0, 2, 1, 3]),
    ];
    for (dims, perm) in cases {
        let n: usize = dims.iter().product();
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.731).sin()).collect();
        let t_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
        let want = permute(&x, dims, perm);
        let last = *t_dims.last().unwrap();
        for with_matmul in [false, true] {
            let mut g = Graph::new("transpose_out");
            let xi = g.input("x", Shape::new(dims, DType::F32));
            let t = g.add_node(
                Op::Transpose {
                    perm: perm.to_vec(),
                },
                vec![xi],
                Shape::new(&t_dims, DType::F32),
            );
            let mut outs = vec![t];
            if with_matmul {
                // t · W, W = identity-ish [last, 3]: reads the transpose.
                let w: Vec<f32> = (0..last * 3)
                    .map(|i| ((i % 3) as f32 + 1.0) * 0.5)
                    .collect();
                let wi = g.input("w", Shape::new(&[last, 3], DType::F32));
                let mut md = t_dims.clone();
                *md.last_mut().unwrap() = 3;
                let m = g.matmul(t, wi, Shape::new(&md, DType::F32));
                outs.push(m);
                g.set_outputs(outs);
                let got = WgpuExecutable::compile(g).run(&[("x", &x), ("w", &w)]);
                assert_eq!(
                    got[0], want,
                    "{dims:?} {perm:?} with matmul: transpose output"
                );
                let rows = want.len() / last;
                for r in 0..rows {
                    for c in 0..3 {
                        let e: f32 = (0..last).map(|k| want[r * last + k] * w[k * 3 + c]).sum();
                        let o = got[1][r * 3 + c];
                        assert!(
                            (o - e).abs() <= 1e-4 * (1.0 + e.abs()),
                            "{dims:?} {perm:?} mm[{r},{c}] {o} vs {e}"
                        );
                    }
                }
            } else {
                g.set_outputs(outs);
                let got = WgpuExecutable::compile(g).run(&[("x", &x)]);
                assert_eq!(got[0], want, "{dims:?} {perm:?}: transpose output");
            }
        }
    }
}
