// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::ConvTranspose2d` with groups, dilation and output padding, on every
//! backend built in, against PyTorch's definition computed here (not
//! against the CPU backend, which a shared bug would agree with).
//!
//! `conv_transpose2d_parity` covers groups = 1 only; grouped layers are what
//! ISTFTNet vocoders (Kokoro, StyleTTS2) use, and where MLX once mixed
//! channels across groups.

use rlx_ir::{DType, Graph, Shape};
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

/// One case: input `[n, c_in, h, w]`, kernel, `c_out / groups`, stride,
/// padding, dilation, output padding, groups.
type Case = (
    [usize; 4],
    [usize; 2],
    usize,
    [usize; 2],
    [usize; 2],
    [usize; 2],
    [usize; 2],
    usize,
);

/// Every input sample scatters its kernel into the output, within its group.
fn reference(x: &[f32], w: &[f32], c: &Case) -> (Vec<f32>, [usize; 4]) {
    let &([n, c_in, h, wd], [kh, kw], cog, s, p, d, op, g) = c;
    let ho = (h - 1) * s[0] + d[0] * (kh - 1) + op[0] + 1 - 2 * p[0];
    let wo = (wd - 1) * s[1] + d[1] * (kw - 1) + op[1] + 1 - 2 * p[1];
    let (cig, c_out) = (c_in / g, cog * g);
    let mut y = vec![0f64; n * c_out * ho * wo];
    for b in 0..n {
        for ci in 0..c_in {
            for i in 0..h {
                for j in 0..wd {
                    let xv = x[((b * c_in + ci) * h + i) * wd + j] as f64;
                    for o in 0..cog {
                        for a in 0..kh {
                            for e in 0..kw {
                                let oi = (i * s[0] + a * d[0]) as isize - p[0] as isize;
                                let oj = (j * s[1] + e * d[1]) as isize - p[1] as isize;
                                if oi < 0 || oj < 0 || oi >= ho as isize || oj >= wo as isize {
                                    continue;
                                }
                                let co = (ci / cig) * cog + o;
                                let wv = w[((ci * cog + o) * kh + a) * kw + e] as f64;
                                y[((b * c_out + co) * ho + oi as usize) * wo + oj as usize] +=
                                    xv * wv;
                            }
                        }
                    }
                }
            }
        }
    }
    (
        y.into_iter().map(|v| v as f32).collect(),
        [n, c_out, ho, wo],
    )
}

#[test]
fn grouped_conv_transpose2d_matches_reference() {
    let cases: [Case; 6] = [
        ([1, 4, 3, 5], [3, 3], 3, [2, 2], [1, 1], [1, 1], [1, 0], 2),
        ([2, 6, 4, 4], [2, 3], 1, [1, 2], [0, 1], [2, 1], [0, 1], 3),
        // ISTFTNet-style depthwise upsampler, length on W.
        ([1, 8, 1, 37], [1, 3], 1, [1, 2], [0, 1], [1, 1], [0, 1], 8),
        ([1, 4, 5, 5], [3, 3], 2, [3, 3], [2, 0], [1, 1], [2, 2], 4),
        ([1, 2, 6, 6], [2, 2], 2, [2, 2], [2, 2], [1, 1], [1, 1], 2),
        ([1, 4, 4, 1], [3, 1], 1, [1, 1], [1, 0], [1, 1], [0, 0], 1),
    ];
    for c in &cases {
        let (shape, k, cog, s, p, d, op, grp) = *c;
        let w_shape = [shape[1], cog, k[0], k[1]];
        let xv: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|i| ((i as f32) * 0.37).sin())
            .collect();
        let wv: Vec<f32> = (0..w_shape.iter().product::<usize>())
            .map(|i| ((i as f32) * 0.53).cos())
            .collect();
        let (want, out) = reference(&xv, &wv, c);
        for dev in devices() {
            let mut g = Graph::new("ct2d_grouped");
            let x = g.input("x", Shape::new(&shape, DType::F32));
            let w = g.input("w", Shape::new(&w_shape, DType::F32));
            let y = g.conv_transpose2d(x, w, k, s, p, d, op, grp);
            g.set_outputs(vec![y]);
            let dims: Vec<usize> = g
                .node(y)
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            assert_eq!(dims, out.to_vec(), "{shape:?} k={k:?}: output shape");
            let got = Session::new(dev)
                .compile(g)
                .run(&[("x", &xv), ("w", &wv)])
                .swap_remove(0);
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                got.len() == want.len() && err < 1e-4,
                "{dev:?} {shape:?} k={k:?} s={s:?} p={p:?} d={d:?} op={op:?} g={grp}: max |d| {err}"
            );
        }
    }
}
