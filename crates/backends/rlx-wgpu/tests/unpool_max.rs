// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `max(MaxPool3dBackward(x, dy), x)` — an unpool to the positions of `x`'s
//! window maxima, maxed with `x` (FastSurfer's decoders) — folds into one
//! `rlx_wgpu.unpool_max` (`unfuse::fold_unpool_max`), either operand order:
//! the same values as the two passes (`RLX_WGPU_NO_FOLD_UNPOOL_MAX=1`) and
//! as the unpool and `max` directly, ties in a window going to its first
//! maximum. A backward with another reader stays.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

#[test]
fn unpool_maxed_with_its_input_folds() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let (c, d, h, w) = (3, 4, 6, 10);
    let sh = |s: &[usize]| Shape::new(s, DType::F32);
    let xs = [1, c, d, h, w];
    let ys = [1, c, d, h / 2, w / 2];
    // a few equal values in a window: the first one wins
    let x: Vec<f32> = (0..c * d * h * w)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 9.0)
        .map(|v| if v.abs() < 1.0 { 0.5 } else { v })
        .collect();
    let dy: Vec<f32> = (0..c * d * (h / 2) * (w / 2))
        .map(|i| ((i * 13 % 47) as f32 - 23.0) / 5.0)
        .collect();
    let folds = |g: &Graph| {
        rlx_wgpu::unfuse::unfuse(g.clone())
            .nodes()
            .iter()
            .filter(|n| matches!(&n.op, Op::Custom { name, .. } if name == rlx_wgpu::unfuse::UNPOOL_MAX))
            .count()
    };
    for (swap, shared) in [(false, false), (true, false), (false, true)] {
        let build = || {
            let mut g = Graph::new("unpool_max");
            let xi = g.input("x", sh(&xs));
            let yi = g.input("dy", sh(&ys));
            let up = g.maxpool3d_backward(xi, yi, vec![1, 2, 2], vec![1, 2, 2], vec![0; 3]);
            let mut m = if swap {
                g.binary(BinaryOp::Max, xi, up, sh(&xs))
            } else {
                g.binary(BinaryOp::Max, up, xi, sh(&xs))
            };
            if shared {
                m = g.binary(BinaryOp::Add, m, up, sh(&xs));
            }
            g.set_outputs(vec![m]);
            g
        };
        let label = format!("swap {swap} shared {shared}");
        assert_eq!(folds(&build()), usize::from(!shared), "{label}: folds");
        let ins: [(&str, &[f32]); 2] = [("x", &x), ("dy", &dy)];
        let folded = WgpuExecutable::compile(build()).run(&ins).remove(0);
        rlx_ir::env::set("RLX_WGPU_NO_FOLD_UNPOOL_MAX", "1");
        assert_eq!(folds(&build()), 0, "{label}: no fold");
        let plain = WgpuExecutable::compile(build()).run(&ins).remove(0);
        rlx_ir::env::clear_overrides();
        assert_eq!(folded, plain, "{label}: folded = two passes");
        // directly: each window's first maximum takes its dy
        let mut up = vec![0f32; x.len()];
        for cc in 0..c {
            for z in 0..d {
                for oy in 0..h / 2 {
                    for ox in 0..w / 2 {
                        let at = |yy: usize, xx: usize| ((cc * d + z) * h + yy) * w + xx;
                        let mut best = at(2 * oy, 2 * ox);
                        for (yy, xx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                            let i = at(2 * oy + yy, 2 * ox + xx);
                            if x[i] > x[best] {
                                best = i;
                            }
                        }
                        up[best] += dy[((cc * d + z) * (h / 2) + oy) * (w / 2) + ox];
                    }
                }
            }
        }
        for (i, &o) in folded.iter().enumerate() {
            let m = up[i].max(x[i]);
            let want = if shared { m + up[i] } else { m };
            assert_eq!(o, want, "{label} at {i}");
        }
    }
}
