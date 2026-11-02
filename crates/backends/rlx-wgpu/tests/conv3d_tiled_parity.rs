// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-wgpu parity for the forward 3-D convolution kernels, aimed at the
//! edges of `conv3d_ctile_w4c8` / `_w4` / `_w2` (eight output channels past
//! four, four / two W positions per thread; `RLX_WGPU_CONV3D_NO_C8=1`,
//! `RLX_WGPU_CONV3D_W2=1` and `RLX_WGPU_CONV3D_NO_W2=1` select the others): output
//! widths that leave the last group short (1, 2, 3, 5, 6, 7, …), stride and
//! dilation (the neighbours' inputs are `sw` apart), padding wider than the
//! volume, `c_out` not a multiple of four, and the bias added in the store
//! (`FusedConvBiasAct`, folded into the conv).

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

fn cpu_run(g: &Graph, inputs: &[(&str, Vec<f32>)]) -> Vec<f32> {
    let plan = rlx_opt::memory::plan_memory(g);
    let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
    let sched = rlx_cpu::thunk::compile_thunks(g, &arena);
    for node in g.nodes() {
        if let Op::Input { name } = &node.op {
            let data = &inputs.iter().find(|(n, _)| n == name).unwrap().1;
            let off = arena.byte_offset(node.id);
            unsafe {
                let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
                for (i, &v) in data.iter().enumerate() {
                    *p.add(i) = v;
                }
            }
        }
    }
    rlx_cpu::thunk::execute_thunks(&sched, arena.raw_buf_mut());
    let out = g.outputs[0];
    let n = g.shape(out).num_elements().unwrap();
    let off = arena.byte_offset(out);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..n).map(|i| *p.add(i)).collect()
    }
}

fn ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 2654435761 + seed * 40503) % 1013) as f32 / 1013.0) * 2.0 - 1.0)
        .collect()
}

/// `(c_in, c_out, [d, h, w], k, stride, pad, dil)`
fn case(c_in: usize, c_out: usize, dims: [usize; 3], k: usize, s: usize, p: usize, dil: usize) {
    let o = |l: usize| (l + 2 * p - dil * (k - 1) - 1) / s + 1;
    let out = [1, c_out, o(dims[0]), o(dims[1]), o(dims[2])];
    let x = ramp(c_in * dims.iter().product::<usize>(), 3);
    let w = ramp(c_out * c_in * k * k * k, 7);
    let b: Vec<f32> = (0..c_out).map(|i| 0.25 * i as f32 - 0.5).collect();
    let sh = |d: &[usize]| Shape::new(d, DType::F32);

    let mut g = Graph::new("conv3d");
    let xi = g.input("x", sh(&[1, c_in, dims[0], dims[1], dims[2]]));
    let wi = g.input("w", sh(&[c_out, c_in, k, k, k]));
    let y = g.conv3d(xi, wi, [s; 3], [p; 3], [dil; 3], 1);
    g.set_outputs(vec![y]);
    let want = cpu_run(&g, &[("x", x.clone()), ("w", w.clone())]);
    let got = WgpuExecutable::compile(g)
        .run(&[("x", &x), ("w", &w)])
        .remove(0);
    let tol = 1e-4 * (c_in * k * k * k) as f32;
    let label = format!("c {c_in}->{c_out} {dims:?} k{k} s{s} p{p} d{dil}");
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (a, e)) in got.iter().zip(&want).enumerate() {
        assert!((a - e).abs() <= tol, "{label} [{i}]: wgpu {a} cpu {e}");
    }

    // the same with a bias, folded into the conv's store
    let mut g = Graph::new("conv3d_bias");
    let xi = g.input("x", sh(&[1, c_in, dims[0], dims[1], dims[2]]));
    let wi = g.input("w", sh(&[c_out, c_in, k, k, k]));
    let bi = g.input("b", sh(&[c_out]));
    let y = g.add_node(
        Op::FusedConvBiasAct {
            kernel_size: vec![k; 3],
            stride: vec![s; 3],
            padding: vec![p; 3],
            dilation: vec![dil; 3],
            groups: 1,
            activation: None,
            has_residual: false,
        },
        vec![xi, wi, bi],
        sh(&out),
    );
    g.set_outputs(vec![y]);
    let got = WgpuExecutable::compile(g)
        .run(&[("x", &x), ("w", &w), ("b", &b)])
        .remove(0);
    let per = out[2] * out[3] * out[4];
    for (i, (a, e)) in got.iter().zip(&want).enumerate() {
        let e = e + b[i / per];
        assert!(
            (a - e).abs() <= tol,
            "{label} biased [{i}]: wgpu {a} cpu {e}"
        );
    }
}

#[test]
fn odd_width_and_channel_remainders() {
    case(3, 5, [4, 5, 7], 3, 1, 1, 1);
    case(2, 6, [3, 4, 1], 3, 1, 1, 1);
    case(3, 4, [3, 3, 2], 3, 1, 1, 1);
    case(2, 9, [3, 4, 3], 3, 1, 1, 1);
    case(5, 8, [4, 3, 6], 3, 1, 1, 1);
    // eight-channel tiles: MindGrab's 15 (one short), and 17 (a tile of one)
    case(15, 15, [5, 4, 9], 3, 1, 4, 4);
    case(3, 17, [3, 4, 5], 3, 1, 1, 1);
}

#[test]
fn strided_and_dilated() {
    case(3, 4, [6, 7, 9], 3, 2, 1, 1);
    case(4, 7, [6, 6, 11], 3, 1, 2, 2);
    case(2, 3, [5, 5, 13], 3, 2, 4, 3);
}

#[test]
fn padding_wider_than_the_volume() {
    // MeshNet's dilation 16 on a small window: whole rows of taps in padding
    case(3, 4, [6, 5, 7], 3, 1, 16, 16);
}

/// An activation reading a biased conv is applied in the conv's store
/// (`fold_conv_activation`) — unless the conv has another reader, which then
/// needs the value before the activation.
#[test]
fn activation_folded_into_the_store() {
    use rlx_ir::op::Activation;
    let (c_in, c_out, dims, k, dil) = (3, 6, [4, 5, 7], 3, 2);
    let x = ramp(c_in * dims.iter().product::<usize>(), 3);
    let w = ramp(c_out * c_in * k * k * k, 7);
    let b: Vec<f32> = (0..c_out).map(|i| 0.25 * i as f32 - 0.5).collect();
    let sh = |d: &[usize]| Shape::new(d, DType::F32);
    let out = sh(&[1, c_out, dims[0], dims[1], dims[2]]);
    // CPU: the plain conv, the bias, then the activation on its own
    let mut g = Graph::new("conv3d");
    let xi = g.input("x", sh(&[1, c_in, dims[0], dims[1], dims[2]]));
    let wi = g.input("w", sh(&[c_out, c_in, k, k, k]));
    let y = g.conv3d(xi, wi, [1; 3], [dil; 3], [dil; 3], 1);
    g.set_outputs(vec![y]);
    let per = dims.iter().product::<usize>();
    let z: Vec<f32> = cpu_run(&g, &[("x", x.clone()), ("w", w.clone())])
        .iter()
        .enumerate()
        .map(|(i, v)| v + b[i / per])
        .collect();
    let reference = |act| {
        let mut g = Graph::new("act");
        let zi = g.input("z", out.clone());
        let a = g.activation(act, zi, out.clone());
        g.set_outputs(vec![a]);
        cpu_run(&g, &[("z", z.clone())])
    };
    for act in [Activation::Relu, Activation::Elu, Activation::Gelu] {
        for both in [false, true] {
            let mut g = Graph::new("conv3d_act");
            let xi = g.input("x", sh(&[1, c_in, dims[0], dims[1], dims[2]]));
            let wi = g.input("w", sh(&[c_out, c_in, k, k, k]));
            let bi = g.input("b", sh(&[c_out]));
            let y = g.add_node(
                Op::FusedConvBiasAct {
                    kernel_size: vec![k; 3],
                    stride: vec![1; 3],
                    padding: vec![dil; 3],
                    dilation: vec![dil; 3],
                    groups: 1,
                    activation: None,
                    has_residual: false,
                },
                vec![xi, wi, bi],
                out.clone(),
            );
            let a = g.activation(act, y, out.clone());
            // a second reader of the conv keeps it unfolded
            g.set_outputs(if both { vec![a, y] } else { vec![a] });
            let want = reference(act);
            let got = WgpuExecutable::compile(g)
                .run(&[("x", &x), ("w", &w), ("b", &b)])
                .remove(0);
            let tol = 1e-4 * (c_in * k * k * k) as f32;
            assert_eq!(got.len(), want.len());
            for (i, (a, e)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - e).abs() <= tol,
                    "{act:?} both={both} [{i}]: wgpu {a} cpu {e}"
                );
            }
        }
    }
}

/// SynthStrip's LeakyReLU, `max(y, α·y)` on a biased conv, applied in the
/// conv's store (`CONV3D_LEAKY`); with the scaled value also an output it
/// cannot fold.
#[test]
fn leaky_relu_folded_into_the_store() {
    use rlx_ir::op::BinaryOp;
    let (c_in, c_out, dims, k) = (4, 5, [5, 4, 9], 3);
    let x = ramp(c_in * dims.iter().product::<usize>(), 5);
    let w = ramp(c_out * c_in * k * k * k, 11);
    let b: Vec<f32> = (0..c_out).map(|i| 0.3 * i as f32 - 0.6).collect();
    let sh = |d: &[usize]| Shape::new(d, DType::F32);
    let out = sh(&[1, c_out, dims[0], dims[1], dims[2]]);
    let mut g = Graph::new("conv3d");
    let xi = g.input("x", sh(&[1, c_in, dims[0], dims[1], dims[2]]));
    let wi = g.input("w", sh(&[c_out, c_in, k, k, k]));
    let y = g.conv3d(xi, wi, [1; 3], [1; 3], [1; 3], 1);
    g.set_outputs(vec![y]);
    let per = dims.iter().product::<usize>();
    let alpha = 0.2f32;
    let z: Vec<f32> = cpu_run(&g, &[("x", x.clone()), ("w", w.clone())])
        .iter()
        .enumerate()
        .map(|(i, v)| v + b[i / per])
        .collect();
    let want: Vec<f32> = z.iter().map(|&v| v.max(v * alpha)).collect();
    for both in [false, true] {
        let mut g = Graph::new("conv3d_leaky");
        let xi = g.input("x", sh(&[1, c_in, dims[0], dims[1], dims[2]]));
        let wi = g.input("w", sh(&[c_out, c_in, k, k, k]));
        let bi = g.input("b", sh(&[c_out]));
        let y = g.add_node(
            Op::FusedConvBiasAct {
                kernel_size: vec![k; 3],
                stride: vec![1; 3],
                padding: vec![1; 3],
                dilation: vec![1; 3],
                groups: 1,
                activation: None,
                has_residual: false,
            },
            vec![xi, wi, bi],
            out.clone(),
        );
        let a = rlx_ir::rf::const_f32(&mut g, alpha, rlx_ir::rf::scalar_f32());
        let s = g.binary(BinaryOp::Mul, y, a, out.clone());
        let m = g.binary(BinaryOp::Max, y, s, out.clone());
        g.set_outputs(if both { vec![m, s] } else { vec![m] });
        let got = WgpuExecutable::compile(g)
            .run(&[("x", &x), ("w", &w), ("b", &b)])
            .remove(0);
        let tol = 1e-4 * (c_in * k * k * k) as f32;
        assert_eq!(got.len(), want.len());
        for (i, (a, e)) in got.iter().zip(&want).enumerate() {
            assert!((a - e).abs() <= tol, "both={both} [{i}]: wgpu {a} cpu {e}");
        }
    }
}

#[test]
fn pointwise_conv_and_channel_argmax_fused() {
    // c_out 7: a remainder tile; channel 5 a copy of channel 2, so they tie
    // and the first must win, as the argmax kernel picks
    let (c_in, c_out, dims) = (5, 7, [3, 5, 9]);
    let per = dims.iter().product::<usize>();
    let x = ramp(c_in * per, 11);
    let mut w = ramp(c_out * c_in, 5);
    let (two, five) = (w[2 * c_in..3 * c_in].to_vec(), 5 * c_in);
    w[five..five + c_in].copy_from_slice(&two);
    let mut b: Vec<f32> = (0..c_out).map(|i| 0.1 * i as f32 - 0.3).collect();
    b[5] = b[2];
    let sh = |d: &[usize]| Shape::new(d, DType::F32);
    let logits = sh(&[1, c_out, dims[0], dims[1], dims[2]]);
    for bias in [true, false] {
        // `with_logits`: the conv also an output, so not fused
        let build = |with_logits: bool| {
            let mut g = Graph::new("conv_argmax");
            let xi = g.input("x", sh(&[1, c_in, dims[0], dims[1], dims[2]]));
            let wi = g.input("w", sh(&[c_out, c_in, 1, 1, 1]));
            let y = if bias {
                let bi = g.input("b", sh(&[c_out]));
                g.add_node(
                    Op::FusedConvBiasAct {
                        kernel_size: vec![1; 3],
                        stride: vec![1; 3],
                        padding: vec![0; 3],
                        dilation: vec![1; 3],
                        groups: 1,
                        activation: None,
                        has_residual: false,
                    },
                    vec![xi, wi, bi],
                    logits.clone(),
                )
            } else {
                g.conv3d(xi, wi, [1; 3], [0; 3], [1; 3], 1)
            };
            let l = g.argmax(y, 1, false, sh(&[1, dims[0], dims[1], dims[2]]));
            g.set_outputs(if with_logits { vec![l, y] } else { vec![l] });
            g
        };
        let mut ins: Vec<(&str, &[f32])> = vec![("x", &x), ("w", &w)];
        if bias {
            ins.push(("b", &b));
        }
        let fused = WgpuExecutable::compile(build(false)).run(&ins).remove(0);
        let mut apart = WgpuExecutable::compile(build(true)).run(&ins);
        let (labels, z) = (apart.remove(0), apart.remove(0));
        assert_eq!(fused, labels, "bias={bias}");
        // and the labels are the first largest channel of the logits
        for v in 0..per {
            let mut best = 0;
            for c in 1..c_out {
                if z[c * per + v] > z[best * per + v] {
                    best = c;
                }
            }
            assert_eq!(fused[v], best as f32, "bias={bias} voxel {v}");
        }
        assert!(
            fused.iter().all(|&l| l != 5.0),
            "a tie went to the later channel"
        );
    }
}

/// `CONV3D_TOPK`: a pointwise conv's `k` largest channels per voxel, their
/// `k + 1`-th as a bound, and their indices — the same values as the plain
/// conv's logits, ranked with the earlier channel first among equals.
#[test]
fn pointwise_conv_top_k() {
    // c_out 11: a remainder tile; channel 6 a copy of channel 3, so they tie
    let (c_in, c_out, dims, k) = (5, 11, [3, 5, 9], 4);
    let per = dims.iter().product::<usize>();
    let x = ramp(c_in * per, 13);
    let mut w = ramp(c_out * c_in, 7);
    let three = w[3 * c_in..4 * c_in].to_vec();
    w[6 * c_in..7 * c_in].copy_from_slice(&three);
    let mut b: Vec<f32> = (0..c_out).map(|i| 0.07 * i as f32 - 0.3).collect();
    b[6] = b[3];
    let sh = |d: &[usize]| Shape::new(d, DType::F32);
    let vol = |c: usize| sh(&[1, c, dims[0], dims[1], dims[2]]);
    let conv = |top: bool| {
        let mut g = Graph::new("conv_topk");
        let xi = g.input("x", vol(c_in));
        let wi = g.input("w", sh(&[c_out, c_in, 1, 1, 1]));
        let bi = g.input("b", sh(&[c_out]));
        let y = if top {
            let attrs: Vec<u8> = [1u32, 1, 1, 0, 0, 0, 1, 1, 1, 1, (k as f32).to_bits()]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            g.add_node(
                Op::Custom {
                    name: rlx_wgpu::unfuse::CONV3D_TOPK.to_string(),
                    num_inputs: 3,
                    attrs,
                },
                vec![xi, wi, bi],
                vol(2 * k + 1),
            )
        } else {
            g.add_node(
                Op::FusedConvBiasAct {
                    kernel_size: vec![1; 3],
                    stride: vec![1; 3],
                    padding: vec![0; 3],
                    dilation: vec![1; 3],
                    groups: 1,
                    activation: None,
                    has_residual: false,
                },
                vec![xi, wi, bi],
                vol(c_out),
            )
        };
        g.set_outputs(vec![y]);
        g
    };
    let ins: Vec<(&str, &[f32])> = vec![("x", &x), ("w", &w), ("b", &b)];
    let top = WgpuExecutable::compile(conv(true)).run(&ins).remove(0);
    let z = WgpuExecutable::compile(conv(false)).run(&ins).remove(0);
    for v in 0..per {
        let mut order: Vec<usize> = (0..c_out).collect();
        // stable: equals keep channel order
        order.sort_by(|&a, &b| z[b * per + v].total_cmp(&z[a * per + v]));
        for r in 0..=k {
            assert_eq!(
                top[r * per + v],
                z[order[r] * per + v],
                "value {r} at voxel {v}"
            );
        }
        for r in 0..k {
            assert_eq!(
                top[(k + 1 + r) * per + v],
                order[r] as f32,
                "channel {r} at voxel {v}"
            );
        }
    }
}
