// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-wgpu parity for newly claimed host / HostOp ops (training + vision).

use rlx::prelude::*;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= tol || (x.is_nan() && y.is_nan()))
}

fn cpu_run(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    Session::new(Device::Cpu)
        .compile(g)
        .run(inputs)
        .into_iter()
        .next()
        .unwrap()
}

fn wgpu_run(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let names: Vec<&str> = inputs.iter().map(|(n, _)| *n).collect();
    let mut exe = WgpuExecutable::compile(g);
    let feed: Vec<(&str, &[f32])> = names
        .iter()
        .zip(inputs.iter().map(|(_, v)| *v))
        .map(|(n, v)| (*n, v))
        .collect();
    exe.run(&feed).into_iter().next().unwrap()
}

#[test]
fn group_norm_backward_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let (n, c, h, w, groups) = (1usize, 8usize, 4usize, 4usize, 2usize);
    let mut g = Graph::new("gn_bwd");
    let x = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let gamma = g.input("gamma", Shape::new(&[c], DType::F32));
    let beta = g.input("beta", Shape::new(&[c], DType::F32));
    let dy = g.input("dy", Shape::new(&[n, c, h, w], DType::F32));
    let dx = g.group_norm_backward_input(x, gamma, beta, dy, groups, 1e-5);
    let dgamma = g.group_norm_backward_gamma(x, dy, Shape::new(&[c], DType::F32), groups, 1e-5);
    let dbeta = g.group_norm_backward_beta(x, dy, Shape::new(&[c], DType::F32), groups, 1e-5);
    g.set_outputs(vec![dx, dgamma, dbeta]);
    let xv: Vec<f32> = (0..n * c * h * w)
        .map(|i| (((i + 1) % 19) as f32 - 9.0) * 0.07)
        .collect();
    let gv: Vec<f32> = (0..c)
        .map(|i| (((i + 3) % 19) as f32 - 9.0) * 0.07)
        .collect();
    let bv: Vec<f32> = (0..c)
        .map(|i| (((i + 5) % 19) as f32 - 9.0) * 0.07)
        .collect();
    let dyv: Vec<f32> = (0..n * c * h * w)
        .map(|i| (((i + 7) % 19) as f32 - 9.0) * 0.07)
        .collect();
    let feed = [
        ("x", xv.as_slice()),
        ("gamma", gv.as_slice()),
        ("beta", bv.as_slice()),
        ("dy", dyv.as_slice()),
    ];
    let want: Vec<f32> = {
        let outs = Session::new(Device::Cpu).compile(g.clone()).run(&feed);
        outs.into_iter().flatten().collect()
    };
    let got: Vec<f32> = {
        let mut exe = WgpuExecutable::compile(g);
        exe.run(&feed).into_iter().flatten().collect()
    };
    assert!(
        close(&got, &want, 1e-4),
        "GroupNormBackward mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn axial_rope2d_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let end_x = 2usize;
    let end_y = 2usize;
    let head_dim = 4usize;
    let num_heads = 1usize;
    let seq = end_x * end_y;
    let hidden = num_heads * head_dim;
    let mut g = Graph::new("axial");
    let x = g.input("x", Shape::new(&[1, seq, hidden], DType::F32));
    let y = g.axial_rope2d(x, end_x, end_y, head_dim, num_heads, 10_000.0, 1);
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 * 0.13).sin()).collect();
    let want = cpu_run(g.clone(), &[("x", &xv)]);
    let got = wgpu_run(g, &[("x", &xv)]);
    assert!(
        close(&got, &want, 1e-4),
        "AxialRope2d mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn maxpool2d_backward_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let n = 1usize;
    let c = 1usize;
    let h = 4usize;
    let w = 4usize;
    let h_out = 2usize;
    let w_out = 2usize;
    let mut g = Graph::new("mpb");
    let x = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let dy = g.input("dy", Shape::new(&[n, c, h_out, w_out], DType::F32));
    let dx = g.maxpool2d_backward(x, dy, vec![2, 2], vec![2, 2], vec![0, 0]);
    g.set_outputs(vec![dx]);
    let xv: Vec<f32> = (0..n * c * h * w).map(|i| (i as f32) * 0.1).collect();
    let dyv: Vec<f32> = (0..n * c * h_out * w_out)
        .map(|i| 1.0 + (i as f32) * 0.25)
        .collect();
    let want = cpu_run(g.clone(), &[("x", &xv), ("dy", &dyv)]);
    let got = wgpu_run(g, &[("x", &xv), ("dy", &dyv)]);
    assert!(
        close(&got, &want, 1e-5),
        "MaxPool2dBackward mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn softmax_ce_with_logits_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let n = 3usize;
    let c = 4usize;
    let mut g = Graph::new("sce");
    let logits = g.input("logits", Shape::new(&[n, c], DType::F32));
    let labels = g.input("labels", Shape::new(&[n], DType::F32));
    let loss = g.softmax_cross_entropy_with_logits(logits, labels);
    g.set_outputs(vec![loss]);
    let lv: Vec<f32> = (0..n * c).map(|i| (i as f32 * 0.17).sin()).collect();
    let lab = vec![0.0f32, 2.0, 1.0];
    let want = cpu_run(g.clone(), &[("logits", &lv), ("labels", &lab)]);
    let got = wgpu_run(g, &[("logits", &lv), ("labels", &lab)]);
    assert!(
        close(&got, &want, 1e-4),
        "SoftmaxCEWithLogits mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn softmax_ce_backward_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let n = 2usize;
    let c = 3usize;
    let mut g = Graph::new("sce_bwd");
    let logits = g.input("logits", Shape::new(&[n, c], DType::F32));
    let labels = g.input("labels", Shape::new(&[n], DType::F32));
    let d_loss = g.input("d_loss", Shape::new(&[n], DType::F32));
    let dx = g.softmax_cross_entropy_backward(logits, labels, d_loss);
    g.set_outputs(vec![dx]);
    let lv: Vec<f32> = (0..n * c).map(|i| (i as f32 * 0.21).cos()).collect();
    let lab = vec![1.0f32, 0.0];
    let dl = vec![1.0f32, 0.5];
    let want = cpu_run(
        g.clone(),
        &[("logits", &lv), ("labels", &lab), ("d_loss", &dl)],
    );
    let got = wgpu_run(g, &[("logits", &lv), ("labels", &lab), ("d_loss", &dl)]);
    assert!(
        close(&got, &want, 1e-4),
        "SoftmaxCEBackward mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn relu_backward_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("relu_bwd");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let dy = g.input("dy", Shape::new(&[4], DType::F32));
    let dx = g.relu_backward(x, dy);
    g.set_outputs(vec![dx]);
    let xv = vec![-1.0f32, 0.0, 0.5, 2.0];
    let dyv = vec![0.1f32, 0.2, 0.3, 0.4];
    let want = cpu_run(g.clone(), &[("x", &xv), ("dy", &dyv)]);
    let got = wgpu_run(g, &[("x", &xv), ("dy", &dyv)]);
    assert!(
        close(&got, &want, 1e-6),
        "ReluBackward mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn activation_backward_silu_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::op::Activation;
    let mut g = Graph::new("act_bwd");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let dy = g.input("dy", Shape::new(&[4], DType::F32));
    let dx = g.activation_backward(Activation::Silu, x, dy);
    g.set_outputs(vec![dx]);
    let xv = vec![-1.0f32, 0.0, 0.5, 2.0];
    let dyv = vec![0.1f32, 0.2, 0.3, 0.4];
    let want = cpu_run(g.clone(), &[("x", &xv), ("dy", &dyv)]);
    let got = wgpu_run(g, &[("x", &xv), ("dy", &dyv)]);
    assert!(
        close(&got, &want, 1e-5),
        "ActivationBackward(Silu) mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn dense_solve_f32_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let n = 2usize;
    let mut g = Graph::new("dense_solve");
    let a = g.input("a", Shape::new(&[n, n], DType::F32));
    let b = g.input("b", Shape::new(&[n], DType::F32));
    let x = g.dense_solve(a, b, Shape::new(&[n], DType::F32));
    g.set_outputs(vec![x]);
    // Well-conditioned 2x2: [[2,1],[0,2]] * [1,2] = [4,4]
    let av = vec![2.0f32, 1.0, 0.0, 2.0];
    let bv = vec![4.0f32, 4.0];
    let want = cpu_run(g.clone(), &[("a", &av), ("b", &bv)]);
    let got = wgpu_run(g, &[("a", &av), ("b", &bv)]);
    assert!(
        close(&got, &want, 1e-4),
        "DenseSolve mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn complex_norm_sq_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("cnorm");
    let z = g.input("z", Shape::new(&[2], DType::C64));
    let y = g.complex_norm_sq(z);
    g.set_outputs(vec![y]);
    // Two complex values as f32 lane pairs (re, im).
    let zv = vec![3.0f32, 4.0, 0.0, 1.0]; // |3+4i|^2=25, |i|^2=1
    let want = cpu_run(g.clone(), &[("z", &zv)]);
    let got = wgpu_run(g, &[("z", &zv)]);
    assert!(
        close(&got, &want, 1e-5),
        "ComplexNormSq mismatch:\n got={got:?}\nwant={want:?}"
    );
    assert!(close(&got, &[25.0, 1.0], 1e-5));
}

#[test]
fn complex_norm_sq_backward_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("cnorm_bwd");
    let z = g.input("z", Shape::new(&[2], DType::C64));
    let gv = g.input("g", Shape::new(&[2], DType::F32));
    let dz = g.complex_norm_sq_backward(z, gv);
    g.set_outputs(vec![dz]);
    let zv = vec![3.0f32, 4.0, 0.0, 1.0];
    let gvals = vec![2.0f32, 0.5];
    let want = cpu_run(g.clone(), &[("z", &zv), ("g", &gvals)]);
    let got = wgpu_run(g, &[("z", &zv), ("g", &gvals)]);
    assert!(
        close(&got, &want, 1e-5),
        "ComplexNormSqBackward mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conjugate_c64_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("conj");
    let z = g.input("z", Shape::new(&[2], DType::C64));
    let y = g.conjugate(z);
    g.set_outputs(vec![y]);
    let zv = vec![3.0f32, 4.0, -1.0, 2.0];
    let want = cpu_run(g.clone(), &[("z", &zv)]);
    let got = wgpu_run(g, &[("z", &zv)]);
    assert!(
        close(&got, &want, 1e-5),
        "Conjugate mismatch:\n got={got:?}\nwant={want:?}"
    );
    assert!(close(&got, &[3.0, -4.0, -1.0, -2.0], 1e-5));
}

#[test]
fn softmax_non_last_axis_host_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("sm_mid");
    let x = g.input("x", Shape::new(&[2, 3, 2], DType::F32));
    let y = g.softmax(x, 1, Shape::new(&[2, 3, 2], DType::F32));
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = (0..12).map(|i| (i as f32 * 0.3).sin()).collect();
    let want = cpu_run(g.clone(), &[("x", &xv)]);
    let got = wgpu_run(g, &[("x", &xv)]);
    assert!(
        close(&got, &want, 1e-4),
        "Softmax mid-axis HostOp mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn batch_norm_inference_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let n = 1usize;
    let c = 2usize;
    let h = 2usize;
    let w = 2usize;
    let mut g = Graph::new("bni");
    let x = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[c], DType::F32));
    let beta = g.param("beta", Shape::new(&[c], DType::F32));
    let mean = g.param("mean", Shape::new(&[c], DType::F32));
    let var = g.param("var", Shape::new(&[c], DType::F32));
    let y = g.batch_norm_inference(x, gamma, beta, mean, var, 1e-5);
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = (0..n * c * h * w).map(|i| i as f32 * 0.5).collect();
    let gv = vec![1.0f32, 0.5];
    let bv = vec![0.0f32, 0.1];
    let mv = vec![1.0f32, 2.0];
    let vv = vec![1.0f32, 4.0];
    let mut cpu_exe = Session::new(Device::Cpu).compile(g.clone());
    cpu_exe.set_param("gamma", &gv);
    cpu_exe.set_param("beta", &bv);
    cpu_exe.set_param("mean", &mv);
    cpu_exe.set_param("var", &vv);
    let want = cpu_exe.run(&[("x", &xv)]).into_iter().next().unwrap();
    let mut gpu_exe = WgpuExecutable::compile(g);
    gpu_exe.set_param("gamma", &gv);
    gpu_exe.set_param("beta", &bv);
    gpu_exe.set_param("mean", &mv);
    gpu_exe.set_param("var", &vv);
    let got = gpu_exe.run(&[("x", &xv)]).into_iter().next().unwrap();
    assert!(
        close(&got, &want, 1e-4),
        "BatchNormInference mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv3d_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("conv3d_parity");
    let x = g.input("x", Shape::new(&[1, 1, 3, 3, 3], DType::F32));
    let w = g.input("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.conv3d(x, w, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = (1..=27).map(|v| v as f32).collect();
    let wv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let want = cpu_run(g.clone(), &[("x", &xv), ("w", &wv)]);
    let got = wgpu_run(g, &[("x", &xv), ("w", &wv)]);
    assert!(
        close(&got, &want, 1e-4),
        "Conv3d mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv_transpose3d_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("ct3d_parity");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let w = g.input("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.conv_transpose3d(x, w, [2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], 1);
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let wv: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let want = cpu_run(g.clone(), &[("x", &xv), ("w", &wv)]);
    let got = wgpu_run(g, &[("x", &xv), ("w", &wv)]);
    assert!(
        close(&got, &want, 1e-4),
        "ConvTranspose3d mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn pool3d_avg_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("pool3d_avg");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.add_node(
        Op::Pool {
            kind: ReduceOp::Mean,
            kernel_size: vec![2, 2, 2],
            stride: vec![1, 1, 1],
            padding: vec![0, 0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 1, 1, 1], DType::F32),
    );
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let want = cpu_run(g.clone(), &[("x", &xv)]);
    let got = wgpu_run(g, &[("x", &xv)]);
    assert!(
        close(&got, &want, 1e-4),
        "Pool3d Mean mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn fake_quantize_perbatch_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::op::{ScaleMode, SteKind};
    let mut g = Graph::new("fq");
    let x = g.input("x", Shape::new(&[6], DType::F32));
    let q = g.add_node(
        Op::FakeQuantize {
            bits: 8,
            axis: None,
            ste: SteKind::Identity,
            scale_mode: ScaleMode::PerBatch,
        },
        vec![x],
        Shape::new(&[6], DType::F32),
    );
    g.set_outputs(vec![q]);
    let xv: Vec<f32> = (0..6).map(|i| 0.07 * (i as f32) - 1.3).collect();
    let want = cpu_run(g.clone(), &[("x", &xv)]);
    let got = wgpu_run(g, &[("x", &xv)]);
    assert!(
        close(&got, &want, 1e-4),
        "FakeQuantize mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn fake_quantize_fixed_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::op::{ScaleMode, SteKind};
    let mut g = Graph::new("fq_fx");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let s = g.param("scale", Shape::new(&[3], DType::F32));
    let q = g.add_node(
        Op::FakeQuantize {
            bits: 8,
            axis: Some(1),
            ste: SteKind::Identity,
            scale_mode: ScaleMode::Fixed,
        },
        vec![x, s],
        Shape::new(&[2, 3], DType::F32),
    );
    g.set_outputs(vec![q]);
    let xv: Vec<f32> = (0..6).map(|i| 0.07 * (i as f32) - 1.3).collect();
    let scale = vec![0.02f32, 0.05, 0.08];
    let mut cpu_exe = Session::new(Device::Cpu).compile(g.clone());
    cpu_exe.set_param("scale", &scale);
    let want = cpu_exe.run(&[("x", &xv)]).into_iter().next().unwrap();
    let mut gpu_exe = WgpuExecutable::compile(g);
    gpu_exe.set_param("scale", &scale);
    let got = gpu_exe.run(&[("x", &xv)]).into_iter().next().unwrap();
    assert!(
        close(&got, &want, 1e-4),
        "FakeQuantize Fixed mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn fake_quantize_perbatch_channel_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::op::{ScaleMode, SteKind};
    let mut g = Graph::new("fq_ch");
    let x = g.input("x", Shape::new(&[2, 3, 2], DType::F32));
    let q = g.add_node(
        Op::FakeQuantize {
            bits: 8,
            axis: Some(1),
            ste: SteKind::Identity,
            scale_mode: ScaleMode::PerBatch,
        },
        vec![x],
        Shape::new(&[2, 3, 2], DType::F32),
    );
    g.set_outputs(vec![q]);
    let xv: Vec<f32> = (0..12).map(|i| 0.11 * (i as f32) - 0.9).collect();
    let want = cpu_run(g.clone(), &[("x", &xv)]);
    let got = wgpu_run(g, &[("x", &xv)]);
    assert!(
        close(&got, &want, 1e-4),
        "FakeQuantize PerBatch channel mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn scaled_matmul_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::{ScaleLayout, ScaledFormat};
    let mut g = Graph::new("smm");
    let a = g.input("a", Shape::new(&[2, 4], DType::F32));
    let b = g.input("b", Shape::new(&[3, 4], DType::F32));
    let y = g.scaled_matmul(a, b, ScaledFormat::F8E4M3, ScaleLayout::PerTensor);
    g.set_outputs(vec![y]);
    let av: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let bv: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05 - 0.2).collect();
    let want = cpu_run(g.clone(), &[("a", &av), ("b", &bv)]);
    let got = wgpu_run(g, &[("a", &av), ("b", &bv)]);
    assert!(
        close(&got, &want, 1e-3),
        "ScaledMatMul mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn fft_butterfly_stage_matches_cpu() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let n_fft = 4u32;
    let half = (n_fft / 2) as usize;
    let mut g = Graph::new("fft_bf");
    let state = g.input("state", Shape::new(&[1, n_fft as usize, 2], DType::F32));
    let gate = g.input("gate", Shape::new(&[half], DType::F32));
    let rev = g.input("rev", Shape::new(&[half], DType::F32));
    let tw_re = g.input("tw_re", Shape::new(&[half], DType::F32));
    let tw_im = g.input("tw_im", Shape::new(&[half], DType::F32));
    let y = g.fft_butterfly_stage(state, gate, rev, tw_re, tw_im, 0, n_fft);
    g.set_outputs(vec![y]);
    let state_v = vec![1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let gate_v = vec![1.0f32, 0.0];
    let rev_v = vec![0.0f32, 0.0];
    let tw_re_v = vec![1.0f32, 1.0];
    let tw_im_v = vec![0.0f32, 0.0];
    let feed = [
        ("state", state_v.as_slice()),
        ("gate", gate_v.as_slice()),
        ("rev", rev_v.as_slice()),
        ("tw_re", tw_re_v.as_slice()),
        ("tw_im", tw_im_v.as_slice()),
    ];
    let want = cpu_run(g.clone(), &feed);
    let got = wgpu_run(g, &feed);
    assert!(
        close(&got, &want, 1e-5),
        "FftButterflyStage mismatch:\n got={got:?}\nwant={want:?}"
    );
}
