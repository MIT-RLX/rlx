// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-MLX parity for newly claimed full-coverage ops
//! (3D + AxialRope2d + Fma + FakeQuantize Fixed/PerBatch + deepened
//! Quant/Complex/LSQ/QMatMul/FftButterfly/Mamba2 natives).

#![cfg(rlx_mlx_host)]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_mlx::{MlxExecutable, MlxMode};

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= tol || (x.is_nan() && y.is_nan()))
}

fn cpu_run(g: &Graph, inputs: &[(&str, Vec<f32>)]) -> Vec<f32> {
    let plan = rlx_opt::memory::plan_memory(g);
    let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
    let sched = rlx_cpu::thunk::compile_thunks(g, &arena);
    for node in g.nodes() {
        if let Op::Input { name } = &node.op {
            let data = &inputs
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("missing input {name}"))
                .1;
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
    let out_id = g.outputs[0];
    let n = g.shape(out_id).num_elements().unwrap();
    let off = arena.byte_offset(out_id);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..n).map(|i| *p.add(i)).collect()
    }
}

fn mlx_run(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Lazy);
    exe.run(inputs).into_iter().next().unwrap()
}

fn cpu_run_typed(g: &Graph, inputs: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let plan = rlx_opt::memory::plan_memory_aligned(g, 64);
    let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
    let sched = rlx_cpu::thunk::compile_thunks(g, &arena);
    for node in g.nodes() {
        if let Op::Input { name } = &node.op {
            let data = &inputs
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("missing input {name}"))
                .1;
            let off = arena.byte_offset(node.id);
            let nbytes = g.node(node.id).shape.size_bytes().unwrap_or(0);
            let buf = arena.raw_buf_mut();
            let n = nbytes.min(data.len()).min(buf.len().saturating_sub(off));
            buf[off..off + n].copy_from_slice(&data[..n]);
        }
    }
    rlx_cpu::thunk::execute_thunks(&sched, arena.raw_buf_mut());
    let out_id = g.outputs[0];
    let nbytes = g.node(out_id).shape.size_bytes().unwrap_or(0);
    let off = arena.byte_offset(out_id);
    arena.raw_buf()[off..off + nbytes].to_vec()
}

fn mlx_run_typed(g: Graph, inputs: &[(&str, &[u8], DType)]) -> Vec<u8> {
    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Lazy);
    exe.run_typed(inputs).into_iter().next().unwrap().0
}

#[test]
fn conv3d_matches_cpu() {
    let mut g = Graph::new("conv3d");
    let x = g.input("x", Shape::new(&[1, 1, 3, 3, 3], DType::F32));
    let w = g.input("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.conv3d(x, w, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = (1..=27).map(|v| v as f32).collect();
    let wv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let want = cpu_run(&g, &[("x", xv.clone()), ("w", wv.clone())]);
    let got = mlx_run(g, &[("x", &xv), ("w", &wv)]);
    assert!(
        close(&got, &want, 1e-4),
        "Conv3d mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv_transpose2d_matches_cpu() {
    let mut g = Graph::new("ct2d");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2], DType::F32));
    let w = g.input("w", Shape::new(&[1, 1, 2, 2], DType::F32));
    let y = g.conv_transpose2d(x, w, [2, 2], [2, 2], [0, 0], [1, 1], [0, 0], 1);
    g.set_outputs(vec![y]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "ConvTranspose2d should lower natively (not host-eval)"
    );
    let xv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let wv: Vec<f32> = vec![1.0, 0.5, 0.25, 2.0];
    let want = cpu_run(&g, &[("x", xv.clone()), ("w", wv.clone())]);
    let got = mlx_run(g, &[("x", &xv), ("w", &wv)]);
    assert!(
        close(&got, &want, 1e-4),
        "ConvTranspose2d mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv_transpose2d_depthwise_matches_cpu() {
    // Grouped/depthwise transpose-conv: MLX's native grouped CT mixes channels
    // ACROSS groups (kokoro ISTFTNet upsampler g=512 → output ~25× too large →
    // garbage audio). We host-eval groups>1. This guards both that it's routed
    // to host AND that the host result matches CPU (the groups=1 test above did
    // not cover grouped, which is how the regression slipped through).
    let mut g = Graph::new("ct2d_dw");
    let x = g.input("x", Shape::new(&[1, 2, 1, 2], DType::F32));
    let w = g.input("w", Shape::new(&[2, 1, 1, 2], DType::F32));
    let y = g.conv_transpose2d(x, w, [1, 2], [1, 1], [0, 0], [1, 1], [0, 0], 2);
    g.set_outputs(vec![y]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_some(),
        "grouped ConvTranspose2d must host-eval (MLX native grouped CT mixes channels)"
    );
    // ch0 x=[1,2] w=[1,0.5]; ch1 x=[3,4] w=[2,3] — channels stay independent.
    let xv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let wv: Vec<f32> = vec![1.0, 0.5, 2.0, 3.0];
    let want = cpu_run(&g, &[("x", xv.clone()), ("w", wv.clone())]);
    let got = mlx_run(g, &[("x", &xv), ("w", &wv)]);
    assert!(
        close(&got, &want, 1e-4),
        "depthwise ConvTranspose2d mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv_transpose3d_matches_cpu() {
    let mut g = Graph::new("ct3d");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let w = g.input("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.conv_transpose3d(x, w, [2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], 1);
    g.set_outputs(vec![y]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "ConvTranspose3d should lower natively (not host-eval)"
    );
    let xv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let wv: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let want = cpu_run(&g, &[("x", xv.clone()), ("w", wv.clone())]);
    let got = mlx_run(g, &[("x", &xv), ("w", &wv)]);
    assert!(
        close(&got, &want, 1e-4),
        "ConvTranspose3d mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn axial_rope2d_matches_cpu() {
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
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "AxialRope2d should lower natively (not host-eval)"
    );
    let xv: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 * 0.13).sin()).collect();
    let want = cpu_run(&g, &[("x", xv.clone())]);
    let got = mlx_run(g, &[("x", &xv)]);
    assert!(
        close(&got, &want, 1e-4),
        "AxialRope2d mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn fma_matches_cpu() {
    let mut g = Graph::new("fma");
    let a = g.input("a", Shape::new(&[4], DType::F32));
    let b = g.input("b", Shape::new(&[4], DType::F32));
    let c = g.input("c", Shape::new(&[4], DType::F32));
    let y = g.add_node(Op::Fma, vec![a, b, c], Shape::new(&[4], DType::F32));
    g.set_outputs(vec![y]);
    let av = vec![1.0f32, 2.0, 3.0, 4.0];
    let bv = vec![0.5f32, 0.5, 0.5, 0.5];
    let cv = vec![1.0f32, 0.0, -1.0, 2.0];
    let want = cpu_run(
        &g,
        &[("a", av.clone()), ("b", bv.clone()), ("c", cv.clone())],
    );
    let got = mlx_run(g, &[("a", &av), ("b", &bv), ("c", &cv)]);
    assert!(
        close(&got, &want, 1e-5),
        "Fma mismatch:\n got={got:?}\nwant={want:?}"
    );
    // Spot-check: a*b+c = [1.5, 1.0, 0.5, 4.0]
    assert!(close(&got, &[1.5, 1.0, 0.5, 4.0], 1e-5));
}

#[test]
fn fake_quantize_perbatch_matches_cpu() {
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
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "FakeQuantize PerBatch should lower natively (not host-eval)"
    );
    let xv: Vec<f32> = (0..6).map(|i| 0.07 * (i as f32) - 1.3).collect();
    let want = cpu_run(&g, &[("x", xv.clone())]);
    let got = mlx_run(g, &[("x", &xv)]);
    assert!(
        close(&got, &want, 1e-4),
        "FakeQuantize PerBatch mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn fake_quantize_fixed_matches_cpu() {
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
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "FakeQuantize Fixed should lower natively (not host-eval)"
    );
    let xv: Vec<f32> = (0..6).map(|i| 0.07 * (i as f32) - 1.3).collect();
    let scale = vec![0.02f32, 0.05, 0.08];
    let want = {
        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
        let sched = rlx_cpu::thunk::compile_thunks(&g, &arena);
        for node in g.nodes() {
            match &node.op {
                Op::Input { name } if name == "x" => {
                    let off = arena.byte_offset(node.id);
                    unsafe {
                        let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
                        for (i, &v) in xv.iter().enumerate() {
                            *p.add(i) = v;
                        }
                    }
                }
                Op::Param { name } if name == "scale" => {
                    let off = arena.byte_offset(node.id);
                    unsafe {
                        let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
                        for (i, &v) in scale.iter().enumerate() {
                            *p.add(i) = v;
                        }
                    }
                }
                _ => {}
            }
        }
        rlx_cpu::thunk::execute_thunks(&sched, arena.raw_buf_mut());
        let out_id = g.outputs[0];
        let n = g.shape(out_id).num_elements().unwrap();
        let off = arena.byte_offset(out_id);
        unsafe {
            let p = arena.raw_buf().as_ptr().add(off) as *const f32;
            (0..n).map(|i| *p.add(i)).collect::<Vec<_>>()
        }
    };
    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Lazy);
    exe.set_param("scale", &scale);
    let got = exe.run(&[("x", &xv)]).into_iter().next().unwrap();
    assert!(
        close(&got, &want, 1e-4),
        "FakeQuantize Fixed mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn fake_quantize_perbatch_channel_matches_cpu() {
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
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "FakeQuantize PerBatch channel should lower natively (not host-eval)"
    );
    let xv: Vec<f32> = (0..12).map(|i| 0.11 * (i as f32) - 0.9).collect();
    let want = cpu_run(&g, &[("x", xv.clone())]);
    let got = mlx_run(g, &[("x", &xv)]);
    assert!(
        close(&got, &want, 1e-4),
        "FakeQuantize PerBatch channel mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn batch_norm_inference_matches_cpu() {
    // Feature dim is the last axis (IR / CPU thunk contract).
    let mut g = Graph::new("bni");
    let x = g.input("x", Shape::new(&[2, 3, 4], DType::F32));
    let gamma = g.input("gamma", Shape::new(&[4], DType::F32));
    let beta = g.input("beta", Shape::new(&[4], DType::F32));
    let mean = g.input("mean", Shape::new(&[4], DType::F32));
    let var = g.input("var", Shape::new(&[4], DType::F32));
    let y = g.batch_norm_inference(x, gamma, beta, mean, var, 1e-5);
    g.set_outputs(vec![y]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "BatchNormInference should lower natively (not host-eval)"
    );
    let xv: Vec<f32> = (0..24).map(|i| i as f32 * 0.25 - 1.5).collect();
    let gv = vec![1.0f32, 0.5, 1.5, 0.75];
    let bv = vec![0.0f32, 0.1, -0.2, 0.05];
    let mv = vec![0.5f32, -0.25, 1.0, 0.0];
    let vv = vec![1.0f32, 4.0, 0.25, 2.25];
    let want = cpu_run(
        &g,
        &[
            ("x", xv.clone()),
            ("gamma", gv.clone()),
            ("beta", bv.clone()),
            ("mean", mv.clone()),
            ("var", vv.clone()),
        ],
    );
    let got = mlx_run(
        g,
        &[
            ("x", &xv),
            ("gamma", &gv),
            ("beta", &bv),
            ("mean", &mv),
            ("var", &vv),
        ],
    );
    assert!(
        close(&got, &want, 1e-4),
        "BatchNormInference mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn batch_norm_inference_backward_input_matches_cpu() {
    let mut g = Graph::new("bni_dx");
    let x = g.input("x", Shape::new(&[2, 3, 4], DType::F32));
    let gamma = g.input("gamma", Shape::new(&[4], DType::F32));
    let mean = g.input("mean", Shape::new(&[4], DType::F32));
    let var = g.input("var", Shape::new(&[4], DType::F32));
    let dy = g.input("dy", Shape::new(&[2, 3, 4], DType::F32));
    let dx = g.batch_norm_inference_backward_input(x, gamma, mean, var, dy, 1e-5);
    g.set_outputs(vec![dx]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "BatchNormInferenceBackwardInput should lower natively (not host-eval)"
    );
    let xv: Vec<f32> = (0..24).map(|i| i as f32 * 0.25 - 1.5).collect();
    let gv = vec![1.0f32, 0.5, 1.5, 0.75];
    let mv = vec![0.5f32, -0.25, 1.0, 0.0];
    let vv = vec![1.0f32, 4.0, 0.25, 2.25];
    let dyv: Vec<f32> = (0..24).map(|i| (i as f32 * 0.17).sin()).collect();
    let want = cpu_run(
        &g,
        &[
            ("x", xv.clone()),
            ("gamma", gv.clone()),
            ("mean", mv.clone()),
            ("var", vv.clone()),
            ("dy", dyv.clone()),
        ],
    );
    let got = mlx_run(
        g,
        &[
            ("x", &xv),
            ("gamma", &gv),
            ("mean", &mv),
            ("var", &vv),
            ("dy", &dyv),
        ],
    );
    assert!(
        close(&got, &want, 1e-4),
        "BatchNormInferenceBackwardInput mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn batch_norm_inference_backward_gamma_matches_cpu() {
    let mut g = Graph::new("bni_dg");
    let x = g.input("x", Shape::new(&[2, 3, 4], DType::F32));
    let mean = g.input("mean", Shape::new(&[4], DType::F32));
    let var = g.input("var", Shape::new(&[4], DType::F32));
    let dy = g.input("dy", Shape::new(&[2, 3, 4], DType::F32));
    let dg =
        g.batch_norm_inference_backward_gamma(x, mean, var, dy, Shape::new(&[4], DType::F32), 1e-5);
    g.set_outputs(vec![dg]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "BatchNormInferenceBackwardGamma should lower natively (not host-eval)"
    );
    let xv: Vec<f32> = (0..24).map(|i| i as f32 * 0.25 - 1.5).collect();
    let mv = vec![0.5f32, -0.25, 1.0, 0.0];
    let vv = vec![1.0f32, 4.0, 0.25, 2.25];
    let dyv: Vec<f32> = (0..24).map(|i| (i as f32 * 0.17).sin()).collect();
    let want = cpu_run(
        &g,
        &[
            ("x", xv.clone()),
            ("mean", mv.clone()),
            ("var", vv.clone()),
            ("dy", dyv.clone()),
        ],
    );
    let got = mlx_run(g, &[("x", &xv), ("mean", &mv), ("var", &vv), ("dy", &dyv)]);
    assert!(
        close(&got, &want, 1e-4),
        "BatchNormInferenceBackwardGamma mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn batch_norm_inference_backward_beta_matches_cpu() {
    let mut g = Graph::new("bni_db");
    let dy = g.input("dy", Shape::new(&[2, 3, 4], DType::F32));
    let db = g.batch_norm_inference_backward_beta(dy, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![db]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "BatchNormInferenceBackwardBeta should lower natively (not host-eval)"
    );
    let dyv: Vec<f32> = (0..24).map(|i| (i as f32 * 0.17).sin()).collect();
    let want = cpu_run(&g, &[("dy", dyv.clone())]);
    let got = mlx_run(g, &[("dy", &dyv)]);
    assert!(
        close(&got, &want, 1e-4),
        "BatchNormInferenceBackwardBeta mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn complex_norm_sq_matches_cpu() {
    let n = 3usize;
    let mut g = Graph::new("cns");
    let z = g.input("z", Shape::new(&[n], DType::C64));
    let y = g.complex_norm_sq(z);
    g.set_outputs(vec![y]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "ComplexNormSq should lower natively (not host-eval)"
    );
    let mut z_bytes = Vec::with_capacity(n * 8);
    for &(re, im) in &[(3.0f32, 4.0f32), (1.0, 0.0), (0.0, 0.0)] {
        z_bytes.extend_from_slice(&re.to_le_bytes());
        z_bytes.extend_from_slice(&im.to_le_bytes());
    }
    let want = cpu_run_typed(&g, &[("z", z_bytes.clone())]);
    let got = mlx_run_typed(g, &[("z", &z_bytes, DType::C64)]);
    let want_f: Vec<f32> = want
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let got_f: Vec<f32> = got
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(
        close(&got_f, &want_f, 1e-5),
        "ComplexNormSq mismatch:\n got={got_f:?}\nwant={want_f:?}"
    );
}

#[test]
fn conjugate_matches_cpu() {
    let n = 2usize;
    let mut g = Graph::new("conj");
    let z = g.input("z", Shape::new(&[n], DType::C64));
    let y = g.conjugate(z);
    g.set_outputs(vec![y]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "Conjugate should lower natively (not host-eval)"
    );
    let mut z_bytes = Vec::new();
    for &(re, im) in &[(1.5f32, -2.5f32), (0.0, 3.0)] {
        z_bytes.extend_from_slice(&re.to_le_bytes());
        z_bytes.extend_from_slice(&im.to_le_bytes());
    }
    let want = cpu_run_typed(&g, &[("z", z_bytes.clone())]);
    let got = mlx_run_typed(g, &[("z", &z_bytes, DType::C64)]);
    assert_eq!(got, want, "Conjugate byte mismatch");
}

#[test]
fn complex_norm_sq_backward_matches_cpu() {
    let n = 2usize;
    let mut g = Graph::new("cns_bwd");
    let z = g.input("z", Shape::new(&[n], DType::C64));
    let gv = g.input("g", Shape::new(&[n], DType::F32));
    let y = g.complex_norm_sq_backward(z, gv);
    g.set_outputs(vec![y]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "ComplexNormSqBackward should lower natively (not host-eval)"
    );
    let mut z_bytes = Vec::new();
    for &(re, im) in &[(2.0f32, 1.0f32), (-1.0, 3.0)] {
        z_bytes.extend_from_slice(&re.to_le_bytes());
        z_bytes.extend_from_slice(&im.to_le_bytes());
    }
    let mut g_bytes = Vec::new();
    for &v in &[0.5f32, -2.0] {
        g_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let want = cpu_run_typed(&g, &[("z", z_bytes.clone()), ("g", g_bytes.clone())]);
    let got = mlx_run_typed(
        g,
        &[("z", &z_bytes, DType::C64), ("g", &g_bytes, DType::F32)],
    );
    assert_eq!(got, want, "ComplexNormSqBackward byte mismatch");
}

#[test]
fn fake_quantize_lsq_matches_cpu() {
    let mut g = Graph::new("lsq");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let s = g.param("scale", Shape::new(&[3], DType::F32));
    let q = g.add_node(
        Op::FakeQuantizeLSQ {
            bits: 8,
            axis: Some(1),
        },
        vec![x, s],
        Shape::new(&[2, 3], DType::F32),
    );
    g.set_outputs(vec![q]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "FakeQuantizeLSQ should lower natively (not host-eval)"
    );
    let xv: Vec<f32> = (0..6).map(|i| 0.07 * (i as f32) - 1.3).collect();
    let scale = vec![0.02f32, 0.05, 0.08];
    let want = {
        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
        let sched = rlx_cpu::thunk::compile_thunks(&g, &arena);
        for node in g.nodes() {
            match &node.op {
                Op::Input { name } if name == "x" => {
                    let off = arena.byte_offset(node.id);
                    unsafe {
                        let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
                        for (i, &v) in xv.iter().enumerate() {
                            *p.add(i) = v;
                        }
                    }
                }
                Op::Param { name } if name == "scale" => {
                    let off = arena.byte_offset(node.id);
                    unsafe {
                        let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
                        for (i, &v) in scale.iter().enumerate() {
                            *p.add(i) = v;
                        }
                    }
                }
                _ => {}
            }
        }
        rlx_cpu::thunk::execute_thunks(&sched, arena.raw_buf_mut());
        let out_id = g.outputs[0];
        let n = g.shape(out_id).num_elements().unwrap();
        let off = arena.byte_offset(out_id);
        unsafe {
            let p = arena.raw_buf().as_ptr().add(off) as *const f32;
            (0..n).map(|i| *p.add(i)).collect::<Vec<_>>()
        }
    };
    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Lazy);
    exe.set_param("scale", &scale);
    let got = exe.run(&[("x", &xv)]).into_iter().next().unwrap();
    assert!(
        close(&got, &want, 1e-4),
        "FakeQuantizeLSQ mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn fake_quantize_lsq_backward_x_matches_cpu() {
    let mut g = Graph::new("lsq_dx");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let s = g.input("scale", Shape::new(&[1], DType::F32));
    let dy = g.input("dy", Shape::new(&[4], DType::F32));
    let dx = g.add_node(
        Op::FakeQuantizeLSQBackwardX {
            bits: 8,
            axis: None,
        },
        vec![x, s, dy],
        Shape::new(&[4], DType::F32),
    );
    g.set_outputs(vec![dx]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "FakeQuantizeLSQBackwardX should lower natively (not host-eval)"
    );
    let xv = vec![0.5f32, 2.0, -3.0, 20.0];
    let sv = vec![0.1f32];
    let dyv = vec![1.0f32, -1.0, 0.5, 2.0];
    let want = cpu_run(
        &g,
        &[
            ("x", xv.clone()),
            ("scale", sv.clone()),
            ("dy", dyv.clone()),
        ],
    );
    let got = mlx_run(g, &[("x", &xv), ("scale", &sv), ("dy", &dyv)]);
    assert!(
        close(&got, &want, 1e-5),
        "FakeQuantizeLSQBackwardX mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn quantize_dequantize_matches_cpu() {
    let mut g = Graph::new("qdq");
    let x = g.input("x", Shape::new(&[8], DType::F32));
    let q = g.quantize(x, 0.05, 3);
    let y = g.dequantize(q, 0.05, 3);
    g.set_outputs(vec![y]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "Quantize/Dequantize should lower natively (not host-eval)"
    );
    let xv: Vec<f32> = (0..8).map(|i| 0.3 * (i as f32) - 1.0).collect();
    let want = cpu_run(&g, &[("x", xv.clone())]);
    let got = mlx_run(g, &[("x", &xv)]);
    assert!(
        close(&got, &want, 1e-5),
        "Quantize/Dequantize mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn q_mat_mul_matches_cpu() {
    let m = 2usize;
    let k = 3usize;
    let n = 2usize;
    let mut g = Graph::new("qmm");
    let x = g.input("x", Shape::new(&[m, k], DType::I8));
    let w = g.input("w", Shape::new(&[k, n], DType::I8));
    let bias = g.input("bias", Shape::new(&[n], DType::I32));
    let y = g.add_node(
        Op::QMatMul {
            x_zp: 1,
            w_zp: -2,
            out_zp: 0,
            mult: 0.01,
        },
        vec![x, w, bias],
        Shape::new(&[m, n], DType::I8),
    );
    g.set_outputs(vec![y]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "QMatMul should lower natively (not host-eval)"
    );
    let xv: Vec<i8> = vec![1, -2, 3, 4, 0, -5];
    let wv: Vec<i8> = vec![1, 2, -1, 0, 3, -2];
    let bv: Vec<i32> = vec![10, -5];
    let x_bytes: Vec<u8> = xv.iter().map(|&v| v as u8).collect();
    let w_bytes: Vec<u8> = wv.iter().map(|&v| v as u8).collect();
    let mut b_bytes = Vec::new();
    for &v in &bv {
        b_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let want = cpu_run_typed(
        &g,
        &[
            ("x", x_bytes.clone()),
            ("w", w_bytes.clone()),
            ("bias", b_bytes.clone()),
        ],
    );
    let got = mlx_run_typed(
        g,
        &[
            ("x", &x_bytes, DType::I8),
            ("w", &w_bytes, DType::I8),
            ("bias", &b_bytes, DType::I32),
        ],
    );
    assert_eq!(got, want, "QMatMul mismatch got={got:?} want={want:?}");
}

#[test]
fn mamba2_matches_cpu() {
    let (b, s, h, p, n) = (1usize, 2usize, 2usize, 2usize, 2usize);
    let mut g = Graph::new("m2");
    let x = g.input("x", Shape::new(&[b, s, h, p], DType::F32));
    let dt = g.input("dt", Shape::new(&[b, s, h], DType::F32));
    let a = g.input("a", Shape::new(&[h], DType::F32));
    let bb = g.input("b", Shape::new(&[b, s, h, n], DType::F32));
    let c = g.input("c", Shape::new(&[b, s, h, n], DType::F32));
    let y = g.mamba2(x, dt, a, bb, c, p, n, Shape::new(&[b, s, h, p], DType::F32));
    g.set_outputs(vec![y]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "Mamba2 should lower natively (not host-eval)"
    );
    let xv: Vec<f32> = (0..b * s * h * p).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let dtv: Vec<f32> = vec![0.2, 0.3, 0.1, 0.4];
    let av: Vec<f32> = vec![-0.5, -1.0];
    let bv: Vec<f32> = (0..b * s * h * n)
        .map(|i| 0.05 * (i as f32 + 1.0))
        .collect();
    let cv: Vec<f32> = (0..b * s * h * n)
        .map(|i| 0.07 * (i as f32 + 1.0))
        .collect();
    let want = cpu_run(
        &g,
        &[
            ("x", xv.clone()),
            ("dt", dtv.clone()),
            ("a", av.clone()),
            ("b", bv.clone()),
            ("c", cv.clone()),
        ],
    );
    let got = mlx_run(
        g,
        &[("x", &xv), ("dt", &dtv), ("a", &av), ("b", &bv), ("c", &cv)],
    );
    assert!(
        close(&got, &want, 1e-4),
        "Mamba2 mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn fft_butterfly_stage_matches_cpu() {
    let batch = 1usize;
    let n_fft = 4usize;
    let stage = 0u32;
    let half = n_fft / 2;
    let mut g = Graph::new("fft_bf");
    let state = g.input("state", Shape::new(&[batch, n_fft * 2], DType::F32));
    let gate = g.input("gate", Shape::new(&[half], DType::F32));
    let rev = g.input("rev", Shape::new(&[half], DType::F32));
    let tw_re = g.input("tw_re", Shape::new(&[half], DType::F32));
    let tw_im = g.input("tw_im", Shape::new(&[half], DType::F32));
    let y = g.fft_butterfly_stage(state, gate, rev, tw_re, tw_im, stage, n_fft as u32);
    g.set_outputs(vec![y]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "FftButterflyStage should lower natively (not host-eval)"
    );
    // Identity-ish: gate all-on, no reverse, twiddle = 1+0i
    let state_v: Vec<f32> = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let gate_v = vec![1.0f32; half];
    let rev_v = vec![0.0f32; half];
    let tw_re_v = vec![1.0f32; half];
    let tw_im_v = vec![0.0f32; half];
    let want = cpu_run(
        &g,
        &[
            ("state", state_v.clone()),
            ("gate", gate_v.clone()),
            ("rev", rev_v.clone()),
            ("tw_re", tw_re_v.clone()),
            ("tw_im", tw_im_v.clone()),
        ],
    );
    let got = mlx_run(
        g,
        &[
            ("state", &state_v),
            ("gate", &gate_v),
            ("rev", &rev_v),
            ("tw_re", &tw_re_v),
            ("tw_im", &tw_im_v),
        ],
    );
    assert!(
        close(&got, &want, 1e-5),
        "FftButterflyStage mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn scaled_quant_scale_per_tensor_matches_cpu() {
    use rlx_ir::{ScaleLayout, ScaledFormat};
    let mut g = Graph::new("sqs");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let s = g.add_node(
        Op::ScaledQuantScale {
            format: ScaledFormat::F8E4M3,
            scale_layout: ScaleLayout::PerTensor,
        },
        vec![x],
        Shape::new(&[1], DType::F32),
    );
    g.set_outputs(vec![s]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "ScaledQuantScale PerTensor should lower natively"
    );
    let xv: Vec<f32> = vec![1.0, -2.0, 0.5, 0.0, 3.0, -1.0, 0.25, 0.75];
    let want = cpu_run(&g, &[("x", xv.clone())]);
    let got = mlx_run(g, &[("x", &xv)]);
    assert!(
        close(&got, &want, 1e-5),
        "ScaledQuantScale mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn scaled_dequantize_per_tensor_matches_cpu() {
    use rlx_ir::{ScaleLayout, ScaledFormat};
    let mut g = Graph::new("sdq");
    let codes = g.input("codes", Shape::new(&[2, 3], DType::U8));
    let scale = g.input("scale", Shape::new(&[1], DType::F32));
    let y = g.scaled_dequantize(codes, scale, ScaledFormat::F8E4M3, ScaleLayout::PerTensor);
    g.set_outputs(vec![y]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "ScaledDequantize PerTensor should lower natively"
    );
    let code_bytes: Vec<u8> = vec![0, 1, 2, 10, 20, 30];
    let scale_v = [0.5f32];
    let want = cpu_run_typed(
        &g,
        &[
            ("codes", code_bytes.clone()),
            (
                "scale",
                scale_v.iter().flat_map(|f| f.to_le_bytes()).collect(),
            ),
        ],
    );
    let got = mlx_run_typed(
        g,
        &[
            ("codes", &code_bytes, DType::U8),
            (
                "scale",
                &scale_v
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect::<Vec<_>>(),
                DType::F32,
            ),
        ],
    );
    let want_f: Vec<f32> = want
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let got_f: Vec<f32> = got
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(
        close(&got_f, &want_f, 1e-5),
        "ScaledDequantize mismatch:\n got={got_f:?}\nwant={want_f:?}"
    );
}

#[test]
fn scaled_matmul_per_tensor_matches_cpu() {
    use rlx_ir::{ScaleLayout, ScaledFormat};
    let m = 2usize;
    let k = 4usize;
    let n = 3usize;
    let fmt = ScaledFormat::F8E4M3;
    let layout = ScaleLayout::PerTensor;
    let mut g = Graph::new("smm");
    let lhs = g.input("lhs", Shape::new(&[m, k], DType::U8));
    let rhs = g.input("rhs", Shape::new(&[n, k], DType::U8));
    let ls = g.input("ls", Shape::new(&[1], DType::F32));
    let rs = g.input("rs", Shape::new(&[1], DType::F32));
    let y = g.add_node(
        Op::ScaledMatMul {
            lhs_format: fmt,
            rhs_format: fmt,
            scale_layout: layout,
            has_bias: false,
        },
        vec![lhs, rhs, ls, rs],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    assert!(
        rlx_mlx::lower::first_host_eval_op(&g).is_none(),
        "ScaledMatMul PerTensor should lower natively"
    );
    let lhs_c: Vec<u8> = (0..m * k).map(|i| (i * 3 + 1) as u8).collect();
    let rhs_c: Vec<u8> = (0..n * k).map(|i| (i * 5 + 7) as u8).collect();
    let ls_v = vec![0.25f32];
    let rs_v = vec![0.5f32];
    let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
    let want = cpu_run_typed(
        &g,
        &[
            ("lhs", lhs_c.clone()),
            ("rhs", rhs_c.clone()),
            ("ls", f32_bytes(&ls_v)),
            ("rs", f32_bytes(&rs_v)),
        ],
    );
    let got = mlx_run_typed(
        g,
        &[
            ("lhs", &lhs_c, DType::U8),
            ("rhs", &rhs_c, DType::U8),
            ("ls", &f32_bytes(&ls_v), DType::F32),
            ("rs", &f32_bytes(&rs_v), DType::F32),
        ],
    );
    let want_f: Vec<f32> = want
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let got_f: Vec<f32> = got
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(
        close(&got_f, &want_f, 1e-4),
        "ScaledMatMul mismatch:\n got={got_f:?}\nwant={want_f:?}"
    );
}

// Asymmetric SDPA (MLA): Q/K scores use `head_dim`, V is read and the
// output written with a *different* `v_head_dim`. DeepSeek/Kimi MLA runs
// qk_head_dim=192, v_head_dim=128. MLX's fast::scaled_dot_product_attention
// supports V having a different last dim natively — the output inherits V's
// head_dim. Here head_dim=8, v_head_dim=4, 2 heads, rank-3, Causal.
#[test]
fn attention_asymmetric_v_head_dim_matches_cpu() {
    use rlx_ir::op::MaskKind;
    let b = 1usize;
    let s = 3usize;
    let nh = 2usize;
    let hd = 8usize; // Q/K head_dim (score width)
    let vhd = 4usize; // V/output head_dim (asymmetric)

    let mut g = Graph::new("attn_mla");
    // rank-3 [B, S, H*D]: Q/K carry head_dim, V carries v_head_dim.
    let q = g.input("q", Shape::new(&[b, s, nh * hd], DType::F32));
    let k = g.input("k", Shape::new(&[b, s, nh * hd], DType::F32));
    let v = g.input("v", Shape::new(&[b, s, nh * vhd], DType::F32));
    let o = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: Some(vhd),
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        // Output is v_head_dim-wide: [B, S, H*v_head_dim].
        Shape::new(&[b, s, nh * vhd], DType::F32),
    );
    g.set_outputs(vec![o]);

    // Deterministic, mildly varied inputs (avoid the degenerate all-equal
    // case where every V head_dim would be indistinguishable).
    let qv: Vec<f32> = (0..b * s * nh * hd)
        .map(|i| ((i as f32) * 0.13).sin() * 0.5)
        .collect();
    let kv: Vec<f32> = (0..b * s * nh * hd)
        .map(|i| ((i as f32) * 0.17 + 1.0).cos() * 0.5)
        .collect();
    let vv: Vec<f32> = (0..b * s * nh * vhd)
        .map(|i| ((i as f32) * 0.29 + 0.3).sin())
        .collect();

    let want = cpu_run(
        &g,
        &[("q", qv.clone()), ("k", kv.clone()), ("v", vv.clone())],
    );
    let got = mlx_run(g, &[("q", &qv), ("k", &kv), ("v", &vv)]);

    // Output must be v_head_dim-wide, not head_dim-wide.
    assert_eq!(
        got.len(),
        b * s * nh * vhd,
        "asymmetric attention output width should be num_heads*v_head_dim"
    );
    let max_delta = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta < 1e-3,
        "asymmetric v_head_dim attention MLX-vs-CPU max|Δ|={max_delta} too large:\n got={got:?}\nwant={want:?}"
    );
}
