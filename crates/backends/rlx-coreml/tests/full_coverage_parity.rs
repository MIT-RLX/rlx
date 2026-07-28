// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-CoreML parity for native MIL depth (Fma / Conv3d / fused).

#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_coreml::CoremlExecutable;
use rlx_coreml::hybrid::{ExecutionPlan, plan_execution};
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Op, Shape};

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

fn coreml_run_with_params(
    g: Graph,
    inputs: &[(&str, &[f32])],
    params: &[(&str, &[f32])],
) -> Vec<f32> {
    // Compile after params are known: bake into MIL.
    let mut exe = CoremlExecutable::compile(g);
    for (name, data) in params {
        exe.set_param(name, data);
    }
    exe.finalize().expect("finalize");
    exe.run(inputs)
        .expect("coreml run")
        .into_iter()
        .next()
        .unwrap()
}

fn assert_mil_only(g: &Graph) {
    let mut g = g.clone();
    rlx_coreml::promote_c64_to_interleaved_f32(&mut g);
    let plan = plan_execution(&g).expect("plan");
    assert!(
        matches!(plan, ExecutionPlan::MilOnly),
        "expected MilOnly (native), got {plan:?}"
    );
}

#[test]
fn conv3d_param_weight_is_mil_native() {
    let mut g = Graph::new("conv3d");
    let x = g.input("x", Shape::new(&[1, 1, 3, 3, 3], DType::F32));
    let w = g.param("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.conv3d(x, w, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
    g.set_outputs(vec![y]);
    assert_mil_only(&g);
    let xv: Vec<f32> = (1..=27).map(|v| v as f32).collect();
    let wv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let want = {
        // CPU needs the param as an input feed via arena write of Param — use Input twin.
        let mut gc = Graph::new("conv3d_cpu");
        let xc = gc.input("x", Shape::new(&[1, 1, 3, 3, 3], DType::F32));
        let wc = gc.input("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
        let yc = gc.conv3d(xc, wc, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
        gc.set_outputs(vec![yc]);
        cpu_run(&gc, &[("x", xv.clone()), ("w", wv.clone())])
    };
    let got = coreml_run_with_params(g, &[("x", &xv)], &[("w", &wv)]);
    assert!(
        close(&got, &want, 1e-4),
        "Conv3d MIL mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv3d_input_weight_host_fallback() {
    let mut g = Graph::new("conv3d_host");
    let x = g.input("x", Shape::new(&[1, 1, 3, 3, 3], DType::F32));
    let w = g.input("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.conv3d(x, w, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
    g.set_outputs(vec![y]);
    let plan = plan_execution(&g).expect("plan");
    assert!(
        !matches!(plan, ExecutionPlan::MilOnly),
        "dynamic 3D weights must host-segment"
    );
    let xv: Vec<f32> = (1..=27).map(|v| v as f32).collect();
    let wv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let want = cpu_run(&g, &[("x", xv.clone()), ("w", wv.clone())]);
    let mut exe = CoremlExecutable::compile(g);
    let got = exe
        .run(&[("x", xv.as_slice()), ("w", wv.as_slice())])
        .expect("run")
        .remove(0);
    assert!(close(&got, &want, 1e-4));
}

#[test]
fn conv_transpose3d_param_weight_is_mil_native() {
    let mut g = Graph::new("ct3d");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let w = g.param("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.conv_transpose3d(x, w, [2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], 1);
    g.set_outputs(vec![y]);
    assert_mil_only(&g);
    let xv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let wv: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let want = {
        let mut gc = Graph::new("ct3d_cpu");
        let xc = gc.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
        let wc = gc.input("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
        let yc = gc.conv_transpose3d(xc, wc, [2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], 1);
        gc.set_outputs(vec![yc]);
        cpu_run(&gc, &[("x", xv.clone()), ("w", wv.clone())])
    };
    let got = coreml_run_with_params(g, &[("x", &xv)], &[("w", &wv)]);
    assert!(
        close(&got, &want, 1e-4),
        "ConvTranspose3d MIL mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn fma_is_mil_native() {
    let mut g = Graph::new("fma");
    let a = g.input("a", Shape::new(&[4], DType::F32));
    let b = g.input("b", Shape::new(&[4], DType::F32));
    let c = g.input("c", Shape::new(&[4], DType::F32));
    let y = g.add_node(Op::Fma, vec![a, b, c], Shape::new(&[4], DType::F32));
    g.set_outputs(vec![y]);
    assert_mil_only(&g);
    let av = vec![1.0f32, 2.0, 3.0, 4.0];
    let bv = vec![0.5f32, 0.5, 0.5, 0.5];
    let cv = vec![1.0f32, 0.0, -1.0, 2.0];
    let want = cpu_run(
        &g,
        &[("a", av.clone()), ("b", bv.clone()), ("c", cv.clone())],
    );
    let mut exe = CoremlExecutable::compile(g);
    let got = exe
        .run(&[
            ("a", av.as_slice()),
            ("b", bv.as_slice()),
            ("c", cv.as_slice()),
        ])
        .expect("run")
        .remove(0);
    assert!(close(&got, &want, 1e-5));
    assert!(close(&got, &[1.5, 1.0, 0.5, 4.0], 1e-5));
}

#[test]
fn fused_matmul_bias_relu_is_mil_native() {
    let mut g = Graph::new("fmba");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[3, 2], DType::F32));
    let b = g.param("b", Shape::new(&[2], DType::F32));
    let y = g.add_node(
        Op::FusedMatMulBiasAct {
            activation: Some(Activation::Relu),
        },
        vec![x, w, b],
        Shape::new(&[2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    assert_mil_only(&g);
    let xv = vec![1.0f32, 0.0, -1.0, 0.5, 0.5, 0.5];
    let wv = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let bv = vec![-0.5f32, 0.0];
    let want = {
        let mut gc = Graph::new("fmba_cpu");
        let xc = gc.input("x", Shape::new(&[2, 3], DType::F32));
        let wc = gc.input("w", Shape::new(&[3, 2], DType::F32));
        let bc = gc.input("b", Shape::new(&[2], DType::F32));
        let yc = gc.add_node(
            Op::FusedMatMulBiasAct {
                activation: Some(Activation::Relu),
            },
            vec![xc, wc, bc],
            Shape::new(&[2, 2], DType::F32),
        );
        gc.set_outputs(vec![yc]);
        cpu_run(
            &gc,
            &[("x", xv.clone()), ("w", wv.clone()), ("b", bv.clone())],
        )
    };
    let got = coreml_run_with_params(g, &[("x", &xv)], &[("w", &wv), ("b", &bv)]);
    assert!(
        close(&got, &want, 1e-4),
        "FusedMatMulBiasAct mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn dense_solve_still_host() {
    let mut g = Graph::new("dense_solve");
    let a = g.input("a", Shape::new(&[2, 2], DType::F32));
    let b = g.input("b", Shape::new(&[2], DType::F32));
    let y = g.add_node(Op::DenseSolve, vec![a, b], Shape::new(&[2], DType::F32));
    g.set_outputs(vec![y]);
    let av = vec![2.0f32, 0.0, 0.0, 4.0];
    let bv = vec![2.0f32, 8.0];
    let want = cpu_run(&g, &[("a", av.clone()), ("b", bv.clone())]);
    let mut exe = CoremlExecutable::compile(g);
    let got = exe
        .run(&[("a", av.as_slice()), ("b", bv.as_slice())])
        .expect("run")
        .remove(0);
    assert!(close(&got, &want, 1e-4));
}

#[test]
fn fake_quantize_fixed_param_is_mil_native() {
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
    assert_mil_only(&g);
    let xv: Vec<f32> = (0..6).map(|i| 0.07 * (i as f32) - 1.3).collect();
    let scale = vec![0.02f32, 0.05, 0.08];
    let want = {
        let mut gc = Graph::new("fq_fx_cpu");
        let xc = gc.input("x", Shape::new(&[2, 3], DType::F32));
        let sc = gc.input("scale", Shape::new(&[3], DType::F32));
        let qc = gc.add_node(
            Op::FakeQuantize {
                bits: 8,
                axis: Some(1),
                ste: SteKind::Identity,
                scale_mode: ScaleMode::Fixed,
            },
            vec![xc, sc],
            Shape::new(&[2, 3], DType::F32),
        );
        gc.set_outputs(vec![qc]);
        cpu_run(&gc, &[("x", xv.clone()), ("scale", scale.clone())])
    };
    let got = coreml_run_with_params(g, &[("x", &xv)], &[("scale", &scale)]);
    assert!(
        close(&got, &want, 1e-4),
        "FakeQuantize Fixed MIL mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn complex_norm_sq_is_mil_native() {
    let n = 3usize;
    let mut g = Graph::new("cns");
    let z = g.input("z", Shape::new(&[n], DType::C64));
    let y = g.complex_norm_sq(z);
    g.set_outputs(vec![y]);
    assert_mil_only(&g);
    assert!(!rlx_coreml::host_exec::is_host_op(&Op::ComplexNormSq));
    // Interleaved re/im as f32 feed (compile promotes C64 → [2n] F32).
    let zv: Vec<f32> = vec![3.0, 4.0, 1.0, 0.0, 0.0, 0.0];
    let want = {
        let mut gc = Graph::new("cns_cpu");
        let zc = gc.input("z", Shape::new(&[n], DType::C64));
        let yc = gc.complex_norm_sq(zc);
        gc.set_outputs(vec![yc]);
        let plan = rlx_opt::memory::plan_memory_aligned(&gc, 64);
        let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
        let sched = rlx_cpu::thunk::compile_thunks(&gc, &arena);
        let off = arena.byte_offset(zc);
        let buf = arena.raw_buf_mut();
        for (i, &v) in zv.iter().enumerate() {
            let b = v.to_le_bytes();
            buf[off + i * 4..off + i * 4 + 4].copy_from_slice(&b);
        }
        rlx_cpu::thunk::execute_thunks(&sched, arena.raw_buf_mut());
        let out_id = gc.outputs[0];
        let nout = gc.shape(out_id).num_elements().unwrap();
        let ooff = arena.byte_offset(out_id);
        unsafe {
            let p = arena.raw_buf().as_ptr().add(ooff) as *const f32;
            (0..nout).map(|i| *p.add(i)).collect::<Vec<_>>()
        }
    };
    let got = coreml_run_with_params(g, &[("z", &zv)], &[]);
    assert!(
        close(&got, &want, 1e-5),
        "ComplexNormSq MIL mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conjugate_is_mil_native() {
    let n = 2usize;
    let mut g = Graph::new("conj");
    let z = g.input("z", Shape::new(&[n], DType::C64));
    let y = g.conjugate(z);
    g.set_outputs(vec![y]);
    assert_mil_only(&g);
    assert!(!rlx_coreml::host_exec::is_host_op(&Op::Conjugate));
    let zv: Vec<f32> = vec![1.5, -2.5, 0.0, 3.0];
    let want = {
        let mut gc = Graph::new("conj_cpu");
        let zc = gc.input("z", Shape::new(&[n], DType::C64));
        let yc = gc.conjugate(zc);
        gc.set_outputs(vec![yc]);
        let plan = rlx_opt::memory::plan_memory_aligned(&gc, 64);
        let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
        let sched = rlx_cpu::thunk::compile_thunks(&gc, &arena);
        let off = arena.byte_offset(zc);
        let buf = arena.raw_buf_mut();
        for (i, &v) in zv.iter().enumerate() {
            buf[off + i * 4..off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        rlx_cpu::thunk::execute_thunks(&sched, arena.raw_buf_mut());
        let out_id = gc.outputs[0];
        let nbytes = gc.node(out_id).shape.size_bytes().unwrap();
        let ooff = arena.byte_offset(out_id);
        let bytes = &arena.raw_buf()[ooff..ooff + nbytes];
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>()
    };
    let got = coreml_run_with_params(g, &[("z", &zv)], &[]);
    assert!(
        close(&got, &want, 1e-5),
        "Conjugate MIL mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn complex_norm_sq_backward_is_mil_native() {
    let n = 2usize;
    let mut g = Graph::new("cns_bwd");
    let z = g.input("z", Shape::new(&[n], DType::C64));
    let gv = g.input("g", Shape::new(&[n], DType::F32));
    let dz = g.complex_norm_sq_backward(z, gv);
    g.set_outputs(vec![dz]);
    assert_mil_only(&g);
    assert!(!rlx_coreml::host_exec::is_host_op(
        &Op::ComplexNormSqBackward
    ));
    let zv: Vec<f32> = vec![1.0, 2.0, -0.5, 0.25];
    let gvals = vec![3.0f32, 0.5];
    let want = {
        let mut gc = Graph::new("cns_bwd_cpu");
        let zc = gc.input("z", Shape::new(&[n], DType::C64));
        let gc_in = gc.input("g", Shape::new(&[n], DType::F32));
        let out = gc.complex_norm_sq_backward(zc, gc_in);
        gc.set_outputs(vec![out]);
        let plan = rlx_opt::memory::plan_memory_aligned(&gc, 64);
        let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
        let sched = rlx_cpu::thunk::compile_thunks(&gc, &arena);
        for (name, data) in [("z", &zv[..]), ("g", &gvals[..])] {
            let nid = gc
                .nodes()
                .iter()
                .find(|nn| matches!(&nn.op, Op::Input { name: n } if n == name))
                .unwrap()
                .id;
            let off = arena.byte_offset(nid);
            let buf = arena.raw_buf_mut();
            for (i, &v) in data.iter().enumerate() {
                buf[off + i * 4..off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        rlx_cpu::thunk::execute_thunks(&sched, arena.raw_buf_mut());
        let out_id = gc.outputs[0];
        let nbytes = gc.node(out_id).shape.size_bytes().unwrap();
        let ooff = arena.byte_offset(out_id);
        arena.raw_buf()[ooff..ooff + nbytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>()
    };
    let got = coreml_run_with_params(g, &[("z", &zv), ("g", &gvals)], &[]);
    assert!(
        close(&got, &want, 1e-5),
        "ComplexNormSqBackward MIL mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn fft_butterfly_stage_is_mil_native() {
    let batch = 1usize;
    let n_fft = 4usize;
    let stage = 0u32;
    let half = n_fft / 2;
    let mut g = Graph::new("fft_bf");
    // Avoid MIL reserved name `state` as a feature id.
    let st = g.input("st", Shape::new(&[batch, n_fft * 2], DType::F32));
    let gate = g.input("gate", Shape::new(&[half], DType::F32));
    let rev = g.input("rev", Shape::new(&[half], DType::F32));
    let tw_re = g.input("tw_re", Shape::new(&[half], DType::F32));
    let tw_im = g.input("tw_im", Shape::new(&[half], DType::F32));
    let y = g.fft_butterfly_stage(st, gate, rev, tw_re, tw_im, stage, n_fft as u32);
    g.set_outputs(vec![y]);
    assert_mil_only(&g);
    assert!(!rlx_coreml::host_exec::is_host_op(&Op::FftButterflyStage {
        stage: 0,
        n_fft: 4
    }));
    let state_v: Vec<f32> = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let gate_v = vec![1.0f32; half];
    let rev_v = vec![0.0f32; half];
    let tw_re_v = vec![1.0f32; half];
    let tw_im_v = vec![0.0f32; half];
    let want = cpu_run(
        &g,
        &[
            ("st", state_v.clone()),
            ("gate", gate_v.clone()),
            ("rev", rev_v.clone()),
            ("tw_re", tw_re_v.clone()),
            ("tw_im", tw_im_v.clone()),
        ],
    );
    let got = coreml_run_with_params(
        g,
        &[
            ("st", &state_v),
            ("gate", &gate_v),
            ("rev", &rev_v),
            ("tw_re", &tw_re_v),
            ("tw_im", &tw_im_v),
        ],
        &[],
    );
    assert!(
        close(&got, &want, 1e-5),
        "FftButterflyStage MIL mismatch:\n got={got:?}\nwant={want:?}"
    );
}
