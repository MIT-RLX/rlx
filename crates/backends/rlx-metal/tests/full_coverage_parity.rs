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

//! CPU-vs-Metal parity for newly claimed full-coverage ops (3D + AxialRope2d + Fma).

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_metal::backend::MetalExecutable;
use rlx_metal::thunk::{Thunk, ThunkSchedule, thunk_name};

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
    let shape = g.shape(out_id);
    let n_elems = shape.num_elements().unwrap();
    // Host f32-lane count: C64/C128 occupy 2/4 lanes per element.
    let n_f32 = match shape.dtype() {
        DType::C64 => n_elems * 2,
        DType::C128 => n_elems * 4,
        _ => n_elems,
    };
    let off = arena.byte_offset(out_id);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..n_f32).map(|i| *p.add(i)).collect()
    }
}

fn metal_run(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let mut exe = MetalExecutable::compile(g);
    exe.run(inputs).into_iter().next().unwrap()
}

fn assert_native_thunk(g: &Graph, want: &str) {
    let plan = rlx_opt::memory::plan_memory(g);
    let arena = rlx_metal::arena::Arena::from_plan(plan);
    let sched = ThunkSchedule::compile(g, &arena);
    let names: Vec<&str> = sched.thunks.iter().map(thunk_name).collect();
    assert!(
        !sched
            .thunks
            .iter()
            .any(|t| matches!(t, Thunk::HostOp { .. })),
        "expected native {want}, got HostOp; schedule={names:?}"
    );
    assert!(
        names.iter().any(|n| *n == want),
        "expected thunk `{want}` in schedule, got {names:?}"
    );
}

#[test]
fn conv3d_matches_cpu() {
    let mut g = Graph::new("conv3d");
    let x = g.input("x", Shape::new(&[1, 1, 3, 3, 3], DType::F32));
    let w = g.input("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.conv3d(x, w, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
    g.set_outputs(vec![y]);
    assert_native_thunk(&g, "conv3d");
    let xv: Vec<f32> = (1..=27).map(|v| v as f32).collect();
    let wv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let want = cpu_run(&g, &[("x", xv.clone()), ("w", wv.clone())]);
    let got = metal_run(g, &[("x", &xv), ("w", &wv)]);
    assert!(
        close(&got, &want, 1e-4),
        "Conv3d mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv_transpose3d_matches_cpu() {
    let mut g = Graph::new("ct3d");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let w = g.input("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.conv_transpose3d(x, w, [2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], 1);
    g.set_outputs(vec![y]);
    assert_native_thunk(&g, "conv_transpose3d");
    let xv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let wv: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let want = cpu_run(&g, &[("x", xv.clone()), ("w", wv.clone())]);
    let got = metal_run(g, &[("x", &xv), ("w", &wv)]);
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
    let xv: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 * 0.13).sin()).collect();
    let want = cpu_run(&g, &[("x", xv.clone())]);
    let got = metal_run(g, &[("x", &xv)]);
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
    let got = metal_run(g, &[("a", &av), ("b", &bv), ("c", &cv)]);
    assert!(
        close(&got, &want, 1e-5),
        "Fma mismatch:\n got={got:?}\nwant={want:?}"
    );
    assert!(close(&got, &[1.5, 1.0, 0.5, 4.0], 1e-5));
}

#[test]
fn relu_backward_matches_cpu() {
    let mut g = Graph::new("relu_bwd");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let dy = g.input("dy", Shape::new(&[4], DType::F32));
    let dx = g.relu_backward(x, dy);
    g.set_outputs(vec![dx]);
    assert_native_thunk(&g, "relu_backward");
    let xv = vec![-1.0f32, 0.0, 0.5, 2.0];
    let dyv = vec![0.1f32, 0.2, 0.3, 0.4];
    let want = cpu_run(&g, &[("x", xv.clone()), ("dy", dyv.clone())]);
    let got = metal_run(g, &[("x", &xv), ("dy", &dyv)]);
    assert!(
        close(&got, &want, 1e-6),
        "ReluBackward mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn activation_backward_silu_matches_cpu() {
    use rlx_ir::op::Activation;
    let mut g = Graph::new("act_bwd");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let dy = g.input("dy", Shape::new(&[4], DType::F32));
    let dx = g.activation_backward(Activation::Silu, x, dy);
    g.set_outputs(vec![dx]);
    assert_native_thunk(&g, "activation_backward");
    let xv = vec![-1.0f32, 0.0, 0.5, 2.0];
    let dyv = vec![0.1f32, 0.2, 0.3, 0.4];
    let want = cpu_run(&g, &[("x", xv.clone()), ("dy", dyv.clone())]);
    let got = metal_run(g, &[("x", &xv), ("dy", &dyv)]);
    assert!(
        close(&got, &want, 1e-5),
        "ActivationBackward(Silu) mismatch:\n got={got:?}\nwant={want:?}"
    );
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
    assert_native_thunk(&g, "fake_quantize_perbatch");
    let xv: Vec<f32> = (0..6).map(|i| 0.07 * (i as f32) - 1.3).collect();
    let want = cpu_run(&g, &[("x", xv.clone())]);
    let got = metal_run(g, &[("x", &xv)]);
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
    assert_native_thunk(&g, "fake_quantize_fixed");
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
    let mut exe = MetalExecutable::compile(g);
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
    assert_native_thunk(&g, "fake_quantize_perbatch");
    let xv: Vec<f32> = (0..12).map(|i| 0.11 * (i as f32) - 0.9).collect();
    let want = cpu_run(&g, &[("x", xv.clone())]);
    let got = metal_run(g, &[("x", &xv)]);
    assert!(
        close(&got, &want, 1e-4),
        "FakeQuantize PerBatch channel mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn complex_norm_sq_matches_cpu() {
    let mut g = Graph::new("cnorm");
    let z = g.input("z", Shape::new(&[2], DType::C64));
    let y = g.complex_norm_sq(z);
    g.set_outputs(vec![y]);
    assert_native_thunk(&g, "complex_norm_sq");
    // Two complex values as f32 lane pairs (re, im): |3+4i|²=25, |i|²=1.
    let zv = vec![3.0f32, 4.0, 0.0, 1.0];
    let want = cpu_run(&g, &[("z", zv.clone())]);
    let got = metal_run(g, &[("z", &zv)]);
    assert!(
        close(&got, &want, 1e-5),
        "ComplexNormSq mismatch:\n got={got:?}\nwant={want:?}"
    );
    assert!(close(&got, &[25.0, 1.0], 1e-5));
}

#[test]
fn complex_norm_sq_backward_matches_cpu() {
    let mut g = Graph::new("cnorm_bwd");
    let z = g.input("z", Shape::new(&[2], DType::C64));
    let gv = g.input("g", Shape::new(&[2], DType::F32));
    let dz = g.complex_norm_sq_backward(z, gv);
    g.set_outputs(vec![dz]);
    assert_native_thunk(&g, "complex_norm_sq_backward");
    // dz = g · z → [(2·3, 2·4), (0.5·0, 0.5·1)] = [(6, 8), (0, 0.5)]
    let zv = vec![3.0f32, 4.0, 0.0, 1.0];
    let gvals = vec![2.0f32, 0.5];
    let want = cpu_run(&g, &[("z", zv.clone()), ("g", gvals.clone())]);
    let got = metal_run(g, &[("z", &zv), ("g", &gvals)]);
    assert!(
        close(&got, &want, 1e-5),
        "ComplexNormSqBackward mismatch:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conjugate_c64_matches_cpu() {
    let mut g = Graph::new("conj");
    let z = g.input("z", Shape::new(&[2], DType::C64));
    let y = g.conjugate(z);
    g.set_outputs(vec![y]);
    assert_native_thunk(&g, "conjugate_c64");
    let zv = vec![3.0f32, 4.0, -1.0, 2.0];
    let want = cpu_run(&g, &[("z", zv.clone())]);
    let got = metal_run(g, &[("z", &zv)]);
    assert!(
        close(&got, &want, 1e-5),
        "Conjugate mismatch:\n got={got:?}\nwant={want:?}"
    );
    assert!(close(&got, &[3.0, -4.0, -1.0, -2.0], 1e-5));
}

#[test]
fn fft_butterfly_stage_matches_cpu() {
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
    assert_native_thunk(&g, "fft_butterfly_stage");
    let state_v = vec![1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let gate_v = vec![1.0f32, 0.0]; // second pair identity-copy
    let rev_v = vec![0.0f32, 0.0];
    let tw_re_v = vec![1.0f32, 1.0];
    let tw_im_v = vec![0.0f32, 0.0];
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
    let got = metal_run(
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
