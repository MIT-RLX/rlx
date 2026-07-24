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

//! CPU-vs-Metal parity for 3-D training backward ops.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_metal::backend::MetalExecutable;
use rlx_metal::thunk::{Thunk, ThunkSchedule, thunk_name};

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
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
    let n_elems = g.shape(out_id).num_elements().unwrap();
    let off = arena.byte_offset(out_id);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..n_elems).map(|i| *p.add(i)).collect()
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
        names.contains(&want),
        "expected thunk `{want}` in schedule, got {names:?}"
    );
}

fn make_conv3d_bwd_input() -> (Graph, Vec<f32>, Vec<f32>) {
    let mut g = Graph::new("c3d_bwd_in");
    let dy = g.input("dy", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let w = g.input("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let dx = g.conv3d_backward_input(
        dy,
        w,
        Shape::new(&[1, 1, 3, 3, 3], DType::F32),
        vec![2, 2, 2],
        vec![1, 1, 1],
        vec![0, 0, 0],
        vec![1, 1, 1],
        1,
    );
    g.set_outputs(vec![dx]);
    let dyv: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let wv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    (g, dyv, wv)
}

fn make_conv3d_bwd_weight() -> (Graph, Vec<f32>, Vec<f32>) {
    let mut g = Graph::new("c3d_bwd_w");
    let x = g.input("x", Shape::new(&[1, 1, 3, 3, 3], DType::F32));
    let dy = g.input("dy", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let dw = g.conv3d_backward_weight(
        x,
        dy,
        Shape::new(&[1, 1, 2, 2, 2], DType::F32),
        vec![2, 2, 2],
        vec![1, 1, 1],
        vec![0, 0, 0],
        vec![1, 1, 1],
        1,
    );
    g.set_outputs(vec![dw]);
    let xv: Vec<f32> = (1..=27).map(|v| v as f32).collect();
    let dyv: Vec<f32> = (1..=8).map(|v| v as f32 * 0.5).collect();
    (g, xv, dyv)
}

fn make_maxpool3d_bwd() -> (Graph, Vec<f32>, Vec<f32>) {
    let mut g = Graph::new("mp3d_bwd");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let dy = g.input("dy", Shape::new(&[1, 1, 1, 1, 1], DType::F32));
    let dx = g.maxpool3d_backward(x, dy, vec![2, 2, 2], vec![1, 1, 1], vec![0, 0, 0]);
    g.set_outputs(vec![dx]);
    let xv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0];
    let dyv = vec![1.0];
    (g, xv, dyv)
}

#[test]
fn conv3d_backward_input_matches_cpu() {
    if !rlx_metal::is_available() {
        eprintln!("[rlx-metal c3d_bwd_in] no Metal device — skipping");
        return;
    }
    let (g, dyv, wv) = make_conv3d_bwd_input();
    assert_native_thunk(&g, "conv3d_backward_input");
    let want = cpu_run(&g, &[("dy", dyv.clone()), ("w", wv.clone())]);
    let got = metal_run(g, &[("dy", &dyv), ("w", &wv)]);
    assert!(
        close(&got, &want, 1e-4),
        "Conv3dBackwardInput Metal vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv3d_backward_weight_matches_cpu() {
    if !rlx_metal::is_available() {
        eprintln!("[rlx-metal c3d_bwd_w] no Metal device — skipping");
        return;
    }
    let (g, xv, dyv) = make_conv3d_bwd_weight();
    assert_native_thunk(&g, "conv3d_backward_weight");
    let want = cpu_run(&g, &[("x", xv.clone()), ("dy", dyv.clone())]);
    let got = metal_run(g, &[("x", &xv), ("dy", &dyv)]);
    assert!(
        close(&got, &want, 1e-4),
        "Conv3dBackwardWeight Metal vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn maxpool3d_backward_matches_cpu() {
    if !rlx_metal::is_available() {
        eprintln!("[rlx-metal mp3d_bwd] no Metal device — skipping");
        return;
    }
    let (g, xv, dyv) = make_maxpool3d_bwd();
    assert_native_thunk(&g, "maxpool3d_backward");
    let want = cpu_run(&g, &[("x", xv.clone()), ("dy", dyv.clone())]);
    let got = metal_run(g, &[("x", &xv), ("dy", &dyv)]);
    assert!(
        close(&got, &want, 1e-5),
        "MaxPool3dBackward Metal vs CPU:\n got={got:?}\nwant={want:?}"
    );
}
