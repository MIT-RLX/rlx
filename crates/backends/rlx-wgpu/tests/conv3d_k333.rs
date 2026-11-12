// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `conv3d_k333_c8` (a 3×3×3 conv, stride 1, padding 1, its weights in
//! workgroup memory) against `conv3d_ctile_w4c8` (`RLX_WGPU_CONV3D_NO_K333=1`)
//! bit for bit, and rlx-cpu to rounding: widths that leave the last group of four short
//! or one past a row, input channels not a multiple of the eight staged at a
//! time, output channel tiles of one, depths and heights of one, a batch, the
//! bias in the store; then a stack half-packed and striped.
//!
//! One test: the switches are process-wide overrides.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

fn cpu_run(g: &Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let plan = rlx_opt::memory::plan_memory(g);
    let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
    let sched = rlx_cpu::thunk::compile_thunks(g, &arena);
    for node in g.nodes() {
        if let Op::Input { name } = &node.op {
            let data = inputs.iter().find(|(n, _)| n == name).unwrap().1;
            let off = arena.byte_offset(node.id);
            unsafe {
                let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
                std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len());
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

fn sh(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// A biased 3×3×3 conv `c_in → c_out` over `[n, c_in, dims]`, with `act`
/// folded into its store.
fn conv(
    g: &mut Graph,
    x: rlx_ir::NodeId,
    w: rlx_ir::NodeId,
    b: rlx_ir::NodeId,
    out: &[usize],
    act: Option<rlx_ir::op::Activation>,
) -> rlx_ir::NodeId {
    let y = g.add_node(
        Op::FusedConvBiasAct {
            kernel_size: vec![3; 3],
            stride: vec![1; 3],
            padding: vec![1; 3],
            dilation: vec![1; 3],
            groups: 1,
            activation: None,
            has_residual: false,
        },
        vec![x, w, b],
        sh(out),
    );
    match act {
        Some(a) => g.activation(a, y, sh(out)),
        None => y,
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The new kernel is the eight-channel tile's sums, and near rlx-cpu's.
fn case(n: usize, c_in: usize, c_out: usize, dims: [usize; 3], biased: bool) {
    let label = format!("n{n} c {c_in}->{c_out} {dims:?} bias={biased}");
    let x = ramp(n * c_in * dims.iter().product::<usize>(), 3);
    let w = ramp(c_out * c_in * 27, 7);
    let b: Vec<f32> = (0..c_out).map(|i| 0.25 * i as f32 - 0.5).collect();
    let out = [n, c_out, dims[0], dims[1], dims[2]];
    let build = || {
        let mut g = Graph::new("k333");
        let xi = g.input("x", sh(&[n, c_in, dims[0], dims[1], dims[2]]));
        let wi = g.input("w", sh(&[c_out, c_in, 3, 3, 3]));
        let y = if biased {
            let bi = g.input("b", sh(&[c_out]));
            conv(&mut g, xi, wi, bi, &out, None)
        } else {
            g.conv3d(xi, wi, [1; 3], [1; 3], [1; 3], 1)
        };
        g.set_outputs(vec![y]);
        g
    };
    let ins: Vec<(&str, &[f32])> = vec![("x", &x), ("w", &w), ("b", &b)];
    let ins = &ins[..if biased { 3 } else { 2 }];
    let got = WgpuExecutable::compile(build()).run(ins).remove(0);
    rlx_ir::env::set("RLX_WGPU_CONV3D_NO_K333", "1");
    let c8 = WgpuExecutable::compile(build()).run(ins).remove(0);
    rlx_ir::env::set("RLX_WGPU_CONV3D_NO_K333", "0");
    assert_eq!(bits(&got), bits(&c8), "{label}: k333 vs c8");
    // rlx-cpu's plain conv, the bias added after as the store adds it
    let mut g = Graph::new("k333_cpu");
    let xi = g.input("x", sh(&[n, c_in, dims[0], dims[1], dims[2]]));
    let wi = g.input("w", sh(&[c_out, c_in, 3, 3, 3]));
    let y = g.conv3d(xi, wi, [1; 3], [1; 3], [1; 3], 1);
    g.set_outputs(vec![y]);
    let per = dims.iter().product::<usize>();
    let cpu: Vec<f32> = cpu_run(&g, &ins[..2])
        .iter()
        .enumerate()
        .map(|(i, &v)| if biased { v + b[i / per % c_out] } else { v })
        .collect();
    let tol = 1e-4 * (c_in * 27) as f32;
    for (i, (a, e)) in got.iter().zip(&cpu).enumerate() {
        assert!((a - e).abs() <= tol, "{label} [{i}]: k333 {a} cpu {e}");
    }
}

#[test]
fn k333_matches_the_eight_channel_tile() {
    for w in [1, 2, 3, 4, 5, 6, 7, 8, 9, 13] {
        case(1, 3, 8, [3, 4, w], true);
    }
    // channels past one staged block of eight, short blocks, tiles of one
    case(1, 9, 17, [4, 3, 6], true);
    case(1, 16, 9, [2, 5, 7], false);
    case(1, 1, 5, [5, 5, 5], true);
    case(1, 20, 15, [3, 6, 10], true);
    // depths and heights of one: only the middle taps
    case(1, 4, 8, [1, 1, 9], true);
    case(1, 4, 8, [1, 6, 4], false);
    // a batch, and past one workgroup of 64 threads
    case(2, 5, 12, [3, 4, 11], true);
    case(1, 8, 24, [6, 16, 40], true);

    // a stack of three, half-packed and striped, against the eight-channel tile
    use rlx_ir::op::Activation;
    let (c, dims) = (10, [8, 48, 60]);
    let vol = [1, c, dims[0], dims[1], dims[2]];
    let build = || {
        let mut g = Graph::new("k333_stack");
        let mut y = g.input("x", sh(&[1, 1, dims[0], dims[1], dims[2]]));
        for l in 0..3 {
            let w = g.input(
                format!("w{l}"),
                sh(&[c, if l == 0 { 1 } else { c }, 3, 3, 3]),
            );
            let b = g.input(format!("b{l}"), sh(&[c]));
            y = conv(&mut g, y, w, b, &vol, Some(Activation::Relu));
        }
        g.set_outputs(vec![y]);
        g
    };
    let mut data: Vec<(String, Vec<f32>)> = vec![("x".into(), ramp(dims.iter().product(), 1))];
    for l in 0..3 {
        data.push((
            format!("w{l}"),
            ramp(c * if l == 0 { 1 } else { c } * 27, 2 + l),
        ));
        data.push((format!("b{l}"), ramp(c, 7 + l)));
    }
    let ins: Vec<(&str, &[f32])> = data
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    let run = || WgpuExecutable::compile(build()).run(&ins).remove(0);
    let k333 = run();
    rlx_ir::env::set("RLX_WGPU_CONV3D_NO_K333", "1");
    assert_eq!(bits(&k333), bits(&run()), "stack k333 vs c8");
    rlx_ir::env::set("RLX_WGPU_CONV3D_NO_K333", "0");
    rlx_ir::env::set("RLX_WGPU_HALF_ACTIVATIONS", "1");
    let half = run();
    rlx_ir::env::set("RLX_WGPU_CONV3D_NO_K333", "1");
    let half_c8 = run();
    rlx_ir::env::set("RLX_WGPU_CONV3D_NO_K333", "0");
    assert_eq!(bits(&half), bits(&half_c8), "half k333 vs half c8");
    // ~0.9 MiB f32 tensors (0.45 packed) in 2 MiB stripes: two-window convs
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "2");
    rlx_ir::env::set("RLX_WGPU_SHARD_STAGE_MIB", "1");
    rlx_ir::env::set("RLX_WGPU_ALLOW_SHARD", "1");
    let striped_half = run();
    rlx_ir::env::set("RLX_WGPU_HALF_ACTIVATIONS", "0");
    let striped = run();
    rlx_ir::env::clear_overrides();
    assert_eq!(
        bits(&striped_half),
        bits(&half),
        "striped half-packed = whole half-packed"
    );
    assert_eq!(bits(&striped), bits(&k333), "striped = whole");
}
