// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// CPU-vs-wgpu parity for conv2d backward (input and weight). wgpu runs them
// on the GPU as the native conv3d backward kernels with a depth of 1, so this
// guards that mapping on the configs a plain LeNet never exercises: padding,
// stride, dilation and groups. (Before, both were host round trips.)

#![cfg(feature = "gpu")]

use rlx_compile::legalize_broadcast::run_with_remap;
use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

mod common;

fn write_slot(arena: &mut Arena, id: NodeId, data: &[f32]) {
    let off = arena.byte_offset(id);
    unsafe {
        let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
        for (i, &v) in data.iter().enumerate() {
            *p.add(i) = v;
        }
    }
}

fn read_slot(arena: &Arena, id: NodeId, n: usize) -> Vec<f32> {
    let off = arena.byte_offset(id);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..n).map(|i| *p.add(i)).collect()
    }
}

fn cpu_run(graph: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let (graph, remap) = run_with_remap(graph);
    let r = |id: NodeId| *remap.get(&id).unwrap_or(&id);
    let plan = rlx_opt::memory::plan_memory(&graph);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&graph, &arena);
    let slots: Vec<_> = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name } => Some((name.as_str(), n.id)),
            _ => None,
        })
        .collect();
    let n_out: usize = graph
        .nodes()
        .iter()
        .find(|n| n.id == r(graph.outputs[0]))
        .map(|n| n.shape.num_elements().unwrap())
        .unwrap();
    for (name, data) in inputs {
        let id = r(slots
            .iter()
            .find(|(n, _)| *n == *name)
            .unwrap_or_else(|| panic!("missing input {name}"))
            .1);
        write_slot(&mut arena, id, data);
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    read_slot(&arena, r(graph.outputs[0]), n_out)
}

fn wgpu_run(graph: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let session = Session::new(Device::Gpu);
    let mut compiled = session.compile_with(graph, &CompileOptions::default());
    compiled.run(inputs).remove(0)
}

fn assert_close(label: &str, cpu: &[f32], gpu: &[f32], tol: f32) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: length mismatch");
    let (mut max, mut argi) = (0f32, 0usize);
    for (i, (a, b)) in cpu.iter().zip(gpu).enumerate() {
        let d = (a - b).abs();
        if d > max {
            max = d;
            argi = i;
        }
    }
    assert!(
        max < tol,
        "{label}: max_abs_diff={max} at idx {argi} (cpu={}, gpu={})",
        cpu[argi],
        gpu[argi]
    );
}

// Deterministic pseudo-random, signed (so max-pool arg-max varies).
fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9e37_79b9);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn out_dim(h: usize, k: usize, s: usize, p: usize, d: usize) -> usize {
    (h + 2 * p - d * (k - 1) - 1) / s + 1
}

struct ConvCfg {
    name: &'static str,
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    k: usize,
    s: usize,
    p: usize,
    d: usize,
    groups: usize,
}

const CONV_CFGS: &[ConvCfg] = &[
    ConvCfg {
        name: "basic",
        n: 2,
        c_in: 3,
        h: 8,
        w: 8,
        c_out: 4,
        k: 3,
        s: 1,
        p: 0,
        d: 1,
        groups: 1,
    },
    ConvCfg {
        name: "pad+stride",
        n: 2,
        c_in: 3,
        h: 9,
        w: 9,
        c_out: 4,
        k: 3,
        s: 2,
        p: 1,
        d: 1,
        groups: 1,
    },
    ConvCfg {
        name: "groups",
        n: 2,
        c_in: 4,
        h: 8,
        w: 8,
        c_out: 4,
        k: 3,
        s: 1,
        p: 1,
        d: 1,
        groups: 2,
    },
    ConvCfg {
        name: "dilation",
        n: 2,
        c_in: 2,
        h: 10,
        w: 10,
        c_out: 3,
        k: 3,
        s: 1,
        p: 2,
        d: 2,
        groups: 1,
    },
    ConvCfg {
        name: "mnist-conv1",
        n: 8,
        c_in: 1,
        h: 28,
        w: 28,
        c_out: 6,
        k: 5,
        s: 1,
        p: 0,
        d: 1,
        groups: 1,
    },
];

#[test]
fn wgpu_conv2d_backward_input_matches_cpu() {
    let _gpu = common::serialize_gpu();
    let f = DType::F32;
    for c in CONV_CFGS {
        let ho = out_dim(c.h, c.k, c.s, c.p, c.d);
        let wo = out_dim(c.w, c.k, c.s, c.p, c.d);
        let cin_g = c.c_in / c.groups;
        let dy_n = c.n * c.c_out * ho * wo;
        let w_n = c.c_out * cin_g * c.k * c.k;
        let mut g = Graph::new("conv_bwd_in");
        let dy = g.input("dy", Shape::new(&[c.n, c.c_out, ho, wo], f));
        let wt = g.input("w", Shape::new(&[c.c_out, cin_g, c.k, c.k], f));
        let dx = g.conv2d_backward_input(
            dy,
            wt,
            Shape::new(&[c.n, c.c_in, c.h, c.w], f),
            vec![c.k, c.k],
            vec![c.s, c.s],
            vec![c.p, c.p],
            vec![c.d, c.d],
            c.groups,
        );
        g.set_outputs(vec![dx]);
        let dyv = fill(dy_n, 11);
        let wv = fill(w_n, 23);
        let want = cpu_run(g.clone(), &[("dy", &dyv), ("w", &wv)]);
        let got = wgpu_run(g, &[("dy", &dyv), ("w", &wv)]);
        assert_close(&format!("conv_bwd_input/{}", c.name), &want, &got, 1e-3);
    }
}

#[test]
fn wgpu_conv2d_backward_weight_matches_cpu() {
    let _gpu = common::serialize_gpu();
    let f = DType::F32;
    for c in CONV_CFGS {
        let ho = out_dim(c.h, c.k, c.s, c.p, c.d);
        let wo = out_dim(c.w, c.k, c.s, c.p, c.d);
        let cin_g = c.c_in / c.groups;
        let x_n = c.n * c.c_in * c.h * c.w;
        let dy_n = c.n * c.c_out * ho * wo;
        let mut g = Graph::new("conv_bwd_w");
        let x = g.input("x", Shape::new(&[c.n, c.c_in, c.h, c.w], f));
        let dy = g.input("dy", Shape::new(&[c.n, c.c_out, ho, wo], f));
        let dw = g.conv2d_backward_weight(
            x,
            dy,
            Shape::new(&[c.c_out, cin_g, c.k, c.k], f),
            vec![c.k, c.k],
            vec![c.s, c.s],
            vec![c.p, c.p],
            vec![c.d, c.d],
            c.groups,
        );
        g.set_outputs(vec![dw]);
        let xv = fill(x_n, 7);
        let dyv = fill(dy_n, 31);
        let want = cpu_run(g.clone(), &[("x", &xv), ("dy", &dyv)]);
        let got = wgpu_run(g, &[("x", &xv), ("dy", &dyv)]);
        assert_close(&format!("conv_bwd_weight/{}", c.name), &want, &got, 2e-3);
    }
}
