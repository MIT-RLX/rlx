// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// CPU-vs-Metal parity for the native GPU training-backward kernels
// (maxpool2d_backward, conv2d_backward_input, conv2d_backward_weight).
// These run as CPU fallbacks on every other backend; the Metal path now
// dispatches dedicated MSL kernels, so this guards the configs that the
// MNIST LeNet bench never exercises: padding, stride, dilation, groups,
// and overlapping pooling windows.

#![cfg(all(target_os = "macos", feature = "metal"))]

use rlx_compile::legalize_broadcast::run_with_remap;
use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

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

fn metal_run(graph: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let session = Session::new(Device::Metal);
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
fn metal_conv2d_backward_input_matches_cpu() {
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
        let got = metal_run(g, &[("dy", &dyv), ("w", &wv)]);
        assert_close(&format!("conv_bwd_input/{}", c.name), &want, &got, 1e-3);
    }
}

#[test]
fn metal_conv2d_backward_weight_matches_cpu() {
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
        let got = metal_run(g, &[("x", &xv), ("dy", &dyv)]);
        assert_close(&format!("conv_bwd_weight/{}", c.name), &want, &got, 2e-3);
    }
}

struct PoolCfg {
    name: &'static str,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    s: usize,
    p: usize,
}

const POOL_CFGS: &[PoolCfg] = &[
    PoolCfg {
        name: "2x2-s2",
        n: 2,
        c: 3,
        h: 8,
        w: 8,
        k: 2,
        s: 2,
        p: 0,
    },
    PoolCfg {
        name: "3x3-s2-pad1-overlap",
        n: 2,
        c: 3,
        h: 9,
        w: 9,
        k: 3,
        s: 2,
        p: 1,
    },
    PoolCfg {
        name: "3x3-s1-overlap",
        n: 2,
        c: 2,
        h: 7,
        w: 7,
        k: 3,
        s: 1,
        p: 0,
    },
];

// Native Metal softmax-cross-entropy forward (with integer labels) + backward,
// vs the CPU fused thunks. Guards the kernels that replace the
// softmax+one-hot(compare/where) decomposition on Metal.
#[test]
fn metal_softmax_cross_entropy_with_logits_matches_cpu() {
    let f = DType::F32;
    for &(n, c) in &[(4usize, 10usize), (7, 3), (128, 10), (3, 257)] {
        let mut g = Graph::new("sce_fwd");
        let logits = g.input("logits", Shape::new(&[n, c], f));
        let labels = g.input("labels", Shape::new(&[n], f));
        let loss = g.softmax_cross_entropy_with_logits(logits, labels);
        g.set_outputs(vec![loss]);
        let lv = fill(n * c, 9);
        let lb: Vec<f32> = (0..n).map(|i| (i * 7 % c) as f32).collect();
        let want = cpu_run(g.clone(), &[("logits", &lv), ("labels", &lb)]);
        let got = metal_run(g, &[("logits", &lv), ("labels", &lb)]);
        assert_close(&format!("sce_fwd/{n}x{c}"), &want, &got, 1e-4);
    }
}

#[test]
fn metal_softmax_cross_entropy_backward_matches_cpu() {
    let f = DType::F32;
    for &(n, c) in &[(4usize, 10usize), (7, 3), (128, 10), (3, 257)] {
        let mut g = Graph::new("sce_bwd");
        let logits = g.input("logits", Shape::new(&[n, c], f));
        let labels = g.input("labels", Shape::new(&[n], f));
        let d_loss = g.input("d_loss", Shape::new(&[n], f));
        let dx = g.softmax_cross_entropy_backward(logits, labels, d_loss);
        g.set_outputs(vec![dx]);
        let lv = fill(n * c, 13);
        let lb: Vec<f32> = (0..n).map(|i| (i * 5 % c) as f32).collect();
        let dl: Vec<f32> = (0..n).map(|i| 1.0 / n as f32 + 0.01 * i as f32).collect();
        let want = cpu_run(
            g.clone(),
            &[("logits", &lv), ("labels", &lb), ("d_loss", &dl)],
        );
        let got = metal_run(g, &[("logits", &lv), ("labels", &lb), ("d_loss", &dl)]);
        assert_close(&format!("sce_bwd/{n}x{c}"), &want, &got, 1e-4);
    }
}

#[test]
fn metal_maxpool2d_backward_matches_cpu() {
    let f = DType::F32;
    for c in POOL_CFGS {
        let ho = out_dim(c.h, c.k, c.s, c.p, 1);
        let wo = out_dim(c.w, c.k, c.s, c.p, 1);
        let x_n = c.n * c.c * c.h * c.w;
        let dy_n = c.n * c.c * ho * wo;
        let mut g = Graph::new("pool_bwd");
        let x = g.input("x", Shape::new(&[c.n, c.c, c.h, c.w], f));
        let dy = g.input("dy", Shape::new(&[c.n, c.c, ho, wo], f));
        let dx = g.maxpool2d_backward(x, dy, vec![c.k, c.k], vec![c.s, c.s], vec![c.p, c.p]);
        g.set_outputs(vec![dx]);
        let xv = fill(x_n, 3);
        let dyv = fill(dy_n, 41);
        let want = cpu_run(g.clone(), &[("x", &xv), ("dy", &dyv)]);
        let got = metal_run(g, &[("x", &xv), ("dy", &dyv)]);
        assert_close(&format!("maxpool_bwd/{}", c.name), &want, &got, 1e-4);
    }
}
