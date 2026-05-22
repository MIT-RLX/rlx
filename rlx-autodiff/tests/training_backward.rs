// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use rlx_autodiff::{grad_with_loss, prepare_graph_for_ad};
use rlx_compile::legalize_broadcast::run_with_remap;
use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

fn run_bwd(bwd: Graph, slots: &[(&str, &[f32])], out_idx: usize) -> Vec<f32> {
    let (bwd, remap) = run_with_remap(bwd);
    let r = |id: NodeId| *remap.get(&id).unwrap_or(&id);
    let plan = rlx_opt::memory::plan_memory(&bwd);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&bwd, &arena);
    for (name, data) in slots {
        let id = bwd
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name: n } if n == name))
            .map(|n| n.id)
            .expect(name);
        let off = arena.byte_offset(r(id));
        unsafe {
            let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
            for (i, &v) in data.iter().enumerate() {
                *p.add(i) = v;
            }
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let out_id = r(bwd.outputs[out_idx]);
    let n = bwd.node(out_id).shape.num_elements().unwrap();
    let off = arena.byte_offset(out_id);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..n).map(|i| *p.add(i)).collect()
    }
}

#[test]
fn legalize_multi_axis_reduce_splits_for_wgpu() {
    let f = DType::F32;
    let mut g = Graph::new("loss");
    let x = g.input("x", Shape::new(&[2, 3, 4], f));
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0, 1, 2],
            keep_dim: false,
        },
        vec![x],
        Shape::from_dims(&[], f),
    );
    g.set_outputs(vec![loss]);
    let prep = prepare_graph_for_ad(g);
    let loss_id = prep.outputs[0];
    let mut stack = vec![loss_id];
    let mut seen = std::collections::HashSet::new();
    let mut multi_on_path = false;
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = prep.node(id);
        if let Op::Reduce { axes, .. } = &node.op {
            multi_on_path |= axes.len() > 1;
        }
        for &inp in &node.inputs {
            stack.push(inp);
        }
    }
    assert!(!multi_on_path, "loss path must not contain multi-axis Reduce after prepare");
}

#[test]
fn rms_norm_backward_kernel_matches_composed() {
    let f = DType::F32;
    let mut g = Graph::new("rms");
    let x = g.input("x", Shape::new(&[2, 4], f));
    let gamma = g.input("gamma", Shape::new(&[4], f));
    let beta = g.input("beta", Shape::new(&[4], f));
    let y = g.add_node(Op::RmsNorm { axis: -1, eps: 1e-5 }, vec![x, gamma, beta], Shape::new(&[2, 4], f));
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0, 1],
            keep_dim: false,
        },
        vec![y],
        Shape::from_dims(&[], f),
    );
    g.set_outputs(vec![loss]);
    let prep = prepare_graph_for_ad(g);
    let bwd = grad_with_loss(&prep, &[x]);
    let xv: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 - 0.2).collect();
    let gv: Vec<f32> = vec![1.0, 1.1, 1.2, 1.3];
    let bv: Vec<f32> = vec![0.01; 4];
    let dx = run_bwd(
        bwd,
        &[("x", &xv), ("gamma", &gv), ("beta", &bv), ("d_output", &[1.0])],
        1,
    );
    assert_eq!(dx.len(), 8);
    assert!(dx.iter().all(|v| v.is_finite()));
}

#[test]
fn group_norm_backward_kernel_smoke() {
    let f = DType::F32;
    let n = 1usize;
    let c = 4usize;
    let h = 2usize;
    let w = 2usize;
    let mut g = Graph::new("gn");
    let x = g.input("x", Shape::new(&[n, c, h, w], f));
    let gamma = g.input("gamma", Shape::new(&[c], f));
    let beta = g.input("beta", Shape::new(&[c], f));
    let y = g.add_node(
        Op::GroupNorm {
            num_groups: 2,
            eps: 1e-5,
        },
        vec![x, gamma, beta],
        Shape::new(&[n, c, h, w], f),
    );
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0, 1, 2, 3],
            keep_dim: false,
        },
        vec![y],
        Shape::from_dims(&[], f),
    );
    g.set_outputs(vec![loss]);
    let prep = prepare_graph_for_ad(g);
    let bwd = grad_with_loss(&prep, &[x]);
    let plane = c * h * w;
    let xv: Vec<f32> = (0..plane).map(|i| 0.1 * i as f32 - 0.3).collect();
    let gv: Vec<f32> = vec![1.0, 1.1, 1.2, 1.3];
    let bv: Vec<f32> = vec![0.01; c];
    let dx = run_bwd(
        bwd,
        &[("x", &xv), ("gamma", &gv), ("beta", &bv), ("d_output", &[1.0])],
        1,
    );
    assert_eq!(dx.len(), plane);
    assert!(dx.iter().all(|v| v.is_finite()));
}

#[test]
fn gated_delta_net_unfused_before_autodiff() {
    let f = DType::F32;
    let mut g = Graph::new("gdn");
    let q = g.input("q", Shape::new(&[1, 2, 2, 4], f));
    let k = g.input("k", Shape::new(&[1, 2, 2, 4], f));
    let v = g.input("v", Shape::new(&[1, 2, 2, 4], f));
    let gv = g.input("g", Shape::new(&[1, 2, 2], f));
    let beta = g.input("beta", Shape::new(&[1, 2, 2], f));
    let y = g.add_node(
        Op::GatedDeltaNet {
            state_size: 4,
            carry_state: false,
        },
        vec![q, k, v, gv, beta],
        Shape::new(&[1, 2, 2, 4], f),
    );
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0, 1, 2, 3],
            keep_dim: false,
        },
        vec![y],
        Shape::from_dims(&[], f),
    );
    g.set_outputs(vec![loss]);
    let prep = prepare_graph_for_ad(g);
    let has_gdn = prep
        .nodes()
        .iter()
        .any(|n| matches!(n.op, Op::GatedDeltaNet { .. }));
    assert!(!has_gdn, "GatedDeltaNet must be unfused before autodiff");
}
