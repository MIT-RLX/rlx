// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! VJP vs finite differences for ScatterElements / GatherNd / GatherElements.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, ScatterNdReduction, Shape};

fn sum_loss(g: &mut Graph, y: NodeId) -> NodeId {
    let rank = g.node(y).shape.rank();
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: (0..rank).collect(),
            keep_dim: false,
        },
        vec![y],
        Shape::from_dims(&[], DType::F32),
    )
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label} len");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "{label}[{i}]: got {a} want {b} (tol {tol})"
        );
    }
}

fn scatter_elements_grad_case(reduction: ScatterNdReduction) {
    let mut g = Graph::new("sel");
    let data = g.param("data", Shape::new(&[2, 4], DType::F32));
    let indices = g.input("indices", Shape::new(&[2, 4], DType::F32));
    let updates = g.param("updates", Shape::new(&[2, 4], DType::F32));
    let y = g.scatter_elements(data, indices, updates, 1, reduction);
    let loss = sum_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[data, updates]);
    let data_init: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 + 1.0).collect();
    let updates_init: Vec<f32> = (0..8).map(|i| (i as f32) * 0.2 + 0.5).collect();
    // Unique destinations along axis=1 (no overwrite collisions).
    let idx = [0.0f32, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0];

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("data", &data_init);
    compiled.set_param("updates", &updates_init);
    let outs = compiled.run(&[("indices", &idx[..]), ("d_output", &[1.0f32])]);
    assert!(outs.len() >= 3);
    let d_data = &outs[1];
    let d_updates = &outs[2];

    let loss_at = |d: &[f32], u: &[f32]| -> f32 {
        let mut fg = Graph::new("fwd");
        let data_i = fg.input("data", Shape::new(&[2, 4], DType::F32));
        let indices_i = fg.input("indices", Shape::new(&[2, 4], DType::F32));
        let updates_i = fg.input("updates", Shape::new(&[2, 4], DType::F32));
        let y = fg.scatter_elements(data_i, indices_i, updates_i, 1, reduction);
        let loss = sum_loss(&mut fg, y);
        fg.set_outputs(vec![loss]);
        rlx::Session::new(rlx::Device::Cpu)
            .compile(fg)
            .run(&[("data", d), ("indices", &idx[..]), ("updates", u)])
            .pop()
            .unwrap()[0]
    };

    let eps = 1e-3f32;
    let mut fd_data = vec![0f32; data_init.len()];
    for i in 0..data_init.len() {
        let mut plus = data_init.clone();
        let mut minus = data_init.clone();
        plus[i] += eps;
        minus[i] -= eps;
        fd_data[i] = (loss_at(&plus, &updates_init) - loss_at(&minus, &updates_init)) / (2.0 * eps);
    }
    let mut fd_upd = vec![0f32; updates_init.len()];
    for i in 0..updates_init.len() {
        let mut plus = updates_init.clone();
        let mut minus = updates_init.clone();
        plus[i] += eps;
        minus[i] -= eps;
        fd_upd[i] = (loss_at(&data_init, &plus) - loss_at(&data_init, &minus)) / (2.0 * eps);
    }

    assert_close(d_data, &fd_data, 2e-2, &format!("{reduction:?} d_data"));
    assert_close(
        d_updates,
        &fd_upd,
        2e-2,
        &format!("{reduction:?} d_updates"),
    );
}

#[test]
fn scatter_elements_vjp_none_matches_fd() {
    scatter_elements_grad_case(ScatterNdReduction::None);
}

#[test]
fn scatter_elements_vjp_add_matches_fd() {
    scatter_elements_grad_case(ScatterNdReduction::Add);
}

#[test]
fn gather_nd_vjp_matches_fd() {
    let mut g = Graph::new("gnd");
    let data = g.param("data", Shape::new(&[4, 3], DType::F32));
    let indices = g.input("indices", Shape::new(&[2, 1], DType::F32));
    let y = g.gather_nd(data, indices, 0, Shape::new(&[2, 3], DType::F32));
    let loss = sum_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[data]);
    let data_init: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let idx = [1.0f32, 3.0];

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("data", &data_init);
    let outs = compiled.run(&[("indices", &idx[..]), ("d_output", &[1.0f32])]);
    let d_data = &outs[1];

    let loss_at = |d: &[f32]| -> f32 {
        let mut fg = Graph::new("fwd");
        let data_i = fg.input("data", Shape::new(&[4, 3], DType::F32));
        let indices_i = fg.input("indices", Shape::new(&[2, 1], DType::F32));
        let y = fg.gather_nd(data_i, indices_i, 0, Shape::new(&[2, 3], DType::F32));
        let loss = sum_loss(&mut fg, y);
        fg.set_outputs(vec![loss]);
        rlx::Session::new(rlx::Device::Cpu)
            .compile(fg)
            .run(&[("data", d), ("indices", &idx[..])])
            .pop()
            .unwrap()[0]
    };

    let eps = 1e-3f32;
    let mut fd = vec![0f32; data_init.len()];
    for i in 0..data_init.len() {
        let mut plus = data_init.clone();
        let mut minus = data_init.clone();
        plus[i] += eps;
        minus[i] -= eps;
        fd[i] = (loss_at(&plus) - loss_at(&minus)) / (2.0 * eps);
    }
    assert_close(d_data, &fd, 2e-2, "gather_nd d_data");
}

#[test]
fn gather_elements_vjp_matches_fd() {
    let mut g = Graph::new("gel");
    let data = g.param("data", Shape::new(&[2, 4], DType::F32));
    let indices = g.input("indices", Shape::new(&[2, 4], DType::F32));
    let y = g.gather_elements(data, indices, 1);
    let loss = sum_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[data]);
    let data_init: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 + 1.0).collect();
    let idx = [0.0f32, 2.0, 1.0, 3.0, 1.0, 0.0, 3.0, 2.0];

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("data", &data_init);
    let outs = compiled.run(&[("indices", &idx[..]), ("d_output", &[1.0f32])]);
    let d_data = &outs[1];

    let loss_at = |d: &[f32]| -> f32 {
        let mut fg = Graph::new("fwd");
        let data_i = fg.input("data", Shape::new(&[2, 4], DType::F32));
        let indices_i = fg.input("indices", Shape::new(&[2, 4], DType::F32));
        let y = fg.gather_elements(data_i, indices_i, 1);
        let loss = sum_loss(&mut fg, y);
        fg.set_outputs(vec![loss]);
        rlx::Session::new(rlx::Device::Cpu)
            .compile(fg)
            .run(&[("data", d), ("indices", &idx[..])])
            .pop()
            .unwrap()[0]
    };

    let eps = 1e-3f32;
    let mut fd = vec![0f32; data_init.len()];
    for i in 0..data_init.len() {
        let mut plus = data_init.clone();
        let mut minus = data_init.clone();
        plus[i] += eps;
        minus[i] -= eps;
        fd[i] = (loss_at(&plus) - loss_at(&minus)) / (2.0 * eps);
    }
    assert_close(d_data, &fd, 2e-2, "gather_elements d_data");
}

/// Backward through a gather whose index is a FLAT list.
///
/// Gather's output rank is `data_rank - 1 + index_rank`, so a rank-1 index is
/// legal and keeps the rank unchanged — but every backend computed the gathered
/// count as `idx_shape.dim(axis)`, which has no dimension at `axis` for this
/// form and panicked with "Shape::dim(1) out of bounds for rank 1". The forward
/// pass worked, so this only bit when a graph needed gradients.
#[test]
fn gather_backward_accepts_a_rank1_index() {
    use rlx_ir::GraphExt;

    const D: usize = 3; // trailing width
    const N: usize = 5; // source positions along the gather axis
    let idx = [3.0f32, 0.0, 4.0, 0.0]; // repeats on purpose: grads must accumulate

    let build = |g: &mut Graph, data: NodeId, indices: NodeId| -> NodeId {
        let y = g.gather_(data, indices, 1);
        sum_loss(g, y)
    };

    let mut g = Graph::new("gather_rank1");
    let data = g.param("data", Shape::new(&[1, N, D], DType::F32));
    let indices = g.input("indices", Shape::new(&[idx.len()], DType::F32));
    let loss = build(&mut g, data, indices);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[data]);
    let data_init: Vec<f32> = (0..N * D).map(|i| (i as f32) * 0.1 + 1.0).collect();
    // Runs on CPU by default; set RLX_FORCE_DEVICE=metal|mlx|gpu|cuda|rocm to
    // check the same fix on an accelerator, since the bug was present verbatim
    // in all five backends and compiling is not the same as executing.
    let device = rlx_ir::env::var("RLX_FORCE_DEVICE")
        .and_then(|s| rlx::parse_device(&s).ok())
        .unwrap_or(rlx::Device::Cpu);
    eprintln!("gather_backward_accepts_a_rank1_index on {device:?}");
    let mut compiled = rlx::Session::new(device).compile(bwd);
    compiled.set_param("data", &data_init);
    let outs = compiled.run(&[("indices", &idx[..]), ("d_output", &[1.0f32])]);
    let d_data = &outs[1];

    // d(sum of gathered)/d(data[p]) is just how many times p was gathered.
    let mut want = vec![0f32; N * D];
    for &p in &idx {
        for k in 0..D {
            want[p as usize * D + k] += 1.0;
        }
    }
    assert_close(d_data, &want, 1e-5, "rank-1 gather d_data");
}
