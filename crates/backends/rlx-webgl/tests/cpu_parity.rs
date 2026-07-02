// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.
//
// Verifies the WebGL lowering (planner + CPU executor) against RLX's own CPU
// autodiff. The WebGL fragment shaders mirror exec_cpu, so passing this test
// proves the numerics the GL path inherits.

use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};

fn build_loss(in_dim: usize, hidden: usize, out_dim: usize) -> (Graph, [NodeId; 4]) {
    let mut g = Graph::new("mlp_loss");
    let x = g.input("x", Shape::new(&[1, in_dim], DType::F32));
    let w1 = g.param("w1", Shape::new(&[in_dim, hidden], DType::F32));
    let b1 = g.param("b1", Shape::new(&[1, hidden], DType::F32));
    let w2 = g.param("w2", Shape::new(&[hidden, out_dim], DType::F32));
    let b2 = g.param("b2", Shape::new(&[1, out_dim], DType::F32));

    let h = g.matmul(x, w1, Shape::new(&[1, hidden], DType::F32));
    let h = g.add(h, b1);
    let h = g.relu(h);
    let y = g.matmul(h, w2, Shape::new(&[1, out_dim], DType::F32));
    let y = g.add(y, b2);

    let target = g.input("target", Shape::new(&[1, out_dim], DType::F32));
    let diff = g.sub(y, target);
    let sq = g.mul(diff, diff);
    let loss = g.sum(sq, vec![0, 1], false);
    g.set_outputs(vec![loss]);
    (g, [w1, b1, w2, b2])
}

fn weights(
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let w1 = (0..in_dim * hidden)
        .map(|i| 0.1 * (i as f32 + 1.0) - 0.2)
        .collect();
    let b1 = (0..hidden).map(|i| 0.05 * i as f32 - 0.1).collect();
    let w2 = (0..hidden * out_dim)
        .map(|i| 0.15 * i as f32 - 0.1)
        .collect();
    let b2 = (0..out_dim).map(|i| 0.02 * i as f32).collect();
    (w1, b1, w2, b2)
}

#[test]
fn webgl_plan_matches_cpu_autodiff() {
    let (in_dim, hidden, out_dim) = (3usize, 4usize, 2usize);
    let x = vec![0.5_f32, -0.3, 0.8];
    let target = vec![0.2_f32, 0.7];
    let (w1, b1, w2, b2) = weights(in_dim, hidden, out_dim);

    let (fwd, params) = build_loss(in_dim, hidden, out_dim);
    let bwd = rlx_autodiff::grad_with_loss(&fwd, &params);

    // Reference: RLX CPU backend.
    let mut sess = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(bwd.clone());
    sess.set_param("w1", &w1);
    sess.set_param("b1", &b1);
    sess.set_param("w2", &w2);
    sess.set_param("b2", &b2);
    let reference = sess.run(&[("x", &x), ("target", &target), ("d_output", &[1.0])]);

    // Under test: WebGL lowering executed on the CPU.
    let plan = rlx_webgl::build_plan(&bwd).expect("plan");
    let got = rlx_webgl::run_cpu(
        &plan,
        &[
            ("x", &x),
            ("target", &target),
            ("d_output", &[1.0]),
            ("w1", &w1),
            ("b1", &b1),
            ("w2", &w2),
            ("b2", &b2),
        ],
    )
    .expect("run_cpu");

    assert_eq!(reference.len(), got.len(), "output count");
    for (oi, (r, g)) in reference.iter().zip(&got).enumerate() {
        assert_eq!(
            r.len(),
            g.len(),
            "output {oi} length: {} vs {}",
            r.len(),
            g.len()
        );
        for (i, (a, b)) in r.iter().zip(g).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
                "output {oi}[{i}]: reference={a} webgl={b}"
            );
        }
    }
}
