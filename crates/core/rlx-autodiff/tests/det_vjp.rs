// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Det` / `Op::LogDet` forward vs a 3×3 cofactor reference, and their VJPs
//! (`∂det/∂A = det·A⁻ᵀ`, `∂logdet/∂A = A⁻ᵀ`) vs finite differences.

use rlx_autodiff::grad_with_loss;
use rlx_ir::{DType, Graph, Shape};

const N: usize = 3;

fn general_a() -> Vec<f32> {
    vec![2.0, 0.3, -0.5, 0.1, 1.7, 0.4, -0.2, 0.6, 2.3]
}
fn ref_det(a: &[f32]) -> f32 {
    a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6])
}
fn scalar() -> Shape {
    Shape::from_dims(&[], DType::F32)
}

fn eval(a: &[f32], logdet: bool) -> f32 {
    let mut g = Graph::new("d");
    let ai = g.input("a", Shape::new(&[N, N], DType::F32));
    let o = if logdet {
        g.logdet(ai, scalar())
    } else {
        g.det(ai, scalar())
    };
    g.set_outputs(vec![o]);
    rlx::Session::new(rlx::Device::Cpu)
        .compile(g)
        .run(&[("a", a)])
        .pop()
        .unwrap()[0]
}

#[test]
fn det_logdet_forward() {
    let a = general_a();
    let want = ref_det(&a);
    assert!((eval(&a, false) - want).abs() < 1e-4, "det vs {want}");
    assert!((eval(&a, true) - want.abs().ln()).abs() < 1e-4, "logdet");
}

fn grad(a: &[f32], logdet: bool) -> Vec<f32> {
    let mut g = Graph::new("g");
    let ai = g.param("a", Shape::new(&[N, N], DType::F32));
    let o = if logdet {
        g.logdet(ai, scalar())
    } else {
        g.det(ai, scalar())
    };
    g.set_outputs(vec![o]);
    let bwd = grad_with_loss(&g, &[ai]);
    let mut c = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    c.set_param("a", a);
    c.run(&[("d_output", &[1.0f32])])[1].clone()
}

#[test]
fn det_logdet_vjp_matches_fd() {
    let a = general_a();
    for logdet in [false, true] {
        let da = grad(&a, logdet);
        let eps = 1e-3f32;
        for k in 0..N * N {
            let (mut ap, mut am) = (a.clone(), a.clone());
            ap[k] += eps;
            am[k] -= eps;
            let fd = (eval(&ap, logdet) - eval(&am, logdet)) / (2.0 * eps);
            assert!(
                (fd - da[k]).abs() <= 2e-2 * (1.0 + fd.abs()),
                "logdet={logdet} d[{k}]: analytic {} vs FD {fd}",
                da[k]
            );
        }
    }
}
