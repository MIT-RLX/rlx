// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Finite-difference correctness of the DECOMPOSED activation derivatives
//! (`rlx_autodiff::activation_deriv_wrt_x`).
//!
//! Backends WITHOUT a native `Op::ActivationBackward` kernel (CUDA, ROCm, …)
//! lower every activation's gradient through this decomposition, so a wrong
//! formula here **silently corrupts their training gradients** while backends
//! with a native kernel (CPU, MLX) stay correct. That exact asymmetry is what
//! made a conv codec DIVERGE on CUDA at LRs where CPU and MLX trained fine: the
//! `Gelu` decomposition had dropped two whole terms (it computed `≈ d/dx tanh(u)`
//! instead of `d/dx[0.5·x·(1+tanh(u))]`), an ~1.0-magnitude gradient error that
//! no structural test caught.
//!
//! This checks each differentiable activation's decomposed derivative against a
//! central finite difference of its own forward — ground truth, on the
//! `RLX_PARITY_DEVICE` (default CPU). Because the decomposition is
//! backend-agnostic, the CPU run alone validates the formula every GPU backend
//! relies on; pointing `RLX_PARITY_DEVICE=cuda` additionally exercises the
//! decomposition-op kernels end-to-end.

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

const F: DType = DType::F32;

fn target() -> Device {
    match std::env::var("RLX_PARITY_DEVICE") {
        Ok(s) => rlx_runtime::parse_device(&s).unwrap_or(Device::Cpu),
        Err(_) => Device::Cpu,
    }
}

fn run1(g: &Graph, dev: Device, x: &[f32]) -> Vec<f32> {
    Session::new(dev).compile(g.clone()).run(&[("x", x)])[0].clone()
}

#[test]
fn activation_deriv_wrt_x_matches_finite_difference() {
    let dev = target();
    if !is_available(dev) {
        eprintln!("activation_deriv_fd: {dev:?} unavailable — skipping");
        return;
    }
    // (kind, safe test points). Domain-restricted kinds (Log/Sqrt/Rsqrt) use
    // positive points; Abs's kink at 0 is excluded by construction.
    let cases: &[(Activation, &[f32])] = &[
        (Activation::Gelu, &[-2.5, -1.0, -0.3, 0.3, 1.0, 2.5]),
        (Activation::GeluApprox, &[-2.5, -1.0, -0.3, 0.3, 1.0, 2.5]),
        (Activation::Silu, &[-2.5, -1.0, -0.3, 0.3, 1.0, 2.5]),
        (Activation::Sigmoid, &[-2.0, -0.5, 0.5, 2.0]),
        (Activation::Tanh, &[-2.0, -0.5, 0.5, 2.0]),
        (Activation::Exp, &[-1.5, -0.5, 0.5, 1.5]),
        (Activation::Sqrt, &[0.2, 0.7, 1.5, 3.0]),
        (Activation::Log, &[0.25, 0.7, 1.5, 3.0]),
        (Activation::Rsqrt, &[0.3, 0.7, 1.5, 3.0]),
        (Activation::Sin, &[-2.0, -0.5, 0.5, 2.0]),
        (Activation::Cos, &[-2.0, -0.5, 0.5, 2.0]),
        (Activation::Recip, &[-2.0, -0.5, 0.5, 2.0]),
    ];
    let h = 1e-2f32;
    let mut failures = Vec::new();
    for (kind, xs) in cases {
        let n = xs.len();
        let shape = Shape::new(&[n], F);

        // graph 1: the DECOMPOSED derivative under test.
        let mut gd = Graph::new("deriv");
        let xin = gd.input("x", shape.clone());
        let d = rlx_autodiff::activation_deriv::activation_deriv_wrt_x(
            &mut gd, *kind, xin, None, &shape,
        );
        gd.set_outputs(vec![d]);
        // graph 2: the forward activation (for the FD reference).
        let mut gf = Graph::new("fwd");
        let xf = gf.input("x", shape.clone());
        let yf = gf.activation(*kind, xf, shape.clone());
        gf.set_outputs(vec![yf]);

        let analytic = run1(&gd, dev, xs);
        let xp: Vec<f32> = xs.iter().map(|v| v + h).collect();
        let xm: Vec<f32> = xs.iter().map(|v| v - h).collect();
        let fp = run1(&gf, dev, &xp);
        let fm = run1(&gf, dev, &xm);

        for i in 0..n {
            let fd = (fp[i] - fm[i]) / (2.0 * h);
            let err = (analytic[i] - fd).abs();
            // FD on f32 is good to ~1e-3; the bug this guards against is ~1.0.
            let tol = 2e-2 + 3e-2 * fd.abs();
            let ok = err <= tol;
            eprintln!(
                "  {kind:?}@x={:+.2}: analytic={:+.5} fd={:+.5} err={:.5} {}",
                xs[i],
                analytic[i],
                fd,
                err,
                if ok { "ok" } else { "MISMATCH" }
            );
            if !ok {
                failures.push(format!("{kind:?}@x={}", xs[i]));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "decomposed activation derivative(s) disagree with finite difference \
         (a backend that decomposes ActivationBackward would train on wrong \
         gradients): {failures:?}"
    );
}
