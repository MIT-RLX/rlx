// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Straight-through QAT autodiff for the native low-precision GEMM.
//! `Op::ScaledMatMul`'s VJP rebuilds the (quantized) operands and runs the
//! ordinary matmul backward, routing the gradient through `ScaledQuantize`'s
//! identity STE to the original f32 operands. So d(sum(lhs·rhsᵀ)) must track the
//! plain f32 matmul gradient (within fp8 reconstruction error).

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, Op, ScaleLayout, ScaledFormat, Shape};

#[test]
fn scaled_matmul_ste_grad_tracks_f32() {
    let (m, k, n) = (3usize, 16usize, 4usize);
    let fmt = ScaledFormat::F8E4M3;
    let layout = ScaleLayout::PerTensor;

    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.21).sin()).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.11).cos() * 0.8).collect();

    let mut g = Graph::new("scaled_grad");
    let lhs_p = g.param("lhs", Shape::new(&[m, k], DType::F32));
    let rhs_p = g.param("rhs", Shape::new(&[n, k], DType::F32));
    let ls = g.add_node(
        Op::ScaledQuantScale {
            format: fmt,
            scale_layout: layout,
        },
        vec![lhs_p],
        Shape::new(&[1], DType::F32),
    );
    let lq = g.add_node(
        Op::ScaledQuantize {
            format: fmt,
            scale_layout: layout,
        },
        vec![lhs_p, ls],
        Shape::new(&[m, k], DType::U8),
    );
    let rs = g.add_node(
        Op::ScaledQuantScale {
            format: fmt,
            scale_layout: layout,
        },
        vec![rhs_p],
        Shape::new(&[1], DType::F32),
    );
    let rq = g.add_node(
        Op::ScaledQuantize {
            format: fmt,
            scale_layout: layout,
        },
        vec![rhs_p, rs],
        Shape::new(&[n, k], DType::U8),
    );
    let out = g.add_node(
        Op::ScaledMatMul {
            lhs_format: fmt,
            rhs_format: fmt,
            scale_layout: layout,
            has_bias: false,
        },
        vec![lq, rq, ls, rs],
        Shape::new(&[m, n], DType::F32),
    );
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0, 1],
            keep_dim: false,
        },
        vec![out],
        Shape::from_dims(&[], DType::F32),
    );
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[lhs_p, rhs_p]);
    let mut c = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    c.set_param("lhs", &lhs);
    c.set_param("rhs", &rhs);
    let outs = c.run(&[("d_output", &[1.0f32])]);
    let d_lhs = &outs[1];
    let d_rhs = &outs[2];
    assert_eq!(d_lhs.len(), m * k);
    assert_eq!(d_rhs.len(), n * k);

    // Analytic f32 gradient of L = Σ_{m,n,p} lhs[m,p]·rhs[n,p]:
    //   dL/dlhs[m,p] = Σ_n rhs[n,p];  dL/drhs[n,p] = Σ_m lhs[m,p].
    let mut ref_lhs = vec![0f32; m * k];
    for mm in 0..m {
        for p in 0..k {
            ref_lhs[mm * k + p] = (0..n).map(|nn| rhs[nn * k + p]).sum();
        }
    }
    let mut ref_rhs = vec![0f32; n * k];
    for nn in 0..n {
        for p in 0..k {
            ref_rhs[nn * k + p] = (0..m).map(|mm| lhs[mm * k + p]).sum();
        }
    }

    let rel = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / (y.abs() + 0.5))
            .fold(0.0f32, f32::max)
    };
    let e_lhs = rel(d_lhs, &ref_lhs);
    let e_rhs = rel(d_rhs, &ref_rhs);
    eprintln!("scaled_matmul STE grad: rel_err lhs={e_lhs:.4} rhs={e_rhs:.4}");
    assert!(e_lhs < 0.1, "d_lhs STE vs f32 rel err {e_lhs}");
    assert!(e_rhs < 0.1, "d_rhs STE vs f32 rel err {e_rhs}");
    assert!(d_lhs.iter().all(|v| v.is_finite()));
    assert!(d_rhs.iter().all(|v| v.is_finite()));
}
