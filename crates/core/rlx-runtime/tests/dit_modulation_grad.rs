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

//! CPU finite-difference checks for DiT modulation autodiff.
//!
//! Default path: fused VJP → packed `AdaLayerNormBackward` /
//! `GatedResidualBackward` (no Expand of `[B,1,D]` modulation).
//! Option-1 path: `unfuse_dit_modulation` then primitive VJPs.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::AdaNormKind;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::FusionOptions;
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{CompileOptions, Device, Session};

const B: usize = 2;
const S: usize = 4;
const D: usize = 6;
const EPS: f32 = 1e-5;
const H: f32 = 1e-3;
const TOL: f32 = 5e-2;

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            seed * (0.17 * t - 0.31 * ((i % 5) as f32) + 0.05 * ((i % 7) as f32))
        })
        .collect()
}

fn no_fusion_opts() -> CompileOptions {
    CompileOptions::new().fusion_opts(FusionOptions {
        skip_fusion: true,
        ..Default::default()
    })
}

fn run_scalar(g: Graph, feeds: &[(&str, &[f32])]) -> f32 {
    let mut c = Session::new(Device::Cpu).compile_with(g, &no_fusion_opts());
    c.run(feeds)[0][0]
}

fn assert_grad_matches_fd(
    name: &str,
    analytic: &[f32],
    base: &[f32],
    mut loss_at: impl FnMut(&[f32]) -> f32,
) {
    assert_eq!(analytic.len(), base.len(), "{name}: grad length");
    for i in 0..base.len() {
        let mut plus = base.to_vec();
        let mut minus = base.to_vec();
        plus[i] += H;
        minus[i] -= H;
        let fd = (loss_at(&plus) - loss_at(&minus)) / (2.0 * H);
        let abs_err = (analytic[i] - fd).abs();
        let rel_err = abs_err / fd.abs().max(1e-6);
        assert!(
            abs_err < TOL || rel_err < TOL,
            "{name}[{i}]: autodiff {} vs FD {fd} (abs {abs_err:e}, rel {rel_err:e})",
            analytic[i],
        );
    }
}

fn ada_loss_graph(norm: AdaNormKind) -> (Graph, NodeId, NodeId, NodeId) {
    let f = DType::F32;
    let mut g = Graph::new("ada_loss");
    let x = g.input("x", Shape::new(&[B, S, D], f));
    let scale = g.input("scale", Shape::new(&[B, 1, D], f));
    let shift = g.input("shift", Shape::new(&[B, 1, D], f));
    let out = g.ada_layer_norm(x, scale, shift, norm, EPS);
    let loss = g.sum(out, vec![0, 1, 2], false);
    g.set_outputs(vec![loss]);
    (g, x, scale, shift)
}

fn gate_loss_graph() -> (Graph, NodeId, NodeId, NodeId) {
    let f = DType::F32;
    let mut g = Graph::new("gate_loss");
    let x = g.input("x", Shape::new(&[B, S, D], f));
    let y = g.input("y", Shape::new(&[B, S, D], f));
    let gate = g.input("gate", Shape::new(&[B, 1, D], f));
    let out = g.gated_residual(x, y, gate);
    let loss = g.sum(out, vec![0, 1, 2], false);
    g.set_outputs(vec![loss]);
    (g, x, y, gate)
}

fn check_ada(norm: AdaNormKind, label: &str, unfuse_first: bool) {
    let x0 = fill(B * S * D, 1.1);
    let scale0 = fill(B * D, 0.7);
    let shift0 = fill(B * D, -0.4);

    let (g, x_n, scale_n, shift_n) = ada_loss_graph(norm);
    let g = if unfuse_first {
        rlx_fusion::unfuse_dit_modulation(g)
    } else {
        g
    };
    // After unfuse, node ids for inputs are remapped — rebuild wrt from names.
    let (g, x_n, scale_n, shift_n) = if unfuse_first {
        let x = g
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name } if name == "x"))
            .map(|n| n.id)
            .expect("x input");
        let scale = g
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name } if name == "scale"))
            .map(|n| n.id)
            .expect("scale input");
        let shift = g
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name } if name == "shift"))
            .map(|n| n.id)
            .expect("shift input");
        (g, x, scale, shift)
    } else {
        (g, x_n, scale_n, shift_n)
    };

    let bwd = grad_with_loss(&g, &[x_n, scale_n, shift_n]);
    if !unfuse_first {
        assert!(
            bwd.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::AdaLayerNormBackward { .. })),
            "{label}: expected fused AdaLayerNormBackward in VJP graph"
        );
    }
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let d_output = [1.0f32];
    let outs = c.run(&[
        ("x", &x0),
        ("scale", &scale0),
        ("shift", &shift0),
        ("d_output", &d_output),
    ]);
    let (dx, dscale, dshift) = (&outs[1], &outs[2], &outs[3]);

    let loss_x = |xv: &[f32]| {
        let (g, _, _, _) = ada_loss_graph(norm);
        run_scalar(g, &[("x", xv), ("scale", &scale0), ("shift", &shift0)])
    };
    let loss_scale = |sv: &[f32]| {
        let (g, _, _, _) = ada_loss_graph(norm);
        run_scalar(g, &[("x", &x0), ("scale", sv), ("shift", &shift0)])
    };
    let loss_shift = |tv: &[f32]| {
        let (g, _, _, _) = ada_loss_graph(norm);
        run_scalar(g, &[("x", &x0), ("scale", &scale0), ("shift", tv)])
    };

    assert_grad_matches_fd(&format!("{label}/dx"), dx, &x0, loss_x);
    assert_grad_matches_fd(&format!("{label}/dscale"), dscale, &scale0, loss_scale);
    assert_grad_matches_fd(&format!("{label}/dshift"), dshift, &shift0, loss_shift);
}

#[test]
fn ada_layer_norm_grad_matches_fd_fused() {
    check_ada(AdaNormKind::LayerNorm, "AdaLayerNorm(LN)/fused", false);
}

#[test]
fn ada_rms_norm_grad_matches_fd_fused() {
    check_ada(AdaNormKind::RmsNorm, "AdaLayerNorm(RMS)/fused", false);
}

#[test]
fn ada_layer_norm_grad_matches_fd_unfused() {
    check_ada(AdaNormKind::LayerNorm, "AdaLayerNorm(LN)/unfuse", true);
}

#[test]
fn gated_residual_grad_matches_fd_fused() {
    let x0 = fill(B * S * D, 0.9);
    let y0 = fill(B * S * D, -0.6);
    let gate0 = fill(B * D, 0.35);

    let (g, x_n, y_n, gate_n) = gate_loss_graph();
    let bwd = grad_with_loss(&g, &[x_n, y_n, gate_n]);
    assert!(
        bwd.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GatedResidualBackward)),
        "expected fused GatedResidualBackward in VJP graph"
    );
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let d_output = [1.0f32];
    let outs = c.run(&[
        ("x", &x0),
        ("y", &y0),
        ("gate", &gate0),
        ("d_output", &d_output),
    ]);
    let (dx, dy, dgate) = (&outs[1], &outs[2], &outs[3]);

    let loss_x = |xv: &[f32]| {
        let (g, _, _, _) = gate_loss_graph();
        run_scalar(g, &[("x", xv), ("y", &y0), ("gate", &gate0)])
    };
    let loss_y = |yv: &[f32]| {
        let (g, _, _, _) = gate_loss_graph();
        run_scalar(g, &[("x", &x0), ("y", yv), ("gate", &gate0)])
    };
    let loss_gate = |gv: &[f32]| {
        let (g, _, _, _) = gate_loss_graph();
        run_scalar(g, &[("x", &x0), ("y", &y0), ("gate", gv)])
    };

    assert_grad_matches_fd("GatedResidual/dx", dx, &x0, loss_x);
    assert_grad_matches_fd("GatedResidual/dy", dy, &y0, loss_y);
    assert_grad_matches_fd("GatedResidual/dgate", dgate, &gate0, loss_gate);
}

#[test]
fn gated_residual_grad_matches_fd_unfused() {
    let x0 = fill(B * S * D, 0.9);
    let y0 = fill(B * S * D, -0.6);
    let gate0 = fill(B * D, 0.35);

    let (g, _, _, _) = gate_loss_graph();
    let g = rlx_fusion::unfuse_dit_modulation(g);
    let x_n = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "x"))
        .map(|n| n.id)
        .unwrap();
    let y_n = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "y"))
        .map(|n| n.id)
        .unwrap();
    let gate_n = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "gate"))
        .map(|n| n.id)
        .unwrap();
    let bwd = grad_with_loss(&g, &[x_n, y_n, gate_n]);
    assert!(
        !bwd.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GatedResidualBackward)),
        "unfuse path should not emit GatedResidualBackward"
    );
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let d_output = [1.0f32];
    let outs = c.run(&[
        ("x", &x0),
        ("y", &y0),
        ("gate", &gate0),
        ("d_output", &d_output),
    ]);
    let (dx, dy, dgate) = (&outs[1], &outs[2], &outs[3]);

    let loss_x = |xv: &[f32]| {
        let (g, _, _, _) = gate_loss_graph();
        run_scalar(g, &[("x", xv), ("y", &y0), ("gate", &gate0)])
    };
    let loss_y = |yv: &[f32]| {
        let (g, _, _, _) = gate_loss_graph();
        run_scalar(g, &[("x", &x0), ("y", yv), ("gate", &gate0)])
    };
    let loss_gate = |gv: &[f32]| {
        let (g, _, _, _) = gate_loss_graph();
        run_scalar(g, &[("x", &x0), ("y", &y0), ("gate", gv)])
    };

    assert_grad_matches_fd("GatedResidual/unfuse/dx", dx, &x0, loss_x);
    assert_grad_matches_fd("GatedResidual/unfuse/dy", dy, &y0, loss_y);
    assert_grad_matches_fd("GatedResidual/unfuse/dgate", dgate, &gate0, loss_gate);
}

#[test]
fn ada_layer_norm_decompose_backward_matches_fd() {
    let x0 = fill(B * S * D, 1.1);
    let scale0 = fill(B * D, 0.7);
    let shift0 = fill(B * D, -0.4);

    let (g, x_n, scale_n, shift_n) = ada_loss_graph(AdaNormKind::LayerNorm);
    let bwd = grad_with_loss(&g, &[x_n, scale_n, shift_n]);
    let bwd = rlx_opt::rlx_autodiff::decompose_backward_ops(bwd);
    assert!(
        !bwd.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::AdaLayerNormBackward { .. })),
        "decompose should expand AdaLayerNormBackward"
    );
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let d_output = [1.0f32];
    let outs = c.run(&[
        ("x", &x0),
        ("scale", &scale0),
        ("shift", &shift0),
        ("d_output", &d_output),
    ]);
    let (dx, dscale, dshift) = (&outs[1], &outs[2], &outs[3]);

    let loss_x = |xv: &[f32]| {
        let (g, _, _, _) = ada_loss_graph(AdaNormKind::LayerNorm);
        run_scalar(g, &[("x", xv), ("scale", &scale0), ("shift", &shift0)])
    };
    let loss_scale = |sv: &[f32]| {
        let (g, _, _, _) = ada_loss_graph(AdaNormKind::LayerNorm);
        run_scalar(g, &[("x", &x0), ("scale", sv), ("shift", &shift0)])
    };
    let loss_shift = |tv: &[f32]| {
        let (g, _, _, _) = ada_loss_graph(AdaNormKind::LayerNorm);
        run_scalar(g, &[("x", &x0), ("scale", &scale0), ("shift", tv)])
    };

    assert_grad_matches_fd("Ada/decompose/dx", dx, &x0, loss_x);
    assert_grad_matches_fd("Ada/decompose/dscale", dscale, &scale0, loss_scale);
    assert_grad_matches_fd("Ada/decompose/dshift", dshift, &shift0, loss_shift);
}

#[test]
fn gated_residual_decompose_backward_matches_fd() {
    let x0 = fill(B * S * D, 0.9);
    let y0 = fill(B * S * D, -0.6);
    let gate0 = fill(B * D, 0.35);

    let (g, x_n, y_n, gate_n) = gate_loss_graph();
    let bwd = grad_with_loss(&g, &[x_n, y_n, gate_n]);
    let bwd = rlx_opt::rlx_autodiff::decompose_backward_ops(bwd);
    assert!(
        !bwd.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GatedResidualBackward)),
        "decompose should expand GatedResidualBackward"
    );
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let d_output = [1.0f32];
    let outs = c.run(&[
        ("x", &x0),
        ("y", &y0),
        ("gate", &gate0),
        ("d_output", &d_output),
    ]);
    let (dx, dy, dgate) = (&outs[1], &outs[2], &outs[3]);

    let loss_x = |xv: &[f32]| {
        let (g, _, _, _) = gate_loss_graph();
        run_scalar(g, &[("x", xv), ("y", &y0), ("gate", &gate0)])
    };
    let loss_y = |yv: &[f32]| {
        let (g, _, _, _) = gate_loss_graph();
        run_scalar(g, &[("x", &x0), ("y", yv), ("gate", &gate0)])
    };
    let loss_gate = |gv: &[f32]| {
        let (g, _, _, _) = gate_loss_graph();
        run_scalar(g, &[("x", &x0), ("y", &y0), ("gate", gv)])
    };

    assert_grad_matches_fd("Gated/decompose/dx", dx, &x0, loss_x);
    assert_grad_matches_fd("Gated/decompose/dy", dy, &y0, loss_y);
    assert_grad_matches_fd("Gated/decompose/dgate", dgate, &gate0, loss_gate);
}

#[test]
fn gated_residual_jvp_matches_fd() {
    use rlx_opt::autodiff_fwd::jvp;

    let x0 = fill(B * S * D, 0.9);
    let y0 = fill(B * S * D, -0.6);
    let gate0 = fill(B * D, 0.35);
    // Directional derivative wrt gate only (cheap FD).
    let tg = fill(B * D, 0.02);

    let (g, _, _, gate_n) = gate_loss_graph();
    // Scalar loss JVP: jvp of sum(gated) wrt gate.
    let jg = jvp(&g, &[gate_n]);
    let mut c = Session::new(Device::Cpu).compile(jg);
    let outs = c.run(&[
        ("x", &x0),
        ("y", &y0),
        ("gate", &gate0),
        ("tangent_gate", &tg),
    ]);
    let primal = outs[0][0];
    let tangent = outs[1][0];

    let loss_at = |gate: &[f32]| {
        let (g, _, _, _) = gate_loss_graph();
        run_scalar(g, &[("x", &x0), ("y", &y0), ("gate", gate)])
    };
    let mut plus = gate0.clone();
    let mut minus = gate0.clone();
    for i in 0..tg.len() {
        plus[i] += H * tg[i];
        minus[i] -= H * tg[i];
    }
    let fd = (loss_at(&plus) - loss_at(&minus)) / (2.0 * H);
    assert!(
        (primal - loss_at(&gate0)).abs() < 1e-5,
        "primal mismatch {primal} vs {}",
        loss_at(&gate0)
    );
    let abs_err = (tangent - fd).abs();
    let rel_err = abs_err / fd.abs().max(1e-6);
    assert!(
        abs_err < TOL || rel_err < TOL,
        "Gated JVP tangent {tangent} vs FD {fd} (abs {abs_err:e}, rel {rel_err:e})"
    );
}

#[test]
fn ada_layer_norm_jvp_matches_fd() {
    use rlx_opt::autodiff_fwd::jvp;

    let x0 = fill(B * S * D, 1.1);
    let scale0 = fill(B * D, 0.7);
    let shift0 = fill(B * D, -0.4);

    // Scale tangent (exact affine path).
    {
        let ts = fill(B * D, 0.02);
        let (g, _, scale_n, _) = ada_loss_graph(AdaNormKind::LayerNorm);
        let jg = jvp(&g, &[scale_n]);
        let mut c = Session::new(Device::Cpu).compile(jg);
        let outs = c.run(&[
            ("x", &x0),
            ("scale", &scale0),
            ("shift", &shift0),
            ("tangent_scale", &ts),
        ]);
        let primal = outs[0][0];
        let tangent = outs[1][0];
        let loss_at = |scale: &[f32]| {
            let (g, _, _, _) = ada_loss_graph(AdaNormKind::LayerNorm);
            run_scalar(g, &[("x", &x0), ("scale", scale), ("shift", &shift0)])
        };
        let mut plus = scale0.clone();
        let mut minus = scale0.clone();
        for i in 0..ts.len() {
            plus[i] += H * ts[i];
            minus[i] -= H * ts[i];
        }
        let fd = (loss_at(&plus) - loss_at(&minus)) / (2.0 * H);
        assert!((primal - loss_at(&scale0)).abs() < 1e-5);
        let abs_err = (tangent - fd).abs();
        let rel_err = abs_err / fd.abs().max(1e-6);
        assert!(
            abs_err < TOL || rel_err < TOL,
            "Ada JVP(scale) tangent {tangent} vs FD {fd} (abs {abs_err:e}, rel {rel_err:e})"
        );
    }

    // x tangent — full LN pushforward (mean/var derivatives).
    {
        let tx = fill(B * S * D, 0.015);
        let (g, x_n, _, _) = ada_loss_graph(AdaNormKind::LayerNorm);
        let jg = jvp(&g, &[x_n]);
        let mut c = Session::new(Device::Cpu).compile(jg);
        let outs = c.run(&[
            ("x", &x0),
            ("scale", &scale0),
            ("shift", &shift0),
            ("tangent_x", &tx),
        ]);
        let primal = outs[0][0];
        let tangent = outs[1][0];
        let loss_at = |x: &[f32]| {
            let (g, _, _, _) = ada_loss_graph(AdaNormKind::LayerNorm);
            run_scalar(g, &[("x", x), ("scale", &scale0), ("shift", &shift0)])
        };
        let mut plus = x0.clone();
        let mut minus = x0.clone();
        for i in 0..tx.len() {
            plus[i] += H * tx[i];
            minus[i] -= H * tx[i];
        }
        let fd = (loss_at(&plus) - loss_at(&minus)) / (2.0 * H);
        assert!((primal - loss_at(&x0)).abs() < 1e-5);
        let abs_err = (tangent - fd).abs();
        let rel_err = abs_err / fd.abs().max(1e-6);
        assert!(
            abs_err < TOL || rel_err < TOL,
            "Ada JVP(x) tangent {tangent} vs FD {fd} (abs {abs_err:e}, rel {rel_err:e})"
        );
    }
}

#[test]
fn dit_modulation_vmap_batches_correctly() {
    use rlx_opt::vmap;

    // Per-example shapes [S,D]; vmap over leading batch.
    let f = DType::F32;
    let mut g = Graph::new("dit_vmap");
    let x = g.input("x", Shape::new(&[S, D], f));
    let scale = g.input("scale", Shape::new(&[1, D], f));
    let shift = g.input("shift", Shape::new(&[1, D], f));
    let y = g.input("y", Shape::new(&[S, D], f));
    let gate = g.input("gate", Shape::new(&[1, D], f));
    let n = g.ada_layer_norm(x, scale, shift, AdaNormKind::LayerNorm, EPS);
    let out = g.gated_residual(n, y, gate);
    g.set_outputs(vec![out]);

    let vg = vmap(&g, &["x", "scale", "shift", "y", "gate"], B);
    assert!(
        !vg.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::AdaLayerNorm { .. } | Op::GatedResidual)),
        "vmap should unfuse DiT modulation before lifting"
    );

    let x0 = fill(B * S * D, 1.1);
    let scale0 = fill(B * D, 0.7);
    let shift0 = fill(B * D, -0.4);
    let y0 = fill(B * S * D, -0.6);
    let gate0 = fill(B * D, 0.35);

    let mut c = Session::new(Device::Cpu).compile_with(vg, &no_fusion_opts());
    let batched = c.run(&[
        ("x", &x0),
        ("scale", &scale0),
        ("shift", &shift0),
        ("y", &y0),
        ("gate", &gate0),
    ])[0]
        .clone();

    // Reference: run each batch slice through the unbatched graph.
    let mut c_ref = Session::new(Device::Cpu).compile_with(g, &no_fusion_opts());
    for b in 0..B {
        let xs = &x0[b * S * D..(b + 1) * S * D];
        let sc = &scale0[b * D..(b + 1) * D];
        let sh = &shift0[b * D..(b + 1) * D];
        let ys = &y0[b * S * D..(b + 1) * S * D];
        let gt = &gate0[b * D..(b + 1) * D];
        let got = &batched[b * S * D..(b + 1) * S * D];
        let want = c_ref.run(&[
            ("x", xs),
            ("scale", sc),
            ("shift", sh),
            ("y", ys),
            ("gate", gt),
        ])[0]
            .clone();
        for i in 0..want.len() {
            let err = (got[i] - want[i]).abs();
            assert!(
                err < 1e-4,
                "vmap[{b}][{i}]: got {} want {} (err {err:e})",
                got[i],
                want[i]
            );
        }
    }
}
