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

//! Import-shaped DiT modulation (torch/ONNX emit LN/RMS + Expand + Mul/Add)
//! must fuse to `AdaLayerNorm` / `GatedResidual` under the Session pipeline.
//!
//! Fixtures mirror post-import IR from FLUX adaLN-Zero and F5-TTS DiT
//! (modulation Linear → Narrow chunks → affine-free norm → Expand modulate →
//! gated residual), without shipping full model weights.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{AdaNormKind, BinaryOp, Op};
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_opt::pass::Pass as _;
use rlx_opt::{FuseAdaLayerNorm, FuseGatedResidual};
use rlx_runtime::{Device, Session};

const B: usize = 2;
const S: usize = 4;
const D: usize = 8;
const EPS: f32 = 1e-5;

/// Minimal FLUX / F5-style adaLN: affine-free LN then `n·(1+expand(scale))+expand(shift)`.
fn import_shaped_ada_ln() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("import_ada");
    let x = g.input("x", Shape::new(&[B, S, D], f));
    let scale = g.input("scale", Shape::new(&[B, 1, D], f));
    let shift = g.input("shift", Shape::new(&[B, 1, D], f));
    let gamma = g.full(&[D], 1.0, f);
    let beta = g.zeros(&[D], f);
    let n = g.layer_norm(x, gamma, beta, -1, EPS, Shape::new(&[B, S, D], f));
    let scale_e = expand_bsd(&mut g, scale);
    let one = g.full(&[1], 1.0, f);
    let one_plus = g.binary(BinaryOp::Add, one, scale_e, Shape::new(&[B, S, D], f));
    let scaled = g.binary(BinaryOp::Mul, n, one_plus, Shape::new(&[B, S, D], f));
    let shift_e = expand_bsd(&mut g, shift);
    let out = g.binary(BinaryOp::Add, scaled, shift_e, Shape::new(&[B, S, D], f));
    g.set_outputs(vec![out]);
    g
}

/// Imported gated residual: `x + expand(gate)·y`.
fn import_shaped_gated() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("import_gate");
    let x = g.input("x", Shape::new(&[B, S, D], f));
    let y = g.input("y", Shape::new(&[B, S, D], f));
    let gate = g.input("gate", Shape::new(&[B, 1, D], f));
    let gate_e = expand_bsd(&mut g, gate);
    let gy = g.binary(BinaryOp::Mul, gate_e, y, Shape::new(&[B, S, D], f));
    let out = g.binary(BinaryOp::Add, x, gy, Shape::new(&[B, S, D], f));
    g.set_outputs(vec![out]);
    g
}

fn expand_bsd(g: &mut Graph, v: NodeId) -> NodeId {
    g.add_node(
        Op::Expand {
            target_shape: vec![B as i64, S as i64, D as i64],
        },
        vec![v],
        Shape::new(&[B, S, D], DType::F32),
    )
}

/// FLUX-style adaLN-Zero fragment after ONNX/torch import:
/// `Linear(c) → [B,3D]` → Narrow shift/scale/gate → reshape `[B,1,D]` →
/// affine-free LN → `n·(1+Expand(scale))+Expand(shift)` → gated residual.
fn flux_adaln_zero_block() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("flux_adaln_zero");
    let x = g.input("x", Shape::new(&[B, S, D], f));
    let c = g.input("c", Shape::new(&[B, D], f));
    let y = g.input("y", Shape::new(&[B, S, D], f));
    let w = g.param("mod_w", Shape::new(&[D, 3 * D], f));
    let bias = g.param("mod_b", Shape::new(&[3 * D], f));
    let proj = g.matmul(c, w, Shape::new(&[B, 3 * D], f));
    let mods = g.binary(BinaryOp::Add, proj, bias, Shape::new(&[B, 3 * D], f));

    let shift_flat = g.narrow_(mods, 1, 0, D);
    let scale_flat = g.narrow_(mods, 1, D, D);
    let gate_flat = g.narrow_(mods, 1, 2 * D, D);
    let shift = g.reshape_(shift_flat, vec![B as i64, 1, D as i64]);
    let scale = g.reshape_(scale_flat, vec![B as i64, 1, D as i64]);
    let gate = g.reshape_(gate_flat, vec![B as i64, 1, D as i64]);

    let gamma = g.full(&[D], 1.0, f);
    let beta = g.zeros(&[D], f);
    let n = g.layer_norm(x, gamma, beta, -1, EPS, Shape::new(&[B, S, D], f));
    let scale_e = expand_bsd(&mut g, scale);
    let one = g.full(&[1], 1.0, f);
    let one_plus = g.binary(BinaryOp::Add, one, scale_e, Shape::new(&[B, S, D], f));
    let scaled = g.binary(BinaryOp::Mul, n, one_plus, Shape::new(&[B, S, D], f));
    let shift_e = expand_bsd(&mut g, shift);
    let h = g.binary(BinaryOp::Add, scaled, shift_e, Shape::new(&[B, S, D], f));

    // FLUX: gate the post-attn tensor `y`, residual on pre-norm `x`; keep `h`
    // live so AdaLN is not DCE'd (tiny mix of gated residual + modulated branch).
    let gate_e = expand_bsd(&mut g, gate);
    let gy = g.binary(BinaryOp::Mul, gate_e, y, Shape::new(&[B, S, D], f));
    let gated = g.binary(BinaryOp::Add, x, gy, Shape::new(&[B, S, D], f));
    let out = g.binary(BinaryOp::Add, gated, h, Shape::new(&[B, S, D], f));
    g.set_outputs(vec![out]);
    g
}

/// F5-TTS DiT identity-form adaLN after import:
/// `n + n·Expand(scale) + Expand(shift)` with affine-free LayerNorm
/// (PyTorch `elementwise_affine=False` export often lowers this way).
fn f5_identity_form_adaln() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("f5_ada_identity");
    let x = g.input("x", Shape::new(&[B, S, D], f));
    let scale = g.input("scale", Shape::new(&[B, 1, D], f));
    let shift = g.input("shift", Shape::new(&[B, 1, D], f));
    let gamma = g.full(&[D], 1.0, f);
    let beta = g.zeros(&[D], f);
    let n = g.layer_norm(x, gamma, beta, -1, EPS, Shape::new(&[B, S, D], f));
    let scale_e = expand_bsd(&mut g, scale);
    let ns = g.binary(BinaryOp::Mul, n, scale_e, Shape::new(&[B, S, D], f));
    let m = g.binary(BinaryOp::Add, n, ns, Shape::new(&[B, S, D], f));
    let shift_e = expand_bsd(&mut g, shift);
    let out = g.binary(BinaryOp::Add, m, shift_e, Shape::new(&[B, S, D], f));
    g.set_outputs(vec![out]);
    g
}

/// F5 ONNX export form: `LN * (1+scale) + shift` with **Reshape** (not Expand)
/// for the `[B,D]→[B,1,D]` broadcast — matches DakeQQ F5_Transformer.onnx.
fn f5_reshape_one_plus_adaln() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("f5_ada_reshape");
    let x = g.input("x", Shape::new(&[B, S, D], f));
    let scale = g.input("scale", Shape::new(&[B, D], f));
    let shift = g.input("shift", Shape::new(&[B, D], f));
    let gamma = g.full(&[D], 1.0, f);
    let beta = g.zeros(&[D], f);
    let n = g.layer_norm(x, gamma, beta, -1, EPS, Shape::new(&[B, S, D], f));
    let scale_r = g.reshape(
        scale,
        vec![B as i64, 1, D as i64],
        Shape::new(&[B, 1, D], f),
    );
    let shift_r = g.reshape(
        shift,
        vec![B as i64, 1, D as i64],
        Shape::new(&[B, 1, D], f),
    );
    let one = g.full(&[1], 1.0, f);
    let one_plus = g.binary(BinaryOp::Add, scale_r, one, Shape::new(&[B, 1, D], f));
    let scaled = g.binary(BinaryOp::Mul, n, one_plus, Shape::new(&[B, S, D], f));
    let out = g.binary(BinaryOp::Add, scaled, shift_r, Shape::new(&[B, S, D], f));
    g.set_outputs(vec![out]);
    g
}

/// F5-style RMS adaLN (`1+scale` form) — some DiT variants use RMSNorm.
fn f5_rms_adaln() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("f5_rms_ada");
    let x = g.input("x", Shape::new(&[B, S, D], f));
    let scale = g.input("scale", Shape::new(&[B, 1, D], f));
    let shift = g.input("shift", Shape::new(&[B, 1, D], f));
    let gamma = g.full(&[D], 1.0, f);
    let beta = g.zeros(&[D], f);
    let n = g.rms_norm(x, gamma, beta, EPS);
    let scale_e = expand_bsd(&mut g, scale);
    let one = g.full(&[1], 1.0, f);
    let one_plus = g.binary(BinaryOp::Add, one, scale_e, Shape::new(&[B, S, D], f));
    let scaled = g.binary(BinaryOp::Mul, n, one_plus, Shape::new(&[B, S, D], f));
    let shift_e = expand_bsd(&mut g, shift);
    let out = g.binary(BinaryOp::Add, scaled, shift_e, Shape::new(&[B, S, D], f));
    g.set_outputs(vec![out]);
    g
}

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| seed * (0.11 * (i as f32) - 0.07 * ((i % 3) as f32)))
        .collect()
}

#[test]
fn fuse_passes_match_import_shaped_adaln_and_gate() {
    let fused = FuseAdaLayerNorm.run(import_shaped_ada_ln());
    assert!(
        fused.nodes().iter().any(|n| matches!(
            n.op,
            Op::AdaLayerNorm {
                norm: AdaNormKind::LayerNorm,
                ..
            }
        )),
        "expected FuseAdaLayerNorm on import-shaped adaLN"
    );

    let fused = FuseGatedResidual.run(import_shaped_gated());
    assert!(
        fused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GatedResidual)),
        "expected FuseGatedResidual on import-shaped gate"
    );
}

#[test]
fn session_compile_fuses_import_shaped_adaln() {
    let g = import_shaped_ada_ln();
    let x = fill(B * S * D, 1.0);
    let scale = fill(B * D, 0.2);
    let shift = fill(B * D, -0.1);
    // Default Session pipeline runs FuseAdaLayerNorm for CPU.
    let mut c = Session::new(Device::Cpu).compile(g);
    let out = c.run(&[("x", &x), ("scale", &scale), ("shift", &shift)])[0].clone();
    assert_eq!(out.len(), B * S * D);
    assert!(out.iter().all(|v| v.is_finite()));

    // Same inputs through an already-fused graph must match.
    let fused = FuseAdaLayerNorm.run(import_shaped_ada_ln());
    let mut c2 = Session::new(Device::Cpu).compile(fused);
    let out2 = c2.run(&[("x", &x), ("scale", &scale), ("shift", &shift)])[0].clone();
    for i in 0..out.len() {
        let err = (out[i] - out2[i]).abs();
        assert!(err < 1e-5, "[{i}] session {} vs fused {}", out[i], out2[i]);
    }
}

#[test]
fn flux_adaln_zero_block_fuses_ada_and_gate() {
    let g = flux_adaln_zero_block();
    let fused_ada = FuseAdaLayerNorm.run(g.clone());
    assert!(
        fused_ada.nodes().iter().any(|n| matches!(
            n.op,
            Op::AdaLayerNorm {
                norm: AdaNormKind::LayerNorm,
                ..
            }
        )),
        "FLUX fixture: expected AdaLayerNorm after FuseAdaLayerNorm"
    );
    let fused = FuseGatedResidual.run(fused_ada);
    assert!(
        fused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GatedResidual)),
        "FLUX fixture: expected GatedResidual after FuseGatedResidual"
    );

    let x = fill(B * S * D, 1.0);
    let c = fill(B * D, 0.3);
    let y = fill(B * S * D, -0.5);
    let w = fill(D * 3 * D, 0.02);
    let b = fill(3 * D, -0.01);
    let mut sess = Session::new(Device::Cpu).compile(g);
    sess.set_param("mod_w", &w);
    sess.set_param("mod_b", &b);
    let out = sess.run(&[("x", &x), ("c", &c), ("y", &y)])[0].clone();
    assert_eq!(out.len(), B * S * D);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn f5_identity_and_rms_forms_fuse() {
    let fused = FuseAdaLayerNorm.run(f5_identity_form_adaln());
    assert!(
        fused.nodes().iter().any(|n| matches!(
            n.op,
            Op::AdaLayerNorm {
                norm: AdaNormKind::LayerNorm,
                ..
            }
        )),
        "F5 identity form should fuse to AdaLayerNorm"
    );

    let fused = FuseAdaLayerNorm.run(f5_rms_adaln());
    assert!(
        fused.nodes().iter().any(|n| matches!(
            n.op,
            Op::AdaLayerNorm {
                norm: AdaNormKind::RmsNorm,
                ..
            }
        )),
        "F5 RMS form should fuse to AdaLayerNorm(RmsNorm)"
    );

    let fused = FuseAdaLayerNorm.run(f5_reshape_one_plus_adaln());
    assert!(
        fused.nodes().iter().any(|n| matches!(
            n.op,
            Op::AdaLayerNorm {
                norm: AdaNormKind::LayerNorm,
                ..
            }
        )),
        "F5 Reshape+(1+scale) form should fuse to AdaLayerNorm"
    );

    let x = fill(B * S * D, 0.9);
    let scale = fill(B * D, 0.15);
    let shift = fill(B * D, -0.08);
    let mut c = Session::new(Device::Cpu).compile(f5_identity_form_adaln());
    let out = c.run(&[("x", &x), ("scale", &scale), ("shift", &shift)])[0].clone();
    let mut c2 = Session::new(Device::Cpu).compile(FuseAdaLayerNorm.run(f5_identity_form_adaln()));
    let out2 = c2.run(&[("x", &x), ("scale", &scale), ("shift", &shift)])[0].clone();
    for i in 0..out.len() {
        let err = (out[i] - out2[i]).abs();
        assert!(
            err < 1e-5,
            "[{i}] F5 identity session {} vs fused {}",
            out[i],
            out2[i]
        );
    }
}
