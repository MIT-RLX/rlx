// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Drives the CoreML backend through the public runtime Session/registry
// API — i.e. exactly how an application selects `Device::Ane`.
#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn ane_device_is_available_and_runs() {
    // Availability through the runtime's own probe.
    assert!(
        rlx_runtime::is_available(Device::Ane),
        "Device::Ane should be available with the coreml feature on Apple"
    );

    // relu(x) end-to-end via Session.
    let mut g = Graph::new("session_relu");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.activation(Activation::Relu, x, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Ane).compile(g);
    let out = compiled.run(&[("x", &[-1.0f32, 2.0, -3.0, 4.0])]).remove(0);
    assert_eq!(out, vec![0.0, 2.0, 0.0, 4.0]);
}

#[test]
fn ane_op_support_introspection() {
    use rlx_ir::Op;

    // The runtime's per-op support probe now knows the CoreML op claim.
    assert!(rlx_runtime::supports(Device::Ane, &Op::MatMul));
    assert!(rlx_runtime::supports(
        Device::Ane,
        &Op::Softmax { axis: -1 }
    ));
    assert!(rlx_runtime::supports(
        Device::Ane,
        &Op::Attention {
            num_heads: 1,
            head_dim: 8,
            v_head_dim: None,
            mask_kind: rlx_ir::op::MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        }
    ));
    // With the `training` feature on, ANE participates in backward: the runtime
    // reports `*Backward` ops as supported (they decompose to — or, for the hot
    // ops, lower through native MIL kernels of — the supported primitive set), so
    // device selection picks Ane for autodiff-produced backward graphs.
    assert!(rlx_runtime::supports(Device::Ane, &Op::ReluBackward));
    assert!(rlx_runtime::supports(
        Device::Ane,
        &Op::RmsNormBackwardInput {
            axis: -1,
            eps: 1e-6
        }
    ));

    // A graph of supported ops is dispatchable; first_unsupported_op finds
    // the gap when one isn't.
    let mut g = Graph::new("ok");
    let x = g.input("x", Shape::new(&[2, 2], DType::F32));
    let y = g.activation(Activation::Relu, x, Shape::new(&[2, 2], DType::F32));
    g.set_outputs(vec![y]);
    assert!(rlx_runtime::supports_graph(Device::Ane, &g));
    assert!(rlx_runtime::first_unsupported_op(Device::Ane, &g).is_none());
}

#[test]
fn ane_matmul_with_param_via_session() {
    // y = x @ W with a baked parameter, driven through Session::set_param.
    let mut g = Graph::new("session_matmul");
    let x = g.input("x", Shape::new(&[1, 3], DType::F32));
    let w = g.param("W", Shape::new(&[3, 2], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[1, 2], DType::F32));
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Ane).compile(g);
    compiled.set_param("W", &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let out = compiled.run(&[("x", &[2.0f32, 3.0, 4.0])]).remove(0);
    // [2,3,4] @ [[1,0],[0,1],[1,1]] = [2+4, 3+4] = [6, 7]
    assert_eq!(out, vec![6.0, 7.0]);
}
