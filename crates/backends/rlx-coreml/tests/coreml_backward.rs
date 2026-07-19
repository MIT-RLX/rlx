// Backward / training on the CoreML (ANE) backend.
//
// Phase 1 validates the *decompose route*: a graph that carries `*Backward` ops
// is (a) claimed by `Device::Ane` for device selection and (b) lowered by
// decomposing each backward op into the supported MIL primitive set — producing
// gradients that match the CPU backend (the oracle). The native MIL backward
// kernels (Phase 2) reuse these same parity checks.
#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_ir::op::{Activation, AdaNormKind, ReduceOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// Relative closeness between a CoreML result and the CPU oracle. Backward /
/// training graphs run fp32 on CPU+GPU (the Neural Engine is fp16 — see
/// `default_compute_units`), so the gradient should match the CPU reference to
/// near fp32 round-off, not the loose fp16 ANE tolerance.
fn assert_close(ane: &[f32], cpu: &[f32], what: &str) {
    assert_eq!(ane.len(), cpu.len(), "{what}: length mismatch");
    assert!(
        ane.iter().all(|v| v.is_finite()),
        "{what}: non-finite ANE output: {ane:?}"
    );
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nc = 0.0f32;
    for (&a, &c) in ane.iter().zip(cpu) {
        max_abs = max_abs.max((a - c).abs() / (1.0 + c.abs()));
        dot += a * c;
        na += a * a;
        nc += c * c;
    }
    let cosine = if na > 0.0 && nc > 0.0 {
        dot / (na.sqrt() * nc.sqrt())
    } else {
        1.0
    };
    assert!(
        max_abs < 1e-3 && cosine > 0.99999,
        "{what}: CoreML vs CPU diverged (max_rel={max_abs}, cosine={cosine})\n  ane={ane:?}\n  cpu={cpu:?}"
    );
}

// ─────────────────────────── capability / selection ───────────────────────────
// These run on any Apple host (no device execution): the dev-dependency builds
// rlx-runtime with the `training` feature, so ANE claims the backward ops.

#[test]
fn ane_claims_backward_ops_for_selection() {
    assert!(rlx_runtime::supports(Device::Ane, &Op::ReluBackward));
    assert!(rlx_runtime::supports(
        Device::Ane,
        &Op::ActivationBackward {
            kind: Activation::Silu
        }
    ));
    assert!(rlx_runtime::supports(
        Device::Ane,
        &Op::RmsNormBackwardInput {
            axis: -1,
            eps: 1e-6
        }
    ));
}

#[test]
fn ane_selects_for_a_differentiated_graph() {
    // forward: loss = sum(silu(x @ W)); backward carries ActivationBackward.
    let (g, w) = silu_matmul_forward();
    let bwd = rlx_opt::grad_with_loss(&g, &[w]);
    // The raw backward graph (pre-decompose) must be claimed by Ane so the
    // runtime's device selection considers it.
    assert!(
        rlx_runtime::supports_graph(Device::Ane, &bwd),
        "Ane should claim the differentiated graph: first gap = {:?}",
        rlx_runtime::first_unsupported_op(Device::Ane, &bwd)
    );
}

// ───────────────────────────── gradient parity ─────────────────────────────
// Apple device execution: ANE-computed gradients ≈ CPU gradients.

/// forward graph `loss = sum(silu(x @ W))`, returning the trainable W node.
fn silu_matmul_forward() -> (Graph, rlx_ir::NodeId) {
    let (b, k, n) = (2usize, 3usize, 4usize);
    let mut g = Graph::new("silu_matmul");
    let x = g.input("x", Shape::new(&[b, k], DType::F32));
    let w = g.param("W", Shape::new(&[k, n], DType::F32));
    let h = g.matmul(x, w, Shape::new(&[b, n], DType::F32));
    let a = g.activation(Activation::Silu, h, Shape::new(&[b, n], DType::F32));
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0, 1],
            keep_dim: false,
        },
        vec![a],
        Shape::from_dims(&[], DType::F32),
    );
    g.set_outputs(vec![loss]);
    (g, w)
}

#[test]
fn ane_silu_matmul_grad_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (g, w) = silu_matmul_forward();
    let bwd = rlx_opt::grad_with_loss(&g, &[w]);

    let x_data: Vec<f32> = (0..2 * 3).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w_init: Vec<f32> = (0..3 * 4).map(|i| (i as f32) * 0.05 - 0.2).collect();

    let run = |device: Device| -> Vec<Vec<f32>> {
        let mut c = Session::new(device).compile(bwd.clone());
        c.set_param("W", &w_init);
        c.run(&[("x", &x_data), ("d_output", &[1.0f32])])
    };
    let cpu = run(Device::Cpu);
    let ane = run(Device::Ane);
    // outputs = [loss, dW]
    assert_close(&ane[0], &cpu[0], "loss");
    assert_close(&ane[1], &cpu[1], "dW");
}

#[test]
fn ane_rms_norm_backward_input_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    // A graph whose output IS the RmsNorm input-gradient op. On CoreML it
    // decomposes to primitives; CPU runs the native kernel.
    let (rows, h) = (2usize, 4usize);
    let (eps, axis) = (1e-6f32, -1i32);
    let mut g = Graph::new("rms_bwd_in");
    let x = g.input("x", Shape::new(&[rows, h], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[h], DType::F32));
    let beta = g.param("beta", Shape::new(&[h], DType::F32));
    let dy = g.input("dy", Shape::new(&[rows, h], DType::F32));
    let dx = g.rms_norm_backward_input(x, gamma, beta, dy, axis, eps);
    g.set_outputs(vec![dx]);

    let x_data: Vec<f32> = (0..rows * h).map(|i| (i as f32) * 0.2 - 0.5).collect();
    let dy_data: Vec<f32> = (0..rows * h).map(|i| 0.1 + (i as f32) * 0.03).collect();
    let gamma_init = vec![1.0f32; h];
    let beta_init = vec![0.0f32; h];

    // Under `training`, `RmsNormBackwardInput` is in COREML_NATIVE_BACKWARD_OPS,
    // so CoreML lowers it through the NATIVE MIL kernel (not decompose). The CPU
    // backend's plain compile doesn't lower a *standalone* backward op, so give it
    // the decomposition as the oracle. Native-ANE ≈ decompose-CPU validates the
    // native kernel mirrors the shared backward math.
    let cpu_g = rlx_opt::rlx_autodiff::decompose_backward_ops_except(g.clone(), &[]);

    let run = |device: Device, graph: &Graph| -> Vec<f32> {
        let mut c = Session::new(device).compile(graph.clone());
        c.set_param("gamma", &gamma_init);
        c.set_param("beta", &beta_init);
        c.run(&[("x", &x_data), ("dy", &dy_data)]).remove(0)
    };
    assert_close(
        &run(Device::Ane, &g),
        &run(Device::Cpu, &cpu_g),
        "rms_norm dx",
    );
}

/// Native RMSNorm gamma/beta backward kernels ≈ decompose oracle on CPU.
#[test]
fn ane_rms_norm_backward_gamma_beta_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (rows, h) = (3usize, 4usize);
    let (eps, axis) = (1e-6f32, -1i32);
    let x_data: Vec<f32> = (0..rows * h).map(|i| (i as f32) * 0.13 - 0.4).collect();
    let dy_data: Vec<f32> = (0..rows * h).map(|i| 0.2 - (i as f32) * 0.02).collect();
    let gamma_init = vec![0.8f32; h];
    let beta_init = vec![0.1f32; h];

    // Build a graph whose single output is the gamma (or beta) gradient.
    let build = |which: &str| -> Graph {
        let mut g = Graph::new("rms_bwd_gb");
        let x = g.input("x", Shape::new(&[rows, h], DType::F32));
        let gamma = g.param("gamma", Shape::new(&[h], DType::F32));
        let beta = g.param("beta", Shape::new(&[h], DType::F32));
        let dy = g.input("dy", Shape::new(&[rows, h], DType::F32));
        let out = match which {
            "gamma" => g.rms_norm_backward_gamma(x, gamma, beta, dy, axis, eps),
            _ => g.rms_norm_backward_beta(x, gamma, beta, dy, axis, eps),
        };
        g.set_outputs(vec![out]);
        g
    };

    let run = |device: Device, graph: &Graph| -> Vec<f32> {
        let mut c = Session::new(device).compile(graph.clone());
        c.set_param("gamma", &gamma_init);
        c.set_param("beta", &beta_init);
        c.run(&[("x", &x_data), ("dy", &dy_data)]).remove(0)
    };

    for which in ["gamma", "beta"] {
        let g = build(which);
        let cpu_g = rlx_opt::rlx_autodiff::decompose_backward_ops_except(g.clone(), &[]);
        assert_close(
            &run(Device::Ane, &g),
            &run(Device::Cpu, &cpu_g),
            &format!("rms_norm d{which}"),
        );
    }
}

/// Native MaxPool2d backward ≈ CPU decompose oracle (incl. an all-equal window
/// — the relu→maxpool tie case — which must route to the first position).
#[test]
fn ane_max_pool2d_backward_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    // [1,2,4,4] → 2x2/2 → [1,2,2,2]. Small enough that the decompose oracle
    // stays under its 4096 cap; the native kernel has no such cap.
    let mut g = Graph::new("mp_bwd");
    let x = g.input("x", Shape::new(&[1, 2, 4, 4], DType::F32));
    let dy = g.input("dy", Shape::new(&[1, 2, 2, 2], DType::F32));
    let dx = g.maxpool2d_backward(x, dy, vec![2, 2], vec![2, 2], vec![0, 0]);
    g.set_outputs(vec![dx]);

    // Channel 0: distinct values (clear argmax). Channel 1: top-left window all
    // zeros (a tie → gradient goes to every position; ANE and CPU must agree).
    let mut x_data = vec![0.0f32; 32];
    for i in 0..16 {
        x_data[i] = (i as f32 * 0.37).sin(); // ch0, no ties
    }
    for i in 0..16 {
        x_data[16 + i] = if i < 2 || (i >= 4 && i < 6) {
            0.0
        } else {
            (i as f32).cos()
        };
    }
    let dy_data: Vec<f32> = (0..8).map(|i| 0.5 + i as f32 * 0.25).collect();

    let cpu_g = rlx_opt::rlx_autodiff::decompose_backward_ops_except(g.clone(), &[]);
    let run = |device: Device, graph: &Graph| -> Vec<f32> {
        Session::new(device)
            .compile(graph.clone())
            .run(&[("x", &x_data), ("dy", &dy_data)])
            .remove(0)
    };
    assert_close(
        &run(Device::Ane, &g),
        &run(Device::Cpu, &cpu_g),
        "maxpool dx",
    );
}

/// Native GroupNorm backward (input + gamma) on the ANE matches the CPU kernel at
/// N=2. The native kernel reshapes `[N,C,H,W] → [N,G,M]` and reduces the group axis;
/// N=2 confirms that reshape keeps batches and groups separate — a scrambled grouping
/// would diverge here even though it agrees at N=1. CPU runs its own native kernel
/// (batch-independent for input; the gamma path is FD-verified for N>1).
#[test]
fn ane_group_norm_backward_matches_cpu_n2() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (n, c, hh, w, ng) = (2usize, 4usize, 2usize, 2usize, 2usize);
    let nel = n * c * hh * w;
    let x_data: Vec<f32> = (0..nel).map(|i| i as f32 * 0.1 - 0.7).collect();
    let dy_data: Vec<f32> = (0..nel).map(|i| 0.2 + 0.05 * i as f32).collect();
    let gamma_init: Vec<f32> = (0..c).map(|i| 0.5 + 0.3 * i as f32).collect();

    let mut gin = Graph::new("gn_in");
    let x = gin.input("x", Shape::new(&[n, c, hh, w], DType::F32));
    let gamma = gin.param("gamma", Shape::new(&[c], DType::F32));
    let beta = gin.param("beta", Shape::new(&[c], DType::F32));
    let dy = gin.input("dy", Shape::new(&[n, c, hh, w], DType::F32));
    let dx = gin.group_norm_backward_input(x, gamma, beta, dy, ng, 1e-5);
    gin.set_outputs(vec![dx]);
    let run_in = |device: Device| -> Vec<f32> {
        let mut cc = Session::new(device).compile(gin.clone());
        cc.set_param("gamma", &gamma_init);
        cc.set_param("beta", &vec![0.0f32; c]);
        cc.run(&[("x", &x_data), ("dy", &dy_data)]).remove(0)
    };
    assert_close(
        &run_in(Device::Ane),
        &run_in(Device::Cpu),
        "group_norm dx N=2",
    );

    let mut gg = Graph::new("gn_g");
    let x2 = gg.input("x", Shape::new(&[n, c, hh, w], DType::F32));
    let dy2 = gg.input("dy", Shape::new(&[n, c, hh, w], DType::F32));
    let dgamma = gg.group_norm_backward_gamma(x2, dy2, Shape::new(&[c], DType::F32), ng, 1e-5);
    gg.set_outputs(vec![dgamma]);
    let run_g = |device: Device| -> Vec<f32> {
        Session::new(device)
            .compile(gg.clone())
            .run(&[("x", &x_data), ("dy", &dy_data)])
            .remove(0)
    };
    assert_close(
        &run_g(Device::Ane),
        &run_g(Device::Cpu),
        "group_norm dgamma N=2",
    );
}

/// Attention backward (dQ/dK/dV) runs on the ANE and matches CPU. Attention has no
/// native MIL backward kernel — its decompose reconstructs the forward as primitives
/// and autodiffs it (matmul + softmax), which CoreML runs directly. This confirms
/// attention training works on-device via that route, so a native fused kernel is a
/// pure-perf optimization, not a correctness gap.
#[test]
fn ane_attention_backward_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    use rlx_ir::op::MaskKind;
    let (b, h, s, d) = (1usize, 1usize, 3usize, 4usize);
    let mut g = Graph::new("attn");
    let q = g.input("q", Shape::new(&[b, h, s, d], DType::F32));
    let k = g.input("k", Shape::new(&[b, h, s, d], DType::F32));
    let v = g.input("v", Shape::new(&[b, h, s, d], DType::F32));
    let y = g.attention_kind(
        q,
        k,
        v,
        h,
        d,
        MaskKind::Causal,
        Shape::new(&[b, h, s, d], DType::F32),
    );
    let loss = g.reduce(
        y,
        ReduceOp::Sum,
        vec![0, 1, 2, 3],
        false,
        Shape::from_dims(&[], DType::F32),
    );
    g.set_outputs(vec![loss]);
    let bwd = rlx_opt::grad_with_loss(&g, &[q, k, v]);

    let nel = b * h * s * d;
    let mk = |seed: f32| -> Vec<f32> {
        (0..nel)
            .map(|i| ((i as f32) * 0.13 + seed).sin() * 0.5)
            .collect()
    };
    let (qd, kd, vd) = (mk(0.0), mk(1.0), mk(2.0));
    let run = |device: Device| -> Vec<Vec<f32>> {
        Session::new(device).compile(bwd.clone()).run(&[
            ("q", &qd),
            ("k", &kd),
            ("v", &vd),
            ("d_output", &[1.0]),
        ])
    };
    let (ane, cpu) = (run(Device::Ane), run(Device::Cpu));
    assert_close(&ane[1], &cpu[1], "attention dQ");
    assert_close(&ane[2], &cpu[2], "attention dK");
    assert_close(&ane[3], &cpu[3], "attention dV");
}

/// Native attention backward also handles the `[B,S,H,D]` operand layout (heads at
/// axis 2 — the Llama/Moshi convention): it transposes to canonical, computes, and
/// transposes the gradient back. `S≠H` keeps the layout unambiguous. dQ/dK/dV match
/// the CPU decompose, confirming the layout wrapper (not just canonical) is correct.
#[test]
fn ane_attention_backward_bshd_layout_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    use rlx_ir::op::MaskKind;
    let (b, s, h, d) = (1usize, 3usize, 2usize, 4usize); // [B,S,H,D], S≠H
    let mut g = Graph::new("attn_bshd");
    let q = g.input("q", Shape::new(&[b, s, h, d], DType::F32));
    let k = g.input("k", Shape::new(&[b, s, h, d], DType::F32));
    let v = g.input("v", Shape::new(&[b, s, h, d], DType::F32));
    let y = g.attention_kind(
        q,
        k,
        v,
        h,
        d,
        MaskKind::Causal,
        Shape::new(&[b, s, h, d], DType::F32),
    );
    let loss = g.reduce(
        y,
        ReduceOp::Sum,
        vec![0, 1, 2, 3],
        false,
        Shape::from_dims(&[], DType::F32),
    );
    g.set_outputs(vec![loss]);
    let bwd = rlx_opt::grad_with_loss(&g, &[q, k, v]);

    let nel = b * s * h * d;
    let mk = |seed: f32| -> Vec<f32> {
        (0..nel)
            .map(|i| ((i as f32) * 0.11 + seed).cos() * 0.5)
            .collect()
    };
    let (qd, kd, vd) = (mk(0.0), mk(1.0), mk(2.0));
    let run = |device: Device| -> Vec<Vec<f32>> {
        Session::new(device).compile(bwd.clone()).run(&[
            ("q", &qd),
            ("k", &kd),
            ("v", &vd),
            ("d_output", &[1.0]),
        ])
    };
    let (ane, cpu) = (run(Device::Ane), run(Device::Cpu));
    assert_close(&ane[1], &cpu[1], "bshd attention dQ");
    assert_close(&ane[2], &cpu[2], "bshd attention dK");
    assert_close(&ane[3], &cpu[3], "bshd attention dV");
}

/// Softmax-cross-entropy loss gradient runs on the ANE (regression for the MIL
/// `log` op that was emitted without its required `epsilon`, failing model load).
#[test]
fn ane_softmax_cross_entropy_grad_runs() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (n, d, c) = (4usize, 5usize, 3usize);
    let mut g = Graph::new("sce");
    let x = g.input("x", Shape::new(&[n, d], DType::F32));
    let w = g.param("W", Shape::new(&[d, c], DType::F32));
    let labels = g.input("labels", Shape::new(&[n], DType::F32));
    let logits = g.matmul(x, w, Shape::new(&[n, c], DType::F32));
    // softmax_cross_entropy_with_logits is per-example [N]; training reduces it
    // to a scalar before backprop (so the loss cotangent seed is scalar).
    let per_ex = g.softmax_cross_entropy_with_logits(logits, labels);
    let loss = g.reduce(
        per_ex,
        ReduceOp::Sum,
        vec![0],
        false,
        Shape::from_dims(&[], DType::F32),
    );
    g.set_outputs(vec![loss]);
    let bwd = rlx_opt::grad_with_loss(&g, &[g.param_id("W").unwrap()]);

    let x_data: Vec<f32> = (0..n * d).map(|i| i as f32 * 0.1).collect();
    let labels = vec![0.0f32, 1.0, 2.0, 1.0];
    let run = |device: Device| -> Vec<Vec<f32>> {
        let mut compiled = Session::new(device).compile(bwd.clone());
        compiled.set_param("W", &vec![0.1f32; d * c]);
        compiled.run(&[("x", &x_data), ("labels", &labels), ("d_output", &[1.0])])
    };
    let (ane, cpu) = (run(Device::Ane), run(Device::Cpu));
    assert_close(&ane[0], &cpu[0], "sce loss");
    assert_close(&ane[1], &cpu[1], "sce dW");
}

/// Automatic-Floating-Point parity: the `AutoMixed` (f16) backward on the ANE
/// points the same direction as the fp32 CPU gradient (cosine ≈ 1), even though
/// f16 round-off loosens the magnitude. Confirms AMP mixed precision is correct,
/// not just finite.
#[test]
fn ane_amp_grad_matches_cpu_direction() {
    use rlx_runtime::PrecisionPolicy;
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (g, w) = silu_matmul_forward();
    let bwd = rlx_opt::grad_with_loss(&g, &[w]);
    let x_data: Vec<f32> = (0..2 * 3).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w_init: Vec<f32> = (0..3 * 4).map(|i| (i as f32) * 0.05 - 0.2).collect();

    let cpu = {
        let mut c = Session::new(Device::Cpu).compile(bwd.clone());
        c.set_param("W", &w_init);
        c.run(&[("x", &x_data), ("d_output", &[1.0f32])]).remove(1)
    };
    let ane = {
        let mut c = Session::new(Device::Ane)
            .with_policy(PrecisionPolicy::AutoMixed)
            .compile(bwd.clone());
        c.set_param("W", &w_init);
        c.run(&[("x", &x_data), ("d_output", &[1.0f32])]).remove(1)
    };
    assert!(
        ane.iter().all(|v| v.is_finite()),
        "AMP dW non-finite: {ane:?}"
    );
    let (mut dot, mut na, mut nc) = (0.0f32, 0.0f32, 0.0f32);
    for (&a, &c) in ane.iter().zip(&cpu) {
        dot += a * c;
        na += a * a;
        nc += c * c;
    }
    let cosine = dot / (na.sqrt() * nc.sqrt());
    assert!(
        cosine > 0.99,
        "AMP f16 grad off-direction: cosine={cosine}\n  ane={ane:?}\n  cpu={cpu:?}"
    );
}

/// Native packed DiT reverse on ANE ≈ CPU (CPU uses its native packed kernel;
/// ANE uses the new MIL compose arm under `training`).
#[test]
fn ane_dit_packed_backward_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (b, s, d) = (2usize, 3usize, 4usize);
    let eps = 1e-5f32;
    let mut g = Graph::new("dit_packed_bwd");
    let x = g.input("x", Shape::new(&[b, s, d], DType::F32));
    let scale = g.input("scale", Shape::new(&[b, 1, d], DType::F32));
    let shift = g.input("shift", Shape::new(&[b, 1, d], DType::F32));
    let dy = g.input("dy", Shape::new(&[b, s, d], DType::F32));
    let packed = g.ada_layer_norm_backward(x, scale, shift, dy, AdaNormKind::LayerNorm, eps);
    g.set_outputs(vec![packed]);

    assert!(
        rlx_runtime::supports(
            Device::Ane,
            &Op::AdaLayerNormBackward {
                norm: AdaNormKind::LayerNorm,
                eps
            }
        ),
        "Ane should claim AdaLayerNormBackward"
    );

    let x_data: Vec<f32> = (0..b * s * d).map(|i| (i as f32) * 0.17 - 0.4).collect();
    let scale_data: Vec<f32> = (0..b * d).map(|i| 0.05 * (i as f32) - 0.1).collect();
    let shift_data: Vec<f32> = (0..b * d).map(|i| -0.03 * (i as f32)).collect();
    let dy_data: Vec<f32> = (0..b * s * d).map(|i| 0.1 + 0.02 * (i as f32)).collect();
    let feeds = [
        ("x", x_data.as_slice()),
        ("scale", scale_data.as_slice()),
        ("shift", shift_data.as_slice()),
        ("dy", dy_data.as_slice()),
    ];

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.run(&feeds).remove(0)
    };
    assert_close(&run(Device::Ane), &run(Device::Cpu), "ada packed reverse");
}
