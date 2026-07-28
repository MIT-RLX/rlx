// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Data-driven cross-backend parity for the standalone elementwise kernels —
//! the **runtime counterpart** to the canonical opcode tables in
//! `rlx_ir::opcodes`.
//!
//! Instead of one hand-written parity test per op with a `macro_rules!` arm per
//! backend (the old pattern, where "supported but never actually parity-tested
//! on backend X" was structurally easy — that's how the silent NeoX-rotate /
//! attention-mask / Compare-broadcast bugs survived), this fans a single table
//! out automatically:
//!
//! * **Forward parity** — every [`Activation`], [`BinaryOp`] and [`CmpOp`]
//!   variant (`*::ALL`) is run on the CPU oracle and on *every backend that is
//!   both compiled in and reports it can run the graph* (`supports_graph`), and
//!   the outputs are compared. Adding a variant to any of those enums extends
//!   the sweep with zero new code — and if a backend's kernel switch ever
//!   disagrees with the canonical opcode id (an off-by-one, a Round/Recip swap,
//!   a copy-paste from the wrong scheme), the wrong *function* is computed and
//!   the parity error jumps to O(1), failing here.
//! * **Gradient parity (CPU, finite differences)** — for the smooth
//!   activations, the analytic VJP from `grad_with_loss` is checked against
//!   central differences. Backward activations decompose to primitives on the
//!   GPU backends, so the CPU is the meaningful autodiff oracle; this catches
//!   sign flips / wrong derivative rules independent of backend.
//!
//! Run the fan-out against real backends with e.g.
//! `cargo test -p rlx-runtime --features metal,mlx,gpu elementwise`. With only
//! the default `cpu` feature, the CPU-oracle self-checks and the FD sweep still
//! run; the cross-backend loops simply find no extra backends and no-op.

#![cfg(feature = "cpu")]
#![allow(clippy::needless_range_loop)]

use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session, is_available, supports_graph};

const N: usize = 64;

/// Backends compiled into this build that are actually present on the box.
/// Each push is `#[cfg]`-gated on its backend feature so we never name a
/// `Device` variant that wasn't built, then filtered by live `is_available`.
fn available_backends() -> Vec<Device> {
    let mut v: Vec<Device> = Vec::new();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    v.push(Device::Metal);
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    v.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    v.push(Device::Gpu);
    #[cfg(feature = "cuda")]
    v.push(Device::Cuda);
    #[cfg(feature = "rocm")]
    v.push(Device::Rocm);
    #[cfg(feature = "vulkan")]
    v.push(Device::Vulkan);
    v.retain(|&d| is_available(d));
    v
}

fn run(device: Device, g: &Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    Session::new(device)
        .compile(g.clone())
        .run(inputs)
        .pop()
        .expect("graph produced no output")
}

/// Worst per-element error, absolute below unit magnitude and relative above —
/// tolerant enough to survive f32 transcendental-intrinsic differences across
/// vendors, tight enough that a wrong opcode (a different function entirely)
/// blows straight past it.
fn worst_rel(cpu: &[f32], got: &[f32]) -> f32 {
    assert_eq!(cpu.len(), got.len(), "length mismatch");
    cpu.iter()
        .zip(got)
        .map(|(c, g)| (c - g).abs() / c.abs().max(1.0))
        .fold(0.0, f32::max)
}

const PARITY_TOL: f32 = 5e-3;

// ── Forward parity ──────────────────────────────────────────────────────────

fn activation_graph(act: Activation) -> Graph {
    let mut g = Graph::new("act");
    let x = g.input("x", Shape::new(&[N], DType::F32));
    let y = g.activation(act, x, Shape::new(&[N], DType::F32));
    g.set_outputs(vec![y]);
    g
}

/// Domain-safe input for a forward activation (positive for log/sqrt/rsqrt,
/// away from 0 for recip, gentle range for tan).
fn activation_input(act: Activation) -> Vec<f32> {
    let base: Vec<f32> = (0..N)
        .map(|i| -2.0 + 4.0 * (i as f32) / (N as f32 - 1.0))
        .collect();
    match act {
        Activation::Log | Activation::Sqrt | Activation::Rsqrt => {
            base.iter().map(|v| v.abs() + 0.5).collect()
        }
        Activation::Recip => base
            .iter()
            .map(|v| if v.abs() < 0.3 { 0.6 } else { *v })
            .collect(),
        Activation::Tan => base.iter().map(|v| v * 0.5).collect(),
        _ => base,
    }
}

#[test]
fn activation_forward_parity() {
    let backends = available_backends();
    eprintln!("activation parity over backends: {backends:?}");
    for &act in &Activation::ALL {
        let g = activation_graph(act);
        let x = activation_input(act);
        let cpu = run(Device::Cpu, &g, &[("x", &x)]);
        assert!(
            cpu.iter().all(|v| v.is_finite()),
            "CPU oracle for {act:?} produced non-finite output"
        );
        for &dev in &backends {
            if !supports_graph(dev, &g) {
                eprintln!("skip {act:?} on {dev:?} (unsupported)");
                continue;
            }
            let got = run(dev, &g, &[("x", &x)]);
            let err = worst_rel(&cpu, &got);
            assert!(
                err < PARITY_TOL,
                "activation {act:?} on {dev:?}: worst_rel={err:.3e} \
                 (opcode/kernel mismatch?)"
            );
        }
    }
}

fn binary_graph(op: BinaryOp) -> Graph {
    let mut g = Graph::new("bin");
    let a = g.input("a", Shape::new(&[N], DType::F32));
    let b = g.input("b", Shape::new(&[N], DType::F32));
    let y = g.binary(op, a, b, Shape::new(&[N], DType::F32));
    g.set_outputs(vec![y]);
    g
}

/// Domain-safe operands for a binary op. Bitwise/shift ops act on
/// integer-valued operands; pow needs a positive base; div/mod need a nonzero
/// divisor.
fn binary_inputs(op: BinaryOp) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..N)
        .map(|i| (i as f32 * 0.19).sin() * 2.0 + 0.05)
        .collect();
    let b: Vec<f32> = (0..N)
        .map(|i| (i as f32 * 0.11).cos() * 2.0 + 0.05)
        .collect();
    match op {
        BinaryOp::Pow => (
            a.iter().map(|v| v.abs() + 0.2).collect(),
            b.iter().map(|v| v * 0.5).collect(),
        ),
        BinaryOp::Div | BinaryOp::Mod => (
            a,
            b.iter()
                .map(|v| if v.abs() < 0.5 { 0.9 } else { *v })
                .collect(),
        ),
        BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => (
            (0..N).map(|i| (i % 13) as f32).collect(),
            (0..N).map(|i| (i % 7) as f32).collect(),
        ),
        BinaryOp::Shl | BinaryOp::Shr => (
            (0..N).map(|i| (i % 17) as f32).collect(),
            (0..N).map(|i| (i % 4) as f32).collect(),
        ),
        _ => (a, b),
    }
}

/// Regression for the negative-base `pow` drift the rlxsl single source fixed:
/// `(-x)^k` for integer `k` must be signed/real on every backend (WGSL/GLSL/MSL
/// `pow()` returns NaN for a negative base, so the emitter now sign-corrects it).
/// Integer exponents keep the result finite, matching Rust `powf` on CPU.
#[test]
fn binary_pow_negative_base_matches_cpu() {
    let backends = available_backends();
    let g = binary_graph(BinaryOp::Pow);
    let a: Vec<f32> = (0..N).map(|i| -1.0 - (i % 5) as f32 * 0.5).collect(); // all negative
    let b: Vec<f32> = (0..N).map(|i| (i % 4) as f32).collect(); // exponents 0,1,2,3
    let cpu = run(Device::Cpu, &g, &[("a", &a), ("b", &b)]);
    assert!(
        cpu.iter().all(|v| v.is_finite()),
        "CPU pow neg-base oracle not finite"
    );
    for &dev in &backends {
        if !supports_graph(dev, &g) {
            continue;
        }
        let got = run(dev, &g, &[("a", &a), ("b", &b)]);
        let err = worst_rel(&cpu, &got);
        assert!(
            err < PARITY_TOL,
            "pow neg-base on {dev:?}: worst_rel={err:.3e} (bare GPU pow NaNs on negative base?)"
        );
    }
}

#[test]
fn binary_forward_parity() {
    let backends = available_backends();
    for &op in &BinaryOp::ALL {
        let g = binary_graph(op);
        let (a, b) = binary_inputs(op);
        let cpu = run(Device::Cpu, &g, &[("a", &a), ("b", &b)]);
        assert!(
            cpu.iter().all(|v| v.is_finite()),
            "CPU oracle for {op:?} produced non-finite output"
        );
        for &dev in &backends {
            if !supports_graph(dev, &g) {
                eprintln!("skip {op:?} on {dev:?} (unsupported)");
                continue;
            }
            let got = run(dev, &g, &[("a", &a), ("b", &b)]);
            let err = worst_rel(&cpu, &got);
            assert!(
                err < PARITY_TOL,
                "binary {op:?} on {dev:?}: worst_rel={err:.3e} (opcode/kernel mismatch?)"
            );
        }
    }
}

fn compare_graph(op: CmpOp) -> Graph {
    let mut g = Graph::new("cmp");
    let a = g.input("a", Shape::new(&[N], DType::F32));
    let b = g.input("b", Shape::new(&[N], DType::F32));
    let mask = g.add_node(Op::Compare(op), vec![a, b], Shape::new(&[N], DType::Bool));
    // Cast bool→f32 so the result reads back as a clean 0/1 vector (matches the
    // `Compare → Cast(F32)` pattern the backends already rely on).
    let y = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![mask],
        Shape::new(&[N], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
fn compare_forward_parity() {
    let backends = available_backends();
    // Overlapping ranges so every predicate yields a mix of 0/1 (and some exact
    // equalities for Eq/Ne).
    let a: Vec<f32> = (0..N).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..N).map(|i| (i % 4) as f32 - 2.0).collect();
    for &op in &CmpOp::ALL {
        let g = compare_graph(op);
        let cpu = run(Device::Cpu, &g, &[("a", &a), ("b", &b)]);
        for &dev in &backends {
            if !supports_graph(dev, &g) {
                eprintln!("skip {op:?} on {dev:?} (unsupported)");
                continue;
            }
            let got = run(dev, &g, &[("a", &a), ("b", &b)]);
            let err = worst_rel(&cpu, &got);
            assert!(
                err < PARITY_TOL,
                "compare {op:?} on {dev:?}: worst_rel={err:.3e} (wrong predicate?)"
            );
        }
    }
}

/// The warm-tier codegen manifest (`rlxsl`) defines each activation's
/// scalar math once. This checks that its interpreter matches the trusted CPU
/// activation kernel for every variant — closing the loop with the GPU parity
/// above: manifest == CPU (here) and generated-GPU == CPU (parity), so the
/// generated GPU kernels are validated transitively. Runs GPU-free.
#[test]
fn kernel_dsl_matches_cpu_activation_oracle() {
    for &act in &Activation::ALL {
        let g = activation_graph(act);
        let x = activation_input(act);
        let cpu = run(Device::Cpu, &g, &[("x", &x)]);
        for (i, &xi) in x.iter().enumerate() {
            let dsl = rlxsl::eval_activation(act, xi);
            let c = cpu[i];
            let err = (dsl - c).abs() / c.abs().max(1.0);
            assert!(
                err < 5e-3,
                "{act:?} at x={xi}: manifest={dsl} cpu_kernel={c} err={err:e}"
            );
        }
    }
}

// ── Gradient parity (CPU finite differences) ─────────────────────────────────

const FD_M: usize = 8;

/// `loss = sum(act(x))` with `x` a param, so `grad_with_loss(&g, &[x])` yields
/// `[loss, d_loss/d_x]`. Returns the graph and the `x` node id.
fn activation_loss_graph(act: Activation) -> (Graph, NodeId) {
    let mut g = Graph::new("actloss");
    let x = g.param("x", Shape::new(&[FD_M], DType::F32));
    let y = g.activation(act, x, Shape::new(&[FD_M], DType::F32));
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![y],
        Shape::new(&[1], DType::F32),
    );
    g.set_outputs(vec![loss]);
    (g, x)
}

fn activation_forward_loss(act: Activation, x: &[f32]) -> f32 {
    let (g, _) = activation_loss_graph(act);
    let mut c = Session::new(Device::Cpu).compile(g);
    c.set_param("x", x);
    c.run(&[])[0][0]
}

fn fd_input(act: Activation) -> Vec<f32> {
    let base: Vec<f32> = (0..FD_M)
        .map(|i| -1.2 + 2.4 * (i as f32) / (FD_M as f32 - 1.0))
        .collect();
    match act {
        Activation::Log | Activation::Sqrt | Activation::Rsqrt => {
            base.iter().map(|v| v.abs() + 0.4).collect()
        }
        Activation::Recip => base
            .iter()
            .map(|v| if v.abs() < 0.3 { 0.6 } else { *v })
            .collect(),
        _ => base,
    }
}

/// Smooth activations only — the piecewise/STE ones (Relu at 0, Round, Floor,
/// Ceil, Sign) have derivatives that intentionally disagree with a finite
/// difference, so they are validated by forward parity, not FD.
const SMOOTH: &[Activation] = &[
    Activation::Gelu,
    Activation::GeluApprox,
    Activation::Silu,
    Activation::Sigmoid,
    Activation::Tanh,
    Activation::Exp,
    Activation::Log,
    Activation::Sqrt,
    Activation::Rsqrt,
    Activation::Softplus,
    Activation::Erf,
    Activation::Mish,
    Activation::Softsign,
    Activation::Atan,
    Activation::Recip,
];

#[test]
fn activation_vjp_finite_difference_cpu() {
    let eps = 1e-3f32;
    let abs_tol = 5e-3f32;
    let rel_tol = 5e-3f32;
    for &act in SMOOTH {
        let x = fd_input(act);
        let (g, x_id) = activation_loss_graph(act);
        let bwd = grad_with_loss(&g, &[x_id]);
        let mut c = Session::new(Device::Cpu).compile(bwd);
        c.set_param("x", &x);
        let outs = c.run(&[("d_output", &[1.0f32])]);
        assert_eq!(outs.len(), 2, "{act:?}: expected [loss, grad_x]");
        let grad = &outs[1];
        assert_eq!(grad.len(), FD_M, "{act:?}: grad length");
        for i in 0..FD_M {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let fd =
                (activation_forward_loss(act, &xp) - activation_forward_loss(act, &xm)) / (2.0 * eps);
            let ad = grad[i];
            let abs_err = (fd - ad).abs();
            let rel_err = abs_err / fd.abs().max(1e-6);
            assert!(
                abs_err < abs_tol || rel_err < rel_tol,
                "{act:?} grad[{i}]: autodiff {ad:e} vs FD {fd:e} (abs {abs_err:e}, rel {rel_err:e})"
            );
        }
    }
}
