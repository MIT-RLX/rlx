// On-device (ANE) training for the CoreML backend.
//
// `MLUpdateTask` only updates legacy NeuralNetwork updatable models, and
// rlx-coreml emits ML Program (MIL) models — so the training session reports the
// task path as unsupported and uses the gradient path: gradients computed on the
// ANE + a host optimizer step. These tests fit a tiny realizable linear model and
// check the loss collapses, the weights converge, the path/eligibility are
// reported honestly, and the ANE gradient step matches a CPU reference.
#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{
    CoremlTrainingSession, Device, MlUpdateEligibility, Optimizer, PrecisionPolicy, TrainConfig,
    UpdatePath,
};
use std::collections::HashMap;

const N: usize = 8;
const D: usize = 3;
const W_TRUE: [f32; D] = [0.5, -1.0, 2.0];

/// `loss = Σ (x·W − target)²`, training `W : [D,1]`.
fn linreg_forward() -> Graph {
    let mut g = Graph::new("linreg");
    let x = g.input("x", Shape::new(&[N, D], DType::F32));
    let w = g.param("W", Shape::new(&[D, 1], DType::F32));
    let target = g.input("target", Shape::new(&[N, 1], DType::F32));
    let pred = g.matmul(x, w, Shape::new(&[N, 1], DType::F32));
    let diff = g.binary(BinaryOp::Sub, pred, target, Shape::new(&[N, 1], DType::F32));
    let sq = g.binary(BinaryOp::Mul, diff, diff, Shape::new(&[N, 1], DType::F32));
    let loss = g.reduce(
        sq,
        ReduceOp::Sum,
        vec![0, 1],
        false,
        Shape::from_dims(&[], DType::F32),
    );
    g.set_outputs(vec![loss]);
    g
}

/// Small features in [-0.5, 0.5] (keeps `xᵀx` well-conditioned for plain SGD)
/// and the exactly-realizable target `x · W_true`.
fn data() -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..N * D)
        .map(|i| ((i as f32 * 7.0 + 1.0) % 11.0) / 11.0 - 0.5)
        .collect();
    let mut target = vec![0.0f32; N];
    for r in 0..N {
        target[r] = (0..D).map(|c| x[r * D + c] * W_TRUE[c]).sum();
    }
    (x, target)
}

#[test]
fn mlupdatetask_falls_back_to_gradient_path() {
    let mut init = HashMap::new();
    init.insert("W".to_string(), vec![0.0f32; D]);
    let sess = CoremlTrainingSession::new(linreg_forward(), &["W"], init, TrainConfig::default());

    // ML Program models aren't MLUpdateTask-updatable → gradient path.
    assert!(matches!(
        sess.mlupdatetask_eligibility(),
        MlUpdateEligibility::Unsupported { .. }
    ));
    assert_eq!(sess.update_path(), UpdatePath::Gradient);
}

#[test]
fn ane_training_converges_linear_model() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (x, target) = data();
    let mut init = HashMap::new();
    init.insert("W".to_string(), vec![0.0f32; D]);
    let cfg = TrainConfig {
        lr: 0.2,
        optimizer: Optimizer::sgd(),
    };
    let mut sess = CoremlTrainingSession::new_on(linreg_forward(), &["W"], init, cfg, Device::Ane);

    let first = sess.step(&[("x", &x), ("target", &target)]).loss;
    let mut last = first;
    for _ in 0..60 {
        let r = sess.step(&[("x", &x), ("target", &target)]);
        assert_eq!(r.path, UpdatePath::Gradient);
        last = r.loss;
    }
    // Realizable problem: the loss should collapse toward zero...
    assert!(
        last < first * 1e-2 && last < 1e-3,
        "loss did not converge: first={first}, last={last}"
    );
    // ...and the weights should approach W_true.
    let w = sess.param("W").unwrap();
    for (got, want) in w.iter().zip(W_TRUE.iter()) {
        assert!((got - want).abs() < 5e-2, "W={w:?} vs {W_TRUE:?}");
    }
}

/// The full `rlx-optim` suite is wired into the training session and runs
/// on-device: AdamW/Lion drive the realizable loss down, and the matrix-aware
/// (Muon) + 2nd-order-ish (Sophia) steppers stay numerically stable on the ANE.
#[test]
fn ane_training_with_optimizer_suite() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (x, target) = data();
    let run = |opt: Optimizer, lr: f32, steps: usize| -> (f32, f32) {
        let mut init = HashMap::new();
        init.insert("W".to_string(), vec![0.0f32; D]);
        let cfg = TrainConfig { lr, optimizer: opt };
        let mut sess =
            CoremlTrainingSession::new_on(linreg_forward(), &["W"], init, cfg, Device::Ane);
        let first = sess.step(&[("x", &x), ("target", &target)]).loss;
        let mut last = first;
        for _ in 0..steps {
            last = sess.step(&[("x", &x), ("target", &target)]).loss;
        }
        (first, last)
    };

    // AdamW reliably collapses the realizable loss toward zero.
    let (f, l) = run(Optimizer::adamw(), 0.1, 200);
    assert!(l.is_finite() && l < f * 0.02, "AdamW: first={f} last={l}");
    // Lion (sign-of-EMA): converges with a smaller LR.
    let (f, l) = run(Optimizer::lion(), 0.02, 200);
    assert!(l.is_finite() && l < f * 0.5, "Lion: first={f} last={l}");
    // Muon (Newton–Schulz, matrix) + Sophia (diagonal-Hessian): wired + stable.
    for (opt, lr) in [(Optimizer::muon(), 0.02f32), (Optimizer::sophia(), 0.05)] {
        let (f, l) = run(opt, lr, 50);
        assert!(
            l.is_finite(),
            "{opt:?}: non-finite loss (first={f} last={l})"
        );
    }
}

/// Gradient accumulation: averaging N micro-batches of the SAME data and stepping
/// must equal one plain step on that data (mean of N identical grads = the grad).
/// Also checks the pending-count bookkeeping.
#[test]
fn ane_gradient_accumulation_matches_single_step() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (x, target) = data();
    let init = || {
        let mut m = HashMap::new();
        m.insert("W".to_string(), vec![0.1f32; D]);
        m
    };
    let cfg = TrainConfig {
        lr: 0.1,
        optimizer: Optimizer::adam(),
    };

    let mut acc = CoremlTrainingSession::new_on(linreg_forward(), &["W"], init(), cfg, Device::Ane);
    for _ in 0..3 {
        acc.accumulate(&[("x", &x), ("target", &target)]);
    }
    assert_eq!(acc.pending_accumulation(), 3);
    assert_eq!(acc.step_accumulated(), 3);
    assert_eq!(acc.pending_accumulation(), 0);

    let mut one = CoremlTrainingSession::new_on(linreg_forward(), &["W"], init(), cfg, Device::Ane);
    one.step(&[("x", &x), ("target", &target)]);

    let (wa, wb) = (acc.param("W").unwrap(), one.param("W").unwrap());
    for (a, b) in wa.iter().zip(wb) {
        assert!(
            (a - b).abs() < 1e-4,
            "accum W {wa:?} vs single-step W {wb:?}"
        );
    }
    // (Micro-batches must share the compiled graph's static shape; accumulation
    // averages several same-shape batches of *different* data in real training.)
}

/// On-device fused momentum-SGD (the update runs on the ANE, velocity is graph I/O)
/// matches the host `rlx-optim` SGD path step-for-step — same `v=μv+g; w−=lr·v`.
#[test]
fn ane_fused_on_device_sgd_matches_host() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (x, target) = data();
    let init = || {
        let mut m = HashMap::new();
        m.insert("W".to_string(), vec![0.05f32; D]);
        m
    };
    let cfg = TrainConfig {
        lr: 0.05,
        optimizer: Optimizer::Sgd { momentum: 0.9 },
    };

    let mut fused =
        CoremlTrainingSession::new_on(linreg_forward(), &["W"], init(), cfg, Device::Ane)
            .with_fused_optimizer();
    assert!(
        fused.fused_active(),
        "fused path should be active for momentum-SGD"
    );
    let mut host =
        CoremlTrainingSession::new_on(linreg_forward(), &["W"], init(), cfg, Device::Ane);
    assert!(!host.fused_active());

    let first = fused.step(&[("x", &x), ("target", &target)]).loss;
    host.step(&[("x", &x), ("target", &target)]);
    let mut last = first;
    for _ in 0..40 {
        last = fused.step(&[("x", &x), ("target", &target)]).loss;
        host.step(&[("x", &x), ("target", &target)]);
    }
    // On-device fused update tracks the host optimizer step-for-step.
    let (wf, wh) = (fused.param("W").unwrap(), host.param("W").unwrap());
    for (a, b) in wf.iter().zip(wh) {
        assert!(
            (a - b).abs() < 1e-3,
            "fused (on-device) W {wf:?} vs host W {wh:?}"
        );
    }
    // ...and the on-device path actually trains (loss collapses).
    assert!(
        last < first * 0.1,
        "fused training did not converge: {first} → {last}"
    );
}

#[test]
fn ane_gradient_step_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (x, target) = data();
    let cfg = TrainConfig {
        lr: 0.1,
        optimizer: Optimizer::adam(),
    };
    let make = |device| {
        let mut init = HashMap::new();
        init.insert("W".to_string(), vec![0.25f32; D]);
        CoremlTrainingSession::new_on(linreg_forward(), &["W"], init, cfg, device)
    };
    let mut ane = make(Device::Ane);
    let mut cpu = make(Device::Cpu);

    // A few steps on each device from the same init; the ANE-computed gradient
    // path must track the CPU reference closely.
    for _ in 0..5 {
        ane.step(&[("x", &x), ("target", &target)]);
        cpu.step(&[("x", &x), ("target", &target)]);
    }
    let (wa, wc) = (ane.param("W").unwrap(), cpu.param("W").unwrap());
    for (a, c) in wa.iter().zip(wc.iter()) {
        assert!((a - c).abs() < 2e-3, "ANE W {wa:?} vs CPU W {wc:?}");
    }
}

// ───────── end-to-end MNIST-in-miniature: maxpool + softmax-CE training ─────────
// A scaled→relu→maxpool→flatten→linear→softmax-CE model trained on the ANE. The
// gradient flows through BOTH the maxpool backward (Bug A) and the softmax-CE
// backward (Bug B) to the trainable params, so a decreasing loss confirms both
// fixes compose in a realistic CNN-shaped loop.
#[test]
fn ane_maxpool_softmax_ce_training_decreases_loss() {
    use rlx_ir::Op;
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    const BATCH: usize = 8;
    const CH: usize = 2;
    const HW: usize = 4; // 4x4 -> 2x2 after 2x2/2 maxpool
    const POOL: usize = HW / 2;
    const FLAT: usize = CH * POOL * POOL; // 8
    const CLASSES: usize = 2;

    // forward: loss = sum_n CE( linear(flatten(maxpool(relu(x * S)))), labels )
    let forward = || -> Graph {
        let mut g = Graph::new("mini_cnn");
        let x = g.input("x", Shape::new(&[BATCH, CH, HW, HW], DType::F32));
        let s = g.param("S", Shape::new(&[1, CH, 1, 1], DType::F32)); // per-channel scale
        let labels = g.input("labels", Shape::new(&[BATCH], DType::F32));
        let scaled = g.binary(
            BinaryOp::Mul,
            x,
            s,
            Shape::new(&[BATCH, CH, HW, HW], DType::F32),
        );
        let act = g.activation(
            rlx_ir::op::Activation::Relu,
            scaled,
            Shape::new(&[BATCH, CH, HW, HW], DType::F32),
        );
        let pooled = g.add_node(
            Op::Pool {
                kind: ReduceOp::Max,
                kernel_size: vec![2, 2],
                stride: vec![2, 2],
                padding: vec![0, 0],
            },
            vec![act],
            Shape::new(&[BATCH, CH, POOL, POOL], DType::F32),
        );
        let flat = g.reshape(
            pooled,
            vec![BATCH as i64, FLAT as i64],
            Shape::new(&[BATCH, FLAT], DType::F32),
        );
        let wfc = g.param("Wfc", Shape::new(&[FLAT, CLASSES], DType::F32));
        let logits = g.matmul(flat, wfc, Shape::new(&[BATCH, CLASSES], DType::F32));
        let per_ex = g.softmax_cross_entropy_with_logits(logits, labels);
        let loss = g.reduce(
            per_ex,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::from_dims(&[], DType::F32),
        );
        g.set_outputs(vec![loss]);
        g
    };

    // Learnable task: class = which channel carries the larger signal. Channel
    // `label` is set high, the other low — separable through max-pool + linear.
    let mut x = vec![0.0f32; BATCH * CH * HW * HW];
    let mut labels = vec![0.0f32; BATCH];
    for n in 0..BATCH {
        let lab = n % CLASSES;
        labels[n] = lab as f32;
        for ch in 0..CH {
            let v = if ch == lab { 1.0 } else { 0.1 };
            for p in 0..HW * HW {
                x[((n * CH + ch) * HW * HW) + p] = v * (1.0 + 0.05 * (p as f32));
            }
        }
    }

    let mut init = HashMap::new();
    init.insert("S".to_string(), vec![1.0f32; CH]);
    init.insert("Wfc".to_string(), vec![0.01f32; FLAT * CLASSES]);
    let cfg = TrainConfig {
        lr: 0.05,
        optimizer: Optimizer::adam(),
    };
    let mut sess = CoremlTrainingSession::new_on(forward(), &["S", "Wfc"], init, cfg, Device::Ane);

    let first = sess.step(&[("x", &x), ("labels", &labels)]).loss;
    let mut last = first;
    for _ in 0..40 {
        let r = sess.step(&[("x", &x), ("labels", &labels)]);
        assert!(r.loss.is_finite(), "loss went non-finite: {}", r.loss);
        last = r.loss;
    }
    // Gradients flow through maxpool + softmax-CE correctly → the loss drops well
    // below its start (perfectly-separable task; the bug made it stall/diverge).
    assert!(
        last < first * 0.5,
        "loss did not decrease through maxpool+CE backward: first={first}, last={last}"
    );
}

/// Mixed-precision (Automatic Floating Point) training: an `AutoMixed` policy
/// makes the backward graph f16, so CoreML runs it on the Neural Engine — the
/// fast, lower-precision path. The realizable linear model still converges.
#[test]
fn ane_amp_f16_training_converges() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (x, target) = data();
    let mut init = HashMap::new();
    init.insert("W".to_string(), vec![0.0f32; D]);
    let cfg = TrainConfig {
        lr: 0.2,
        optimizer: Optimizer::sgd(),
    };
    // `.with_precision_policy(AutoMixed)` → f16 compute → ANE (fast).
    let mut sess = CoremlTrainingSession::new_on(linreg_forward(), &["W"], init, cfg, Device::Ane)
        .with_precision_policy(PrecisionPolicy::AutoMixed);

    let first = sess.step(&[("x", &x), ("target", &target)]).loss;
    let mut last = first;
    for _ in 0..60 {
        let r = sess.step(&[("x", &x), ("target", &target)]);
        assert!(r.loss.is_finite(), "mixed-precision loss went non-finite");
        last = r.loss;
    }
    // f16 gradients are noisier than fp32, so a looser bar — but it must learn.
    assert!(
        last < first * 0.1,
        "AMP/f16 training did not converge: first={first}, last={last}"
    );
}
