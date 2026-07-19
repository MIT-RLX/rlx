// Native (on-device) `Op::Lstm` on CoreML/ANE, unrolled over the sequence into
// MIL primitives. Verified against the CPU backend (the reference executor)
// through the public Session API — no CPU host-eval in the ANE path.
#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn approx(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    let mx = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(mx <= tol, "max abs diff {mx} > {tol}");
}

fn mk(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

fn build(b: usize, s: usize, inp: usize, h: usize, layers: usize, bidir: bool) -> Graph {
    let f = DType::F32;
    let dirs = if bidir { 2 } else { 1 };
    let in_l = |l: usize| if l == 0 { inp } else { dirs * h };
    let wih_total: usize = (0..layers).map(|l| dirs * 4 * h * in_l(l)).sum();
    let whh_total = layers * dirs * 4 * h * h;
    let bias_total = layers * dirs * 4 * h;

    let mut g = Graph::new("lstm");
    let x = g.input("x", Shape::new(&[b, s, inp], f));
    let wih = g.input("w_ih", Shape::new(&[wih_total], f));
    let whh = g.input("w_hh", Shape::new(&[whh_total], f));
    let bias = g.input("bias", Shape::new(&[bias_total], f));
    let y = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: layers,
            bidirectional: bidir,
            carry: false,
        },
        vec![x, wih, whh, bias],
        Shape::new(&[b, s, dirs * h], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn run_parity(b: usize, s: usize, inp: usize, h: usize, layers: usize, bidir: bool) {
    let dirs = if bidir { 2 } else { 1 };
    let in_l = |l: usize| if l == 0 { inp } else { dirs * h };
    let xd = mk(b * s * inp, 1);
    let wihd = mk((0..layers).map(|l| dirs * 4 * h * in_l(l)).sum(), 2);
    let whhd = mk(layers * dirs * 4 * h * h, 3);
    let bd = mk(layers * dirs * 4 * h, 4);
    let feed: Vec<(&str, &[f32])> =
        vec![("x", &xd), ("w_ih", &wihd), ("w_hh", &whhd), ("bias", &bd)];

    let mut cpu = Session::new(Device::Cpu).compile(build(b, s, inp, h, layers, bidir));
    let cpu_out = cpu.run(&feed).remove(0);
    let mut ane = Session::new(Device::Ane).compile(build(b, s, inp, h, layers, bidir));
    let ane_out = ane.run(&feed).remove(0);
    approx(&ane_out, &cpu_out, 1e-3);
}

/// Proof the `carry = false` LSTM graph lowers *fully* to a single MIL program
/// with no host segment — i.e. it runs entirely on-device, never touching the
/// CPU `execute_lstm_f32` fallback. (`carry = true` would segment to the host.)
#[test]
fn bilstm_is_fully_native_mil() {
    use rlx_coreml::hybrid::{ExecutionPlan, plan_execution};
    let plan = plan_execution(&build(1, 6, 5, 4, 2, true)).expect("plan");
    assert!(
        matches!(plan, ExecutionPlan::MilOnly),
        "carry=false BiLSTM must be MilOnly (no CPU host segment), got {plan:?}"
    );
}

#[test]
fn lstm_vs_cpu() {
    run_parity(2, 5, 4, 4, 1, false);
}

#[test]
fn bilstm_vs_cpu() {
    run_parity(2, 5, 4, 4, 1, true);
}

/// Multi-layer bidirectional — exercises the per-layer `wih_cursor` offsets
/// (layer 0 reads `inp`, later layers read `dirs*h`).
#[test]
fn multilayer_bilstm_vs_cpu() {
    run_parity(1, 6, 5, 4, 3, true);
}

/// Kokoro StyleTTS2 encoder shape: H=256, bidirectional, single layer.
#[test]
fn bilstm_kokoro_shape_vs_cpu() {
    run_parity(1, 8, 128, 256, 1, true);
}
