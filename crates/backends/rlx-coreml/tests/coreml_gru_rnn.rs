// Native (on-device) `Op::Gru` / `Op::Rnn` (carry = false) on CoreML/ANE,
// unrolled into MIL primitives. Verified vs the CPU reference kernels through
// the public Session API — no CPU host-eval in the ANE path.
#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_coreml::hybrid::{ExecutionPlan, plan_execution};
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

/// Packed `(w_ih, w_hh, bias)` element counts — derived from the canonical
/// `rlx_cpu::thunk::rnn_expected_lens` so the test can never mis-size a weight
/// (batch/seq are irrelevant to these three).
fn dims(
    _b: usize,
    _s: usize,
    inp: usize,
    h: usize,
    layers: usize,
    bidir: bool,
    gates: usize,
) -> (usize, usize, usize) {
    let ex = rlx_cpu::thunk::rnn_expected_lens(gates, 1, 1, inp, h, layers, bidir);
    (ex.w_ih, ex.w_hh, ex.bias)
}

fn build_gru(b: usize, s: usize, inp: usize, h: usize, layers: usize, bidir: bool) -> Graph {
    let f = DType::F32;
    let dirs = if bidir { 2 } else { 1 };
    let (wt, ht, bt) = dims(b, s, inp, h, layers, bidir, 3);
    let mut g = Graph::new("gru");
    let x = g.input("x", Shape::new(&[b, s, inp], f));
    let a = g.input("w_ih", Shape::new(&[wt], f));
    let c = g.input("w_hh", Shape::new(&[ht], f));
    let d = g.input("b_ih", Shape::new(&[bt], f));
    let e = g.input("b_hh", Shape::new(&[bt], f));
    let y = g.add_node(
        Op::Gru {
            hidden_size: h,
            num_layers: layers,
            bidirectional: bidir,
            carry: false,
        },
        vec![x, a, c, d, e],
        Shape::new(&[b, s, dirs * h], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn build_rnn(
    b: usize,
    s: usize,
    inp: usize,
    h: usize,
    layers: usize,
    bidir: bool,
    relu: bool,
) -> Graph {
    let f = DType::F32;
    let dirs = if bidir { 2 } else { 1 };
    let (wt, ht, bt) = dims(b, s, inp, h, layers, bidir, 1);
    let mut g = Graph::new("rnn");
    let x = g.input("x", Shape::new(&[b, s, inp], f));
    let a = g.input("w_ih", Shape::new(&[wt], f));
    let c = g.input("w_hh", Shape::new(&[ht], f));
    let d = g.input("bias", Shape::new(&[bt], f));
    let y = g.add_node(
        Op::Rnn {
            hidden_size: h,
            num_layers: layers,
            bidirectional: bidir,
            carry: false,
            relu,
        },
        vec![x, a, c, d],
        Shape::new(&[b, s, dirs * h], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn gru_parity(b: usize, s: usize, inp: usize, h: usize, l: usize, bidir: bool) {
    let (wt, ht, bt) = dims(b, s, inp, h, l, bidir, 3);
    let (xd, wih, whh, bih, bhh) = (
        mk(b * s * inp, 1),
        mk(wt, 2),
        mk(ht, 3),
        mk(bt, 4),
        mk(bt, 5),
    );
    let feed: Vec<(&str, &[f32])> = vec![
        ("x", &xd),
        ("w_ih", &wih),
        ("w_hh", &whh),
        ("b_ih", &bih),
        ("b_hh", &bhh),
    ];
    let cpu = Session::new(Device::Cpu)
        .compile(build_gru(b, s, inp, h, l, bidir))
        .run(&feed)
        .remove(0);
    let ane = Session::new(Device::Ane)
        .compile(build_gru(b, s, inp, h, l, bidir))
        .run(&feed)
        .remove(0);
    approx(&ane, &cpu, 1e-3);
}

fn rnn_parity(b: usize, s: usize, inp: usize, h: usize, l: usize, bidir: bool, relu: bool) {
    let (wt, ht, bt) = dims(b, s, inp, h, l, bidir, 1);
    let (xd, wih, whh, bias) = (mk(b * s * inp, 1), mk(wt, 2), mk(ht, 3), mk(bt, 4));
    let feed: Vec<(&str, &[f32])> =
        vec![("x", &xd), ("w_ih", &wih), ("w_hh", &whh), ("bias", &bias)];
    let cpu = Session::new(Device::Cpu)
        .compile(build_rnn(b, s, inp, h, l, bidir, relu))
        .run(&feed)
        .remove(0);
    let ane = Session::new(Device::Ane)
        .compile(build_rnn(b, s, inp, h, l, bidir, relu))
        .run(&feed)
        .remove(0);
    approx(&ane, &cpu, 1e-3);
}

#[test]
fn gru_vs_cpu() {
    gru_parity(2, 5, 4, 4, 1, false);
}

#[test]
fn bigru_multilayer_vs_cpu() {
    gru_parity(1, 6, 5, 4, 2, true);
}

/// Kokoro-scale GRU shape (H=256, bidirectional).
#[test]
fn bigru_wide_vs_cpu() {
    gru_parity(1, 8, 128, 256, 1, true);
}

#[test]
fn rnn_tanh_bi_vs_cpu() {
    rnn_parity(2, 5, 4, 4, 1, true, false);
}

#[test]
fn rnn_relu_multilayer_vs_cpu() {
    rnn_parity(1, 6, 5, 4, 3, false, true);
}

/// Kokoro-scale RNN shape (H=256, bidirectional).
#[test]
fn rnn_wide_bi_vs_cpu() {
    rnn_parity(1, 8, 128, 256, 1, true, false);
}

/// GRU and RNN (carry = false) plan as a single MIL program — fully on-device,
/// no CPU host segment.
#[test]
fn gru_rnn_are_fully_native_mil() {
    for plan in [
        plan_execution(&build_gru(1, 6, 5, 4, 2, true)).expect("gru plan"),
        plan_execution(&build_rnn(1, 6, 5, 4, 2, true, false)).expect("rnn plan"),
    ] {
        assert!(
            matches!(plan, ExecutionPlan::MilOnly),
            "must be MilOnly, got {plan:?}"
        );
    }
}
