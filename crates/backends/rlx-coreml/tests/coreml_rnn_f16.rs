// fp16 (ANE-preferred) LSTM / GRU / RNN on CoreML. Compiling at `float_dtype =
// F16` demotes the graph to f16 storage; the recurrent unrolls now emit their
// intermediate tensors in the model's float dtype (not hardcoded f32), so they
// run entirely in fp16 on the Neural Engine. Verified against the f32 CPU
// reference within fp16 tolerance.
#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_coreml::CoremlExecutable;
use rlx_coreml::mil::LowerOptions;
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn mk(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

fn maxd(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn f16_opts() -> LowerOptions {
    LowerOptions {
        float_dtype: DType::F16,
        ..Default::default()
    }
}

/// Compile `g` at f16 on CoreML, feed `feed`, and compare to the f32 CPU
/// reference within an fp16-appropriate tolerance.
fn f16_vs_cpu(label: &str, build: impl Fn() -> Graph, feed: &[(&str, &[f32])]) {
    let cpu = Session::new(Device::Cpu)
        .compile(build())
        .run(feed)
        .remove(0);
    let mut e = CoremlExecutable::compile_with_lower_opts(build(), f16_opts());
    let ane = e.run(feed).expect("f16 run").remove(0);
    let d = maxd(&ane, &cpu);
    // Relative magnitude of the reference, for context.
    let rms = (cpu.iter().map(|v| v * v).sum::<f32>() / cpu.len() as f32).sqrt();
    println!("{label} f16: max|Δ|={d:.4}  (ref rms={rms:.3})");
    assert!(
        d < 5e-2,
        "{label} f16 vs f32-CPU exceeds fp16 tolerance: {d}"
    );
}

fn dims(inp: usize, h: usize, layers: usize, bidir: bool, gates: usize) -> (usize, usize, usize) {
    let ex = rlx_cpu::thunk::rnn_expected_lens(gates, 1, 1, inp, h, layers, bidir);
    (ex.w_ih, ex.w_hh, ex.bias)
}

#[test]
fn lstm_f16_vs_cpu() {
    let (b, s, inp, h, l, bi) = (2usize, 5usize, 4usize, 4usize, 1usize, true);
    let f = DType::F32;
    let d = if bi { 2 } else { 1 };
    let (wt, ht, bt) = dims(inp, h, l, bi, 4);
    let (xd, wih, whh, bias) = (mk(b * s * inp, 1), mk(wt, 2), mk(ht, 3), mk(bt, 4));
    let build = || {
        let mut g = Graph::new("lstm");
        let x = g.input("x", Shape::new(&[b, s, inp], f));
        let a = g.input("w_ih", Shape::new(&[wt], f));
        let c = g.input("w_hh", Shape::new(&[ht], f));
        let e = g.input("bias", Shape::new(&[bt], f));
        let y = g.add_node(
            Op::Lstm {
                hidden_size: h,
                num_layers: l,
                bidirectional: bi,
                carry: false,
            },
            vec![x, a, c, e],
            Shape::new(&[b, s, d * h], f),
        );
        g.set_outputs(vec![y]);
        g
    };
    f16_vs_cpu(
        "LSTM",
        build,
        &[("x", &xd), ("w_ih", &wih), ("w_hh", &whh), ("bias", &bias)],
    );
}

#[test]
fn gru_f16_vs_cpu() {
    let (b, s, inp, h, l, bi) = (2usize, 5usize, 4usize, 4usize, 1usize, true);
    let f = DType::F32;
    let d = if bi { 2 } else { 1 };
    let (wt, ht, bt) = dims(inp, h, l, bi, 3);
    let (xd, wih, whh, bih, bhh) = (
        mk(b * s * inp, 1),
        mk(wt, 2),
        mk(ht, 3),
        mk(bt, 4),
        mk(bt, 5),
    );
    let build = || {
        let mut g = Graph::new("gru");
        let x = g.input("x", Shape::new(&[b, s, inp], f));
        let a = g.input("w_ih", Shape::new(&[wt], f));
        let c = g.input("w_hh", Shape::new(&[ht], f));
        let di = g.input("b_ih", Shape::new(&[bt], f));
        let dh = g.input("b_hh", Shape::new(&[bt], f));
        let y = g.add_node(
            Op::Gru {
                hidden_size: h,
                num_layers: l,
                bidirectional: bi,
                carry: false,
            },
            vec![x, a, c, di, dh],
            Shape::new(&[b, s, d * h], f),
        );
        g.set_outputs(vec![y]);
        g
    };
    f16_vs_cpu(
        "GRU",
        build,
        &[
            ("x", &xd),
            ("w_ih", &wih),
            ("w_hh", &whh),
            ("b_ih", &bih),
            ("b_hh", &bhh),
        ],
    );
}

#[test]
fn rnn_f16_vs_cpu() {
    let (b, s, inp, h, l, bi) = (2usize, 5usize, 4usize, 4usize, 1usize, true);
    let f = DType::F32;
    let d = if bi { 2 } else { 1 };
    let (wt, ht, bt) = dims(inp, h, l, bi, 1);
    let (xd, wih, whh, bias) = (mk(b * s * inp, 1), mk(wt, 2), mk(ht, 3), mk(bt, 4));
    let build = || {
        let mut g = Graph::new("rnn");
        let x = g.input("x", Shape::new(&[b, s, inp], f));
        let a = g.input("w_ih", Shape::new(&[wt], f));
        let c = g.input("w_hh", Shape::new(&[ht], f));
        let e = g.input("bias", Shape::new(&[bt], f));
        let y = g.add_node(
            Op::Rnn {
                hidden_size: h,
                num_layers: l,
                bidirectional: bi,
                carry: false,
                relu: false,
            },
            vec![x, a, c, e],
            Shape::new(&[b, s, d * h], f),
        );
        g.set_outputs(vec![y]);
        g
    };
    f16_vs_cpu(
        "RNN",
        build,
        &[("x", &xd), ("w_ih", &wih), ("w_hh", &whh), ("bias", &bias)],
    );
}
