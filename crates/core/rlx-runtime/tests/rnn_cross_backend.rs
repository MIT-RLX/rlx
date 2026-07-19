//! Cross-backend agreement guard: the LSTM / GRU / Elman-RNN family
//! (`carry = false`) must produce matching output on CPU, MLX (Metal), and
//! CoreML/ANE. CPU is the reference kernel; MLX and CoreML run their native
//! on-device unrolls. Catches per-backend divergence directly (not just each
//! backend vs CPU in isolation). Needs `--features mlx,coreml` on macOS.
#![cfg(all(target_os = "macos", feature = "mlx", feature = "coreml"))]

use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn mk(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

fn maxd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Run `build` on CPU, MLX and ANE with the same feed; assert all three agree.
fn agree(label: &str, build: impl Fn() -> Graph, feed: &[(&str, &[f32])]) {
    let cpu = Session::new(Device::Cpu)
        .compile(build())
        .run(feed)
        .remove(0);
    let mlx = Session::new(Device::Mlx)
        .compile(build())
        .run(feed)
        .remove(0);
    let ane = Session::new(Device::Ane)
        .compile(build())
        .run(feed)
        .remove(0);
    let (dm, da) = (maxd(&cpu, &mlx), maxd(&cpu, &ane));
    println!(
        "{label}: CPU-vs-MLX={dm:.3e}  CPU-vs-ANE={da:.3e}  MLX-vs-ANE={:.3e}",
        maxd(&mlx, &ane)
    );
    assert!(dm < 1e-3, "{label}: MLX disagrees with CPU ({dm})");
    assert!(da < 1e-3, "{label}: ANE disagrees with CPU ({da})");
    assert!(maxd(&mlx, &ane) < 2e-3, "{label}: MLX disagrees with ANE");
}

// (b, s, inp, h, layers, bidir); H=4 bidirectional multi-layer exercises the
// per-(layer,dir) weight offsets on every backend.
const CFG: (usize, usize, usize, usize, usize, bool) = (2, 5, 4, 4, 2, true);

#[test]
fn lstm_agrees_cpu_mlx_coreml() {
    let (b, s, inp, h, l, bi) = CFG;
    let f = DType::F32;
    let ex = rlx_cpu::thunk::rnn_expected_lens(4, b, s, inp, h, l, bi);
    let d = if bi { 2 } else { 1 };
    let (xd, wih, whh, bias) = (mk(ex.x, 1), mk(ex.w_ih, 2), mk(ex.w_hh, 3), mk(ex.bias, 4));
    let build = || {
        let mut g = Graph::new("lstm");
        let x = g.input("x", Shape::new(&[b, s, inp], f));
        let a = g.input("w_ih", Shape::new(&[ex.w_ih], f));
        let c = g.input("w_hh", Shape::new(&[ex.w_hh], f));
        let e = g.input("bias", Shape::new(&[ex.bias], f));
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
    agree(
        "LSTM",
        build,
        &[("x", &xd), ("w_ih", &wih), ("w_hh", &whh), ("bias", &bias)],
    );
}

#[test]
fn gru_agrees_cpu_mlx_coreml() {
    let (b, s, inp, h, l, bi) = CFG;
    let f = DType::F32;
    let ex = rlx_cpu::thunk::rnn_expected_lens(3, b, s, inp, h, l, bi);
    let d = if bi { 2 } else { 1 };
    let (xd, wih, whh, bih, bhh) = (
        mk(ex.x, 1),
        mk(ex.w_ih, 2),
        mk(ex.w_hh, 3),
        mk(ex.bias, 4),
        mk(ex.bias, 5),
    );
    let build = || {
        let mut g = Graph::new("gru");
        let x = g.input("x", Shape::new(&[b, s, inp], f));
        let a = g.input("w_ih", Shape::new(&[ex.w_ih], f));
        let c = g.input("w_hh", Shape::new(&[ex.w_hh], f));
        let di = g.input("b_ih", Shape::new(&[ex.bias], f));
        let dh = g.input("b_hh", Shape::new(&[ex.bias], f));
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
    agree(
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
fn rnn_agrees_cpu_mlx_coreml() {
    let (b, s, inp, h, l, bi) = CFG;
    let f = DType::F32;
    let ex = rlx_cpu::thunk::rnn_expected_lens(1, b, s, inp, h, l, bi);
    let d = if bi { 2 } else { 1 };
    let (xd, wih, whh, bias) = (mk(ex.x, 1), mk(ex.w_ih, 2), mk(ex.w_hh, 3), mk(ex.bias, 4));
    let build = || {
        let mut g = Graph::new("rnn");
        let x = g.input("x", Shape::new(&[b, s, inp], f));
        let a = g.input("w_ih", Shape::new(&[ex.w_ih], f));
        let c = g.input("w_hh", Shape::new(&[ex.w_hh], f));
        let e = g.input("bias", Shape::new(&[ex.bias], f));
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
    agree(
        "RNN",
        build,
        &[("x", &xd), ("w_ih", &wih), ("w_hh", &whh), ("bias", &bias)],
    );
}
