// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Gru` / `Op::Rnn` correctness vs independent packed-weight
//! references. Both decompose via `unfuse`, so this also exercises the
//! MLX/CoreML/TPU and autodiff path.

use rlx_ir::*;
use rlx_runtime::{Device, Session};

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}
fn seq(n: usize, m: usize, off: f32, scale: f32) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 % m) as f32 - off) * scale).collect()
}

struct Cfg {
    b: usize,
    s: usize,
    inp: usize,
    h: usize,
    layers: usize,
    bidir: bool,
}
impl Cfg {
    fn d(&self) -> usize {
        if self.bidir { 2 } else { 1 }
    }
    fn in_l(&self, l: usize) -> usize {
        if l == 0 { self.inp } else { self.d() * self.h }
    }
    fn wih_total(&self, gates: usize) -> usize {
        (0..self.layers)
            .map(|l| self.d() * gates * self.h * self.in_l(l))
            .sum()
    }
    fn whh_total(&self, gates: usize) -> usize {
        self.layers * self.d() * gates * self.h * self.h
    }
    fn b_total(&self, gates: usize) -> usize {
        self.layers * self.d() * gates * self.h
    }
}

#[allow(clippy::too_many_arguments)]
fn ref_gru(
    cfg: &Cfg,
    x: &[f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
) -> Vec<f32> {
    let (b, s, h, d) = (cfg.b, cfg.s, cfg.h, cfg.d());
    let g3 = 3 * h;
    let mut layer_in = x.to_vec();
    let mut in_l = cfg.inp;
    let mut wcur = 0usize;
    for l in 0..cfg.layers {
        let ow = d * h;
        let mut lo = vec![0f32; b * s * ow];
        let wb = g3 * in_l;
        for dir in 0..d {
            let ld = l * d + dir;
            let wih = &w_ih[wcur + dir * wb..][..wb];
            let whh = &w_hh[ld * g3 * h..][..g3 * h];
            let bih = &b_ih[ld * g3..][..g3];
            let bhh = &b_hh[ld * g3..][..g3];
            for bi in 0..b {
                let mut hh = vec![0f32; h];
                for step in 0..s {
                    let t = if dir == 0 { step } else { s - 1 - step };
                    let xt = &layer_in[(bi * s + t) * in_l..][..in_l];
                    let mut hn = vec![0f32; h];
                    for k in 0..h {
                        let gate = |gi: usize| {
                            let xi: f32 = bih[gi * h + k]
                                + (0..in_l)
                                    .map(|j| wih[(gi * h + k) * in_l + j] * xt[j])
                                    .sum::<f32>();
                            let hi: f32 = bhh[gi * h + k]
                                + (0..h)
                                    .map(|j| whh[(gi * h + k) * h + j] * hh[j])
                                    .sum::<f32>();
                            (xi, hi)
                        };
                        let (xr, hr) = gate(0);
                        let (xz, hz) = gate(1);
                        let (xn, hnn) = gate(2);
                        let r = sigmoid(xr + hr);
                        let z = sigmoid(xz + hz);
                        let n = (xn + r * hnn).tanh();
                        let h_new = (1.0 - z) * n + z * hh[k];
                        hn[k] = h_new;
                        lo[(bi * s + t) * ow + dir * h + k] = h_new;
                    }
                    hh = hn;
                }
            }
        }
        wcur += d * wb;
        layer_in = lo;
        in_l = ow;
    }
    layer_in
}

fn ref_rnn(cfg: &Cfg, x: &[f32], w_ih: &[f32], w_hh: &[f32], bias: &[f32], relu: bool) -> Vec<f32> {
    let (b, s, h, d) = (cfg.b, cfg.s, cfg.h, cfg.d());
    let act = |v: f32| if relu { v.max(0.0) } else { v.tanh() };
    let mut layer_in = x.to_vec();
    let mut in_l = cfg.inp;
    let mut wcur = 0usize;
    for l in 0..cfg.layers {
        let ow = d * h;
        let mut lo = vec![0f32; b * s * ow];
        let wb = h * in_l;
        for dir in 0..d {
            let ld = l * d + dir;
            let wih = &w_ih[wcur + dir * wb..][..wb];
            let whh = &w_hh[ld * h * h..][..h * h];
            let bs = &bias[ld * h..][..h];
            for bi in 0..b {
                let mut hh = vec![0f32; h];
                for step in 0..s {
                    let t = if dir == 0 { step } else { s - 1 - step };
                    let xt = &layer_in[(bi * s + t) * in_l..][..in_l];
                    let mut hn = vec![0f32; h];
                    for k in 0..h {
                        let xi: f32 =
                            bs[k] + (0..in_l).map(|j| wih[k * in_l + j] * xt[j]).sum::<f32>();
                        let hi: f32 = (0..h).map(|j| whh[k * h + j] * hh[j]).sum::<f32>();
                        let h_new = act(xi + hi);
                        hn[k] = h_new;
                        lo[(bi * s + t) * ow + dir * h + k] = h_new;
                    }
                    hh = hn;
                }
            }
        }
        wcur += d * wb;
        layer_in = lo;
        in_l = ow;
    }
    layer_in
}

fn check(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} len");
    for i in 0..actual.len() {
        assert!(
            (actual[i] - expected[i]).abs() < 1e-4,
            "{what} mismatch at {i}: {} vs {}",
            actual[i],
            expected[i]
        );
    }
}

fn run_gru(cfg: &Cfg) {
    let f = DType::F32;
    let x = seq(cfg.b * cfg.s * cfg.inp, 13, 6.0, 0.08);
    let w_ih = seq(cfg.wih_total(3), 11, 5.0, 0.04);
    let w_hh = seq(cfg.whh_total(3), 7, 3.0, 0.04);
    let b_ih = seq(cfg.b_total(3), 5, 2.0, 0.04);
    let b_hh = seq(cfg.b_total(3), 6, 3.0, 0.04);
    let expected = ref_gru(cfg, &x, &w_ih, &w_hh, &b_ih, &b_hh);

    let mut g = Graph::new("gru");
    let xi = g.input("x", Shape::new(&[cfg.b, cfg.s, cfg.inp], f));
    let wih = g.input("w_ih", Shape::new(&[cfg.wih_total(3)], f));
    let whh = g.input("w_hh", Shape::new(&[cfg.whh_total(3)], f));
    let bih = g.input("b_ih", Shape::new(&[cfg.b_total(3)], f));
    let bhh = g.input("b_hh", Shape::new(&[cfg.b_total(3)], f));
    let y = g.add_node(
        Op::Gru {
            hidden_size: cfg.h,
            num_layers: cfg.layers,
            bidirectional: cfg.bidir,
            carry: false,
        },
        vec![xi, wih, whh, bih, bhh],
        Shape::new(&[cfg.b, cfg.s, cfg.d() * cfg.h], f),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let actual = c
        .run(&[
            ("x", &x),
            ("w_ih", &w_ih),
            ("w_hh", &w_hh),
            ("b_ih", &b_ih),
            ("b_hh", &b_hh),
        ])
        .pop()
        .unwrap();
    check(&actual, &expected, "gru");
}

fn run_rnn(cfg: &Cfg, relu: bool) {
    let f = DType::F32;
    let x = seq(cfg.b * cfg.s * cfg.inp, 13, 6.0, 0.08);
    let w_ih = seq(cfg.wih_total(1), 11, 5.0, 0.04);
    let w_hh = seq(cfg.whh_total(1), 7, 3.0, 0.04);
    let bias = seq(cfg.b_total(1), 5, 2.0, 0.04);
    let expected = ref_rnn(cfg, &x, &w_ih, &w_hh, &bias, relu);

    let mut g = Graph::new("rnn");
    let xi = g.input("x", Shape::new(&[cfg.b, cfg.s, cfg.inp], f));
    let wih = g.input("w_ih", Shape::new(&[cfg.wih_total(1)], f));
    let whh = g.input("w_hh", Shape::new(&[cfg.whh_total(1)], f));
    let bias_n = g.input("bias", Shape::new(&[cfg.b_total(1)], f));
    let y = g.add_node(
        Op::Rnn {
            hidden_size: cfg.h,
            num_layers: cfg.layers,
            bidirectional: cfg.bidir,
            carry: false,
            relu,
        },
        vec![xi, wih, whh, bias_n],
        Shape::new(&[cfg.b, cfg.s, cfg.d() * cfg.h], f),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let actual = c
        .run(&[("x", &x), ("w_ih", &w_ih), ("w_hh", &w_hh), ("bias", &bias)])
        .pop()
        .unwrap();
    check(&actual, &expected, "rnn");
}

#[test]
fn gru_single_layer() {
    run_gru(&Cfg {
        b: 2,
        s: 5,
        inp: 3,
        h: 4,
        layers: 1,
        bidir: false,
    });
}
#[test]
fn gru_multi_layer_bidirectional() {
    run_gru(&Cfg {
        b: 2,
        s: 4,
        inp: 3,
        h: 4,
        layers: 2,
        bidir: true,
    });
}
#[test]
fn rnn_tanh_single_layer() {
    run_rnn(
        &Cfg {
            b: 2,
            s: 5,
            inp: 3,
            h: 4,
            layers: 1,
            bidir: false,
        },
        false,
    );
}
#[test]
fn rnn_relu_multi_layer_bidirectional() {
    run_rnn(
        &Cfg {
            b: 2,
            s: 4,
            inp: 3,
            h: 4,
            layers: 2,
            bidir: true,
        },
        true,
    );
}

/// Regression guard for the class of bug that once masqueraded as a "CoreML GRU
/// bug": a mis-sized recurrent weight must fail LOUDLY at compile time (the
/// f32 kernels read `w_hh` through raw pointers, so an under-sized buffer would
/// otherwise read out of bounds and return silent garbage). Here `w_hh` is
/// under-sized by the `×hidden` factor — compiling must panic with a clear
/// message naming `w_hh`.
#[test]
#[should_panic(expected = "w_hh")]
fn undersized_w_hh_panics_loudly() {
    let f = DType::F32;
    let (b, s, inp, h) = (1usize, 3usize, 4usize, 4usize);
    let mut g = Graph::new("bad_gru");
    let x = g.input("x", Shape::new(&[b, s, inp], f));
    let wih = g.input("w_ih", Shape::new(&[3 * h * inp], f));
    // BUG: w_hh should be [3h·h]; supply [3h] (missing the ×h factor).
    let whh = g.input("w_hh", Shape::new(&[3 * h], f));
    let bih = g.input("b_ih", Shape::new(&[3 * h], f));
    let bhh = g.input("b_hh", Shape::new(&[3 * h], f));
    let y = g.add_node(
        Op::Gru {
            hidden_size: h,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        vec![x, wih, whh, bih, bhh],
        Shape::new(&[b, s, h], f),
    );
    g.set_outputs(vec![y]);
    // Panics inside compile_gru's `check_rnn_lens`.
    let _ = Session::new(Device::Cpu).compile(g);
}
