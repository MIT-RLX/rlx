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
//! `Op::Lstm` extended forms — multi-layer, bidirectional, and decode
//! carry — checked against an independent packed-weight reference, on the
//! native CPU kernel and the `unfuse` decomposition (MLX/CoreML/TPU path).

use rlx_ir::*;
use rlx_runtime::{Device, Session};

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

#[allow(clippy::too_many_arguments)]
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
    fn wih_total(&self) -> usize {
        (0..self.layers)
            .map(|l| self.d() * 4 * self.h * self.in_l(l))
            .sum()
    }
    fn whh_total(&self) -> usize {
        self.layers * self.d() * 4 * self.h * self.h
    }
    fn bias_total(&self) -> usize {
        self.layers * self.d() * 4 * self.h
    }
}

/// Packed-weight reference matching `execute_lstm_f32`. Returns `y`.
fn ref_lstm(
    cfg: &Cfg,
    x: &[f32],
    w_ih: &[f32],
    w_hh: &[f32],
    bias: &[f32],
    h0: &[f32],
    c0: &[f32],
    carry: bool,
) -> Vec<f32> {
    let (b, s, h, d) = (cfg.b, cfg.s, cfg.h, cfg.d());
    let four_h = 4 * h;
    let mut layer_in = x.to_vec();
    let mut in_l = cfg.inp;
    let mut wih_cursor = 0usize;
    for l in 0..cfg.layers {
        let out_width = d * h;
        let mut layer_out = vec![0f32; b * s * out_width];
        let wih_block = four_h * in_l;
        for dir in 0..d {
            let ld = l * d + dir;
            let wih = &w_ih[wih_cursor + dir * wih_block..][..wih_block];
            let whh = &w_hh[ld * four_h * h..][..four_h * h];
            let bs = &bias[ld * four_h..][..four_h];
            for bi in 0..b {
                let mut hh = vec![0f32; h];
                let mut cc = vec![0f32; h];
                if carry {
                    hh.copy_from_slice(&h0[(ld * b + bi) * h..][..h]);
                    cc.copy_from_slice(&c0[(ld * b + bi) * h..][..h]);
                }
                for step in 0..s {
                    let t = if dir == 0 { step } else { s - 1 - step };
                    let xt = &layer_in[(bi * s + t) * in_l..][..in_l];
                    let mut z = vec![0f32; four_h];
                    for (r, zr) in z.iter_mut().enumerate() {
                        let mut acc = bs[r];
                        for (j, &xj) in xt.iter().enumerate() {
                            acc += wih[r * in_l + j] * xj;
                        }
                        for (j, &hj) in hh.iter().enumerate() {
                            acc += whh[r * h + j] * hj;
                        }
                        *zr = acc;
                    }
                    for k in 0..h {
                        let i_g = sigmoid(z[k]);
                        let f_g = sigmoid(z[h + k]);
                        let g_g = z[2 * h + k].tanh();
                        let o_g = sigmoid(z[3 * h + k]);
                        let c_new = f_g * cc[k] + i_g * g_g;
                        cc[k] = c_new;
                        let h_new = o_g * c_new.tanh();
                        hh[k] = h_new;
                        layer_out[(bi * s + t) * out_width + dir * h + k] = h_new;
                    }
                }
            }
        }
        wih_cursor += d * wih_block;
        layer_in = layer_out;
        in_l = out_width;
    }
    layer_in
}

fn seq(n: usize, m: usize, off: f32, scale: f32) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 % m) as f32 - off) * scale).collect()
}

fn run_case(cfg: &Cfg, carry: bool, decompose: bool) {
    let d = cfg.d();
    let x = seq(cfg.b * cfg.s * cfg.inp, 13, 6.0, 0.08);
    let w_ih = seq(cfg.wih_total(), 11, 5.0, 0.04);
    let w_hh = seq(cfg.whh_total(), 7, 3.0, 0.04);
    let bias = seq(cfg.bias_total(), 5, 2.0, 0.04);
    let state_n = cfg.layers * d * cfg.b * cfg.h;
    let h0 = if carry {
        seq(state_n, 9, 4.0, 0.05)
    } else {
        vec![]
    };
    let c0 = if carry {
        seq(state_n, 6, 3.0, 0.05)
    } else {
        vec![]
    };

    let expected = ref_lstm(cfg, &x, &w_ih, &w_hh, &bias, &h0, &c0, carry);

    let f = DType::F32;
    let mut g = Graph::new("lstm_variant");
    let x_in = g.input("x", Shape::new(&[cfg.b, cfg.s, cfg.inp], f));
    let wih = g.input("w_ih", Shape::new(&[cfg.wih_total()], f));
    let whh = g.input("w_hh", Shape::new(&[cfg.whh_total()], f));
    let bs = g.input("bias", Shape::new(&[cfg.bias_total()], f));
    let out_shape = Shape::new(&[cfg.b, cfg.s, d * cfg.h], f);
    let mut ins = vec![x_in, wih, whh, bs];
    if carry {
        let h0n = g.input("h0", Shape::new(&[cfg.layers * d, cfg.b, cfg.h], f));
        let c0n = g.input("c0", Shape::new(&[cfg.layers * d, cfg.b, cfg.h], f));
        ins.push(h0n);
        ins.push(c0n);
    }
    let y = g.add_node(
        Op::Lstm {
            hidden_size: cfg.h,
            num_layers: cfg.layers,
            bidirectional: cfg.bidir,
            carry,
        },
        ins,
        out_shape,
    );
    g.set_outputs(vec![y]);

    let g = if decompose {
        rlx_fusion::unfuse::unfuse_fused_for_autodiff(g)
    } else {
        g
    };
    if decompose {
        assert!(
            !g.nodes().iter().any(|n| matches!(n.op, Op::Lstm { .. })),
            "Op::Lstm should decompose"
        );
    }

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let mut slots: Vec<(&str, &[f32])> = vec![
        ("x", x.as_slice()),
        ("w_ih", w_ih.as_slice()),
        ("w_hh", w_hh.as_slice()),
        ("bias", bias.as_slice()),
    ];
    if carry {
        slots.push(("h0", h0.as_slice()));
        slots.push(("c0", c0.as_slice()));
    }
    let actual = compiled.run(&slots).pop().unwrap();

    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        assert!(
            (actual[i] - expected[i]).abs() < 1e-4,
            "carry={carry} decompose={decompose} mismatch at {i}: {} vs {}",
            actual[i],
            expected[i]
        );
    }
}

#[test]
fn lstm_bidirectional() {
    let cfg = Cfg {
        b: 2,
        s: 5,
        inp: 3,
        h: 4,
        layers: 1,
        bidir: true,
    };
    run_case(&cfg, false, false); // native
    run_case(&cfg, false, true); // decomposed
}

#[test]
fn lstm_multi_layer() {
    let cfg = Cfg {
        b: 2,
        s: 5,
        inp: 3,
        h: 4,
        layers: 3,
        bidir: false,
    };
    run_case(&cfg, false, false);
    run_case(&cfg, false, true);
}

#[test]
fn lstm_multi_layer_bidirectional() {
    let cfg = Cfg {
        b: 2,
        s: 4,
        inp: 3,
        h: 4,
        layers: 2,
        bidir: true,
    };
    run_case(&cfg, false, false);
    run_case(&cfg, false, true);
}

#[test]
fn lstm_decode_carry() {
    // h0/c0 nonzero initial state must feed the recurrence; y reflects it.
    let cfg = Cfg {
        b: 2,
        s: 4,
        inp: 3,
        h: 4,
        layers: 2,
        bidir: false,
    };
    run_case(&cfg, true, false); // native carry
    run_case(&cfg, true, true); // decomposed carry (seeds h0/c0)
}
