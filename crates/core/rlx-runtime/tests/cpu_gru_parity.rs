// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Gru` cross-backend parity. Validates the native CPU kernel and the
//! `unfuse` decomposition (the path MLX / CoreML / CUDA-host / ROCm / wgpu /
//! TPU and autodiff take) against an independent host reference, and runs the
//! same graph on every GPU backend available at build time (Metal, wgpu).
//!
//! Reference is PyTorch/ONNX-GRU semantics: per `(layer,dir)` packed weights,
//! gate order **r, z, n**, `linear_before_reset = 1`, separate `b_ih`/`b_hh`,
//! bidirectional output concatenated on the hidden axis (`[batch, seq, D*h]`).

use rlx_ir::*;
use rlx_runtime::{Device, Session};

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

#[derive(Clone, Copy)]
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
            .map(|l| self.d() * 3 * self.h * self.in_l(l))
            .sum()
    }
    fn whh_total(&self) -> usize {
        self.layers * self.d() * 3 * self.h * self.h
    }
    fn b_total(&self) -> usize {
        self.layers * self.d() * 3 * self.h
    }
    fn out_len(&self) -> usize {
        self.b * self.s * self.d() * self.h
    }
}

/// Independent host reference (gate order r, z, n; `n = tanh(xn + r·hnn)`).
fn reference_gru(
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

fn seq(n: usize, seed: usize, off: f32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * seed % 13) as f32 - off) * scale)
        .collect()
}

fn inputs(cfg: &Cfg) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        seq(cfg.b * cfg.s * cfg.inp, 7, 6.0, 0.1),
        seq(cfg.wih_total(), 5, 5.0, 0.04),
        seq(cfg.whh_total(), 3, 3.0, 0.04),
        seq(cfg.b_total(), 2, 2.0, 0.05),
        seq(cfg.b_total(), 4, 2.0, 0.05),
    )
}

fn build_gru_graph(cfg: &Cfg) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("gru");
    let x = g.input("x", Shape::new(&[cfg.b, cfg.s, cfg.inp], f));
    let wih = g.input("w_ih", Shape::new(&[cfg.wih_total()], f));
    let whh = g.input("w_hh", Shape::new(&[cfg.whh_total()], f));
    let bih = g.input("b_ih", Shape::new(&[cfg.b_total()], f));
    let bhh = g.input("b_hh", Shape::new(&[cfg.b_total()], f));
    let y = g.gru(
        x,
        wih,
        whh,
        bih,
        bhh,
        None,
        cfg.h,
        cfg.layers,
        cfg.bidir,
        Shape::new(&[cfg.b, cfg.s, cfg.d() * cfg.h], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn run_on(cfg: &Cfg, device: Device) -> Vec<f32> {
    let (x, wih, whh, bih, bhh) = inputs(cfg);
    let mut c = Session::new(device).compile(build_gru_graph(cfg));
    c.run(&[
        ("x", x.as_slice()),
        ("w_ih", wih.as_slice()),
        ("w_hh", whh.as_slice()),
        ("b_ih", bih.as_slice()),
        ("b_hh", bhh.as_slice()),
    ])
    .pop()
    .unwrap()
}

fn assert_close(what: &str, actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "{what} length");
    let mut max = 0f32;
    for i in 0..actual.len() {
        max = max.max((actual[i] - expected[i]).abs());
    }
    assert!(max <= tol, "{what}: max abs diff {max} > {tol}");
    eprintln!("{what}: max abs diff {max:.2e} (n={})", actual.len());
}

/// Configs covering the OCR shape (bidirectional) plus uni/multi-layer.
fn cfgs() -> Vec<(&'static str, Cfg)> {
    vec![
        (
            "uni-1L",
            Cfg {
                b: 2,
                s: 4,
                inp: 3,
                h: 5,
                layers: 1,
                bidir: false,
            },
        ),
        (
            "bi-1L",
            Cfg {
                b: 2,
                s: 5,
                inp: 4,
                h: 6,
                layers: 1,
                bidir: true,
            },
        ),
        (
            "bi-2L",
            Cfg {
                b: 2,
                s: 6,
                inp: 4,
                h: 8,
                layers: 2,
                bidir: true,
            },
        ),
        (
            "bi-1L-wide",
            Cfg {
                b: 1,
                s: 7,
                inp: 16,
                h: 64,
                layers: 1,
                bidir: true,
            },
        ),
    ]
}

#[test]
fn gru_cpu_native_matches_reference() {
    for (name, cfg) in cfgs() {
        let (x, wih, whh, bih, bhh) = inputs(&cfg);
        let expected = reference_gru(&cfg, &x, &wih, &whh, &bih, &bhh);
        assert_eq!(expected.len(), cfg.out_len());
        let actual = run_on(&cfg, Device::Cpu);
        assert_close(&format!("cpu {name}"), &actual, &expected, 1e-4);
    }
}

#[test]
fn gru_unfuse_decomposition_matches_reference() {
    // The decomposed graph is the path MLX / CoreML / CUDA-host / ROCm / wgpu /
    // TPU and autodiff take. It must reproduce the native kernel / reference.
    for (name, cfg) in cfgs() {
        let (x, wih, whh, bih, bhh) = inputs(&cfg);
        let expected = reference_gru(&cfg, &x, &wih, &whh, &bih, &bhh);
        let decomposed = rlx_fusion::unfuse::unfuse_fused_for_autodiff(build_gru_graph(&cfg));
        assert!(
            !decomposed
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Gru { .. })),
            "Op::Gru should be decomposed away by unfuse ({name})"
        );
        let mut c = Session::new(Device::Cpu).compile(decomposed);
        let actual = c
            .run(&[
                ("x", x.as_slice()),
                ("w_ih", wih.as_slice()),
                ("w_hh", whh.as_slice()),
                ("b_ih", bih.as_slice()),
                ("b_hh", bhh.as_slice()),
            ])
            .pop()
            .unwrap();
        assert_close(&format!("unfuse {name}"), &actual, &expected, 1e-4);
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn gru_metal_matches_cpu_and_reference() {
    for (name, cfg) in cfgs() {
        let (x, wih, whh, bih, bhh) = inputs(&cfg);
        let expected = reference_gru(&cfg, &x, &wih, &whh, &bih, &bhh);
        let metal = run_on(&cfg, Device::Metal);
        assert_close(&format!("metal {name} vs ref"), &metal, &expected, 1e-4);
        let cpu = run_on(&cfg, Device::Cpu);
        assert_close(&format!("metal {name} vs cpu"), &metal, &cpu, 1e-4);
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn gru_mlx_matches_reference() {
    for (name, cfg) in cfgs() {
        let (x, wih, whh, bih, bhh) = inputs(&cfg);
        let expected = reference_gru(&cfg, &x, &wih, &whh, &bih, &bhh);
        let mlx = run_on(&cfg, Device::Mlx);
        assert_close(&format!("mlx {name} vs ref"), &mlx, &expected, 1e-4);
    }
}

/// Throughput at OCR-recognition scale (seq≈50, hidden 256, bidirectional).
/// `cargo test -p rlx-runtime --features metal --test cpu_gru_parity gru_bench -- --ignored --nocapture`
#[test]
#[ignore = "benchmark, not a correctness test; run with --ignored --nocapture for timings"]
fn gru_bench() {
    use std::time::Instant;
    let cfg = Cfg {
        b: 1,
        s: 50,
        inp: 128,
        h: 256,
        layers: 1,
        bidir: true,
    };
    let (x, wih, whh, bih, bhh) = inputs(&cfg);
    #[allow(unused_mut)]
    let mut devices = vec![("cpu", Device::Cpu)];
    #[cfg(all(target_os = "macos", feature = "metal"))]
    devices.push(("metal", Device::Metal));
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    devices.push(("mlx", Device::Mlx));
    for (name, dev) in devices {
        let mut c = Session::new(dev).compile(build_gru_graph(&cfg));
        let inp = [
            ("x", x.as_slice()),
            ("w_ih", wih.as_slice()),
            ("w_hh", whh.as_slice()),
            ("b_ih", bih.as_slice()),
            ("b_hh", bhh.as_slice()),
        ];
        for _ in 0..3 {
            c.run(&inp);
        }
        let n = 50;
        let t = Instant::now();
        for _ in 0..n {
            c.run(&inp);
        }
        let per = t.elapsed() / n;
        eprintln!("[gru_bench] {name}: {per:?}/iter (s=50,h=256,bidir,1L)");
    }
}

#[test]
#[cfg(feature = "gpu")]
fn gru_wgpu_matches_reference() {
    for (name, cfg) in cfgs() {
        let (x, wih, whh, bih, bhh) = inputs(&cfg);
        let expected = reference_gru(&cfg, &x, &wih, &whh, &bih, &bhh);
        let gpu = run_on(&cfg, Device::Gpu);
        assert_close(&format!("wgpu {name} vs ref"), &gpu, &expected, 1e-4);
    }
}
