// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `NumberFormat::F64` on an f32 graph: a transformer block (RmsNorm, Rope,
//! grouped-query causal Attention, SiLU MLP, Softmax) built in f32 and run in
//! f64 on the CPU, against a host f64 reference written out here.
//!
//! The f32 run of the same graph is the control: it must miss the reference
//! by f32 rounding, so a pass that silently left the graph in f32 fails.

#![cfg(feature = "cpu")]

mod common;

use rlx_ir::op::{Activation, BinaryOp, MaskKind, RopeStyle};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompileOptions, Device, NumberFormat, Session};

const S: usize = 6;
const D: usize = 16;
const NH: usize = 4;
const NKV: usize = 2;
const DH: usize = 4;
const KV: usize = NKV * DH;
const FF: usize = 24;
const EPS: f32 = 1e-6;

fn sh(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn block() -> Graph {
    let mut g = Graph::new("f64_block");
    let x = g.input("x", sh(&[1, S, D]));
    let p = |g: &mut Graph, n: &str, d: &[usize]| g.param(n, sh(d));
    let (gam, bet) = (p(&mut g, "g", &[D]), p(&mut g, "b", &[D]));
    let (wq, wk, wv, wo) = (
        p(&mut g, "wq", &[D, D]),
        p(&mut g, "wk", &[D, KV]),
        p(&mut g, "wv", &[D, KV]),
        p(&mut g, "wo", &[D, D]),
    );
    let (wg, wu, wd) = (
        p(&mut g, "wg", &[D, FF]),
        p(&mut g, "wu", &[D, FF]),
        p(&mut g, "wd", &[FF, D]),
    );
    let (cos, sin) = (
        p(&mut g, "cos", &[S, DH / 2]),
        p(&mut g, "sin", &[S, DH / 2]),
    );
    let n = g.add_node(
        Op::RmsNorm { axis: -1, eps: EPS },
        vec![x, gam, bet],
        sh(&[1, S, D]),
    );
    let q = g.matmul(n, wq, sh(&[1, S, D]));
    let k = g.matmul(n, wk, sh(&[1, S, KV]));
    let v = g.matmul(n, wv, sh(&[1, S, KV]));
    let rope = Op::Rope {
        head_dim: DH,
        n_rot: DH,
        style: RopeStyle::NeoX,
    };
    let q = g.add_node(rope.clone(), vec![q, cos, sin], sh(&[1, S, D]));
    let k = g.add_node(rope, vec![k, cos, sin], sh(&[1, S, KV]));
    let a = g.add_node(
        Op::Attention {
            num_heads: NH,
            head_dim: DH,
            v_head_dim: None,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        sh(&[1, S, D]),
    );
    let o = g.matmul(a, wo, sh(&[1, S, D]));
    let h = g.binary(BinaryOp::Add, o, x, sh(&[1, S, D]));
    let gate = g.matmul(h, wg, sh(&[1, S, FF]));
    let gate = g.add_node(
        Op::Activation(Activation::Silu),
        vec![gate],
        sh(&[1, S, FF]),
    );
    let up = g.matmul(h, wu, sh(&[1, S, FF]));
    let m = g.binary(BinaryOp::Mul, gate, up, sh(&[1, S, FF]));
    let dn = g.matmul(m, wd, sh(&[1, S, D]));
    let y = g.binary(BinaryOp::Add, dn, h, sh(&[1, S, D]));
    let sm = g.add_node(Op::Softmax { axis: -1 }, vec![y], sh(&[1, S, D]));
    g.set_outputs(vec![y, sm]);
    g
}

struct Weights(Vec<(&'static str, Vec<f32>)>);

impl Weights {
    fn get(&self, n: &str) -> Vec<f64> {
        let v = &self.0.iter().find(|(k, _)| *k == n).unwrap().1;
        v.iter().map(|&x| x as f64).collect()
    }
}

fn weights() -> (Vec<f32>, Weights) {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut r = |scale: f32| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
    };
    let mut w = Vec::new();
    w.push(("g", (0..D).map(|_| 1.0 + r(0.2)).collect()));
    w.push(("b", (0..D).map(|_| r(0.1)).collect()));
    for (n, len) in [
        ("wq", D * D),
        ("wk", D * KV),
        ("wv", D * KV),
        ("wo", D * D),
        ("wg", D * FF),
        ("wu", D * FF),
        ("wd", FF * D),
    ] {
        w.push((n, (0..len).map(|_| r(0.5)).collect()));
    }
    let (mut c, mut si) = (Vec::new(), Vec::new());
    for p in 0..S {
        for i in 0..DH / 2 {
            let a = p as f32 / 100f32.powf(2.0 * i as f32 / DH as f32);
            c.push(a.cos());
            si.push(a.sin());
        }
    }
    w.push(("cos", c));
    w.push(("sin", si));
    let x = (0..S * D).map(|_| r(2.0)).collect();
    (x, Weights(w))
}

fn mm(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut o = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            o[i * n + j] = (0..k).map(|t| a[i * k + t] * b[t * n + j]).sum();
        }
    }
    o
}

fn softmax_rows(v: &mut [f64], n: usize) {
    for r in v.chunks_exact_mut(n) {
        let m = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let s: f64 = r
            .iter_mut()
            .map(|x| {
                *x = (*x - m).exp();
                *x
            })
            .sum();
        r.iter_mut().for_each(|x| *x /= s);
    }
}

fn rope(v: &mut [f64], width: usize, cos: &[f64], sin: &[f64]) {
    let half = DH / 2;
    for p in 0..S {
        for h in 0..width / DH {
            let at = p * width + h * DH;
            for i in 0..half {
                let (a, b) = (v[at + i], v[at + half + i]);
                let (c, s) = (cos[p * half + i], sin[p * half + i]);
                v[at + i] = a * c - b * s;
                v[at + half + i] = b * c + a * s;
            }
        }
    }
}

/// The block in f64, written out directly.
fn reference(x: &[f32], w: &Weights) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let (g, b) = (w.get("g"), w.get("b"));
    let mut n = vec![0.0; S * D];
    for p in 0..S {
        let r = &x[p * D..][..D];
        let ms = r.iter().map(|v| v * v).sum::<f64>() / D as f64;
        let inv = 1.0 / (ms + EPS as f64).sqrt();
        for i in 0..D {
            n[p * D + i] = r[i] * inv * g[i] + b[i];
        }
    }
    let mut q = mm(&n, &w.get("wq"), S, D, D);
    let mut k = mm(&n, &w.get("wk"), S, D, KV);
    let v = mm(&n, &w.get("wv"), S, D, KV);
    let (cos, sin) = (w.get("cos"), w.get("sin"));
    rope(&mut q, D, &cos, &sin);
    rope(&mut k, KV, &cos, &sin);
    let mut a = vec![0.0; S * D];
    let scale = (DH as f64).powf(-0.5);
    for h in 0..NH {
        let kh = h / (NH / NKV);
        for qi in 0..S {
            let mut sc: Vec<f64> = (0..=qi)
                .map(|ki| {
                    (0..DH)
                        .map(|d| q[qi * D + h * DH + d] * k[ki * KV + kh * DH + d])
                        .sum::<f64>()
                        * scale
                })
                .collect();
            softmax_rows(&mut sc, qi + 1);
            for d in 0..DH {
                a[qi * D + h * DH + d] = (0..=qi).map(|ki| sc[ki] * v[ki * KV + kh * DH + d]).sum();
            }
        }
    }
    let o = mm(&a, &w.get("wo"), S, D, D);
    let h: Vec<f64> = o.iter().zip(&x).map(|(a, b)| a + b).collect();
    let gate = mm(&h, &w.get("wg"), S, D, FF);
    let up = mm(&h, &w.get("wu"), S, D, FF);
    let m: Vec<f64> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    let dn = mm(&m, &w.get("wd"), S, FF, D);
    let y: Vec<f64> = dn.iter().zip(&h).map(|(a, b)| a + b).collect();
    let mut sm = y.clone();
    softmax_rows(&mut sm, D);
    (y, sm)
}

fn run(opts: &CompileOptions, x: &[f32], w: &Weights) -> Vec<Vec<f32>> {
    let mut cg = Session::new(Device::Cpu).compile_with(block(), opts);
    for (n, v) in &w.0 {
        cg.set_param(n, v);
    }
    cg.run(&[("x", x)])
}

fn max_rel(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .map(|(g, w)| (*g as f64 - w).abs() / scale)
        .fold(0.0, f64::max)
}

#[test]
fn f64_format_runs_an_f32_block_in_f64() {
    let (x, w) = weights();
    let (y, sm) = reference(&x, &w);
    // Outputs come back as f32, so compare against the reference rounded to
    // f32: an f64 computation lands on it within an ulp, an f32 one does not.
    let f64_out = run(
        &CompileOptions::new().number_format(NumberFormat::F64),
        &x,
        &w,
    );
    let f32_out = run(&CompileOptions::new(), &x, &w);
    let rounded = |v: &[f64]| v.iter().map(|&x| x as f32 as f64).collect::<Vec<_>>();
    let (e64, e32) = (
        max_rel(&f64_out[0], &rounded(&y)),
        max_rel(&f32_out[0], &rounded(&y)),
    );
    let (s64, s32) = (
        max_rel(&f64_out[1], &rounded(&sm)),
        max_rel(&f32_out[1], &rounded(&sm)),
    );
    eprintln!("block: f64 {e64:.2e}  f32 {e32:.2e} | softmax: f64 {s64:.2e}  f32 {s32:.2e}");
    assert!(e64 < 1e-7 && s64 < 1e-7, "f64 run off the f64 reference");
    assert!(
        e32 > 4.0 * e64.max(1e-9),
        "f32 control matched too: was the graph promoted?"
    );
}

/// The promoted block read before its casts back to f32, so the kernels are
/// checked at f64 resolution rather than through f32 outputs.
#[test]
fn f64_kernels_match_the_reference_to_f64_rounding() {
    use rlx_opt::rlx_compile::promote_f64::promote_to_f64;
    let (x, w) = weights();
    let (y, sm) = reference(&x, &w);
    let mut g = promote_to_f64(&block()).graph;
    let outs: Vec<_> = g.outputs.iter().map(|&o| g.node(o).inputs[0]).collect();
    g.set_outputs(outs);
    let mut opts = CompileOptions::new().number_format(NumberFormat::F64);
    opts.fusion_opts.skip_fusion = true;
    let mut cg = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, v) in &w.0 {
        let wide: Vec<u8> = v.iter().flat_map(|&x| (x as f64).to_le_bytes()).collect();
        cg.set_param_typed(n, &wide, DType::F64);
    }
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = cg.run_typed(&[("x", &xb, DType::F32)]);
    let f64s = |b: &[u8]| -> Vec<f64> {
        b.as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect()
    };
    for ((bytes, dt), want, name) in [(&out[0], &y, "block"), (&out[1], &sm, "softmax")] {
        assert_eq!(*dt, DType::F64);
        let got = f64s(bytes);
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        let err = got
            .iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs() / scale)
            .fold(0.0, f64::max);
        eprintln!("{name}: max rel {err:.2e}");
        assert!(err < 1e-13, "{name}: {err:.2e}");
    }
}

/// The block in double-single: forced `f64x2` on the CPU (the reference df
/// kernels) and `f64` on every GPU backend built in, where norms, softmax,
/// SiLU, RoPE and attention are decomposed into df primitives. Read at f64
/// resolution, as above.
#[test]
fn double_single_runs_the_block_on_every_backend() {
    use rlx_opt::rlx_compile::promote_f64::promote_to_f64;
    let (x, w) = weights();
    let (y, sm) = reference(&x, &w);
    // Pushed per GPU feature; mutable only when one is on.
    #[allow(unused_mut)]
    let mut runs = vec![(Device::Cpu, NumberFormat::F64x2)];
    #[cfg(feature = "metal")]
    runs.push((Device::Metal, NumberFormat::F64));
    #[cfg(feature = "gpu")]
    runs.push((Device::Gpu, NumberFormat::F64));
    #[cfg(feature = "mlx")]
    runs.push((Device::Mlx, NumberFormat::F64));
    #[cfg(feature = "cuda")]
    runs.push((Device::Cuda, NumberFormat::F64));
    #[cfg(feature = "vulkan")]
    runs.push((Device::Vulkan, NumberFormat::F64));
    #[cfg(feature = "rocm")]
    runs.push((Device::Rocm, NumberFormat::F64));
    let mut bad = Vec::new();
    for (device, format) in runs {
        if common::skip_unless(device) {
            continue;
        }
        let _gpu = common::GpuTestGuard::acquire(device);
        let mut g = promote_to_f64(&block()).graph;
        let outs: Vec<_> = g.outputs.iter().map(|&o| g.node(o).inputs[0]).collect();
        g.set_outputs(outs);
        let mut opts = CompileOptions::new().number_format(format);
        opts.fusion_opts.skip_fusion = true;
        let mut cg = Session::new(device).compile_with(g, &opts);
        for (n, v) in &w.0 {
            let wide: Vec<u8> = v.iter().flat_map(|&x| (x as f64).to_le_bytes()).collect();
            cg.set_param_typed(n, &wide, DType::F64);
        }
        let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = cg.run_typed(&[("x", &xb, DType::F32)]);
        for ((bytes, dt), want, name) in [(&out[0], &y, "block"), (&out[1], &sm, "softmax")] {
            assert_eq!(*dt, DType::F64, "{device:?} {name}");
            let got: Vec<f64> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect();
            let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
            let err = got
                .iter()
                .zip(want)
                .map(|(g, w)| (g - w).abs() / scale)
                .fold(0.0, f64::max);
            eprintln!("{device:?} {format}: {name} max rel {err:.2e}");
            if err.is_nan() || err >= 1e-11 {
                bad.push(format!("{device:?} {name}: {err:.2e}"));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "double-single off the f64 reference: {bad:?}"
    );
}

#[test]
fn f64_mean_reduction_runs_natively_on_the_cpu() {
    // A mean whose f32 sum loses digits: f32 spacing at 1e8 is 8, so
    // 1e8 + 4 + 4 + 4 stays 1e8 in f32 (mean 2.5e7) but is 1e8 + 12 in f64
    // (mean 25000003, which narrows to 25000004 ≠ 2.5e7).
    let mut g = Graph::new("f64_mean");
    let x = g.input("x", sh(&[2, 4]));
    let m = g.add_node(
        Op::Reduce {
            op: rlx_ir::op::ReduceOp::Mean,
            axes: vec![1],
            keep_dim: false,
        },
        vec![x],
        sh(&[2]),
    );
    g.set_outputs(vec![m]);
    let opts = CompileOptions {
        number_format: Some(NumberFormat::F64),
        ..Default::default()
    };
    let mut c = Session::new(Device::Cpu).compile_with(g, &opts);
    let xs = [1e8f32, 4.0, 4.0, 4.0, 2.0, 4.0, 6.0, 8.0];
    let out = c.run(&[("x", &xs[..])]).remove(0);
    assert_eq!(out, vec![25_000_004.0, 5.0]);
    assert_ne!(out[0], 2.5e7, "an f32 sum would give 2.5e7");
}

#[test]
fn softmax_cross_entropy_in_double_single_matches_f64() {
    // Logits spread over a wide range so the f32 log seed alone is visibly
    // short of f64; one Newton step must close the gap.
    let (n, c) = (3usize, 5usize);
    let z: Vec<f32> = (0..n * c)
        .map(|i| ((i * 37 % 11) as f32 - 5.0) * 1.7)
        .collect();
    let labels = [2.0f32, 0.0, 4.0];
    let dl = [1.0f32, 0.5, 2.0];
    let mut g = Graph::new("sce_df");
    let zi = g.input("z", sh(&[n, c]));
    let yi = g.input("y", sh(&[n]));
    let di = g.input("dl", sh(&[n]));
    let l = g.add_node(Op::SoftmaxCrossEntropyWithLogits, vec![zi, yi], sh(&[n]));
    let dz = g.add_node(
        Op::SoftmaxCrossEntropyBackward,
        vec![zi, yi, di],
        sh(&[n, c]),
    );
    g.set_outputs(vec![l, dz]);
    let opts = CompileOptions {
        number_format: Some(NumberFormat::F64x2),
        ..Default::default()
    };
    let mut cg = Session::new(Device::Cpu).compile_with(g, &opts);
    let out = cg.run(&[("z", &z[..]), ("y", &labels[..]), ("dl", &dl[..])]);
    // f64 host reference.
    for r in 0..n {
        let row: Vec<f64> = z[r * c..(r + 1) * c].iter().map(|&v| v as f64).collect();
        let m = row.iter().cloned().fold(f64::MIN, f64::max);
        let s: f64 = row.iter().map(|v| (v - m).exp()).sum();
        let lab = labels[r] as usize;
        let loss = m + s.ln() - row[lab];
        assert!(
            ((out[0][r] as f64) - loss).abs() <= 1e-6 * loss.abs().max(1.0),
            "loss row {r}: {} vs {loss}",
            out[0][r]
        );
        for k in 0..c {
            let p = (row[k] - m).exp() / s;
            let want = (p - f64::from(k == lab)) * dl[r] as f64;
            let got = out[1][r * c + k] as f64;
            assert!((got - want).abs() <= 1e-6, "dz[{r},{k}]: {got} vs {want}");
        }
    }
}
