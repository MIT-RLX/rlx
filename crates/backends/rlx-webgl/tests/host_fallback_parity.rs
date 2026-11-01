// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Ops with no WebGL lowering run on rlx-cpu through `Step::Host`: sampling,
// quantized I/O and matmuls, SVD/QR/eigen, registered custom ops, complex
// helpers, log-mel. Each is checked against RLX's CPU backend (the same
// kernels, so the bound is tight), and the plan is checked to use the host
// step.

use std::sync::Arc;

use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};
use rlx_ir::op::{QrPart, SvdPart};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, OpExtension, Shape, register_op};

fn f32s(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(f).collect()
}
fn sh(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

/// CPU reference (typed I/O: every input encoded as its declared dtype, every
/// output decoded by its dtype) vs the WebGL plan, which must use a host step.
fn parity(g: Graph, inputs: &[(&str, &[f32])]) {
    let dtype_of = |name: &str| {
        g.nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name: m } | Op::Param { name: m } if m == name))
            .map(|n| n.shape.dtype())
            .expect("input")
    };
    let enc: Vec<(&str, Vec<u8>, DType)> = inputs
        .iter()
        .map(|&(n, v)| {
            let dt = dtype_of(n);
            (n, encode(v, dt), dt)
        })
        .collect();
    let is_param = |name: &str| {
        g.nodes()
            .iter()
            .any(|n| matches!(&n.op, Op::Param { name: m } if m == name))
    };
    let mut sess = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g.clone());
    for (n, b, d) in enc.iter().filter(|(n, ..)| is_param(n)) {
        sess.set_param_typed(n, b, *d);
    }
    let refs: Vec<(&str, &[u8], DType)> = enc
        .iter()
        .filter(|(n, ..)| !is_param(n))
        .map(|(n, b, d)| (*n, b.as_slice(), *d))
        .collect();
    // `run_typed` may hand back a sub-4-byte output in a 4-byte-per-element
    // buffer (values first); keep the output's own element count.
    let reference: Vec<Vec<f32>> = sess
        .run_typed(&refs)
        .iter()
        .zip(&g.outputs)
        .map(|((b, dt), &id)| {
            let mut v = decode(b, *dt);
            v.truncate(g.node(id).shape.num_elements().unwrap_or(0));
            v
        })
        .collect();
    check(&g, inputs, &reference);
}

fn encode(v: &[f32], dt: DType) -> Vec<u8> {
    match dt {
        DType::F64 => v.iter().flat_map(|&x| (x as f64).to_le_bytes()).collect(),
        DType::I8 => v.iter().map(|&x| x as i8 as u8).collect(),
        DType::I32 => v.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect(),
        _ => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
    }
}

fn decode(b: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F64 => b
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        DType::I8 => b.iter().map(|&x| x as i8 as f32).collect(),
        DType::U8 => b.iter().map(|&x| x as f32).collect(),
        DType::I32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c) as f32)
            .collect(),
        DType::I64 => b
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c) as f32)
            .collect(),
        _ => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
    }
}

fn check(g: &Graph, inputs: &[(&str, &[f32])], reference: &[Vec<f32>]) {
    let plan = rlx_webgl::build_plan(g).expect("plan");
    assert!(
        plan.steps
            .iter()
            .any(|s| matches!(s, rlx_webgl::Step::Host { .. })),
        "expected a host step"
    );
    let got = rlx_webgl::run_cpu(&plan, inputs).expect("run_cpu");
    assert_eq!(reference.len(), got.len(), "output count");
    for (oi, (r, gv)) in reference.iter().zip(&got).enumerate() {
        assert_eq!(r.len(), gv.len(), "output {oi} length");
        for (i, (a, b)) in r.iter().zip(gv).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * (1.0 + a.abs()) || (a.is_nan() && b.is_nan()),
                "output {oi}[{i}]: reference={a} webgl={b}"
            );
        }
    }
}

#[test]
fn sample_seeded() {
    let logits = f32s(2 * 10, |i| (i as f32 * 0.7).sin() * 3.0);
    let mut g = Graph::new("sample");
    let l = g.input("l", sh(&[2, 10]));
    let y = g.sample(l, 5, 0.9, 0.8, 7, sh(&[2]));
    g.set_outputs(vec![y]);
    parity(g, &[("l", &logits)]);
}

#[test]
fn quantize_dequantize() {
    let xs = f32s(4 * 6, |i| (i as f32 * 0.37).sin() * 2.0);
    let mut g = Graph::new("q");
    let x = g.input("x", sh(&[4, 6]));
    let q = g.quantize(x, 0.02, 3);
    let dq = g.dequantize(q, 0.02, 3);
    let qc = g.quantize_per_channel(
        x,
        1,
        f32s(6, |i| 0.01 + 0.005 * i as f32),
        vec![0, 1, -1, 2, 0, 0],
    );
    g.set_outputs(vec![q, dq, qc]);
    parity(g, &[("x", &xs)]);
}

#[test]
fn qmatmul_int8() {
    let (m, k, n) = (3usize, 8usize, 5usize);
    let xq = f32s(m * k, |i| ((i * 13 % 51) as i32 - 25) as f32);
    let wq = f32s(k * n, |i| ((i * 29 % 61) as i32 - 30) as f32);
    let bias = f32s(n, |i| (i as i32 * 40 - 80) as f32);
    let mut g = Graph::new("qmm");
    let x = g.input("x", Shape::new(&[m, k], DType::I8));
    let w = g.input("w", Shape::new(&[k, n], DType::I8));
    let b = g.input("b", Shape::new(&[n], DType::I32));
    let y = g.q_matmul(x, w, b, 1, -2, 3, 0.0125, Shape::new(&[m, n], DType::I8));
    g.set_outputs(vec![y]);

    parity(g, &[("x", &xq), ("w", &wq), ("b", &bias)]);
}

#[test]
fn dequant_matmul_int8_block() {
    let (m, k, n, block) = (2usize, 64usize, 8usize, 32usize);
    let nb = k / block;
    let x = f32s(m * k, |i| (i as f32 * 0.017).sin());
    let w = f32s(k * n, |i| (((i * 37 + 11) % 201) as i32 - 100) as f32);
    let scale = f32s(nb * n, |i| 0.02 + 0.001 * (i % 7) as f32);
    let zp = vec![0.0f32; nb * n];
    let mut g = Graph::new("dq");
    let xi = g.input("x", sh(&[m, k]));
    let wi = g.param("w", Shape::new(&[k * n], DType::I8));
    let si = g.param("scale", sh(&[nb, n]));
    let zi = g.param("zp", sh(&[nb, n]));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::Int8Block {
                block_size: block as u32,
            },
        },
        vec![xi, wi, si, zi],
        sh(&[m, n]),
    );
    g.set_outputs(vec![y]);
    parity(g, &[("x", &x), ("w", &w), ("scale", &scale), ("zp", &zp)]);
}

#[test]
fn svd_qr_eigh() {
    let (m, n) = (5usize, 3usize);
    let a = f32s(m * n, |i| {
        ((i * 7 % 11) as f32 - 5.0) * 0.3 + if i % 4 == 0 { 1.0 } else { 0.0 }
    });
    let s = f32s(4 * 4, |i| {
        let (r, c) = (i / 4, i % 4);
        1.0 / (1.0 + (r + c) as f32) + if r == c { 2.0 + r as f32 } else { 0.0 }
    });
    let mut g = Graph::new("dec");
    let ai = g.input("a", sh(&[m, n]));
    // `eigh` (rlx-cpu's `core.eigh`) works in f64.
    let si = g.input("s", Shape::new(&[4, 4], DType::F64));
    let u = g.svd(ai, SvdPart::U, sh(&[m, n]));
    let sv = g.svd(ai, SvdPart::S, sh(&[n]));
    let vt = g.svd(ai, SvdPart::Vt, sh(&[n, n]));
    let q = g.qr(ai, QrPart::Q, sh(&[m, n]));
    let r = g.qr(ai, QrPart::R, sh(&[n, n]));
    let (w, v) = g.eigh(si);
    g.set_outputs(vec![u, sv, vt, q, r, w, v]);
    parity(g, &[("a", &a), ("s", &s)]);
}

#[test]
fn complex_helpers() {
    let xs = f32s(2 * 6, |i| (i as f32 * 0.9).cos());
    let mut g = Graph::new("cplx");
    let x = g.input("x", sh(&[2, 6]));
    let c = g.add_node(
        Op::Cast { to: DType::C64 },
        vec![x],
        Shape::new(&[2, 6], DType::C64),
    );
    let f = g.fft(c, false);
    let conj = g.conjugate(f);
    let im = g.add_node(Op::Cast { to: DType::F32 }, vec![conj], sh(&[2, 6]));
    let nsq = g.complex_norm_sq(f);
    g.set_outputs(vec![im, nsq]);
    parity(g, &[("x", &xs)]);
}

#[test]
fn log_mel() {
    // n_fft = 8: spectrum [.., 2·8] (re plane | im plane), filters [n_mels, 8/2 + 1].
    let spec = f32s(3 * 16, |i| 0.1 + (i as f32 * 0.3).sin());
    let filt = f32s(4 * 5, |i| ((i % 5) as f32 * 0.1).max(0.05));
    let mut g = Graph::new("logmel");
    let s = g.input("s", sh(&[3, 16]));
    let f = g.input("f", sh(&[4, 5]));
    let y = g.log_mel(s, f);
    g.set_outputs(vec![y]);
    parity(g, &[("s", &spec), ("f", &filt)]);
}

// ── A registered (non-collective) custom op ───────────────────────────────

struct CubeIr;
impl OpExtension for CubeIr {
    fn name(&self) -> &str {
        "rlx_webgl_test.cube"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}

struct CubeCpu;
impl CpuKernel for CubeCpu {
    fn name(&self) -> &str {
        "rlx_webgl_test.cube"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f32("cube input")?;
        let out = output.expect_f32_mut("cube output")?;
        for (o, v) in out.iter_mut().zip(x) {
            *o = v * v * v;
        }
        Ok(())
    }
}

#[test]
fn registered_custom_op() {
    register_op(Arc::new(CubeIr));
    register_cpu_kernel(Arc::new(CubeCpu));
    let xs = f32s(7, |i| i as f32 * 0.5 - 1.5);
    let mut g = Graph::new("custom");
    let x = g.input("x", sh(&[7]));
    let y = g.custom_op("rlx_webgl_test.cube", vec![], vec![x]);
    let z = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Neg),
        vec![y],
        sh(&[7]),
    );
    g.set_outputs(vec![z]);
    parity(g, &[("x", &xs)]);
}
