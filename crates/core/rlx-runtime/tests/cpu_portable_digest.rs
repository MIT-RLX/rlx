// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPU backend's portable arithmetic is one function of the inputs on
//! every host: a corpus of graphs (GEMMs of every dispatch class, the
//! transcendental activations, softmax, the norms, reductions) hashes to the
//! same digest whether portable is compiled in (`--features rlx-cpu/portable`)
//! or switched on at run time (`RLX_CPU_PORTABLE=1` /
//! `rlx_cpu::config::set_portable`), on macOS, Linux and x86-64 alike.
//! (Own test binary: the switch is process-wide.)

use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use rlx_runtime::{Device, Session};

/// FNV-1a over every output's bits of every graph in the corpus — the same on
/// macOS arm64 and Linux x86-64, feature or run-time switch.
const PORTABLE_DIGEST: &str = "28c53cb6ef67b9c3";

fn data(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.618_034).fract() * 4.0 - 2.0)
        .collect()
}

fn corpus() -> Vec<(Graph, Vec<(&'static str, Vec<f32>)>)> {
    let f = DType::F32;
    let mut out = Vec::new();
    // GEMM: tiny m (the small-GEMM kernel), mid, large (BLAS / threaded), bias
    for (m, k, n) in [
        (1, 64, 96),
        (4, 33, 40),
        (8, 128, 128),
        (37, 70, 51),
        (256, 256, 256),
    ] {
        let mut g = Graph::new("mm");
        let a = g.input("a", Shape::new(&[m, k], f));
        let b = g.input("b", Shape::new(&[k, n], f));
        let bias = g.input("bias", Shape::new(&[n], f));
        let y = g.mm(a, b);
        let z = g.add(y, bias);
        let t = g.transpose_(a, vec![1, 0]);
        let u = g.mm(t, a);
        g.set_outputs(vec![y, z, u]);
        out.push((
            g,
            vec![
                ("a", data(m * k, 0.3)),
                ("b", data(k * n, 0.7)),
                ("bias", data(n, 0.1)),
            ],
        ));
    }
    // attention (the SDPA dot / weighted-sum kernels), 2 heads of 16
    let (s_len, heads, hd) = (9, 2, 16);
    let mut g = Graph::new("attn");
    let sh = Shape::new(&[1, s_len, heads * hd], f);
    let q = g.input("q", sh.clone());
    let k = g.input("k", sh.clone());
    let v = g.input("v", sh);
    let mask = g.input("m", Shape::new(&[1, s_len], f));
    let a = g.attention_(q, k, v, mask, heads, hd);
    g.set_outputs(vec![a]);
    let n = s_len * heads * hd;
    out.push((
        g,
        vec![
            ("q", data(n, 0.11)),
            ("k", data(n, 0.23)),
            ("v", data(n, 0.37)),
            ("m", vec![0.0; s_len]),
        ],
    ));
    // elementwise transcendentals, softmax, norms, reductions
    let (r, c) = (13, 300);
    let mut g = Graph::new("ew");
    let x = g.input("x", Shape::new(&[r, c], f));
    let gamma = g.input("g", Shape::new(&[c], f));
    let beta = g.input("be", Shape::new(&[c], f));
    let ax = g.abs(x);
    let outs: Vec<NodeId> = vec![
        g.gelu(x),
        g.gelu_approx(x),
        g.silu(x),
        g.tanh(x),
        g.sigmoid(x),
        g.exp(x),
        g.log(ax),
        g.sqrt(ax),
        g.rsqrt(ax),
        g.recip(x),
        g.sm(x, -1),
        g.rms_norm(x, gamma, beta, 1e-6),
        g.sum(x, vec![1], false),
        g.mean(x, vec![0], true),
    ];
    g.set_outputs(outs);
    out.push((
        g,
        vec![
            ("x", data(r * c, 0.9)),
            ("g", data(c, 0.2)),
            ("be", data(c, 0.5)),
        ],
    ));
    out
}

fn digest() -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for (gi, (g, inputs)) in corpus().into_iter().enumerate() {
        let mut c = Session::new(Device::Cpu).compile(g);
        c.finalize_params();
        let ins: Vec<(&str, &[f32])> = inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        for (oi, o) in c.run(&ins).into_iter().enumerate() {
            let mut ho = 0xcbf2_9ce4_8422_2325u64;
            for v in o {
                for b in v.to_bits().to_le_bytes() {
                    h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
                    ho = (ho ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
                }
            }
            // per output, to find what moved
            eprintln!("graph {gi} output {oi}: {ho:016x}");
        }
    }
    format!("{h:016x}")
}

#[test]
fn portable_cpu_arithmetic_is_one_digest() {
    if !cfg!(feature = "cpu") {
        return;
    }
    rlx_cpu::config::set_portable(true).expect("the portable mode is still open");
    let d = digest();
    eprintln!("portable digest {d}");
    assert_eq!(d, PORTABLE_DIGEST, "the portable CPU arithmetic moved");
}
