// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Wave execution (`wave-exec`): independent branches of a graph run
//! concurrently. Checks that the plan finds the branches and that the
//! concurrent run matches an in-test reference.
#![cfg(feature = "wave-exec")]

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{WavePlan, compile_thunks, execute_thunks};
use rlx_ir::{DType, Graph, Op, Shape};

const M: usize = 4;
const K: usize = 64;
const BRANCHES: usize = 6;
const EPS: f32 = 1e-6;

fn fill(seed: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7919 + seed * 104_729) % 1000) as f32 / 500.0) - 1.0)
        .collect()
}

fn reference(x: &[f32], w: &[f32], g: &[f32], b: &[f32]) -> Vec<f32> {
    let mut h = vec![0f32; M * K];
    for r in 0..M {
        for c in 0..K {
            h[r * K + c] = (0..K).map(|i| x[r * K + i] * w[i * K + c]).sum();
        }
    }
    let mut out = vec![0f32; M * K];
    for r in 0..M {
        let row = &h[r * K..(r + 1) * K];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / K as f32;
        let inv = 1.0 / (ms + EPS).sqrt();
        for c in 0..K {
            out[r * K + c] = row[c] * inv * g[c] + b[c];
        }
    }
    out
}

#[test]
fn parallel_branches_match_reference() {
    let f32s = |d: &[usize]| Shape::new(d, DType::F32);
    let mut g = Graph::new("waves");
    let x = g.input("x", f32s(&[M, K]));
    let gamma = g.input("g", f32s(&[K]));
    let beta = g.input("b", f32s(&[K]));
    let mut outs = Vec::new();
    for i in 0..BRANCHES {
        let w = g.input(format!("w{i}"), f32s(&[K, K]));
        let h = g.matmul(x, w, f32s(&[M, K]));
        let n = g.add_node(
            Op::RmsNorm { axis: -1, eps: EPS },
            vec![h, gamma, beta],
            f32s(&[M, K]),
        );
        outs.push(n);
    }
    g.set_outputs(outs.clone());

    let mut feeds: Vec<(String, Vec<f32>)> = vec![
        ("x".into(), fill(1, M * K)),
        ("g".into(), fill(2, K)),
        ("b".into(), fill(3, K)),
    ];
    for i in 0..BRANCHES {
        feeds.push((format!("w{i}"), fill(10 + i, K * K)));
    }

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    assert!(
        WavePlan::build(&sched.thunks).parallel_thunks() >= BRANCHES,
        "independent branches were not grouped into a shared level"
    );

    for node in g.nodes() {
        if let Op::Input { name } = &node.op {
            let data = &feeds.iter().find(|(n, _)| n == name).unwrap().1;
            arena.slice_mut(node.id)[..data.len()].copy_from_slice(data);
        }
    }
    // Repeat so a race has more than one chance to show.
    for _ in 0..50 {
        execute_thunks(&sched, arena.raw_buf_mut());
    }

    let get = |n: &str| &feeds.iter().find(|(k, _)| k == n).unwrap().1;
    for (i, &o) in outs.iter().enumerate() {
        let want = reference(get("x"), get(&format!("w{i}")), get("g"), get("b"));
        let got = &arena.slice_mut(o)[..M * K];
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-4, "branch {i}: max abs err {err}");
    }
}
