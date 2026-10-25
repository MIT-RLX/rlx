// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// RLX — combined / split graphs compute what the originals do (CPU).

#![cfg(feature = "cpu")]

use std::collections::HashMap;

use rlx_ir::compose::{parallel, sequential};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape, Splice};
use rlx_runtime::{Device, Session};

mod common;

const B: usize = 2;
const D: usize = 8;

fn f32s(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

/// `y = tanh(x @ w + b)`.
fn dense() -> Graph {
    let mut g = Graph::new("dense");
    let x = g.input("x", f32s(&[B, D]));
    let w = g.param("w", f32s(&[D, D]));
    let b = g.param("b", f32s(&[D]));
    let mm = g.matmul(x, w, f32s(&[B, D]));
    let y = g.add(mm, b);
    let y = g.tanh(y);
    g.set_outputs(vec![y]);
    g
}

/// Three dense layers with their own weights, plus a side output `exp(x)`
/// that shares nothing with the stack but its input.
fn model() -> Graph {
    let mut g = Graph::new("model");
    let mut h = None;
    for l in 0..3 {
        let mut splice = Splice::new().prefix(format!("l{l}."));
        if let Some(h) = h {
            splice = splice.bind("x", h);
        }
        h = Some(g.splice(&dense(), &splice).unwrap().outputs[0]);
    }
    let x = g.input_id("l0.x").unwrap();
    let side = g.exp(x);
    g.set_outputs(vec![h.unwrap(), side]);
    g
}

fn values(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 1000) as f32 / 1000.0 - 0.5)
        .collect()
}

/// Every leaf any of these graphs read, with deterministic data.
fn feeds() -> HashMap<String, Vec<f32>> {
    let mut f = HashMap::new();
    for (k, l) in (0..3).enumerate() {
        f.insert(format!("l{l}.w"), values(D * D, 11 + k as u32));
        f.insert(format!("l{l}.b"), values(D, 23 + k as u32));
    }
    f.insert("w".into(), values(D * D, 11));
    f.insert("b".into(), values(D, 23));
    f.insert("l0.x".into(), values(B * D, 5));
    f.insert("x".into(), values(B * D, 5));
    f
}

fn run(g: &Graph, feeds: &HashMap<String, Vec<f32>>) -> Vec<Vec<f32>> {
    let mut inputs = Vec::new();
    let mut params = Vec::new();
    for n in g.nodes() {
        match &n.op {
            Op::Input { name } => inputs.push(name.clone()),
            Op::Param { name } => params.push(name.clone()),
            _ => {}
        }
    }
    let mut c = Session::new(Device::Cpu).compile(g.clone());
    for p in &params {
        c.set_param(p, &feeds[p]);
    }
    let named: Vec<(&str, &[f32])> = inputs
        .iter()
        .map(|n| (n.as_str(), feeds[n].as_slice()))
        .collect();
    c.run(&named)
}

fn assert_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= 1e-5, "[{i}] {x} vs {y}");
    }
}

#[test]
fn sequential_matches_running_each_graph() {
    let _gpu = common::serialize_gpu();
    let f = feeds();
    // Shared names: both layers read the same `w`/`b` (tied weights).
    let chained = sequential("two", &[&dense(), &dense()]).unwrap();
    let out = run(&chained, &f);

    let first = run(&dense(), &f);
    let mut f2 = f.clone();
    f2.insert("x".into(), first[0].clone());
    let second = run(&dense(), &f2);
    assert_close(&out[0], &second[0]);
}

#[test]
fn parallel_matches_running_each_graph() {
    let _gpu = common::serialize_gpu();
    let f = feeds();
    let mut ex = Graph::new("exp");
    let x = ex.input("x", f32s(&[B, D]));
    let y = ex.exp(x);
    ex.set_outputs(vec![y]);

    let both = parallel("both", &[&dense(), &ex]).unwrap();
    let out = run(&both, &f);
    assert_close(&out[0], &run(&dense(), &f)[0]);
    assert_close(&out[1], &run(&ex, &f)[0]);
}

#[test]
fn partition_parts_run_in_order_match_whole() {
    let _gpu = common::serialize_gpu();
    let g = model();
    let f = feeds();
    let whole = run(&g, &f);

    let n = g.len();
    let p = g.partition(3, |id| (id.0 as usize * 3 / n).min(2)).unwrap();
    let mut env = f.clone();
    let mut part_outs = Vec::new();
    for part in &p.parts {
        let outs = run(&part.graph, &env);
        for (name, v) in part.exports.iter().zip(&outs) {
            env.insert(name.clone(), v.clone());
        }
        part_outs.push(outs);
    }
    for (k, &(pi, j)) in p.outputs.iter().enumerate() {
        assert_close(&part_outs[pi][j], &whole[k]);
    }

    let merged = run(&p.merge().unwrap(), &f);
    for (a, b) in merged.iter().zip(&whole) {
        assert_close(a, b);
    }
}

#[test]
fn split_at_and_components_match_whole() {
    let _gpu = common::serialize_gpu();
    let g = model();
    let f = feeds();
    let whole = run(&g, &f);

    // Cut after the first layer's output (the second matmul's left operand).
    let cut = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::MatMul))
        .nth(1)
        .unwrap()
        .inputs[0];
    let p = g.split_at(&[cut]).unwrap();
    let head = run(&p.parts[0].graph, &f);
    let mut env = f.clone();
    for (name, v) in p.parts[0].exports.iter().zip(&head) {
        env.insert(name.clone(), v.clone());
    }
    let tail = run(&p.parts[1].graph, &env);
    let parts = [head, tail];
    for (k, &(pi, j)) in p.outputs.iter().enumerate() {
        assert_close(&parts[pi][j], &whole[k]);
    }

    // Independent parts import nothing: each runs on its own.
    let comps = g.components().unwrap();
    assert_eq!(comps.parts.len(), 2);
    let outs: Vec<_> = comps.parts.iter().map(|p| run(&p.graph, &f)).collect();
    for (k, &(pi, j)) in comps.outputs.iter().enumerate() {
        assert_close(&outs[pi][j], &whole[k]);
    }
    let merged = run(&comps.merge().unwrap(), &f);
    for (a, b) in merged.iter().zip(&whole) {
        assert_close(a, b);
    }
}
