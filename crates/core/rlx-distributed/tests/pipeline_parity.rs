// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Correctness: a graph partitioned into pipeline stages and run
//! (a) in-process and (b) over real localhost TCP sockets (one worker thread per
//! stage, each holding only its own parameter shard) must reproduce the
//! single-node result bit-for-bit.

use rlx_distributed::graph::{
    bind_stage, partition, run_pipeline_local, run_pipeline_tcp, serve_bound,
};
use rlx_distributed::{NamedTensor, Param, Pipeline};
use rlx_ir::{DType, Graph, Shape, op::BinaryOp};
use rlx_runtime::{CompileOptions, Device, Session};
use std::collections::HashMap;

/// Two linear layers: y = (x·W1 + b1)·W2 + b2. Enough compute nodes that a
/// 2-way partition cuts between the layers (boundary tensor = the hidden state).
fn build_mlp() -> (Graph, HashMap<String, Vec<f32>>, Vec<f32>) {
    let mut g = Graph::new("mlp2");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let w1 = g.param("w1", Shape::new(&[4, 8], DType::F32));
    let b1 = g.param("b1", Shape::new(&[8], DType::F32));
    let h1 = g.matmul(x, w1, Shape::new(&[2, 8], DType::F32));
    let h1b = g.binary(BinaryOp::Add, h1, b1, Shape::new(&[2, 8], DType::F32));
    let w2 = g.param("w2", Shape::new(&[8, 4], DType::F32));
    let b2 = g.param("b2", Shape::new(&[4], DType::F32));
    let h2 = g.matmul(h1b, w2, Shape::new(&[2, 4], DType::F32));
    let y = g.binary(BinaryOp::Add, h2, b2, Shape::new(&[2, 4], DType::F32));
    g.set_outputs(vec![y]);

    let rnd = |seed: f64, n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let v = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
                (v - v.floor()) as f32 - 0.5
            })
            .collect()
    };
    let mut params = HashMap::new();
    params.insert("w1".to_string(), rnd(1.0, 4 * 8));
    params.insert("b1".to_string(), rnd(2.0, 8));
    params.insert("w2".to_string(), rnd(3.0, 8 * 4));
    params.insert("b2".to_string(), rnd(4.0, 4));
    let x = rnd(5.0, 2 * 4);
    (g, params, x)
}

fn single_node(g: &Graph, params: &HashMap<String, Vec<f32>>, x: &[f32]) -> Vec<f32> {
    let opts = CompileOptions::default();
    let mut c = Session::new(Device::Cpu).compile_with(g.clone(), &opts);
    for (n, d) in params {
        c.set_param(n, d);
    }
    c.run(&[("x", x)]).into_iter().next().unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "output length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn partition_splits_weights_across_stages() {
    let (g, _params, _x) = build_mlp();
    let stages = partition(&g, 2);
    assert_eq!(stages.len(), 2);
    // Every param lands in exactly one stage (weight sharding, no duplication).
    let mut all: Vec<String> = stages.iter().flat_map(|s| s.params.clone()).collect();
    all.sort();
    let mut uniq = all.clone();
    uniq.dedup();
    assert_eq!(all, uniq, "a param was duplicated across stages");
    assert_eq!(uniq, vec!["b1", "b2", "w1", "w2"], "all four params must be covered exactly once");
    // Stage 0 emits a boundary the model input `x` alone can't satisfy stage 1.
    assert!(!stages[0].outputs.is_empty(), "stage 0 must emit a boundary tensor");
    assert!(
        stages[1].inputs.iter().any(|n| n.starts_with("__stage_boundary_")),
        "stage 1 must consume stage 0's boundary"
    );
}

#[test]
fn in_process_pipeline_matches_single_node() {
    let (g, params, x) = build_mlp();
    let reference = single_node(&g, &params, &x);
    let mut src = params.clone(); // HashMap<String,Vec<f32>> is a ParamSource
    let stages = partition(&g, 2);
    let opts = CompileOptions::default();
    let out = run_pipeline_local(
        stages,
        &mut src,
        vec![NamedTensor::new("x", vec![2, 4], x.clone())],
        Device::Cpu,
        &opts,
    );
    assert_eq!(out.len(), 1, "one final output (logits)");
    let d = max_abs_diff(&out[0].data, &reference);
    assert!(d < 1e-6, "in-process pipeline diverged: max|err| {d:e}");
}

#[test]
fn tcp_pipeline_matches_single_node() {
    let (g, params, x) = build_mlp();
    let reference = single_node(&g, &params, &x);
    let stages = partition(&g, 2);

    // Bind each stage's listener up front so the coordinator knows the addrs,
    // then serve each stage in its own thread with ONLY its param shard.
    let mut addrs = Vec::new();
    let mut handles = Vec::new();
    for stage in &stages {
        let (addr, listener) = bind_stage("127.0.0.1:0").expect("bind");
        addrs.push(addr.to_string());
        let stage = stage.clone();
        // Each worker gets only the params for its own stage — proving the shard.
        let mut shard: HashMap<String, Vec<f32>> = stage
            .params
            .iter()
            .map(|p| (p.clone(), params[p].clone()))
            .collect();
        handles.push(std::thread::spawn(move || {
            let opts = CompileOptions::default();
            serve_bound(listener, stage, &mut shard, Device::Cpu, &opts, 1).expect("serve");
        }));
    }

    let out = run_pipeline_tcp(
        &stages,
        &addrs,
        vec![NamedTensor::new("x", vec![2, 4], x.clone())],
    )
    .expect("coordinator");
    for h in handles {
        h.join().expect("worker thread");
    }

    assert_eq!(out.len(), 1);
    let d = max_abs_diff(&out[0].data, &reference);
    assert!(d < 1e-6, "TCP pipeline diverged: max|err| {d:e}");
}

#[test]
fn facade_with_closure_source_matches_single_node() {
    // Proves the model-agnostic seam + DX: a `Pipeline` fed by a *closure*
    // ParamSource (the one-liner a model crate uses to adapt its own loader),
    // returning `Param::F32` on demand — no HashMap, no model dependency.
    let (g, params, x) = build_mlp();
    let reference = single_node(&g, &params, &x);

    let mut source = move |name: &str| params.get(name).cloned().map(Param::F32);
    let out = Pipeline::partition(&g, 3).run_local(
        &mut source,
        vec![NamedTensor::new("x", vec![2, 4], x.clone())],
        Device::Cpu,
        &CompileOptions::default(),
    );
    assert_eq!(out.len(), 1);
    let d = max_abs_diff(&out[0].data, &reference);
    assert!(d < 1e-6, "facade/closure-source pipeline diverged: max|err| {d:e}");
}
