// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx_runtime::model::distributed`: a package function split over a 3-rank
//! loopback mesh gives what one node gives; an LM function decodes the same
//! tokens; a package trains data-parallel (every param, and LoRA over a frozen
//! base) and its loss falls. Workers run `serve` — no package, no model code.
#![cfg(all(feature = "model-io", feature = "training", feature = "cpu"))]

use std::sync::Arc;

use rlx_driver::{Node, ProcessGroup};
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::Device;
use rlx_runtime::dist::{TrainOptions, WireDtype};
use rlx_runtime::model::distributed::{
    InferJob, StepLayout, exchange_speeds, parse_dims, run_local, serve, text_samples, train_job,
    worker,
};
use rlx_runtime::model::{Tensor, rlxp};
use serde_json::json;

const B: usize = 8;
const D: usize = 4;
const H: usize = 16;
const C: usize = 3;
const N: usize = 96;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn rand(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

/// `logits = relu(relu(x·w1 + b1)·w2)·w3 · head`: four matmuls to split.
fn mlp() -> Graph {
    let mut g = Graph::new("forward");
    let x = g.input("x", f32s(&[B, D]));
    let w1 = g.param("w1", f32s(&[D, H]));
    let b1 = g.param("b1", f32s(&[1, H]));
    let w2 = g.param("w2", f32s(&[H, H]));
    let w3 = g.param("w3", f32s(&[H, H]));
    let head = g.param("head", f32s(&[H, C]));
    let h = g.mm(x, w1);
    let h = g.add(h, b1);
    let h = g.relu(h);
    let h = g.mm(h, w2);
    let h = g.relu(h);
    let h = g.mm(h, w3);
    let h = g.relu(h);
    let y = g.mm(h, head);
    g.node_mut(y).name = Some("logits".into());
    g.set_outputs(vec![y]);
    g
}

/// The MLP package; `head` is a checkpoint tensor stored `[out, in]` and
/// referenced transposed (the package's own layout logic must apply).
fn mlp_package(dir: &std::path::Path) -> String {
    let head = rand(C * H, 11);
    let bytes: Vec<u8> = head.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![C, H], &bytes).unwrap();
    let st = safetensors::serialize(vec![("head.weight".to_string(), view)], None).unwrap();
    std::fs::write(dir.join("ckpt.safetensors"), st).unwrap();
    let path = dir.join("mlp.rlxp");
    rlxp::save_package(
        &path,
        &mlp(),
        &[
            Tensor::f32("w1", vec![D, H], &rand(D * H, 5)),
            Tensor::f32("b1", vec![1, H], &rand(H, 7)),
            Tensor::f32("w2", vec![H, H], &rand(H * H, 9)),
            Tensor::f32("w3", vec![H, H], &rand(H * H, 13)),
        ],
        &json!({
            "name": "mlp",
            "weights": [{"uri": "safetensors://ckpt.safetensors",
                         "map": {"head": {"tensor": "head.weight", "transpose": true}}}],
        }),
    )
    .unwrap();
    path.to_string_lossy().into()
}

/// A dataset safetensors: `x [N, D]`, `labels [N]` (I64, as token ids are).
fn dataset(dir: &std::path::Path) -> String {
    let x = rand(N * D, 3);
    let labels: Vec<i64> = (0..N)
        .map(|r| {
            let row = &x[r * D..(r + 1) * D];
            ((0..D).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap() % C) as i64
        })
        .collect();
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let lb: Vec<u8> = labels.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tx =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![N, D], &xb).unwrap();
    let tl = safetensors::tensor::TensorView::new(safetensors::Dtype::I64, vec![N], &lb).unwrap();
    let st = safetensors::serialize(
        vec![("x".to_string(), tx), ("labels".to_string(), tl)],
        None,
    )
    .unwrap();
    let p = dir.join("data.safetensors");
    std::fs::write(&p, st).unwrap();
    p.to_string_lossy().into()
}

/// `world` ranks on loopback: workers `serve`, rank 0 runs `lead`.
fn cluster<T: Send + 'static>(world: u32, lead: impl FnOnce(&ProcessGroup) -> T) -> T {
    let ports: Vec<String> = (0..world)
        .map(|_| {
            std::net::TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .to_string()
        })
        .collect();
    let workers: Vec<_> = (1..world)
        .map(|r| {
            let p = ports.clone();
            std::thread::spawn(move || {
                let g: Arc<ProcessGroup> = Node::new(r, world)
                    .peers(p.iter().map(String::as_str))
                    .unwrap()
                    .connect()
                    .unwrap();
                let did = serve(&g, false).unwrap();
                g.barrier().unwrap();
                did
            })
        })
        .collect();
    let g = Node::new(0, world)
        .peers(ports.iter().map(String::as_str))
        .unwrap()
        .connect()
        .unwrap();
    let out = lead(&g);
    g.barrier().unwrap();
    for w in workers {
        eprintln!("worker: {}", w.join().unwrap());
    }
    out
}

#[test]
fn a_split_function_gives_what_one_node_gives() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = mlp_package(dir.path());
    let dims = parse_dims("").unwrap();
    let requests: Vec<Vec<f32>> = (0..3).map(|i| rand(B * D, 100 + i)).collect();
    let local: Vec<Vec<f32>> = requests
        .iter()
        .map(|x| {
            run_local(&pkg, "forward", &dims, Device::Cpu, &[("x", x)])
                .unwrap()
                .remove(0)
        })
        .collect();
    let (shares, got) = cluster(3, |g| {
        let mut job =
            InferJob::start(g, &pkg, "forward", &dims, &[], "cpu", None, WireDtype::F32).unwrap();
        let got: Vec<Vec<f32>> = requests
            .iter()
            .map(|x| job.run(g, &[("x", x)]).unwrap().remove(0))
            .collect();
        let shares = job.shares.clone();
        job.finish(g).unwrap();
        (shares, got)
    });
    assert_eq!(shares.len(), 3);
    assert!(
        shares.iter().all(|&s| s > 0.0),
        "every rank got work: {shares:?}"
    );
    for (a, b) in got.iter().zip(&local) {
        assert_eq!(a, b, "the split computes exactly what one node computes");
    }
}

/// Calibration times every stage, and a re-laid-out step (forced here:
/// any gain) is deployed to the workers mid-job and still computes what
/// one node computes.
#[test]
fn a_calibrated_relayout_gives_what_one_node_gives() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = mlp_package(dir.path());
    let dims = parse_dims("").unwrap();
    let requests: Vec<Vec<f32>> = (0..3).map(|i| rand(B * D, 200 + i)).collect();
    let local: Vec<Vec<f32>> = requests
        .iter()
        .map(|x| {
            run_local(&pkg, "forward", &dims, Device::Cpu, &[("x", x)])
                .unwrap()
                .remove(0)
        })
        .collect();
    let (speeds, relaid, got) = cluster(3, |g| {
        let mut job = InferJob::start_with_layout(
            g,
            &pkg,
            "forward",
            &dims,
            &[1.0, 1.0, 1.0],
            "cpu",
            None,
            WireDtype::F32,
            &[],
            &StepLayout::default(),
        )
        .unwrap();
        // A gain below -1 re-lays the step out whatever it is estimated at.
        let (speeds, relaid) = job
            .calibrate(g, &[("x".into(), requests[0].clone())], 2, -10.0)
            .unwrap();
        let got: Vec<Vec<f32>> = requests
            .iter()
            .map(|x| job.run(g, &[("x", x)]).unwrap().remove(0))
            .collect();
        job.finish(g).unwrap();
        (speeds, relaid, got)
    });
    assert!(relaid, "the forced relayout did not happen");
    assert_eq!(speeds.len(), 3);
    assert!(
        speeds.iter().all(|&s| s > 0.0),
        "every stage was timed: {speeds:?}"
    );
    for (a, b) in got.iter().zip(&local) {
        assert_eq!(
            a, b,
            "the relaid split computes exactly what one node computes"
        );
    }
}

/// A re-layout that would move weights for longer than it saves over the
/// job is not made: forced, it happens with no step count, and over links
/// slow enough that the weights take hours it is skipped for a short job.
#[test]
fn calibration_skips_a_relayout_its_transfer_does_not_pay_for() {
    use rlx_runtime::model::distributed::Links;
    let dir = tempfile::tempdir().unwrap();
    let pkg = mlp_package(dir.path());
    let dims = parse_dims("").unwrap();
    let x = rand(B * D, 500);
    let want = run_local(&pkg, "forward", &dims, Device::Cpu, &[("x", &x)])
        .unwrap()
        .remove(0);
    let slow = Links {
        latency: vec![vec![1e-3; 3]; 3],
        bandwidth: vec![vec![1e-3; 3]; 3],
        order: None,
    };
    for steps in [None, Some(10)] {
        let (pkg, dims, x, slow) = (pkg.clone(), dims.clone(), x.clone(), slow.clone());
        let (relaid, got) = cluster(3, move |g| {
            let mut job = InferJob::start_with_layout(
                g,
                &pkg,
                "forward",
                &dims,
                // Uneven, so the measured speeds lay the step out differently.
                &[100.0, 1.0, 1.0],
                "cpu",
                None,
                WireDtype::F32,
                &[],
                &StepLayout {
                    links: Some(slow),
                    ..Default::default()
                },
            )
            .unwrap();
            let (_, relaid) = job
                .calibrate_for(g, &[("x".into(), x.clone())], 2, -10.0, steps)
                .unwrap();
            let got = job.run(g, &[("x", &x)]).unwrap().remove(0);
            job.finish(g).unwrap();
            (relaid, got)
        });
        assert_eq!(got, want);
        assert_eq!(relaid, steps.is_none(), "steps {steps:?}");
    }
}

/// A worker serving one job after another keeps its stage: the second job
/// with the same plan sends no weights, and still computes what one node
/// computes.
#[test]
fn a_kept_worker_reuses_its_stage_for_the_next_job() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = mlp_package(dir.path());
    let dims = parse_dims("").unwrap();
    let x = rand(B * D, 300);
    let want = run_local(&pkg, "forward", &dims, Device::Cpu, &[("x", &x)])
        .unwrap()
        .remove(0);
    let world = 3u32;
    let ports: Vec<String> = (0..world)
        .map(|_| {
            std::net::TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .to_string()
        })
        .collect();
    let workers: Vec<_> = (1..world)
        .map(|r| {
            let p = ports.clone();
            std::thread::spawn(move || {
                let g: Arc<ProcessGroup> = Node::new(r, world)
                    .peers(p.iter().map(String::as_str))
                    .unwrap()
                    .connect()
                    .unwrap();
                // Two jobs on one thread: the stage kept between them.
                for _ in 0..2 {
                    serve(&g, false).unwrap();
                }
                g.barrier().unwrap();
            })
        })
        .collect();
    let g = Node::new(0, world)
        .peers(ports.iter().map(String::as_str))
        .unwrap()
        .connect()
        .unwrap();
    let mut kept = Vec::new();
    for _ in 0..2 {
        let mut job =
            InferJob::start(&g, &pkg, "forward", &dims, &[], "cpu", None, WireDtype::F32).unwrap();
        kept.push(job.kept);
        assert_eq!(job.run(&g, &[("x", &x)]).unwrap().remove(0), want);
        job.finish(&g).unwrap();
    }
    g.barrier().unwrap();
    for w in workers {
        w.join().unwrap();
    }
    assert_eq!(
        kept,
        vec![0, 2],
        "the second job should find both workers' stages kept"
    );
}

/// The next job on persistent workers plans by the layout the last one
/// ended with, calibrated and re-laid out: every worker keeps its stage.
#[test]
fn a_calibrated_layout_is_kept_for_the_next_job() {
    use rlx_runtime::model::distributed::exchange_speeds_links;
    let dir = tempfile::tempdir().unwrap();
    let pkg = mlp_package(dir.path());
    let dims = parse_dims("").unwrap();
    let x = rand(B * D, 400);
    let want = run_local(&pkg, "forward", &dims, Device::Cpu, &[("x", &x)])
        .unwrap()
        .remove(0);
    let world = 3u32;
    let ports: Vec<String> = (0..world)
        .map(|_| {
            std::net::TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .to_string()
        })
        .collect();
    let workers: Vec<_> = (1..world)
        .map(|r| {
            let p = ports.clone();
            std::thread::spawn(move || {
                let g: Arc<ProcessGroup> = Node::new(r, world)
                    .peers(p.iter().map(String::as_str))
                    .unwrap()
                    .connect()
                    .unwrap();
                for _ in 0..2 {
                    exchange_speeds_links(&g, Device::Cpu).unwrap();
                    serve(&g, false).unwrap();
                }
                g.barrier().unwrap();
            })
        })
        .collect();
    let g = Node::new(0, world)
        .peers(ports.iter().map(String::as_str))
        .unwrap()
        .connect()
        .unwrap();
    let mut kept = Vec::new();
    let mut orders = Vec::new();
    for job in 0..2 {
        let (speeds, links) = exchange_speeds_links(&g, Device::Cpu).unwrap();
        let mut j = InferJob::start_with_layout(
            &g,
            &pkg,
            "forward",
            &dims,
            &speeds,
            "cpu",
            None,
            WireDtype::F32,
            &[],
            &StepLayout {
                links: Some(links),
                ..Default::default()
            },
        )
        .unwrap();
        kept.push(j.kept);
        if job == 0 {
            // A gain below -1 re-lays the step out whatever it is estimated at.
            let (_, relaid) = j
                .calibrate(&g, &[("x".into(), x.clone())], 2, -10.0)
                .unwrap();
            assert!(relaid);
        }
        orders.push(j.order.clone());
        assert_eq!(j.run(&g, &[("x", &x)]).unwrap().remove(0), want);
        j.finish(&g).unwrap();
    }
    g.barrier().unwrap();
    for w in workers {
        w.join().unwrap();
    }
    assert_eq!(
        orders[0], orders[1],
        "the second job keeps the calibrated order"
    );
    assert_eq!(
        kept,
        vec![0, 2],
        "the second job should find both workers' stages kept"
    );
}

/// `logits[1, S, V] = embed[tokens] · out`: a token LM in miniature.
fn tiny_lm(dir: &std::path::Path, s: usize, v: usize) -> String {
    let d = 6;
    let mut g = Graph::new("forward");
    let tok = g.input("tokens", Shape::new(&[1, s], DType::I32));
    let emb = g.param("embed", f32s(&[v, d]));
    let w = g.param("out", f32s(&[d, v]));
    let flat = g.reshape_(tok, vec![s as i64]);
    let e = g.gather(emb, flat, 0, f32s(&[s, d]));
    let l = g.mm(e, w);
    let l = g.reshape_(l, vec![1, s as i64, v as i64]);
    g.set_outputs(vec![l]);
    let path = dir.join("lm.rlxp");
    rlxp::save_package(
        &path,
        &g,
        &[
            Tensor::f32("embed", vec![v, d], &rand(v * d, 21)),
            Tensor::f32("out", vec![d, v], &rand(d * v, 23)),
        ],
        &json!({"name": "tiny-lm"}),
    )
    .unwrap();
    path.to_string_lossy().into()
}

#[test]
fn an_lm_function_decodes_the_same_tokens_split() {
    let dir = tempfile::tempdir().unwrap();
    let (s, v) = (12, 17);
    let pkg = tiny_lm(dir.path(), s, v);
    let dims = parse_dims("").unwrap();
    // Greedy decode on one node.
    let mut tokens = vec![3u32, 5, 1];
    let mut want = Vec::new();
    for _ in 0..6 {
        let mut feed = vec![0f32; s];
        for (f, &t) in feed.iter_mut().zip(&tokens) {
            *f = t as f32;
        }
        let l = run_local(&pkg, "forward", &dims, Device::Cpu, &[("tokens", &feed)])
            .unwrap()
            .remove(0);
        let row = &l[(tokens.len() - 1) * v..tokens.len() * v];
        let next = (0..v).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap() as u32;
        tokens.push(next);
        want.push(next);
    }
    let got = cluster(2, |g| {
        let mut job =
            InferJob::start(g, &pkg, "forward", &dims, &[], "cpu", None, WireDtype::Auto).unwrap();
        let out = job.generate(g, &[3, 5, 1], 6).unwrap();
        job.finish(g).unwrap();
        out
    });
    assert_eq!(got, want);
}

#[test]
fn a_package_trains_data_parallel_all_params_and_lora() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = mlp_package(dir.path());
    let data = dataset(dir.path());
    let loss = json!({"cross_entropy": {"logits": "logits", "labels": "labels"}});
    for (what, train) in [
        ("all", json!("all")),
        (
            "lora",
            json!({"lora": {"targets": ["w2", "w3"], "rank": 4}}),
        ),
    ] {
        let opts = json!({"function": "forward", "loss": loss, "train": train,
                          "optimizer": {"lr": 0.02}});
        let job = train_job(&pkg, &opts, &data, 30, "cpu", TrainOptions::default()).unwrap();
        assert_eq!(job.samples, N);
        if what == "lora" {
            assert!(
                !job.spec.frozen.is_empty(),
                "LoRA's base travels as frozen params"
            );
        }
        let before = job.loss(&Default::default(), Device::Cpu);
        let (m, trained) = cluster(3, |g| job.run(g, false).unwrap());
        let after = job.loss(&trained, Device::Cpu);
        eprintln!("{what}: loss {before} -> {after} ({} syncs)", m.syncs);
        assert!(after < 0.7 * before, "{what}: {before} -> {after}");
        // The trained values save as the package trainer saves them.
        let out = dir.path().join(format!("{what}.rlxp"));
        job.save(&trained, &out, what).unwrap();
        assert!(out.exists());
    }
}

#[test]
fn text_becomes_next_token_windows() {
    let ids: Vec<u32> = (10..20).collect();
    let names = |n: &[&str]| -> std::collections::HashSet<String> {
        n.iter().map(|s| s.to_string()).collect()
    };
    // A `last`-position LM: windows of 4 every 3 tokens, labels the token
    // after each window, the position its last index.
    let inputs = vec![
        ("input_ids".to_string(), vec![1, 1, 4]),
        ("last".to_string(), vec![1, 1]),
        ("labels".to_string(), vec![1, 1]),
    ];
    let v = text_samples(&inputs, &names(&["input_ids", "last"]), &ids, 3).unwrap();
    assert_eq!(v["input_ids"], [10., 11., 12., 13., 13., 14., 15., 16.]);
    assert_eq!(v["last"], [3., 3.]);
    assert_eq!(v["labels"], [14., 17.]);
    // Whole-sequence labels with a mask: the window shifted by one, ones.
    let inputs = vec![
        ("tokens".to_string(), vec![1, 4]),
        ("mask".to_string(), vec![1, 4]),
        ("targets".to_string(), vec![1, 4]),
    ];
    let v = text_samples(&inputs, &names(&["tokens", "mask"]), &ids, 5).unwrap();
    assert_eq!(v["tokens"], [10., 11., 12., 13., 15., 16., 17., 18.]);
    assert_eq!(v["targets"], [11., 12., 13., 14., 16., 17., 18., 19.]);
    assert_eq!(v["mask"], [1.0; 8]);
    // Too little text for one window and its next token.
    assert!(text_samples(&inputs, &names(&["tokens", "mask"]), &ids[..4], 1).is_err());
}

/// A rank that is not `dist_node` (the iOS/Android node, through rlx-ffi)
/// runs [`worker`]; rank 0 runs `dist_node`'s own sequence — a barrier once
/// connected, the speed exchange, the job, then its post-model and final
/// barriers. The two must
/// agree on every collective, or the mesh waits forever: an iPad that went
/// straight to the speed exchange deadlocked a 4-rank cluster this way.
#[test]
fn a_worker_rank_matches_dist_nodes_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let (s, v) = (12, 17);
    let pkg = tiny_lm(dir.path(), s, v);
    let world = 3u32;
    let ports: Vec<String> = (0..world)
        .map(|_| {
            std::net::TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .to_string()
        })
        .collect();
    let workers: Vec<_> = (1..world)
        .map(|r| {
            let p = ports.clone();
            std::thread::spawn(move || {
                let g: Arc<ProcessGroup> = Node::new(r, world)
                    .peers(p.iter().map(String::as_str))
                    .unwrap()
                    .connect()
                    .unwrap();
                worker(&g, Device::Cpu, false).unwrap()
            })
        })
        .collect();
    let (tx, rx) = std::sync::mpsc::channel();
    let lead_pkg = pkg.clone();
    std::thread::spawn(move || {
        let g = Node::new(0, world)
            .peers(ports.iter().map(String::as_str))
            .unwrap()
            .connect()
            .unwrap();
        g.barrier().unwrap();
        let speeds = exchange_speeds(&g, Device::Cpu).unwrap();
        let dims = parse_dims("").unwrap();
        let mut job = InferJob::start(
            &g,
            &lead_pkg,
            "forward",
            &dims,
            &speeds,
            "cpu",
            None,
            WireDtype::F32,
        )
        .unwrap();
        let out = job.generate(&g, &[3, 5, 1], 4).unwrap();
        job.finish(&g).unwrap();
        g.barrier().unwrap(); // post-model
        g.barrier().unwrap(); // final
        tx.send((speeds.len(), out)).unwrap();
    });
    let (n, out) = rx
        .recv_timeout(std::time::Duration::from_secs(120))
        .expect("the ranks deadlocked: worker() is out of step with dist_node's sequence");
    assert_eq!(n, world as usize, "every rank reported a speed");
    assert_eq!(out.len(), 4);
    for w in workers {
        assert!(w.join().unwrap().contains("stage"), "workers served stages");
    }
}
