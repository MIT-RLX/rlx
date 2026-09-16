// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A node joins a **training** run, not just an inference pipeline.
//!
//! `serve_worker` covers ship-graph inference; this covers the other half of
//! what a non-desktop rank is for. The worker is model-agnostic — it receives a
//! `TrainSpec` and trains a model it has no code for — and the cross-rank
//! gradient reduce rides on the group's own `all_reduce`, so the node driver
//! needs no dependency on the in-graph collectives crate.
//!
//! The assertion that matters is **parameter agreement**: after a data-parallel
//! run the two ranks must hold bit-identical weights. Each rank sees a different
//! data shard, so if the all-reduce did not happen they would drift apart while
//! each still converging nicely on its own shard — a failure that looks exactly
//! like success from either side alone.

mod common;

use rlx_autodiff as _;
use rlx_driver::{NetTransport, ProcessGroup};
use rlx_ir::op::{BinaryOp, Op, ReduceOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::dist::node::{mean_reduce, serve_trainer_here};
use rlx_runtime::dist::{self, DataRef, TrainSpec, WeightRef};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;
use std::thread;

const F: DType = DType::F32;
const D: usize = 3;
const PER_RANK: usize = 16;
const BATCH: usize = 4;

fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn write_f32(dir: &std::path::Path, name: &str, vals: &[f32]) -> String {
    let path = dir.join(name);
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&path, &bytes).unwrap();
    format!("file://{}", path.display())
}

/// `loss = mean((X·w - y)^2)` over a shard, with the backward graph attached.
/// `shard` selects which half of the data this rank trains on.
fn spec_for(dir: &std::path::Path, shard: usize, world: usize) -> TrainSpec {
    let m = PER_RANK * world;
    let mut g = Graph::new("lr");
    let x = g.input("X", Shape::new(&[BATCH, D], F));
    let w = g.param("w", Shape::new(&[D, 1], F));
    let y = g.input("y", Shape::new(&[BATCH, 1], F));
    let pred = g.matmul(x, w, Shape::new(&[BATCH, 1], F));
    let diff = g.binary(BinaryOp::Sub, pred, y, Shape::new(&[BATCH, 1], F));
    let sq = g.binary(BinaryOp::Mul, diff, diff, Shape::new(&[BATCH, 1], F));
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Mean,
            axes: vec![0, 1],
            keep_dim: false,
        },
        vec![sq],
        Shape::from_dims(&[], F),
    );
    g.set_outputs(vec![loss]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[w]);

    let w_true = seeded(D, 7);
    let x_data = seeded(m * D, 3);
    let y_data: Vec<f32> = (0..m)
        .map(|i| (0..D).map(|k| x_data[i * D + k] * w_true[k]).sum())
        .collect();

    // Same starting weights on every rank; the shards differ.
    let w_uri = write_f32(dir, "w0.bin", &[0.0f32; D]);
    let x_uri = write_f32(dir, "X.bin", &x_data);
    let y_uri = write_f32(dir, "y.bin", &y_data);

    TrainSpec {
        graph: bwd,
        params: vec![WeightRef {
            name: "w".into(),
            uri: w_uri,
            packed: false,
        }],
        grad_start: 1,
        loss_index: 0,
        data: vec![
            DataRef {
                input: "X".into(),
                uri: x_uri,
                elem: D,
                shard_start: shard * PER_RANK,
                shard_len: PER_RANK,
            },
            DataRef {
                input: "y".into(),
                uri: y_uri,
                elem: 1,
                shard_start: shard * PER_RANK,
                shard_len: PER_RANK,
            },
        ],
        seed_input: Some("d_output".into()),
        momentum: 0.0,
        lr_per_epoch: vec![0.3; 60],
        batch: BATCH,
        device: "cpu".into(),
        grad_group: 0,
        push_data: false,
    }
}

#[test]
fn a_worker_node_joins_a_data_parallel_training_run() {
    let _gpu = common::serialize_gpu();
    let world: u32 = 2;
    let dir = std::env::temp_dir().join(format!("rlx_node_train_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let listeners: Vec<TcpListener> = (0..world)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();

    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let (addrs, dir) = (addrs.clone(), dir.clone());
            thread::spawn(move || -> Result<Vec<f32>, String> {
                let rank = rank as u32;
                let t = NetTransport::from_listener(rank, world, listener, addrs, 1 << 20)
                    .map_err(|e| e.to_string())?;
                let group = Arc::new(ProcessGroup::new(Arc::new(t)));

                if rank == 0 {
                    // Coordinator: ship the far shard to the worker, then train
                    // its own shard in lockstep with it.
                    dist::ship_train(&group, 1, &spec_for(&dir, 1, world as usize))?;
                    let spec = spec_for(&dir, 0, world as usize);
                    let (metrics, params) = dist::run_train(
                        &spec,
                        world,
                        dist::uri_resolver,
                        |flat| mean_reduce(&group, flat),
                        false,
                    )?;
                    assert!(
                        metrics.last_loss < metrics.first_loss,
                        "coordinator loss did not fall: {} -> {}",
                        metrics.first_loss,
                        metrics.last_loss
                    );
                    Ok(params.into_iter().next().unwrap().1)
                } else {
                    // The node under test: no model code, just a spec off the wire.
                    let report = serve_trainer_here(&group, dist::uri_resolver, false)?;
                    assert_eq!(report.rank, 1);
                    assert!(
                        report.metrics.samples > 0,
                        "worker trained on nothing: {:?}",
                        report.metrics
                    );
                    assert!(
                        report.metrics.last_loss < report.metrics.first_loss,
                        "worker loss did not fall: {} -> {}",
                        report.metrics.first_loss,
                        report.metrics.last_loss
                    );
                    Ok(report.params.into_iter().next().unwrap().1)
                }
            })
        })
        .collect();

    let out: Vec<Vec<f32>> = handles
        .into_iter()
        .enumerate()
        .map(|(i, h)| {
            h.join()
                .unwrap_or_else(|_| panic!("rank {i} panicked"))
                .unwrap_or_else(|e| panic!("rank {i}: {e}"))
        })
        .collect();
    std::fs::remove_dir_all(&dir).ok();

    // The point of the exercise: the replicas agree. Each rank saw a different
    // shard, so without the all-reduce these would differ while both still
    // looked like they were converging.
    assert_eq!(
        out[0], out[1],
        "ranks disagree after training — the gradient all-reduce did not happen"
    );
    assert!(
        out[0].iter().any(|v| v.abs() > 1e-3),
        "weights never moved off their zero init: {:?}",
        out[0]
    );
}

/// Guard the guard for `assert_eq!(out[0], out[1])` above.
///
/// That assertion only means something if the two shards would otherwise
/// produce different weights. Train each shard alone — identity reduce, a
/// cluster of one — and require the results to differ. If they did not, the
/// distributed test would pass whether or not the all-reduce ran.
#[test]
fn the_two_shards_disagree_without_a_reduce() {
    let dir = std::env::temp_dir().join(format!("rlx_node_shard_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let solo = |shard: usize| -> Vec<f32> {
        let spec = spec_for(&dir, shard, 2);
        let (_m, params) = dist::run_train(
            &spec,
            1, // a cluster of one: the identity reduce is the whole group
            dist::uri_resolver,
            |flat| flat.to_vec(),
            false,
        )
        .expect("solo training");
        params.into_iter().next().unwrap().1
    };
    let a = solo(0);
    let b = solo(1);
    std::fs::remove_dir_all(&dir).ok();

    assert_ne!(
        a, b,
        "the shards train to the same weights, so the distributed test's \
         parameter-agreement assertion proves nothing"
    );
}
