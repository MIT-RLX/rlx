// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx_runtime::dist::run_train_group` over a real loopback TCP mesh (one
//! thread per rank): every `TrainOptions` path converges, all ranks finish with
//! the same parameters, ZeRO-1 reproduces plain data-parallel exactly,
//! push-data reproduces shared-filesystem data, and a checkpointed run resumes
//! to the same result as an uninterrupted one.

use rlx_driver::{Node, ProcessGroup};
use rlx_ir::op::{BinaryOp, Op, ReduceOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::dist::{self, DataRef, TrainOptions, TrainSpec, WeightRef, WireDtype};
use std::net::{Ipv4Addr, TcpListener};
use std::sync::Arc;

const F: DType = DType::F32;
const D: usize = 3;
const M: usize = 96;
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

/// Noiseless linear regression `y = X·w_true`, MSE; the spec covers the whole
/// dataset (`run_train_group` slices it per rank).
fn spec(dir: &std::path::Path, epochs: usize, options: TrainOptions) -> (TrainSpec, Vec<f32>) {
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
    let x_data = seeded(M * D, 3);
    let y_data: Vec<f32> = (0..M)
        .map(|i| (0..D).map(|k| x_data[i * D + k] * w_true[k]).sum())
        .collect();
    let s = TrainSpec {
        graph: bwd,
        params: vec![WeightRef {
            name: "w".into(),
            uri: write_f32(dir, "w0.bin", &[0.0; D]),
            packed: false,
        }],
        grad_start: 1,
        loss_index: 0,
        data: vec![
            DataRef {
                input: "X".into(),
                uri: write_f32(dir, "X.bin", &x_data),
                elem: D,
                shard_start: 0,
                shard_len: M,
            },
            DataRef {
                input: "y".into(),
                uri: write_f32(dir, "y.bin", &y_data),
                elem: 1,
                shard_start: 0,
                shard_len: M,
            },
        ],
        seed_input: Some("d_output".into()),
        momentum: 0.5,
        lr_per_epoch: vec![0.2; epochs],
        batch: BATCH,
        device: "cpu".into(),
        grad_group: 0,
        push_data: false,
        options: Some(options),
        frozen: Vec::new(),
    };
    (s, w_true)
}

/// Run `world` ranks on loopback threads; returns each rank's final `w`.
fn run_cluster(world: u32, mut make: impl FnMut(u32) -> TrainSpec) -> Vec<Vec<f32>> {
    let ports: Vec<String> = (0..world)
        .map(|_| {
            TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
                .unwrap()
                .local_addr()
                .unwrap()
                .to_string()
        })
        .collect();
    let handles: Vec<_> = (0..world)
        .map(|r| {
            let spec = make(r);
            let peers = ports.clone();
            std::thread::spawn(move || {
                let g: Arc<ProcessGroup> = Node::new(r, world)
                    .peers(peers.iter().map(String::as_str))
                    .unwrap()
                    .connect()
                    .unwrap();
                let (m, params) = dist::run_train_group(&spec, &g, |_| Vec::new(), false)
                    .expect("run_train_group");
                assert!(m.syncs > 0);
                g.barrier().unwrap();
                params.into_iter().next().unwrap().1
            })
        })
        .collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Train with `opts` on 3 ranks and check convergence + replica agreement.
fn converges(opts: TrainOptions, epochs: usize, tol: f32) -> Vec<f32> {
    let dir = tempfile::tempdir().unwrap();
    let (s, w_true) = spec(dir.path(), epochs, opts.clone());
    let ws = run_cluster(3, |_| s.clone());
    for w in &ws[1..] {
        assert_eq!(w, &ws[0], "replicas diverged under {opts:?}");
    }
    let e = max_err(&ws[0], &w_true);
    assert!(e < tol, "{opts:?}: |w - w_true| = {e}");
    ws[0].clone()
}

#[test]
fn plain_data_parallel_converges() {
    converges(TrainOptions::default(), 40, 1e-3);
}

#[test]
fn accumulation_and_bf16_wire_converge() {
    converges(
        TrainOptions {
            accum: 2,
            wire: WireDtype::Bf16,
            ..Default::default()
        },
        60,
        2e-2,
    );
}

#[test]
fn overlap_converges() {
    converges(
        TrainOptions {
            overlap: true,
            ..Default::default()
        },
        40,
        1e-3,
    );
}

#[test]
fn local_sgd_converges() {
    converges(
        TrainOptions {
            local_steps: 3,
            ..Default::default()
        },
        40,
        1e-3,
    );
}

#[test]
fn balanced_plan_converges() {
    converges(
        TrainOptions {
            balance: true,
            ..Default::default()
        },
        40,
        1e-3,
    );
}

#[test]
fn every_local_device_as_a_lane_converges() {
    // `device: "all"` = one lane per physical unit here (CPU + GPU on a Mac,
    // CPU alone on a GPU-less runner); balance sizes each lane's share.
    let dir = tempfile::tempdir().unwrap();
    let (mut s, w_true) = spec(
        dir.path(),
        40,
        TrainOptions {
            balance: true,
            ..Default::default()
        },
    );
    s.device = "all".into();
    let ws = run_cluster(2, |_| s.clone());
    assert_eq!(ws[0], ws[1]);
    assert!(max_err(&ws[0], &w_true) < 1e-3, "{:?} vs {w_true:?}", ws[0]);
}

#[test]
fn every_number_format_trains() {
    // Error floor per format: noiseless regression converges to the format's
    // resolution (~2^-mantissa of |w| ≈ 1), loss-scaled for the narrow ones.
    for (fmt, tol) in [
        ("f64", 1e-3),
        ("tf32", 3e-3),
        ("bf16", 3e-2),
        ("f16", 5e-3),
        ("f8e4m3", 0.2),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let (s, w_true) = spec(
            dir.path(),
            40,
            TrainOptions {
                number_format: Some(fmt.into()),
                ..Default::default()
            },
        );
        let ws = run_cluster(2, |_| s.clone());
        assert_eq!(ws[0], ws[1], "{fmt}: replicas diverged");
        let e = max_err(&ws[0], &w_true);
        assert!(e < tol, "{fmt}: |w - w_true| = {e} (tol {tol})");
    }
}

/// f64 training on every local GPU backend: native f64 on the CPU, double-
/// single where the backend has no f64 (Metal, wgpu …).
#[test]
fn f64_trains_on_every_local_device() {
    for dev in rlx_runtime::available_devices() {
        if matches!(
            dev,
            rlx_runtime::Device::Cpu | rlx_runtime::Device::Ane | rlx_runtime::Device::Xdna
        ) {
            continue;
        }
        let dir = tempfile::tempdir().unwrap();
        let (mut s, w_true) = spec(
            dir.path(),
            40,
            TrainOptions {
                number_format: Some("f64".into()),
                ..Default::default()
            },
        );
        s.device = rlx_runtime::device_label(dev).into();
        let ws = run_cluster(1, |_| s.clone());
        let e = max_err(&ws[0], &w_true);
        assert!(e < 1e-3, "{dev:?}: |w - w_true| = {e}");
    }
}

#[test]
fn sixteen_bit_wires_with_error_feedback_and_auto() {
    for wire in [WireDtype::Bf16, WireDtype::F16, WireDtype::Auto] {
        converges(
            TrainOptions {
                wire,
                ..Default::default()
            },
            40,
            1e-2,
        );
    }
}

#[test]
fn a_bad_number_format_is_rejected_on_every_rank() {
    let o = TrainOptions {
        number_format: Some("f7q9".into()),
        ..Default::default()
    };
    assert!(o.validate().is_err());
    // f16 starts loss-scaled, bf16 does not.
    let f16 = TrainOptions {
        number_format: Some("f16".into()),
        ..Default::default()
    };
    assert_eq!(f16.initial_loss_scale(), 1024.0);
    let bf16 = TrainOptions {
        number_format: Some("bf16".into()),
        ..Default::default()
    };
    assert_eq!(bf16.initial_loss_scale(), 1.0);
}

#[test]
fn zero1_matches_plain_data_parallel_exactly() {
    // reduce-scatter + slice update + all-gather is the same arithmetic as the
    // ring all-reduce + full update.
    let plain = converges(TrainOptions::default(), 20, 1e-1);
    let zero = converges(
        TrainOptions {
            zero: true,
            ..Default::default()
        },
        20,
        1e-1,
    );
    assert_eq!(plain, zero);
}

#[test]
fn push_data_matches_shared_filesystem() {
    let dir = tempfile::tempdir().unwrap();
    let (s, _) = spec(dir.path(), 10, TrainOptions::default());
    let shared = run_cluster(3, |_| s.clone());
    // Workers get URIs that do not exist: everything must come from rank 0.
    let pushed = run_cluster(3, |r| {
        let mut p = s.clone();
        p.push_data = true;
        if r > 0 {
            for d in &mut p.data {
                d.uri = "file:///nonexistent/rlx".into();
            }
            p.params[0].uri = "file:///nonexistent/rlx".into();
        }
        p
    });
    assert_eq!(shared, pushed);
}

#[test]
fn checkpoint_resume_matches_uninterrupted_run() {
    let dir = tempfile::tempdir().unwrap();
    let ck = dir.path().join("ckpt").to_string_lossy().to_string();
    let with_ckpt = |epochs| {
        spec(
            dir.path(),
            epochs,
            TrainOptions {
                checkpoint_dir: Some(ck.clone()),
                ..Default::default()
            },
        )
        .0
    };
    let (full, _) = spec(dir.path(), 8, TrainOptions::default());
    let straight = run_cluster(3, |_| full.clone());
    // "Crash" after 3 epochs, then relaunch the 8-epoch job: it resumes at 3.
    let first = with_ckpt(3);
    run_cluster(3, |_| first.clone());
    let second = with_ckpt(8);
    let resumed = run_cluster(3, |_| second.clone());
    assert_eq!(resumed, straight);
}

#[test]
fn a_dead_rank_fails_the_job_instead_of_hanging() {
    let dir = tempfile::tempdir().unwrap();
    let (s, _) = spec(dir.path(), 200, TrainOptions::default());
    let ports: Vec<String> = (0..2)
        .map(|_| {
            TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
                .unwrap()
                .local_addr()
                .unwrap()
                .to_string()
        })
        .collect();
    let p1 = ports.clone();
    // Rank 1 joins, then exits before training (a crashed node).
    let dead = std::thread::spawn(move || {
        let _g = Node::new(1, 2)
            .peers(p1.iter().map(String::as_str))
            .unwrap()
            .connect()
            .unwrap();
    });
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let g = Node::new(0, 2)
            .peers(ports.iter().map(String::as_str))
            .unwrap()
            .connect()
            .unwrap();
        tx.send(dist::run_train_group(&s, &g, |_| Vec::new(), false).map(|_| ()))
            .unwrap();
    });
    dead.join().unwrap();
    let out = rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("training hung on a dead rank");
    let err = out.expect_err("must fail");
    assert!(err.contains("rank 1"), "{err}");
}

/// A rank that is not `dist_node` (the phone node) runs
/// [`dist::trainserve_worker`]; the master runs `rlx-vision-bench`'s sequence
/// — a barrier once connected, `ship_train` to each worker, its own
/// `run_train_group`, then the post-train and final barriers. Out of step,
/// every rank waits forever, so a deadline turns that into a failure.
#[test]
fn trainserve_worker_matches_the_masters_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let (s, w_true) = spec(dir.path(), 30, TrainOptions::default());
    let world = 3u32;
    let ports: Vec<String> = (0..world)
        .map(|_| {
            TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
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
                dist::trainserve_worker(&g, false).unwrap()
            })
        })
        .collect();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let g = Node::new(0, world)
            .peers(ports.iter().map(String::as_str))
            .unwrap()
            .connect()
            .unwrap();
        g.barrier().unwrap();
        for r in 1..world {
            dist::ship_train(&g, r, &s).unwrap();
        }
        let (_, params) = dist::run_train_group(&s, &g, |_| Vec::new(), false).unwrap();
        g.barrier().unwrap(); // post-train
        g.barrier().unwrap(); // final
        tx.send(params.into_iter().next().unwrap().1).unwrap();
    });
    let w = rx
        .recv_timeout(std::time::Duration::from_secs(120))
        .expect("deadlock: trainserve_worker is out of step with the master");
    assert!(max_err(&w, &w_true) < 1e-3, "{w:?} vs {w_true:?}");
    for h in workers {
        assert!(h.join().unwrap().syncs > 0, "every worker trained");
    }
}
