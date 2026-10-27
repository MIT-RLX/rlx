// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lane fallback in `run_train_group`: a lane that fails while compiling, or
//! panics mid-run, hands its micro-batches to another lane (a fresh CPU lane
//! when none is left) and the job trains to the same parameters. Faults are
//! injected with `RLX_TRAIN_LANE_FAULT`, so this lives in its own test binary
//! (one test, scenarios in sequence) to keep the variable out of other tests.

use rlx_driver::{Node, ProcessGroup};
use rlx_ir::op::{BinaryOp, Op, ReduceOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::dist::{self, DataRef, TrainOptions, TrainSpec, WeightRef};
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

#[test]
fn a_failing_lane_hands_its_work_to_another_lane() {
    let dir = tempfile::tempdir().unwrap();
    let run = |device: &str, balance: bool, fault: Option<&str>| {
        // SAFETY: the only test in this binary; no other thread reads env.
        unsafe {
            match fault {
                Some(f) => std::env::set_var("RLX_TRAIN_LANE_FAULT", f),
                None => std::env::remove_var("RLX_TRAIN_LANE_FAULT"),
            }
        }
        rlx_ir::env::reload();
        let (mut s, w_true) = spec(
            dir.path(),
            30,
            TrainOptions {
                balance,
                ..Default::default()
            },
        );
        s.device = device.into();
        let ws = run_cluster(2, |_| s.clone());
        assert_eq!(ws[0], ws[1], "replicas diverged ({device}, {fault:?})");
        (ws[0].clone(), w_true)
    };
    let (reference, w_true) = run("cpu,cpu", false, None);
    assert!(max_err(&reference, &w_true) < 1e-3);
    for (device, balance, fault, what) in [
        (
            "cpu,cpu",
            false,
            "1@0",
            "lane 1 cannot compile: lane 0 runs its share",
        ),
        (
            "cpu,cpu",
            true,
            "1@0",
            "balanced: the failed lane drops out of the plan",
        ),
        (
            "cpu,cpu",
            false,
            "1@5",
            "lane 1 panics mid-run: lane 0 re-runs the rest",
        ),
        (
            "cpu",
            false,
            "0@4",
            "the only lane panics: a fresh CPU lane takes over",
        ),
        (
            "cpu",
            false,
            "0@0",
            "the only lane cannot compile: a CPU lane replaces it",
        ),
    ] {
        let (w, _) = run(device, balance, Some(fault));
        let e = max_err(&w, &w_true);
        eprintln!(
            "{what}: |w - w_true| = {e:.2e}, vs reference {:.2e}",
            max_err(&w, &reference)
        );
        assert!(e < 1e-3, "{what}: {w:?} vs {w_true:?}");
        if !balance {
            // Same micro-batches, same plan: only the summation order moved.
            assert!(
                max_err(&w, &reference) < 1e-5,
                "{what}: {w:?} vs {reference:?}"
            );
        }
    }
}
