// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! End-to-end `rlx_runtime::dist::run_train` checks — the ship-graph generic
//! trainer, driven WITHOUT a network by passing an in-process `reduce` closure.
//!
//! Covers the correctness-sensitive paths that separate a single-machine run
//! from a cluster run:
//!   * single-lane data-parallel training converges (the `Mean`/weighted reduce
//!     with `lane_count = 1` reproduces plain SGD);
//!   * `device:"all"` intra-node multi-lane (CPU + every GPU) still converges —
//!     the sample-weighted reduce keeps a >1-lane node unbiased;
//!   * the uneven-shard guard fails fast (a clear error) instead of deadlocking
//!     the cross-worker all-reduce.

use rlx_ir::op::{BinaryOp, Op, ReduceOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::dist::{self, DataRef, TrainSpec, WeightRef};

const F: DType = DType::F32;

/// Deterministic pseudo-random f32 in [-1, 1).
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

/// Noiseless linear-regression training job: `y = X · w_true`, MSE loss, one
/// trainable param `w`. Returns `(spec, w_true)`; data + init live under `dir`.
fn regression_spec(
    dir: &std::path::Path,
    device: &str,
    m: usize,
    d: usize,
    batch: usize,
) -> (TrainSpec, Vec<f32>) {
    // Forward: loss = mean_b( (X·w - y)^2 ), a scalar.
    let mut g = Graph::new("lr");
    let x = g.input("X", Shape::new(&[batch, d], F));
    let w = g.param("w", Shape::new(&[d, 1], F));
    let y = g.input("y", Shape::new(&[batch, 1], F));
    let pred = g.matmul(x, w, Shape::new(&[batch, 1], F));
    let diff = g.binary(BinaryOp::Sub, pred, y, Shape::new(&[batch, 1], F));
    let sq = g.binary(BinaryOp::Mul, diff, diff, Shape::new(&[batch, 1], F));
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
    // bwd.outputs = [loss, grad_w]; the seed input is `d_output`.
    let bwd = rlx_autodiff::grad_with_loss(&g, &[w]);

    // Deterministic problem, exactly recoverable.
    let w_true: Vec<f32> = seeded(d, 7);
    let x_data: Vec<f32> = seeded(m * d, 3);
    let y_data: Vec<f32> = (0..m)
        .map(|i| (0..d).map(|k| x_data[i * d + k] * w_true[k]).sum())
        .collect();
    let w0 = vec![0.0f32; d]; // identical init on every rank

    let w_uri = write_f32(dir, "w0.bin", &w0);
    let x_uri = write_f32(dir, "X.bin", &x_data);
    let y_uri = write_f32(dir, "y.bin", &y_data);

    let spec = TrainSpec {
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
                elem: d,
                shard_start: 0,
                shard_len: m,
            },
            DataRef {
                input: "y".into(),
                uri: y_uri,
                elem: 1,
                shard_start: 0,
                shard_len: m,
            },
        ],
        seed_input: Some("d_output".into()),
        momentum: 0.0,
        lr_per_epoch: vec![0.3; 300],
        batch,
        device: device.into(),
        grad_group: 0,
        push_data: false,
    };
    (spec, w_true)
}

/// In-process "cluster of one": the reduce is the identity, so `run_train`'s
/// weighted bucket `[grads…, lane_count]` comes back unchanged and it divides by
/// the local lane count — exactly one node's contribution.
fn identity_reduce(flat: &[f32]) -> Vec<f32> {
    flat.to_vec()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn single_lane_converges() {
    let dir = tempfile::tempdir().unwrap();
    let (spec, w_true) = regression_spec(dir.path(), "cpu", 16, 3, 4);
    let (metrics, params) =
        dist::run_train(&spec, 1, |_| Vec::new(), identity_reduce, false).unwrap();
    let w = &params[0].1;
    eprintln!(
        "single-lane: loss {:.3e}→{:.3e}, ‖w-w*‖∞={:.2e}",
        metrics.first_loss,
        metrics.last_loss,
        max_err(w, &w_true)
    );
    assert!(
        metrics.last_loss < metrics.first_loss * 0.1,
        "loss did not fall enough"
    );
    assert!(
        max_err(w, &w_true) < 1e-2,
        "did not recover w_true: {w:?} vs {w_true:?}"
    );
}

#[test]
fn multi_lane_all_converges() {
    // `device:"all"` fans across every local backend (CPU + any GPU). On a
    // single-backend host this is identical to `single_lane_converges`; on a Mac
    // it exercises the real multi-lane branch + sample-weighted reduce.
    let dir = tempfile::tempdir().unwrap();
    let (spec, w_true) = regression_spec(dir.path(), "all", 16, 3, 4);
    let (metrics, params) =
        dist::run_train(&spec, 1, |_| Vec::new(), identity_reduce, false).unwrap();
    let w = &params[0].1;
    eprintln!(
        "multi-lane lanes={:?}: loss {:.3e}→{:.3e}, ‖w-w*‖∞={:.2e}",
        metrics.lanes,
        metrics.first_loss,
        metrics.last_loss,
        max_err(w, &w_true)
    );
    assert!(
        metrics.last_loss < metrics.first_loss * 0.1,
        "loss did not fall enough"
    );
    assert!(
        max_err(w, &w_true) < 1e-2,
        "did not recover w_true: {w:?} vs {w_true:?}"
    );
}

#[test]
fn uneven_shards_fail_fast_not_deadlock() {
    // Simulate a cluster (world=2) whose ranks disagree on batch count: a reduce
    // that reports a variance across the guard's [mean_b, mean_b²] probe. The
    // guard must turn what would be a silent all-reduce deadlock into a clear
    // error, IDENTICALLY on every rank.
    let dir = tempfile::tempdir().unwrap();
    let (spec, _w) = regression_spec(dir.path(), "cpu", 16, 3, 4); // batches = 4
    let reduce = |flat: &[f32]| -> Vec<f32> {
        if flat.len() == 2 {
            // mean_b = b, but mean_b² inflated → variance ≫ 0 (uneven shards).
            vec![flat[0], flat[1] * 4.0]
        } else {
            flat.to_vec()
        }
    };
    let err = dist::run_train(&spec, 2, |_| Vec::new(), reduce, false).unwrap_err();
    eprintln!("guard: {err}");
    assert!(
        err.contains("uneven shards"),
        "expected uneven-shard error, got: {err}"
    );
}

#[test]
fn even_shards_pass_guard_and_train() {
    // Same world=2 path, but a faithful (identity) reduce → zero variance → the
    // guard passes and training proceeds (here the "cluster" is mocked to this
    // one rank, so it simply converges).
    let dir = tempfile::tempdir().unwrap();
    let (spec, w_true) = regression_spec(dir.path(), "cpu", 16, 3, 4);
    let (metrics, params) =
        dist::run_train(&spec, 2, |_| Vec::new(), identity_reduce, false).unwrap();
    assert!(metrics.last_loss < metrics.first_loss * 0.1);
    assert!(max_err(&params[0].1, &w_true) < 1e-2);
}
