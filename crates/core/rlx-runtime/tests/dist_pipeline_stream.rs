// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx_runtime::dist::{pipeline_run, serve_stage_stream}` over a loopback TCP
//! mesh: micro-batches come back complete, in order, through every stage, on
//! each activation wire width.

use rlx_driver::{Node, ProcessGroup};
use rlx_runtime::dist::{self, WireDtype};
use std::net::{Ipv4Addr, TcpListener};
use std::sync::Arc;

/// Stage r scales by (r + 2): a pure scale keeps wire rounding relative (an
/// added offset would cancel and magnify it).
fn stage(r: u32) -> impl FnMut(&[f32]) -> Vec<f32> {
    move |x: &[f32]| x.iter().map(|v| v * (r as f32 + 2.0)).collect()
}

fn run(world: u32, wire: WireDtype, micro: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let ports: Vec<String> = (0..world)
        .map(|_| {
            TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
                .unwrap()
                .local_addr()
                .unwrap()
                .to_string()
        })
        .collect();
    let n_micro = micro.len();
    let workers: Vec<_> = (1..world)
        .map(|r| {
            let p = ports.clone();
            std::thread::spawn(move || {
                let g: Arc<ProcessGroup> = Node::new(r, world)
                    .peers(p.iter().map(String::as_str))
                    .unwrap()
                    .connect()
                    .unwrap();
                let mut f = stage(r);
                dist::serve_stage_stream(&g, &mut f, n_micro, wire).unwrap();
                g.barrier().unwrap();
            })
        })
        .collect();
    let g = Node::new(0, world)
        .peers(ports.iter().map(String::as_str))
        .unwrap()
        .connect()
        .unwrap();
    let mut f0 = stage(0);
    let out = dist::pipeline_run(&g, &mut f0, &micro, wire).unwrap();
    g.barrier().unwrap();
    for w in workers {
        w.join().unwrap();
    }
    out
}

fn reference(world: u32, x: &[f32]) -> Vec<f32> {
    (0..world).fold(x.to_vec(), |h, r| stage(r)(&h))
}

#[test]
fn streams_every_micro_batch_in_order_through_every_stage() {
    let micro: Vec<Vec<f32>> = (0..24)
        .map(|i| (0..64).map(|j| (i * 64 + j) as f32 * 1e-3).collect())
        .collect();
    let out = run(3, WireDtype::F32, micro.clone());
    assert_eq!(out.len(), micro.len());
    for (o, x) in out.iter().zip(&micro) {
        assert_eq!(o, &reference(3, x));
    }
}

#[test]
fn sixteen_bit_activations_stay_within_their_precision() {
    let micro: Vec<Vec<f32>> = (0..8)
        .map(|i| {
            (0..32)
                .map(|j| ((i * 32 + j) as f32 * 0.37).sin())
                .collect()
        })
        .collect();
    // 3 hops, each rounding to the wire: ≤ 3·2⁻⁸ relative for bf16 (8-bit
    // mantissa), ≤ 3·2⁻¹¹ for f16.
    for (wire, tol) in [(WireDtype::Bf16, 2e-2), (WireDtype::F16, 2e-3)] {
        let out = run(3, wire, micro.clone());
        for (o, x) in out.iter().zip(&micro) {
            let r = reference(3, x);
            for (a, b) in o.iter().zip(&r) {
                assert!(
                    (a - b).abs() <= tol * b.abs().max(1.0),
                    "{wire:?}: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn single_rank_runs_stage_zero_locally() {
    let out = run(1, WireDtype::F32, vec![vec![1.0, 2.0]]);
    assert_eq!(out, vec![vec![2.0, 4.0]]);
}
