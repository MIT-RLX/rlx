// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A heterogeneous mesh where one rank is **pre-synthesized hardware**.
//!
//! rank 0 (coordinator) → rank 1 (general worker, compiles a shipped stage)
//!                      → rank 2 (fixed-function rank, e.g. an FPGA bitstream)
//!                      → rank 0
//!
//! The fixed-function rank stands in for a board via [`LoopbackFixedFunction`],
//! so the mesh-side contract — capability handshake, placement refusal, shape
//! enforcement, activation streaming — is exercised with no hardware attached.

use rlx_driver::{NetTransport, ProcessGroup};
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::dist::node::{
    FixedFunction, LoopbackFixedFunction, NodeControl, NodeRole, collect_caps,
    serve_fixed_function_n, serve_worker_n,
};
use rlx_runtime::dist::{StageSpec, recv_activation, send_activation, ship_stage};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;
use std::thread;

const N: usize = 4;
const STAGE: &str = "relu4-int8-v1";

/// The stage the board implements: ×3 on 4 elements.
fn board_fn(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v * 3.0).collect()
}

/// The stage the general worker compiles: relu on 4 elements.
fn worker_graph() -> Graph {
    let mut g = Graph::new("relu_stage");
    let shape = Shape::new(&[1, N], DType::F32);
    let x = g.input("x", shape.clone());
    let y = g.activation(Activation::Relu, x, shape);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn fixed_function_rank_joins_mesh_and_streams() {
    let world: u32 = 3;
    let listeners: Vec<TcpListener> = (0..world)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();

    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let addrs = addrs.clone();
            thread::spawn(move || -> Result<Vec<f32>, String> {
                let rank = rank as u32;
                let t = NetTransport::from_listener(rank, world, listener, addrs, 1 << 20)
                    .map_err(|e| e.to_string())?;
                let group = Arc::new(ProcessGroup::new(Arc::new(t)));

                match rank {
                    // ── coordinator ──────────────────────────────────────
                    0 => {
                        let caps = collect_caps(&group)?;
                        assert_eq!(caps.len(), 2, "expected caps from ranks 1..3");

                        // The board rank must advertise itself as fixed-function.
                        let fpga = caps
                            .iter()
                            .find(|c| matches!(c.role, NodeRole::FixedFunction(_)))
                            .expect("no fixed-function rank announced");
                        assert_eq!(fpga.rank, 2);

                        // Placement must REFUSE a stage the bitstream doesn't
                        // implement — the whole point of the handshake.
                        let wrong = StageSpec::new(worker_graph(), "x", "y", "cpu")
                            .stage_id("some-other-datapath");
                        let err = fpga.accepts(&wrong).unwrap_err();
                        assert!(
                            err.contains(STAGE) && err.contains("some-other-datapath"),
                            "refusal should name both stages, got: {err}"
                        );

                        // An untagged stage is also refused (a general stage
                        // cannot be silently clocked into a fixed datapath).
                        let untagged = StageSpec::new(worker_graph(), "x", "y", "cpu");
                        assert!(fpga.accepts(&untagged).is_err());

                        // The matching stage is accepted.
                        let right = StageSpec::new(worker_graph(), "x", "y", "cpu").stage_id(STAGE);
                        fpga.accepts(&right).expect("matching stage_id must accept");

                        // A general worker accepts anything.
                        let general = caps
                            .iter()
                            .find(|c| matches!(c.role, NodeRole::Worker))
                            .expect("no general worker announced");
                        general
                            .accepts(&untagged)
                            .expect("worker accepts any stage");

                        // Drive one activation through the pipeline.
                        ship_stage(&group, 1, &untagged)?;
                        send_activation(&group, 1, &[-1.0, 2.0, -3.0, 4.0])?;
                        let out = recv_activation(&group, 2)?;
                        Ok(out)
                    }
                    // ── general worker: compiles the shipped stage ────────
                    1 => {
                        let caps = rlx_runtime::dist::node::NodeCaps {
                            rank,
                            role: NodeRole::Worker,
                            devices: vec!["cpu".into()],
                            platform: rlx_runtime::dist::node::platform_tag().into(),
                        };
                        serve_worker_n(&group, &caps, |_| Vec::new(), &NodeControl::steps(1))?;
                        Ok(Vec::new())
                    }
                    // ── fixed-function rank: pre-synthesized datapath ─────
                    _ => {
                        let ff = FixedFunction::new(STAGE, N, N, DType::F32);
                        let mut board = LoopbackFixedFunction::new(STAGE, board_fn);
                        let report = serve_fixed_function_n(
                            &group,
                            &ff,
                            &mut board,
                            &NodeControl::steps(1),
                        )?;
                        assert_eq!(report.activations, 1);
                        Ok(Vec::new())
                    }
                }
            })
        })
        .collect();

    let mut out = Vec::new();
    for (i, h) in handles.into_iter().enumerate() {
        let r = h.join().expect("thread panicked");
        let v = r.unwrap_or_else(|e| panic!("rank {i}: {e}"));
        if i == 0 {
            out = v;
        }
    }

    // relu([-1,2,-3,4]) = [0,2,0,4], then ×3 on the board = [0,6,0,12].
    assert_eq!(out, vec![0.0, 6.0, 0.0, 12.0]);
}

/// A fixed-function rank refuses an activation whose length its datapath was
/// not synthesized for, instead of clocking a wrong-shaped feed into fabric.
#[test]
fn fixed_function_rank_rejects_wrong_shape() {
    let world: u32 = 2;
    let listeners: Vec<TcpListener> = (0..world)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();

    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let addrs = addrs.clone();
            thread::spawn(move || -> Result<(), String> {
                let rank = rank as u32;
                let t = NetTransport::from_listener(rank, world, listener, addrs, 1 << 20)
                    .map_err(|e| e.to_string())?;
                let group = Arc::new(ProcessGroup::new(Arc::new(t)));
                if rank == 0 {
                    let _ = collect_caps(&group)?;
                    // 5 elements into a datapath synthesized for 4.
                    send_activation(&group, 1, &[1.0, 2.0, 3.0, 4.0, 5.0])?;
                    Ok(())
                } else {
                    let ff = FixedFunction::new(STAGE, N, N, DType::F32);
                    let mut board = LoopbackFixedFunction::new(STAGE, board_fn);
                    let err =
                        serve_fixed_function_n(&group, &ff, &mut board, &NodeControl::steps(1))
                            .expect_err("wrong-length activation must be refused");
                    assert!(
                        err.contains("takes 4 elems, got 5"),
                        "unexpected error: {err}"
                    );
                    Ok(())
                }
            })
        })
        .collect();

    for (i, h) in handles.into_iter().enumerate() {
        h.join()
            .expect("thread panicked")
            .unwrap_or_else(|e| panic!("rank {i}: {e}"));
    }
}

/// A board whose stage id disagrees with the rank's advertised descriptor is a
/// configuration error, caught before the mesh handshake.
#[test]
fn board_stage_id_must_match_descriptor() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let addr = listener.local_addr().unwrap();
    let t = NetTransport::from_listener(0, 1, listener, vec![addr], 1 << 16).unwrap();
    let group = Arc::new(ProcessGroup::new(Arc::new(t)));

    let ff = FixedFunction::new(STAGE, N, N, DType::F32);
    let mut board = LoopbackFixedFunction::new("a-different-bitstream", board_fn);
    let err = serve_fixed_function_n(&group, &ff, &mut board, &NodeControl::steps(1)).unwrap_err();
    assert!(
        err.contains("a-different-bitstream") && err.contains(STAGE),
        "unexpected error: {err}"
    );
}

/// A fixed-function rank must be refused a **training** job, at placement.
///
/// This is the failure the capability handshake exists to prevent, and it is
/// silent: a `TrainSpec` shipped to a bitstream lands on a tag that rank never
/// reads, and the coordinator then waits forever on the first gradient
/// all-reduce, because a barrier needs every rank to arrive. No error, no
/// timeout — the run simply stops.
///
/// The test never ships anything, so a regression here fails rather than hangs.
#[test]
fn a_fixed_function_rank_is_refused_a_training_job() {
    let world: u32 = 2;
    let listeners: Vec<TcpListener> = (0..world)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();

    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let addrs = addrs.clone();
            thread::spawn(move || -> Result<(), String> {
                let rank = rank as u32;
                let t = NetTransport::from_listener(rank, world, listener, addrs, 1 << 20)
                    .map_err(|e| e.to_string())?;
                let group = Arc::new(ProcessGroup::new(Arc::new(t)));
                if rank == 0 {
                    let caps = collect_caps(&group)?;
                    let fpga = caps
                        .iter()
                        .find(|c| matches!(c.role, NodeRole::FixedFunction(_)))
                        .expect("the board announced itself");
                    let err = fpga
                        .can_train()
                        .expect_err("a bitstream has no backward pass");
                    assert!(
                        err.contains(STAGE) && err.contains("cannot"),
                        "refusal should name the stage and say why: {err}"
                    );
                    // A general worker is still allowed to train.
                    let worker = rlx_runtime::dist::node::NodeCaps {
                        rank: 9,
                        role: NodeRole::Worker,
                        devices: vec!["cpu".into()],
                        platform: "test".into(),
                    }
                    .can_train();
                    assert!(worker.is_ok(), "a worker must still be allowed: {worker:?}");
                    Ok(())
                } else {
                    // Announce as a board, then leave — nothing is shipped.
                    let ff = FixedFunction::new(STAGE, N, N, DType::F32);
                    let caps = rlx_runtime::dist::node::NodeCaps {
                        rank,
                        role: NodeRole::FixedFunction(ff),
                        devices: vec!["fpga".into()],
                        platform: "test".into(),
                    };
                    rlx_runtime::dist::node::send_caps(&group, &caps)
                }
            })
        })
        .collect();

    for (i, h) in handles.into_iter().enumerate() {
        h.join()
            .unwrap_or_else(|_| panic!("rank {i} panicked"))
            .unwrap_or_else(|e| panic!("rank {i}: {e}"));
    }
}
