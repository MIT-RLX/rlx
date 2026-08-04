//! End-to-end loopback test for the MoE expert-parallel offload: a 2-rank
//! `TcpTransport` (rank 0 = orchestrator, rank 1 = worker). The orchestrator
//! dispatches one MoE layer; the worker computes via a deterministic
//! `ExpertProvider`; the orchestrator gathers + sums. Exercises the real wire
//! path (encode → send → recv → decode → compute → encode → recv → sum).

use rlx_distributed::{
    ExpertProvider, ExpertShards, TcpTransport, dispatch_experts, free_loopback_ports,
    serve_expert_worker, shutdown_expert_workers,
};
use std::net::{Ipv4Addr, SocketAddr};

/// Deterministic provider: `out[token] += Σ_slot prob · id · mn[token]` over owned
/// experts. Owns every expert here (single worker).
struct Dummy;
impl ExpertProvider for Dummy {
    fn owns(&self, _e: u32) -> bool {
        true
    }
    fn compute(
        &mut self,
        _layer: u32,
        mn: &[f32],
        rows: usize,
        hidden: usize,
        ids: &[u32],
        probs: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        let k = ids.len() / rows;
        let mut out = vec![0f32; rows * hidden];
        for r in 0..rows {
            for s in 0..k {
                let e = ids[r * k + s] as f32;
                let p = probs[r * k + s];
                for h in 0..hidden {
                    out[r * hidden + h] += p * e * mn[r * hidden + h];
                }
            }
        }
        Ok(out)
    }
}

#[test]
fn expert_offload_loopback_roundtrip() {
    const HEAP: usize = 64 * 1024 * 1024;
    let ports = free_loopback_ports(2).expect("ports");
    let peers: Vec<SocketAddr> = ports
        .iter()
        .map(|&p| SocketAddr::from((Ipv4Addr::LOCALHOST, p)))
        .collect();

    // rank 1 = worker (concurrent bind → mesh handshake).
    let peers_w = peers.clone();
    let worker = std::thread::spawn(move || {
        let t = TcpTransport::bind(1, 2, peers_w, HEAP).expect("worker bind");
        let mut p = Dummy;
        serve_expert_worker(&t, 0, &mut p).expect("serve");
    });

    // rank 0 = orchestrator.
    let t = TcpTransport::bind(0, 2, peers, HEAP).expect("orch bind");
    let shards = ExpertShards::round_robin(8, &[1]); // all experts → worker rank 1
    let (rows, hidden) = (2usize, 4usize);
    let ids = vec![1u32, 3, 5, 7]; // rows=2, top_k=2
    let probs = vec![0.5f32, 0.5, 0.25, 0.75];
    let mn = vec![2.0f32; rows * hidden];

    let out = dispatch_experts(&t, &shards, 0, &mn, rows, hidden, &ids, &probs).expect("dispatch");

    // token0: (0.5·1 + 0.5·3)·2 = 4.0 ; token1: (0.25·5 + 0.75·7)·2 = 13.0
    assert_eq!(out.len(), rows * hidden);
    for h in 0..hidden {
        assert!((out[h] - 4.0).abs() < 1e-4, "tok0[{h}] = {}", out[h]);
        assert!(
            (out[hidden + h] - 13.0).abs() < 1e-4,
            "tok1[{h}] = {}",
            out[hidden + h]
        );
    }

    shutdown_expert_workers(&t, &[1]).expect("shutdown");
    worker.join().expect("worker join");
}
