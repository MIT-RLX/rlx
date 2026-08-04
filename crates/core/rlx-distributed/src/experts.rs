// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Model-agnostic **MoE expert-parallel offload**.
//!
//! The pattern (complements the layer-pipeline [`crate::pipeline`]): ONE node runs
//! the sequential/recurrent backbone (attention + router), and the routed-expert
//! FFNs — the bandwidth-heavy, embarrassingly-parallel part — are dispatched to
//! **worker** ranks that hold the experts, sharded by expert id, on their LOCAL
//! storage. Only the hidden state + fired expert ids cross the wire (KB/token); the
//! experts (which can be TBs) never do. Each worker computes the contribution of
//! the experts IT owns; the orchestrator sums the partials.
//!
//! This is reusable by ANY MoE model (DeepSeek, Llama-4, Kimi, …) via one seam,
//! [`ExpertProvider`] — the model supplies how to page + compute a set of experts
//! for a layer; everything else (sharding, transport, gather/sum) lives here.
//!
//! Wiring: the orchestrator calls [`dispatch_experts`] per MoE layer; each worker
//! runs [`serve_expert_worker`]; [`shutdown_expert_workers`] ends the loop.

use anyhow::Result;
use rlx_driver::Transport;

/// Message tags on the [`Transport`].
const TAG_REQ: u32 = 0xE00;
const TAG_RESP: u32 = 0xE01;
/// Sentinel `layer` that tells a worker to stop serving.
const LAYER_SHUTDOWN: u32 = u32::MAX;

/// The model-specific half of expert offload: compute the summed contribution of
/// the experts THIS worker owns for one MoE layer.
///
/// - `mn`: FFN input, row-major `[rows * hidden]`.
/// - `ids` / `probs`: routing, row-major `[rows * top_k]` (each token's fired
///   expert ids and gate weights); `top_k = ids.len() / rows`.
/// - returns `[rows * hidden]` — `Σ_{token, slot : owns(id)} prob · expert_id(mn_token)`,
///   i.e. zero-initialised and accumulating only the OWNED experts (the orchestrator
///   sums across workers, so non-owned slots must contribute nothing here).
///
/// The provider holds its shard (`owns`) and pages experts from local storage; how
/// it computes (native `ScaledGroupedMatMul`, a fused dequant-GEMM, dense, …) is up
/// to the model.
pub trait ExpertProvider: Send {
    /// Does this worker hold expert `e` locally?
    fn owns(&self, e: u32) -> bool;
    /// Compute the owned-expert partial for `layer` (see trait docs).
    fn compute(
        &mut self,
        layer: u32,
        mn: &[f32],
        rows: usize,
        hidden: usize,
        ids: &[u32],
        probs: &[f32],
    ) -> Result<Vec<f32>>;
}

/// Static expert→rank shard map (`rank_of[expert_id]` = owning worker rank).
/// Flat/quantile-balanced routing ⇒ a round-robin or bandwidth-weighted split.
/// An expert mapped to [`ExpertShards::LOCAL`] is NOT dispatched — the orchestrator
/// owns it locally (e.g. an overflow shard that didn't fit on any worker).
#[derive(Clone, Debug)]
pub struct ExpertShards {
    pub rank_of: Vec<u32>,
}

impl ExpertShards {
    /// Sentinel rank: this expert is served by the ORCHESTRATOR locally, not a worker.
    pub const LOCAL: u32 = u32::MAX;

    /// Round-robin `num_experts` across `worker_ranks`.
    pub fn round_robin(num_experts: usize, worker_ranks: &[u32]) -> Self {
        assert!(!worker_ranks.is_empty(), "no worker ranks");
        let rank_of = (0..num_experts)
            .map(|e| worker_ranks[e % worker_ranks.len()])
            .collect();
        Self { rank_of }
    }

    /// Assign `num_experts` to `worker_ranks` in proportion to `weights` (e.g. NVMe
    /// bandwidth), contiguously — the higher-weight nodes get more experts.
    pub fn weighted(num_experts: usize, worker_ranks: &[u32], weights: &[f64]) -> Self {
        assert_eq!(worker_ranks.len(), weights.len());
        let total: f64 = weights.iter().sum();
        let mut rank_of = Vec::with_capacity(num_experts);
        let (mut w, mut acc) = (0usize, weights[0] / total * num_experts as f64);
        for e in 0..num_experts {
            while (e as f64) >= acc && w + 1 < worker_ranks.len() {
                w += 1;
                acc += weights[w] / total * num_experts as f64;
            }
            rank_of.push(worker_ranks[w]);
        }
        Self { rank_of }
    }

    /// Distinct WORKER ranks that own at least one of `ids` (skips [`Self::LOCAL`]
    /// experts, which the orchestrator computes itself).
    pub fn owners(&self, ids: &[u32]) -> Vec<u32> {
        let mut r: Vec<u32> = ids
            .iter()
            .filter_map(|&e| self.rank_of.get(e as usize).copied())
            .filter(|&rank| rank != Self::LOCAL)
            .collect();
        r.sort_unstable();
        r.dedup();
        r
    }
}

// ── wire encoding (length-prefixed LE, mirrors graph/transport.rs) ──────────────

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn get_u32(b: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes([b[*o], b[*o + 1], b[*o + 2], b[*o + 3]]);
    *o += 4;
    v
}
fn put_f32s(b: &mut Vec<u8>, xs: &[f32]) {
    for &x in xs {
        b.extend_from_slice(&x.to_le_bytes());
    }
}
fn get_f32s(b: &[u8], o: &mut usize, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(f32::from_le_bytes([b[*o], b[*o + 1], b[*o + 2], b[*o + 3]]));
        *o += 4;
    }
    v
}

/// req = [layer][rows][hidden][k][ids:k×u32][probs:k×f32][mn:rows*hidden×f32]
fn encode_req(
    layer: u32,
    rows: usize,
    hidden: usize,
    ids: &[u32],
    probs: &[f32],
    mn: &[f32],
) -> Vec<u8> {
    let mut b = Vec::with_capacity(16 + ids.len() * 8 + mn.len() * 4);
    put_u32(&mut b, layer);
    put_u32(&mut b, rows as u32);
    put_u32(&mut b, hidden as u32);
    put_u32(&mut b, ids.len() as u32);
    for &e in ids {
        put_u32(&mut b, e);
    }
    put_f32s(&mut b, probs);
    put_f32s(&mut b, mn);
    b
}

#[allow(clippy::type_complexity)]
fn decode_req(b: &[u8]) -> (u32, usize, usize, Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut o = 0usize;
    let layer = get_u32(b, &mut o);
    let rows = get_u32(b, &mut o) as usize;
    let hidden = get_u32(b, &mut o) as usize;
    let k = get_u32(b, &mut o) as usize;
    if layer == LAYER_SHUTDOWN {
        return (layer, 0, 0, Vec::new(), Vec::new(), Vec::new());
    }
    let ids: Vec<u32> = (0..k).map(|_| get_u32(b, &mut o)).collect();
    let probs = get_f32s(b, &mut o, k);
    let mn = get_f32s(b, &mut o, rows * hidden);
    (layer, rows, hidden, ids, probs, mn)
}

fn t_send(t: &dyn Transport, to: u32, tag: u32, bytes: &[u8]) -> Result<()> {
    t.send_bytes(to, tag, bytes)
        .map_err(|e| anyhow::anyhow!("expert transport send: {e:?}"))
}
fn t_recv(t: &dyn Transport, from: u32, tag: u32) -> Result<Vec<u8>> {
    t.recv_bytes(from, tag)
        .map_err(|e| anyhow::anyhow!("expert transport recv: {e:?}"))
}

/// **Orchestrator**: dispatch one MoE layer's routed experts to the workers that
/// own them, then gather + SUM the partials. Returns `[rows*hidden]`. Experts stay
/// local to each worker; only `mn` + ids cross the wire.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_experts(
    t: &dyn Transport,
    shards: &ExpertShards,
    layer: u32,
    mn: &[f32],
    rows: usize,
    hidden: usize,
    ids: &[u32],
    probs: &[f32],
) -> Result<Vec<f32>> {
    let owners = shards.owners(ids);
    let req = encode_req(layer, rows, hidden, ids, probs, mn);
    for &w in &owners {
        t_send(t, w, TAG_REQ, &req)?;
    }
    let mut acc = vec![0f32; rows * hidden];
    for &w in &owners {
        let resp = t_recv(t, w, TAG_RESP)?;
        let mut o = 0usize;
        let part = get_f32s(&resp, &mut o, rows * hidden);
        for (a, b) in acc.iter_mut().zip(&part) {
            *a += *b;
        }
    }
    Ok(acc)
}

/// **Worker**: serve expert requests from `orchestrator` until a shutdown sentinel,
/// computing each layer's owned-expert partial via `provider` and returning it.
pub fn serve_expert_worker(
    t: &dyn Transport,
    orchestrator: u32,
    provider: &mut dyn ExpertProvider,
) -> Result<()> {
    loop {
        let bytes = t_recv(t, orchestrator, TAG_REQ)?;
        let (layer, rows, hidden, ids, probs, mn) = decode_req(&bytes);
        if layer == LAYER_SHUTDOWN {
            return Ok(());
        }
        let out = provider.compute(layer, &mn, rows, hidden, &ids, &probs)?;
        debug_assert_eq!(out.len(), rows * hidden, "expert partial wrong length");
        let mut b = Vec::with_capacity(out.len() * 4);
        put_f32s(&mut b, &out);
        t_send(t, orchestrator, TAG_RESP, &b)?;
    }
}

/// **Orchestrator**: tell every worker rank to stop serving.
pub fn shutdown_expert_workers(t: &dyn Transport, worker_ranks: &[u32]) -> Result<()> {
    let mut b = Vec::new();
    put_u32(&mut b, LAYER_SHUTDOWN);
    put_u32(&mut b, 0);
    put_u32(&mut b, 0);
    put_u32(&mut b, 0);
    for &w in worker_ranks {
        t_send(t, w, TAG_REQ, &b)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shards_round_robin_and_weighted() {
        let s = ExpertShards::round_robin(8, &[1, 2]);
        assert_eq!(s.rank_of, vec![1, 2, 1, 2, 1, 2, 1, 2]);
        assert_eq!(s.owners(&[0, 1, 2, 4]), vec![1, 2]);
        // weighted 3:1 → ~6 experts to rank 1, ~2 to rank 2.
        let w = ExpertShards::weighted(8, &[1, 2], &[3.0, 1.0]);
        let n1 = w.rank_of.iter().filter(|&&r| r == 1).count();
        assert!((5..=6).contains(&n1), "got {n1}");
    }

    #[test]
    fn req_roundtrips() {
        let (rows, hidden) = (2usize, 4usize);
        let ids = vec![3u32, 7, 1, 9];
        let probs = vec![0.5f32, 0.5, 0.25, 0.75];
        let mn: Vec<f32> = (0..rows * hidden).map(|i| i as f32 * 0.1).collect();
        let enc = encode_req(5, rows, hidden, &ids, &probs, &mn);
        let (l, r, h, i2, p2, m2) = decode_req(&enc);
        assert_eq!((l, r, h), (5, rows, hidden));
        assert_eq!(i2, ids);
        assert_eq!(p2, probs);
        assert_eq!(m2, mn);
    }

    #[test]
    fn shutdown_decodes() {
        let mut b = Vec::new();
        put_u32(&mut b, LAYER_SHUTDOWN);
        for _ in 0..3 {
            put_u32(&mut b, 0);
        }
        let (l, ..) = decode_req(&b);
        assert_eq!(l, LAYER_SHUTDOWN);
    }
}
