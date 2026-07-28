// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! WideEP EPLB: expert-parallel load balancing across ranks.
//!
//! Unlike TIDE ([`rlx_runtime::ExpertPool`]), which chooses which experts are
//! **device-resident** on one GPU, EPLB chooses which **rank owns** each
//! expert. Hits are summed across the EP group, then experts are reassigned
//! so predicted token load is balanced while keeping a fixed per-rank shard
//! size (`num_experts / world_size`).
//!
//! When `num_slots > num_experts`, [`rebalance_with_replicas`] places extra
//! copies of hot experts on under-loaded ranks (DeepSeek-style EPLB slots).
//! Dispatch picks a replica with [`pick_dispatch_rank`].
//!
//! Online updates: [`register_ep_placement`] / [`register_ep_replicas`]
//! override attrs-baked placement for MoE EP kernels without recompiling.
//! After a remap, [`migrate_to_replica_map`] exchanges weight slabs over
//! `all_to_all_v` so ranks need not keep a full expert bank.

use crate::moe_ep::MoeEpConfig;
use rlx_driver::{CollectiveError, ProcessGroup, ReduceKind};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

fn placements() -> &'static RwLock<HashMap<u64, Arc<[u32]>>> {
    static P: OnceLock<RwLock<HashMap<u64, Arc<[u32]>>>> = OnceLock::new();
    P.get_or_init(|| RwLock::new(HashMap::new()))
}

fn replica_maps() -> &'static RwLock<HashMap<u64, Arc<EpReplicaMap>>> {
    static R: OnceLock<RwLock<HashMap<u64, Arc<EpReplicaMap>>>> = OnceLock::new();
    R.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Expert→rank multi-map plus per-rank local slot lists (GroupedMatMul order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpReplicaMap {
    /// `owners[e]` = ranks holding a copy of expert `e` (non-empty).
    pub owners: Vec<Vec<u32>>,
    /// `local_slots[rank]` = global expert ids in this rank's weight bank order.
    pub local_slots: Vec<Vec<u32>>,
}

impl EpReplicaMap {
    pub fn num_experts(&self) -> usize {
        self.owners.len()
    }

    pub fn world_size(&self) -> usize {
        self.local_slots.len()
    }

    pub fn slots_per_rank(&self) -> usize {
        self.local_slots.first().map(|s| s.len()).unwrap_or(0)
    }
}

/// Publish an expert→rank map for `group_id` (read by MoE EP kernels).
/// Clears any replica map for the same id.
pub fn register_ep_placement(group_id: u64, placement: Vec<u32>) {
    replica_maps().write().unwrap().remove(&group_id);
    placements()
        .write()
        .unwrap()
        .insert(group_id, Arc::from(placement));
}

/// Publish a replica map (takes precedence over single-owner placement).
pub fn register_ep_replicas(group_id: u64, map: EpReplicaMap) {
    placements().write().unwrap().remove(&group_id);
    replica_maps()
        .write()
        .unwrap()
        .insert(group_id, Arc::new(map));
}

/// Drop runtime placement / replica map for `group_id`.
pub fn unregister_ep_placement(group_id: u64) {
    placements().write().unwrap().remove(&group_id);
    replica_maps().write().unwrap().remove(&group_id);
}

/// Alias for clearing replica registration (same as [`unregister_ep_placement`]).
pub fn unregister_ep_replicas(group_id: u64) {
    unregister_ep_placement(group_id);
}

/// Current runtime single-owner placement, if any.
pub fn lookup_ep_placement(group_id: u64) -> Option<Arc<[u32]>> {
    placements().read().unwrap().get(&group_id).cloned()
}

/// Current runtime replica map, if any.
pub fn lookup_ep_replicas(group_id: u64) -> Option<Arc<EpReplicaMap>> {
    replica_maps().read().unwrap().get(&group_id).cloned()
}

/// Default static map: expert `e` → rank `e % world_size`.
pub fn default_placement(num_experts: u32, world_size: u32) -> Vec<u32> {
    assert!(world_size >= 1);
    (0..num_experts).map(|e| e % world_size).collect()
}

/// Resolve expert→primary-rank for a MoE EP config (first owner if replicas).
pub fn resolve_placement(cfg: &MoeEpConfig) -> Vec<u32> {
    if let Some(map) = lookup_ep_replicas(cfg.group_id) {
        return map.owners.iter().map(|o| o[0]).collect();
    }
    if let Some(p) = lookup_ep_placement(cfg.group_id) {
        assert_eq!(
            p.len(),
            cfg.num_experts as usize,
            "ep placement len must equal num_experts"
        );
        return p.to_vec();
    }
    if !cfg.placement.is_empty() {
        return cfg.placement.clone();
    }
    default_placement(cfg.num_experts, cfg.world_size)
}

/// Owner rank of `expert` (primary / sole owner). Prefer [`pick_dispatch_rank`]
/// when replicas may be registered.
pub fn owner_of(cfg: &MoeEpConfig, expert: u32) -> u32 {
    pick_dispatch_rank(cfg, expert, 0)
}

/// Choose a destination rank for `expert` on token row `token_row`.
///
/// With replicas: `owners[e][token_row % n]`. Without: single-owner placement.
pub fn pick_dispatch_rank(cfg: &MoeEpConfig, expert: u32, token_row: u32) -> u32 {
    if let Some(map) = lookup_ep_replicas(cfg.group_id) {
        let o = &map.owners[expert as usize];
        debug_assert!(!o.is_empty());
        return o[(token_row as usize) % o.len()];
    }
    let p = resolve_placement(cfg);
    p[expert as usize]
}

/// Local GroupedMatMul index of `expert` on `dest_rank`.
pub fn local_id_on_rank(cfg: &MoeEpConfig, expert: u32, dest_rank: u32) -> u32 {
    if let Some(map) = lookup_ep_replicas(cfg.group_id) {
        let slots = &map.local_slots[dest_rank as usize];
        return slots
            .iter()
            .position(|&e| e == expert)
            .expect("expert missing from dest_rank local_slots") as u32;
    }
    let p = resolve_placement(cfg);
    let owner = p[expert as usize];
    debug_assert_eq!(owner, dest_rank);
    let mut local = 0u32;
    for e in 0..=expert {
        if p[e as usize] == owner {
            if e == expert {
                return local;
            }
            local += 1;
        }
    }
    0
}

/// Local index of `expert` among experts owned by its primary owner rank.
pub fn local_id_on_owner(cfg: &MoeEpConfig, expert: u32) -> u32 {
    let dest = pick_dispatch_rank(cfg, expert, 0);
    local_id_on_rank(cfg, expert, dest)
}

/// Experts / slots on `rank` (GroupedMatMul order).
pub fn experts_on_rank(cfg: &MoeEpConfig, rank: u32) -> Vec<u32> {
    if let Some(map) = lookup_ep_replicas(cfg.group_id) {
        return map.local_slots[rank as usize].clone();
    }
    let p = resolve_placement(cfg);
    (0..cfg.num_experts)
        .filter(|&e| p[e as usize] == rank)
        .collect()
}

/// Rebalance so each rank owns exactly `num_experts / world_size` experts
/// and predicted load (`sum of hits`) is as even as possible.
///
/// Requires `num_experts % world_size == 0`. Experts are considered in
/// descending hit order and assigned to the eligible rank with the lowest
/// current load (tie-break: lowest rank id).
pub fn rebalance_placement(hits: &[u64], world_size: u32) -> Result<Vec<u32>, String> {
    let num_experts = hits.len();
    let w = world_size as usize;
    if w == 0 {
        return Err("rebalance_placement: world_size must be >= 1".into());
    }
    if num_experts == 0 {
        return Err("rebalance_placement: num_experts must be >= 1".into());
    }
    if !num_experts.is_multiple_of(w) {
        return Err(format!(
            "rebalance_placement: num_experts {num_experts} not divisible by world {w}"
        ));
    }
    let per_rank = num_experts / w;
    let mut order: Vec<usize> = (0..num_experts).collect();
    order.sort_by(|&a, &b| hits[b].cmp(&hits[a]).then_with(|| a.cmp(&b)));

    let mut placement = vec![0u32; num_experts];
    let mut load = vec![0u64; w];
    let mut count = vec![0usize; w];

    for &e in &order {
        let mut best = None;
        for r in 0..w {
            if count[r] >= per_rank {
                continue;
            }
            match best {
                None => best = Some(r),
                Some(b) => {
                    if load[r] < load[b] || (load[r] == load[b] && r < b) {
                        best = Some(r);
                    }
                }
            }
        }
        let r = best.ok_or_else(|| "rebalance_placement: no rank with free slots".to_string())?;
        placement[e] = r as u32;
        load[r] = load[r].saturating_add(hits[e]);
        count[r] += 1;
    }
    Ok(placement)
}

/// Like [`rebalance_placement`], then fill `num_slots - num_experts` extra slots
/// with replicas of the hottest experts on ranks that do not already hold them.
///
/// Requires `num_slots >= num_experts`, `num_slots % world_size == 0`, and
/// `num_experts % world_size == 0`.
pub fn rebalance_with_replicas(
    hits: &[u64],
    world_size: u32,
    num_slots: u32,
) -> Result<EpReplicaMap, String> {
    let num_experts = hits.len();
    let w = world_size as usize;
    let slots = num_slots as usize;
    if slots < num_experts {
        return Err(format!(
            "rebalance_with_replicas: num_slots {slots} < num_experts {num_experts}"
        ));
    }
    if !slots.is_multiple_of(w) {
        return Err(format!(
            "rebalance_with_replicas: num_slots {slots} not divisible by world {w}"
        ));
    }
    let primary = rebalance_placement(hits, world_size)?;
    let slots_per_rank = slots / w;

    let mut owners: Vec<Vec<u32>> = (0..num_experts).map(|e| vec![primary[e]]).collect();
    let mut local_slots: Vec<Vec<u32>> = vec![Vec::new(); w];
    for e in 0..num_experts {
        local_slots[primary[e] as usize].push(e as u32);
    }
    let mut free: Vec<usize> = local_slots
        .iter()
        .map(|s| slots_per_rank.saturating_sub(s.len()))
        .collect();
    let mut load: Vec<u64> = (0..w)
        .map(|r| local_slots[r].iter().map(|&e| hits[e as usize]).sum())
        .collect();

    let mut order: Vec<usize> = (0..num_experts).collect();
    order.sort_by(|&a, &b| hits[b].cmp(&hits[a]).then_with(|| a.cmp(&b)));

    let extra = slots - num_experts;
    for _ in 0..extra {
        let mut placed = false;
        for &e in &order {
            let mut best: Option<usize> = None;
            for r in 0..w {
                if free[r] == 0 {
                    continue;
                }
                if local_slots[r].contains(&(e as u32)) {
                    continue;
                }
                match best {
                    None => best = Some(r),
                    Some(b) => {
                        if load[r] < load[b] || (load[r] == load[b] && r < b) {
                            best = Some(r);
                        }
                    }
                }
            }
            if let Some(r) = best {
                local_slots[r].push(e as u32);
                owners[e].push(r as u32);
                owners[e].sort_unstable();
                free[r] -= 1;
                load[r] = load[r].saturating_add(hits[e]);
                placed = true;
                break;
            }
        }
        if !placed {
            return Err(
                "rebalance_with_replicas: cannot place remaining replicas (no free remote slots)"
                    .into(),
            );
        }
    }

    for slots in &mut local_slots {
        slots.sort_unstable();
    }

    Ok(EpReplicaMap {
        owners,
        local_slots,
    })
}

/// Sum per-expert hit counts across the process group (`all_reduce` Sum).
///
/// Counts are exchanged as `f32` (exact for integer hits up to 2²⁴).
pub fn all_reduce_hits(
    group: &ProcessGroup,
    local_hits: &[u64],
) -> Result<Vec<u64>, CollectiveError> {
    let mut buf: Vec<f32> = local_hits.iter().map(|&c| c as f32).collect();
    group.all_reduce(&mut buf, ReduceKind::Sum)?;
    Ok(buf.iter().map(|&v| v as u64).collect())
}

/// Pack the expert weight slabs owned by `rank` from a full `[E, K, N]` bank.
pub fn shard_expert_weights(
    full: &[f32],
    num_experts: usize,
    k: usize,
    n: usize,
    placement: &[u32],
    rank: u32,
) -> Vec<f32> {
    assert_eq!(full.len(), num_experts * k * n);
    assert_eq!(placement.len(), num_experts);
    let stride = k * n;
    let mut out = Vec::new();
    for e in 0..num_experts {
        if placement[e] == rank {
            let start = e * stride;
            out.extend_from_slice(&full[start..start + stride]);
        }
    }
    out
}

/// Pack weight slabs for `local_slots` (global expert ids, GroupedMatMul order).
pub fn shard_expert_weights_slots(
    full: &[f32],
    num_experts: usize,
    k: usize,
    n: usize,
    local_slots: &[u32],
) -> Vec<f32> {
    assert_eq!(full.len(), num_experts * k * n);
    let stride = k * n;
    let mut out = Vec::with_capacity(local_slots.len() * stride);
    for &e in local_slots {
        let e = e as usize;
        assert!(e < num_experts);
        let start = e * stride;
        out.extend_from_slice(&full[start..start + stride]);
    }
    out
}

/// Build an [`EpReplicaMap`] from a single-owner placement (no replicas).
pub fn replica_map_from_placement(placement: &[u32], world_size: u32) -> EpReplicaMap {
    let num_experts = placement.len();
    let w = world_size as usize;
    let mut owners: Vec<Vec<u32>> = (0..num_experts).map(|e| vec![placement[e]]).collect();
    let mut local_slots: Vec<Vec<u32>> = vec![Vec::new(); w];
    for e in 0..num_experts {
        local_slots[placement[e] as usize].push(e as u32);
    }
    for slots in &mut local_slots {
        slots.sort_unstable();
    }
    // Keep owners sorted for determinism.
    for o in &mut owners {
        o.sort_unstable();
    }
    EpReplicaMap {
        owners,
        local_slots,
    }
}

/// Migrate this rank's expert weight bank from `old_map` layout to `new_map`.
///
/// Each expert's **primary** old owner (`owners[e][0]`) sends the slab to every
/// rank that holds `e` under `new_map` (via [`ProcessGroup::all_to_all_v`]).
/// Returns the packed `[slots_per_rank, K, N]` bank for `new_map.local_slots[rank]`.
///
/// Ranks must call this collectively with matching maps. `old_weights` must
/// match `old_map.local_slots[rank]` (row-major expert slabs of `k * n` f32).
pub fn migrate_to_replica_map(
    group: &ProcessGroup,
    old_map: &EpReplicaMap,
    old_weights: &[f32],
    new_map: &EpReplicaMap,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, CollectiveError> {
    let rank = group.rank() as usize;
    let w = group.world_size() as usize;
    if old_map.world_size() != w || new_map.world_size() != w {
        return Err(CollectiveError::TransportError {
            reason: format!(
                "migrate: map world {}/{} != group world {w}",
                old_map.world_size(),
                new_map.world_size()
            ),
        });
    }
    if old_map.num_experts() != new_map.num_experts() {
        return Err(CollectiveError::TransportError {
            reason: "migrate: num_experts mismatch between maps".into(),
        });
    }
    let stride = k * n;
    let old_slots = &old_map.local_slots[rank];
    if old_weights.len() != old_slots.len() * stride {
        return Err(CollectiveError::LengthMismatch {
            expected: old_slots.len() * stride,
            got: old_weights.len(),
        });
    }

    // Inventory of slabs this rank can source.
    let mut inventory: HashMap<u32, Vec<f32>> = HashMap::new();
    for (i, &e) in old_slots.iter().enumerate() {
        let slab = old_weights[i * stride..(i + 1) * stride].to_vec();
        inventory.insert(e, slab);
    }

    // Primary old owner sends to every new holder.
    let mut buckets: Vec<Vec<f32>> = (0..w).map(|_| Vec::new()).collect();
    // Parallel meta: list of expert ids in each send bucket (same order as slabs).
    let mut send_experts: Vec<Vec<u32>> = (0..w).map(|_| Vec::new()).collect();
    for e in 0..old_map.num_experts() {
        let primary = old_map.owners[e][0] as usize;
        if primary != rank {
            continue;
        }
        let slab = inventory
            .get(&(e as u32))
            .ok_or_else(|| CollectiveError::TransportError {
                reason: format!("migrate: primary rank {rank} missing expert {e}"),
            })?;
        for &dest in &new_map.owners[e] {
            let d = dest as usize;
            buckets[d].extend_from_slice(slab);
            send_experts[d].push(e as u32);
        }
    }

    // Exchange expert-id headers first (counts = number of u32 ids), then slabs.
    // Encode ids as f32 for a single all_to_all_v path (exact for ids < 2^24).
    let id_counts: Vec<usize> = send_experts.iter().map(|v| v.len()).collect();
    let mut id_send = Vec::new();
    for v in &send_experts {
        for &e in v {
            id_send.push(e as f32);
        }
    }
    let (id_recv, id_recv_counts) = group.all_to_all_v(&id_send, &id_counts)?;

    let slab_counts: Vec<usize> = buckets.iter().map(|b| b.len()).collect();
    let mut slab_send = Vec::with_capacity(slab_counts.iter().sum());
    for b in &buckets {
        slab_send.extend_from_slice(b);
    }
    let (slab_recv, slab_recv_counts) = group.all_to_all_v(&slab_send, &slab_counts)?;

    // Rebuild inventory from received slabs (overwrite/extend).
    let mut got: HashMap<u32, Vec<f32>> = HashMap::new();
    let mut id_off = 0usize;
    let mut slab_off = 0usize;
    for src in 0..w {
        let n_ids = id_recv_counts[src];
        let n_slab = slab_recv_counts[src];
        if n_ids * stride != n_slab {
            return Err(CollectiveError::TransportError {
                reason: format!(
                    "migrate: from {src}: {n_ids} ids but slab elems {n_slab} (stride {stride})"
                ),
            });
        }
        for i in 0..n_ids {
            let e = id_recv[id_off + i] as u32;
            let start = slab_off + i * stride;
            got.insert(e, slab_recv[start..start + stride].to_vec());
        }
        id_off += n_ids;
        slab_off += n_slab;
    }

    // Assemble new local bank.
    let new_slots = &new_map.local_slots[rank];
    let mut out = Vec::with_capacity(new_slots.len() * stride);
    for &e in new_slots {
        let slab = got.get(&e).ok_or_else(|| CollectiveError::TransportError {
            reason: format!("migrate: rank {rank} missing expert {e} after exchange"),
        })?;
        out.extend_from_slice(slab);
    }
    Ok(out)
}

/// Migrate between single-owner placements (wraps [`migrate_to_replica_map`]).
pub fn migrate_to_placement(
    group: &ProcessGroup,
    old_placement: &[u32],
    old_weights: &[f32],
    new_placement: &[u32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>, CollectiveError> {
    let w = group.world_size();
    let old_map = replica_map_from_placement(old_placement, w);
    let new_map = replica_map_from_placement(new_placement, w);
    migrate_to_replica_map(group, &old_map, old_weights, &new_map, k, n)
}

/// Count hits from f32 expert indices (GroupedMatMul / TopK convention).
pub fn count_hits_f32(expert_idx: &[f32], num_experts: usize) -> Vec<u64> {
    let mut counts = vec![0u64; num_experts];
    for &v in expert_idx {
        let e = v as usize;
        if e < num_experts {
            counts[e] += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_driver::{NetTransport, ProcessGroup};
    use std::net::{Ipv4Addr, SocketAddr, TcpListener};
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn default_placement_round_robin() {
        assert_eq!(default_placement(4, 2), vec![0, 1, 0, 1]);
    }

    #[test]
    fn rebalance_spreads_hot_experts() {
        let hits = vec![100u64, 1, 90, 1];
        let p = rebalance_placement(&hits, 2).unwrap();
        assert_eq!(p.iter().filter(|&&r| r == 0).count(), 2);
        assert_eq!(p.iter().filter(|&&r| r == 1).count(), 2);
        assert_ne!(p[0], p[2], "hot experts 0 and 2 should split: {p:?}");
    }

    #[test]
    fn rebalance_rejects_uneven_world() {
        assert!(rebalance_placement(&[1, 1, 1], 2).is_err());
    }

    #[test]
    fn replicas_add_hot_copies() {
        let hits = vec![100u64, 1, 90, 1];
        let map = rebalance_with_replicas(&hits, 2, 6).unwrap();
        assert_eq!(map.slots_per_rank(), 3);
        assert_eq!(map.local_slots[0].len(), 3);
        assert_eq!(map.local_slots[1].len(), 3);
        assert!(
            map.owners[0].len() > 1 || map.owners[2].len() > 1,
            "expected a hot replica: {:?}",
            map.owners
        );
        for (e, o) in map.owners.iter().enumerate() {
            assert!(!o.is_empty(), "expert {e} has no owner");
        }
    }

    #[test]
    fn pick_dispatch_uses_replicas() {
        let cfg = MoeEpConfig::new(77, 2, 0, 2, 4, 4).with_num_slots(4);
        let map = EpReplicaMap {
            owners: vec![vec![0, 1], vec![1]],
            local_slots: vec![vec![0], vec![0, 1]],
        };
        register_ep_replicas(77, map);
        assert_eq!(pick_dispatch_rank(&cfg, 0, 0), 0);
        assert_eq!(pick_dispatch_rank(&cfg, 0, 1), 1);
        assert_eq!(local_id_on_rank(&cfg, 0, 1), 0);
        assert_eq!(local_id_on_rank(&cfg, 1, 1), 1);
        unregister_ep_replicas(77);
    }

    #[test]
    fn registry_overrides_attrs() {
        let cfg = MoeEpConfig::new(55, 2, 0, 4, 8, 4).with_placement(vec![0, 0, 1, 1]);
        assert_eq!(resolve_placement(&cfg), vec![0, 0, 1, 1]);
        register_ep_placement(55, vec![0, 1, 0, 1]);
        assert_eq!(resolve_placement(&cfg), vec![0, 1, 0, 1]);
        unregister_ep_placement(55);
        assert_eq!(resolve_placement(&cfg), vec![0, 0, 1, 1]);
    }

    #[test]
    fn shard_weights_follows_placement() {
        let e = 2usize;
        let k = 2usize;
        let n = 2usize;
        let full = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let placement = vec![1u32, 0];
        let r0 = shard_expert_weights(&full, e, k, n, &placement, 0);
        let r1 = shard_expert_weights(&full, e, k, n, &placement, 1);
        assert_eq!(r0, vec![2.0, 2.0, 2.0, 2.0]);
        assert_eq!(r1, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn shard_slots_duplicates_replica_weights() {
        let full = vec![1.0, 1.0, 2.0, 2.0];
        let slots = [0u32, 0, 1];
        let packed = shard_expert_weights_slots(&full, 2, 1, 2, &slots);
        assert_eq!(packed, vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn migrate_swaps_shards_across_ranks() {
        // Start: rank0 has expert0, rank1 has expert1.
        // After: swapped placement.
        let world = 2u32;
        let k = 2usize;
        let n = 2usize;
        let stride = k * n;
        let listeners: Vec<TcpListener> = (0..world)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let addrs = addrs.clone();
                thread::spawn(move || {
                    let rank = rank as u32;
                    let t =
                        NetTransport::from_listener(rank, world, listener, addrs, 1 << 20).unwrap();
                    let g = ProcessGroup::new(Arc::new(t));
                    let old_placement = vec![0u32, 1];
                    let new_placement = vec![1u32, 0];
                    let old_w = if rank == 0 {
                        vec![1.0, 1.0, 1.0, 1.0] // expert 0
                    } else {
                        vec![2.0, 2.0, 2.0, 2.0] // expert 1
                    };
                    let new_w =
                        migrate_to_placement(&g, &old_placement, &old_w, &new_placement, k, n)
                            .unwrap();
                    assert_eq!(new_w.len(), stride);
                    if rank == 0 {
                        assert_eq!(new_w, vec![2.0, 2.0, 2.0, 2.0]);
                    } else {
                        assert_eq!(new_w, vec![1.0, 1.0, 1.0, 1.0]);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn migrate_replicas_distributes_hot_slab() {
        let world = 2u32;
        let k = 1usize;
        let n = 2usize;
        let listeners: Vec<TcpListener> = (0..world)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        // old: e0@0, e1@1  →  new: e0 on both ranks, e1 on rank1 only (3 slots?
        // Use map: local_slots [[0], [0,1]] — rank0 needs e0 (already has), rank1 needs e0+e1
        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let addrs = addrs.clone();
                thread::spawn(move || {
                    let rank = rank as u32;
                    let t =
                        NetTransport::from_listener(rank, world, listener, addrs, 1 << 20).unwrap();
                    let g = ProcessGroup::new(Arc::new(t));
                    let old = EpReplicaMap {
                        owners: vec![vec![0], vec![1]],
                        local_slots: vec![vec![0], vec![1]],
                    };
                    let new = EpReplicaMap {
                        owners: vec![vec![0, 1], vec![1]],
                        local_slots: vec![vec![0], vec![0, 1]],
                    };
                    let old_w = if rank == 0 {
                        vec![3.0, 3.0]
                    } else {
                        vec![4.0, 4.0]
                    };
                    let got = migrate_to_replica_map(&g, &old, &old_w, &new, k, n).unwrap();
                    if rank == 0 {
                        assert_eq!(got, vec![3.0, 3.0]);
                    } else {
                        assert_eq!(got, vec![3.0, 3.0, 4.0, 4.0]);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn all_reduce_hits_sums_across_ranks() {
        let world = 2u32;
        let listeners: Vec<TcpListener> = (0..world)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let addrs = addrs.clone();
                thread::spawn(move || {
                    let rank = rank as u32;
                    let t =
                        NetTransport::from_listener(rank, world, listener, addrs, 1 << 20).unwrap();
                    let g = ProcessGroup::new(Arc::new(t));
                    let local = if rank == 0 {
                        vec![3u64, 0, 1, 0]
                    } else {
                        vec![1u64, 5, 0, 2]
                    };
                    all_reduce_hits(&g, &local).unwrap()
                })
            })
            .collect();
        for h in handles {
            let got = h.join().unwrap();
            assert_eq!(got, vec![4, 5, 1, 2]);
        }
    }
}
