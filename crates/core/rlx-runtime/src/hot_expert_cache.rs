// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hot-expert device slot cache for MoE offload (the "hotstore" of the
//! hot-on-GPU / cold-on-CPU split).
//!
//! One [`HotExpertCache`] models a single MoE layer's expert projection stack:
//! `num_slots` device-resident slots drawn from `num_experts` total experts. An
//! [`crate::ExpertPool`] chooses *which* experts are hot (by usage heatmap +
//! hysteresis); this cache tracks *where* each hot expert physically lives on the
//! device (its slot) and emits the minimal set of host→device copies
//! ([`SlotLoad`]) needed to bring the slots in line with the pool's resident set.
//!
//! Backend-agnostic on purpose: the runtime owns the slot↔expert bookkeeping and
//! the routing split; a backend (CUDA / ROCm) owns the actual device buffers, the
//! H2D copies named by [`SlotLoad`], and the grouped-matmul kernel that runs over
//! the slot buffer. Cold experts (no slot) never occupy device memory — that is
//! where the VRAM saving comes from — and are computed on the host from the full
//! [`crate::MoeExpertStore`].

/// One host→device load: copy `expert`'s weights into device `slot`, overwriting
/// whatever `evicted` expert previously occupied it (`None` if the slot was empty).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotLoad {
    pub slot: usize,
    pub expert: usize,
    pub evicted: Option<usize>,
}

/// Per-token routing after the slot split: `slot_idx[t]` is the device slot for a
/// token whose expert is resident, or `cold_sentinel` when the token's expert is
/// cold (to be computed on the host). `cold` lists `(token_index, global_expert)`
/// for every cold token so the host path can gather/scatter them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotRoute {
    pub slot_idx: Vec<u32>,
    pub cold: Vec<(usize, usize)>,
}

/// Device slot cache for one MoE layer.
#[derive(Debug, Clone)]
pub struct HotExpertCache {
    num_experts: usize,
    num_slots: usize,
    /// slot -> expert currently loaded (`None` = empty slot).
    slot_to_expert: Vec<Option<usize>>,
    /// expert -> slot (`None` = cold / not device-resident).
    expert_to_slot: Vec<Option<usize>>,
}

impl HotExpertCache {
    /// New cache with the first `min(num_slots, num_experts)` experts pre-loaded,
    /// matching [`ExpertPool::new`](crate::ExpertPool)'s initial residency so the
    /// two structures start consistent (no spurious first-refresh reload).
    pub fn new(num_experts: usize, num_slots: usize) -> Self {
        let num_slots = num_slots.min(num_experts);
        let mut slot_to_expert = vec![None; num_slots];
        let mut expert_to_slot = vec![None; num_experts];
        for s in 0..num_slots {
            slot_to_expert[s] = Some(s);
            expert_to_slot[s] = Some(s);
        }
        Self {
            num_experts,
            num_slots,
            slot_to_expert,
            expert_to_slot,
        }
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Device slot holding `expert`, or `None` if the expert is cold.
    pub fn slot_of(&self, expert: usize) -> Option<usize> {
        self.expert_to_slot.get(expert).copied().flatten()
    }

    /// Whether `expert` currently occupies a device slot.
    pub fn is_resident(&self, expert: usize) -> bool {
        self.slot_of(expert).is_some()
    }

    /// slot -> expert table (`None` = empty slot).
    pub fn slot_to_expert(&self) -> &[Option<usize>] {
        &self.slot_to_expert
    }

    /// expert -> slot table (`None` = cold).
    pub fn expert_to_slot(&self) -> &[Option<usize>] {
        &self.expert_to_slot
    }

    /// Bring device slots in line with `resident_mask` (typically
    /// [`ExpertPool::resident_mask`](crate::ExpertPool::resident_mask)), keeping
    /// experts that stay resident in their existing slot (no reload). Returns the
    /// minimal set of host→device loads for newly-resident experts.
    ///
    /// Two clean passes so leftover slots (when fewer experts are hot than there
    /// are slots) are always left consistently empty rather than dangling on a
    /// now-cold expert.
    pub fn reconcile(&mut self, resident_mask: &[bool]) -> Vec<SlotLoad> {
        debug_assert_eq!(resident_mask.len(), self.num_experts);
        let resident = |e: usize| resident_mask.get(e).copied().unwrap_or(false);

        // Pass 1: free slots holding a now-cold expert (or already empty),
        // remembering the prior occupant for the `evicted` field.
        let mut free_slots: Vec<(usize, Option<usize>)> = Vec::new();
        for slot in 0..self.num_slots {
            match self.slot_to_expert[slot] {
                Some(e) if resident(e) => {} // stays put
                Some(e) => {
                    self.expert_to_slot[e] = None;
                    self.slot_to_expert[slot] = None;
                    free_slots.push((slot, Some(e)));
                }
                None => free_slots.push((slot, None)),
            }
        }

        // Newly-resident experts needing a slot (ascending id for determinism).
        let newly: Vec<usize> = (0..self.num_experts)
            .filter(|&e| resident(e) && self.expert_to_slot[e].is_none())
            .collect();
        debug_assert!(
            newly.len() <= free_slots.len(),
            "resident_mask has more hot experts ({}) than free slots ({})",
            newly.len(),
            free_slots.len()
        );

        // Pass 2: assign each newly-hot expert to a freed slot, emitting the load.
        let mut loads = Vec::with_capacity(newly.len());
        for (idx, &e) in newly.iter().enumerate() {
            let Some(&(slot, evicted)) = free_slots.get(idx) else {
                break;
            };
            self.slot_to_expert[slot] = Some(e);
            self.expert_to_slot[e] = Some(slot);
            loads.push(SlotLoad {
                slot,
                expert: e,
                evicted,
            });
        }
        loads
    }

    /// Split per-token routed expert indices into slot indices + a cold-token list.
    /// Resident tokens get their device slot; cold tokens get `cold_sentinel` in
    /// `slot_idx` and an entry in `cold`.
    pub fn route(&self, expert_idx: &[u32], cold_sentinel: u32) -> SlotRoute {
        let mut slot_idx = Vec::with_capacity(expert_idx.len());
        let mut cold = Vec::new();
        for (t, &e) in expert_idx.iter().enumerate() {
            let e = e as usize;
            match self.slot_of(e) {
                Some(slot) => slot_idx.push(slot as u32),
                None => {
                    slot_idx.push(cold_sentinel);
                    cold.push((t, e));
                }
            }
        }
        SlotRoute { slot_idx, cold }
    }
}

/// Reconcile a whole model's per-layer caches against per-layer resident masks
/// (e.g. `pools.iter().map(|p| p.resident_mask())`). Returns per-layer load lists.
pub fn reconcile_layers(caches: &mut [HotExpertCache], masks: &[Vec<bool>]) -> Vec<Vec<SlotLoad>> {
    caches
        .iter_mut()
        .zip(masks.iter())
        .map(|(cache, mask)| cache.reconcile(mask))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExpertPool, ExpertPoolConfig, ExpertRefreshPolicy};

    #[test]
    fn starts_consistent_with_expert_pool() {
        let cache = HotExpertCache::new(8, 3);
        // Experts 0,1,2 pre-loaded into slots 0,1,2 (mirrors ExpertPool::new).
        assert_eq!(cache.slot_of(0), Some(0));
        assert_eq!(cache.slot_of(2), Some(2));
        assert_eq!(cache.slot_of(3), None);
        assert!(cache.is_resident(1));
        assert!(!cache.is_resident(7));
    }

    #[test]
    fn reconcile_keeps_stable_experts_in_place() {
        let mut cache = HotExpertCache::new(8, 3); // slots: 0->0,1->1,2->2
        // New hot set {1,2,5}: expert 0 leaves, expert 5 arrives; 1,2 stay put.
        let mut mask = vec![false; 8];
        for e in [1, 2, 5] {
            mask[e] = true;
        }
        let loads = cache.reconcile(&mask);
        // Only one load: expert 5 into the slot freed by expert 0 (slot 0).
        assert_eq!(
            loads,
            vec![SlotLoad {
                slot: 0,
                expert: 5,
                evicted: Some(0),
            }]
        );
        assert_eq!(cache.slot_of(1), Some(1), "stable expert keeps its slot");
        assert_eq!(cache.slot_of(2), Some(2), "stable expert keeps its slot");
        assert_eq!(cache.slot_of(5), Some(0));
        assert_eq!(cache.slot_of(0), None);
    }

    #[test]
    fn reconcile_leaves_extra_slots_empty_and_consistent() {
        let mut cache = HotExpertCache::new(8, 4); // slots 0..4 -> experts 0..4
        // Only two experts are hot now.
        let mut mask = vec![false; 8];
        mask[6] = true;
        mask[1] = true; // 1 already resident (slot 1)
        let loads = cache.reconcile(&mask);
        // Expert 6 loads into the first freed slot (slot 0, evicting expert 0).
        assert_eq!(loads.len(), 1);
        assert_eq!(loads[0].expert, 6);
        assert!(cache.is_resident(1) && cache.is_resident(6));
        // Every cold expert must have no slot (no dangling expert_to_slot entries).
        for e in [0, 2, 3, 4, 5, 7] {
            assert!(
                !cache.is_resident(e),
                "cold expert {e} still marked resident"
            );
        }
        // Slot table has exactly two occupied slots.
        let occupied = cache
            .slot_to_expert()
            .iter()
            .filter(|s| s.is_some())
            .count();
        assert_eq!(occupied, 2);
    }

    #[test]
    fn route_splits_hot_and_cold_tokens() {
        let mut cache = HotExpertCache::new(8, 3);
        let mask = {
            let mut m = vec![false; 8];
            for e in [1, 2, 5] {
                m[e] = true;
            }
            m
        };
        cache.reconcile(&mask); // 1->1, 2->2, 5->0
        // Tokens routed to experts [5, 3, 1, 7, 2]; 3 and 7 are cold.
        let r = cache.route(&[5, 3, 1, 7, 2], u32::MAX);
        assert_eq!(r.slot_idx, vec![0, u32::MAX, 1, u32::MAX, 2]);
        assert_eq!(r.cold, vec![(1, 3), (3, 7)]);
    }

    #[test]
    fn tracks_expert_pool_placement_end_to_end() {
        // The cache should exactly mirror an ExpertPool's resident set after each
        // heatmap refresh, with bounded loads.
        let mut pool = ExpertPool::new(ExpertPoolConfig::new(
            8,
            3,
            ExpertRefreshPolicy::EveryForward,
        ));
        let mut cache = HotExpertCache::new(8, 3);

        // Route heavily to experts 4,5,6 → they should become resident.
        pool.refresh_from_indices(&[4, 4, 5, 5, 6, 6, 0]);
        let loads = reconcile_layers(&mut [cache.clone()], &[pool.resident_mask()]);
        cache.reconcile(&pool.resident_mask());
        // Cache residency must equal pool residency.
        for e in 0..8 {
            assert_eq!(
                cache.is_resident(e),
                pool.is_gpu_resident(e),
                "expert {e} residency mismatch"
            );
        }
        assert!(!loads[0].is_empty());
    }
}
