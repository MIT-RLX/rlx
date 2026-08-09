// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end proof of the hot-on-GPU / cold-on-CPU MoE split on the CPU, using the
//! exact runtime API sequence a GPU backend will drive each decode step:
//!
//! 1. `ExpertPool::refresh_from_indices` — usage-heatmap placement (+ hysteresis).
//! 2. `HotExpertCache::reconcile` — minimal host→device load plan for the new hot set.
//! 3. apply ONLY the emitted `SlotLoad`s to the device slot buffer (incremental H2D).
//! 4. `HotExpertCache::route` — split tokens into hot(slot)/cold.
//! 5. hot tokens over the slot buffer + `cold_grouped_matmul` over the host store.
//!
//! The invariant under test: across many refreshes, applying only the incremental
//! loads keeps the slot buffer correct, so the split stays **byte-identical** to a
//! full grouped matmul over all experts. If the GPU port matches this reference, it
//! is correct.

use rlx_cpu::moe_split::{cold_grouped_matmul, grouped_matmul_reference};
use rlx_runtime::{
    ExpertPool, ExpertPoolConfig, ExpertRefreshPolicy, HotExpertCache, HysteresisConfig,
};

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn next_expert(&mut self, num_experts: usize) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize % num_experts) as u32
    }
}

const SENTINEL: u32 = u32::MAX;

/// Run one decode step of the split and assert byte-parity with the full path.
fn step(
    pool: &mut ExpertPool,
    cache: &mut HotExpertCache,
    slot_weights: &mut [f32],
    weights_all: &[f32],
    input: &[f32],
    expert_idx: &[u32],
    m: usize,
    k: usize,
    n: usize,
) {
    // 1. Heatmap placement.
    pool.refresh_from_indices(expert_idx);
    // 2. Reconcile device slots to the new hot set.
    let loads = cache.reconcile(&pool.resident_mask());
    // 3. Apply ONLY the incremental loads (simulated H2D copy expert→slot).
    let stride = k * n;
    for load in &loads {
        slot_weights[load.slot * stride..(load.slot + 1) * stride]
            .copy_from_slice(&weights_all[load.expert * stride..(load.expert + 1) * stride]);
    }
    // Cache residency must equal pool residency at all times.
    for e in 0..pool.num_experts() {
        assert_eq!(
            cache.is_resident(e),
            pool.is_gpu_resident(e),
            "residency mismatch e={e}"
        );
    }

    // 4. Route.
    let route = cache.route(expert_idx, SENTINEL);

    // 5. Hot over slots + cold over host store.
    let mut split = vec![0f32; m * n];
    // Hot tokens grouped by slot.
    use std::collections::BTreeMap;
    let mut by_slot: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (t, &s) in route.slot_idx.iter().enumerate() {
        if s != SENTINEL {
            by_slot.entry(s).or_default().push(t);
        }
    }
    for (slot, tokens) in by_slot {
        let w = &slot_weights[slot as usize * stride..(slot as usize + 1) * stride];
        let cnt = tokens.len();
        let mut pin = vec![0f32; cnt * k];
        for (r, &t) in tokens.iter().enumerate() {
            pin[r * k..(r + 1) * k].copy_from_slice(&input[t * k..(t + 1) * k]);
        }
        let mut pout = vec![0f32; cnt * n];
        // Route through the same public sgemm the cold helper uses via a 1-expert store.
        rlx_cpu::moe_split::cold_grouped_matmul(
            &pin,
            k,
            n,
            &(0..cnt).map(|r| (r, 0usize)).collect::<Vec<_>>(),
            w,
            &mut pout,
        );
        for (r, &t) in tokens.iter().enumerate() {
            split[t * n..(t + 1) * n].copy_from_slice(&pout[r * n..(r + 1) * n]);
        }
    }
    // Cold tokens over the full host store.
    cold_grouped_matmul(input, k, n, &route.cold, weights_all, &mut split);

    // Oracle.
    let full = grouped_matmul_reference(input, weights_all, expert_idx, m, k, n);
    assert_eq!(
        split, full,
        "hot/cold split diverged from full grouped matmul"
    );
}

#[test]
fn dynamic_refresh_stays_byte_exact() {
    let (num_experts, budget, m, k, n) = (16usize, 4usize, 48usize, 6usize, 5usize);
    let mut rng = Lcg(0xDEAD_BEEF);
    let input: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
    let weights_all: Vec<f32> = (0..num_experts * k * n).map(|_| rng.next_f32()).collect();

    // Hysteresis ON — exercises the anti-thrash placement path end-to-end.
    let cfg = ExpertPoolConfig::new(num_experts, budget, ExpertRefreshPolicy::EveryForward)
        .with_hysteresis(HysteresisConfig {
            margin: 0.25,
            min_dwell: 2,
        });
    let mut pool = ExpertPool::new(cfg);
    let mut cache = HotExpertCache::new(num_experts, budget);
    // Slot buffer starts holding experts 0..budget (matches ExpertPool/HotExpertCache init).
    let stride = k * n;
    let mut slot_weights = vec![0f32; budget * stride];
    for s in 0..budget {
        slot_weights[s * stride..(s + 1) * stride]
            .copy_from_slice(&weights_all[s * stride..(s + 1) * stride]);
    }

    // Many steps with shifting routing "hot spots" so the resident set churns.
    for s in 0..30 {
        // Bias routing toward a moving window of experts to make some experts hot.
        let hot_base = (s * 3) % num_experts;
        let expert_idx: Vec<u32> = (0..m)
            .map(|t| {
                if t % 2 == 0 {
                    ((hot_base + (t % 3)) % num_experts) as u32
                } else {
                    rng.next_expert(num_experts)
                }
            })
            .collect();
        step(
            &mut pool,
            &mut cache,
            &mut slot_weights,
            &weights_all,
            &input,
            &expert_idx,
            m,
            k,
            n,
        );
    }
}

#[test]
fn all_cold_when_budget_zero_equivalent() {
    // Budget 1 but route only to experts that are never hot enough to stay — still exact.
    let (num_experts, budget, m, k, n) = (8usize, 1usize, 20usize, 4usize, 4usize);
    let mut rng = Lcg(7);
    let input: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
    let weights_all: Vec<f32> = (0..num_experts * k * n).map(|_| rng.next_f32()).collect();
    let mut pool = ExpertPool::new(ExpertPoolConfig::new(
        num_experts,
        budget,
        ExpertRefreshPolicy::EveryForward,
    ));
    let mut cache = HotExpertCache::new(num_experts, budget);
    let stride = k * n;
    let mut slot_weights = vec![0f32; budget * stride];
    slot_weights.copy_from_slice(&weights_all[0..budget * stride]);

    for _ in 0..10 {
        let expert_idx: Vec<u32> = (0..m).map(|_| rng.next_expert(num_experts)).collect();
        step(
            &mut pool,
            &mut cache,
            &mut slot_weights,
            &weights_all,
            &input,
            &expert_idx,
            m,
            k,
            n,
        );
    }
}
