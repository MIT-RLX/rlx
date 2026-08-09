// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! MoE expert residency pool (TIDE-style predictive offload).
//!
//! Mirrors the policy in [ims-kdks/TIDE](https://github.com/ims-kdks/TIDE)
//! `LLaDA2MoeSparseMoeBlock`: rank experts by token hits, refresh placement
//! every τ steps, paired promote/demote to limit PCIe churn.
//!
//! Router logits and expert indices are unchanged — placement only.

use std::collections::{HashMap, HashSet};

/// When to re-run hit counting and expert placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertRefreshPolicy {
    /// Refresh on every forward (τ = 1; Mixtral-Offload-style).
    EveryForward,
    /// Autoregressive decode: refresh every N generated tokens / steps.
    EveryDecodeSteps(usize),
    /// Diffusion block decode: refresh every N denoise steps within a block
    /// (`jump_steps` in the TIDE reference repo).
    EveryDenoiseSteps(usize),
}

impl Default for ExpertRefreshPolicy {
    fn default() -> Self {
        Self::EveryDenoiseSteps(1)
    }
}

/// Per-forward hint from the runner (maps to TIDE `refresh_experts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoEExecMode {
    /// Reuse current GPU/CPU placement (`moe_infer`).
    Reuse,
    /// Recompute placement from this step's routing (`moe_infer_with_expert_refresh`).
    Refresh,
}

/// Anti-thrash policy for slot swaps (mirrors the llama.cpp hot-expert cache
/// hysteresis gate + dwell). Both knobs default to *off* so existing callers keep
/// the plain top-S paired-swap behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HysteresisConfig {
    /// A challenger expert must exceed an incumbent's hit count by this *fraction*
    /// before it may evict it (e.g. `0.25` = "must be ≥25% hotter"). `0.0` disables
    /// the margin so any strictly-hotter expert wins.
    pub margin: f32,
    /// An incumbent may not be evicted until it has been resident for at least this
    /// many refreshes (dwell / minimum-hold). `0` disables the dwell gate.
    pub min_dwell: u64,
}

impl Default for HysteresisConfig {
    fn default() -> Self {
        Self {
            margin: 0.0,
            min_dwell: 0,
        }
    }
}

impl HysteresisConfig {
    /// Whether either gate is active (otherwise placement uses the plain paired swap).
    pub fn is_active(&self) -> bool {
        self.margin > 0.0 || self.min_dwell > 0
    }
}

/// Configuration for [`ExpertPool`].
#[derive(Debug, Clone)]
pub struct ExpertPoolConfig {
    pub num_experts: usize,
    /// Max experts resident on the accelerator per MoE layer.
    pub gpu_budget: usize,
    pub refresh: ExpertRefreshPolicy,
    /// Slot-swap anti-thrash gates (default: disabled).
    pub hysteresis: HysteresisConfig,
}

impl ExpertPoolConfig {
    pub fn new(num_experts: usize, gpu_budget: usize, refresh: ExpertRefreshPolicy) -> Self {
        Self {
            num_experts,
            gpu_budget: gpu_budget.min(num_experts),
            refresh,
            hysteresis: HysteresisConfig::default(),
        }
    }

    /// Attach a hysteresis / dwell policy (builder).
    pub fn with_hysteresis(mut self, hysteresis: HysteresisConfig) -> Self {
        self.hysteresis = hysteresis;
        self
    }

    /// All experts pinned on device (offload disabled).
    pub fn all_resident(num_experts: usize) -> Self {
        Self::new(num_experts, num_experts, ExpertRefreshPolicy::EveryForward)
    }
}

/// Result of one placement refresh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertRefreshResult {
    pub target_gpu: Vec<usize>,
    pub promotions: usize,
    pub demotions: usize,
}

/// Cumulative counters (TIDE `offload_stats`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpertPoolStats {
    pub refreshes: u64,
    pub promotions: u64,
    pub demotions: u64,
}

/// Tracks which logical experts are GPU-resident and applies TIDE placement updates.
#[derive(Debug, Clone)]
pub struct ExpertPool {
    num_experts: usize,
    gpu_budget: usize,
    refresh: ExpertRefreshPolicy,
    resident: HashSet<usize>,
    /// Steps since last refresh (decode / denoise counter).
    steps_since_refresh: usize,
    stats: ExpertPoolStats,
    /// Slot-swap anti-thrash gates.
    hysteresis: HysteresisConfig,
    /// Monotonic refresh clock (never reset — dwell is measured against it, unlike
    /// [`ExpertPoolStats::refreshes`] which [`reset_step_stats`] zeroes).
    refresh_clock: u64,
    /// Refresh clock at which each currently-resident expert last became resident.
    resident_since: HashMap<usize, u64>,
}

impl ExpertPool {
    pub fn new(config: ExpertPoolConfig) -> Self {
        let gpu_budget = config.gpu_budget.min(config.num_experts);
        let mut resident = HashSet::new();
        let mut resident_since = HashMap::new();
        for e in 0..gpu_budget {
            resident.insert(e);
            resident_since.insert(e, 0);
        }
        Self {
            num_experts: config.num_experts,
            gpu_budget,
            refresh: config.refresh,
            resident,
            steps_since_refresh: 0,
            stats: ExpertPoolStats::default(),
            hysteresis: config.hysteresis,
            refresh_clock: 0,
            resident_since,
        }
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn gpu_budget(&self) -> usize {
        self.gpu_budget
    }

    pub fn refresh_policy(&self) -> ExpertRefreshPolicy {
        self.refresh
    }

    pub fn stats(&self) -> &ExpertPoolStats {
        &self.stats
    }

    /// TIDE `LLaDA2MoeSparseMoeBlock.reset_stats()` — clear per-step counters before next forward.
    pub fn reset_step_stats(&mut self) {
        self.stats = ExpertPoolStats::default();
    }

    pub fn resident_gpu_experts(&self) -> impl Iterator<Item = usize> + '_ {
        self.resident.iter().copied()
    }

    /// Bitmask for [`crate::CompiledGraph::set_moe_resident_experts`].
    pub fn resident_mask(&self) -> Vec<bool> {
        (0..self.num_experts)
            .map(|e| self.resident.contains(&e))
            .collect()
    }

    pub fn is_gpu_resident(&self, expert: usize) -> bool {
        self.resident.contains(&expert)
    }

    /// Whether offload is active (budget < total experts).
    pub fn offload_enabled(&self) -> bool {
        self.gpu_budget < self.num_experts
    }

    /// TIDE `generate`: `refresh_experts = prefill_block || (offload && step % τ == 0)`.
    pub fn should_refresh(
        &self,
        mode: MoEExecMode,
        denoise_step: usize,
        is_prefill_block: bool,
    ) -> bool {
        if !self.offload_enabled() {
            return false;
        }
        match mode {
            MoEExecMode::Refresh => true,
            MoEExecMode::Reuse => {
                if is_prefill_block {
                    return true;
                }
                match self.refresh {
                    ExpertRefreshPolicy::EveryForward => true,
                    ExpertRefreshPolicy::EveryDecodeSteps(n)
                    | ExpertRefreshPolicy::EveryDenoiseSteps(n) => {
                        let interval = n.max(1);
                        denoise_step.is_multiple_of(interval)
                    }
                }
            }
        }
    }

    /// Advance the step counter; returns whether this forward should refresh.
    pub fn on_forward_step(
        &mut self,
        mode: MoEExecMode,
        denoise_step: usize,
        is_prefill_block: bool,
    ) -> bool {
        let refresh = self.should_refresh(mode, denoise_step, is_prefill_block);
        if refresh {
            self.steps_since_refresh = 0;
        } else {
            self.steps_since_refresh = self.steps_since_refresh.saturating_add(1);
        }
        refresh
    }

    /// Count token hits per expert from flat or per-token indices (TIDE `bincount`).
    pub fn count_hits(expert_idx: &[u32], num_experts: usize) -> Vec<u64> {
        let mut counts = vec![0u64; num_experts];
        for &e in expert_idx {
            let e = e as usize;
            if e < num_experts {
                counts[e] += 1;
            }
        }
        counts
    }

    /// Top-`gpu_budget` experts by hit count (TIDE `torch.topk` on bincount).
    pub fn target_gpu_from_counts(counts: &[u64], gpu_budget: usize) -> Vec<usize> {
        let mut ranked: Vec<(u64, usize)> = counts
            .iter()
            .enumerate()
            .filter(|&(_, c)| *c > 0)
            .map(|(e, &c)| (c, e))
            .collect();
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        ranked
            .into_iter()
            .take(gpu_budget)
            .map(|(_, e)| e)
            .collect()
    }

    /// TIDE `update_expert_placement` + hit-based target selection.
    ///
    /// When a [`HysteresisConfig`] gate is active this routes through
    /// [`Self::refresh_from_counts`] (margin + dwell); otherwise it is the plain
    /// top-S paired swap.
    pub fn refresh_from_indices(&mut self, expert_idx: &[u32]) -> ExpertRefreshResult {
        let counts = Self::count_hits(expert_idx, self.num_experts);
        self.refresh_from_counts(&counts)
    }

    /// Refresh placement from per-expert hit counts, honoring the hysteresis / dwell
    /// gates when active.
    pub fn refresh_from_counts(&mut self, counts: &[u64]) -> ExpertRefreshResult {
        if self.hysteresis.is_active() {
            self.apply_hysteresis_placement(counts)
        } else {
            let target_order = Self::target_gpu_from_counts(counts, self.gpu_budget);
            self.apply_target_placement(&target_order)
        }
    }

    /// Refresh clock at which `expert` last became resident (0 if not resident).
    pub fn resident_since(&self, expert: usize) -> u64 {
        self.resident_since.get(&expert).copied().unwrap_or(0)
    }

    pub fn hysteresis(&self) -> HysteresisConfig {
        self.hysteresis
    }

    /// Mark `experts` as freshly (re)loaded at the current clock — record-keeping only
    /// (does not change residency); used to reset dwell after a manual slot fill.
    fn touch_resident_since(&mut self, e: usize) {
        self.resident_since.insert(e, self.refresh_clock);
    }

    /// Apply a precomputed target GPU set (paired promote/demote, no hysteresis).
    pub fn apply_target_placement(&mut self, target_order: &[usize]) -> ExpertRefreshResult {
        self.refresh_clock += 1;
        let clock = self.refresh_clock;
        let target_set: HashSet<usize> = target_order.iter().copied().collect();

        let to_promote: Vec<usize> = target_order
            .iter()
            .copied()
            .filter(|e| !self.resident.contains(e))
            .collect();
        let can_demote: Vec<usize> = self
            .resident
            .iter()
            .copied()
            .filter(|e| !target_set.contains(e))
            .collect();
        let to_demote: Vec<usize> = can_demote.iter().copied().take(to_promote.len()).collect();

        let mut new_resident = target_set;
        for e in can_demote.iter().skip(to_promote.len()) {
            new_resident.insert(*e);
        }

        // Maintain dwell bookkeeping: newly-resident experts start their dwell now;
        // drop timestamps for anything evicted.
        for &e in &to_promote {
            self.resident_since.insert(e, clock);
        }
        self.resident_since.retain(|e, _| new_resident.contains(e));

        let promotions = to_promote.len();
        let demotions = to_demote.len();
        self.resident = new_resident;
        self.stats.refreshes += 1;
        self.stats.promotions += promotions as u64;
        self.stats.demotions += demotions as u64;

        ExpertRefreshResult {
            target_gpu: target_order.to_vec(),
            promotions,
            demotions,
        }
    }

    /// Hysteresis-gated placement: promote the hottest non-resident experts, but only
    /// evict an incumbent when the challenger is `margin`-hotter AND the incumbent has
    /// been resident at least `min_dwell` refreshes. Challengers are considered
    /// hottest-first and eviction candidates weakest-first, so once the hottest
    /// challenger fails to beat the weakest evictable incumbent we stop — bounding
    /// churn without a hard paired-swap cap.
    fn apply_hysteresis_placement(&mut self, counts: &[u64]) -> ExpertRefreshResult {
        self.refresh_clock += 1;
        let clock = self.refresh_clock;
        let budget = self.gpu_budget;
        let margin = self.hysteresis.margin.max(0.0) as f64;
        let dwell = self.hysteresis.min_dwell;
        let hits = |e: usize| counts.get(e).copied().unwrap_or(0);

        let desired = Self::target_gpu_from_counts(counts, budget);
        let desired_set: HashSet<usize> = desired.iter().copied().collect();

        // Challengers: desired experts not yet resident (already hottest-first).
        let to_add: Vec<usize> = desired
            .iter()
            .copied()
            .filter(|e| !self.resident.contains(e))
            .collect();

        // Eviction candidates: resident experts not in the desired set, weakest-first
        // (fewest hits, then longest-resident, then lowest id for determinism).
        let mut evictable: Vec<usize> = self
            .resident
            .iter()
            .copied()
            .filter(|e| !desired_set.contains(e))
            .collect();
        evictable.sort_by(|&a, &b| {
            hits(a)
                .cmp(&hits(b))
                .then_with(|| self.resident_since(a).cmp(&self.resident_since(b)))
                .then_with(|| a.cmp(&b))
        });

        let mut promotions = 0usize;
        let mut demotions = 0usize;
        let mut ev_idx = 0usize;

        for c in to_add {
            let c_hits = hits(c) as f64;
            if self.resident.len() < budget {
                // Free slot — promote with no eviction.
                self.resident.insert(c);
                self.touch_resident_since(c);
                promotions += 1;
                continue;
            }
            // Locate the weakest unprotected incumbent this challenger can displace.
            let mut victim: Option<usize> = None;
            while ev_idx < evictable.len() {
                let i = evictable[ev_idx];
                let held = clock.saturating_sub(self.resident_since(i));
                if held < dwell {
                    // Dwell-protected: keeps its slot, skip permanently this round.
                    ev_idx += 1;
                    continue;
                }
                if c_hits >= hits(i) as f64 * (1.0 + margin) {
                    victim = Some(i);
                    ev_idx += 1;
                }
                // Weakest unprotected incumbent decided this challenger's fate.
                break;
            }
            match victim {
                Some(i) => {
                    self.resident.remove(&i);
                    self.resident_since.remove(&i);
                    self.resident.insert(c);
                    self.touch_resident_since(c);
                    promotions += 1;
                    demotions += 1;
                }
                None => break, // colder challengers can't do better — stop.
            }
        }

        self.stats.refreshes += 1;
        self.stats.promotions += promotions as u64;
        self.stats.demotions += demotions as u64;

        ExpertRefreshResult {
            target_gpu: desired,
            promotions,
            demotions,
        }
    }
}

/// Per-layer resident bitmasks (TIDE placement; one row per MoE FFN in forward order).
pub fn per_layer_resident_masks(pools: &[ExpertPool]) -> Vec<Vec<bool>> {
    pools.iter().map(|p| p.resident_mask()).collect()
}

/// Union of GPU-resident experts across per-layer pools (legacy single graph mask).
pub fn merged_resident_mask(pools: &[ExpertPool]) -> Vec<bool> {
    let Some(first) = pools.first() else {
        return Vec::new();
    };
    let n = first.num_experts();
    (0..n)
        .map(|e| pools.iter().any(|p| p.is_gpu_resident(e)))
        .collect()
}

pub fn gpu_expert_budget_from_vram(
    free_bytes: usize,
    reserve_bytes: usize,
    expert_param_bytes: usize,
    num_moe_layers: usize,
    max_gpu_experts_per_layer: usize,
    num_experts: usize,
) -> usize {
    if expert_param_bytes == 0 || num_moe_layers == 0 {
        return max_gpu_experts_per_layer.min(num_experts);
    }
    let usable = free_bytes.saturating_sub(reserve_bytes);
    let per_layer = usable / (expert_param_bytes.saturating_mul(num_moe_layers));
    per_layer.min(max_gpu_experts_per_layer).min(num_experts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_layer_masks_differ_from_merged_union() {
        let mut p0 = ExpertPool::new(ExpertPoolConfig::new(
            4,
            2,
            ExpertRefreshPolicy::EveryForward,
        ));
        let mut p1 = ExpertPool::new(ExpertPoolConfig::new(
            4,
            2,
            ExpertRefreshPolicy::EveryForward,
        ));
        p0.refresh_from_indices(&[0, 1]);
        p1.refresh_from_indices(&[2, 3]);
        let pools = [p0, p1];
        let merged = merged_resident_mask(&pools);
        assert_eq!(merged, vec![true, true, true, true]);
        let per = per_layer_resident_masks(&pools);
        assert_eq!(per[0], vec![true, true, false, false]);
        assert_eq!(per[1], vec![false, false, true, true]);
    }

    #[test]
    fn count_hits_matches_bincount() {
        let idx = [1u32, 0, 1, 2, 1];
        let c = ExpertPool::count_hits(&idx, 4);
        assert_eq!(c, [1, 3, 1, 0]);
    }

    #[test]
    fn target_gpu_picks_top_by_count() {
        let counts = [10, 50, 30, 0, 50];
        let t = ExpertPool::target_gpu_from_counts(&counts, 3);
        assert_eq!(t, vec![1, 4, 2]); // tie-break: lower expert id first
    }

    #[test]
    fn paired_swap_limits_demotions() {
        let mut pool = ExpertPool::new(ExpertPoolConfig::new(
            8,
            2,
            ExpertRefreshPolicy::EveryForward,
        ));
        pool.resident = [0, 1].into_iter().collect();
        let r = pool.apply_target_placement(&[6, 7]);
        assert_eq!(r.promotions, 2);
        assert_eq!(r.demotions, 2);
        assert_eq!(pool.resident, [6, 7].into_iter().collect::<HashSet<_>>());
    }

    #[test]
    fn paired_swap_keeps_extra_residents() {
        let mut pool = ExpertPool::new(ExpertPoolConfig::new(
            8,
            4,
            ExpertRefreshPolicy::EveryForward,
        ));
        pool.resident = [0, 1, 2, 3].into_iter().collect();
        // Target overlaps heavily — paired demotion leaves one former GPU expert
        // on device (matches TIDE `can_demote[len(to_promote):]`).
        let r = pool.apply_target_placement(&[2, 3, 4, 5]);
        assert_eq!(r.promotions, 2);
        assert_eq!(r.demotions, 2);
        assert_eq!(pool.resident.len(), 4);
        for e in [2, 3, 4, 5] {
            assert!(pool.is_gpu_resident(e));
        }
        assert!(!pool.is_gpu_resident(0));
    }

    #[test]
    fn jump_steps_refresh_schedule() {
        let pool = ExpertPool::new(ExpertPoolConfig::new(
            256,
            64,
            ExpertRefreshPolicy::EveryDenoiseSteps(3),
        ));
        assert!(pool.should_refresh(MoEExecMode::Reuse, 0, false));
        assert!(!pool.should_refresh(MoEExecMode::Reuse, 1, false));
        assert!(!pool.should_refresh(MoEExecMode::Reuse, 2, false));
        assert!(pool.should_refresh(MoEExecMode::Reuse, 3, false));
        assert!(pool.should_refresh(MoEExecMode::Reuse, 0, true)); // prefill block
    }

    #[test]
    fn hysteresis_margin_protects_incumbent() {
        let cfg = ExpertPoolConfig::new(4, 2, ExpertRefreshPolicy::EveryForward).with_hysteresis(
            HysteresisConfig {
                margin: 0.5,
                min_dwell: 0,
            },
        );
        let mut pool = ExpertPool::new(cfg); // resident {0,1}

        // Expert 2 is hotter than incumbent 1 but not by the 50% margin → no swap.
        let r = pool.refresh_from_counts(&[10, 10, 12, 0]);
        assert_eq!(r.promotions, 0);
        assert_eq!(r.demotions, 0);
        assert!(!pool.is_gpu_resident(2));
        assert!(pool.is_gpu_resident(1));

        // Now expert 2 clears the margin (20 ≥ 10 * 1.5) → it displaces expert 1.
        let r = pool.refresh_from_counts(&[10, 10, 20, 0]);
        assert_eq!(r.promotions, 1);
        assert_eq!(r.demotions, 1);
        assert!(pool.is_gpu_resident(2));
        assert!(!pool.is_gpu_resident(1));
    }

    #[test]
    fn dwell_delays_eviction() {
        let cfg = ExpertPoolConfig::new(4, 2, ExpertRefreshPolicy::EveryForward).with_hysteresis(
            HysteresisConfig {
                margin: 0.0,
                min_dwell: 3,
            },
        );
        let mut pool = ExpertPool::new(cfg); // resident {0,1}, resident_since = clock 0

        // Expert 2 is far hotter, but the incumbents haven't served their dwell yet.
        for _ in 0..2 {
            let r = pool.refresh_from_counts(&[0, 0, 100, 0]);
            assert_eq!(r.promotions, 0, "dwell must protect fresh incumbents");
            assert!(!pool.is_gpu_resident(2));
        }
        // Third refresh: clock reaches 3, dwell satisfied → expert 2 gets a slot.
        let r = pool.refresh_from_counts(&[0, 0, 100, 0]);
        assert_eq!(r.promotions, 1);
        assert!(pool.is_gpu_resident(2));
    }

    #[test]
    fn hysteresis_disabled_matches_plain_topk() {
        let counts = [3u64, 9, 1, 7, 0, 5];
        // Plain path.
        let mut plain = ExpertPool::new(ExpertPoolConfig::new(
            6,
            3,
            ExpertRefreshPolicy::EveryForward,
        ));
        plain.refresh_from_counts(&counts);
        // Default (disabled) hysteresis path.
        let mut hyst = ExpertPool::new(
            ExpertPoolConfig::new(6, 3, ExpertRefreshPolicy::EveryForward)
                .with_hysteresis(HysteresisConfig::default()),
        );
        hyst.refresh_from_counts(&counts);
        assert_eq!(plain.resident_mask(), hyst.resident_mask());
        // Top-3 by count = experts 1(9), 3(7), 5(5).
        assert_eq!(
            plain.resident_mask(),
            vec![false, true, false, true, false, true]
        );
    }

    #[test]
    fn vram_budget_formula() {
        let b = gpu_expert_budget_from_vram(
            40 * 1024 * 1024 * 1024,
            2 * 1024 * 1024 * 1024,
            50 * 1024 * 1024,
            20,
            128,
            256,
        );
        assert!(b > 0 && b <= 128);
    }
}
