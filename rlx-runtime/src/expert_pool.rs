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

//! MoE expert residency pool (TIDE-style predictive offload).
//!
//! Mirrors the policy in [ims-kdks/TIDE](https://github.com/ims-kdks/TIDE)
//! `LLaDA2MoeSparseMoeBlock`: rank experts by token hits, refresh placement
//! every τ steps, paired promote/demote to limit PCIe churn.
//!
//! Router logits and expert indices are unchanged — placement only.

use std::collections::HashSet;

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

/// Configuration for [`ExpertPool`].
#[derive(Debug, Clone)]
pub struct ExpertPoolConfig {
    pub num_experts: usize,
    /// Max experts resident on the accelerator per MoE layer.
    pub gpu_budget: usize,
    pub refresh: ExpertRefreshPolicy,
}

impl ExpertPoolConfig {
    pub fn new(num_experts: usize, gpu_budget: usize, refresh: ExpertRefreshPolicy) -> Self {
        Self {
            num_experts,
            gpu_budget: gpu_budget.min(num_experts),
            refresh,
        }
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
}

impl ExpertPool {
    pub fn new(config: ExpertPoolConfig) -> Self {
        let gpu_budget = config.gpu_budget.min(config.num_experts);
        let mut resident = HashSet::new();
        for e in 0..gpu_budget {
            resident.insert(e);
        }
        Self {
            num_experts: config.num_experts,
            gpu_budget,
            refresh: config.refresh,
            resident,
            steps_since_refresh: 0,
            stats: ExpertPoolStats::default(),
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
    pub fn refresh_from_indices(&mut self, expert_idx: &[u32]) -> ExpertRefreshResult {
        let counts = Self::count_hits(expert_idx, self.num_experts);
        let target_order = Self::target_gpu_from_counts(&counts, self.gpu_budget);
        self.apply_target_placement(&target_order)
    }

    /// Apply a precomputed target GPU set (paired promote/demote).
    pub fn apply_target_placement(&mut self, target_order: &[usize]) -> ExpertRefreshResult {
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
