// RLX — versatile ML compiler + runtime.
//! **Centralized resource budget for large-model inference** — the single source
//! of truth for how much a model may hold resident, applied uniformly across ALL
//! large models (DeepSeek-V4, Kimi-K3, Llama4, GLM-MoE, …). It unifies the two
//! knobs that were previously scattered:
//!   * **RAM budget** — how much weight memory may be resident
//!     ([`crate::memory_estimate::soft_memory_budget_bytes`]).
//!   * **Experts-per-time** — how many MoE experts stay resident per layer (the
//!     offload/paging cache size; feeds [`crate::expert_pool::ExpertPoolConfig`]
//!     and `CompiledGraph::set_moe_resident_experts`).
//!
//! Model crates read one [`ResourceBudget`] (from env, or overridden by a model's
//! config) and hand it to the runtime — they never re-implement budgeting. Env:
//! `RLX_MAX_RAM_BYTES` (bytes; else `RLX_SOFT_MEMORY_BUDGET_BYTES` / physical RAM)
//! and `RLX_MAX_RESIDENT_EXPERTS` (per layer; else derived from the RAM budget).

use crate::expert_pool::{ExpertPoolConfig, ExpertRefreshPolicy};
use crate::memory_estimate::soft_memory_budget_bytes;

/// How much a model may hold resident. `None` fields mean "unbounded / derive".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceBudget {
    /// Max resident weight RAM (bytes). `None` → the physical-RAM-derived soft
    /// budget ([`soft_memory_budget_bytes`]).
    pub max_ram_bytes: Option<usize>,
    /// Max MoE experts resident PER LAYER. `None` → derive from `max_ram_bytes`
    /// (or all experts if RAM is unbounded).
    pub max_resident_experts: Option<usize>,
}

impl ResourceBudget {
    /// Unbounded: full RAM, all experts resident.
    pub const UNBOUNDED: Self = Self {
        max_ram_bytes: None,
        max_resident_experts: None,
    };

    /// Read from the environment. `RLX_MAX_RAM_BYTES` and `RLX_MAX_RESIDENT_EXPERTS`
    /// override; absent fields fall back to derivation at query time.
    pub fn from_env() -> Self {
        let max_ram_bytes = std::env::var("RLX_MAX_RAM_BYTES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok());
        let max_resident_experts = std::env::var("RLX_MAX_RESIDENT_EXPERTS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok());
        Self {
            max_ram_bytes,
            max_resident_experts,
        }
    }

    /// The effective RAM ceiling: explicit `max_ram_bytes`, else the soft
    /// physical-RAM budget (`None` only when RAM is unknown and unset).
    pub fn effective_ram_bytes(&self) -> Option<usize> {
        self.max_ram_bytes.or_else(soft_memory_budget_bytes)
    }

    /// How many experts to keep resident PER LAYER: honor an explicit
    /// `max_resident_experts`, else derive from the RAM ceiling by reserving
    /// `backbone_bytes` (attention/norms/embed) and dividing the remainder by
    /// `bytes_per_expert`. Always in `1..=num_experts`.
    pub fn resident_experts(
        &self,
        num_experts: usize,
        bytes_per_expert: usize,
        backbone_bytes: usize,
    ) -> usize {
        if let Some(e) = self.max_resident_experts {
            return e.clamp(1, num_experts.max(1));
        }
        match self.effective_ram_bytes() {
            Some(ram) => {
                let for_experts = ram.saturating_sub(backbone_bytes);
                (for_experts / bytes_per_expert.max(1)).clamp(1, num_experts.max(1))
            }
            None => num_experts,
        }
    }

    /// Build an [`ExpertPoolConfig`] honoring this budget (offload kicks in when the
    /// resident count is below `num_experts`).
    pub fn expert_pool_config(
        &self,
        num_experts: usize,
        bytes_per_expert: usize,
        backbone_bytes: usize,
        refresh: ExpertRefreshPolicy,
    ) -> ExpertPoolConfig {
        ExpertPoolConfig::new(
            num_experts,
            self.resident_experts(num_experts, bytes_per_expert, backbone_bytes),
            refresh,
        )
    }

    /// Whether weight offload/streaming should engage: the model doesn't fit the
    /// RAM ceiling, or an explicit expert cap is below the expert count.
    pub fn needs_offload(&self, model_bytes: usize, num_experts: usize) -> bool {
        self.max_resident_experts.is_some_and(|e| e < num_experts)
            || self
                .effective_ram_bytes()
                .is_some_and(|ram| model_bytes > ram)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_expert_cap_wins() {
        let b = ResourceBudget {
            max_ram_bytes: Some(1 << 40),
            max_resident_experts: Some(8),
        };
        assert_eq!(b.resident_experts(256, 1 << 20, 0), 8);
        assert!(b.needs_offload(1 << 30, 256));
    }

    #[test]
    fn derive_experts_from_ram() {
        // 20 GB RAM, 4 GB backbone, 1 GB/expert → 16 experts fit.
        let b = ResourceBudget {
            max_ram_bytes: Some(20 * (1 << 30)),
            max_resident_experts: None,
        };
        assert_eq!(b.resident_experts(256, 1 << 30, 4 * (1 << 30)), 16);
    }

    #[test]
    fn clamps_and_unbounded() {
        // Tiny RAM → at least 1 expert.
        let tiny = ResourceBudget {
            max_ram_bytes: Some(1),
            max_resident_experts: None,
        };
        assert_eq!(tiny.resident_experts(256, 1 << 30, 0), 1);
        // UNBOUNDED still honors the physical-RAM soft budget (a safety floor), so
        // with tiny per-expert bytes everything fits → all experts.
        assert_eq!(ResourceBudget::UNBOUNDED.resident_experts(256, 1, 0), 256);
        // A model that comfortably fits needs no offload under UNBOUNDED.
        assert!(!ResourceBudget::UNBOUNDED.needs_offload(1 << 20, 256));
    }
}
