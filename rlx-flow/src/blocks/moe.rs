// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! MoE FFN placeholder for Gemma 4 recipes (dense path ignores this).

/// Routed-expert FFN stage (stub — wired when `rlx-flow` MoE kernels land).
#[derive(Debug, Clone)]
pub struct MoeFfnStage {
    pub prefix: String,
    pub num_experts: usize,
    pub top_k: usize,
    pub n_embd: usize,
    pub n_ff: usize,
}

impl MoeFfnStage {
    pub fn hf(
        prefix: impl Into<String>,
        num_experts: usize,
        top_k: usize,
        n_embd: usize,
        n_ff: usize,
    ) -> Self {
        Self {
            prefix: prefix.into(),
            num_experts,
            top_k,
            n_embd,
            n_ff,
        }
    }
}
