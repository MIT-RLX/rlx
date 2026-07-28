// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for TIDE-style expert pool (reference: ims-kdks/TIDE).

use rlx_runtime::{
    ExpertPool, ExpertPoolConfig, ExpertRefreshPolicy, ExpertRefreshResult, MoEExecMode,
};

/// Reproduce TIDE `moe_infer_with_expert_refresh` target selection on fixed indices.
#[test]
fn refresh_matches_topk_by_hit_count() {
    // 4 tokens, top_k=2 style flat indices (duplicate experts allowed).
    let flat: Vec<u32> = vec![1, 1, 0, 2];
    let mut pool = ExpertPool::new(ExpertPoolConfig::new(
        4,
        2,
        ExpertRefreshPolicy::EveryForward,
    ));
    let ExpertRefreshResult { target_gpu, .. } = pool.refresh_from_indices(&flat);
    assert_eq!(target_gpu, vec![1, 0]); // expert 1: 2 hits, expert 0: 1 hit
}

#[test]
fn denoise_jump_steps_aligns_with_tide_main() {
    // main.py uses jump_steps=2 → refresh on steps 0, 2, 4, ...
    let pool = ExpertPool::new(ExpertPoolConfig::new(
        256,
        128,
        ExpertRefreshPolicy::EveryDenoiseSteps(2),
    ));
    for step in 0..8 {
        let expect = step % 2 == 0;
        assert_eq!(
            pool.should_refresh(MoEExecMode::Reuse, step, false),
            expect,
            "step {step}"
        );
    }
}

#[test]
fn explicit_refresh_mode() {
    let pool = ExpertPool::new(ExpertPoolConfig::new(
        8,
        4,
        ExpertRefreshPolicy::EveryDenoiseSteps(100),
    ));
    assert!(pool.should_refresh(MoEExecMode::Refresh, 99, false));
    assert!(!pool.should_refresh(MoEExecMode::Reuse, 99, false));
}
