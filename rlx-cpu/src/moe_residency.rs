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

//! Per-forward MoE expert residency mask (TIDE placement) for CPU dispatch.
//!
//! Set by [`rlx_runtime::CompiledGraph::set_moe_resident_experts`] before
//! `run`. [`crate::thunk::GroupedMatMul`] reads the mask for accounting;
//! numerics still use the full expert stack in the arena (lossless on CPU).

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Per-expert host pointers for one MoE layer (gate/up/down).
#[derive(Debug, Clone)]
pub struct LayerHostBind {
    pub gate: Vec<*const f32>,
    pub up: Vec<*const f32>,
    pub down: Vec<*const f32>,
    pub stride: usize,
}

/// Host weight lookup installed before forward (TIDE CPU/GPU fallback path).
#[derive(Debug, Clone)]
pub struct MoeHostBind {
    pub layers: Vec<LayerHostBind>,
}

// Host weight pointers are installed on the calling thread; safe to share
// across the RwLock used for TIDE residency bookkeeping.
unsafe impl Send for LayerHostBind {}
unsafe impl Sync for LayerHostBind {}
unsafe impl Send for MoeHostBind {}
unsafe impl Sync for MoeHostBind {}

static HOST_BIND: RwLock<Option<MoeHostBind>> = RwLock::new(None);
static LAST_STATS: RwLock<Option<MoeResidencyStats>> = RwLock::new(None);
/// Monotonic GroupedMatMul ordinal within one forward (layer = ord/3, matrix = ord%3).
static GMM_ORD: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Default, Clone)]
pub struct MoeResidencyStats {
    pub gpu_expert_calls: u64,
    pub cpu_expert_calls: u64,
    pub gpu_tokens: u64,
    pub cpu_tokens: u64,
}

struct MoeResidencyCtx {
    /// Union mask (legacy): expert on device if any layer has it resident.
    merged: Option<Arc<[bool]>>,
    /// TIDE per MoE layer (forward order); takes precedence over [`merged`].
    per_layer: Option<Arc<Vec<Arc<[bool]>>>>,
    stats: MoeResidencyStats,
}

thread_local! {
    static CTX: RefCell<Option<MoeResidencyCtx>> = RefCell::new(None);
}

/// Install merged (union) residency mask for the current thread until [`clear_mask`].
pub fn set_mask(mask: Option<Arc<[bool]>>) {
    CTX.with(|c| {
        *c.borrow_mut() = Some(MoeResidencyCtx {
            merged: mask,
            per_layer: None,
            stats: MoeResidencyStats::default(),
        });
    });
}

/// Install per-layer masks (one row per MoE FFN in forward order).
pub fn set_per_layer_masks(layers: Option<Arc<Vec<Arc<[bool]>>>>) {
    CTX.with(|c| {
        *c.borrow_mut() = Some(MoeResidencyCtx {
            merged: None,
            per_layer: layers,
            stats: MoeResidencyStats::default(),
        });
    });
}

pub fn clear_mask() {
    CTX.with(|c| *c.borrow_mut() = None);
}

pub fn bind_host_weights(bind: Option<MoeHostBind>) {
    *HOST_BIND.write().unwrap() = bind;
}

pub fn reset_gmm_counters() {
    GMM_ORD.store(0, Ordering::Relaxed);
}

/// Next MoE GroupedMatMul ordinal for this forward (call once per kernel).
pub fn next_gmm_ord() -> usize {
    GMM_ORD.fetch_add(1, Ordering::Relaxed)
}

/// Host expert weight pointer for `ord` (gate/up/down = ord%3, layer = ord/3).
pub fn host_expert_weight_ptr(ord: usize, expert: usize) -> Option<*const f32> {
    let bind = HOST_BIND.read().unwrap();
    let bind = bind.as_ref()?;
    let layer = bind.layers.get(ord / 3)?;
    let ptrs = match ord % 3 {
        0 => &layer.gate,
        1 => &layer.up,
        _ => &layer.down,
    };
    ptrs.get(expert).copied()
}

pub fn peek_stats() -> Option<MoeResidencyStats> {
    CTX.with(|c| c.borrow().as_ref().map(|ctx| ctx.stats.clone()))
}

/// Stats from the most recent forward on this thread (set when the residency guard drops).
pub fn take_last_forward_stats() -> Option<MoeResidencyStats> {
    LAST_STATS.write().unwrap().take()
}

pub(crate) fn stash_last_forward_stats(stats: MoeResidencyStats) {
    *LAST_STATS.write().unwrap() = Some(stats);
}

fn expert_on_device_inner(ctx: &MoeResidencyCtx, layer: Option<usize>, e: usize) -> bool {
    if let Some(layers) = ctx.per_layer.as_ref() {
        if let Some(li) = layer {
            return layers
                .get(li)
                .and_then(|m| m.get(e).copied())
                .unwrap_or(true);
        }
    }
    ctx.merged
        .as_ref()
        .and_then(|m| m.get(e).copied())
        .unwrap_or(true)
}

/// True when expert `e` is GPU-resident for MoE layer `layer` (GMM ord / 3).
pub fn expert_on_device_for_layer(layer: usize, e: usize) -> bool {
    CTX.with(|c| {
        let borrow = c.borrow();
        let Some(ctx) = borrow.as_ref() else {
            return true;
        };
        expert_on_device_inner(ctx, Some(layer), e)
    })
}

/// True when expert `e` is marked GPU-resident (merged mask if no per-layer table).
pub fn expert_on_device(e: usize) -> bool {
    expert_on_device_for_layer(0, e)
}

pub fn record_expert_tokens(layer: usize, e: usize, num_tokens: usize) {
    if num_tokens == 0 {
        return;
    }
    CTX.with(|c| {
        let mut borrow = c.borrow_mut();
        let Some(ctx) = borrow.as_mut() else {
            return;
        };
        let on_device = expert_on_device_inner(ctx, Some(layer), e);
        if on_device {
            ctx.stats.gpu_expert_calls += 1;
            ctx.stats.gpu_tokens += num_tokens as u64;
        } else {
            ctx.stats.cpu_expert_calls += 1;
            ctx.stats.cpu_tokens += num_tokens as u64;
        }
    });
}

/// Take stats and clear the thread-local context.
pub fn take_stats() -> Option<MoeResidencyStats> {
    CTX.with(|c| c.borrow_mut().take().map(|ctx| ctx.stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn per_layer_masks_are_layer_local() {
        let per = Arc::new(vec![
            Arc::from([false, true, true, true]),
            Arc::from([true, false, true, true]),
        ]);
        set_per_layer_masks(Some(per));
        assert!(!expert_on_device_for_layer(0, 0));
        assert!(expert_on_device_for_layer(0, 1));
        assert!(expert_on_device_for_layer(1, 0));
        assert!(!expert_on_device_for_layer(1, 1));
        clear_mask();
    }
}
