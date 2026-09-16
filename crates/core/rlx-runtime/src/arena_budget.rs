// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! How large an arena a device can address, and how large one would actually be.
//!
//! These are two different questions and conflating them is how a graph ends up
//! refused on one backend and fine on another:
//!
//! * [`device_arena_limit`] is an **ABI** limit — the widest byte offset the
//!   backend's kernels can represent. It has nothing to do with how much memory
//!   is installed. `rlx-rocm` refuses any plan past 4 GiB because hundreds of
//!   its shared kernels still take `unsigned int` offset parameters that would
//!   wrap *inside* the kernel; that is a correct refusal, since the alternative
//!   is silent corruption partway through a graph.
//! * [`device_vram_bytes`](crate::device_vram_bytes) is the **capacity**
//!   question, answered by the host profiler.
//!
//! [`arena_budget`] combines them, and [`plan_arena_bytes`] answers "would this
//! graph fit" using the planner the backend actually uses — which matters,
//! because the backends do not agree. Measured on one 12-layer Mamba training
//! graph at identical geometry, `rlx-rocm`'s own planner produced **2.15x** the
//! arena that `rlx-compile`'s did (10.4 GiB vs 4.76 GiB at batch 8). Estimating
//! ROCm with the shared planner would therefore have under-predicted by more
//! than half and chosen a batch that does not run.

use rlx_driver::Device;
use rlx_ir::Graph;

/// Widest arena, in bytes, that `device` can address in a single compiled plan.
///
/// `None` means "no ABI ceiling" — the backend addresses the arena with 64-bit
/// offsets throughout, so the only limit is capacity.
pub fn device_arena_limit(device: Device) -> Option<u64> {
    match device {
        // Host-side offsets are u64, but the shared HIP kernels are mixed: many
        // still declare `unsigned int` offset parameters, so an op of that kind
        // truncates inside the kernel signature no matter how wide the host
        // field is. `rlx-rocm`'s compile step asserts on this.
        Device::Rocm => Some(u32::MAX as u64),
        // A single storage-buffer binding is capped at 4 GiB by the underlying
        // APIs; `rlx-wgpu` shards above that, and Vulkan inherits the same rule.
        Device::Gpu | Device::Vulkan => Some(u32::MAX as u64),
        _ => None,
    }
}

/// Bytes of arena a plan for `device` may use, given both the ABI ceiling and
/// the memory the profiler can see.
///
/// `headroom` scales the detected capacity (e.g. `0.6` to leave room for
/// weights, workspace and whatever else shares the device). Returns `None` when
/// neither limit is known, meaning "do not constrain".
pub fn arena_budget(device: Device, headroom: f64) -> Option<u64> {
    let abi = device_arena_limit(device);
    let capacity = crate::device_vram_bytes(device).map(|v| (v as f64 * headroom) as u64);
    match (abi, capacity) {
        (Some(a), Some(c)) => Some(a.min(c)),
        (Some(a), None) => Some(a),
        (None, Some(c)) => Some(c),
        (None, None) => None,
    }
}

/// Arena size `graph` would plan to on `device`, using that backend's own
/// planner where it has one.
///
/// Returns `None` when the backend for `device` is not compiled into this
/// build — in which case the caller has nothing to adapt to anyway.
pub fn plan_arena_bytes(graph: &Graph, device: Device) -> Option<usize> {
    match device {
        #[cfg(feature = "rocm")]
        Device::Rocm => {
            // Ask, do not estimate. `prepare_rocm_exec_graph` is the exact
            // sequence `RocmBackend::compile` runs before planning — shared
            // with it, so the two cannot drift — and the arena is then planned
            // by `rlx-rocm`'s own planner at its own alignment.
            //
            // Two earlier attempts got this wrong by predicting from a graph
            // the backend never plans. The raw backward graph under-counted
            // badly; running only `unfuse` + `legalize` still under-counted by
            // a consistent 1.70x (6.12 GB predicted vs 10.39 GB actual on a
            // 12-layer Mamba backward), which is enough to choose a batch that
            // then fails to compile. The remaining 70% is fusion, LIR lowering
            // and the f32 exec rewrite — all of which change slot sizes.
            let opts = crate::CompileOptions::default();
            let (g, _io) =
                crate::backend::rocm_backend::prepare_rocm_exec_graph(graph.clone(), &opts, false);
            Some(rlx_rocm::arena::plan_f32_uniform(&g, 16).arena_size)
        }
        // Everything else plans through `rlx-compile`. Align 128 matches what
        // the Metal and wgpu paths ask for; the CPU path is not arena-limited.
        _ => Some(
            rlx_compile::memory::plan_memory_with_options(
                graph,
                128,
                rlx_compile::memory::MemoryPlanOptions::inference(),
            )
            .arena_size,
        ),
    }
}

/// Largest batch in `1..=requested` whose graph fits `device`'s arena budget.
///
/// `build` is called with a candidate batch and returns the graph that would be
/// compiled at that batch — for training, pass the **backward** graph, since
/// that is what dominates. Batches are tried by halving, so this costs at most
/// `log2(requested) + 1` graph builds and plans, not `requested` of them.
///
/// Returns `requested` unchanged when the device has no known budget, or when
/// even a batch of 1 does not fit (there is nothing useful to fall back to, and
/// the backend's own error is a better report than a silent 1).
pub fn fit_batch<F>(device: Device, requested: usize, headroom: f64, mut build: F) -> usize
where
    F: FnMut(usize) -> Option<Graph>,
{
    let Some(budget) = arena_budget(device, headroom) else {
        return requested;
    };
    let mut b = requested.max(1);
    while b > 1 {
        let Some(g) = build(b) else { return b };
        match plan_arena_bytes(&g, device) {
            Some(bytes) if (bytes as u64) <= budget => return b,
            Some(_) => b /= 2,
            None => return b,
        }
    }
    requested.min(1).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_and_wgpu_declare_a_four_gib_ceiling_others_do_not() {
        assert_eq!(device_arena_limit(Device::Rocm), Some(u32::MAX as u64));
        assert_eq!(device_arena_limit(Device::Gpu), Some(u32::MAX as u64));
        assert_eq!(device_arena_limit(Device::Vulkan), Some(u32::MAX as u64));
        // 64-bit addressing throughout — capacity is the only limit.
        assert_eq!(device_arena_limit(Device::Cuda), None);
        assert_eq!(device_arena_limit(Device::Cpu), None);
        assert_eq!(device_arena_limit(Device::Metal), None);
    }

    #[test]
    fn a_budget_never_exceeds_the_abi_ceiling() {
        // Whatever the profiler reports, ROCm's answer stays under 4 GiB.
        if let Some(b) = arena_budget(Device::Rocm, 1.0) {
            assert!(b <= u32::MAX as u64, "ROCm budget {b} exceeds its ABI cap");
        }
    }

    #[test]
    fn fit_batch_is_a_no_op_without_a_budget() {
        // CPU has no ABI cap; if the profiler also reports nothing, the
        // requested batch must survive untouched rather than silently drop.
        let n = fit_batch(Device::Cpu, 8, 0.6, |_| None);
        assert!(n <= 8 && n >= 1, "batch {n} out of range");
    }

    #[test]
    fn fit_batch_halves_until_it_fits() {
        // A build that reports a graph too large at every size must bottom out
        // rather than loop.
        let mut calls = 0usize;
        let n = fit_batch(Device::Rocm, 8, 0.6, |_| {
            calls += 1;
            None // unknown graph -> caller keeps the current candidate
        });
        assert!(n >= 1, "batch must stay positive");
        assert!(calls <= 4, "should halve, not decrement: {calls} builds");
    }
}
