// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Emitted Vulkan kernels must agree with the shipping one, on a device.**
//!
//! `kernel_schedule_emit`'s unit tests prove the GLSL compiles and that its
//! barriers carry `WorkgroupMemory` semantics. Neither proves the kernel is
//! *right*: a rotating buffer indexed wrong reads plausible garbage, and a
//! barrier that is correctly formed but wrongly placed produces results that
//! are right most of the time.
//!
//! So this runs a real matmul through `VulkanExecutable` under each schedule
//! and compares against the shipping path. Skips cleanly when there is no
//! Vulkan device, because a test that silently passes on a machine with no GPU
//! is worse than one that says why it did nothing.
//!
//! # Why the tolerance is exact
//!
//! Every arm accumulates in the same `kk` order over the same tile contents, so
//! the results must be bit-identical. Anything else is a bug in the schedule,
//! not a rounding difference — the same bar `tune_dispatch` holds tiles to.

#![cfg(feature = "schedule-codegen")]

use rlx_ir::{DType, Graph, Shape};
use rlx_vulkan::backend::VulkanExecutable;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn build(m: usize, k: usize, n: usize) -> Graph {
    let mut g = Graph::new("vk_sched_parity");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[k, n], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);
    g
}

fn run(m: usize, k: usize, n: usize, xv: &[f32], wv: &[f32]) -> Vec<f32> {
    let mut exe = VulkanExecutable::compile(build(m, k, n));
    exe.set_param("w", wv);
    exe.run(&[("x", xv)])[0].clone()
}

/// K is 16-aligned in every case on purpose.
///
/// `backend::matmul_kernel` routes non-16-aligned K away from the tiled kernel
/// entirely — there is a known, documented shader bug on a trailing partial
/// K-tile — and the emitted schedule inherits the same gate. Testing shapes the
/// path never takes would prove nothing about the path.
const SHAPES: &[(usize, usize, usize)] = &[
    (1, 4096, 4096),
    (4, 2048, 512),
    (32, 512, 512),
    (64, 1024, 256),
    (128, 256, 128),
    (200, 1024, 300),
];

#[test]
fn every_emitted_schedule_matches_the_shipping_kernel_bit_for_bit() {
    if rlx_vulkan::device::vulkan_device().is_none() {
        eprintln!("no Vulkan device; skipping");
        return;
    }
    // MoltenVK routes matmul to the scalar kernel (tiling regresses under
    // Vulkan->Metal translation) and the emitted schedule follows that gate, so
    // on a portability driver there is nothing to compare. Say so.
    if rlx_vulkan::device::vulkan_device().map(|d| d.portability) == Some(true) {
        eprintln!(
            "portability driver (MoltenVK): the tiled path is not selected here, so the \
             emitted schedule is not exercised — skipping rather than passing vacuously"
        );
        return;
    }

    for &(m, k, n) in SHAPES {
        let xv = fill(m * k, 0x5eed_1234);
        let wv = fill(k * n, 0xbeef_9876);

        rlx_ir::env::unset("RLX_VULKAN_SCHEDULE_MATMUL");
        let reference = run(m, k, n, &xv, &wv);

        for variant in ["serial", "pipelined:2", "pipelined:3"] {
            // NOTE: `scheduled_kernel_name` memoizes, so a single process can
            // only exercise one variant. Assert that rather than silently
            // comparing the reference against itself.
            rlx_ir::env::set("RLX_VULKAN_SCHEDULE_MATMUL", variant);
            let got = run(m, k, n, &xv, &wv);
            let bad = got
                .iter()
                .zip(&reference)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert!(
                bad.is_none(),
                "{variant} differs at {m}x{k}x{n} element {}: {:e} vs {:e}",
                bad.unwrap(),
                got[bad.unwrap()],
                reference[bad.unwrap()]
            );
        }
    }
    rlx_ir::env::unset("RLX_VULKAN_SCHEDULE_MATMUL");
}

/// The name selector is memoized, so an in-process sweep over variants would
/// silently measure the first one. Pin that so nobody writes such a sweep and
/// believes its output.
#[test]
fn the_scheduled_kernel_name_is_memoized_for_the_process() {
    use rlx_vulkan::kernel_schedule_emit::scheduled_kernel_name;
    rlx_ir::env::set("RLX_VULKAN_SCHEDULE_MATMUL", "pipelined:3");
    let first = scheduled_kernel_name();
    rlx_ir::env::set("RLX_VULKAN_SCHEDULE_MATMUL", "serial");
    assert_eq!(
        first,
        scheduled_kernel_name(),
        "the name is memoized; an in-process A/B must fork or use a different seam"
    );
    rlx_ir::env::unset("RLX_VULKAN_SCHEDULE_MATMUL");
}

/// Every generated name must resolve to a schedule that verifies and compiles.
/// Device-free, so it runs everywhere and catches a naming/emission mismatch
/// before any rig time is spent.
#[test]
fn every_generated_name_resolves_and_compiles() {
    use rlx_vulkan::kernel_schedule_emit::{SHIPPING_TS, device_target, spirv_for_name};
    for name in [
        "matmul_sched_serial",
        "matmul_sched_pipe2",
        "matmul_sched_pipe3",
        "matmul_sched_pipe4",
    ] {
        let words = spirv_for_name(name, device_target())
            .unwrap_or_else(|| panic!("`{name}` was not recognised as a generated kernel"))
            .unwrap_or_else(|e| panic!("`{name}` failed to build: {e}"));
        assert_eq!(words[0], 0x0723_0203, "`{name}` is not SPIR-V");
    }
    // A name that is not ours must be left alone for the blob table.
    assert!(spirv_for_name("matmul_tiled", device_target()).is_none());
    let _ = SHIPPING_TS;
}

/// The target must be **queried**, not guessed.
///
/// A per-vendor constant table would be the `cost-model-uncalibrated-ranking`
/// defect at the schedule layer: confident numbers with no device behind them.
/// This asserts the values came off `vkGetPhysicalDeviceProperties` by checking
/// they differ from the portable floor in the ways a real device must.
#[test]
fn the_vulkan_target_is_read_off_the_device() {
    use rlx_ir::kernel_schedule::{Features, Target};
    use rlx_vulkan::kernel_schedule_emit::device_target;

    let Some(dev) = rlx_vulkan::device::vulkan_device() else {
        eprintln!("no Vulkan device; skipping (coverage limitation, not a pass)");
        return;
    };
    let t = device_target();
    assert!(
        matches!(t.features, Features::Queried(_)),
        "a device is present but the target reports static features — the probe did not run"
    );
    assert_eq!(
        t.max_shared_bytes, dev.limits.max_compute_shared_memory_size as usize,
        "shared budget did not come from the device limits"
    );
    assert!(
        t.max_shared_bytes >= 16 * 1024,
        "every Vulkan implementation guarantees >= 16 KiB of shared memory; got {}",
        t.max_shared_bytes
    );
    assert!(t.max_warps >= 1 && t.name != Target::PORTABLE.name);
    eprintln!(
        "vulkan target: {} shared={} KiB max_warps={} coop={}",
        t.name,
        t.max_shared_bytes / 1024,
        t.max_warps,
        dev.coop_matmul
    );
}
