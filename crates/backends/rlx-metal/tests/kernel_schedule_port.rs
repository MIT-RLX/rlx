// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Metal port, checked against the MSL that actually ships.
//!
//! Gated exactly like the module it tests. `kernel_schedule_port` is
//! `#[cfg(rlx_metal_host)]` because it scans the shipping MSL, and a workspace
//! `cargo test` builds every crate's test targets on every platform — so an
//! ungated test importing a host-gated module fails to COMPILE on Linux and
//! takes the whole workspace suite down with it. That is the
//! `linux_workspace_test_gate` trap, and gating the module without gating its
//! test only moves it.
#![cfg(rlx_metal_host)]

use rlx_ir::kernel_schedule::{
    Access, Action, Feature, Instruction, KernelScheduleError, Layout, Space, Target,
    verify_kernel_schedule,
};
use rlx_metal::kernel_schedule_port::{
    METAL_TARGET, SIMDGROUP_TILE, TG_TILE, hgemm_simd_4x4_schedule, simdgroup_strides_in, verify,
};

#[test]
fn the_metal_schedule_verifies() {
    verify().unwrap_or_else(|e| panic!("hgemm_simd_4x4 schedule is unsound: {e:?}"));
}

#[test]
fn the_declared_stride_matches_every_simdgroup_load_in_the_source() {
    // THE check. `simdgroup_load(a, &A_tg[...], 32)` — that 32 is the row
    // stride, and it is read out of the shipping MSL rather than retyped here.
    // A stride that disagrees with the region compiles cleanly and reads the
    // wrong elements.
    let strides = simdgroup_strides_in("hgemm_simd_4x4");
    // Exactly two: the pair of `simdgroup_load` calls over threadgroup memory.
    //
    // The `simdgroup_store` is deliberately absent — it writes device memory
    // with `N`, a runtime kernel argument, so its stride is not a literal and
    // is outside what a source scan can check. Stating that is the honest
    // scope of this gate: it covers on-chip staging, where the stride is fixed
    // at compile time and where a mismatch is silent, and not the global
    // epilogue, where the stride comes from the caller.
    assert_eq!(
        strides,
        vec![32, 32],
        "expected both threadgroup simdgroup_loads to pass a literal stride; if the \
         kernel changed shape, revisit the scan rather than relaxing this"
    );

    let sched = hgemm_simd_4x4_schedule();
    let tg = sched
        .regions
        .iter()
        .find(|r| r.name == "A_tg")
        .expect("A_tg declared");
    let declared = tg.layout.strides[0];
    assert_eq!(
        declared, TG_TILE,
        "A_tg is [32,32] row-major, so row stride 32"
    );

    // Every threadgroup-sourced load must be told exactly that stride. The
    // store writes device memory with a runtime stride `N`, which is not a
    // literal and so is not in `strides`.
    let from_shared: Vec<usize> = strides.iter().copied().filter(|s| *s == declared).collect();
    assert!(
        from_shared.len() >= 2,
        "both simdgroup_loads should pass the declared stride {declared}; got {strides:?}"
    );
}

#[test]
fn a_stride_disagreeing_with_the_region_is_caught() {
    // Simulate the defect: the schedule commits to a stride the instruction's
    // tile cannot divide.
    let mut s = hgemm_simd_4x4_schedule();
    if let Some(a) = s.regions.iter_mut().find(|r| r.name == "A_tg") {
        // 33 is not a multiple of the 8-wide simdgroup tile.
        a.layout = Layout {
            offset_bytes: 0,
            strides: vec![33, 1],
            swizzle: rlx_ir::kernel_schedule::Swizzle::None,
        };
        a.dims = vec![TG_TILE, 33];
    }
    let errors = verify_kernel_schedule(&s, METAL_TARGET);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::OperandStrideNotTiled { .. })),
        "a row stride the 8x8 tile cannot divide must be reported: {errors:?}"
    );
}

#[test]
fn a_target_without_the_8x8_shape_rejects_the_kernel() {
    // The wgpu/Vulkan case in miniature: sm_86 has cooperative matrices, but
    // at m16n8k16 — NOT the 8x8x8 shape `simdgroup_half8x8` needs. A bare
    // "has tensor cores" flag could not tell these apart.
    assert!(METAL_TARGET.has(Feature::CoopMatrix { m: 8, n: 8, k: 8 }));
    assert!(!Target::CUDA_SM86.has(Feature::CoopMatrix { m: 8, n: 8, k: 8 }));

    let errors = verify_kernel_schedule(&hgemm_simd_4x4_schedule(), Target::CUDA_SM86);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::UnsupportedFeature { .. })),
        "an 8x8x8 kernel must be rejected on a target offering only 16x8x16: {errors:?}"
    );
}

#[test]
fn the_scalar_path_is_unconstrained() {
    // `via: None` means a scalar loop, which no instruction contract governs.
    // Without this, adding operand checks would have silently constrained every
    // non-tensor-core kernel in the tree.
    let mut s = hgemm_simd_4x4_schedule();
    for acts in s.body.values_mut() {
        for a in acts.iter_mut() {
            if let Action::Compute { via, .. } = a {
                *via = None;
            }
        }
    }
    if let Some(a) = s.regions.iter_mut().find(|r| r.name == "A_tg") {
        a.dims = vec![TG_TILE, 33];
        a.layout = Layout::row_major(&[TG_TILE, 33]);
    }
    let errors = verify_kernel_schedule(&s, METAL_TARGET);
    assert!(
        !errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::OperandStrideNotTiled { .. })),
        "a scalar-path region must not be held to a tensor-core operand contract: {errors:?}"
    );
}

#[test]
fn the_threadgroup_budget_is_derived_from_the_declarations() {
    // 2 tiles x 32x32 x 2 bytes (f16) = 4 KiB, well inside Apple's 32 KiB.
    let s = hgemm_simd_4x4_schedule();
    assert_eq!(s.bytes_in(Space::Shared), 2 * TG_TILE * TG_TILE * 2);
    assert_eq!(s.warps().len(), 16, "16 simdgroups");
}

#[test]
fn dropping_a_threadgroup_barrier_is_caught() {
    let mut s = hgemm_simd_4x4_schedule();
    let acts = s.body.get_mut("threadgroup").unwrap();
    acts.retain(
        |a| !matches!(a, Action::Arrive { barrier, .. } | Action::Wait { barrier, .. } if barrier == "tiles_filled"),
    );
    let errors = verify_kernel_schedule(&s, METAL_TARGET);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::MissingReadBarrier { .. })),
        "a missing threadgroup_barrier must be reported: {errors:?}"
    );
}

#[test]
fn the_extractor_finds_nothing_for_an_unknown_kernel() {
    // Guards the source scan from silently returning an empty list that a
    // caller then reads as "all strides agree".
    assert!(simdgroup_strides_in("no_such_kernel_exists").is_empty());
    let _ = Access::plain("x");
    let _ = Instruction::CoopMatrix {
        m: SIMDGROUP_TILE,
        n: SIMDGROUP_TILE,
        k: SIMDGROUP_TILE,
    };
}
