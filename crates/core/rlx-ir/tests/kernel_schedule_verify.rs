// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The schedule IR earns its place only if its verifier catches things nothing
//! else in rlx can. These tests are that claim, stated as failures.
//!
//! Every hazard below is constructed by corrupting a schedule that is
//! otherwise valid, so a checker that inspects nothing cannot pass.

use rlx_ir::DType;
use rlx_ir::kernel_schedule::{
    Access, Action, Barrier, Feature, KernelSchedule, KernelScheduleError, Layout, Region, Role,
    Space, Swizzle, Target, lower, verify_kernel_schedule,
};

/// rlx's actual tiled matmul, expressed as a schedule: a loader role stages
/// A/B tiles into shared memory, an mma role consumes them, and a barrier
/// gates the handoff. This is `kernels/matmul.cu`'s structure — the part that
/// today exists only inside the CUDA text, where nothing can check it.
fn tiled_matmul() -> KernelSchedule {
    let mut s = KernelSchedule::new("matmul_64x64x16");
    s.stages = 2;
    s.regions = vec![
        Region {
            name: "a_tile".into(),
            space: Space::Shared,
            dims: vec![64, 16],
            dtype: DType::F32,
            stages: 2,
            layout: Layout::row_major(&[64, 16]),
        },
        Region {
            name: "b_tile".into(),
            space: Space::Shared,
            dims: vec![16, 64],
            dtype: DType::F32,
            stages: 2,
            layout: Layout::row_major(&[16, 64]),
        },
        Region {
            name: "acc".into(),
            space: Space::Register,
            dims: vec![4, 4],
            dtype: DType::F32,
            stages: 1,
            layout: Layout::row_major(&[4, 4]),
        },
    ];
    s.roles = vec![
        Role {
            name: "load".into(),
            warps: vec![0, 1],
        },
        Role {
            name: "mma".into(),
            warps: vec![2, 3],
        },
    ];
    s.barriers = vec![Barrier {
        name: "tile_ready".into(),
        producers: vec!["load".into()],
        consumers: vec!["mma".into()],
        count: 1,
    }];
    for stage in 0..2 {
        s.body.entry("load".into()).or_default().extend([
            Action::Load {
                access: Access::plain("a_tile"),
                stage,
            },
            Action::Load {
                access: Access::plain("b_tile"),
                stage,
            },
            Action::Arrive {
                barrier: "tile_ready".into(),
                stage,
            },
        ]);
        s.body.entry("mma".into()).or_default().extend([
            Action::Wait {
                barrier: "tile_ready".into(),
                stage,
            },
            Action::Compute {
                reads: vec![Access::plain("a_tile"), Access::plain("b_tile")],
                writes: vec![Access::plain("acc")],
                stage,
                via: None,
            },
        ]);
    }
    s.body.entry("mma".into()).or_default().push(Action::Store {
        access: Access::plain("acc"),
        stage: 1,
    });
    s.requires = vec![Feature::CoopMatrix { m: 16, n: 8, k: 16 }];
    s
}

#[test]
fn the_real_matmul_schedule_verifies() {
    // If a faithful transcription of rlx's own kernel fails, the verifier is
    // wrong, not the kernel.
    let errors = verify_kernel_schedule(&tiled_matmul(), Target::CUDA_SM86);
    assert!(errors.is_empty(), "valid schedule rejected: {errors:?}");
}

/// A single-role K iteration over a region of `stages` buffers: read slot
/// `read_stage`, then stage into slot `write_stage`, with no barrier between.
///
/// Whether that is a hazard depends entirely on whether the two slots are the
/// same buffer, which is the distinction the write-after-read rule missed until
/// a multi-stage matmul was ported onto it.
fn rotating_buffer(stages: usize, read_stage: usize, write_stage: usize) -> KernelSchedule {
    let mut s = KernelSchedule::new("rotating");
    s.stages = stages;
    s.regions = vec![
        Region {
            name: "tile".into(),
            space: Space::Shared,
            dims: vec![16, 16],
            dtype: DType::F32,
            stages,
            layout: Layout::row_major(&[16, 16]),
        },
        Region {
            name: "acc".into(),
            space: Space::Register,
            dims: vec![4, 4],
            dtype: DType::F32,
            stages: 1,
            layout: Layout::row_major(&[4, 4]),
        },
    ];
    s.roles = vec![Role {
        name: "block".into(),
        warps: vec![0, 1, 2, 3],
    }];
    s.barriers = vec![Barrier {
        name: "filled".into(),
        producers: vec!["block".into()],
        consumers: vec!["block".into()],
        count: 1,
    }];
    s.body.insert(
        "block".into(),
        vec![
            Action::Wait {
                barrier: "filled".into(),
                stage: read_stage,
            },
            Action::Compute {
                reads: vec![Access::plain("tile")],
                writes: vec![Access::plain("acc")],
                stage: read_stage,
                via: None,
            },
            // The staging write. Same buffer as the read above, or not.
            Action::Load {
                access: Access::plain("tile"),
                stage: write_stage,
            },
            Action::Arrive {
                barrier: "filled".into(),
                stage: write_stage,
            },
        ],
    );
    s
}

#[test]
fn overwriting_the_buffer_just_read_still_needs_a_barrier() {
    // One stage: read and write are the same memory, and this is exactly the
    // second `__syncthreads()` in matmul.cu's K loop. The rule must still fire.
    let errors = verify_kernel_schedule(&rotating_buffer(1, 0, 0), Target::CUDA_SM86);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::MissingOverwriteBarrier { .. })),
        "a write-after-read on one buffer must be reported: {errors:?}"
    );
}

#[test]
fn staging_into_a_different_slot_is_not_a_write_after_read() {
    // Two stages, read slot 0, write slot 1: different buffers, so there is no
    // hazard and demanding a barrier would make every software pipeline
    // unexpressible. Keying the rule on the region NAME alone got this wrong.
    let errors = verify_kernel_schedule(&rotating_buffer(2, 0, 1), Target::CUDA_SM86);
    assert!(
        errors.is_empty(),
        "a staged write into another slot is not a hazard: {errors:?}"
    );
}

#[test]
fn a_rotation_that_wraps_onto_the_slot_it_read_is_still_caught() {
    // The stage index is taken modulo the declared depth, so slot 2 of a
    // 2-deep region IS slot 0. A rule that compared raw stage indices would
    // wave this through — the regression the fix must not introduce.
    let errors = verify_kernel_schedule(&rotating_buffer(2, 0, 2), Target::CUDA_SM86);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::MissingOverwriteBarrier { .. })),
        "stage 2 of a 2-deep region is slot 0 and must be caught: {errors:?}"
    );
}

#[test]
fn a_missing_barrier_is_a_race_the_op_graph_cannot_see() {
    // THE motivating case. Drop the synchronization and the graph is still a
    // perfectly correct matmul — the arithmetic, shapes and representations are
    // all unchanged. What changed is that one role now reads a buffer while
    // another writes it.
    let mut s = tiled_matmul();
    s.barriers.clear();
    for acts in s.body.values_mut() {
        acts.retain(|a| !matches!(a, Action::Arrive { .. } | Action::Wait { .. }));
    }
    let errors = verify_kernel_schedule(&s, Target::CUDA_SM86);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::UnsynchronizedHandoff { .. })),
        "an unsynchronized cross-role handoff must be reported: {errors:?}"
    );
}

#[test]
fn a_wait_with_no_arrival_is_a_hang() {
    let mut s = tiled_matmul();
    // Remove every arrival but keep the waits: the classic deadlock.
    s.body
        .get_mut("load")
        .unwrap()
        .retain(|a| !matches!(a, Action::Arrive { .. }));
    let errors = verify_kernel_schedule(&s, Target::CUDA_SM86);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::DeadlockWait { .. })),
        "a wait no role arrives at must be reported: {errors:?}"
    );
}

#[test]
fn an_unmeetable_arrival_count_is_a_hang() {
    let mut s = tiled_matmul();
    s.barriers[0].count = 4; // only one role arrives
    let errors = verify_kernel_schedule(&s, Target::CUDA_SM86);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::UnreachableCount { .. })),
        "a barrier needing more arrivals than exist must be reported: {errors:?}"
    );
}

#[test]
fn shared_memory_over_budget_is_caught_before_launch() {
    // The failure that shows up as CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES at run
    // time, derived here from the declarations instead.
    let mut s = tiled_matmul();
    s.regions[0].dims = vec![512, 512];
    let errors = verify_kernel_schedule(&s, Target::CUDA_SM86);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::SharedOverBudget { .. })),
        "over-budget shared memory must be reported: {errors:?}"
    );
    // And the portable floor is stricter than sm_86, so a schedule can be
    // legal on one target and not another — which is the point of naming it.
    // Sized to land BETWEEN the two limits: 2 tiles x 2560 elems x 4 B x 2
    // stages = 40 KiB, which fits sm_86's 48 KiB and busts the 32 KiB floor.
    let mut s = tiled_matmul();
    s.regions[0].dims = vec![64, 40];
    s.regions[1].dims = vec![40, 64];
    let sm86 = verify_kernel_schedule(&s, Target::CUDA_SM86);
    let portable = verify_kernel_schedule(&s, Target::PORTABLE);
    assert!(sm86.is_empty(), "should fit sm_86's 48 KiB: {sm86:?}");
    assert!(
        portable
            .iter()
            .any(|e| matches!(e, KernelScheduleError::SharedOverBudget { .. })),
        "should NOT fit the 32 KiB portable floor: {portable:?}"
    );
}

#[test]
fn two_roles_cannot_claim_the_same_warp() {
    let mut s = tiled_matmul();
    s.roles[1].warps = vec![1, 2]; // warp 1 also belongs to `load`
    let errors = verify_kernel_schedule(&s, Target::CUDA_SM86);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::WarpConflict { .. })),
        "overlapping warp assignment must be reported: {errors:?}"
    );
}

#[test]
fn undeclared_names_are_rejected() {
    let mut s = tiled_matmul();
    s.body.get_mut("mma").unwrap().push(Action::Compute {
        reads: vec![Access::plain("c_tile")],
        writes: vec![Access::plain("acc")],
        stage: 0,
        via: None,
    });
    let errors = verify_kernel_schedule(&s, Target::CUDA_SM86);
    assert!(
        errors.iter().any(|e| matches!(
            e,
            KernelScheduleError::UndeclaredName { kind: "region", .. }
        )),
        "a region that was never declared must be reported: {errors:?}"
    );
}

#[test]
fn a_stage_past_the_pipeline_depth_is_rejected() {
    let mut s = tiled_matmul();
    s.body.get_mut("load").unwrap().push(Action::Load {
        access: Access::plain("a_tile"),
        stage: 7,
    });
    let errors = verify_kernel_schedule(&s, Target::CUDA_SM86);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::StageOutOfRange { .. })),
        "a stage index past the declared depth must be reported: {errors:?}"
    );
}

#[test]
fn metadata_is_derived_not_declared() {
    // CAKE's fourth property: the mechanical consequences of the declarations
    // are computed, not written out. Nothing states the byte total or the warp
    // set, so neither can drift from the regions and roles that imply them.
    let s = tiled_matmul();
    let expected = (64 * 16 * 4 * 2) + (16 * 64 * 4 * 2);
    assert_eq!(s.bytes_in(Space::Shared), expected);
    assert_eq!(s.warps().into_iter().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    assert_eq!(s.bytes_in(Space::Register), 4 * 4 * 4);
}

// ── Layout commitments (CAKE B.4) ───────────────────────────────────────────

#[test]
fn a_transposed_read_of_the_same_region_is_caught() {
    // THE case this exists for. `rocm-gguf-transposed`: one side treats the
    // operand as row-major [n,k], the other reads it column-major. The dims
    // are identical, so no shape check can see it — the numbers just come out
    // wrong.
    let mut s = tiled_matmul();
    let acts = s.body.get_mut("mma").unwrap();
    for a in acts.iter_mut() {
        if let Action::Compute { reads, .. } = a {
            reads[1] = Access::with("b_tile", Layout::col_major(&[16, 64]));
        }
    }
    let errors = verify_kernel_schedule(&s, Target::CUDA_SM86);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::LayoutDisagreement { .. })),
        "a transposed commitment on a shared region must be reported: {errors:?}"
    );
}

#[test]
fn agreeing_explicit_commitments_are_fine() {
    // Guards the check from firing whenever a layout is stated at all. Naming
    // the region's own layout explicitly must be a no-op.
    let mut s = tiled_matmul();
    let acts = s.body.get_mut("mma").unwrap();
    for a in acts.iter_mut() {
        if let Action::Compute { reads, .. } = a {
            reads[0] = Access::with("a_tile", Layout::row_major(&[64, 16]));
        }
    }
    let errors = verify_kernel_schedule(&s, Target::CUDA_SM86);
    assert!(
        errors.is_empty(),
        "restating the declared layout is not a conflict: {errors:?}"
    );
}

#[test]
fn a_layout_that_runs_past_its_region_is_caught() {
    let mut s = tiled_matmul();
    let acts = s.body.get_mut("load").unwrap();
    if let Action::Load { access, .. } = &mut acts[0] {
        // Strides far too large for a [64,16] region.
        *access = Access::with(
            "a_tile",
            Layout {
                offset_bytes: 0,
                strides: vec![4096, 1],
                swizzle: Swizzle::None,
            },
        );
    }
    let errors = verify_kernel_schedule(&s, Target::CUDA_SM86);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::LayoutOutOfRegion { .. })),
        "a view addressing past its region must be reported: {errors:?}"
    );
}

// ── Exact target match (CAKE B.5) ───────────────────────────────────────────

#[test]
fn a_missing_capability_is_reported_not_stepped_down() {
    // sm_70 has tensor cores but no async copy. A schedule needing it must be
    // REJECTED, not silently lowered to a synchronous path — the silent
    // step-down is how an ineligible Metal variant reached a shape it could
    // not handle.
    let mut s = tiled_matmul();
    s.requires.push(Feature::AsyncCopy);
    assert!(
        verify_kernel_schedule(&s, Target::CUDA_SM86).is_empty(),
        "sm_86 provides async copy"
    );
    let errors = verify_kernel_schedule(&s, Target::CUDA_SM70);
    assert!(
        errors.iter().any(|e| matches!(
            e,
            KernelScheduleError::UnsupportedFeature {
                feature: Feature::AsyncCopy,
                ..
            }
        )),
        "sm_70 lacks async copy and must say so: {errors:?}"
    );
}

#[test]
fn a_swizzled_commitment_requires_the_capability_even_if_undeclared() {
    // Using a feature is a requirement whether or not the author remembered to
    // declare it. Otherwise `requires` is documentation, not a contract.
    let mut s = tiled_matmul();
    s.regions[0].layout.swizzle = Swizzle::Xor(8);
    assert!(
        verify_kernel_schedule(&s, Target::CUDA_SM86).is_empty(),
        "sm_86 supports swizzle"
    );
    let errors = verify_kernel_schedule(&s, Target::PORTABLE);
    assert!(
        errors.iter().any(|e| matches!(
            e,
            KernelScheduleError::UnsupportedFeature {
                feature: Feature::Swizzle,
                ..
            }
        )),
        "the portable floor has no swizzle and must say so: {errors:?}"
    );
}

// ── Lowering derives, never restates (CAKE §2.2) ────────────────────────────

#[test]
fn lowering_derives_offsets_slots_and_identity() {
    let s = tiled_matmul();
    let l = lower(&s, Target::CUDA_SM86).expect("valid schedule lowers");

    // Shared regions are packed in declaration order; `acc` is a register
    // region and must not consume shared bytes.
    assert_eq!(l.region_offsets.get("a_tile"), Some(&0));
    assert_eq!(l.region_offsets.get("b_tile"), Some(&(64 * 16 * 4 * 2)));
    assert!(
        !l.region_offsets.contains_key("acc"),
        "registers are not shared memory"
    );
    assert_eq!(l.shared_bytes, s.bytes_in(Space::Shared));

    assert_eq!(l.barrier_slots.get("tile_ready"), Some(&0));
    assert_eq!(l.role_base_warp.get("load"), Some(&0));
    assert_eq!(l.role_base_warp.get("mma"), Some(&2));
    assert_eq!(l.threads, 4 * 32);
}

#[test]
fn an_invalid_schedule_does_not_lower() {
    // Deriving addresses for a program that cannot run would produce
    // confident-looking metadata for a race.
    let mut s = tiled_matmul();
    s.barriers.clear();
    for acts in s.body.values_mut() {
        acts.retain(|a| !matches!(a, Action::Arrive { .. } | Action::Wait { .. }));
    }
    let err = lower(&s, Target::CUDA_SM86).expect_err("an unsynchronized schedule must not lower");
    assert!(
        err.iter()
            .any(|e| matches!(e, KernelScheduleError::UnsynchronizedHandoff { .. }))
    );
}
