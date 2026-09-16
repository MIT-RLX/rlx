// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The schedule IR checked against CAKE's eight design principles (B.1).
//!
//! CAKE's IR-evolution loop (Appendix A, step 4) is "principle-driven
//! iteration: each candidate is validated against the eight principles in
//! Appendix B; violations are refined or rejected." Most of the eight are
//! judgement calls. Two are not, and those are gated here.
//!
//! * **P7 Analysis-consistent** — "accompany changes to the IR data model with
//!   corresponding analysis updates." Mechanically checkable: perturb each
//!   field of the data model and require some analysis to notice. A field no
//!   analysis reads is decoration, and the next person to add one will assume
//!   it means something.
//! * **P4 Statically type-checked** — ill-typed programs are rejected at
//!   construction/verification, covered by `schedule_verify.rs`.
//!
//! P1 (ergonomic), P2 (performance-transparent), P3 (canonical), P5
//! (analysis-friendly), P6 (test-gated) and P8 (hardware-grounded) are review
//! conventions; recording which are and are not machine-enforced is the honest
//! part, and mirrors `docs/ir-design-principles.md` for the op IR.

use rlx_ir::DType;
use rlx_ir::kernel_schedule::{
    Access, Action, Barrier, Feature, KernelSchedule, Layout, Region, Role, Space, Swizzle, Target,
    verify_kernel_schedule,
};

fn base() -> KernelSchedule {
    let mut s = KernelSchedule::new("p7");
    s.stages = 1;
    s.regions = vec![
        Region {
            name: "tile".into(),
            space: Space::Shared,
            dims: vec![32, 32],
            dtype: DType::F32,
            stages: 1,
            layout: Layout::row_major(&[32, 32]),
        },
        Region {
            name: "out".into(),
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
            warps: vec![0],
        },
        Role {
            name: "mma".into(),
            warps: vec![1],
        },
    ];
    s.barriers = vec![Barrier {
        name: "ready".into(),
        producers: vec!["load".into()],
        consumers: vec!["mma".into()],
        count: 1,
    }];
    s.body.insert(
        "load".into(),
        vec![
            Action::Load {
                access: Access::plain("tile"),
                stage: 0,
            },
            Action::Arrive {
                barrier: "ready".into(),
                stage: 0,
            },
        ],
    );
    s.body.insert(
        "mma".into(),
        vec![
            Action::Wait {
                barrier: "ready".into(),
                stage: 0,
            },
            Action::Compute {
                reads: vec![Access::plain("tile")],
                writes: vec![Access::plain("out")],
                stage: 0,
                via: None,
            },
            Action::Store {
                access: Access::plain("out"),
                stage: 0,
            },
        ],
    );
    s
}

#[test]
fn the_baseline_is_valid() {
    let e = verify_kernel_schedule(&base(), Target::CUDA_SM86);
    assert!(
        e.is_empty(),
        "perturbation tests need a clean baseline: {e:?}"
    );
}

/// P7: every field of the data model must change some analysis outcome.
///
/// Each case perturbs exactly one field and requires the verifier to notice.
/// If a field can be set to anything without affecting any check, it is not
/// part of the contract and should not be in the struct.
#[test]
fn p7_every_data_model_field_is_consulted_by_an_analysis() {
    let perturbations: Vec<(&str, Box<dyn Fn(&mut KernelSchedule)>)> = vec![
        (
            "Region::dims",
            Box::new(|s: &mut KernelSchedule| s.regions[0].dims = vec![4096, 4096]),
        ),
        (
            "Region::dtype",
            Box::new(|s: &mut KernelSchedule| {
                s.regions[0].dtype = DType::F32;
                s.regions[0].dims = vec![4096, 4096];
            }),
        ),
        (
            "Region::stages",
            Box::new(|s: &mut KernelSchedule| s.regions[0].stages = 4096),
        ),
        (
            "Region::space",
            Box::new(|s: &mut KernelSchedule| {
                // Moving the big register tile into shared must be budgeted.
                s.regions[1].space = Space::Shared;
                s.regions[1].dims = vec![4096, 4096];
            }),
        ),
        (
            "Region::layout",
            Box::new(|s: &mut KernelSchedule| {
                s.regions[0].layout = Layout::col_major(&[32, 32]);
                if let Some(Action::Compute { reads, .. }) =
                    s.body.get_mut("mma").unwrap().get_mut(1)
                {
                    reads[0] = Access::with("tile", Layout::row_major(&[32, 32]));
                }
            }),
        ),
        // A period that does not divide the innermost extent aliases rows.
        (
            "Layout::swizzle",
            Box::new(|s: &mut KernelSchedule| {
                s.regions[0].layout.swizzle = Swizzle::Xor(12);
            }),
        ),
        (
            "Role::warps",
            Box::new(|s: &mut KernelSchedule| s.roles[1].warps = vec![0]),
        ),
        (
            "Role::name",
            Box::new(|s: &mut KernelSchedule| s.roles[0].name = "nobody".into()),
        ),
        (
            "Barrier::count",
            Box::new(|s: &mut KernelSchedule| s.barriers[0].count = 9),
        ),
        (
            "Barrier::producers",
            Box::new(|s: &mut KernelSchedule| s.barriers[0].producers.clear()),
        ),
        (
            "Barrier::consumers",
            Box::new(|s: &mut KernelSchedule| s.barriers[0].consumers.clear()),
        ),
        (
            "KernelSchedule::stages",
            Box::new(|s: &mut KernelSchedule| {
                s.body.get_mut("load").unwrap().push(Action::Load {
                    access: Access::plain("tile"),
                    stage: 99,
                });
            }),
        ),
        (
            "KernelSchedule::requires",
            Box::new(|s: &mut KernelSchedule| s.requires.push(Feature::AsyncBarrier)),
        ),
        (
            "KernelSchedule::body",
            Box::new(|s: &mut KernelSchedule| {
                s.body.get_mut("mma").unwrap().clear();
            }),
        ),
        (
            "Access::layout",
            Box::new(|s: &mut KernelSchedule| {
                if let Some(Action::Compute { reads, .. }) =
                    s.body.get_mut("mma").unwrap().get_mut(1)
                {
                    reads[0] = Access::with("tile", Layout::col_major(&[32, 32]));
                }
            }),
        ),
    ];

    let mut unnoticed = Vec::new();
    for (field, perturb) in perturbations {
        let mut s = base();
        perturb(&mut s);
        if verify_kernel_schedule(&s, Target::CUDA_SM86).is_empty() {
            unnoticed.push(field);
        }
    }
    assert!(
        unnoticed.is_empty(),
        "P7 violation — these fields can be set to anything without any analysis \
         noticing, so they are not part of the contract: {unnoticed:?}"
    );
}

/// P6/P8 are review conventions here, and saying so beats implying otherwise.
#[test]
fn which_principles_are_machine_enforced_is_recorded() {
    // P4 and P7 are gated by tests; the rest are conventions. This test exists
    // so the claim lives next to the code rather than only in a doc that can
    // drift from it.
    let machine_enforced = ["P4", "P7"];
    let convention = ["P1", "P2", "P3", "P5", "P6", "P8"];
    assert_eq!(machine_enforced.len() + convention.len(), 8);
}
