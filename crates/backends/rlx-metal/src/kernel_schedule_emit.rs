// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The schedule generates the kernel** — `KernelSchedule` -> MSL.
//!
//! The CUDA twin of this module is `rlx_gpu_kernels::kernel_schedule_emit`.
//! Same argument, different machine: `rlx_ir::kernel_schedule` is a
//! backend-neutral schedule IR, so if it is worth anything it must be able to
//! generate more than one target's text.
//!
//! The kernel here is [`crate::kernels`]'s `sgemm_tiled` — Metal's structural
//! analogue of `kernels/matmul.cu`: a `TILE x TILE` threadgroup tile, one
//! scalar accumulator per thread, and **two** `threadgroup_barrier` calls per K
//! iteration. Those two barriers exist for the same reason CUDA's two
//! `__syncthreads()` do, and they are removable for the same reason: they only
//! protect a single pair of `threadgroup` buffers being reused across
//! iterations.
//!
//! # What the declaration buys on Apple GPUs
//!
//! Metal has no `cp.async`, so the Ampere half of the CUDA experiment does not
//! transfer. The *structural* half does. Declaring `Region::stages = N` gives
//! the emitter N rotating buffers, and the write for iteration `t + N - 1` then
//! lands in the buffer iteration `t - 1` finished reading before the barrier at
//! the top of iteration `t`. The second barrier has nothing left to protect and
//! is deleted rather than skipped:
//!
//! | schedule | `threadgroup` buffers | barriers / K iteration |
//! |---|---|---|
//! | [`sgemm_tiled_schedule`] | 1 pair | 2 |
//! | [`sgemm_tiled_pipelined_schedule`] | N pairs | 1 |
//!
//! This is a *better-posed* experiment than the CUDA one in one respect. The
//! CUDA pipeline needs 16-byte-aligned `cp.async`, so it only runs on full
//! blocks and four of that sweep's twelve shapes cannot reach it at all. Here
//! the staging keeps `sgemm_tiled`'s bounds-checked loads, so the pipelined
//! path runs at **every** shape and the treatment always applies.
//!
//! # Scope, stated
//!
//! Per-thread index algebra is the emitter's knowledge, not the schedule's —
//! `rlx_ir::kernel_schedule` deliberately does not model it. What the schedule
//! contributes is the machine structure: buffer extents, staging depth, barrier
//! placement, threads per group, and the capabilities the target must have.
//!
//! Also: `sgemm_tiled` is **not** Metal's default GEMM. `cost::pick_sgemm`
//! prefers MPS and the `simdgroup_matrix` variants, and reaches the scalar
//! tiled kernel only when those are ineligible. A win measured here is a win on
//! the scalar-tiled path, not on rlx's fastest Metal GEMM, and the A/B example
//! says so in its own output.

use rlx_ir::DType;
use rlx_ir::kernel_schedule::{
    Access, Action, Barrier, Feature, KernelSchedule, KernelScheduleError, Layout, Region, Role,
    Space, Target, lower,
};

use crate::kernel_schedule_port::BLOCK_ROLE;

/// The threadgroup tile edge `sgemm_tiled` ships with (`constant uint TILE = 16`).
pub const SHIPPING_TILE: usize = 16;

/// Entry point name of the emitted kernel.
///
/// Deliberately not `sgemm_tiled`: the baseline arm compiles the real library,
/// and two kernels of the same name in one process is the kind of ambiguity
/// that makes an A/B measure the wrong thing.
pub const EMITTED_ENTRY: &str = "sgemm_sched";

/// The shipping MSL library, so the baseline arm is the real kernel rather than
/// a copy of it.
///
/// `crate::kernels::msl_source()` is `pub(crate)`; re-exporting it here is what
/// lets the A/B example outside this crate compare against what actually ships.
pub fn shipping_msl() -> String {
    crate::kernels::msl_source()
}

/// Why a schedule could not be emitted as MSL.
#[derive(Debug, Clone, PartialEq)]
pub enum EmitError {
    /// The schedule does not verify against the target.
    Unsound(Vec<KernelScheduleError>),
    /// A region the GEMM family requires is missing.
    MissingRegion { name: &'static str },
    /// A declared region's extents disagree with the tile edge.
    RegionTileMismatch {
        region: &'static str,
        declared: Vec<usize>,
        from_tile: Vec<usize>,
    },
    /// The roles imply a different thread count than `TILE x TILE`.
    ThreadCountMismatch { from_roles: usize, from_tile: usize },
    /// More than one role. Warp specialization is expressible in the IR and is
    /// not implemented here; rejecting beats silently flattening it.
    MultiRole { roles: usize },
    /// `stages` is inconsistent across the staged regions and the schedule.
    StageDisagreement {
        schedule: usize,
        region: &'static str,
        declared: usize,
    },
    /// Apple GPUs have no `cp.async`. A schedule that declares it is asking for
    /// a machine this backend is not.
    UnsupportedOnMetal { feature: &'static str },
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsound(errors) => {
                let joined = errors
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; ");
                write!(f, "schedule does not verify: {joined}")
            }
            Self::MissingRegion { name } => {
                write!(f, "GEMM emitter requires a region named `{name}`")
            }
            Self::RegionTileMismatch {
                region,
                declared,
                from_tile,
            } => write!(
                f,
                "region `{region}` declares {declared:?} but the tile implies {from_tile:?}"
            ),
            Self::ThreadCountMismatch {
                from_roles,
                from_tile,
            } => write!(
                f,
                "roles imply {from_roles} threads, the tile implies {from_tile}"
            ),
            Self::MultiRole { roles } => write!(
                f,
                "{roles} roles declared; this emitter lowers single-role (uniform \
                 threadgroup) schedules only"
            ),
            Self::StageDisagreement {
                schedule,
                region,
                declared,
            } => write!(
                f,
                "schedule declares {schedule} stages but region `{region}` declares {declared}"
            ),
            Self::UnsupportedOnMetal { feature } => write!(
                f,
                "`{feature}` has no Metal equivalent — refused rather than lowered to \
                 something else"
            ),
        }
    }
}

impl std::error::Error for EmitError {}

fn gemm_regions(tile: usize, stages: usize) -> Vec<Region> {
    vec![
        Region {
            name: "Asub".into(),
            space: Space::Shared,
            dims: vec![tile, tile],
            dtype: DType::F32,
            stages,
            layout: Layout::row_major(&[tile, tile]),
        },
        Region {
            name: "Bsub".into(),
            space: Space::Shared,
            dims: vec![tile, tile],
            dtype: DType::F32,
            stages,
            layout: Layout::row_major(&[tile, tile]),
        },
        Region {
            name: "sum".into(),
            space: Space::Register,
            dims: vec![1, 1],
            dtype: DType::F32,
            stages: 1,
            layout: Layout::row_major(&[1, 1]),
        },
    ]
}

fn gemm_role(tile: usize) -> Vec<Role> {
    // `TILE x TILE` threads, 32 lanes per simdgroup.
    let simdgroups = (tile * tile).div_ceil(32) as u32;
    vec![Role {
        name: BLOCK_ROLE.into(),
        warps: (0..simdgroups).collect(),
    }]
}

/// `sgemm_tiled` as a typed schedule: one buffer pair, two barriers.
///
/// The mirror of `rlx_gpu_kernels::kernel_schedule_port::matmul_schedule`, and
/// the arm that makes the experiment attributable — emitting *this* should
/// reproduce the shipping kernel, so any delta the pipelined arm shows is the
/// pipeline rather than the code generator.
pub fn sgemm_tiled_schedule(tile: usize) -> KernelSchedule {
    let mut s = KernelSchedule::new(format!("sgemm_tiled_{tile}"));
    s.stages = 1;
    s.requires = vec![];
    s.regions = gemm_regions(tile, 1);
    s.roles = gemm_role(tile);
    s.barriers = vec![
        Barrier {
            name: "tiles_filled".into(),
            producers: vec![BLOCK_ROLE.into()],
            consumers: vec![BLOCK_ROLE.into()],
            count: 1,
        },
        Barrier {
            name: "tiles_consumed".into(),
            producers: vec![BLOCK_ROLE.into()],
            consumers: vec![BLOCK_ROLE.into()],
            count: 1,
        },
    ];
    s.body.insert(
        BLOCK_ROLE.into(),
        vec![
            Action::Load {
                access: Access::plain("Asub"),
                stage: 0,
            },
            Action::Load {
                access: Access::plain("Bsub"),
                stage: 0,
            },
            Action::Arrive {
                barrier: "tiles_filled".into(),
                stage: 0,
            },
            Action::Wait {
                barrier: "tiles_filled".into(),
                stage: 0,
            },
            Action::Compute {
                reads: vec![Access::plain("Asub"), Access::plain("Bsub")],
                writes: vec![Access::plain("sum")],
                stage: 0,
                via: None,
            },
            Action::Arrive {
                barrier: "tiles_consumed".into(),
                stage: 0,
            },
            Action::Wait {
                barrier: "tiles_consumed".into(),
                stage: 0,
            },
            Action::Store {
                access: Access::plain("sum"),
                stage: 0,
            },
        ],
    );
    s
}

/// The same GEMM with `stages` rotating buffer pairs and **one** barrier.
///
/// Two declarations differ from [`sgemm_tiled_schedule`]: `Region::stages` and
/// the absence of `tiles_consumed`. The second is only sound because of the
/// first, and the verifier is what checks that — before this schedule could be
/// expressed at all, `rlx_ir`'s write-after-read rule keyed on the region name
/// and reported every rotating buffer as a race.
pub fn sgemm_tiled_pipelined_schedule(tile: usize, stages: usize) -> KernelSchedule {
    let mut s = KernelSchedule::new(format!("sgemm_pipe{stages}_{tile}"));
    s.stages = stages;
    // No AsyncCopy: Apple GPUs have no cp.async. The win here is the barrier
    // and the earlier issue of the next tile's loads, not an async engine.
    s.requires = vec![];
    s.regions = gemm_regions(tile, stages);
    s.roles = gemm_role(tile);
    s.barriers = vec![Barrier {
        name: "tiles_filled".into(),
        producers: vec![BLOCK_ROLE.into()],
        consumers: vec![BLOCK_ROLE.into()],
        count: 1,
    }];
    let next = stages - 1;
    s.body.insert(
        BLOCK_ROLE.into(),
        vec![
            Action::Wait {
                barrier: "tiles_filled".into(),
                stage: 0,
            },
            Action::Compute {
                reads: vec![Access::plain("Asub"), Access::plain("Bsub")],
                writes: vec![Access::plain("sum")],
                stage: 0,
                via: None,
            },
            Action::Load {
                access: Access::plain("Asub"),
                stage: next,
            },
            Action::Load {
                access: Access::plain("Bsub"),
                stage: next,
            },
            Action::Arrive {
                barrier: "tiles_filled".into(),
                stage: next,
            },
            Action::Store {
                access: Access::plain("sum"),
                stage: 0,
            },
        ],
    );
    s
}

/// What the emitter read out of the schedule, so a caller can assert on it
/// rather than regexing generated MSL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitFacts {
    pub stages: usize,
    pub barriers_per_k_iter: usize,
    pub threadgroup_bytes: usize,
    pub threads: usize,
}

/// Build the schedule these parameters describe.
///
/// The parameters are the single source of truth; the schedule is derived from
/// them. Writing a stage depth in two places is how the Rust-side and
/// shader-side views of a kernel drift apart.
pub fn schedule_for(p: &crate::apple_params::AppleKernelParams) -> KernelSchedule {
    let mut s = if p.stages_value() > 1 {
        sgemm_tiled_pipelined_schedule(p.tile_value(), p.stages_value())
    } else {
        sgemm_tiled_schedule(p.tile_value())
    };
    // The staged regions must carry the parameter's precision, or `lower()`
    // prices the schedule in f32 while the kernel stages f16 — and
    // `EmitFacts.threadgroup_bytes` then disagrees with
    // `AppleKernelParams::threadgroup_bytes` about the same quantity. Two
    // statements of one fact drifting apart is exactly what this emitter exists
    // to prevent; it showed up as 2048 vs 1024 in an A/B header.
    if p.precision_value() == crate::apple_params::Precision::F16Storage {
        for r in s.regions.iter_mut() {
            if r.space == Space::Shared {
                r.dtype = DType::F16;
            }
        }
    }
    s
}

/// Emit MSL for a parameter set. **The entry point to prefer.**
///
/// Everything the kernel does is decided here: tile edge, rotation depth,
/// barrier scope and staged precision all come from `p`, so a configuration is
/// one value that can be logged, pasted back, and predicted before it runs.
pub fn emit_for(
    p: &crate::apple_params::AppleKernelParams,
    target: Target,
) -> Result<(String, EmitFacts), EmitError> {
    emit_msl_with(&schedule_for(p), p.tile_value(), target, Some(p))
}

/// Emit MSL for `sched` at `tile`, checked against `target`.
pub fn emit_msl(
    sched: &KernelSchedule,
    tile: usize,
    target: Target,
) -> Result<(String, EmitFacts), EmitError> {
    emit_msl_with(sched, tile, target, None)
}

/// [`emit_msl`] with an optional parameter set controlling barrier scope and
/// staged precision. `None` means the defaults (threadgroup barrier, f32).
pub fn emit_msl_with(
    sched: &KernelSchedule,
    tile: usize,
    target: Target,
    params: Option<&crate::apple_params::AppleKernelParams>,
) -> Result<(String, EmitFacts), EmitError> {
    if sched.roles.len() != 1 {
        return Err(EmitError::MultiRole {
            roles: sched.roles.len(),
        });
    }
    if sched.requires.contains(&Feature::AsyncCopy) {
        return Err(EmitError::UnsupportedOnMetal {
            feature: "AsyncCopy",
        });
    }
    let lowered = lower(sched, target).map_err(EmitError::Unsound)?;
    if lowered.threads != tile * tile {
        return Err(EmitError::ThreadCountMismatch {
            from_roles: lowered.threads,
            from_tile: tile * tile,
        });
    }

    let region = |name: &'static str| -> Result<&Region, EmitError> {
        sched
            .regions
            .iter()
            .find(|r| r.name == name)
            .ok_or(EmitError::MissingRegion { name })
    };
    let a = region("Asub")?;
    let b = region("Bsub")?;
    region("sum")?;
    for (r, name) in [(a, "Asub"), (b, "Bsub")] {
        if r.dims != [tile, tile] {
            return Err(EmitError::RegionTileMismatch {
                region: name,
                declared: r.dims.clone(),
                from_tile: vec![tile, tile],
            });
        }
    }

    let stages = sched.stages.max(1);
    for (r, name) in [(a, "Asub"), (b, "Bsub")] {
        if r.stages.max(1) != stages {
            return Err(EmitError::StageDisagreement {
                schedule: stages,
                region: name,
                declared: r.stages,
            });
        }
    }

    let body = sched.body.get(BLOCK_ROLE).map(Vec::as_slice).unwrap_or(&[]);
    // Not a constant of the emitter: however many `Wait`s the role declares.
    let barriers_per_k_iter = body
        .iter()
        .filter(|x| matches!(x, Action::Wait { .. }))
        .count()
        .max(1);

    let facts = EmitFacts {
        stages,
        barriers_per_k_iter,
        threadgroup_bytes: lowered.shared_bytes,
        threads: lowered.threads,
    };

    // The two tokens `AppleKernelParams` controls in the body. Derived once so
    // there is a single place each is written down.
    use crate::apple_params::{Precision, SyncScope};
    let stage_ty = match params.map(|p| p.precision_value()) {
        Some(Precision::F16Storage) => "half",
        _ => "float",
    };
    let barrier = match params.map(|p| p.sync_value()) {
        // Only orders 32 threads. Correct exactly when the handoff does not
        // leave a simdgroup — the caller declares that; the emitter cannot
        // check it, and says so rather than pretending to.
        Some(SyncScope::Simdgroup) => "simdgroup_barrier(mem_flags::mem_threadgroup);",
        _ => "threadgroup_barrier(mem_flags::mem_threadgroup);",
    };

    let mut src = String::with_capacity(4096);
    src.push_str(&format!(
        "// @generated by rlx_metal::kernel_schedule_emit from KernelSchedule `{}`.\n\
         // Do not edit; edit the schedule.\n\
         //\n\
         // Derived, not authored:\n\
         //   stages              = {}   (KernelSchedule::stages / Region::stages)\n\
         //   barriers per K iter = {}   (count of Action::Wait in the role body)\n\
         //   threadgroup bytes   = {}   (lower().shared_bytes)\n\
         //   threads/group       = {}   (lower().threads, from Role::warps)\n\
         //   buffer offsets      = {:?}\n\
         #include <metal_stdlib>\n\
         using namespace metal;\n\n\
         constant uint TILE = {tile};\n\
         constant uint STAGES = {};\n\n",
        sched.name,
        facts.stages,
        facts.barriers_per_k_iter,
        facts.threadgroup_bytes,
        facts.threads,
        lowered.region_offsets,
        stages,
    ));

    // Signature is `sgemm_tiled`'s, so the emitted kernel is a drop-in and the
    // A/B varies one thing.
    src.push_str(&format!(
        "kernel void {EMITTED_ENTRY}(\n\
         \x20   device const float* A [[buffer(0)]],\n\
         \x20   device const float* B [[buffer(1)]],\n\
         \x20   device float* C       [[buffer(2)]],\n\
         \x20   constant uint& M      [[buffer(3)]],\n\
         \x20   constant uint& K      [[buffer(4)]],\n\
         \x20   constant uint& N      [[buffer(5)]],\n\
         \x20   uint2 gid  [[thread_position_in_grid]],\n\
         \x20   uint2 tid  [[thread_position_in_threadgroup]],\n\
         \x20   uint2 tgid [[threadgroup_position_in_grid]]\n\
         ) {{\n"
    ));

    // `[STAGES]` is the only difference from the shipping declaration, and it
    // comes from `Region::stages`.
    src.push_str(&format!(
        "    threadgroup {stage_ty} Asub[STAGES][TILE][TILE];\n\
         \x20   threadgroup {stage_ty} Bsub[STAGES][TILE][TILE];\n\n\
         \x20   uint row = tgid.y * TILE + tid.y;\n\
         \x20   uint col = tgid.x * TILE + tid.x;\n\n\
         \x20   float sum = 0.0;\n\
         \x20   uint num_tiles = (K + TILE - 1) / TILE;\n\n"
    ));

    // The staging load, bounds-checked exactly as `sgemm_tiled` does it. Shared
    // between prologue and steady state so the two cannot drift apart.
    let stage_load = |t_expr: &str, buf_expr: &str| -> String {
        format!(
            "        {{\n\
             \x20           uint a_col = ({t_expr}) * TILE + tid.x;\n\
             \x20           uint b_row = ({t_expr}) * TILE + tid.y;\n\
             \x20           Asub[{buf_expr}][tid.y][tid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0;\n\
             \x20           Bsub[{buf_expr}][tid.y][tid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0;\n\
             \x20       }}\n"
        )
    };
    let compute = |buf_expr: &str| -> String {
        format!(
            "        for (uint k = 0; k < TILE; ++k) {{\n\
             \x20           sum += Asub[{buf_expr}][tid.y][k] * Bsub[{buf_expr}][k][tid.x];\n\
             \x20       }}\n"
        )
    };

    if stages > 1 {
        src.push_str("    // Prologue: fill STAGES-1 buffers before the first compute.\n");
        src.push_str("    for (uint s = 0; s + 1 < STAGES; ++s) {\n");
        src.push_str("        if (s < num_tiles) {\n");
        src.push_str(&stage_load("s", "s"));
        src.push_str("        }\n    }\n\n");
        src.push_str("    for (uint t = 0; t < num_tiles; ++t) {\n");
        src.push_str(&format!(
            "        {barrier}  // Action::Wait `tiles_filled` \
             ({barriers_per_k_iter}/iter declared)\n\n"
        ));
        src.push_str(&compute("t % STAGES"));
        src.push_str(
            "\n        // Stage the tile needed STAGES-1 iterations from now, into the\n\
             \x20       // buffer iteration t-1 finished reading before the barrier above.\n\
             \x20       // That ordering is why no second barrier is needed here.\n\
             \x20       uint kt = t + STAGES - 1;\n\
             \x20       if (kt < num_tiles) {\n",
        );
        src.push_str(&stage_load("kt", "kt % STAGES"));
        src.push_str("        }\n    }\n");
    } else {
        src.push_str("    for (uint t = 0; t < num_tiles; ++t) {\n");
        src.push_str(&stage_load("t", "0"));
        src.push_str(&format!(
            "        {barrier}  // Action::Wait `tiles_filled`\n\n"
        ));
        src.push_str(&compute("0"));
        src.push_str(&format!(
            "\n        {barrier}  // Action::Wait `tiles_consumed`\n    }}\n"
        ));
    }

    src.push_str(
        "\n    if (row < M && col < N) {\n\
         \x20       C[row * N + col] = sum;\n\
         \x20   }\n}\n",
    );

    Ok((src, facts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_schedule_port::METAL_TARGET;

    #[test]
    fn the_shipping_schedule_emits_one_stage_and_two_barriers() {
        let (src, facts) = emit_msl(
            &sgemm_tiled_schedule(SHIPPING_TILE),
            SHIPPING_TILE,
            METAL_TARGET,
        )
        .unwrap();
        assert_eq!(facts.stages, 1);
        assert_eq!(facts.barriers_per_k_iter, 2);
        assert_eq!(facts.threads, 256);
        // 16*16*4 * 2 buffers = 2048.
        assert_eq!(facts.threadgroup_bytes, 2048);
        assert_eq!(src.matches("threadgroup_barrier").count(), 2);
        assert!(src.contains("constant uint STAGES = 1;"));
    }

    #[test]
    fn the_pipelined_schedule_emits_rotation_and_one_barrier() {
        let sched = sgemm_tiled_pipelined_schedule(SHIPPING_TILE, 2);
        let (src, facts) = emit_msl(&sched, SHIPPING_TILE, METAL_TARGET).unwrap();
        assert_eq!(facts.stages, 2);
        assert_eq!(facts.barriers_per_k_iter, 1);
        assert_eq!(facts.threadgroup_bytes, 4096);
        assert_eq!(src.matches("threadgroup_barrier").count(), 1);
        assert!(src.contains("t % STAGES"));
        assert!(src.contains("kt % STAGES"));
    }

    /// The declared depth must reach the generated text, or the pipelined arm
    /// would silently measure the same structure and read as a free win.
    #[test]
    fn stage_depth_reaches_the_emitted_source() {
        for stages in 2..=4 {
            let sched = sgemm_tiled_pipelined_schedule(SHIPPING_TILE, stages);
            let (src, facts) = emit_msl(&sched, SHIPPING_TILE, METAL_TARGET).unwrap();
            assert!(src.contains(&format!("constant uint STAGES = {stages};")));
            assert_eq!(facts.threadgroup_bytes, 2048 * stages);
        }
    }

    /// A capability Metal does not have is refused, not lowered to something
    /// else. `Feature::AsyncCopy` is the CUDA schedule's, and reusing it here
    /// by accident must not silently produce a serial kernel.
    #[test]
    fn async_copy_is_refused_on_metal() {
        let mut sched = sgemm_tiled_pipelined_schedule(SHIPPING_TILE, 2);
        sched.requires = vec![Feature::AsyncCopy];
        assert!(matches!(
            emit_msl(&sched, SHIPPING_TILE, METAL_TARGET),
            Err(EmitError::UnsupportedOnMetal { .. })
        ));
    }

    /// Threadgroup memory is budgeted, and a rotation multiplies it. Apple's
    /// floor is 32 KiB; a depth that busts it is a launch failure, so it is
    /// caught before any MSL exists.
    #[test]
    fn an_over_deep_rotation_busts_the_threadgroup_budget() {
        // 64x64 f32 tiles are 16 KiB each; 2 of them x 2 stages = 64 KiB.
        let sched = sgemm_tiled_pipelined_schedule(64, 2);
        let err = emit_msl(&sched, 64, METAL_TARGET).unwrap_err();
        assert!(
            matches!(&err, EmitError::Unsound(es)
                if es.iter().any(|e| matches!(
                    e, KernelScheduleError::SharedOverBudget { .. }))),
            "expected a shared-budget finding, got {err}"
        );
    }

    /// `AppleKernelParams` must reach the emitted MSL, or the knobs are
    /// decoration. This is the end-to-end version of `apple_params`'s
    /// `every_parameter_reaches_the_msl`, which only checks `defines()`.
    #[test]
    fn params_drive_the_emitted_kernel_end_to_end() {
        use crate::apple_params::{AppleKernelParams, Precision, SyncScope};

        let p = AppleKernelParams::default()
            .stages(2)
            .precision(Precision::F16Storage)
            .sync(SyncScope::Simdgroup);
        let (src, facts) = emit_for(&p, METAL_TARGET).expect("params emit");

        assert_eq!(facts.stages, 2, "stage depth did not reach the schedule");
        assert!(
            src.contains("threadgroup half Asub"),
            "precision did not reach the MSL"
        );
        assert!(
            src.contains("simdgroup_barrier"),
            "sync scope did not reach the MSL"
        );
        assert!(
            !src.contains("threadgroup_barrier(mem"),
            "the old barrier survived"
        );
        // THE invariant, not a magic number: the schedule's byte accounting and
        // the parameters' must agree. They did not before `schedule_for`
        // propagated the precision — the facts said 4096 (priced in f32) while
        // the params said 2048 (f16), and both numbers appeared in the same A/B
        // header.
        assert_eq!(
            facts.threadgroup_bytes,
            p.threadgroup_bytes(),
            "EmitFacts and AppleKernelParams disagree about threadgroup bytes"
        );
        // 2 tiles x 16x16 x 2 B (f16) x 2 stages.
        assert_eq!(
            facts.threadgroup_bytes, 2048,
            "2 stages x 16x16 x f16 accounting"
        );
    }

    /// The default parameter set must reproduce the shipping structure exactly
    /// — one stage, two threadgroup barriers, f32. If the defaults drifted,
    /// every A/B baseline would silently move with them.
    #[test]
    fn default_params_reproduce_the_shipping_structure() {
        use crate::apple_params::AppleKernelParams;
        let (src, facts) =
            emit_for(&AppleKernelParams::default(), METAL_TARGET).expect("default emits");
        assert_eq!(facts.stages, 1);
        assert_eq!(facts.barriers_per_k_iter, 2);
        assert!(src.contains("threadgroup float Asub"));
        assert_eq!(src.matches("threadgroup_barrier").count(), 2);
    }

    /// An unsound schedule never reaches the Metal compiler.
    #[test]
    fn an_unverifiable_schedule_is_rejected_before_emission() {
        let mut sched = sgemm_tiled_schedule(SHIPPING_TILE);
        sched.barriers[0].producers.clear();
        assert!(matches!(
            emit_msl(&sched, SHIPPING_TILE, METAL_TARGET),
            Err(EmitError::Unsound(_))
        ));
    }

    /// The emitted serial kernel must carry the same bounds-checked staging and
    /// write-back as `sgemm_tiled`, or the baseline comparison is against a
    /// different algorithm. Read the shipping text rather than trusting a copy.
    #[test]
    fn the_emitted_staging_does_not_drift_from_sgemm_tiled() {
        let (src, _) = emit_msl(
            &sgemm_tiled_schedule(SHIPPING_TILE),
            SHIPPING_TILE,
            METAL_TARGET,
        )
        .unwrap();
        let shipping = shipping_msl();
        for fragment in [
            "(row < M && a_col < K) ? A[row * K + a_col] : 0.0",
            "(b_row < K && col < N) ? B[b_row * N + col] : 0.0",
            "C[row * N + col] = sum;",
        ] {
            assert!(
                shipping.contains(fragment),
                "sgemm_tiled no longer contains `{fragment}` — update the emitter"
            );
            assert!(
                src.contains(fragment),
                "the emitted kernel is missing `{fragment}`"
            );
        }
    }
}
