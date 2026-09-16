// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The schedule generates the kernel** — `KernelSchedule` -> OpenCL C.
//!
//! Fifth target for the same schedule IR, after CUDA/HIP
//! (`rlx_gpu_kernels::kernel_schedule_emit`), MSL (`rlx_metal`), WGSL
//! (`rlx_wgpu`) and GLSL (`rlx_vulkan`).
//!
//! # Two limitations, stated before anything else
//!
//! This port is **not** in the same evidential state as the other four, and
//! saying so is the point of putting it first.
//!
//! **1. There is no shipping baseline to A/B against.** `kernels/matmul.cl` is
//! one work-item per output element with *no* `__local` memory and *no*
//! `barrier()` — its own comment calls it "Naive (no tiling/SLM) — the
//! correctness baseline". The transformation these emitters exist to test
//! (rotate the staged tiles, drop one barrier per K iteration) has nothing to
//! remove there. So the comparison this module supports is
//! `serial-tiled` vs `pipelined-tiled`, both emitted, with the shipping naive
//! kernel as a *separate* reference point. A "tiled beats naive" result would
//! be true and uninteresting; it is not the claim under test.
//!
//! **2. It is unmeasured.** Compilation needs Intel's `ocloc` (see
//! `build.rs`), which is absent on this project's macOS box, its NVIDIA rig and
//! its AMD rig alike — and so is any Intel GPU to run it on. The emitted text
//! is exercised by the tests below; nothing here has executed on hardware.
//! [`compile_spirv_ocloc`] returns a named error rather than a fallback,
//! matching how the Vulkan port treats a missing glslang.
//!
//! That second point is a coverage limitation, not a pass. rlx-vulkan's port
//! showed exactly why the distinction matters: an emitted kernel that compiled
//! cleanly, validated cleanly, and asserted the right SPIR-V barrier operand
//! was still numerically wrong on real hardware at large K.
//!
//! # What is derived
//!
//! | emitted | derived from |
//! |---|---|
//! | `__local` array extents | [`Region`] in [`Space::Shared`] |
//! | leading `[STAGES]` rank | `Region::stages` |
//! | local-memory budget, checked at emit | `lower().shared_bytes` |
//! | `reqd_work_group_size` | `Role::warps` via `lower().threads` |
//! | `barrier(CLK_LOCAL_MEM_FENCE)` placement | position of `Action::Wait` |
//! | K-loop rotation modulus | `KernelSchedule::stages` |

use rlx_ir::DType;
use rlx_ir::kernel_schedule::{
    Access, Action, Barrier, Feature, KernelSchedule, KernelScheduleError, Layout, Region, Role,
    Space, Target, lower,
};

/// The single role: the whole work-group.
pub const BLOCK_ROLE: &str = "workgroup";

/// Tile edge the emitted kernels use.
///
/// Not read off a shipping kernel, because the shipping kernel has no tile —
/// see the module docs. 16 matches the other backends' tiled GEMMs so the
/// emitted OpenCL is structurally comparable to them.
pub const DEFAULT_TS: usize = 16;

/// Entry point of the emitted kernel. Distinct from `matmul` so the naive
/// shipping kernel and a generated one can coexist in a process.
pub const EMITTED_ENTRY: &str = "matmul_sched";

/// Why a schedule could not be emitted as OpenCL C.
#[derive(Debug, Clone, PartialEq)]
pub enum EmitError {
    Unsound(Vec<KernelScheduleError>),
    MissingRegion {
        name: &'static str,
    },
    RegionTileMismatch {
        region: &'static str,
        declared: Vec<usize>,
        from_tile: Vec<usize>,
    },
    ThreadCountMismatch {
        from_roles: usize,
        from_tile: usize,
    },
    MultiRole {
        roles: usize,
    },
    StageDisagreement {
        schedule: usize,
        region: &'static str,
        declared: usize,
    },
    /// A capability this emitter has no OpenCL-C lowering for.
    UnsupportedInOpenCl {
        feature: &'static str,
    },
    /// `ocloc` missing, or it rejected the emitted source.
    Compile(String),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsound(errors) => write!(
                f,
                "schedule does not verify: {}",
                errors
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
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
                "roles imply {from_roles} work-items, the tile implies {from_tile}"
            ),
            Self::MultiRole { roles } => write!(
                f,
                "{roles} roles declared; this emitter lowers single-role work-groups only"
            ),
            Self::StageDisagreement {
                schedule,
                region,
                declared,
            } => write!(
                f,
                "schedule declares {schedule} stages but region `{region}` declares {declared}"
            ),
            Self::UnsupportedInOpenCl { feature } => write!(
                f,
                "`{feature}` has no OpenCL-C lowering here — refused rather than emitted \
                 as another dialect's instruction"
            ),
            Self::Compile(why) => write!(f, "emitted OpenCL C did not compile: {why}"),
        }
    }
}

impl std::error::Error for EmitError {}

fn gemm_regions(ts: usize, stages: usize) -> Vec<Region> {
    vec![
        Region {
            name: "As".into(),
            space: Space::Shared,
            dims: vec![ts, ts],
            dtype: DType::F32,
            stages,
            layout: Layout::row_major(&[ts, ts]),
        },
        Region {
            name: "Bs".into(),
            space: Space::Shared,
            dims: vec![ts, ts],
            dtype: DType::F32,
            stages,
            layout: Layout::row_major(&[ts, ts]),
        },
        Region {
            name: "acc".into(),
            space: Space::Register,
            dims: vec![1, 1],
            dtype: DType::F32,
            stages: 1,
            layout: Layout::row_major(&[1, 1]),
        },
    ]
}

fn gemm_role(ts: usize) -> Vec<Role> {
    vec![Role {
        name: BLOCK_ROLE.into(),
        warps: (0..((ts * ts).div_ceil(32) as u32).max(1)).collect(),
    }]
}

/// A tiled GEMM with one buffer pair and two barriers.
///
/// The *control* schedule: it is what a hand-written tiled OpenCL kernel would
/// be, so a `pipelined` delta measured against it is the rotation rather than
/// the difference between tiled and naive.
pub fn matmul_tiled_schedule(ts: usize) -> KernelSchedule {
    let mut s = KernelSchedule::new(format!("ocl_matmul_tiled_{ts}"));
    s.stages = 1;
    s.regions = gemm_regions(ts, 1);
    s.roles = gemm_role(ts);
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
                access: Access::plain("As"),
                stage: 0,
            },
            Action::Load {
                access: Access::plain("Bs"),
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
                reads: vec![Access::plain("As"), Access::plain("Bs")],
                writes: vec![Access::plain("acc")],
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
                access: Access::plain("acc"),
                stage: 0,
            },
        ],
    );
    s
}

/// The same GEMM with `stages` rotating tile pairs and **one** barrier.
pub fn matmul_pipelined_schedule(ts: usize, stages: usize) -> KernelSchedule {
    let mut s = KernelSchedule::new(format!("ocl_matmul_pipe{stages}_{ts}"));
    s.stages = stages;
    s.regions = gemm_regions(ts, stages);
    s.roles = gemm_role(ts);
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
                reads: vec![Access::plain("As"), Access::plain("Bs")],
                writes: vec![Access::plain("acc")],
                stage: 0,
                via: None,
            },
            Action::Load {
                access: Access::plain("As"),
                stage: next,
            },
            Action::Load {
                access: Access::plain("Bs"),
                stage: next,
            },
            Action::Arrive {
                barrier: "tiles_filled".into(),
                stage: next,
            },
            Action::Store {
                access: Access::plain("acc"),
                stage: 0,
            },
        ],
    );
    s
}

/// What the emitter read out of the schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitFacts {
    pub stages: usize,
    pub barriers_per_k_iter: usize,
    pub local_bytes: usize,
    pub work_items: usize,
}

/// Emit OpenCL C for `sched` at tile edge `ts`, checked against `target`.
///
/// The signature matches `kernels/matmul.cl`'s arena convention (one `__global
/// float*` plus scalar offsets), so an emitted kernel is argument-compatible
/// with the existing launcher.
pub fn emit_opencl(
    sched: &KernelSchedule,
    ts: usize,
    target: Target,
) -> Result<(String, EmitFacts), EmitError> {
    if sched.roles.len() != 1 {
        return Err(EmitError::MultiRole {
            roles: sched.roles.len(),
        });
    }
    if sched.requires.contains(&Feature::AsyncCopy) {
        // OpenCL has `async_work_group_copy`, but nothing here emits it, and a
        // capability list must describe what rlx can drive.
        return Err(EmitError::UnsupportedInOpenCl {
            feature: "AsyncCopy",
        });
    }
    let lowered = lower(sched, target).map_err(EmitError::Unsound)?;
    if lowered.threads != ts * ts {
        return Err(EmitError::ThreadCountMismatch {
            from_roles: lowered.threads,
            from_tile: ts * ts,
        });
    }

    let region = |name: &'static str| -> Result<&Region, EmitError> {
        sched
            .regions
            .iter()
            .find(|r| r.name == name)
            .ok_or(EmitError::MissingRegion { name })
    };
    let a = region("As")?;
    let b = region("Bs")?;
    region("acc")?;
    for (r, name) in [(a, "As"), (b, "Bs")] {
        if r.dims != [ts, ts] {
            return Err(EmitError::RegionTileMismatch {
                region: name,
                declared: r.dims.clone(),
                from_tile: vec![ts, ts],
            });
        }
    }
    let stages = sched.stages.max(1);
    for (r, name) in [(a, "As"), (b, "Bs")] {
        if r.stages.max(1) != stages {
            return Err(EmitError::StageDisagreement {
                schedule: stages,
                region: name,
                declared: r.stages,
            });
        }
    }

    let body = sched.body.get(BLOCK_ROLE).map(Vec::as_slice).unwrap_or(&[]);
    let barriers_per_k_iter = body
        .iter()
        .filter(|x| matches!(x, Action::Wait { .. }))
        .count()
        .max(1);

    let facts = EmitFacts {
        stages,
        barriers_per_k_iter,
        local_bytes: lowered.shared_bytes,
        work_items: lowered.threads,
    };

    let mut s = String::with_capacity(4096);
    s.push_str(&format!(
        "// @generated by rlx_oneapi::kernel_schedule_emit from KernelSchedule `{}`.\n\
         // Do not edit; edit the schedule.\n\
         //\n\
         // Derived, not authored:\n\
         //   stages              = {}   (KernelSchedule::stages / Region::stages)\n\
         //   barriers per K iter = {}   (count of Action::Wait in the role body)\n\
         //   local bytes         = {}   (lower().shared_bytes)\n\
         //   work-items/group    = {}   (lower().threads, from Role::warps)\n\
         //   buffer offsets      = {:?}\n\
         //\n\
         // UNMEASURED: no Intel device or `ocloc` was available when this was\n\
         // written. The text is tested; the kernel has never run.\n\n",
        sched.name,
        facts.stages,
        facts.barriers_per_k_iter,
        facts.local_bytes,
        facts.work_items,
        lowered.region_offsets,
    ));
    s.push_str(&format!("#define TS {ts}u\n#define STAGES {stages}u\n\n"));
    s.push_str(&format!(
        "__attribute__((reqd_work_group_size({ts}, {ts}, 1)))\n\
         __kernel void {EMITTED_ENTRY}(__global float* arena,\n\
         \x20                    uint M, uint K, uint N,\n\
         \x20                    uint off_a, uint off_b, uint off_out,\n\
         \x20                    uint batch, uint a_bs, uint b_bs) {{\n"
    ));
    s.push_str(&format!(
        "    __local float As[{}];\n    __local float Bs[{}];\n\n",
        stages * ts * ts,
        stages * ts * ts
    ));
    s.push_str(
        "    uint lx = get_local_id(0);\n\
         \x20   uint ly = get_local_id(1);\n\
         \x20   uint col = get_global_id(0);\n\
         \x20   uint row = get_global_id(1);\n\
         \x20   uint bz  = get_global_id(2);\n\n\
         \x20   uint a_base = off_a + bz * a_bs;\n\
         \x20   uint b_base = off_b + bz * b_bs;\n\n\
         \x20   float acc = 0.0f;\n\
         \x20   uint tiles = (K + TS - 1u) / TS;\n\n",
    );

    let stage_load = |t: &str, buf: &str, ind: &str| -> String {
        format!(
            "{ind}{{\n\
             {ind}    uint base = ({buf}) * TS * TS;\n\
             {ind}    uint a_col = ({t}) * TS + lx;\n\
             {ind}    uint b_row = ({t}) * TS + ly;\n\
             {ind}    As[base + ly * TS + lx] = (row < M && a_col < K)\n\
             {ind}        ? arena[a_base + row * K + a_col] : 0.0f;\n\
             {ind}    Bs[base + ly * TS + lx] = (b_row < K && col < N)\n\
             {ind}        ? arena[b_base + b_row * N + col] : 0.0f;\n\
             {ind}}}\n"
        )
    };
    let compute = |buf: &str, ind: &str| -> String {
        format!(
            "{ind}{{\n\
             {ind}    uint base = ({buf}) * TS * TS;\n\
             {ind}    for (uint kk = 0u; kk < TS; kk++) {{\n\
             {ind}        acc += As[base + ly * TS + kk] * Bs[base + kk * TS + lx];\n\
             {ind}    }}\n\
             {ind}}}\n"
        )
    };
    // `CLK_LOCAL_MEM_FENCE` is not decoration. rlx-vulkan's port shipped a
    // control barrier with no explicit shared-memory fence and produced wrong
    // results at K = 4096 on real hardware while passing at small K.
    const BARRIER: &str = "barrier(CLK_LOCAL_MEM_FENCE);";

    if stages > 1 {
        s.push_str("    for (uint sidx = 0u; sidx + 1u < STAGES; sidx++) {\n");
        s.push_str("        if (sidx < tiles) {\n");
        s.push_str(&stage_load("sidx", "sidx", "            "));
        s.push_str("        }\n    }\n\n");
        s.push_str("    for (uint t = 0u; t < tiles; t++) {\n");
        s.push_str(&format!(
            "        {BARRIER}  // Action::Wait `tiles_filled` \
             ({barriers_per_k_iter}/iter declared)\n\n"
        ));
        s.push_str(&compute("t % STAGES", "        "));
        s.push_str(
            "\n        // Stage the tile needed STAGES-1 iterations from now, into the\n\
             \x20       // buffer iteration t-1 finished reading before the barrier above.\n\
             \x20       uint kt = t + STAGES - 1u;\n\
             \x20       if (kt < tiles) {\n",
        );
        s.push_str(&stage_load("kt", "kt % STAGES", "            "));
        s.push_str("        }\n    }\n");
    } else {
        s.push_str("    for (uint t = 0u; t < tiles; t++) {\n");
        s.push_str(&stage_load("t", "0u", "        "));
        s.push_str(&format!(
            "        {BARRIER}  // Action::Wait `tiles_filled`\n\n"
        ));
        s.push_str(&compute("0u", "        "));
        s.push_str(&format!(
            "\n        {BARRIER}  // Action::Wait `tiles_consumed`\n    }}\n"
        ));
    }
    s.push_str(
        "\n    if (row < M && col < N && bz < batch) {\n\
         \x20       arena[off_out + bz * (M * N) + row * N + col] = acc;\n\
         \x20   }\n}\n",
    );
    Ok((s, facts))
}

/// Compile emitted OpenCL C to Kernel-flavor SPIR-V with Intel's `ocloc`.
///
/// Returns `Err` naming the missing tool rather than substituting anything.
/// `build.rs` already treats `ocloc` as opt-in and best-effort for the same
/// reason: it ships only with the Intel toolchain.
pub fn compile_spirv_ocloc(src: &str, device: &str) -> Result<Vec<u8>, EmitError> {
    let tool = rlx_ir::env::var("RLX_OCLOC").unwrap_or_else(|| "ocloc".to_string());
    let dir = std::env::temp_dir().join(format!("rlx-ocl-emit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| EmitError::Compile(format!("tmpdir: {e}")))?;
    let path = dir.join("k.cl");
    std::fs::write(&path, src).map_err(|e| EmitError::Compile(format!("write: {e}")))?;
    let out = std::process::Command::new(&tool)
        .args(["compile", "-file"])
        .arg(&path)
        .args(["-device", device, "-spv_only", "-out_dir"])
        .arg(&dir)
        .output()
        .map_err(|e| {
            EmitError::Compile(format!(
                "`{tool}` not runnable ({e}). Intel's offline compiler ships with the oneAPI \
                 / Compute-Runtime toolchain; set RLX_OCLOC to its path. There is no correct \
                 in-process fallback."
            ))
        })?;
    if !out.status.success() {
        return Err(EmitError::Compile(format!(
            "ocloc: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let spv = std::fs::read_dir(&dir)
        .map_err(|e| EmitError::Compile(format!("read out_dir: {e}")))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().map(|x| x == "spv").unwrap_or(false))
        .ok_or_else(|| EmitError::Compile("ocloc produced no .spv".into()))?;
    let bytes = std::fs::read(&spv).map_err(|e| EmitError::Compile(format!("read spv: {e}")))?;
    let _ = std::fs::remove_dir_all(&dir);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TARGET: Target = Target::PORTABLE;

    #[test]
    fn the_tiled_schedule_emits_one_stage_and_two_barriers() {
        let (src, facts) = emit_opencl(&matmul_tiled_schedule(DEFAULT_TS), DEFAULT_TS, TARGET)
            .expect("tiled schedule emits");
        assert_eq!(facts.stages, 1);
        assert_eq!(facts.barriers_per_k_iter, 2);
        assert_eq!(facts.work_items, 256);
        assert_eq!(facts.local_bytes, 2048);
        assert_eq!(src.matches("barrier(CLK_LOCAL_MEM_FENCE);").count(), 2);
        assert!(src.contains("reqd_work_group_size(16, 16, 1)"));
    }

    #[test]
    fn the_pipelined_schedule_emits_rotation_and_one_barrier() {
        let (src, facts) = emit_opencl(
            &matmul_pipelined_schedule(DEFAULT_TS, 3),
            DEFAULT_TS,
            TARGET,
        )
        .expect("pipelined schedule emits");
        assert_eq!(facts.stages, 3);
        assert_eq!(facts.barriers_per_k_iter, 1);
        assert_eq!(facts.local_bytes, 6144);
        assert_eq!(src.matches("barrier(CLK_LOCAL_MEM_FENCE);").count(), 1);
        assert!(src.contains("t % STAGES"));
        assert!(src.contains("__local float As[768]"));
    }

    /// Every emitted barrier must carry `CLK_LOCAL_MEM_FENCE`.
    ///
    /// A bare control barrier is precisely how the Vulkan port produced wrong
    /// numbers at large K while passing every static check, so this asserts the
    /// fence rather than merely counting barriers.
    #[test]
    fn every_emitted_barrier_carries_the_local_mem_fence() {
        for sched in [
            matmul_tiled_schedule(DEFAULT_TS),
            matmul_pipelined_schedule(DEFAULT_TS, 2),
            matmul_pipelined_schedule(DEFAULT_TS, 4),
        ] {
            let (src, _) = emit_opencl(&sched, DEFAULT_TS, TARGET).unwrap();
            let total = src.matches("barrier(").count();
            let fenced = src.matches("barrier(CLK_LOCAL_MEM_FENCE)").count();
            assert_eq!(
                total, fenced,
                "`{}` emitted a barrier without CLK_LOCAL_MEM_FENCE",
                sched.name
            );
        }
    }

    #[test]
    fn stage_depth_reaches_the_emitted_source() {
        for stages in 2..=4 {
            let (src, facts) = emit_opencl(
                &matmul_pipelined_schedule(DEFAULT_TS, stages),
                DEFAULT_TS,
                TARGET,
            )
            .unwrap();
            assert!(src.contains(&format!("#define STAGES {stages}u")));
            assert_eq!(facts.local_bytes, 2048 * stages);
        }
    }

    #[test]
    fn an_over_deep_rotation_busts_the_local_budget() {
        let err = emit_opencl(
            &matmul_pipelined_schedule(DEFAULT_TS, 17),
            DEFAULT_TS,
            TARGET,
        )
        .unwrap_err();
        assert!(
            matches!(&err, EmitError::Unsound(es)
                if es.iter().any(|e| matches!(
                    e, KernelScheduleError::SharedOverBudget { .. }))),
            "expected a local-memory budget finding, got {err}"
        );
    }

    #[test]
    fn async_copy_is_refused() {
        let mut sched = matmul_pipelined_schedule(DEFAULT_TS, 2);
        sched.requires = vec![Feature::AsyncCopy];
        assert!(matches!(
            emit_opencl(&sched, DEFAULT_TS, TARGET),
            Err(EmitError::UnsupportedInOpenCl { .. })
        ));
    }

    #[test]
    fn an_unverifiable_schedule_is_rejected_before_emission() {
        let mut sched = matmul_tiled_schedule(DEFAULT_TS);
        sched.barriers[0].producers.clear();
        assert!(matches!(
            emit_opencl(&sched, DEFAULT_TS, TARGET),
            Err(EmitError::Unsound(_))
        ));
    }

    /// The shipping kernel really does have no tile and no barrier, which is
    /// why this port has no A/B baseline. Pin that, so if `matmul.cl` ever
    /// grows one, someone is told the comparison became possible.
    #[test]
    fn the_shipping_matmul_is_still_untiled() {
        const SHIPPING: &str = include_str!("../kernels/matmul.cl");
        assert!(
            !SHIPPING.contains("__local") && !SHIPPING.contains("barrier("),
            "kernels/matmul.cl now uses local memory or barriers — an A/B baseline for the \
             rotation exists, and this module's 'no baseline' caveat is out of date"
        );
    }

    /// Missing `ocloc` is a named coverage limitation, not a silent fallback.
    #[test]
    fn a_missing_ocloc_is_reported_rather_than_worked_around() {
        rlx_ir::env::set("RLX_OCLOC", "/nonexistent/ocloc-for-this-test");
        let (src, _) = emit_opencl(&matmul_tiled_schedule(DEFAULT_TS), DEFAULT_TS, TARGET).unwrap();
        let err = compile_spirv_ocloc(&src, "pvc").unwrap_err();
        rlx_ir::env::unset("RLX_OCLOC");
        match err {
            EmitError::Compile(why) => assert!(
                why.contains("not runnable") && why.contains("no correct in-process fallback"),
                "the error must name the missing tool and refuse a fallback: {why}"
            ),
            other => panic!("expected a Compile error, got {other}"),
        }
    }
}
