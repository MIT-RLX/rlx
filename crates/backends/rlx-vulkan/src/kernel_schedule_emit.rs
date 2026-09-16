// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The schedule generates the kernel** — `KernelSchedule` -> GLSL -> SPIR-V.
//!
//! Fourth target for the same schedule IR, after CUDA
//! (`rlx_gpu_kernels::kernel_schedule_emit`, which HIP shares), MSL
//! (`rlx_metal`) and WGSL (`rlx_wgpu`). The kernel is
//! `shaders/precompiled/matmul_tiled.comp` — the same tiled GEMM shape as
//! `matmul.cu`, Metal's `sgemm_tiled` and `matmul.wgsl`, with the same two
//! barriers per K iteration. Its own comment names the removable one: *"all
//! reads done before the next tile overwrites As/Bs"*.
//!
//! # Why Vulkan is worth the fourth port
//!
//! It is the only backend in this tree that reaches **AMD, Intel and discrete
//! NVIDIA through one source**, so a schedule measured here separates "this
//! transformation is bad" from "this vendor's barriers are cheap" without
//! rewriting the kernel per vendor.
//!
//! # Runtime compilation, and why it needs an external glslang
//!
//! `build.rs` compiles `shaders/*.comp` with naga and the crate advertises "no
//! runtime shader compilation". An emitted kernel has to break that, so this
//! module is feature-gated and is the *only* place the crate compiles a shader
//! at run time — and the only place it needs a tool off `PATH`.
//!
//! `matmul_tiled.comp` is committed as a glslang-built `.spv` with a note that
//! naga's GLSL frontend lacks `memoryBarrierShared` and that its bare
//! `barrier()` "does not enforce the cross-subgroup shared-memory visibility
//! this kernel needs (K > 16 read stale tiles → wrong results)".
//!
//! That note is **correct**, and this port tried to work around it and failed.
//! The attempt is worth recording because the reasoning looked sound:
//!
//! * naga 29 really has no memory-barrier intrinsic — `memoryBarrierShared`,
//!   `groupMemoryBarrier` and `memoryBarrier` all fail to parse;
//! * but its bare `barrier()` lowers to
//!   `OpControlBarrier(Workgroup, Workgroup, AcquireRelease | WorkgroupMemory)`
//!   — semantics `0x108`, which reads like a correct shared-memory
//!   release/acquire, and
//!   [`tests::the_emitted_barrier_carries_workgroup_memory_semantics`] asserts
//!   exactly that operand and passes;
//! * **and the kernel was still wrong.** Built that way, the `serial` arm —
//!   GLSL-identical to the shipping kernel — disagreed with it at
//!   `1 x 4096 x 4096` on an RTX 3080 Ti (1.0612e3 vs 1.0570e3) while agreeing
//!   at small K.
//!
//! So the SPIR-V operand check is **necessary but not sufficient**, and it is
//! the kind of check that reads like proof. `emit_spirv` therefore shells out
//! to `glslangValidator` and emits `memoryBarrierShared(); barrier();`;
//! [`BarrierStyle::BareNaga`] stays reachable only so the failure can be
//! reproduced. With glslang, all three schedules are bit-exact against the
//! shipping kernel on NVIDIA Vulkan.
//!
//! When no glslang is on `PATH`, [`compile_spirv_glslang`] returns an error
//! naming it rather than falling back to the build known to be wrong at large
//! K — a coverage limitation reported is worth more than a silent substitution.
//!
//! # What is derived
//!
//! | emitted | derived from |
//! |---|---|
//! | `shared` array extents | [`Region`] in [`Space::Shared`] |
//! | leading `[STAGES]` rank | `Region::stages` |
//! | shared budget, checked at emit | `lower().shared_bytes` |
//! | `local_size_x/y` | `Role::warps` via `lower().threads` |
//! | `barrier()` placement | position of `Action::Wait` in the body |
//! | K-loop rotation modulus | `KernelSchedule::stages` |

use rlx_ir::DType;
use rlx_ir::kernel_schedule::{
    Access, Action, Barrier, Feature, KernelSchedule, KernelScheduleError, Layout, Region, Role,
    Space, Target, lower,
};

/// The single role: the whole workgroup.
pub const BLOCK_ROLE: &str = "workgroup";

/// Tile edge `matmul_tiled.comp` ships with (`#define TS 16u`).
pub const SHIPPING_TS: usize = 16;

/// Why a schedule could not be emitted as GLSL.
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
    /// More than one role. GLSL compute has no portable subgroup-specialization
    /// story, so this is refused rather than flattened into a uniform kernel.
    MultiRole {
        roles: usize,
    },
    StageDisagreement {
        schedule: usize,
        region: &'static str,
        declared: usize,
    },
    /// A capability GLSL-via-naga cannot express here.
    UnsupportedInGlsl {
        feature: &'static str,
    },
    /// naga rejected the emitted GLSL, or SPIR-V validation failed.
    ///
    /// Carried rather than panicked because the emitter's whole contract is
    /// that a failure is a localized repair target.
    Compile(String),
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
                "{roles} roles declared; GLSL compute has no portable subgroup \
                 specialization, so this is refused rather than flattened"
            ),
            Self::StageDisagreement {
                schedule,
                region,
                declared,
            } => write!(
                f,
                "schedule declares {schedule} stages but region `{region}` declares {declared}"
            ),
            Self::UnsupportedInGlsl { feature } => write!(
                f,
                "`{feature}` has no portable GLSL equivalent here — refused rather than \
                 lowered to something else"
            ),
            Self::Compile(why) => write!(f, "emitted GLSL did not compile: {why}"),
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
    let groups = (ts * ts).div_ceil(32) as u32;
    vec![Role {
        name: BLOCK_ROLE.into(),
        warps: (0..groups.max(1)).collect(),
    }]
}

/// `matmul_tiled.comp` as a typed schedule: one tile pair, two barriers.
pub fn matmul_tiled_schedule(ts: usize) -> KernelSchedule {
    let mut s = KernelSchedule::new(format!("vk_matmul_tiled_{ts}"));
    s.stages = 1;
    s.requires = vec![];
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
pub fn matmul_tiled_pipelined_schedule(ts: usize, stages: usize) -> KernelSchedule {
    let mut s = KernelSchedule::new(format!("vk_matmul_pipe{stages}_{ts}"));
    s.stages = stages;
    s.requires = vec![];
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

/// How the emitted barrier is spelled, and therefore which compiler can build
/// the shader.
///
/// This is not a style preference. `matmul_tiled.comp` carries an explicit
/// `memoryBarrierShared()` and a note that without it "K > 16 read stale tiles
/// → wrong results". Emitting bare `barrier()` and compiling with naga
/// reproduced exactly that: the `serial` arm — byte-equivalent in GLSL to the
/// shipping kernel — disagreed with it at K = 4096 on an RTX 3080 Ti, while
/// agreeing on small K.
///
/// The SPIR-V operand check ([`tests::the_emitted_barrier_carries_workgroup_memory_semantics`])
/// passes for the naga build, so `OpControlBarrier` carrying
/// `AcquireRelease | WorkgroupMemory` is **necessary but not sufficient**. That
/// is worth stating precisely, because it is the kind of check that reads like
/// proof and is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierStyle {
    /// `memoryBarrierShared(); barrier();` — needs an external glslang.
    /// The only style validated against real hardware.
    Explicit,
    /// Bare `barrier()` — parses under naga, which has no memory-barrier
    /// intrinsic. **Known to produce wrong results at large K on NVIDIA.**
    /// Kept reachable only so the failure is reproducible on demand.
    BareNaga,
}

/// What the emitter read out of the schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitFacts {
    pub stages: usize,
    pub barriers_per_k_iter: usize,
    pub shared_bytes: usize,
    pub threads: usize,
    /// Which barrier spelling was emitted. Recorded so a result cannot be
    /// reported without saying whether it came from the validated path.
    pub barrier_style: BarrierStyle,
}

/// Emit GLSL for `sched` at tile edge `ts`, checked against `target`.
pub fn emit_glsl(
    sched: &KernelSchedule,
    ts: usize,
    target: Target,
) -> Result<(String, EmitFacts), EmitError> {
    emit_glsl_styled(sched, ts, target, BarrierStyle::Explicit)
}

/// [`emit_glsl`] with an explicit barrier spelling.
///
/// Only [`BarrierStyle::Explicit`] is validated on hardware; the bare form is
/// reachable so the naga failure stays reproducible.
pub fn emit_glsl_styled(
    sched: &KernelSchedule,
    ts: usize,
    target: Target,
    barrier_style: BarrierStyle,
) -> Result<(String, EmitFacts), EmitError> {
    if sched.roles.len() != 1 {
        return Err(EmitError::MultiRole {
            roles: sched.roles.len(),
        });
    }
    if sched.requires.contains(&Feature::AsyncCopy) {
        return Err(EmitError::UnsupportedInGlsl {
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
        shared_bytes: lowered.shared_bytes,
        threads: lowered.threads,
        barrier_style,
    };
    // The one line the style controls. `memoryBarrierShared()` is what
    // `matmul_tiled.comp` carries and what real hardware turned out to need.
    let barrier = match barrier_style {
        BarrierStyle::Explicit => "memoryBarrierShared(); barrier();",
        BarrierStyle::BareNaga => "barrier();",
    };

    let mut s = String::with_capacity(4096);
    s.push_str(&format!(
        "#version 450\n\
         // @generated by rlx_vulkan::kernel_schedule_emit from KernelSchedule `{}`.\n\
         // Do not edit; edit the schedule.\n\
         //\n\
         // Derived, not authored:\n\
         //   stages              = {}   (KernelSchedule::stages / Region::stages)\n\
         //   barriers per K iter = {}   (count of Action::Wait in the role body)\n\
         //   shared bytes        = {}   (lower().shared_bytes)\n\
         //   threads/group       = {}   (lower().threads, from Role::warps)\n\
         //   buffer offsets      = {:?}\n\
         //\n\
         // Bare `barrier()` deliberately: naga 29's GLSL frontend has no memory\n\
         // barrier intrinsic, and its `barrier()` lowers to OpControlBarrier with\n\
         // AcquireRelease|WorkgroupMemory semantics, which is what this needs.\n",
        sched.name,
        facts.stages,
        facts.barriers_per_k_iter,
        facts.shared_bytes,
        facts.threads,
        lowered.region_offsets,
    ));
    s.push_str(&format!("#define TS {ts}u\n#define STAGES {stages}u\n\n"));
    // Binding and push-constant block are `matmul_tiled.comp`'s, verbatim, so
    // the emitted module drops into the shared descriptor-set layout.
    s.push_str(&format!(
        "layout(local_size_x = {ts}, local_size_y = {ts}) in;\n\n\
         layout(std430, binding = 0) buffer Arena {{ float data[]; }};\n\n\
         layout(push_constant) uniform PC {{\n\
         \x20   uint m;\n\
         \x20   uint k;\n\
         \x20   uint n;\n\
         \x20   uint a_off;\n\
         \x20   uint b_off;\n\
         \x20   uint c_off;\n\
         \x20   uint batch;\n\
         \x20   uint a_bs;\n\
         \x20   uint b_bs;\n\
         \x20   uint c_bs;\n\
         }} pc;\n\n"
    ));
    s.push_str(&format!(
        "shared float As[{}];\nshared float Bs[{}];\n\n",
        stages * ts * ts,
        stages * ts * ts
    ));
    s.push_str(
        "void main() {\n\
         \x20   uint lx = gl_LocalInvocationID.x;\n\
         \x20   uint ly = gl_LocalInvocationID.y;\n\
         \x20   uint col = gl_GlobalInvocationID.x;\n\
         \x20   uint row = gl_GlobalInvocationID.y;\n\
         \x20   uint bz  = gl_GlobalInvocationID.z;\n\n\
         \x20   uint a_base = pc.a_off + bz * pc.a_bs;\n\
         \x20   uint b_base = pc.b_off + bz * pc.b_bs;\n\n\
         \x20   float acc = 0.0;\n\
         \x20   uint tiles = (pc.k + TS - 1u) / TS;\n\n",
    );

    // Staging, bounds-checked exactly as the shipping kernel does it. Shared
    // between prologue and steady state so the two cannot drift.
    let stage_load = |t: &str, buf: &str, ind: &str| -> String {
        format!(
            "{ind}{{\n\
             {ind}    uint base = ({buf}) * TS * TS;\n\
             {ind}    uint a_col = ({t}) * TS + lx;\n\
             {ind}    uint b_row = ({t}) * TS + ly;\n\
             {ind}    As[base + ly * TS + lx] = (row < pc.m && a_col < pc.k)\n\
             {ind}        ? data[a_base + row * pc.k + a_col] : 0.0;\n\
             {ind}    Bs[base + ly * TS + lx] = (b_row < pc.k && col < pc.n)\n\
             {ind}        ? data[b_base + b_row * pc.n + col] : 0.0;\n\
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

    if stages > 1 {
        s.push_str("    // Prologue: fill STAGES-1 buffers before the first compute.\n");
        s.push_str("    for (uint sidx = 0u; sidx + 1u < STAGES; sidx++) {\n");
        s.push_str("        if (sidx < tiles) {\n");
        s.push_str(&stage_load("sidx", "sidx", "            "));
        s.push_str("        }\n    }\n\n");
        s.push_str("    for (uint t = 0u; t < tiles; t++) {\n");
        s.push_str(&format!(
            "        {barrier}  // Action::Wait `tiles_filled` \
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
            "        {barrier}  // Action::Wait `tiles_filled`\n\n"
        ));
        s.push_str(&compute("0u", "        "));
        s.push_str(&format!(
            "\n        {barrier}  // Action::Wait `tiles_consumed`\n    }}\n"
        ));
    }

    s.push_str(
        "\n    if (row < pc.m && col < pc.n && bz < pc.batch) {\n\
         \x20       data[pc.c_off + bz * pc.c_bs + row * pc.n + col] = acc;\n\
         \x20   }\n}\n",
    );

    Ok((s, facts))
}

/// Compile emitted GLSL with an external `glslangValidator`.
///
/// The crate otherwise needs no external toolchain, and this is the one place
/// that is not true. It is not a preference: naga's GLSL frontend has no
/// memory-barrier intrinsic at all, and a kernel built without one disagreed
/// with the shipping kernel at K = 4096 on an RTX 3080 Ti — while agreeing at
/// small K, which is what makes it dangerous rather than merely broken.
///
/// Returns `Err` naming the missing tool rather than falling back to the naga
/// path. Silently substituting a build known to be wrong at large K is exactly
/// the "quietly lowered to something the target does have" failure the schedule
/// IR exists to prevent.
pub fn compile_spirv_glslang(glsl: &str) -> Result<Vec<u32>, EmitError> {
    use std::io::Write as _;
    let tool = glslang_path().ok_or_else(|| {
        EmitError::Compile(
            "glslangValidator not found on PATH (set RLX_GLSLANG to its path). The emitted \
             Vulkan kernel needs `memoryBarrierShared()`, which naga's GLSL frontend cannot \
             parse, so there is no correct in-process fallback."
                .to_string(),
        )
    })?;
    // Unique per CALL, not per process. Keying on the pid alone made two
    // concurrent compiles share `k.comp`/`k.spv` — each overwrote the other's
    // source and the first to finish `remove_dir_all`'d the directory the
    // second was still reading. It passed when run alone and failed under
    // cargo's parallel test threads, which is the worst way for a race to
    // present.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("rlx-vk-emit-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| EmitError::Compile(format!("tmpdir: {e}")))?;
    let src = dir.join("k.comp");
    let out = dir.join("k.spv");
    {
        let mut f = std::fs::File::create(&src)
            .map_err(|e| EmitError::Compile(format!("write source: {e}")))?;
        f.write_all(glsl.as_bytes())
            .map_err(|e| EmitError::Compile(format!("write source: {e}")))?;
    }
    let output = std::process::Command::new(&tool)
        .args(["-V", "--target-env", "vulkan1.3", "-S", "comp"])
        .arg(&src)
        .arg("-o")
        .arg(&out)
        .output()
        .map_err(|e| EmitError::Compile(format!("spawn {tool}: {e}")))?;
    if !output.status.success() {
        return Err(EmitError::Compile(format!(
            "glslang: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let bytes = std::fs::read(&out).map_err(|e| EmitError::Compile(format!("read spv: {e}")))?;
    let _ = std::fs::remove_dir_all(&dir);
    if !bytes.len().is_multiple_of(4) {
        return Err(EmitError::Compile("spv length not a multiple of 4".into()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Locate `glslangValidator`: `RLX_GLSLANG` wins, else PATH.
pub fn glslang_path() -> Option<String> {
    if let Some(p) = rlx_ir::env::var("RLX_GLSLANG") {
        return Some(p);
    }
    let probe = std::process::Command::new("glslangValidator")
        .arg("--version")
        .output()
        .ok()?;
    probe
        .status
        .success()
        .then(|| "glslangValidator".to_string())
}

/// Compile emitted GLSL to SPIR-V words with naga.
///
/// **Not correct for this emitter's kernels at large K** — see
/// [`BarrierStyle`]. Kept public because it is how the failure is reproduced,
/// and because it is the only in-process path when no glslang exists.
///
/// Separate from [`emit_glsl`] so the text is testable without a device *and*
/// the compile failure is its own named error.
pub fn compile_spirv(glsl: &str) -> Result<Vec<u32>, EmitError> {
    use naga::ShaderStage;
    use naga::back::spv;
    use naga::front::glsl::{Frontend, Options};
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    let options = Options::from(ShaderStage::Compute);
    let module = Frontend::default()
        .parse(&options, glsl)
        .map_err(|e| EmitError::Compile(format!("naga parse: {e:?}")))?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| EmitError::Compile(format!("naga validate: {e:?}")))?;
    spv::write_vec(&module, &info, &spv::Options::default(), None)
        .map_err(|e| EmitError::Compile(format!("naga spv: {e:?}")))
}

/// Emit and compile in one step.
pub fn emit_spirv(
    sched: &KernelSchedule,
    ts: usize,
    target: Target,
) -> Result<(Vec<u32>, EmitFacts), EmitError> {
    let (glsl, facts) = emit_glsl(sched, ts, target)?;
    Ok((compile_spirv_glslang(&glsl)?, facts))
}

/// Emit and compile with a chosen barrier style and compiler.
///
/// [`BarrierStyle::BareNaga`] builds in-process and is **known wrong at large
/// K**; it exists so the failure is reproducible without editing the emitter.
pub fn emit_spirv_styled(
    sched: &KernelSchedule,
    ts: usize,
    target: Target,
    style: BarrierStyle,
) -> Result<(Vec<u32>, EmitFacts), EmitError> {
    let (glsl, facts) = emit_glsl_styled(sched, ts, target, style)?;
    let words = match style {
        BarrierStyle::Explicit => compile_spirv_glslang(&glsl)?,
        BarrierStyle::BareNaga => compile_spirv(&glsl)?,
    };
    Ok((words, facts))
}

// ── Reachability from the backend's kernel dispatch ─────────────────────
//
// `backend.rs` selects a matmul by `&'static str` name and `Kernels::pipeline`
// resolves that against the blobs `build.rs` embedded. An emitted kernel has no
// blob, so it needs a name the dispatcher can carry and the cache can resolve.
// The set is bounded (one per supported schedule), which is what makes leaking
// the names acceptable rather than a slow leak keyed on workload.

/// Prefix marking a kernel name as generated rather than embedded.
pub const SCHEDULED_PREFIX: &str = "matmul_sched_";

/// The schedule-verifier [`Target`] for the attached Vulkan device.
///
/// Built by **querying the device**, not by naming it. Vulkan spans AMD, Intel,
/// NVIDIA and MoltenVK, so a table of per-vendor constants would be a guess
/// that goes stale; `vkGetPhysicalDeviceProperties` already reports exactly the
/// two facts a schedule is checked against, and `VulkanDevice` already stores
/// them.
///
/// * `max_shared_bytes` &larr; `limits.max_compute_shared_memory_size`
/// * `max_warps` &larr; `limits.max_compute_work_group_invocations / 32`
/// * `CoopMatrix` &larr; `coop_matmul`, which is set only when the device
///   advertises a usable 16x16x16 config **and** the features its kernel needs
///   were enabled — so this reports what rlx can actually drive, not what the
///   hardware might have.
///
/// This is [`rlx_ir::kernel_schedule::Features::Queried`] in spirit: measured
/// on this machine rather than assumed from an architecture name, which is the
/// distinction `CostCalibration` draws for cost models and
/// `cost-model-uncalibrated-ranking` is the defect for getting wrong.
///
/// With no device, falls back to [`Target::PORTABLE`]. That under-reports both
/// the budget and the capability list, which is the safe direction: a schedule
/// wrongly refused is recoverable, one wrongly admitted is a launch failure or
/// silent corruption.
pub fn device_target() -> Target {
    use rlx_ir::kernel_schedule::{FeatureSet, Features};
    let Some(dev) = crate::device::vulkan_device() else {
        return Target::PORTABLE;
    };
    let mut features = FeatureSet::default();
    if dev.coop_matmul {
        // The shape is part of the capability, not a detail — `coop_matmul` is
        // gated on the 16x16x16 f16.f16->f32 config specifically.
        features.push(Feature::CoopMatrix {
            m: 16,
            n: 16,
            k: 16,
        });
    }
    // Every Vulkan implementation supports a shared-memory control barrier with
    // memory semantics; that is what `barrier()` compiles to. It is not
    // `AsyncCopy`, and nothing here claims one.
    features.push(Feature::AsyncBarrier);

    Target {
        // Leaked once per process; the device is a `&'static` singleton, so the
        // set of names is bounded by one.
        name: Box::leak(dev.name.clone().into_boxed_str()),
        max_shared_bytes: dev.limits.max_compute_shared_memory_size as usize,
        max_warps: (dev.limits.max_compute_work_group_invocations as usize / 32).max(1),
        features: Features::Queried(features),
    }
}

/// The kernel name selected by `RLX_VULKAN_SCHEDULE_MATMUL`, if any.
///
/// `None` means the shipping shaders are in play and nothing changes. An
/// unrecognized value is reported once and treated as unset — never silently
/// accepted, because a typo that quietly runs the baseline makes an A/B measure
/// one path twice.
pub fn scheduled_kernel_name() -> Option<&'static str> {
    use std::sync::OnceLock;
    static NAME: OnceLock<Option<&'static str>> = OnceLock::new();
    *NAME.get_or_init(|| {
        let v = rlx_ir::env::var("RLX_VULKAN_SCHEDULE_MATMUL")?;
        let v = v.trim().to_ascii_lowercase();
        if v == "default" || v == "0" {
            return None;
        }
        let suffix = if v == "serial" {
            "serial".to_string()
        } else if let Some(rest) = v.strip_prefix("pipelined") {
            let stages: usize = rest
                .trim_start_matches([':', '='])
                .parse()
                .unwrap_or(2)
                .max(2);
            format!("pipe{stages}")
        } else {
            eprintln!(
                "rlx-vulkan: RLX_VULKAN_SCHEDULE_MATMUL={v:?} is not one of \
                 default|serial|pipelined[:N] — using the shipping shaders"
            );
            return None;
        };
        // Bounded set (serial + pipe2..pipeN), so this leak is a constant.
        Some(&*Box::leak(
            format!("{SCHEDULED_PREFIX}{suffix}").into_boxed_str(),
        ))
    })
}

/// The schedule behind a generated kernel name, if the name is one of ours.
pub fn schedule_for_name(name: &str, ts: usize) -> Option<KernelSchedule> {
    let suffix = name.strip_prefix(SCHEDULED_PREFIX)?;
    if suffix == "serial" {
        return Some(matmul_tiled_schedule(ts));
    }
    let stages: usize = suffix.strip_prefix("pipe")?.parse().ok()?;
    Some(matmul_tiled_pipelined_schedule(ts, stages.max(2)))
}

/// SPIR-V for a generated kernel name, or `None` if the name is not ours.
///
/// Returns `Err` rather than falling back to the shipping shader: an arm that
/// silently ran the baseline would report parity rather than absence.
pub fn spirv_for_name(name: &str, target: Target) -> Option<Result<Vec<u32>, EmitError>> {
    let sched = schedule_for_name(name, SHIPPING_TS)?;
    Some(emit_spirv(&sched, SHIPPING_TS, target).map(|(w, facts)| {
        if rlx_ir::env::flag("RLX_VERBOSE") {
            eprintln!(
                "rlx-vulkan: `{name}` emitted from `{}` — {facts:?}",
                sched.name
            );
        }
        w
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The portable floor. `matmul_tiled.comp` is the discrete-GPU path but is
    /// written to run anywhere, so its schedule must clear the floor.
    const TARGET: Target = Target::PORTABLE;

    #[test]
    fn the_shipping_schedule_emits_one_stage_and_two_barriers() {
        let (glsl, facts) =
            emit_glsl(&matmul_tiled_schedule(SHIPPING_TS), SHIPPING_TS, TARGET).unwrap();
        assert_eq!(facts.stages, 1);
        assert_eq!(facts.barriers_per_k_iter, 2);
        assert_eq!(facts.threads, 256);
        // 16*16*4 * 2 buffers = 2048.
        assert_eq!(facts.shared_bytes, 2048);
        // Count statements, not the word: the generated header comment explains
        // the `barrier()` lowering and would otherwise be counted as a barrier.
        assert_eq!(glsl.matches("barrier();").count(), 2);
        assert!(glsl.contains("#define STAGES 1u"));
        assert!(glsl.contains("local_size_x = 16, local_size_y = 16"));
    }

    #[test]
    fn the_pipelined_schedule_emits_rotation_and_one_barrier() {
        let (glsl, facts) = emit_glsl(
            &matmul_tiled_pipelined_schedule(SHIPPING_TS, 3),
            SHIPPING_TS,
            TARGET,
        )
        .unwrap();
        assert_eq!(facts.stages, 3);
        assert_eq!(facts.barriers_per_k_iter, 1);
        assert_eq!(facts.shared_bytes, 6144);
        assert_eq!(glsl.matches("barrier();").count(), 1);
        assert!(glsl.contains("t % STAGES"));
        assert!(glsl.contains("kt % STAGES"));
        // The rotation must actually widen the shared arrays.
        assert!(glsl.contains("shared float As[768]"));
    }

    /// Every emitted variant must survive naga. This is the device-free half of
    /// the correctness story: a shader that does not compile cannot be measured,
    /// and finding that out on the rig wastes a sync-build-run round trip.
    #[test]
    fn every_emitted_variant_compiles_to_spirv() {
        let mut scheds = vec![matmul_tiled_schedule(SHIPPING_TS)];
        for stages in 2..=4 {
            scheds.push(matmul_tiled_pipelined_schedule(SHIPPING_TS, stages));
        }
        if glslang_path().is_none() {
            eprintln!(
                "glslangValidator not found — skipping. This is a coverage limitation, \
                 not a pass: the shipped emit path cannot be exercised here."
            );
            return;
        }
        for s in &scheds {
            let words = emit_spirv(s, SHIPPING_TS, TARGET)
                .unwrap_or_else(|e| panic!("`{}` did not compile: {e}", s.name))
                .0;
            assert!(words.len() > 16, "`{}` produced a stub module", s.name);
            assert_eq!(words[0], 0x0723_0203, "not a SPIR-V magic number");
        }
    }

    /// **The claim this module rests on.**
    ///
    /// `matmul_tiled.comp` is committed as a glslang build because bare
    /// `barrier()` was believed not to order shared memory. If that were true
    /// of naga 29, every emitted kernel here would be racy — right most of the
    /// time and wrong for K > TS, which no smoke test would catch.
    ///
    /// So assert the operand rather than trusting either the note or the
    /// toolchain: `OpControlBarrier` (opcode 224) must carry memory semantics
    /// including `WorkgroupMemory` (0x100).
    #[test]
    fn the_emitted_barrier_carries_workgroup_memory_semantics() {
        // Deliberately the naga build: this test is about what naga emits.
        let (words, _) = emit_spirv_styled(
            &matmul_tiled_schedule(SHIPPING_TS),
            SHIPPING_TS,
            TARGET,
            BarrierStyle::BareNaga,
        )
        .unwrap();

        // Collect OpConstant id -> value so the semantics operand resolves.
        let mut consts = std::collections::HashMap::new();
        let mut i = 5usize;
        while i < words.len() {
            let wc = (words[i] >> 16) as usize;
            if wc == 0 {
                break;
            }
            if (words[i] & 0xFFFF) == 43 && i + 3 < words.len() {
                consts.insert(words[i + 2], words[i + 3]);
            }
            i += wc;
        }

        const WORKGROUP_MEMORY: u32 = 0x100;
        let mut found = 0usize;
        let mut i = 5usize;
        while i < words.len() {
            let wc = (words[i] >> 16) as usize;
            if wc == 0 {
                break;
            }
            if (words[i] & 0xFFFF) == 224 && i + 3 < words.len() {
                let semantics = consts.get(&words[i + 3]).copied().unwrap_or(0);
                if semantics & WORKGROUP_MEMORY != 0 {
                    found += 1;
                }
            }
            i += wc;
        }
        assert!(
            found >= 2,
            "expected both naga barriers to carry WorkgroupMemory semantics, found {found}"
        );
        // AND YET: this passing does not make the naga build correct. The
        // `serial` arm built this way disagreed with the shipping kernel at
        // K = 4096 on an RTX 3080 Ti. The operand is necessary, not sufficient
        // — which is why `emit_spirv` uses glslang and this style is opt-in.
    }

    #[test]
    fn an_over_deep_rotation_busts_the_shared_budget() {
        // PORTABLE's floor is 32 KiB; 2 KiB per stage, so 17 stages busts it.
        let err = emit_glsl(
            &matmul_tiled_pipelined_schedule(SHIPPING_TS, 17),
            SHIPPING_TS,
            TARGET,
        )
        .unwrap_err();
        assert!(
            matches!(&err, EmitError::Unsound(es)
                if es.iter().any(|e| matches!(
                    e, KernelScheduleError::SharedOverBudget { .. }))),
            "expected a shared-budget finding, got {err}"
        );
    }

    #[test]
    fn async_copy_is_refused_in_glsl() {
        let mut sched = matmul_tiled_pipelined_schedule(SHIPPING_TS, 2);
        sched.requires = vec![Feature::AsyncCopy];
        assert!(matches!(
            emit_glsl(&sched, SHIPPING_TS, TARGET),
            Err(EmitError::UnsupportedInGlsl { .. })
        ));
    }

    #[test]
    fn an_unverifiable_schedule_is_rejected_before_emission() {
        let mut sched = matmul_tiled_schedule(SHIPPING_TS);
        sched.barriers[0].producers.clear();
        assert!(matches!(
            emit_glsl(&sched, SHIPPING_TS, TARGET),
            Err(EmitError::Unsound(_))
        ));
    }

    /// Anti-drift against the shipping shader, the same rule the CUDA, Metal
    /// and wgpu ports carry: read the real source, do not trust a copy.
    #[test]
    fn the_emitted_body_does_not_drift_from_matmul_tiled_comp() {
        const SHIPPING: &str = include_str!("../shaders/precompiled/matmul_tiled.comp");
        let (glsl, _) =
            emit_glsl(&matmul_tiled_schedule(SHIPPING_TS), SHIPPING_TS, TARGET).unwrap();
        for fragment in [
            "uint tiles = (pc.k + TS - 1u) / TS;",
            "data[pc.c_off + bz * pc.c_bs + row * pc.n + col] = acc;",
        ] {
            assert!(
                SHIPPING.contains(fragment),
                "matmul_tiled.comp no longer contains `{fragment}` — update the emitter"
            );
            assert!(
                glsl.contains(fragment),
                "emitted shader missing `{fragment}`"
            );
        }
        // Geometry must agree, or the emitter describes a different kernel.
        assert!(SHIPPING.contains("#define TS 16u"));
        assert!(glsl.contains("#define TS 16u"));
    }
}
