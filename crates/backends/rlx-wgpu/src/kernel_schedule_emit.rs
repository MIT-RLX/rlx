// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The schedule generates the kernel** — `KernelSchedule` -> WGSL.
//!
//! Third target for the same schedule IR, after
//! `rlx_gpu_kernels::kernel_schedule_emit` (CUDA) and
//! `rlx_metal::kernel_schedule_emit` (MSL). The kernel is
//! [`crate::kernels::MATMUL_WGSL`]'s `matmul` entry — the same tiled GEMM
//! shape as `kernels/matmul.cu` and Metal's `sgemm_tiled`, with the same two
//! `workgroupBarrier()` calls per K iteration. `matmul.wgsl` even names the
//! second one in its own header: *"4. workgroupBarrier — wait before reusing
//! the tiles."* Reusing the tiles is exactly what a rotation stops doing.
//!
//! # Why this target specifically
//!
//! Not for coverage. The same WGSL runs on **Metal on Apple silicon and on
//! Vulkan on NVIDIA**, which is the only way to tell two hypotheses apart:
//!
//! * the rotation costs more in occupancy than the barrier saves — a property
//!   of the transformation, which should then lose on both;
//! * Apple GPUs have unusually cheap barriers and tight threadgroup memory — a
//!   property of the vendor, which should then lose on Metal and not on Vulkan.
//!
//! The Metal measurement alone cannot distinguish those. One WGSL source
//! compiled to both backends can, because the *only* thing that varies is the
//! GPU.
//!
//! # What is derived
//!
//! | emitted | derived from |
//! |---|---|
//! | `var<workgroup>` extents | [`Region`] in [`Space::Shared`] |
//! | leading `array<_, STAGES>` rank | `Region::stages` |
//! | workgroup memory budget, checked at emit | `lower().shared_bytes` |
//! | `@workgroup_size` | `Role::warps` via `lower().threads` |
//! | `workgroupBarrier()` placement | position of `Action::Wait` in the body |
//! | K-loop rotation modulus | `KernelSchedule::stages` |
//!
//! Per-thread index algebra is the emitter's, not the schedule's — the same
//! division of labour the CUDA and Metal emitters document.
//!
//! # No async copy
//!
//! WGSL has no `cp.async` analogue, so `Feature::AsyncCopy` is refused rather
//! than lowered to a plain load that would quietly measure a different thing.
//! This is the structural arm only.

use rlx_ir::DType;
use rlx_ir::kernel_schedule::{
    Access, Action, Barrier, Feature, KernelSchedule, KernelScheduleError, Layout, Region, Role,
    Space, Target, lower,
};

/// The single role in `matmul.wgsl`: the whole workgroup.
pub const BLOCK_ROLE: &str = "workgroup";

/// Entry point of the emitted shader.
///
/// Deliberately not `matmul`: the baseline arm compiles the shipping module,
/// and two entry points of the same name in one process is how an A/B ends up
/// measuring the wrong one.
pub const EMITTED_ENTRY: &str = "matmul_sched";

/// `matmul.wgsl`'s tile geometry. These mirror its `const TILE_M/N/K` and
/// `RM`/`RN`, and the emitter's index algebra depends on them, so they are
/// stated once here rather than spread through format strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgslTile {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub rm: usize,
    pub rn: usize,
}

impl WgslTile {
    /// What `matmul.wgsl` ships: 32x32 output tile, K depth 16, 4x4 registers,
    /// 8x8 workgroup.
    pub const SHIPPING: Self = Self {
        tile_m: 32,
        tile_n: 32,
        tile_k: 16,
        rm: 4,
        rn: 4,
    };

    pub const fn wg_m(&self) -> usize {
        self.tile_m / self.rm
    }
    pub const fn wg_n(&self) -> usize {
        self.tile_n / self.rn
    }
    pub const fn threads(&self) -> usize {
        self.wg_m() * self.wg_n()
    }
}

/// Why a schedule could not be emitted as WGSL.
#[derive(Debug, Clone, PartialEq)]
pub enum EmitError {
    /// The schedule does not verify against the target.
    Unsound(Vec<KernelScheduleError>),
    /// A region the GEMM family requires is missing.
    MissingRegion { name: &'static str },
    /// A declared region's extents disagree with the tile geometry.
    RegionTileMismatch {
        region: &'static str,
        declared: Vec<usize>,
        from_tile: Vec<usize>,
    },
    /// The roles imply a different workgroup size than the tile does.
    ThreadCountMismatch { from_roles: usize, from_tile: usize },
    /// More than one role. Warp specialization is expressible in the IR; WGSL
    /// has no way to give distinct work to distinct subgroups portably, and
    /// pretending otherwise would emit a uniform kernel under a specialized
    /// schedule's name.
    MultiRole { roles: usize },
    /// `stages` is inconsistent across the staged regions and the schedule.
    StageDisagreement {
        schedule: usize,
        region: &'static str,
        declared: usize,
    },
    /// A capability WGSL does not portably expose.
    UnsupportedInWgsl { feature: &'static str },
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
                "{roles} roles declared; WGSL has no portable warp specialization, so this \
                 is refused rather than flattened"
            ),
            Self::StageDisagreement {
                schedule,
                region,
                declared,
            } => write!(
                f,
                "schedule declares {schedule} stages but region `{region}` declares {declared}"
            ),
            Self::UnsupportedInWgsl { feature } => write!(
                f,
                "`{feature}` has no portable WGSL equivalent — refused rather than lowered \
                 to something else"
            ),
        }
    }
}

impl std::error::Error for EmitError {}

fn gemm_regions(tile: WgslTile, stages: usize) -> Vec<Region> {
    vec![
        Region {
            name: "tile_a".into(),
            space: Space::Shared,
            dims: vec![tile.tile_m, tile.tile_k],
            dtype: DType::F32,
            stages,
            layout: Layout::row_major(&[tile.tile_m, tile.tile_k]),
        },
        Region {
            name: "tile_b".into(),
            space: Space::Shared,
            dims: vec![tile.tile_k, tile.tile_n],
            dtype: DType::F32,
            stages,
            layout: Layout::row_major(&[tile.tile_k, tile.tile_n]),
        },
        Region {
            name: "acc".into(),
            space: Space::Register,
            dims: vec![tile.rm, tile.rn],
            dtype: DType::F32,
            stages: 1,
            layout: Layout::row_major(&[tile.rm, tile.rn]),
        },
    ]
}

fn gemm_role(tile: WgslTile) -> Vec<Role> {
    // WGSL has no warp concept, but `Role::warps` is how the IR counts threads
    // and `lower()` turns it back into a thread count. 32 is the divisor it
    // uses; the emitter checks the result against the tile rather than
    // trusting it.
    let groups = tile.threads().div_ceil(32) as u32;
    vec![Role {
        name: BLOCK_ROLE.into(),
        warps: (0..groups.max(1)).collect(),
    }]
}

/// `matmul.wgsl` as a typed schedule: one tile pair, two barriers.
///
/// The arm that makes a pipelined delta attributable — emitting this should
/// reproduce the shipping shader, so anything the pipelined arm shows is the
/// rotation rather than the code generator.
pub fn matmul_schedule(tile: WgslTile) -> KernelSchedule {
    let mut s = KernelSchedule::new(format!("wgsl_matmul_{}x{}", tile.tile_m, tile.tile_k));
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
                access: Access::plain("tile_a"),
                stage: 0,
            },
            Action::Load {
                access: Access::plain("tile_b"),
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
                reads: vec![Access::plain("tile_a"), Access::plain("tile_b")],
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
pub fn matmul_pipelined_schedule(tile: WgslTile, stages: usize) -> KernelSchedule {
    let mut s = KernelSchedule::new(format!("wgsl_matmul_pipe{stages}"));
    s.stages = stages;
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
                reads: vec![Access::plain("tile_a"), Access::plain("tile_b")],
                writes: vec![Access::plain("acc")],
                stage: 0,
                via: None,
            },
            Action::Load {
                access: Access::plain("tile_a"),
                stage: next,
            },
            Action::Load {
                access: Access::plain("tile_b"),
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
    pub workgroup_bytes: usize,
    pub threads: usize,
}

/// Emit WGSL for `sched` at `tile`, checked against `target`.
///
/// The emitted entry takes the same `arena` + `Params` bindings as
/// `matmul.wgsl`, so it drops into the same bind group and the experiment
/// varies one thing.
pub fn emit_wgsl(
    sched: &KernelSchedule,
    tile: WgslTile,
    target: Target,
) -> Result<(String, EmitFacts), EmitError> {
    if sched.roles.len() != 1 {
        return Err(EmitError::MultiRole {
            roles: sched.roles.len(),
        });
    }
    if sched.requires.contains(&Feature::AsyncCopy) {
        return Err(EmitError::UnsupportedInWgsl {
            feature: "AsyncCopy",
        });
    }
    let lowered = lower(sched, target).map_err(EmitError::Unsound)?;
    if lowered.threads != tile.threads() {
        return Err(EmitError::ThreadCountMismatch {
            from_roles: lowered.threads,
            from_tile: tile.threads(),
        });
    }

    let region = |name: &'static str| -> Result<&Region, EmitError> {
        sched
            .regions
            .iter()
            .find(|r| r.name == name)
            .ok_or(EmitError::MissingRegion { name })
    };
    let a = region("tile_a")?;
    let b = region("tile_b")?;
    region("acc")?;
    for (r, name, want) in [
        (a, "tile_a", [tile.tile_m, tile.tile_k]),
        (b, "tile_b", [tile.tile_k, tile.tile_n]),
    ] {
        if r.dims != want {
            return Err(EmitError::RegionTileMismatch {
                region: name,
                declared: r.dims.clone(),
                from_tile: want.to_vec(),
            });
        }
    }

    let stages = sched.stages.max(1);
    for (r, name) in [(a, "tile_a"), (b, "tile_b")] {
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
        workgroup_bytes: lowered.shared_bytes,
        threads: lowered.threads,
    };

    let (tm, tn, tk, rm, rn) = (tile.tile_m, tile.tile_n, tile.tile_k, tile.rm, tile.rn);
    // `matmul.wgsl` splits the cooperative load as RM rows x (TILE_K/WG_N)
    // columns for A and (TILE_K/WG_M) rows x RN columns for B. Derived rather
    // than hardcoded to 2, which only held at the shipping geometry.
    let a_cols_per_thread = tk / tile.wg_n();
    let b_rows_per_thread = tk / tile.wg_m();

    let mut src = String::with_capacity(6144);
    src.push_str(&format!(
        "// @generated by rlx_wgpu::kernel_schedule_emit from KernelSchedule `{}`.\n\
         // Do not edit; edit the schedule.\n\
         //\n\
         // Derived, not authored:\n\
         //   stages              = {}   (KernelSchedule::stages / Region::stages)\n\
         //   barriers per K iter = {}   (count of Action::Wait in the role body)\n\
         //   workgroup bytes     = {}   (lower().shared_bytes)\n\
         //   threads/group       = {}   (lower().threads, from Role::warps)\n\
         //   buffer offsets      = {:?}\n\n",
        sched.name,
        facts.stages,
        facts.barriers_per_k_iter,
        facts.workgroup_bytes,
        facts.threads,
        lowered.region_offsets,
    ));

    // Bindings and Params are `matmul.wgsl`'s, verbatim, so the bind group is
    // shared and the ABI cannot drift.
    src.push_str(
        "struct Params {\n\
         \x20   m: u32,\n\
         \x20   k: u32,\n\
         \x20   n: u32,\n\
         \x20   a_off: u32,\n\
         \x20   b_off: u32,\n\
         \x20   c_off: u32,\n\
         \x20   batch: u32,\n\
         \x20   a_batch_stride: u32,\n\
         \x20   b_batch_stride: u32,\n\
         \x20   c_batch_stride: u32,\n\
         \x20   has_bias: u32,\n\
         \x20   bias_off: u32,\n\
         \x20   act_id: u32,\n\
         \x20   _p0: u32, _p1: u32, _p2: u32,\n\
         };\n\n\
         @group(0) @binding(0) var<storage, read_write> arena: array<f32>;\n\
         @group(0) @binding(1) var<uniform>              params: Params;\n\n",
    );

    src.push_str(&format!(
        "const TILE_M: u32 = {tm}u;\n\
         const TILE_N: u32 = {tn}u;\n\
         const TILE_K: u32 = {tk}u;\n\
         const RM: u32 = {rm}u;\n\
         const RN: u32 = {rn}u;\n\
         const STAGES: u32 = {stages}u;\n\n"
    ));

    // The `array<_, STAGES>` rank is the only difference from the shipping
    // declaration, and it comes from `Region::stages`.
    src.push_str(&format!(
        "var<workgroup> tile_a: array<array<array<f32, {tk}>, {tm}>, {stages}>;\n\
         var<workgroup> tile_b: array<array<array<f32, {tn}>, {tk}>, {stages}>;\n\n"
    ));

    // Activation epilogue, carried from `matmul.wgsl`. Not part of the machine
    // schedule — no region, role or barrier describes it — so it is reproduced
    // rather than claimed as derived. Only the identity case is exercised by
    // the A/B, but emitting a stub would make the kernel non-substitutable.
    src.push_str(
        "fn apply_act(x: f32) -> f32 {\n\
         \x20   var v = x;\n\
         \x20   if (params.act_id == 0xFFFFu) { return v; }\n\
         \x20   switch (params.act_id) {\n\
         \x20       case 0u: { v = max(v, 0.0); }\n\
         \x20       case 1u: { v = 1.0 / (1.0 + exp(-clamp(v, -88.0, 88.0))); }\n\
         \x20       case 2u: { v = tanh(clamp(v, -15.0, 15.0)); }\n\
         \x20       case 5u: { v = sqrt(v); }\n\
         \x20       case 7u: { v = -v; }\n\
         \x20       case 8u: { v = abs(v); }\n\
         \x20       case 10u: { v = v / (1.0 + exp(-clamp(v, -88.0, 88.0))); }\n\
         \x20       default: {}\n\
         \x20   }\n\
         \x20   return v;\n\
         }\n\n",
    );

    src.push_str(&format!(
        "@compute @workgroup_size({}, {})\n\
         fn {EMITTED_ENTRY}(\n\
         \x20   @builtin(local_invocation_id) lid: vec3<u32>,\n\
         \x20   @builtin(workgroup_id)        wid: vec3<u32>,\n\
         ) {{\n",
        tile.wg_n(),
        tile.wg_m()
    ));

    // Preamble is `matmul.wgsl`'s, including the never-early-return rule its
    // comment records: a `gid`-gated return upstream of a barrier is varying
    // control flow to FXC and gets rejected (X4026).
    src.push_str(
        "    let bz = wid.z;\n\
         \x20   let in_batch = bz < params.batch;\n\
         \x20   let bz_safe = select(0u, bz, in_batch);\n\n\
         \x20   let lr = lid.y;\n\
         \x20   let lc = lid.x;\n\
         \x20   let row_base = wid.y * TILE_M + lr * RM;\n\
         \x20   let col_base = wid.x * TILE_N + lc * RN;\n\n\
         \x20   let a_base = params.a_off + bz_safe * params.a_batch_stride;\n\
         \x20   let b_base = params.b_off + bz_safe * params.b_batch_stride;\n\
         \x20   let c_base = params.c_off + bz_safe * params.c_batch_stride;\n\n\
         \x20   var acc: array<array<f32, 4>, 4>;\n\
         \x20   for (var i: u32 = 0u; i < RM; i = i + 1u) {\n\
         \x20       for (var j: u32 = 0u; j < RN; j = j + 1u) {\n\
         \x20           acc[i][j] = 0.0;\n\
         \x20       }\n\
         \x20   }\n\n\
         \x20   let n_tiles = (params.k + TILE_K - 1u) / TILE_K;\n\n",
    );

    // Bounds-checked cooperative staging, shared between prologue and steady
    // state so the two cannot drift apart.
    let stage_load = |t_expr: &str, buf: &str, indent: &str| -> String {
        format!(
            "{indent}for (var i: u32 = 0u; i < RM; i = i + 1u) {{\n\
             {indent}    let m_local = lr * RM + i;\n\
             {indent}    let global_row = wid.y * TILE_M + m_local;\n\
             {indent}    for (var j: u32 = 0u; j < {a_cols_per_thread}u; j = j + 1u) {{\n\
             {indent}        let k_local = lc * {a_cols_per_thread}u + j;\n\
             {indent}        let global_k = ({t_expr}) * TILE_K + k_local;\n\
             {indent}        if (in_batch && global_row < params.m && global_k < params.k) {{\n\
             {indent}            tile_a[{buf}][m_local][k_local] = arena[a_base + global_row * params.k + global_k];\n\
             {indent}        }} else {{\n\
             {indent}            tile_a[{buf}][m_local][k_local] = 0.0;\n\
             {indent}        }}\n\
             {indent}    }}\n\
             {indent}}}\n\
             {indent}for (var i: u32 = 0u; i < {b_rows_per_thread}u; i = i + 1u) {{\n\
             {indent}    let k_local = lr * {b_rows_per_thread}u + i;\n\
             {indent}    let global_k = ({t_expr}) * TILE_K + k_local;\n\
             {indent}    for (var j: u32 = 0u; j < RN; j = j + 1u) {{\n\
             {indent}        let n_local = lc * RN + j;\n\
             {indent}        let global_col = wid.x * TILE_N + n_local;\n\
             {indent}        if (in_batch && global_k < params.k && global_col < params.n) {{\n\
             {indent}            tile_b[{buf}][k_local][n_local] = arena[b_base + global_k * params.n + global_col];\n\
             {indent}        }} else {{\n\
             {indent}            tile_b[{buf}][k_local][n_local] = 0.0;\n\
             {indent}        }}\n\
             {indent}    }}\n\
             {indent}}}\n"
        )
    };
    let compute = |buf: &str, indent: &str| -> String {
        format!(
            "{indent}for (var kk: u32 = 0u; kk < TILE_K; kk = kk + 1u) {{\n\
             {indent}    var a_reg: array<f32, 4>;\n\
             {indent}    var b_reg: array<f32, 4>;\n\
             {indent}    for (var i: u32 = 0u; i < RM; i = i + 1u) {{\n\
             {indent}        a_reg[i] = tile_a[{buf}][lr * RM + i][kk];\n\
             {indent}    }}\n\
             {indent}    for (var j: u32 = 0u; j < RN; j = j + 1u) {{\n\
             {indent}        b_reg[j] = tile_b[{buf}][kk][lc * RN + j];\n\
             {indent}    }}\n\
             {indent}    for (var i: u32 = 0u; i < RM; i = i + 1u) {{\n\
             {indent}        for (var j: u32 = 0u; j < RN; j = j + 1u) {{\n\
             {indent}            acc[i][j] = acc[i][j] + a_reg[i] * b_reg[j];\n\
             {indent}        }}\n\
             {indent}    }}\n\
             {indent}}}\n"
        )
    };

    if stages > 1 {
        src.push_str("    // Prologue: fill STAGES-1 buffers before the first compute.\n");
        src.push_str("    for (var s: u32 = 0u; s + 1u < STAGES; s = s + 1u) {\n");
        src.push_str(&stage_load("s", "s", "        "));
        src.push_str("    }\n\n");
        src.push_str("    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {\n");
        src.push_str(&format!(
            "        workgroupBarrier();  // Action::Wait `tiles_filled` \
             ({barriers_per_k_iter}/iter declared)\n\n"
        ));
        src.push_str(&compute("t % STAGES", "        "));
        src.push_str(
            "\n        // Stage the tile needed STAGES-1 iterations from now, into the\n\
             \x20       // buffer iteration t-1 finished reading before the barrier above.\n\
             \x20       // That ordering is why no second barrier is needed here.\n\
             \x20       let kt = t + STAGES - 1u;\n\
             \x20       if (kt < n_tiles) {\n",
        );
        src.push_str(&stage_load("kt", "kt % STAGES", "            "));
        src.push_str("        }\n    }\n");
    } else {
        src.push_str("    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {\n");
        src.push_str(&stage_load("t", "0u", "        "));
        src.push_str("\n        workgroupBarrier();  // Action::Wait `tiles_filled`\n\n");
        src.push_str(&compute("0u", "        "));
        src.push_str("\n        workgroupBarrier();  // Action::Wait `tiles_consumed`\n    }\n");
    }

    src.push_str(
        "\n    for (var i: u32 = 0u; i < RM; i = i + 1u) {\n\
         \x20       let global_row = row_base + i;\n\
         \x20       if (in_batch && global_row < params.m) {\n\
         \x20           for (var j: u32 = 0u; j < RN; j = j + 1u) {\n\
         \x20               let global_col = col_base + j;\n\
         \x20               if (global_col < params.n) {\n\
         \x20                   var v = acc[i][j];\n\
         \x20                   if (params.has_bias != 0u) {\n\
         \x20                       v = v + arena[params.bias_off + global_col];\n\
         \x20                   }\n\
         \x20                   v = apply_act(v);\n\
         \x20                   arena[c_base + global_row * params.n + global_col] = v;\n\
         \x20               }\n\
         \x20           }\n\
         \x20       }\n\
         \x20   }\n}\n",
    );

    Ok((src, facts))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The portable floor: `matmul.wgsl` ships with "no extensions, no subgroup
    /// ops; runs identically on Metal/Vulkan/DX12/WebGPU", so the schedule
    /// describing it must clear the floor rather than any one vendor's ceiling.
    const TARGET: Target = Target::PORTABLE;
    const TILE: WgslTile = WgslTile::SHIPPING;

    #[test]
    fn the_shipping_schedule_emits_one_stage_and_two_barriers() {
        let (src, facts) = emit_wgsl(&matmul_schedule(TILE), TILE, TARGET).unwrap();
        assert_eq!(facts.stages, 1);
        assert_eq!(facts.barriers_per_k_iter, 2);
        assert_eq!(facts.threads, 64);
        // (32*16 + 16*32) * 4 = 4096.
        assert_eq!(facts.workgroup_bytes, 4096);
        assert_eq!(src.matches("workgroupBarrier()").count(), 2);
        assert!(src.contains("const STAGES: u32 = 1u;"));
        assert!(src.contains("@workgroup_size(8, 8)"));
    }

    #[test]
    fn the_pipelined_schedule_emits_rotation_and_one_barrier() {
        let (src, facts) = emit_wgsl(&matmul_pipelined_schedule(TILE, 2), TILE, TARGET).unwrap();
        assert_eq!(facts.stages, 2);
        assert_eq!(facts.barriers_per_k_iter, 1);
        assert_eq!(facts.workgroup_bytes, 8192);
        assert_eq!(src.matches("workgroupBarrier()").count(), 1);
        assert!(src.contains("t % STAGES"));
        assert!(src.contains("kt % STAGES"));
    }

    #[test]
    fn stage_depth_reaches_the_emitted_source() {
        for stages in 2..=4 {
            let (src, facts) =
                emit_wgsl(&matmul_pipelined_schedule(TILE, stages), TILE, TARGET).unwrap();
            assert!(src.contains(&format!("const STAGES: u32 = {stages}u;")));
            assert_eq!(facts.workgroup_bytes, 4096 * stages);
        }
    }

    /// Every WebGPU implementation guarantees only 16 KiB of workgroup storage.
    /// At 4 KiB per stage a 5-deep rotation busts it, and that is caught before
    /// any shader text exists rather than at pipeline creation.
    #[test]
    fn an_over_deep_rotation_busts_the_workgroup_budget() {
        // PORTABLE's floor is 32 KiB; 9 stages x 4 KiB = 36 KiB.
        let err = emit_wgsl(&matmul_pipelined_schedule(TILE, 9), TILE, TARGET).unwrap_err();
        assert!(
            matches!(&err, EmitError::Unsound(es)
                if es.iter().any(|e| matches!(
                    e, KernelScheduleError::SharedOverBudget { .. }))),
            "expected a workgroup-budget finding, got {err}"
        );
    }

    /// `Feature::AsyncCopy` is the CUDA schedule's. Reusing it here by accident
    /// must not silently produce a kernel with no async anything in it.
    #[test]
    fn async_copy_is_refused_in_wgsl() {
        let mut sched = matmul_pipelined_schedule(TILE, 2);
        sched.requires = vec![Feature::AsyncCopy];
        assert!(matches!(
            emit_wgsl(&sched, TILE, TARGET),
            Err(EmitError::UnsupportedInWgsl { .. })
        ));
    }

    #[test]
    fn an_unverifiable_schedule_is_rejected_before_emission() {
        let mut sched = matmul_schedule(TILE);
        sched.barriers[0].producers.clear();
        assert!(matches!(
            emit_wgsl(&sched, TILE, TARGET),
            Err(EmitError::Unsound(_))
        ));
    }

    /// The staging, inner product and write-back must match the shipping
    /// shader, or the baseline comparison is against a different algorithm.
    /// Read `matmul.wgsl` rather than trusting a copy — the same anti-drift
    /// rule the CUDA and Metal ports carry.
    #[test]
    fn the_emitted_body_does_not_drift_from_matmul_wgsl() {
        let (src, _) = emit_wgsl(&matmul_schedule(TILE), TILE, TARGET).unwrap();
        let shipping = crate::kernels::MATMUL_WGSL;
        for fragment in [
            "let bz_safe = select(0u, bz, in_batch);",
            "let n_tiles = (params.k + TILE_K - 1u) / TILE_K;",
            "v = v + arena[params.bias_off + global_col];",
            "arena[c_base + global_row * params.n + global_col] = v;",
        ] {
            assert!(
                shipping.contains(fragment),
                "matmul.wgsl no longer contains `{fragment}` — update the emitter"
            );
            assert!(
                src.contains(fragment),
                "the emitted shader is missing `{fragment}`"
            );
        }
        // Geometry must agree too, or the emitter is describing a kernel that
        // is not the one being measured against.
        for c in [
            "const TILE_M: u32 = 32u;",
            "const TILE_N: u32 = 32u;",
            "const TILE_K: u32 = 16u;",
        ] {
            assert!(shipping.contains(c), "matmul.wgsl geometry changed: {c}");
            assert!(src.contains(c));
        }
    }
}
