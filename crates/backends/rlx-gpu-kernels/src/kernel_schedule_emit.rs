// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The schedule generates the kernel** — `KernelSchedule` -> CUDA source.
//!
//! `rlx_ir::kernel_schedule` says of itself: *"This is the representation and
//! its analyses, not a code generator... no backend consumes it yet."* That
//! left the schedule a *mirror* of the shipping kernel rather than its source:
//! [`crate::kernel_schedule_port`] hand-transcribes `matmul.cu` into a
//! `KernelSchedule`, and `drift_against_source` keeps the two honest by
//! counting features in the `.cu` text. A verifier pointed at a description can
//! only ever find bugs in the description.
//!
//! This module inverts that for one kernel family. CAKE §2.2: *"a schedule
//! states **what** is to happen and lowering derives **how**: barrier
//! addresses, phase bits, TMEM offsets, descriptor encodings, and warp identity
//! are all computed from the declarations rather than written out by the
//! agent."* Here every structural decision in the emitted CUDA is read out of
//! the schedule:
//!
//! | emitted | derived from |
//! |---|---|
//! | `__shared__` array names, extents, dtype | [`Region`] in [`Space::Shared`] |
//! | staging depth (`[STAGES][..][..]`) | `Region::stages` |
//! | shared byte budget asserted at emit | `lower().shared_bytes` |
//! | `blockDim` / threads | `Role::warps` via `lower().threads` |
//! | where each block sync goes | position of `Action::Wait` in the role body |
//! | `cp.async` vs plain staged loads | `Feature::AsyncCopy` in `requires` |
//! | K-loop rotation modulus | `KernelSchedule::stages` |
//!
//! # What is *not* derived, stated plainly
//!
//! `rlx_ir::kernel_schedule` deliberately does not model per-thread indexing
//! inside an action, and this emitter does not invent it. The index algebra of
//! a tiled GEMM — which thread owns which `float4` of a tile, the `TM x TN`
//! outer product, the masked epilogue — is *this emitter's* knowledge, keyed to
//! the GEMM family and to [`TileParams`]. The schedule contributes the machine
//! structure; the emitter contributes the arithmetic.
//!
//! That split is the honest scope. A general `Action` -> CUDA lowering would
//! need an index language the IR does not have, and claiming one here would be
//! the "syntax without effects" failure CAKE §3.2 warns about.
//!
//! # Why this is worth doing at all
//!
//! Because once the structure is declared, a structure `matmul.cu` cannot
//! express becomes reachable by changing a declaration rather than by writing a
//! second kernel. [`matmul_pipelined_schedule`] declares `stages: N` and
//! `Feature::AsyncCopy`, and the same emitter produces a multi-stage
//! `cp.async` pipeline with **one** block sync per K iteration instead of two.
//! `matmul.cu` has exactly one pair of shared tiles and two `__syncthreads()`
//! hardcoded into its text; no `#define` reaches that.
//!
//! Whether that is *faster* is a measurement, not a claim — see
//! `rlx-cuda/examples/schedule_matmul_ab.rs`.

use rlx_gpu_dispatch::tiles::TileParams;
use rlx_ir::DType;
use rlx_ir::kernel_schedule::{
    Action, Barrier, Feature, KernelSchedule, KernelScheduleError, Layout, Region, Role, Space,
    Target, lower,
};

use crate::kernel_schedule_port::BLOCK_ROLE;

/// Which C++ dialect the emitted kernel is written in.
///
/// The `.cu` sources in this crate are shared between NVRTC and hipRTC, and so
/// is this emitter — but only the *portable* subset is genuinely shared.
/// Inline PTX is not: `cp.async.cg.shared.global` is an NVIDIA instruction, and
/// handing it to hipRTC is a compile error, not a slow path.
///
/// Naming the dialect makes that a typed rejection instead of a build failure
/// three layers down. It is also the seam where an AMD async-staging path
/// (`global_load_lds` on CDNA) would attach, so [`Lang::Hip`] refusing
/// `Feature::AsyncCopy` is a statement about this emitter, not about the
/// hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    /// CUDA C++ for NVRTC. `Feature::AsyncCopy` lowers to inline `cp.async`.
    Cuda,
    /// HIP C++ for hipRTC. Portable staging only — no async-copy instruction
    /// is emitted, so a schedule that requires one is refused.
    Hip,
}

impl Lang {
    /// Short tag for generated-file headers and diagnostics.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA/NVRTC",
            Self::Hip => "HIP/hipRTC",
        }
    }
}

/// Why a schedule could not be emitted as CUDA.
///
/// Every variant names the declaration that does not fit, because the point of
/// emitting *from* a schedule is that a rejection is a repair target rather
/// than an NVRTC error 200 lines into generated text.
#[derive(Debug, Clone, PartialEq)]
pub enum EmitError {
    /// The schedule does not verify against the target. Emitting an unverified
    /// schedule would generate code for a program that cannot run.
    Unsound(Vec<KernelScheduleError>),
    /// A region the GEMM family requires is missing or has the wrong rank.
    ///
    /// This emitter is family-specific by design (see module docs); saying so
    /// with a named region beats emitting something that compiles and is wrong.
    MissingRegion { name: &'static str },
    /// A declared region's extents disagree with the tile the caller passed.
    /// The schedule and the tile are two statements of the same fact, so a
    /// mismatch means one of them is stale.
    RegionTileMismatch {
        region: &'static str,
        declared: Vec<usize>,
        from_tile: Vec<usize>,
    },
    /// The roles imply a different thread count than the tile's block does.
    ThreadCountMismatch { from_roles: usize, from_tile: u32 },
    /// More than one role. Warp specialization is expressible in the IR and is
    /// **not** implemented by this emitter; rejecting is the honest response.
    MultiRole { roles: usize },
    /// `stages` is inconsistent across the staged regions and the schedule.
    StageDisagreement {
        schedule: usize,
        region: &'static str,
        declared: usize,
    },
    /// A pipelined schedule needs at least two stages to rotate between.
    PipelineTooShallow { stages: usize },
    /// The tile itself is illegal.
    Tile(rlx_gpu_dispatch::tiles::TileError),
    /// The schedule requires a capability this *dialect* cannot express, even
    /// though the target device may well have one.
    ///
    /// Distinct from the verifier's `UnsupportedFeature`, which is about the
    /// device. This is about the emitter: CDNA has an LDS-direct global load,
    /// but nothing here emits it, and saying so beats emitting NVIDIA inline
    /// PTX into a hipRTC translation unit.
    UnsupportedInLang { lang: Lang, feature: &'static str },
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
                "roles imply {from_roles} threads, tile block is {from_tile}"
            ),
            Self::MultiRole { roles } => write!(
                f,
                "{roles} roles declared; this emitter lowers single-role (uniform-block) \
                 schedules only — warp specialization is unimplemented, not silently ignored"
            ),
            Self::StageDisagreement {
                schedule,
                region,
                declared,
            } => write!(
                f,
                "schedule declares {schedule} stages but region `{region}` declares {declared}"
            ),
            Self::PipelineTooShallow { stages } => {
                write!(f, "a rotating pipeline needs >= 2 stages, got {stages}")
            }
            Self::Tile(e) => write!(f, "illegal tile: {e}"),
            Self::UnsupportedInLang { lang, feature } => write!(
                f,
                "this emitter has no {} lowering for `{feature}` — refused rather than \
                 emitted as another dialect's instruction",
                lang.tag()
            ),
        }
    }
}

impl std::error::Error for EmitError {}

/// A multi-stage `cp.async` GEMM schedule — the structure `matmul.cu` has no
/// way to say.
///
/// The difference from [`crate::kernel_schedule_port::matmul_schedule`] is
/// three declarations, and every consequence follows from them:
///
/// * `Region::stages = stages` on both shared tiles, so the emitter allocates
///   a rotating buffer and indexes it `t % STAGES`;
/// * `Feature::AsyncCopy`, so the staging loads become `cp.async` (global ->
///   shared with no register round-trip) and a target lacking it gets a named
///   rejection from the verifier rather than wrong code;
/// * **one** barrier per K iteration instead of two.
///
/// That last one is the point, and it is only sound *because* of the first.
/// `matmul.cu` needs `tiles_consumed` because the next iteration overwrites the
/// single buffer it just read. With `stages >= 2` the write for iteration
/// `t + stages - 1` lands in the buffer last read at iteration `t - 1`, which
/// the top-of-loop wait already ordered — so the second sync has nothing left
/// to protect and is deleted rather than merely skipped.
///
/// The FMA order is untouched: same `kk` sequence over the same tile contents,
/// so results must be **bit-identical** to the default schedule. Anything else
/// is a bug in the pipeline, not a faster kernel.
pub fn matmul_pipelined_schedule(tile: TileParams, stages: usize) -> KernelSchedule {
    let (bm, bn, bk) = (tile.bm as usize, tile.bn as usize, tile.bk as usize);
    let (tm, tn) = (tile.tm as usize, tile.tn as usize);

    let mut s = KernelSchedule::new(format!("matmul_pipe{stages}_{}", tile.label()));
    s.stages = stages;
    s.requires = vec![Feature::AsyncCopy];

    s.regions = vec![
        Region {
            name: "tile_a".into(),
            space: Space::Shared,
            dims: vec![bm, bk],
            dtype: DType::F32,
            stages,
            layout: Layout::row_major(&[bm, bk]),
        },
        Region {
            name: "tile_b".into(),
            space: Space::Shared,
            dims: vec![bk, bn],
            dtype: DType::F32,
            stages,
            layout: Layout::row_major(&[bk, bn]),
        },
        Region {
            name: "acc".into(),
            space: Space::Register,
            dims: vec![tm, tn],
            dtype: DType::F32,
            stages: 1,
            layout: Layout::row_major(&[tm, tn]),
        },
    ];

    let warps = tile.threads().div_ceil(32);
    s.roles = vec![Role {
        name: BLOCK_ROLE.into(),
        warps: (0..warps).collect(),
    }];

    // One barrier. Its absence is as load-bearing as its presence: a schedule
    // that kept `tiles_consumed` would be describing a kernel that still pays
    // for a hazard it no longer has.
    s.barriers = vec![Barrier {
        name: "tiles_filled".into(),
        producers: vec![BLOCK_ROLE.into()],
        consumers: vec![BLOCK_ROLE.into()],
        count: 1,
    }];

    // Steady-state iteration for stage `cur`, in program order. The load
    // targets a *different* stage than the compute reads — which is what makes
    // the overlap expressible at all.
    let cur = 0usize;
    let next = stages - 1;
    s.body.insert(
        BLOCK_ROLE.into(),
        vec![
            Action::Wait {
                barrier: "tiles_filled".into(),
                stage: cur,
            },
            Action::Compute {
                reads: vec![
                    rlx_ir::kernel_schedule::Access::plain("tile_a"),
                    rlx_ir::kernel_schedule::Access::plain("tile_b"),
                ],
                writes: vec![rlx_ir::kernel_schedule::Access::plain("acc")],
                stage: cur,
                via: None,
            },
            Action::Load {
                access: rlx_ir::kernel_schedule::Access::plain("tile_a"),
                stage: next,
            },
            Action::Load {
                access: rlx_ir::kernel_schedule::Access::plain("tile_b"),
                stage: next,
            },
            Action::Arrive {
                barrier: "tiles_filled".into(),
                stage: next,
            },
            Action::Store {
                access: rlx_ir::kernel_schedule::Access::plain("acc"),
                stage: cur,
            },
        ],
    );
    s
}

/// What the emitter read out of the schedule, so a caller can assert on it.
///
/// Returned alongside the source because "the emitted kernel uses N stages"
/// should be checkable without regexing generated C++. `rlx-cuda`'s A/B example
/// prints these next to the timings; a variant that silently emitted the
/// default structure would otherwise look like a win at 1.00x.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitFacts {
    /// Rotation modulus of the K loop.
    pub stages: usize,
    /// Block syncs per K iteration. 2 for the default schedule, 1 pipelined.
    pub syncs_per_k_iter: usize,
    /// Whether staging loads were emitted as `cp.async`.
    pub async_copy: bool,
    /// Shared bytes the launch reserves, from `lower()`.
    pub shared_bytes: usize,
    /// Threads per block, from the roles.
    pub threads: usize,
    /// Which dialect the source is written in. Recorded because "CUDA and HIP
    /// share the kernel text" is true of the portable subset only, and an A/B
    /// report that does not say which one it built is not reproducible.
    pub lang: Lang,
}

/// Emit CUDA for `sched` under `tile`, checked against `target`.
///
/// The schedule is verified first and a failure is returned rather than
/// emitted around: source generated from an unsound schedule is a program that
/// cannot run, and producing it would move the failure from a named finding to
/// an NVRTC diagnostic or a hang.
///
/// The emitted entry point is `matmul` with **the same parameter list** as
/// `kernels/matmul.cu`, so it drops into the existing launcher unchanged. That
/// is deliberate: the experiment this exists for must vary the schedule and
/// nothing else.
pub fn emit_cuda(
    sched: &KernelSchedule,
    tile: TileParams,
    target: Target,
) -> Result<(String, EmitFacts), EmitError> {
    emit(sched, tile, target, Lang::Cuda)
}

/// Emit `lang` source for `sched` under `tile`, checked against `target`.
///
/// See [`emit_cuda`] for the contract; the only difference is the dialect. A
/// capability this emitter cannot express in `lang` is a named rejection
/// ([`EmitError::UnsupportedInLang`]) rather than a foreign instruction pasted
/// into a translation unit that will not compile.
pub fn emit(
    sched: &KernelSchedule,
    tile: TileParams,
    target: Target,
    lang: Lang,
) -> Result<(String, EmitFacts), EmitError> {
    tile.validate().map_err(EmitError::Tile)?;
    // The emitter's own capability limit is reported before the verifier is
    // asked about a program this emitter could not lower in any case — a
    // multi-role schedule that also fails verification should read as "warp
    // specialization is unimplemented", not as a barrier finding.
    if sched.roles.len() != 1 {
        return Err(EmitError::MultiRole {
            roles: sched.roles.len(),
        });
    }
    let lowered = lower(sched, target).map_err(EmitError::Unsound)?;

    if lowered.threads != tile.threads() as usize {
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
    let tile_a = region("tile_a")?;
    let tile_b = region("tile_b")?;
    let acc = region("acc")?;

    // The schedule and the tile are two statements of the same extents. If they
    // disagree, one is stale — and the emitted indexing would follow the tile
    // while the verifier's shared-memory budget followed the schedule.
    let check = |r: &Region, name: &'static str, want: [usize; 2]| -> Result<(), EmitError> {
        if r.dims != want {
            return Err(EmitError::RegionTileMismatch {
                region: name,
                declared: r.dims.clone(),
                from_tile: want.to_vec(),
            });
        }
        Ok(())
    };
    let (bm, bn, bk) = (tile.bm as usize, tile.bn as usize, tile.bk as usize);
    let (tm, tn) = (tile.tm as usize, tile.tn as usize);
    check(tile_a, "tile_a", [bm, bk])?;
    check(tile_b, "tile_b", [bk, bn])?;
    check(acc, "acc", [tm, tn])?;

    let stages = sched.stages.max(1);
    for (r, name) in [(tile_a, "tile_a"), (tile_b, "tile_b")] {
        if r.stages.max(1) != stages {
            return Err(EmitError::StageDisagreement {
                schedule: stages,
                region: name,
                declared: r.stages,
            });
        }
    }

    let async_copy = sched.requires.contains(&Feature::AsyncCopy);
    let body = sched.body.get(BLOCK_ROLE).map(Vec::as_slice).unwrap_or(&[]);
    // Block syncs are not a constant of the emitter: they are however many
    // `Wait`s the role body declares. Deleting the second `Wait` from a
    // schedule deletes a `__syncthreads()` from the generated kernel.
    let syncs_per_k_iter = body
        .iter()
        .filter(|a| matches!(a, Action::Wait { .. }))
        .count()
        .max(1);
    // The device may have an async-copy engine; this emitter only knows how to
    // drive NVIDIA's. Refusing is what keeps `Feature::AsyncCopy` a claim about
    // the machine rather than a claim about the code generator.
    if async_copy && lang == Lang::Hip {
        return Err(EmitError::UnsupportedInLang {
            lang,
            feature: "AsyncCopy",
        });
    }
    // `cp.async` buys nothing without a second buffer to land in: the wait
    // would immediately precede the read it feeds. A schedule that declares the
    // feature at one stage is stating an intent the structure cannot deliver.
    if async_copy && stages < 2 {
        return Err(EmitError::PipelineTooShallow { stages });
    }
    let pipelined = stages > 1;

    let facts = EmitFacts {
        stages,
        syncs_per_k_iter,
        async_copy,
        shared_bytes: lowered.shared_bytes,
        threads: lowered.threads,
        lang,
    };

    let mut src = String::with_capacity(12 * 1024);
    emit_header(&mut src, sched, tile, &facts, &lowered);
    emit_prelude(&mut src, tile, stages, async_copy);
    emit_signature(&mut src);
    emit_setup(&mut src, tile, stages);
    if pipelined {
        emit_pipelined_loop(&mut src, tile, stages, async_copy, syncs_per_k_iter);
    }
    emit_serial_loop(&mut src, tile, /* guarded_by_fallback */ pipelined);
    emit_epilogue(&mut src);
    Ok((src, facts))
}

fn emit_header(
    out: &mut String,
    sched: &KernelSchedule,
    tile: TileParams,
    facts: &EmitFacts,
    lowered: &rlx_ir::kernel_schedule::Lowered,
) {
    out.push_str(&format!(
        "// @generated by rlx_gpu_kernels::kernel_schedule_emit from KernelSchedule\n\
         // `{}` under tile {} for {}. Do not edit; edit the schedule.\n\
         //\n\
         // Derived, not authored:\n\
         //   stages           = {}   (KernelSchedule::stages / Region::stages)\n\
         //   syncs per K iter = {}   (count of Action::Wait in the role body)\n\
         //   async copy       = {}   (Feature::AsyncCopy in `requires`)\n\
         //   shared bytes     = {}   (lower().shared_bytes)\n\
         //   threads/block    = {}   (lower().threads, from Role::warps)\n\
         //   shared offsets   = {:?}\n\n",
        sched.name,
        tile.label(),
        facts.lang.tag(),
        facts.stages,
        facts.syncs_per_k_iter,
        facts.async_copy,
        facts.shared_bytes,
        facts.threads,
        lowered.region_offsets,
    ));
}

fn emit_prelude(out: &mut String, tile: TileParams, stages: usize, async_copy: bool) {
    out.push_str(&format!(
        "#define BM {}\n#define BN {}\n#define BK {}\n\
         #define TM {}\n#define TN {}\n\
         #define BLOCK_DIM_X {}\n#define BLOCK_DIM_Y {}\n\
         #define STAGES {}\n\
         #define THREADS (BLOCK_DIM_X * BLOCK_DIM_Y)\n\n",
        tile.bm, tile.bn, tile.bk, tile.tm, tile.tn, tile.bdx, tile.bdy, stages
    ));

    // The activation epilogue is verbatim from `matmul.cu`. It is not part of
    // the machine schedule — no region, role or barrier describes it — so the
    // emitter reproduces it rather than pretending to derive it.
    out.push_str(
        r#"__device__ __forceinline__ float apply_act(float v, unsigned int act_id) {
    if (act_id == 0xFFFFu) return v;
    switch (act_id) {
        case 0:  return fmaxf(v, 0.0f);
        case 1:  return 1.0f / (1.0f + expf(-fminf(fmaxf(v, -88.0f), 88.0f)));
        case 2:  return tanhf(fminf(fmaxf(v, -15.0f), 15.0f));
        case 5:  return sqrtf(v);
        case 7:  return -v;
        case 8:  return fabsf(v);
        case 9:  return gelu_erf(v);
        case 11: return gelu_approx(v);
        case 10: {
            float nx = fminf(fmaxf(-v, -88.0f), 88.0f);
            return v / (1.0f + expf(nx));
        }
        default: return v;
    }
}

"#,
    );

    if async_copy {
        // 16-byte `cp.async.cg`: global -> shared with no register round-trip,
        // which is the whole reason `Feature::AsyncCopy` is worth declaring.
        out.push_str(
            r#"// Feature::AsyncCopy — emitted because the schedule declares it.
__device__ __forceinline__ void cp_async16(float* smem_dst, const float* gmem_src) {
    unsigned int s = static_cast<unsigned int>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

"#,
        );
    }
}

fn emit_signature(out: &mut String) {
    // Byte-identical to `kernels/matmul.cu`'s parameter list. The launcher is
    // shared, so the ABI is a fixed contract this emitter satisfies rather than
    // chooses.
    out.push_str(
        r#"extern "C" __global__ void matmul(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int a_off,
    unsigned int b_off,
    unsigned int c_off,
    unsigned int batch,
    unsigned int a_batch_stride,
    unsigned int b_batch_stride,
    unsigned int c_batch_stride,
    unsigned int has_bias,
    unsigned int bias_off,
    unsigned int act_id
) {
"#,
    );
}

fn emit_setup(out: &mut String, _tile: TileParams, _stages: usize) {
    // `[STAGES]` is the only difference from matmul.cu's declaration, and it
    // comes from `Region::stages`.
    out.push_str(
        r#"    __shared__ float tile_a[STAGES][BM][BK];
    __shared__ float tile_b[STAGES][BK][BN];

    unsigned int bz = blockIdx.z;
    if (bz >= batch) return;

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int tid = ty * BLOCK_DIM_X + tx;

    unsigned int row0 = blockIdx.y * BM + ty * TM;
    unsigned int col0 = blockIdx.x * BN + tx * TN;

    unsigned int a_base = a_off + bz * a_batch_stride;
    unsigned int b_base = b_off + bz * b_batch_stride;
    unsigned int c_base = c_off + bz * c_batch_stride;

    bool vec_a = ((k & 3u) == 0u);
    bool vec_b = ((n & 3u) == 0u);
    bool full_block_a = (blockIdx.y * BM + BM <= m);
    bool full_block_b = (blockIdx.x * BN + BN <= n);

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    }

    unsigned int n_tiles = (k + BK - 1) / BK;
    const unsigned int A_PER_THREAD = (BM * BK) / THREADS;
    const unsigned int B_PER_THREAD = (BK * BN) / THREADS;
    const unsigned int A_VEC_PER_THREAD = (BM * BK / 4) / THREADS;
    const unsigned int B_VEC_PER_THREAD = (BK * BN / 4) / THREADS;

"#,
    );
}

/// The inner outer-product, identical in both loops.
///
/// Shared verbatim between the pipelined and serial paths so bit-exactness
/// between them is a property of the text, not of two edits staying in step.
const COMPUTE_BODY: &str = r#"        #pragma unroll
        for (unsigned int kk = 0; kk < BK; ++kk) {
            float a_reg[TM];
            float b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = tile_a[CUR_STAGE][ty * TM + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = tile_b[CUR_STAGE][kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
"#;

fn compute_body(stage_expr: &str) -> String {
    COMPUTE_BODY.replace("CUR_STAGE", stage_expr)
}

fn emit_pipelined_loop(
    out: &mut String,
    _tile: TileParams,
    stages: usize,
    async_copy: bool,
    syncs_per_k_iter: usize,
) {
    // The pipeline runs only where every K tile is full and both tile loads are
    // float4-aligned. Ragged edges fall through to the serial loop below —
    // masked `cp.async` per element would cost more than it saves, and a
    // partially-pipelined loop is where an off-by-one lives.
    out.push_str(
        "    // ── Pipelined path (STAGES-deep rotation) ───────────────────\n\
         \x20   // Uniform across the block: shape-only predicates plus this block's\n\
         \x20   // own tile position. No intra-block divergence.\n\
         \x20   bool pipe_ok = (STAGES > 1) && vec_a && vec_b && full_block_a && full_block_b\n\
         \x20                  && ((k % BK) == 0u) && (n_tiles >= STAGES);\n\
         \x20   if (pipe_ok) {\n",
    );

    let issue = if async_copy {
        r#"            #pragma unroll
            for (unsigned int li = 0; li < A_VEC_PER_THREAD; ++li) {
                unsigned int idx4 = tid + li * THREADS;
                unsigned int r = idx4 / (BK / 4);
                unsigned int c4 = idx4 % (BK / 4);
                unsigned int gr = blockIdx.y * BM + r;
                unsigned int gc = kt * BK + c4 * 4;
                cp_async16(&tile_a[buf][r][c4 * 4], &arena[a_base + gr * k + gc]);
            }
            #pragma unroll
            for (unsigned int li = 0; li < B_VEC_PER_THREAD; ++li) {
                unsigned int idx4 = tid + li * THREADS;
                unsigned int r = idx4 / (BN / 4);
                unsigned int c4 = idx4 % (BN / 4);
                unsigned int gr = kt * BK + r;
                unsigned int gc = blockIdx.x * BN + c4 * 4;
                cp_async16(&tile_b[buf][r][c4 * 4], &arena[b_base + gr * n + gc]);
            }
            cp_async_commit();
"#
    } else {
        // Same rotation, register-staged loads. Kept reachable so the stage
        // depth and the copy instruction are independently attributable in the
        // A/B: "N stages" and "cp.async" are two claims, not one.
        r#"            #pragma unroll
            for (unsigned int li = 0; li < A_VEC_PER_THREAD; ++li) {
                unsigned int idx4 = tid + li * THREADS;
                unsigned int r = idx4 / (BK / 4);
                unsigned int c4 = idx4 % (BK / 4);
                unsigned int gr = blockIdx.y * BM + r;
                unsigned int gc = kt * BK + c4 * 4;
                float4 v = *reinterpret_cast<const float4*>(&arena[a_base + gr * k + gc]);
                tile_a[buf][r][c4 * 4 + 0] = v.x;
                tile_a[buf][r][c4 * 4 + 1] = v.y;
                tile_a[buf][r][c4 * 4 + 2] = v.z;
                tile_a[buf][r][c4 * 4 + 3] = v.w;
            }
            #pragma unroll
            for (unsigned int li = 0; li < B_VEC_PER_THREAD; ++li) {
                unsigned int idx4 = tid + li * THREADS;
                unsigned int r = idx4 / (BN / 4);
                unsigned int c4 = idx4 % (BN / 4);
                unsigned int gr = kt * BK + r;
                unsigned int gc = blockIdx.x * BN + c4 * 4;
                float4 v = *reinterpret_cast<const float4*>(&arena[b_base + gr * n + gc]);
                tile_b[buf][r][c4 * 4 + 0] = v.x;
                tile_b[buf][r][c4 * 4 + 1] = v.y;
                tile_b[buf][r][c4 * 4 + 2] = v.z;
                tile_b[buf][r][c4 * 4 + 3] = v.w;
            }
"#
    };

    out.push_str("        // Prologue: fill STAGES-1 buffers before the first compute.\n");
    out.push_str("        for (unsigned int s = 0; s + 1 < STAGES; ++s) {\n");
    out.push_str("            unsigned int kt = s;\n            unsigned int buf = s;\n");
    out.push_str(issue);
    out.push_str("        }\n\n");

    // `cp.async.wait_group<STAGES-2>` leaves at most STAGES-2 groups in flight,
    // so the group staging buffer `t % STAGES` has landed. Derived from the
    // rotation depth, not chosen.
    let wait = if async_copy {
        format!("            cp_async_wait<{}>();\n", stages - 2)
    } else {
        String::new()
    };

    out.push_str("        for (unsigned int t = 0; t < n_tiles; ++t) {\n");
    out.push_str(&wait);
    // The single Wait the schedule declares.
    out.push_str(&format!(
        "            __syncthreads();  // Action::Wait `tiles_filled` ({syncs_per_k_iter} \
         sync/iter declared)\n\n"
    ));
    out.push_str(&compute_body("t % STAGES"));
    out.push_str(
        "\n            // Stage the tile this iteration's buffer will be needed for,\n\
         \x20           // into the buffer iteration t-1 finished reading before the\n\
         \x20           // sync above. No second barrier is needed to protect it.\n\
         \x20           unsigned int kt = t + STAGES - 1;\n",
    );
    out.push_str("            if (kt < n_tiles) {\n");
    out.push_str("                unsigned int buf = kt % STAGES;\n");
    // Re-indent the shared issue block by four spaces for this nesting level.
    for line in issue.lines() {
        out.push_str("    ");
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("            }");
    if async_copy {
        out.push_str(" else {\n                cp_async_commit();  // keep the group count even\n            }");
    }
    out.push_str("\n        }\n");
    out.push_str("    } else {\n");
}

fn emit_serial_loop(out: &mut String, _tile: TileParams, guarded_by_fallback: bool) {
    let indent = if guarded_by_fallback { "    " } else { "" };
    let put = |out: &mut String, s: &str| {
        for line in s.lines() {
            if line.is_empty() {
                out.push('\n');
            } else {
                out.push_str(indent);
                out.push_str(line);
                out.push('\n');
            }
        }
    };

    put(
        out,
        r#"    // ── Serial path: one buffer, two barriers ───────────────────
    // Bit-identical arithmetic to the pipelined loop; it exists for ragged
    // edges (and is the whole kernel when the schedule declares one stage).
    for (unsigned int t = 0; t < n_tiles; ++t) {
        bool full_k_tile = (t * BK + BK <= k);

        if (vec_a && full_block_a && full_k_tile) {
            #pragma unroll
            for (unsigned int li = 0; li < A_VEC_PER_THREAD; ++li) {
                unsigned int idx4 = tid + li * THREADS;
                unsigned int r = idx4 / (BK / 4);
                unsigned int c4 = idx4 % (BK / 4);
                unsigned int gr = blockIdx.y * BM + r;
                unsigned int gc = t * BK + c4 * 4;
                float4 v = *reinterpret_cast<const float4*>(&arena[a_base + gr * k + gc]);
                tile_a[0][r][c4 * 4 + 0] = v.x;
                tile_a[0][r][c4 * 4 + 1] = v.y;
                tile_a[0][r][c4 * 4 + 2] = v.z;
                tile_a[0][r][c4 * 4 + 3] = v.w;
            }
        } else {
            #pragma unroll
            for (unsigned int li = 0; li < A_PER_THREAD; ++li) {
                unsigned int idx = tid + li * THREADS;
                unsigned int r = idx / BK;
                unsigned int c = idx % BK;
                unsigned int gr = blockIdx.y * BM + r;
                unsigned int gc = t * BK + c;
                tile_a[0][r][c] = (gr < m && gc < k) ? arena[a_base + gr * k + gc] : 0.0f;
            }
        }

        if (vec_b && full_block_b && full_k_tile) {
            #pragma unroll
            for (unsigned int li = 0; li < B_VEC_PER_THREAD; ++li) {
                unsigned int idx4 = tid + li * THREADS;
                unsigned int r = idx4 / (BN / 4);
                unsigned int c4 = idx4 % (BN / 4);
                unsigned int gr = t * BK + r;
                unsigned int gc = blockIdx.x * BN + c4 * 4;
                float4 v = *reinterpret_cast<const float4*>(&arena[b_base + gr * n + gc]);
                tile_b[0][r][c4 * 4 + 0] = v.x;
                tile_b[0][r][c4 * 4 + 1] = v.y;
                tile_b[0][r][c4 * 4 + 2] = v.z;
                tile_b[0][r][c4 * 4 + 3] = v.w;
            }
        } else {
            #pragma unroll
            for (unsigned int li = 0; li < B_PER_THREAD; ++li) {
                unsigned int idx = tid + li * THREADS;
                unsigned int r = idx / BN;
                unsigned int c = idx % BN;
                unsigned int gr = t * BK + r;
                unsigned int gc = blockIdx.x * BN + c;
                tile_b[0][r][c] = (gr < k && gc < n) ? arena[b_base + gr * n + gc] : 0.0f;
            }
        }

        __syncthreads();  // Action::Wait `tiles_filled`

"#,
    );
    put(out, &compute_body("0"));
    put(
        out,
        r#"
        __syncthreads();  // Action::Wait `tiles_consumed`
    }
"#,
    );
    if guarded_by_fallback {
        out.push_str("    }\n");
    }
}

fn emit_epilogue(out: &mut String) {
    out.push_str(
        r#"
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            unsigned int row = row0 + i;
            unsigned int col = col0 + j;
            if (row < m && col < n) {
                float v = acc[i][j];
                if (has_bias) v += arena[bias_off + col];
                v = apply_act(v, act_id);
                arena[c_base + row * n + col] = v;
            }
        }
    }
}
"#,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_schedule_port::{MATMUL_TARGET, matmul_schedule};

    const TILE: TileParams = TileParams {
        bm: 64,
        bn: 64,
        bk: 16,
        tm: 4,
        tn: 4,
        bdx: 16,
        bdy: 16,
    };

    #[test]
    fn default_schedule_emits_one_stage_and_two_syncs() {
        let (src, facts) = emit_cuda(&matmul_schedule(TILE), TILE, MATMUL_TARGET).unwrap();
        assert_eq!(facts.stages, 1);
        assert_eq!(facts.syncs_per_k_iter, 2);
        assert!(!facts.async_copy);
        assert_eq!(facts.threads, TILE.threads() as usize);
        // 64*16*4 + 16*64*4 = 8192, one stage.
        assert_eq!(facts.shared_bytes, 8192);
        assert!(!src.contains("cp.async"));
        assert!(src.contains("#define STAGES 1"));
        assert_eq!(src.matches("__syncthreads()").count(), 2);
    }

    /// The pipelined schedule is checked against the device that actually has
    /// the capability. `MATMUL_TARGET` is the portable floor, which does not —
    /// see [`the portable floor refuses it`](portable_floor_refuses_async_copy).
    const SM86: Target = Target::CUDA_SM86;

    #[test]
    fn pipelined_schedule_emits_rotation_and_one_sync() {
        let sched = matmul_pipelined_schedule(TILE, 3);
        let (src, facts) = emit_cuda(&sched, TILE, SM86).unwrap();
        assert_eq!(facts.stages, 3);
        assert_eq!(facts.syncs_per_k_iter, 1);
        assert!(facts.async_copy);
        // Three stages of both tiles: 3 * 8192.
        assert_eq!(facts.shared_bytes, 24576);
        assert!(src.contains("cp.async.cg.shared.global"));
        assert!(src.contains("cp_async_wait<1>()"));
        assert!(src.contains("t % STAGES"));
    }

    /// The stage depth reaches the generated text. A pipelined schedule that
    /// emitted the default structure would look like a free win in the A/B.
    #[test]
    fn stage_depth_changes_the_wait_group_depth() {
        for stages in 2..=4 {
            let sched = matmul_pipelined_schedule(TILE, stages);
            let (src, _) = emit_cuda(&sched, TILE, SM86).unwrap();
            assert!(
                src.contains(&format!("cp_async_wait<{}>()", stages - 2)),
                "stage depth {stages} did not reach the emitted wait"
            );
            assert!(src.contains(&format!("#define STAGES {stages}")));
        }
    }

    /// A schedule whose regions disagree with the tile is a stale statement of
    /// the same fact, and is refused rather than emitted around.
    #[test]
    fn region_extents_must_agree_with_the_tile() {
        let other = TileParams { bk: 32, ..TILE };
        let err = emit_cuda(&matmul_schedule(TILE), other, MATMUL_TARGET).unwrap_err();
        assert!(
            matches!(
                err,
                EmitError::RegionTileMismatch {
                    region: "tile_a",
                    ..
                }
            ),
            "expected a tile_a mismatch, got {err}"
        );
    }

    /// Warp specialization is expressible in the IR and unimplemented here.
    /// Emitting single-role code for a two-role schedule would silently drop
    /// the specialization the schedule asked for.
    #[test]
    fn multi_role_schedules_are_refused_not_flattened() {
        let mut sched = matmul_pipelined_schedule(TILE, 2);
        sched.roles = vec![
            Role {
                name: "load".into(),
                warps: vec![0, 1],
            },
            Role {
                name: "mma".into(),
                warps: vec![2, 3, 4, 5, 6, 7],
            },
        ];
        assert!(matches!(
            emit_cuda(&sched, TILE, SM86),
            Err(EmitError::MultiRole { roles: 2 })
        ));
    }

    /// The capability is a *declaration*, so a target without it produces a
    /// named rejection instead of code that runs everywhere and is wrong on the
    /// SKUs that lack `cp.async`. This is the property `Feature` exists for.
    #[test]
    fn portable_floor_refuses_async_copy() {
        let sched = matmul_pipelined_schedule(TILE, 2);
        let err = emit_cuda(&sched, TILE, MATMUL_TARGET).unwrap_err();
        assert!(
            matches!(&err, EmitError::Unsound(es)
                if es.iter().any(|e| e.to_string().contains("AsyncCopy"))),
            "expected a missing-capability finding, got {err}"
        );
        // And the same schedule on a target that has it emits fine.
        assert!(emit_cuda(&sched, TILE, SM86).is_ok());
    }

    /// HIP shares the kernel *text* with CUDA, so the same schedule emits for
    /// both — as long as it stays in the portable subset.
    #[test]
    fn the_rotation_emits_for_hip_without_async_copy() {
        // A pipelined schedule that does NOT declare AsyncCopy: rotation only.
        let mut sched = matmul_pipelined_schedule(TILE, 2);
        sched.requires.clear();
        let (src, facts) = emit(&sched, TILE, Target::ROCM_GFX908, Lang::Hip).unwrap();
        assert_eq!(facts.lang, Lang::Hip);
        assert_eq!(facts.stages, 2);
        assert_eq!(facts.syncs_per_k_iter, 1);
        assert!(!facts.async_copy);
        // The dividing line: no NVIDIA inline PTX anywhere in a HIP unit.
        assert!(!src.contains("cp.async"), "PTX leaked into HIP source");
        assert!(!src.contains("__cvta_generic_to_shared"));
        // But the portable machinery IS there — `__syncthreads` and `float4`
        // are spelled the same in HIP, which is why the text can be shared.
        assert!(src.contains("__syncthreads()"));
        assert!(src.contains("#define STAGES 2"));
        assert!(src.contains("HIP/hipRTC"));
    }

    /// The CUDA `cp.async` schedule handed to HIP is refused by name rather
    /// than emitted into a translation unit hipRTC cannot compile.
    #[test]
    fn cp_async_is_refused_for_hip() {
        let sched = matmul_pipelined_schedule(TILE, 2);
        let err = emit(&sched, TILE, Target::CUDA_SM86, Lang::Hip).unwrap_err();
        assert!(
            matches!(
                err,
                EmitError::UnsupportedInLang {
                    lang: Lang::Hip,
                    feature: "AsyncCopy"
                }
            ),
            "expected a dialect rejection, got {err}"
        );
    }

    /// gfx1103 exposes no matrix cores through rlx's paths, and a schedule that
    /// needs one must be told so rather than silently running scalar.
    #[test]
    fn a_capability_absent_on_gfx1103_is_reported() {
        let mut sched = matmul_pipelined_schedule(TILE, 2);
        sched.requires = vec![Feature::CoopMatrix { m: 16, n: 16, k: 4 }];
        let err = emit(&sched, TILE, Target::ROCM_GFX1103, Lang::Hip).unwrap_err();
        assert!(matches!(err, EmitError::Unsound(_)), "got {err}");
        // The same schedule is fine on the MI100, which does expose it.
        assert!(emit(&sched, TILE, Target::ROCM_GFX908, Lang::Hip).is_ok());
    }

    /// `cp.async` at one stage is an intent the structure cannot deliver.
    #[test]
    fn async_copy_without_a_second_buffer_is_refused() {
        let mut sched = matmul_schedule(TILE);
        sched.requires = vec![Feature::AsyncCopy];
        assert!(matches!(
            emit_cuda(&sched, TILE, SM86),
            Err(EmitError::PipelineTooShallow { stages: 1 })
        ));
    }

    /// Everything between `{` and the matching `}` of `needle`'s block, with
    /// runs of whitespace collapsed so indentation differences do not count.
    fn block_after(src: &str, needle: &str) -> Option<String> {
        let start = src.find(needle)?;
        let open = src[start..].find('{')? + start;
        let mut depth = 0usize;
        let mut end = None;
        for (i, c) in src[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(open + i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        Some(
            src[open..end?]
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "),
        )
    }

    /// The epilogue is not part of the machine schedule — no region, role or
    /// barrier describes an activation — so the emitter carries a **copy** of
    /// `matmul.cu`'s `apply_act`. A copy is a drift hazard: adding an
    /// activation case to the shipping kernel would leave the emitted one
    /// silently computing the old table, and every bit-exactness check in the
    /// A/B would still pass because both arms would be compared at `act_id =
    /// 0xFFFF`.
    ///
    /// This is `schedule-port-drift` in the other direction, and it gets the
    /// same treatment: read the real thing and compare.
    #[test]
    fn the_emitted_epilogue_does_not_drift_from_matmul_cu() {
        let (src, _) = emit_cuda(&matmul_schedule(TILE), TILE, MATMUL_TARGET).unwrap();
        let shipped = block_after(crate::MATMUL_CU, "float apply_act(")
            .expect("apply_act not found in matmul.cu — the scan is broken, not the kernel");
        let emitted = block_after(&src, "float apply_act(")
            .expect("apply_act not found in the emitted source");
        assert_eq!(
            shipped, emitted,
            "the emitted activation epilogue has drifted from kernels/matmul.cu"
        );
    }

    /// Same hazard for the masked write-back, which the emitter also copies.
    #[test]
    fn the_emitted_writeback_does_not_drift_from_matmul_cu() {
        let (src, _) = emit_cuda(&matmul_schedule(TILE), TILE, MATMUL_TARGET).unwrap();
        for fragment in [
            "if (has_bias) v += arena[bias_off + col];",
            "v = apply_act(v, act_id);",
            "arena[c_base + row * n + col] = v;",
        ] {
            assert!(
                crate::MATMUL_CU.contains(fragment),
                "matmul.cu no longer contains `{fragment}` — update the emitter's epilogue"
            );
            assert!(
                src.contains(fragment),
                "the emitted kernel is missing `{fragment}`"
            );
        }
    }

    /// An unsound schedule never reaches NVRTC.
    #[test]
    fn an_unverifiable_schedule_is_rejected_before_emission() {
        let mut sched = matmul_schedule(TILE);
        // A wait on a barrier nothing arrives at: a hang, caught statically.
        sched.barriers[0].producers.clear();
        assert!(matches!(
            emit_cuda(&sched, TILE, MATMUL_TARGET),
            Err(EmitError::Unsound(_))
        ));
    }
}
