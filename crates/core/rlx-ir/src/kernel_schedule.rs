// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **A typed, hardware-explicit schedule representation.**
//!
//! rlx's `Op` graph says *what* to compute. Nothing in the tree says *how the
//! machine is driven* — which thread groups take which roles, how deeply a
//! buffer is staged, which barrier gates which handoff. That lives inside
//! hand-written MSL/CUDA/WGSL text, where no analysis can reach it.
//!
//! The cost of that shows up as a specific, recurring failure mode: every
//! synchronization bug in this tree has had to be bisected by hand, which is
//! why `RLX_VULKAN_NOBARRIER`, `RLX_VULKAN_FULLBARRIER`,
//! `RLX_METAL_CONCURRENT_NOBARRIER` and `RLX_WGPU_ONE_OP_PER_PASS` all exist.
//! Those are not features; they are the debugging tools you need when barriers
//! are invisible to the compiler.
//!
//! CAKE §2.2 states the alternative: "a schedule states *what* is to happen and
//! lowering derives *how*: barrier addresses, phase bits, TMEM offsets,
//! descriptor encodings, and warp identity are all computed from the
//! declarations rather than written out by the agent." Four properties do the
//! work — a type-checked vocabulary, declared resources, explicit roles, and
//! auto-derived metadata.
//!
//! # "Schedule" is overloaded — this is the intra-kernel sense
//!
//! Three things in this tree carry the name, at two different levels:
//!
//! | name | scope | meaning |
//! |---|---|---|
//! | `MemoryPlan::schedule` | whole graph | topological order the ops run in |
//! | `PhaseSchedule` | whole graph | which phase each op belongs to |
//! | [`KernelSchedule`] | **one kernel** | how that kernel drives the machine |
//!
//! The first two order *between* ops. This one describes what happens *inside*
//! one — which roles exist, which barrier gates which handoff, how deeply a
//! buffer is staged. `KernelSchedule` rather than `Schedule` precisely because
//! the shorter name was already taken by the graph-level sense.
//!
//! `TileParams`-shaped facts (block tile, micro tile) are a *subset* of a
//! kernel schedule: they fix the region dimensions and nothing else.
//! `rlx_gpu_kernels::kernel_schedule_port` derives the rest. That type lives
//! in `rlx-gpu-dispatch`, which this crate does not depend on.
//!
//! # Scope of this module
//!
//! This is the **representation and its analyses**, not a code generator. It
//! deliberately does not lower to MSL or PTX, and no backend consumes it yet.
//! That ordering is the point: CAKE's argument for a schedule IR is that it
//! makes analysis possible before compilation, so the analyses are the
//! deliverable. A schedule language with no checks would be syntax.
//!
//! It is also strictly **additive** — a new type alongside `Op`, not a change
//! to it. Downstream crates construct `Op` heavily and match on it rarely, so
//! extending the op vocabulary would have been the breaking path; this is not.
//!
//! # What the verifier catches that nothing else can
//!
//! [`verify_kernel_schedule`] answers questions that are undecidable at the op-graph
//! level because the op graph has no notion of concurrency:
//!
//! * a consumer reading a buffer without waiting on the barrier that fills it
//!   (a data race that manifests as intermittently wrong numbers),
//! * a role waiting on a barrier no role ever arrives at (a hang),
//! * a barrier whose declared arrival count cannot be met (a hang),
//! * shared memory over the target's budget (a launch failure),
//! * two accesses to one region committing to different physical addressing
//!   (the `rocm-gguf-transposed` class: same shapes, different strides),
//! * a required capability the target does not provide.
//!
//! # Scope of the analysis
//!
//! Stated because a gate that implies more than it checks is worse than none,
//! and CAKE Appendix C says the same of its own: this "is a pre-compile gate
//! only within its modeled domain; it does not prove global GPU correctness or
//! capture all microarchitectural behavior... both false positives and false
//! negatives can occur."
//!
//! Concretely, [`verify_kernel_schedule`] does **not** model: per-thread indexing
//! inside an action, register pressure, occupancy, instruction latency,
//! memory-ordering subtleties below the barrier level, or anything about the
//! numerics computed. A clean report means the declared schedule is
//! self-consistent and fits the declared target — not that the kernel is
//! correct.

use std::collections::{BTreeMap, BTreeSet};

use crate::DType;

/// Where a declared buffer lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Space {
    /// Device-global memory.
    Global,
    /// Threadgroup / shared memory. Budgeted per block.
    Shared,
    /// Per-thread registers / accumulators.
    Register,
}

/// How a tile is permuted in on-chip memory to avoid bank conflicts.
///
/// A tag, not an algebra. CAKE B.4 takes this position deliberately, against
/// CuTe's layout algebra and Triton's linear layouts: "the agent writes down
/// the concrete commitments... and the compiler carries the burden of deciding
/// whether those commitments are legal."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swizzle {
    None,
    /// XOR-swizzle with the given period in elements.
    Xor(u32),
}

/// A **concrete layout commitment** on a view of a region.
///
/// This is the piece that makes a region more than a shape annotation. rlx has
/// shipped exactly one class of defect that nothing could catch without it:
/// `rocm-gguf-transposed`, where a kernel issued `sgemm(N, N)` against weights
/// physically laid out `[n, k]`. Both sides were internally consistent; they
/// simply disagreed about which axis was contiguous, and no shape check can
/// see that because the shapes matched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// Byte offset of this view within its region.
    pub offset_bytes: usize,
    /// Stride per dimension, in ELEMENTS. `[k, 1]` is row-major over `[n, k]`;
    /// `[1, n]` is the same buffer read column-major.
    pub strides: Vec<usize>,
    pub swizzle: Swizzle,
}

impl Layout {
    /// Row-major over `dims`.
    pub fn row_major(dims: &[usize]) -> Self {
        let mut strides = vec![1usize; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        Self {
            offset_bytes: 0,
            strides,
            swizzle: Swizzle::None,
        }
    }

    /// Column-major over `dims` — the same bytes, transposed reading.
    pub fn col_major(dims: &[usize]) -> Self {
        let mut strides = vec![1usize; dims.len()];
        for i in 1..dims.len() {
            strides[i] = strides[i - 1] * dims[i - 1];
        }
        Self {
            offset_bytes: 0,
            strides,
            swizzle: Swizzle::None,
        }
    }

    /// Highest element index this view can address, given `dims`.
    pub fn extent_elems(&self, dims: &[usize]) -> usize {
        dims.iter()
            .zip(&self.strides)
            .map(|(d, s)| d.saturating_sub(1) * s)
            .sum::<usize>()
            + 1
    }

    /// Whether two commitments describe the same physical addressing.
    pub fn agrees_with(&self, other: &Layout) -> bool {
        self.offset_bytes == other.offset_bytes
            && self.strides == other.strides
            && self.swizzle == other.swizzle
    }
}

/// A declared memory region. Declared once, so the schedule knows the shape,
/// dtype and lifetime of every buffer — CAKE's "declared resources".
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    pub name: String,
    pub space: Space,
    pub dims: Vec<usize>,
    pub dtype: DType,
    /// Pipeline depth: how many stages of this region are live at once.
    pub stages: usize,
    /// The region's own layout commitment. Every access must agree with it
    /// unless the access declares its own.
    pub layout: Layout,
}

impl Region {
    pub fn bytes(&self) -> usize {
        self.dims.iter().product::<usize>() * self.dtype.size_bytes() * self.stages.max(1)
    }
}

/// A named group of threads with assigned work. Roles are explicit so every
/// cross-role handoff is visible rather than an implicit convention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Role {
    pub name: String,
    /// Which warps/simdgroups this role occupies.
    pub warps: Vec<u32>,
}

/// A producer→consumer synchronization point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Barrier {
    pub name: String,
    /// Roles that arrive at this barrier.
    pub producers: Vec<String>,
    /// Roles that wait on it.
    pub consumers: Vec<String>,
    /// Arrivals required before waiters proceed.
    pub count: u32,
}

/// The hardware instruction an action is issued through, when it is not a
/// plain scalar loop.
///
/// CAKE B.4's second half: commitments must "satisfy the target instruction
/// and resource contracts". Resource contracts are about fitting the region;
/// instruction contracts are about what the *instruction* requires of its
/// operands, which is a different question and the one that catches
/// tensor-core-path errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    /// A cooperative-matrix multiply-accumulate over an `m x n x k` tile.
    ///
    /// Metal spells this `simdgroup_multiply_accumulate` over
    /// `simdgroup_float8x8`; Vulkan spells it `OpCooperativeMatrixMulAddKHR`;
    /// CUDA spells it `mma.sync`. The contract is the same shape in all three:
    /// the operand's row stride must be a whole number of tiles, and the
    /// region must tile evenly.
    CoopMatrix { m: u32, n: u32, k: u32 },
}

/// One access to a region, with the layout the accessor commits to.
///
/// `layout: None` means "the region's declared layout". Naming one explicitly
/// is how a transposed read is *stated* rather than implied — and therefore how
/// it becomes checkable.
#[derive(Debug, Clone, PartialEq)]
pub struct Access {
    pub region: String,
    pub layout: Option<Layout>,
}

impl Access {
    /// Access a region on its declared layout.
    pub fn plain(region: impl Into<String>) -> Self {
        Self {
            region: region.into(),
            layout: None,
        }
    }

    /// Access a region on an explicitly committed layout.
    pub fn with(region: impl Into<String>, layout: Layout) -> Self {
        Self {
            region: region.into(),
            layout: Some(layout),
        }
    }
}

/// One action a role performs.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Stage `region` from global into its declared space.
    Load { access: Access, stage: usize },
    /// Compute over `reads`, producing `writes`.
    Compute {
        reads: Vec<Access>,
        writes: Vec<Access>,
        stage: usize,
        /// `None` = a scalar loop, which constrains nothing. `Some` brings the
        /// instruction's operand contract into play.
        via: Option<Instruction>,
    },
    /// Write `region` back out.
    Store { access: Access, stage: usize },
    /// Signal arrival at `barrier`.
    Arrive { barrier: String, stage: usize },
    /// Block until `barrier` is satisfied.
    Wait { barrier: String, stage: usize },
}

impl Action {
    pub fn stage(&self) -> usize {
        match self {
            Self::Load { stage, .. }
            | Self::Compute { stage, .. }
            | Self::Store { stage, .. }
            | Self::Arrive { stage, .. }
            | Self::Wait { stage, .. } => *stage,
        }
    }

    /// Every region this action touches, with its committed layout.
    pub fn accesses(&self) -> Vec<&Access> {
        match self {
            Self::Load { access, .. } | Self::Store { access, .. } => vec![access],
            Self::Compute { reads, writes, .. } => reads.iter().chain(writes).collect(),
            _ => vec![],
        }
    }

    fn barrier(&self) -> Option<&str> {
        match self {
            Self::Arrive { barrier, .. } | Self::Wait { barrier, .. } => Some(barrier),
            _ => None,
        }
    }
}

/// A complete machine schedule for one kernel.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KernelSchedule {
    pub name: String,
    pub regions: Vec<Region>,
    pub roles: Vec<Role>,
    pub barriers: Vec<Barrier>,
    /// Per-role action sequence, in program order.
    pub body: BTreeMap<String, Vec<Action>>,
    /// Pipeline depth.
    pub stages: usize,
    /// Capabilities this schedule needs. Declared, so a target that lacks one
    /// produces a named rejection instead of a silent fallback.
    pub requires: Vec<Feature>,
}

impl KernelSchedule {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: 1,
            ..Default::default()
        }
    }

    /// Total declared bytes in `space`. Auto-derived from the declarations —
    /// nothing states it separately, so it cannot drift.
    pub fn bytes_in(&self, space: Space) -> usize {
        self.regions
            .iter()
            .filter(|r| r.space == space)
            .map(Region::bytes)
            .sum()
    }

    /// Warps used by any role. Derived, not declared.
    pub fn warps(&self) -> BTreeSet<u32> {
        self.roles
            .iter()
            .flat_map(|r| r.warps.iter().copied())
            .collect()
    }
}

/// A capability a schedule may require of its target.
///
/// CAKE B.5 tabulates these per SKU (`mma.sync`, `wgmma`, TMA, TMEM, clusters,
/// async barriers) precisely so a schedule that needs one is *rejected* on a
/// target lacking it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Feature {
    /// A cooperative-matrix / tensor-core MMA instruction of a specific shape.
    ///
    /// The shape is part of the capability, not a detail: CUDA fixes
    /// `mma.sync.m16n8k16` by ISA, Metal's `simdgroup_float8x8` is 8x8, and
    /// Vulkan *enumerates* the supported `(M, N, K, types)` combinations per
    /// device via `VkCooperativeMatrixPropertiesKHR`. A bare "has tensor
    /// cores" flag cannot express any of that, and rlx's wgpu path currently
    /// assumes 8x8 f16 works whenever the feature bit is present — an
    /// unchecked assumption a device is free to violate.
    CoopMatrix { m: u32, n: u32, k: u32 },
    /// Asynchronous global→shared copy.
    AsyncCopy,
    /// Hardware barriers with an arrival count (as opposed to a full block sync).
    AsyncBarrier,
    /// Swizzled shared-memory addressing.
    Swizzle,
}

/// Most capabilities a single device is expected to expose.
///
/// Bounded so [`FeatureSet`] stays `Copy` and `Target` keeps its 156 by-value
/// call sites. Overflow is recorded rather than dropped — see
/// [`FeatureSet::truncated`].
pub const MAX_QUERIED_FEATURES: usize = 16;

/// Capabilities read back from a live device.
///
/// Fixed capacity so this is `Copy`; `truncated` exists because silently
/// dropping the 17th capability would make an *unqueried* feature
/// indistinguishable from an *unsupported* one, and the whole point of this
/// type is that the difference is reportable.
#[derive(Debug, Clone, Copy)]
pub struct FeatureSet {
    items: [Option<Feature>; MAX_QUERIED_FEATURES],
    len: usize,
    /// The device offered more capabilities than fit.
    pub truncated: bool,
}

impl Default for FeatureSet {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureSet {
    pub const fn new() -> Self {
        Self {
            items: [None; MAX_QUERIED_FEATURES],
            len: 0,
            truncated: false,
        }
    }

    /// Record a capability. Returns `false` when the set is full, and sets
    /// [`Self::truncated`].
    pub fn push(&mut self, f: Feature) -> bool {
        if self.len >= MAX_QUERIED_FEATURES {
            self.truncated = true;
            return false;
        }
        self.items[self.len] = Some(f);
        self.len += 1;
        true
    }

    pub fn contains(&self, f: Feature) -> bool {
        self.items[..self.len].contains(&Some(f))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Where a target's capability list came from.
///
/// The distinction is not cosmetic. CUDA and Metal capabilities are ISA facts
/// and belong in a compile-time table. Vulkan's cooperative-matrix support is
/// **enumerated per device** at run time via `VkCooperativeMatrixPropertiesKHR`
/// — the supported `(M, N, K, types, scope)` combinations are not knowable from
/// the API version, and rlx's wgpu path currently assumes 8x8 f16 works
/// whenever the feature bit is set.
#[derive(Debug, Clone, Copy)]
pub enum Features {
    /// From an architecture table baked in at compile time.
    Static(&'static [Feature]),
    /// Read back from a live device.
    Queried(FeatureSet),
}

/// Hardware limits and capabilities a schedule is checked against.
#[derive(Debug, Clone, Copy)]
pub struct Target {
    pub name: &'static str,
    pub max_shared_bytes: usize,
    pub max_warps: usize,
    /// Capabilities this target provides. A schedule requiring anything absent
    /// is REPORTED, never quietly lowered to something the target does have.
    pub features: Features,
}

impl Target {
    pub fn has(&self, f: Feature) -> bool {
        match &self.features {
            Features::Static(list) => list.contains(&f),
            Features::Queried(set) => set.contains(f),
        }
    }

    /// `true` when the capability list may be incomplete, so a `false` from
    /// [`Self::has`] means "not known to be supported" rather than "known to be
    /// unsupported".
    pub fn capabilities_may_be_incomplete(&self) -> bool {
        matches!(&self.features, Features::Queried(s) if s.truncated)
    }
}

impl Target {
    /// Conservative floor across the GPUs rlx targets: 32 KiB of shared memory
    /// and 32 warps. Deliberately the *minimum* rather than any one device's
    /// ceiling — a schedule that clears this runs everywhere, and a schedule
    /// tuned to one device's larger limit should say which device.
    pub const PORTABLE: Self = Self {
        name: "portable-floor",
        max_shared_bytes: 32 * 1024,
        max_warps: 32,
        features: Features::Static(&[]),
    };
    pub const CUDA_SM86: Self = Self {
        name: "sm_86",
        max_shared_bytes: 48 * 1024,
        max_warps: 32,
        // sm_86: mma.sync m16n8k16 is the canonical fp16 shape.
        features: Features::Static(&[
            Feature::CoopMatrix { m: 16, n: 8, k: 16 },
            Feature::AsyncCopy,
            Feature::Swizzle,
        ]),
    };
    /// Ampere-class without the async-copy path — the older SKU that makes
    /// "requires a feature" a real rejection rather than a formality.
    pub const CUDA_SM70: Self = Self {
        name: "sm_70",
        max_shared_bytes: 48 * 1024,
        max_warps: 32,
        features: Features::Static(&[Feature::CoopMatrix { m: 16, n: 8, k: 16 }]),
    };
    /// CDNA2-class compute GPU (gfx908 MI100 is this project's AMD rig).
    ///
    /// 64 KiB LDS per workgroup, 32 warps (AMD calls them wavefronts; the IR's
    /// unit is 32 lanes and gfx908 waves are 64, so a role's warp count is a
    /// *thread* budget here rather than a hardware wave count — `lower()`
    /// multiplies by 32 either way, which is what the emitter checks against).
    ///
    /// `AsyncCopy` is deliberately ABSENT even though CDNA has `global_load_lds`:
    /// a capability list is a claim about what rlx can *drive*, and no emitter
    /// in this tree emits that instruction. Listing it would let a schedule
    /// verify and then fail to compile, which is the failure mode `Feature`
    /// exists to prevent.
    pub const ROCM_GFX908: Self = Self {
        name: "gfx908",
        max_shared_bytes: 64 * 1024,
        max_warps: 32,
        features: Features::Static(&[Feature::CoopMatrix { m: 16, n: 16, k: 4 }]),
    };
    /// RDNA3 integrated GPU (gfx1103, the 780M on the same rig).
    ///
    /// Same 64 KiB LDS; no matrix cores exposed through the paths rlx uses, so
    /// no `CoopMatrix`. An empty capability list is a real answer, not a
    /// placeholder.
    pub const ROCM_GFX1103: Self = Self {
        name: "gfx1103",
        max_shared_bytes: 64 * 1024,
        max_warps: 32,
        features: Features::Static(&[]),
    };
    pub const METAL_APPLE: Self = Self {
        name: "apple-gpu",
        max_shared_bytes: 32 * 1024,
        max_warps: 32,
        // Apple GPUs expose simdgroup_float8x8 / simdgroup_half8x8.
        features: Features::Static(&[Feature::CoopMatrix { m: 8, n: 8, k: 8 }, Feature::Swizzle]),
    };
}

/// A schedule defect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelScheduleError {
    /// A consumer reads a region another role writes, with no barrier between.
    UnsynchronizedHandoff {
        region: String,
        producer: String,
        consumer: String,
    },
    /// A role waits on a barrier nobody arrives at.
    DeadlockWait { role: String, barrier: String },
    /// A barrier's required arrival count exceeds the roles that can arrive.
    UnreachableCount {
        barrier: String,
        count: u32,
        producers: usize,
    },
    /// Declared shared memory exceeds the target's budget.
    SharedOverBudget {
        bytes: usize,
        limit: usize,
        target: &'static str,
    },
    /// Two roles claim the same warp.
    WarpConflict { warp: u32, roles: Vec<String> },
    /// A reference to a name that was never declared.
    UndeclaredName { kind: &'static str, name: String },
    /// A declared role with no actions.
    EmptyRole { role: String },
    /// An action names a stage past the declared pipeline depth.
    StageOutOfRange {
        role: String,
        stage: usize,
        stages: usize,
    },
    /// Two accesses to the same region commit to different physical
    /// addressing. The `rocm-gguf-transposed` class.
    LayoutDisagreement {
        region: String,
        a: String,
        b: String,
        detail: String,
    },
    /// The schedule requires a capability the target does not provide.
    ///
    /// Reported rather than worked around: CAKE B.5 requires "an exact target
    /// match and reports missing device or toolchain support rather than
    /// stepping a schedule down to another architecture." rlx's backends step
    /// down silently today, which is how an ineligible kernel variant reached
    /// a shape it could not handle.
    UnsupportedFeature {
        feature: Feature,
        target: &'static str,
    },
    /// One role writes a shared region and reads it back with no barrier
    /// between — a missing `__syncthreads()`.
    ///
    /// Distinct from [`Self::UnsynchronizedHandoff`], which is about two
    /// roles. The uniform-block kernels rlx actually ships have ONE role that
    /// both stages and consumes, so the cross-role check never fires on them
    /// and this is the hazard that matters.
    MissingReadBarrier {
        role: String,
        region: String,
        wrote_at: usize,
        read_at: usize,
    },
    /// A role reads a shared region and a later action overwrites it with no
    /// barrier between — write-after-read across a pipeline iteration.
    ///
    /// This is what the *second* `__syncthreads()` in a tiled matmul's K loop
    /// prevents: without it the next iteration's staging can land while
    /// threads are still reading the current tile.
    MissingOverwriteBarrier {
        role: String,
        region: String,
        read_at: usize,
        overwrote_at: usize,
    },
    /// An operand's row stride is not a whole number of instruction tiles.
    ///
    /// On Metal this is the `elements_per_row` argument to `simdgroup_load`
    /// disagreeing with the region it reads — which compiles cleanly and reads
    /// the wrong elements.
    OperandStrideNotTiled {
        region: String,
        stride: usize,
        tile: u32,
        instruction: Instruction,
    },
    /// An operand region does not tile evenly under the instruction's shape.
    OperandNotTileable {
        region: String,
        dims: Vec<usize>,
        m: u32,
        k: u32,
    },
    /// A swizzle period that does not divide the row it permutes.
    ///
    /// An XOR swizzle exists to spread consecutive rows across memory banks.
    /// If the period does not divide the innermost extent, it aliases rows
    /// onto each other instead — the addressing is still "valid" and the
    /// bank conflicts it was added to remove come back, silently.
    IllegalSwizzle {
        region: String,
        period: u32,
        innermost: usize,
    },
    /// A committed layout addresses past the end of its region.
    LayoutOutOfRegion {
        region: String,
        role: String,
        extent_elems: usize,
        region_elems: usize,
    },
}

impl std::fmt::Display for KernelScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsynchronizedHandoff {
                region,
                producer,
                consumer,
            } => write!(
                f,
                "role '{consumer}' reads '{region}' written by role '{producer}' with no \
                 barrier between them — a race, not a slow path"
            ),
            Self::DeadlockWait { role, barrier } => {
                write!(
                    f,
                    "role '{role}' waits on barrier '{barrier}' that no role arrives at"
                )
            }
            Self::UnreachableCount {
                barrier,
                count,
                producers,
            } => write!(
                f,
                "barrier '{barrier}' needs {count} arrival(s) but only {producers} role(s) \
                 arrive at it"
            ),
            Self::SharedOverBudget {
                bytes,
                limit,
                target,
            } => write!(
                f,
                "declared shared memory {bytes} B exceeds {target}'s {limit} B budget"
            ),
            Self::WarpConflict { warp, roles } => {
                write!(f, "warp {warp} claimed by roles {roles:?}")
            }
            Self::UndeclaredName { kind, name } => write!(f, "undeclared {kind}: '{name}'"),
            Self::EmptyRole { role } => write!(f, "role '{role}' has no actions"),
            Self::StageOutOfRange {
                role,
                stage,
                stages,
            } => write!(
                f,
                "role '{role}' names stage {stage} but the pipeline declares {stages}"
            ),
            Self::UnsupportedFeature { feature, target } => write!(
                f,
                "schedule requires {feature:?} which target '{target}' does not provide — \
                 no step-down is attempted"
            ),
            Self::MissingReadBarrier {
                role,
                region,
                wrote_at,
                read_at,
            } => write!(
                f,
                "role '{role}' writes '{region}' at action {wrote_at} and reads it at \
                 {read_at} with no barrier between — the write may not be visible"
            ),
            Self::MissingOverwriteBarrier {
                role,
                region,
                read_at,
                overwrote_at,
            } => write!(
                f,
                "role '{role}' reads '{region}' at action {read_at} and overwrites it at \
                 {overwrote_at} with no barrier between — the overwrite may land mid-read"
            ),
            Self::OperandStrideNotTiled {
                region,
                stride,
                tile,
                instruction,
            } => write!(
                f,
                "operand '{region}' has row stride {stride}, not a multiple of the \
                 {tile}-wide tile {instruction:?} requires"
            ),
            Self::OperandNotTileable { region, dims, m, k } => write!(
                f,
                "operand '{region}' with dims {dims:?} does not tile evenly under {m}x{k}"
            ),
            Self::IllegalSwizzle {
                region,
                period,
                innermost,
            } => write!(
                f,
                "region '{region}': XOR-swizzle period {period} must be a power of two that \
                 divides the innermost extent {innermost}, or it aliases rows instead of \
                 spreading them"
            ),
            Self::LayoutDisagreement {
                region,
                a,
                b,
                detail,
            } => write!(
                f,
                "region '{region}': '{a}' and '{b}' commit to different layouts ({detail}) — \
                 the shapes match and the addressing does not"
            ),
            Self::LayoutOutOfRegion {
                region,
                role,
                extent_elems,
                region_elems,
            } => write!(
                f,
                "role '{role}' commits to a layout over '{region}' spanning {extent_elems} \
                 element(s), past the region's {region_elems}"
            ),
        }
    }
}

/// Check a schedule against a target.
///
/// Pure and device-free: this is the pre-compile gate CAKE's Table 1 calls
/// program safety and schedule semantics, at a level the op graph cannot
/// express.
/// Check a kernel schedule against a target.
///
/// Pure and device-free: the pre-compile gate CAKE's Table 1 calls program
/// safety and schedule semantics, at a level the op graph cannot express.
///
/// Split into one function per check class rather than one long body — the
/// classes are the design (they name different repair targets), and burying
/// them in section comments inside a 400-line function made that invisible.
pub fn verify_kernel_schedule(sched: &KernelSchedule, target: Target) -> Vec<KernelScheduleError> {
    let mut errors = Vec::new();
    check_declarations_resolve(sched, &mut errors);
    check_roles(sched, target, &mut errors);
    check_resources(sched, target, &mut errors);
    check_synchronization(sched, &mut errors);
    check_intra_role_ordering(sched, &mut errors);
    check_target_capabilities(sched, target, &mut errors);
    check_operand_contracts(sched, target, &mut errors);
    check_layout_commitments(sched, &mut errors);
    errors
}

fn check_declarations_resolve(sched: &KernelSchedule, errors: &mut Vec<KernelScheduleError>) {
    let role_names: BTreeSet<&str> = sched.roles.iter().map(|r| r.name.as_str()).collect();
    let region_names: BTreeSet<&str> = sched.regions.iter().map(|r| r.name.as_str()).collect();
    let barrier_names: BTreeSet<&str> = sched.barriers.iter().map(|b| b.name.as_str()).collect();
    // ── Declarations resolve ────────────────────────────────────────────
    for b in &sched.barriers {
        for r in b.producers.iter().chain(&b.consumers) {
            if !role_names.contains(r.as_str()) {
                errors.push(KernelScheduleError::UndeclaredName {
                    kind: "role",
                    name: r.clone(),
                });
            }
        }
    }
    for (role, actions) in &sched.body {
        if !role_names.contains(role.as_str()) {
            errors.push(KernelScheduleError::UndeclaredName {
                kind: "role",
                name: role.clone(),
            });
        }
        for a in actions {
            if a.stage() >= sched.stages.max(1) {
                errors.push(KernelScheduleError::StageOutOfRange {
                    role: role.clone(),
                    stage: a.stage(),
                    stages: sched.stages.max(1),
                });
            }
            for acc in a.accesses() {
                if !region_names.contains(acc.region.as_str()) {
                    errors.push(KernelScheduleError::UndeclaredName {
                        kind: "region",
                        name: acc.region.clone(),
                    });
                }
            }
            if let Some(b) = a.barrier()
                && !barrier_names.contains(b)
            {
                errors.push(KernelScheduleError::UndeclaredName {
                    kind: "barrier",
                    name: b.to_string(),
                });
            }
        }
    }
}

fn check_roles(sched: &KernelSchedule, target: Target, errors: &mut Vec<KernelScheduleError>) {
    // ── Roles ───────────────────────────────────────────────────────────
    let mut warp_owner: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for r in &sched.roles {
        if sched.body.get(&r.name).is_none_or(|a| a.is_empty()) {
            errors.push(KernelScheduleError::EmptyRole {
                role: r.name.clone(),
            });
        }
        for w in &r.warps {
            warp_owner.entry(*w).or_default().push(r.name.clone());
        }
    }
    for (warp, roles) in warp_owner {
        if roles.len() > 1 {
            errors.push(KernelScheduleError::WarpConflict { warp, roles });
        }
    }
    if sched.warps().len() > target.max_warps {
        // Reported through the shared-budget arm's sibling: warp count over the
        // target is a launch failure exactly like shared memory is.
        errors.push(KernelScheduleError::SharedOverBudget {
            bytes: sched.warps().len(),
            limit: target.max_warps,
            target: target.name,
        });
    }
}

fn check_resources(sched: &KernelSchedule, target: Target, errors: &mut Vec<KernelScheduleError>) {
    // ── Resources ───────────────────────────────────────────────────────
    let shared = sched.bytes_in(Space::Shared);
    if shared > target.max_shared_bytes {
        errors.push(KernelScheduleError::SharedOverBudget {
            bytes: shared,
            limit: target.max_shared_bytes,
            target: target.name,
        });
    }
}

fn check_synchronization(sched: &KernelSchedule, errors: &mut Vec<KernelScheduleError>) {
    // ── Synchronization ─────────────────────────────────────────────────
    for b in &sched.barriers {
        let arrivers: usize = b
            .producers
            .iter()
            .filter(|p| {
                sched.body.get(*p).is_some_and(|acts| {
                    acts.iter()
                        .any(|a| matches!(a, Action::Arrive { barrier, .. } if barrier == &b.name))
                })
            })
            .count();
        if (b.count as usize) > arrivers {
            errors.push(KernelScheduleError::UnreachableCount {
                barrier: b.name.clone(),
                count: b.count,
                producers: arrivers,
            });
        }
    }
    for (role, actions) in &sched.body {
        for a in actions {
            if let Action::Wait { barrier, .. } = a {
                let anyone_arrives = sched.body.values().any(|acts| {
                    acts.iter()
                        .any(|x| matches!(x, Action::Arrive { barrier: b2, .. } if b2 == barrier))
                });
                if !anyone_arrives {
                    errors.push(KernelScheduleError::DeadlockWait {
                        role: role.clone(),
                        barrier: barrier.clone(),
                    });
                }
            }
        }
    }

    // Cross-role handoff: a region written by one role and read by another
    // must have a barrier the writer arrives at and the reader waits on.
    //
    // This is the check the op graph fundamentally cannot make. It has no
    // concept of two roles at all, so a missing barrier is not a wrong graph —
    // it is a correct graph compiled into a race.
    let mut writers: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut readers: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (role, actions) in &sched.body {
        for a in actions {
            match a {
                Action::Load { access, .. } => {
                    writers.entry(&access.region).or_default().insert(role);
                }
                Action::Compute { reads, writes, .. } => {
                    for r in reads {
                        readers.entry(&r.region).or_default().insert(role);
                    }
                    for w in writes {
                        writers.entry(&w.region).or_default().insert(role);
                    }
                }
                Action::Store { access, .. } => {
                    readers.entry(&access.region).or_default().insert(role);
                }
                _ => {}
            }
        }
    }
    for (region, ws) in &writers {
        let Some(rs) = readers.get(region) else {
            continue;
        };
        for w in ws {
            for r in rs {
                if w == r {
                    continue;
                }
                let synced = sched.barriers.iter().any(|b| {
                b.producers.iter().any(|p| p == w)
                    && b.consumers.iter().any(|c| c == r)
                    && sched.body.get(*w).is_some_and(|acts| {
                        acts.iter().any(|a| matches!(a, Action::Arrive { barrier, .. } if barrier == &b.name))
                    })
                    && sched.body.get(*r).is_some_and(|acts| {
                        acts.iter().any(|a| matches!(a, Action::Wait { barrier, .. } if barrier == &b.name))
                    })
            });
                if !synced {
                    errors.push(KernelScheduleError::UnsynchronizedHandoff {
                        region: (*region).to_string(),
                        producer: (*w).to_string(),
                        consumer: (*r).to_string(),
                    });
                }
            }
        }
    }
}

fn check_intra_role_ordering(sched: &KernelSchedule, errors: &mut Vec<KernelScheduleError>) {
    // ── Intra-role ordering ─────────────────────────────────────────────
    //
    // Within one role, a write and a later read of the same on-chip region
    // must be separated by a barrier, and so must a read and a later
    // overwrite. Both are what the two `__syncthreads()` calls in a tiled
    // matmul's K loop buy; dropping either produces a kernel that is correct
    // most of the time.
    //
    // Only `Shared` regions are checked: registers are private to a thread, so
    // no barrier can or should order them.
    let shared: BTreeSet<&str> = sched
        .regions
        .iter()
        .filter(|r| r.space == Space::Shared)
        .map(|r| r.name.as_str())
        .collect();
    // Declared pipeline depth per region. A region with `stages > 1` is
    // `stages` distinct buffers, and an action names which one it touches.
    //
    // Keying the hazard maps on the region NAME alone was a false positive that
    // rejected every rotating buffer: in a staged pipeline the read at slot
    // `cur` and the store at slot `next` are different memory, so no barrier
    // between them is required and demanding one makes the schedule
    // unexpressible. It was found by porting a multi-stage matmul, which is the
    // only way a rule that only ever saw one-stage kernels could be found
    // wrong. At `stages == 1` every slot collapses to 0 and the behaviour is
    // exactly what it was.
    let depth: BTreeMap<&str, usize> = sched
        .regions
        .iter()
        .map(|r| (r.name.as_str(), r.stages.max(1)))
        .collect();
    // Which physical buffer of `region` an action at `stage` touches.
    let slot = |region: &str, stage: usize| -> usize {
        let d = depth.get(region).copied().unwrap_or(1);
        if d <= 1 { 0 } else { stage % d }
    };
    for (role, actions) in &sched.body {
        // Last index at which this role wrote / read each region SLOT, and the
        // last barrier it synchronized at.
        let mut wrote: BTreeMap<(&str, usize), usize> = BTreeMap::new();
        let mut read: BTreeMap<(&str, usize), usize> = BTreeMap::new();
        let mut last_sync: Option<usize> = None;
        for (i, a) in actions.iter().enumerate() {
            if matches!(a, Action::Wait { .. } | Action::Arrive { .. }) {
                last_sync = Some(i);
                continue;
            }
            let (reads, writes): (Vec<&str>, Vec<&str>) = match a {
                Action::Load { access, .. } => (vec![], vec![access.region.as_str()]),
                Action::Store { access, .. } => (vec![access.region.as_str()], vec![]),
                Action::Compute { reads, writes, .. } => (
                    reads.iter().map(|x| x.region.as_str()).collect(),
                    writes.iter().map(|x| x.region.as_str()).collect(),
                ),
                _ => (vec![], vec![]),
            };
            let stage = a.stage();
            for r in &reads {
                if !shared.contains(r) {
                    continue;
                }
                if let Some(w) = wrote.get(&(*r, slot(r, stage)))
                    && last_sync.is_none_or(|s| s < *w)
                {
                    errors.push(KernelScheduleError::MissingReadBarrier {
                        role: role.clone(),
                        region: (*r).to_string(),
                        wrote_at: *w,
                        read_at: i,
                    });
                }
            }
            for w in &writes {
                if !shared.contains(w) {
                    continue;
                }
                if let Some(rd) = read.get(&(*w, slot(w, stage)))
                    && last_sync.is_none_or(|s| s < *rd)
                {
                    errors.push(KernelScheduleError::MissingOverwriteBarrier {
                        role: role.clone(),
                        region: (*w).to_string(),
                        read_at: *rd,
                        overwrote_at: i,
                    });
                }
            }
            for r in reads {
                read.insert((r, slot(r, stage)), i);
            }
            for w in writes {
                wrote.insert((w, slot(w, stage)), i);
            }
        }
    }
}

fn check_target_capabilities(
    sched: &KernelSchedule,
    target: Target,
    errors: &mut Vec<KernelScheduleError>,
) {
    // ── Target capabilities ─────────────────────────────────────────────
    for f in &sched.requires {
        if !target.has(*f) {
            errors.push(KernelScheduleError::UnsupportedFeature {
                feature: *f,
                target: target.name,
            });
        }
    }
    // A swizzled commitment is itself a capability requirement, whether or not
    // the schedule remembered to declare it.
    let uses_swizzle = sched
        .regions
        .iter()
        .any(|r| r.layout.swizzle != Swizzle::None)
        || sched.body.values().flatten().any(|a| {
            a.accesses().iter().any(|acc| {
                acc.layout
                    .as_ref()
                    .is_some_and(|l| l.swizzle != Swizzle::None)
            })
        });
    if uses_swizzle && !target.has(Feature::Swizzle) {
        errors.push(KernelScheduleError::UnsupportedFeature {
            feature: Feature::Swizzle,
            target: target.name,
        });
    }
}

fn check_operand_contracts(
    sched: &KernelSchedule,
    target: Target,
    errors: &mut Vec<KernelScheduleError>,
) {
    // ── Instruction operand contracts ───────────────────────────────────
    //
    // Distinct from the resource checks below. Those ask "does this view fit
    // the buffer"; this asks "does the instruction accept an operand shaped
    // like this". A `simdgroup_load` told the wrong `elements_per_row`
    // satisfies every resource check and still reads the wrong elements.
    let region_lookup: BTreeMap<&str, &Region> =
        sched.regions.iter().map(|r| (r.name.as_str(), r)).collect();
    for actions in sched.body.values() {
        for a in actions {
            let Action::Compute {
                reads,
                writes,
                via: Some(instr),
                ..
            } = a
            else {
                continue;
            };
            let Instruction::CoopMatrix { m, n, k } = *instr;

            // The target must offer this exact shape. Vulkan enumerates the
            // supported combinations per device, so "has tensor cores" is not
            // an answer.
            if !target.has(Feature::CoopMatrix { m, n, k }) {
                errors.push(KernelScheduleError::UnsupportedFeature {
                    feature: Feature::CoopMatrix { m, n, k },
                    target: target.name,
                });
            }

            for acc in reads.iter().chain(writes) {
                let Some(region) = region_lookup.get(acc.region.as_str()) else {
                    continue;
                };
                let layout = acc.layout.as_ref().unwrap_or(&region.layout);
                // Row stride must be a whole number of tiles wide, or a tile
                // load straddles rows.
                let stride = *layout.strides.first().unwrap_or(&1);
                if !stride.is_multiple_of(n as usize) {
                    errors.push(KernelScheduleError::OperandStrideNotTiled {
                        region: acc.region.clone(),
                        stride,
                        tile: n,
                        instruction: *instr,
                    });
                }
                // And the region itself must tile evenly.
                let rows = *region.dims.first().unwrap_or(&1);
                let cols = *region.dims.last().unwrap_or(&1);
                if !rows.is_multiple_of(m as usize) || !cols.is_multiple_of(k as usize) {
                    errors.push(KernelScheduleError::OperandNotTileable {
                        region: acc.region.clone(),
                        dims: region.dims.clone(),
                        m,
                        k,
                    });
                }
            }
        }
    }
}

fn check_layout_commitments(sched: &KernelSchedule, errors: &mut Vec<KernelScheduleError>) {
    // ── Layout commitments ──────────────────────────────────────────────
    //
    // CAKE B.4: "The compiler checks that these commitments remain mutually
    // consistent along the program's data flow and satisfy the target
    // instruction and resource contracts."
    //
    // Shape checking cannot do this. In `rocm-gguf-transposed` both sides
    // agreed the operand was [n, k]; they disagreed about which axis was
    // contiguous. Same dims, different addressing, wrong numbers.
    let region_by_name: BTreeMap<&str, &Region> =
        sched.regions.iter().map(|r| (r.name.as_str(), r)).collect();
    let mut committed: BTreeMap<&str, (String, &Layout)> = BTreeMap::new();
    for (role, actions) in &sched.body {
        for a in actions {
            for acc in a.accesses() {
                let Some(region) = region_by_name.get(acc.region.as_str()) else {
                    continue;
                };
                let layout = acc.layout.as_ref().unwrap_or(&region.layout);

                // The swizzle must actually de-conflict rather than alias.
                if let Swizzle::Xor(period) = layout.swizzle {
                    let innermost = *region.dims.last().unwrap_or(&1);
                    let pow2 = period.is_power_of_two();
                    if !pow2 || period == 0 || !innermost.is_multiple_of(period as usize) {
                        errors.push(KernelScheduleError::IllegalSwizzle {
                            region: region.name.clone(),
                            period,
                            innermost,
                        });
                    }
                }

                // Fits inside the region it views.
                let region_elems: usize = region.dims.iter().product();
                let extent = layout.extent_elems(&region.dims);
                let off_elems = layout.offset_bytes / region.dtype.size_bytes().max(1);
                if off_elems + extent > region_elems {
                    errors.push(KernelScheduleError::LayoutOutOfRegion {
                        region: region.name.clone(),
                        role: role.clone(),
                        extent_elems: off_elems + extent,
                        region_elems,
                    });
                }

                // Agrees with every other commitment on the same region.
                match committed.get(acc.region.as_str()) {
                    Some((other_role, other)) if !layout.agrees_with(other) => {
                        errors.push(KernelScheduleError::LayoutDisagreement {
                            region: acc.region.clone(),
                            a: other_role.clone(),
                            b: role.clone(),
                            detail: format!(
                                "strides {:?}/swizzle {:?} vs strides {:?}/swizzle {:?}",
                                other.strides, other.swizzle, layout.strides, layout.swizzle
                            ),
                        });
                    }
                    Some(_) => {}
                    None => {
                        committed.insert(acc.region.as_str(), (role.clone(), layout));
                    }
                }
            }
        }
    }
}

/// Metadata derived from the declarations during lowering.
///
/// CAKE §2.2's fourth property: "barrier addresses, phase bits, TMEM offsets,
/// descriptor encodings, and warp identity are all computed from the
/// declarations rather than written out by the agent." Nothing here is stated
/// by the schedule author, so none of it can drift from what was declared.
#[derive(Debug, Clone, PartialEq)]
pub struct Lowered {
    /// Byte offset of each region within its space.
    pub region_offsets: BTreeMap<String, usize>,
    /// Barrier index, assigned in declaration order.
    pub barrier_slots: BTreeMap<String, u32>,
    /// First warp of each role — the role's identity at run time.
    pub role_base_warp: BTreeMap<String, u32>,
    /// Total shared bytes the launch must reserve.
    pub shared_bytes: usize,
    /// Threads per block implied by the roles.
    pub threads: usize,
}

/// Derive the mechanical consequences of a schedule's declarations.
///
/// Deliberately NOT a code generator: it emits the numbers a backend would
/// need, not MSL or PTX. rlx already has five hand-written codegen paths, and
/// replacing them is a far larger commitment than this module has earned. What
/// this demonstrates is the property that matters — that the addresses, slots
/// and identities are *computed*, so an agent (or a human) never writes them
/// down and never gets them wrong.
///
/// Returns `Err` with the verifier's findings when the schedule is invalid:
/// lowering an unverified schedule would derive metadata for a program that
/// cannot run.
pub fn lower(sched: &KernelSchedule, target: Target) -> Result<Lowered, Vec<KernelScheduleError>> {
    let errors = verify_kernel_schedule(sched, target);
    if !errors.is_empty() {
        return Err(errors);
    }
    const WARP_LANES: usize = 32;

    let mut region_offsets = BTreeMap::new();
    let mut cursor = 0usize;
    for r in sched.regions.iter().filter(|r| r.space == Space::Shared) {
        // Align each region to its element size so a swizzled view never
        // straddles a bank boundary it did not intend to.
        let align = r.dtype.size_bytes().max(1);
        cursor = cursor.div_ceil(align) * align;
        region_offsets.insert(r.name.clone(), cursor);
        cursor += r.bytes();
    }
    let shared_bytes = cursor;

    let barrier_slots = sched
        .barriers
        .iter()
        .enumerate()
        .map(|(i, b)| (b.name.clone(), i as u32))
        .collect();

    let role_base_warp = sched
        .roles
        .iter()
        .filter_map(|r| r.warps.iter().min().map(|w| (r.name.clone(), *w)))
        .collect();

    Ok(Lowered {
        region_offsets,
        barrier_slots,
        role_base_warp,
        shared_bytes,
        threads: sched.warps().len() * WARP_LANES,
    })
}
