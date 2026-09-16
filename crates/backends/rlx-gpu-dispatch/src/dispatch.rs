// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU dispatch table — **shape-keyed** kernel-variant selection.
//!
//! `rlx-cpu`'s `dispatch.rs` already states the principle: *kernel-variant
//! selection is a data lookup, not scattered match arms in dispatch sites*, with
//! compile-time per-arch defaults and runtime overrides filled by measurement.
//! That table is CPU-only. The GPU backends kept the scattered form — a global
//! `use_wmma()` flag, one hand-picked `attention_wmma_min_work` constant applied
//! to every shape on every arch — even though they already ship several physical
//! schedules per logical op (`matmul` / `_bt` / `_wmma` / `_tma`; `attention` /
//! `_row` / `_warp` / `_wmma` / `_wmma_d128`). Those *are* dispatch families
//! routed by a magic number.
//!
//! This is the GPU twin of that table, with one addition the CPU version does
//! not need: the key includes a **shape bucket**. A GEMM at `m=1` (decode) and a
//! GEMM at `m=4096` (prefill) want different physical schedules, and a single
//! threshold cannot express that.
//!
//! Three layers, in increasing authority:
//!
//! 1. [`default_choice`] — compile-time defaults per `(arch, op, bucket)`. Every
//!    default reproduces the historical hand-written decision, so an untuned
//!    process behaves exactly as before.
//! 2. Runtime **overrides** ([`set_override`]) — what a measured tuning run
//!    learned. Persisted with [`save_overrides`] / [`load_overrides`] so the
//!    measurement is paid once per machine.
//! 3. Forced `Override`s from env — an operator pinning a variant for A/B.
//!
//! Shared by `rlx-cuda` and `rlx-rocm` because the kernel sources are shared.

use crate::tiles::TileParams;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

// ── Keys ────────────────────────────────────────────────────────────────────

/// A GPU architecture, at the granularity where the best schedule actually
/// changes. Free-form rather than an enum: new silicon should add a row to a
/// table, not a variant to a type every backend matches on.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GpuArch(pub String);

impl GpuArch {
    /// NVIDIA compute capability, e.g. `(8, 6)` → `sm_86`.
    pub fn cuda(major: u32, minor: u32) -> Self {
        Self(format!("sm_{major}{minor}"))
    }
    /// AMD GCN/RDNA target, e.g. `gfx908`.
    pub fn rocm(gfx: &str) -> Self {
        Self(gfx.to_string())
    }
    /// Apple GPU family, e.g. `apple9` — the granularity at which Metal's own
    /// cost model already varies (`MetalHwModel::gpu_family`).
    pub fn metal(family: &str) -> Self {
        Self(format!("metal-{family}"))
    }
    /// A wgpu adapter. wgpu runs over Metal / Vulkan / DX12 / GL and the best
    /// path differs per *backend* as much as per chip, so both go in the key.
    pub fn wgpu(backend: &str, adapter: &str) -> Self {
        // Adapter strings carry spaces and vendor punctuation; the cache format
        // is tab-separated, so only tabs and newlines actually need removing.
        let clean: String = adapter
            .chars()
            .map(|c| if c == '\t' || c == '\n' { ' ' } else { c })
            .collect();
        Self(format!("wgpu-{backend}-{}", clean.trim()))
    }
    /// Fallback for a device we have no tuned row for.
    pub fn unknown() -> Self {
        Self("unknown".to_string())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    /// True for NVIDIA `sm_*` targets.
    pub fn is_cuda(&self) -> bool {
        self.0.starts_with("sm_")
    }
}

/// Op family a dispatch decision is keyed against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GpuOp {
    /// Dense fp32 GEMM (`matmul`, `matmul_bt`, `matmul_wmma`, `matmul_tma`).
    Matmul,
    /// SDPA (`attention`, `attention_row`, `attention_warp`, `attention_wmma*`).
    Attention,
}

impl GpuOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Matmul => "matmul",
            Self::Attention => "attention",
        }
    }
}

/// A coarse bucket over the problem *shape* — one `floor(log2)` band per axis.
///
/// **This used to be `floor(log2(m·k·n))`, a single number, and that was wrong.**
/// Total work is invariant to aspect ratio, so one bucket held every shape with
/// the same product:
///
/// ```text
/// m=1        k=4096  n=4096   ← decode GEMV
/// m=4096     k=4096  n=1      ← tall-skinny, the opposite schedule
/// m=64       k=4096  n=64     ← squarish
/// m=16777216 k=1     n=1      ← a vector
/// ```
///
/// All four landed in bucket 24, so a tuner that measured *one* of them
/// installed its winner for all four. Aspect ratio is the entire
/// decode-vs-prefill distinction the bucket exists to capture, which made the
/// collapsed key precisely useless for its own purpose. Per-axis bands keep
/// those four apart while staying coarse enough that a measured table
/// generalizes — CAKE's split: tune an exact shape, then group measured seeds
/// into shape buckets behind an explicit fallback.
///
/// Note the resolution a bucket still costs. The historical attention threshold
/// is `batch·heads·seq_q >= 12288`, which is not a band edge — 8192 and 12288
/// share a band. That is why [`default_choice`] takes the exact [`Workload`]
/// rather than the bucket: **defaults stay bit-for-bit historical, only measured
/// overrides are bucketed.**
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShapeBucket(pub [u8; BUCKET_AXES]);

/// Axes a bucket discriminates on. Three covers GEMM's `(m, k, n)`; attention
/// uses two and leaves the third zero.
pub const BUCKET_AXES: usize = 3;

/// Per-axis bands above this collapse into the last one.
pub const MAX_BUCKET: u8 = 40;

impl ShapeBucket {
    /// `floor(log2(x))` for one axis, clamped to [`MAX_BUCKET`]. `0` and `1`
    /// both map to band 0 — a size-1 axis (decode's `m`, a GEMV's `n`) is the
    /// case worth separating, and it lands in its own band by construction.
    pub fn band(x: u64) -> u8 {
        if x == 0 {
            return 0;
        }
        (63 - x.leading_zeros() as u8).min(MAX_BUCKET)
    }

    /// Bucket from per-axis sizes, most significant axis first.
    pub fn of_axes(axes: [u64; BUCKET_AXES]) -> Self {
        Self([
            Self::band(axes[0]),
            Self::band(axes[1]),
            Self::band(axes[2]),
        ])
    }

    /// Stable text form for the persisted cache: `band.band.band`.
    pub fn encode(&self) -> String {
        format!("{}.{}.{}", self.0[0], self.0[1], self.0[2])
    }

    /// Inverse of [`ShapeBucket::encode`].
    ///
    /// Returns `None` for the old single-number form, which is what makes a v1
    /// cache degrade to defaults instead of being silently misread: `"24"` has
    /// one field where three are required, so it cannot be mistaken for a band
    /// triple.
    pub fn decode(s: &str) -> Option<Self> {
        let mut it = s.split('.');
        let a = it.next()?.parse().ok()?;
        let b = it.next()?.parse().ok()?;
        let c = it.next()?.parse().ok()?;
        if it.next().is_some() {
            return None;
        }
        Some(Self([a, b, c]))
    }
}

/// An exact workload — what dispatch is actually deciding about.
///
/// Carries the raw dimensions, not a bucket, so a default can reproduce a
/// non-power-of-two threshold exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Workload {
    /// Dense GEMM: `[m,k] · [k,n]`.
    Matmul { m: usize, k: usize, n: usize },
    /// SDPA. The size metric is `batch·heads·seq_q` — the count of query rows,
    /// which is what the historical WMMA threshold measures ("is the launch big
    /// enough to amortize the per-block setup"). `head_dim` and the mask are
    /// *eligibility* gates in the backend, not tuning inputs, so they stay out
    /// of the key.
    Attention {
        batch: usize,
        heads: usize,
        seq_q: usize,
    },
}

impl Workload {
    /// Which op family this workload dispatches within.
    pub fn op(&self) -> GpuOp {
        match self {
            Self::Matmul { .. } => GpuOp::Matmul,
            Self::Attention { .. } => GpuOp::Attention,
        }
    }

    /// The scalar work metric the bucket and the historical thresholds use.
    pub fn work(&self) -> u64 {
        match self {
            Self::Matmul { m, k, n } => (*m as u64)
                .saturating_mul(*k as u64)
                .saturating_mul(*n as u64),
            Self::Attention {
                batch,
                heads,
                seq_q,
            } => (*batch as u64)
                .saturating_mul(*heads as u64)
                .saturating_mul(*seq_q as u64),
        }
    }

    /// The coarse key an override is stored under.
    ///
    /// GEMM discriminates on all three of `(m, k, n)` — the aspect ratio *is*
    /// the schedule decision. Attention discriminates on `(batch·heads, seq_q)`:
    /// the row count and the per-row work are independent knobs, and decode
    /// (`seq_q = 1`) versus prefill (`seq_q = 4096`) at the same `batch·heads`
    /// is exactly the split a single product would erase.
    pub fn bucket(&self) -> ShapeBucket {
        match self {
            Self::Matmul { m, k, n } => ShapeBucket::of_axes([*m as u64, *k as u64, *n as u64]),
            Self::Attention {
                batch,
                heads,
                seq_q,
            } => ShapeBucket::of_axes([
                (*batch as u64).saturating_mul(*heads as u64),
                *seq_q as u64,
                0,
            ]),
        }
    }

    /// The full override key for this workload on `arch`.
    pub fn key(&self, arch: &GpuArch) -> DispatchKey {
        DispatchKey {
            arch: arch.clone(),
            op: self.op(),
            bucket: self.bucket(),
        }
    }
}

/// The full lookup key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DispatchKey {
    pub arch: GpuArch,
    pub op: GpuOp,
    pub bucket: ShapeBucket,
}

// ── Values ──────────────────────────────────────────────────────────────────

/// Metal's dense-sgemm schedules. Mirrors `rlx_metal::cost::SgemmVariant`.
///
/// Metal has no scalar tiled `matmul` kernel to parameterize — dense f32 GEMM
/// runs through MPS or one of seven hand-written simdgroup kernels — so the
/// tunable decision there is *which variant*, not *which tile*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalSgemm {
    Mps,
    Simd4x4,
    Simd64,
    Simd64SplitK,
    Simd,
    SimdPadded,
    Tiled,
    Naive,
}

/// wgpu's dense-matmul compute paths. Mirrors `rlx_wgpu::backend::MatmulCompute`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WgpuMatmul {
    F32,
    F16,
    Coop16,
    CoopF32,
    CoopF16Vk,
    Bf16Packed,
}

/// Which physical schedule to run, and how it is parameterized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Choice {
    /// The scalar tiled `matmul` kernel with an explicit tile.
    MatmulTiled(TileParams),
    /// The WMMA tensor-core `matmul_wmma` kernel (fixed tile, no epilogue).
    MatmulWmma,
    /// A Metal dense-sgemm variant.
    MetalSgemm(MetalSgemm),
    /// A wgpu dense-matmul compute path.
    WgpuMatmul(WgpuMatmul),
    /// The scalar per-element `attention` kernel.
    AttentionScalar,
    /// The row-parallel `attention_row` kernel.
    AttentionRow,
    /// The warp-per-row `attention_warp` kernel.
    AttentionWarp,
    /// The WMMA attention kernel (`head_dim` selects the d64 / d128 entry).
    AttentionWmma,
}

impl MetalSgemm {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Mps => "mps",
            Self::Simd4x4 => "simd4x4",
            Self::Simd64 => "simd64",
            Self::Simd64SplitK => "simd64_splitk",
            Self::Simd => "simd",
            Self::SimdPadded => "simd_padded",
            Self::Tiled => "tiled",
            Self::Naive => "naive",
        }
    }
    fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "mps" => Self::Mps,
            "simd4x4" => Self::Simd4x4,
            "simd64" => Self::Simd64,
            "simd64_splitk" => Self::Simd64SplitK,
            "simd" => Self::Simd,
            "simd_padded" => Self::SimdPadded,
            "tiled" => Self::Tiled,
            "naive" => Self::Naive,
            _ => return None,
        })
    }

    /// Shape constraints the *kernel* requires, independent of any tuning
    /// preference. **Not** advisory: `Simd4x4` writes a full 32×32 tile with no
    /// bottom-row mask, so running it at `m % 32 != 0` overruns C and corrupts
    /// whatever sits next in the arena. The backend's cascade already encodes
    /// these; naming them here lets an override be checked against the same
    /// rule instead of a second copy of it.
    ///
    /// Device-level eligibility (is MPS present? enough occupancy?) is *not*
    /// here — that lives with the backend, which is the only thing that can
    /// answer it.
    pub fn shape_eligible(&self, m: usize, k: usize, n: usize) -> bool {
        let a = |x: usize, d: usize| x.is_multiple_of(d);
        match self {
            Self::Mps | Self::Naive | Self::SimdPadded => true,
            Self::Simd4x4 => a(m, 32) && a(k, 32) && a(n, 32),
            Self::Simd64 => a(m, 64) && a(n, 64) && a(k, 8),
            // The split count itself is the backend's `pick_ksplits`; here we
            // only assert the tile alignment it also requires.
            Self::Simd64SplitK => a(m, 64) && a(n, 64) && a(k, 8),
            Self::Simd => a(m, 8) && a(k, 8) && a(n, 8),
            Self::Tiled => m >= 16 && n >= 16,
        }
    }
}

impl WgpuMatmul {
    fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Coop16 => "coop16",
            Self::CoopF32 => "coop_f32",
            Self::CoopF16Vk => "coop_f16_vk",
            Self::Bf16Packed => "bf16_packed",
        }
    }
    fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "f32" => Self::F32,
            "f16" => Self::F16,
            "coop16" => Self::Coop16,
            "coop_f32" => Self::CoopF32,
            "coop_f16_vk" => Self::CoopF16Vk,
            "bf16_packed" => Self::Bf16Packed,
            _ => return None,
        })
    }
}

impl Choice {
    /// The op family this choice belongs to — used to reject a mismatched
    /// override before it can route an attention shape to a GEMM kernel.
    pub fn op(&self) -> GpuOp {
        match self {
            Self::MatmulTiled(_) | Self::MatmulWmma | Self::MetalSgemm(_) | Self::WgpuMatmul(_) => {
                GpuOp::Matmul
            }
            Self::AttentionScalar
            | Self::AttentionRow
            | Self::AttentionWarp
            | Self::AttentionWmma => GpuOp::Attention,
        }
    }

    /// Stable text form for the persisted tuning cache.
    pub fn encode(&self) -> String {
        match self {
            Self::MatmulTiled(t) => format!("matmul_tiled:{}", t.label()),
            Self::MatmulWmma => "matmul_wmma".to_string(),
            Self::MetalSgemm(v) => format!("metal_sgemm:{}", v.as_str()),
            Self::WgpuMatmul(v) => format!("wgpu_matmul:{}", v.as_str()),
            Self::AttentionScalar => "attention_scalar".to_string(),
            Self::AttentionRow => "attention_row".to_string(),
            Self::AttentionWarp => "attention_warp".to_string(),
            Self::AttentionWmma => "attention_wmma".to_string(),
        }
    }

    /// Inverse of [`Choice::encode`]. Unknown text is `None` rather than a
    /// panic: a cache written by a newer build must not break an older one.
    pub fn decode(s: &str) -> Option<Self> {
        if let Some(label) = s.strip_prefix("matmul_tiled:") {
            let t = decode_tile_label(label)?;
            // A persisted tile from another build could be illegal here.
            t.validate().ok()?;
            return Some(Self::MatmulTiled(t));
        }
        if let Some(v) = s.strip_prefix("metal_sgemm:") {
            return MetalSgemm::from_str(v).map(Self::MetalSgemm);
        }
        if let Some(v) = s.strip_prefix("wgpu_matmul:") {
            return WgpuMatmul::from_str(v).map(Self::WgpuMatmul);
        }
        match s {
            "matmul_wmma" => Some(Self::MatmulWmma),
            "attention_scalar" => Some(Self::AttentionScalar),
            "attention_row" => Some(Self::AttentionRow),
            "attention_warp" => Some(Self::AttentionWarp),
            "attention_wmma" => Some(Self::AttentionWmma),
            _ => None,
        }
    }

    /// The Metal variant this choice names, if any — the shape a backend uses
    /// to ask "did the table pick something for me?" without matching on every
    /// other backend's variants.
    pub fn as_metal_sgemm(&self) -> Option<MetalSgemm> {
        match self {
            Self::MetalSgemm(v) => Some(*v),
            _ => None,
        }
    }

    /// The wgpu compute path this choice names, if any.
    pub fn as_wgpu_matmul(&self) -> Option<WgpuMatmul> {
        match self {
            Self::WgpuMatmul(v) => Some(*v),
            _ => None,
        }
    }
}

/// Parse `64x64x16_t4x4_b16x16` back into a [`TileParams`].
fn decode_tile_label(s: &str) -> Option<TileParams> {
    let mut parts = s.split('_');
    let dims = parts.next()?;
    let t = parts.next()?.strip_prefix('t')?;
    let b = parts.next()?.strip_prefix('b')?;
    if parts.next().is_some() {
        return None;
    }
    let mut d = dims.split('x');
    let (bm, bn, bk) = (
        d.next()?.parse().ok()?,
        d.next()?.parse().ok()?,
        d.next()?.parse().ok()?,
    );
    if d.next().is_some() {
        return None;
    }
    let mut tt = t.split('x');
    let (tm, tn) = (tt.next()?.parse().ok()?, tt.next()?.parse().ok()?);
    if tt.next().is_some() {
        return None;
    }
    let mut bb = b.split('x');
    let (bdx, bdy) = (bb.next()?.parse().ok()?, bb.next()?.parse().ok()?);
    if bb.next().is_some() {
        return None;
    }
    Some(TileParams {
        bm,
        bn,
        bk,
        tm,
        tn,
        bdx,
        bdy,
    })
}

// ── Compile-time defaults ───────────────────────────────────────────────────

/// The default schedule for a workload, reproducing today's hand-written
/// routing **exactly**.
///
/// Matmul: the historical 64×64×16 tile at every shape — the single hardcoded
/// `#define` block, expressed as data. WMMA stays opt-in (it is a global flag
/// today, not a shape decision) so the default table never routes to it.
///
/// Attention: the historical `batch·heads·seq_q >= 12288` boundary between the
/// scalar/row family and WMMA. `attention_warp` is the measured-better member of
/// the row family below that boundary (it took `attention_row` from 82.6% to
/// 17.2% of decode GPU time) — exactly the kind of decision that belongs in a
/// table rather than an `if`. Eligibility (head_dim, mask kind, softcap, sm70+)
/// stays with the backend: those are capability gates, not tuning.
///
/// `arch` is unused today — every default is arch-independent, which is itself
/// the finding: nothing in the current routing adapts to the device. It stays in
/// the signature because adding a per-arch default should be a table row, not a
/// signature change at every call site.
pub fn default_choice(arch: &GpuArch, workload: &Workload) -> Choice {
    let _ = arch;
    match workload {
        Workload::Matmul { .. } => Choice::MatmulTiled(TileParams::DEFAULT_MATMUL),
        Workload::Attention { .. } => {
            if workload.work() >= ATTENTION_WMMA_MIN_WORK {
                Choice::AttentionWmma
            } else {
                Choice::AttentionWarp
            }
        }
    }
}

/// The historical `RLX_CUDA_ATTENTION_WMMA_MIN_WORK` default. Exact, not a
/// bucket edge — see [`ShapeBucket`].
pub const ATTENTION_WMMA_MIN_WORK: u64 = 12_288;

/// Every Metal sgemm variant, for tuners to enumerate.
pub const METAL_SGEMM_VARIANTS: &[MetalSgemm] = &[
    MetalSgemm::Mps,
    MetalSgemm::Simd4x4,
    MetalSgemm::Simd64,
    MetalSgemm::Simd64SplitK,
    MetalSgemm::Simd,
    MetalSgemm::SimdPadded,
    MetalSgemm::Tiled,
    MetalSgemm::Naive,
];

/// Every wgpu matmul compute path, for tuners to enumerate.
pub const WGPU_MATMUL_VARIANTS: &[WgpuMatmul] = &[
    WgpuMatmul::F32,
    WgpuMatmul::F16,
    WgpuMatmul::Coop16,
    WgpuMatmul::CoopF32,
    WgpuMatmul::CoopF16Vk,
    WgpuMatmul::Bf16Packed,
];

// ── Runtime override table ──────────────────────────────────────────────────

/// A measured override plus the evidence behind it.
///
/// The table used to store a bare [`Choice`], which made it impossible to answer
/// the question a route report exists to answer: *why* is this shape on this
/// schedule, and how much did it actually buy? CAKE's Figure 9 reports a
/// per-route speedup precisely because the aggregate hides the spread — its
/// Flash-KMeans routes range from 1.037× to 3.757× under an overall 1.803×, so a
/// single number would conceal both the big win and the routes doing nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverrideRecord {
    pub choice: Choice,
    /// Measured speedup over the compile-time default on the tuned shape.
    /// `None` for an override installed without timing evidence (a hand-pinned
    /// cache row, or a loader reading a record that predates this column).
    pub speedup: Option<f32>,
    /// How many held-out shapes in the same bucket confirmed the winner. `0`
    /// means the override was never checked outside the shape it was tuned on —
    /// worth surfacing, since that is the leakage CAKE §6 warns about.
    pub holdouts: u16,
}

impl OverrideRecord {
    /// An override with no timing evidence attached.
    pub fn bare(choice: Choice) -> Self {
        Self {
            choice,
            speedup: None,
            holdouts: 0,
        }
    }
}

#[derive(Default)]
struct Table {
    overrides: Mutex<BTreeMap<DispatchKey, OverrideRecord>>,
}

fn table() -> &'static Table {
    static T: OnceLock<Table> = OnceLock::new();
    T.get_or_init(Table::default)
}

/// Record a measured winner for `(arch, op, bucket)`.
///
/// Rejected — with the reason — when the choice belongs to a different op
/// family or names an illegal tile. A tuning run that produced garbage must not
/// be able to install a schedule that reads out of bounds; this is the same
/// pre-execution gate discipline as [`crate::tiles::TileParams::validate`],
/// applied at the table's write edge rather than the compiler's.
pub fn set_override(key: DispatchKey, choice: Choice) -> Result<(), OverrideError> {
    set_override_measured(key, OverrideRecord::bare(choice))
}

/// Record a measured winner together with the evidence behind it — see
/// [`OverrideRecord`]. `set_override` is the evidence-free shorthand.
pub fn set_override_measured(
    key: DispatchKey,
    record: OverrideRecord,
) -> Result<(), OverrideError> {
    let choice = record.choice;
    if choice.op() != key.op {
        return Err(OverrideError::OpMismatch {
            key_op: key.op,
            choice_op: choice.op(),
        });
    }
    if let Choice::MatmulTiled(t) = choice {
        t.validate()
            .map_err(|e| OverrideError::IllegalTile(e.to_string()))?;
    }
    table()
        .overrides
        .lock()
        .expect("gpu dispatch table poisoned")
        .insert(key, record);
    Ok(())
}

/// Why an override was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverrideError {
    /// The choice runs a different op family than the key selects.
    OpMismatch { key_op: GpuOp, choice_op: GpuOp },
    /// The choice names a tile that violates a kernel invariant.
    IllegalTile(String),
}

impl std::fmt::Display for OverrideError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpMismatch { key_op, choice_op } => write!(
                f,
                "override for {} names a {} kernel",
                key_op.as_str(),
                choice_op.as_str()
            ),
            Self::IllegalTile(e) => write!(f, "override names an illegal tile: {e}"),
        }
    }
}

impl std::error::Error for OverrideError {}

/// Resolve the schedule for a workload: the measured override for its bucket if
/// one exists, else the exact compile-time default.
pub fn resolve(arch: &GpuArch, workload: &Workload) -> Choice {
    if let Some(c) = table()
        .overrides
        .lock()
        .expect("gpu dispatch table poisoned")
        .get(&workload.key(arch))
    {
        return c.choice;
    }
    default_choice(arch, workload)
}

/// Convenience: resolve a GEMM by its dimensions.
pub fn resolve_matmul(arch: &GpuArch, m: usize, k: usize, n: usize) -> Choice {
    resolve(arch, &Workload::Matmul { m, k, n })
}

/// Convenience: resolve attention by its query-row dimensions.
pub fn resolve_attention(arch: &GpuArch, batch: usize, heads: usize, seq_q: usize) -> Choice {
    resolve(
        arch,
        &Workload::Attention {
            batch,
            heads,
            seq_q,
        },
    )
}

/// Drop every override (test hook, and the reset a re-tune wants).
pub fn clear_overrides() {
    table()
        .overrides
        .lock()
        .expect("gpu dispatch table poisoned")
        .clear();
}

/// Snapshot the override table, sorted — for persisting and for reporting what
/// a tuning run actually changed.
pub fn overrides() -> Vec<(DispatchKey, OverrideRecord)> {
    table()
        .overrides
        .lock()
        .expect("gpu dispatch table poisoned")
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect()
}

// ── Persistence ─────────────────────────────────────────────────────────────

/// Serialize the override table to the on-disk tuning-cache format.
///
/// One `arch\top\tbucket\tchoice\tspeedup\tholdouts` record per line —
/// deliberately not JSON, so the file needs no serde dependency in this crate and
/// stays diffable and hand-editable when someone wants to pin a schedule.
///
/// The two evidence columns are **optional trailing fields**: the loader reads
/// four and ignores the rest, so a hand-written 4-column row still applies and an
/// older build reading a newer file is unaffected. No version bump needed for an
/// additive tail — unlike the v1→v2 bucket change, which altered the meaning of an
/// existing column and therefore had to be rejected outright.
pub fn save_overrides() -> String {
    // v2: the bucket column became a per-axis band triple (`m.k.n`). A v1 cache
    // holds single numbers there, which `ShapeBucket::decode` rejects — so an
    // old file degrades to compile-time defaults rather than being reinterpreted
    // under the new key, where a decode-tuned entry could land on a prefill shape.
    let mut s = String::from(
        "# rlx gpu dispatch tuning cache v2\n# arch\top\tbucket\tchoice\tspeedup\tholdouts\n",
    );
    for (k, r) in overrides() {
        s.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            k.arch.as_str(),
            k.op.as_str(),
            k.bucket.encode(),
            r.choice.encode(),
            r.speedup
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "-".into()),
            r.holdouts,
        ));
    }
    s
}

/// Load overrides written by [`save_overrides`]. Returns the number applied.
///
/// Unparseable and illegal records are **skipped, not fatal**: a stale cache
/// from an older build must degrade to the compile-time defaults rather than
/// take the process down. Skips are reported so a tuner can notice its cache
/// went stale.
pub fn load_overrides(text: &str) -> LoadReport {
    let mut report = LoadReport::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut f = line.split('\t');
        let (Some(arch), Some(op), Some(bucket), Some(choice)) =
            (f.next(), f.next(), f.next(), f.next())
        else {
            report.skipped += 1;
            continue;
        };
        let op = match op {
            "matmul" => GpuOp::Matmul,
            "attention" => GpuOp::Attention,
            _ => {
                report.skipped += 1;
                continue;
            }
        };
        let (Some(bucket), Some(choice)) = (ShapeBucket::decode(bucket), Choice::decode(choice))
        else {
            report.skipped += 1;
            continue;
        };
        // Optional evidence columns; absent or unparseable means "no evidence
        // recorded", which is different from "zero speedup" and is reported as
        // such by `explain`.
        let speedup = f.next().and_then(|v| v.parse::<f32>().ok());
        let holdouts = f.next().and_then(|v| v.parse::<u16>().ok()).unwrap_or(0);
        let key = DispatchKey {
            arch: GpuArch(arch.to_string()),
            op,
            bucket,
        };
        match set_override_measured(
            key,
            OverrideRecord {
                choice,
                speedup,
                holdouts,
            },
        ) {
            Ok(()) => report.applied += 1,
            Err(_) => report.skipped += 1,
        }
    }
    report
}

/// Outcome of [`load_overrides`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LoadReport {
    pub applied: usize,
    pub skipped: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiles::MATMUL_TILE_CANDIDATES;

    /// `resolve` reads a process-global table.
    static LOCK: Mutex<()> = Mutex::new(());

    fn clean(f: impl FnOnce()) {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_overrides();
        f();
        clear_overrides();
    }

    fn arch() -> GpuArch {
        GpuArch::cuda(8, 6)
    }

    fn key(op: GpuOp, bucket: [u8; BUCKET_AXES]) -> DispatchKey {
        DispatchKey {
            arch: arch(),
            op,
            bucket: ShapeBucket(bucket),
        }
    }

    /// A GEMM sitting in the band triple `[0, 0, lg]`.
    fn gemm_in_bucket(lg: u8) -> Workload {
        let w = Workload::Matmul {
            m: 1,
            k: 1,
            n: 1usize << lg,
        };
        assert_eq!(w.bucket(), ShapeBucket([0, 0, lg]));
        w
    }

    #[test]
    fn bands_are_floor_log2_per_axis() {
        assert_eq!(ShapeBucket::band(0), 0);
        assert_eq!(ShapeBucket::band(1), 0);
        assert_eq!(ShapeBucket::band(2), 1);
        assert_eq!(ShapeBucket::band(1023), 9);
        assert_eq!(ShapeBucket::band(1024), 10);
        assert_eq!(ShapeBucket::band(u64::MAX), MAX_BUCKET);
        assert_eq!(
            Workload::Matmul {
                m: 1,
                k: 4096,
                n: 4096
            }
            .bucket(),
            ShapeBucket([0, 12, 12])
        );
    }

    /// Decode (m=1) and prefill (m=4096) at the same weight must not collide —
    /// that separation is the whole reason the key carries a shape bucket.
    #[test]
    fn decode_and_prefill_land_in_different_buckets() {
        let decode = Workload::Matmul {
            m: 1,
            k: 4096,
            n: 4096,
        }
        .bucket();
        let prefill = Workload::Matmul {
            m: 4096,
            k: 4096,
            n: 4096,
        }
        .bucket();
        assert_ne!(decode, prefill);
        assert!(prefill > decode);
    }

    /// THE regression this key exists for. Under the old single-number bucket
    /// (`floor(log2(m*k*n))`) all four of these shared bucket 24, so a tuner that
    /// measured the decode GEMV installed its winner for a tall-skinny GEMM, a
    /// square one, and a plain vector. Every pair must now be distinct.
    #[test]
    fn equal_work_opposite_aspect_ratios_do_not_share_a_bucket() {
        let shapes = [
            (1usize, 4096usize, 4096usize), // decode GEMV
            (4096, 4096, 1),                // tall-skinny
            (64, 4096, 64),                 // squarish
            (1 << 24, 1, 1),                // a vector
        ];
        // Same total work, by construction.
        let work: Vec<u64> = shapes
            .iter()
            .map(|(m, k, n)| {
                Workload::Matmul {
                    m: *m,
                    k: *k,
                    n: *n,
                }
                .work()
            })
            .collect();
        assert!(
            work.iter().all(|w| *w == work[0]),
            "test premise broken: shapes must have equal work, got {work:?}"
        );
        // …and pairwise-distinct buckets.
        let buckets: Vec<ShapeBucket> = shapes
            .iter()
            .map(|(m, k, n)| {
                Workload::Matmul {
                    m: *m,
                    k: *k,
                    n: *n,
                }
                .bucket()
            })
            .collect();
        for i in 0..buckets.len() {
            for j in (i + 1)..buckets.len() {
                assert_ne!(
                    buckets[i], buckets[j],
                    "{:?} and {:?} still collide at {:?}",
                    shapes[i], shapes[j], buckets[i]
                );
            }
        }
    }

    /// Attention: decode and prefill at the same `batch*heads` must separate.
    #[test]
    fn attention_decode_and_prefill_do_not_share_a_bucket() {
        let decode = Workload::Attention {
            batch: 1,
            heads: 32,
            seq_q: 1,
        }
        .bucket();
        let prefill = Workload::Attention {
            batch: 1,
            heads: 32,
            seq_q: 4096,
        }
        .bucket();
        assert_ne!(decode, prefill);
    }

    /// A v1 cache (single-number bucket column) must be REJECTED, not reread
    /// under the new key — a decode-tuned v1 entry reinterpreted as a band
    /// triple could land on an unrelated shape.
    #[test]
    fn v1_cache_records_are_rejected() {
        assert_eq!(ShapeBucket::decode("24"), None);
        assert_eq!(
            ShapeBucket::decode("0.12.12"),
            Some(ShapeBucket([0, 12, 12]))
        );
        assert_eq!(ShapeBucket::decode("0.12.12.5"), None);
        clean(|| {
            let v1 = "# rlx gpu dispatch tuning cache v1\n# arch\top\tbucket\tchoice\n\
                      sm_86\tmatmul\t24\tmatmul_tiled:32x32x32_t2x2_b16x16\n";
            let r = load_overrides(v1);
            assert_eq!(r.applied, 0, "a v1 record must not be applied");
            assert_eq!(r.skipped, 1);
        });
    }

    /// The default must reproduce the historical routing EXACTLY, or making
    /// dispatch data-driven would itself be a behavior change.
    #[test]
    fn defaults_reproduce_the_historical_routing() {
        // Matmul: the one hardcoded tile, at every shape.
        for w in [
            Workload::Matmul { m: 1, k: 1, n: 1 },
            Workload::Matmul {
                m: 1,
                k: 4096,
                n: 4096,
            },
            Workload::Matmul {
                m: 8192,
                k: 8192,
                n: 8192,
            },
        ] {
            assert_eq!(
                default_choice(&arch(), &w),
                Choice::MatmulTiled(TileParams::DEFAULT_MATMUL)
            );
        }
        // Attention: the exact `batch*heads*seq_q >= 12288` boundary.
        let at = |batch, heads, seq_q| {
            default_choice(
                &arch(),
                &Workload::Attention {
                    batch,
                    heads,
                    seq_q,
                },
            )
        };
        assert_eq!(at(1, 12, 1024), Choice::AttentionWmma); // 12288, exactly at
        assert_eq!(at(1, 12, 1023), Choice::AttentionWarp); // 12276, just below
        assert_eq!(at(1, 16, 1), Choice::AttentionWarp); // decode
    }

    /// The exact threshold is not a band edge — `heads=8` and `heads=12` at the
    /// same `seq_q` share a band yet route differently. This is the resolution a
    /// bucket gives up, and the reason defaults consume the exact workload.
    #[test]
    fn bucket_resolution_is_coarser_than_the_default_threshold() {
        assert_eq!(ShapeBucket::band(8), ShapeBucket::band(12));
        let below = Workload::Attention {
            batch: 1,
            heads: 8,
            seq_q: 1024,
        }; // 8192
        let above = Workload::Attention {
            batch: 1,
            heads: 12,
            seq_q: 1024,
        }; // 12288
        assert_eq!(below.bucket(), above.bucket());
        assert_ne!(
            default_choice(&arch(), &below),
            default_choice(&arch(), &above)
        );
    }

    #[test]
    fn override_wins_over_default_for_its_key_only() {
        clean(|| {
            let tuned = MATMUL_TILE_CANDIDATES[0];
            let w = gemm_in_bucket(12);
            set_override(w.key(&arch()), Choice::MatmulTiled(tuned)).expect("legal");
            assert_eq!(resolve(&arch(), &w), Choice::MatmulTiled(tuned));
            // A neighbouring bucket is untouched.
            assert_eq!(
                resolve(&arch(), &gemm_in_bucket(13)),
                Choice::MatmulTiled(TileParams::DEFAULT_MATMUL)
            );
            // …and so is the same bucket on another device.
            assert_eq!(
                resolve(&GpuArch::rocm("gfx908"), &w),
                Choice::MatmulTiled(TileParams::DEFAULT_MATMUL)
            );
        });
    }

    /// The write-edge gate: a tuner that produced nonsense cannot install a
    /// schedule that would read out of bounds on the GPU.
    #[test]
    fn rejects_illegal_tile_override() {
        clean(|| {
            let illegal = TileParams {
                bk: 15,
                ..TileParams::DEFAULT_MATMUL
            };
            let w = gemm_in_bucket(12);
            let err = set_override(w.key(&arch()), Choice::MatmulTiled(illegal)).unwrap_err();
            assert!(matches!(err, OverrideError::IllegalTile(_)));
            // …and the default still stands.
            assert_eq!(
                resolve(&arch(), &w),
                Choice::MatmulTiled(TileParams::DEFAULT_MATMUL)
            );
        });
    }

    #[test]
    fn rejects_cross_family_override() {
        clean(|| {
            let err =
                set_override(key(GpuOp::Matmul, [0, 0, 12]), Choice::AttentionWarp).unwrap_err();
            assert!(matches!(err, OverrideError::OpMismatch { .. }));
        });
    }

    #[test]
    fn choice_round_trips_through_the_cache_format() {
        let mut all = vec![
            Choice::MatmulTiled(TileParams::DEFAULT_MATMUL),
            Choice::MatmulTiled(MATMUL_TILE_CANDIDATES[0]),
            Choice::MatmulTiled(MATMUL_TILE_CANDIDATES[4]),
            Choice::MatmulWmma,
            Choice::AttentionScalar,
            Choice::AttentionRow,
            Choice::AttentionWarp,
            Choice::AttentionWmma,
        ];
        all.extend(METAL_SGEMM_VARIANTS.iter().copied().map(Choice::MetalSgemm));
        all.extend(WGPU_MATMUL_VARIANTS.iter().copied().map(Choice::WgpuMatmul));
        for c in all {
            assert_eq!(Choice::decode(&c.encode()), Some(c), "round-trip {c:?}");
        }
    }

    /// Every Metal/wgpu variant belongs to the matmul family — an override that
    /// leaked into the attention key would route a GEMM kernel at an attention
    /// shape.
    #[test]
    fn backend_variants_are_matmul_family() {
        for v in METAL_SGEMM_VARIANTS {
            assert_eq!(Choice::MetalSgemm(*v).op(), GpuOp::Matmul);
        }
        for v in WGPU_MATMUL_VARIANTS {
            assert_eq!(Choice::WgpuMatmul(*v).op(), GpuOp::Matmul);
        }
    }

    /// The alignment rules are correctness, not preference: `Simd4x4` writes an
    /// unmasked 32×32 tile, so admitting it at `m % 32 != 0` overruns C into
    /// the next tensor in the arena.
    #[test]
    fn metal_shape_eligibility_rejects_misaligned_tiles() {
        assert!(MetalSgemm::Simd4x4.shape_eligible(64, 64, 64));
        assert!(!MetalSgemm::Simd4x4.shape_eligible(16, 1536, 2048)); // m % 32 != 0
        assert!(MetalSgemm::Simd64.shape_eligible(128, 64, 128));
        assert!(!MetalSgemm::Simd64.shape_eligible(128, 64, 96)); // n % 64 != 0
        assert!(!MetalSgemm::Simd.shape_eligible(8, 8, 12)); // n % 8 != 0
        // The unconstrained ones really are unconstrained.
        for v in [MetalSgemm::Mps, MetalSgemm::Naive, MetalSgemm::SimdPadded] {
            assert!(v.shape_eligible(7, 13, 29), "{v:?} should accept any shape");
        }
    }

    #[test]
    fn arch_labels_are_tab_safe_for_the_cache_format() {
        // wgpu adapter names are vendor strings; a tab would split a record.
        let a = GpuArch::wgpu("Metal", "Apple\tM4 Pro");
        assert!(!a.as_str().contains('\t'));
        assert_ne!(GpuArch::metal("apple9"), GpuArch::metal("apple8"));
        assert_ne!(
            GpuArch::wgpu("Metal", "Apple M4 Pro"),
            GpuArch::wgpu("Vulkan", "Apple M4 Pro")
        );
    }

    #[test]
    fn table_round_trips_through_save_load() {
        clean(|| {
            let w1 = gemm_in_bucket(12);
            // 2^20 query rows — a big prefill, on the ROCm rig's arch.
            let rocm = GpuArch::rocm("gfx908");
            let w2 = Workload::Attention {
                batch: 1,
                heads: 1,
                seq_q: 1 << 20,
            };
            set_override(
                w1.key(&arch()),
                Choice::MatmulTiled(MATMUL_TILE_CANDIDATES[3]),
            )
            .unwrap();
            set_override(w2.key(&rocm), Choice::AttentionRow).unwrap();
            let text = save_overrides();
            clear_overrides();
            assert_eq!(resolve(&rocm, &w2), default_choice(&rocm, &w2));
            let r = load_overrides(&text);
            assert_eq!(
                r,
                LoadReport {
                    applied: 2,
                    skipped: 0
                }
            );
            assert_eq!(
                resolve(&arch(), &w1),
                Choice::MatmulTiled(MATMUL_TILE_CANDIDATES[3])
            );
            assert_eq!(resolve(&rocm, &w2), Choice::AttentionRow);
        });
    }

    /// A cache written by a newer build (unknown variant, illegal tile, short
    /// record) must degrade to defaults rather than abort the process.
    #[test]
    fn stale_cache_records_are_skipped_not_fatal() {
        clean(|| {
            let text = "# header\n\
                        sm_86\tmatmul\t0.0.12\tmatmul_from_the_future\n\
                        sm_86\tmatmul\t0.0.13\tmatmul_tiled:64x64x15_t4x4_b16x16\n\
                        sm_86\ttensor_soup\t0.0.14\tattention_row\n\
                        sm_86\tmatmul\tnot_a_number\tmatmul_wmma\n\
                        truncated\n\
                        sm_86\tattention\t0.15.0\tattention_row\n";
            let r = load_overrides(text);
            assert_eq!(r.applied, 1);
            assert_eq!(r.skipped, 5);
            // batch*heads == 1 (band 0), seq_q == 2^15 (band 15) — the one good
            // record.
            let attn = Workload::Attention {
                batch: 1,
                heads: 1,
                seq_q: 1 << 15,
            };
            assert_eq!(resolve(&arch(), &attn), Choice::AttentionRow);
            // The illegal-tile record did not land.
            assert_eq!(
                resolve(&arch(), &gemm_in_bucket(13)),
                Choice::MatmulTiled(TileParams::DEFAULT_MATMUL)
            );
        });
    }

    /// A route report must distinguish measured routes from defaulted ones, and
    /// carry the evidence for the measured ones — otherwise it cannot answer the
    /// question it exists for.
    #[test]
    fn explain_separates_measured_routes_from_defaults() {
        clean(|| {
            let tuned = MATMUL_TILE_CANDIDATES[0];
            let w_tuned = gemm_in_bucket(12);
            let w_default = gemm_in_bucket(13);
            set_override_measured(
                w_tuned.key(&arch()),
                OverrideRecord {
                    choice: Choice::MatmulTiled(tuned),
                    speedup: Some(1.97),
                    holdouts: 2,
                },
            )
            .unwrap();

            let routes = explain(&arch(), &[w_tuned, w_default]);
            assert_eq!(routes.len(), 2, "two buckets, two routes");

            let m = routes
                .iter()
                .find(|r| matches!(r.source, RouteSource::Measured { .. }))
                .expect("the tuned bucket must report as measured");
            assert_eq!(m.choice, Choice::MatmulTiled(tuned));
            assert_eq!(
                m.source,
                RouteSource::Measured {
                    speedup: Some(1.97),
                    holdouts: 2
                }
            );

            let d = routes
                .iter()
                .find(|r| r.source == RouteSource::Default)
                .expect("the untuned bucket must report as default, not be omitted");
            assert_eq!(d.choice, Choice::MatmulTiled(TileParams::DEFAULT_MATMUL));

            // Rendering mentions both, and the speedup.
            let text = render_routes(&arch(), &routes);
            assert!(text.contains("measured"), "{text}");
            assert!(text.contains("default"), "{text}");
            assert!(text.contains("1.97x"), "{text}");
        });
    }

    /// Shapes sharing a bucket AND a schedule collapse into one row with a shape
    /// count — that count is what makes an over-broad bucket visible.
    #[test]
    fn explain_groups_shapes_that_share_a_route() {
        clean(|| {
            // Three shapes, all band [0, 12, 12], all on the default.
            let ws = [
                Workload::Matmul {
                    m: 1,
                    k: 4096,
                    n: 4096,
                },
                Workload::Matmul {
                    m: 1,
                    k: 5000,
                    n: 6000,
                },
                Workload::Matmul {
                    m: 1,
                    k: 7000,
                    n: 4096,
                },
            ];
            assert!(ws.iter().all(|w| w.bucket() == ws[0].bucket()));
            let routes = explain(&arch(), &ws);
            assert_eq!(routes.len(), 1);
            assert_eq!(routes[0].shapes, 3);
        });
    }

    /// An override tuned on one shape with no held-out confirmation must be
    /// flagged — that is exactly the leakage the holdout check exists to prevent,
    /// and a report that stayed quiet about it would launder the problem.
    #[test]
    fn render_flags_measured_routes_with_no_holdout_evidence() {
        clean(|| {
            let w = gemm_in_bucket(12);
            set_override_measured(
                w.key(&arch()),
                OverrideRecord {
                    choice: Choice::MatmulTiled(MATMUL_TILE_CANDIDATES[0]),
                    speedup: Some(1.4),
                    holdouts: 0,
                },
            )
            .unwrap();
            let text = render_routes(&arch(), &explain(&arch(), &[w]));
            assert!(
                text.contains("no held-out confirmation"),
                "an unvalidated override must be called out:\n{text}"
            );
        });
    }

    /// The evidence columns must survive a save/load round-trip, and a 4-column
    /// row (hand-written, or from a build predating the columns) must still apply.
    #[test]
    fn evidence_columns_round_trip_and_stay_backward_compatible() {
        clean(|| {
            let w = gemm_in_bucket(12);
            set_override_measured(
                w.key(&arch()),
                OverrideRecord {
                    choice: Choice::MatmulTiled(MATMUL_TILE_CANDIDATES[0]),
                    speedup: Some(1.97),
                    holdouts: 2,
                },
            )
            .unwrap();
            let text = save_overrides();
            clear_overrides();
            assert_eq!(load_overrides(&text).applied, 1);
            let rec = overrides()[0].1;
            assert_eq!(rec.holdouts, 2);
            assert!((rec.speedup.unwrap() - 1.97).abs() < 1e-3);

            // A 4-column row still loads; evidence is simply absent.
            clear_overrides();
            let short = "sm_86\tmatmul\t0.0.12\tmatmul_tiled:32x32x16_t4x4_b8x8\n";
            assert_eq!(load_overrides(short).applied, 1);
            let rec = overrides()[0].1;
            assert_eq!(rec.speedup, None);
            assert_eq!(rec.holdouts, 0);
        });
    }

    // ── Boundary / tail / coverage (CAKE §6) ────────────────────────────────
    //
    // §6: "validation covers representative and held-out inputs, boundary and
    // tail cases, overlapping or missing guards, and the fallback path." Held-out
    // inputs are the tuner's job; the rest are properties of the KEY and belong
    // here.

    /// Band edges are where a bucketed key is most likely to misroute: 1023 and
    /// 1024 differ by one element and land in different bands, so an override
    /// tuned at one must not leak across. Overlap is impossible by construction
    /// (the key is a function), so the real risk is a tuned route silently
    /// claiming its neighbour.
    #[test]
    fn band_edges_do_not_leak_a_tuned_route_to_the_neighbour() {
        clean(|| {
            let below = Workload::Matmul {
                m: 1,
                k: 1024,
                n: 1023,
            };
            let at = Workload::Matmul {
                m: 1,
                k: 1024,
                n: 1024,
            };
            assert_ne!(
                below.bucket(),
                at.bucket(),
                "1023 and 1024 must fall in different bands"
            );
            let tuned = MATMUL_TILE_CANDIDATES[0];
            set_override(at.key(&arch()), Choice::MatmulTiled(tuned)).unwrap();
            assert_eq!(resolve(&arch(), &at), Choice::MatmulTiled(tuned));
            assert_eq!(
                resolve(&arch(), &below),
                Choice::MatmulTiled(TileParams::DEFAULT_MATMUL),
                "the shape one element below the band edge must keep the default"
            );
        });
    }

    /// Tail cases: degenerate and extreme shapes must resolve, not panic. A
    /// dispatcher that dies on a 1-element or saturating shape is worse than one
    /// that routes it suboptimally.
    #[test]
    fn tail_shapes_resolve_without_panicking() {
        clean(|| {
            for w in [
                Workload::Matmul { m: 0, k: 0, n: 0 },
                Workload::Matmul { m: 1, k: 1, n: 1 },
                Workload::Matmul {
                    m: usize::MAX,
                    k: usize::MAX,
                    n: usize::MAX,
                },
                Workload::Attention {
                    batch: 0,
                    heads: 0,
                    seq_q: 0,
                },
                Workload::Attention {
                    batch: 1,
                    heads: 1,
                    seq_q: 1,
                },
            ] {
                // Must not panic, and must land inside the clamped band range.
                let b = w.bucket();
                assert!(b.0.iter().all(|band| *band <= MAX_BUCKET));
                let _ = resolve(&arch(), &w);
            }
        });
    }

    /// The route map must be TOTAL over the declared domain — every shape
    /// resolves, and `explain` accounts for every one of them. A "missing guard"
    /// in this design shows up as a shape that no route covers.
    #[test]
    fn every_declared_shape_is_covered_by_exactly_one_route() {
        clean(|| {
            let domain: Vec<Workload> = vec![
                Workload::Matmul {
                    m: 1,
                    k: 1024,
                    n: 1024,
                },
                Workload::Matmul {
                    m: 1,
                    k: 4096,
                    n: 4096,
                },
                Workload::Matmul {
                    m: 32,
                    k: 4096,
                    n: 4096,
                },
                Workload::Matmul {
                    m: 512,
                    k: 2048,
                    n: 2048,
                },
                Workload::Matmul {
                    m: 2048,
                    k: 2048,
                    n: 2048,
                },
                Workload::Attention {
                    batch: 1,
                    heads: 32,
                    seq_q: 1,
                },
                Workload::Attention {
                    batch: 1,
                    heads: 32,
                    seq_q: 4096,
                },
            ];
            let routes = explain(&arch(), &domain);
            // Every shape is attributed to exactly one route.
            let covered: usize = routes.iter().map(|r| r.shapes).sum();
            assert_eq!(
                covered,
                domain.len(),
                "route report lost {} shape(s)",
                domain.len() - covered
            );
            // And the fallback path is real: with no overrides installed, every
            // route is the compile-time default rather than an absence.
            assert!(routes.iter().all(|r| r.source == RouteSource::Default));
            assert!(!routes.is_empty());
        });
    }

    /// Attention's two axes must both matter. If `seq_q` were folded into the
    /// row count, decode and prefill would share a route again.
    #[test]
    fn attention_bands_separate_both_axes() {
        let rows_differ = Workload::Attention {
            batch: 1,
            heads: 8,
            seq_q: 128,
        }
        .bucket();
        let seq_differs = Workload::Attention {
            batch: 1,
            heads: 32,
            seq_q: 32,
        }
        .bucket();
        // Same product (1024) but different (rows, seq) split.
        assert_ne!(rows_differ, seq_differs);
    }

    /// Journal records must round-trip through the line format and, crucially,
    /// must survive a detail string containing tabs or newlines — a regression
    /// list is free text, and one stray tab would shift every later column.
    #[test]
    fn decision_records_encode_safely() {
        let r = DecisionRecord {
            at_unix_s: 1_771_000_000,
            arch: GpuArch::cuda(8, 6),
            op: GpuOp::Matmul,
            bucket: ShapeBucket([5, 12, 12]),
            outcome: DecisionOutcome::RejectedByHoldout {
                choice: Choice::MatmulTiled(MATMUL_TILE_CANDIDATES[0]),
                speedup: 1.39,
                detail: "(63,4096,6000)\t0.86x\nsecond line".to_string(),
            },
            tuned_shape: "32x4096x4096".to_string(),
        };
        let line = r.encode();
        assert!(!line.contains('\n'), "a newline would split the record");
        assert_eq!(
            line.matches('\t').count(),
            7,
            "exactly 8 columns expected: {line}"
        );
        assert!(line.contains("rejected-holdout"));
        assert!(line.contains("32x4096x4096"));

        // All three outcomes render, and the header names the columns.
        for outcome in [
            DecisionOutcome::Installed {
                choice: Choice::MatmulTiled(TileParams::DEFAULT_MATMUL),
                speedup: 1.97,
                holdouts: 2,
            },
            DecisionOutcome::DefaultHeld { best_speedup: 1.00 },
        ] {
            let rec = DecisionRecord {
                outcome,
                ..r.clone()
            };
            assert_eq!(rec.encode().matches('\t').count(), 7);
        }
        let hdr = decision_journal_header();
        assert!(hdr.contains("append-only"));
        // Both header lines must be comments with no stray indent, or a reader
        // (or `awk -F'\t'`) sees a phantom leading field.
        for line in hdr.lines() {
            assert!(
                line.starts_with('#'),
                "journal header line is not a comment: {line:?}"
            );
        }
        assert_eq!(render_decisions(&[r.clone(), r]).lines().count(), 2);
    }

    #[test]
    fn arch_labels_separate_vendors() {
        assert!(GpuArch::cuda(8, 6).is_cuda());
        assert!(!GpuArch::rocm("gfx908").is_cuda());
        assert_ne!(GpuArch::cuda(8, 6), GpuArch::cuda(9, 0));
    }
}

// ── Route reporting ─────────────────────────────────────────────────────────

/// Why a workload is on the schedule it is on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RouteSource {
    /// The compile-time default fired — no measurement exists for this bucket.
    Default,
    /// A measured override fired. `speedup` is what it bought on the tuned shape
    /// (`None` if the record carried no timing), `holdouts` how many unseen shapes
    /// confirmed it.
    Measured { speedup: Option<f32>, holdouts: u16 },
}

impl RouteSource {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Measured { .. } => "measured",
        }
    }
}

/// One row of a route report: a bucket, the schedule it routes to, and the
/// evidence for that choice.
#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    pub bucket: ShapeBucket,
    pub op: GpuOp,
    pub choice: Choice,
    pub source: RouteSource,
    /// How many of the reported workloads landed on this route.
    pub shapes: usize,
}

/// Resolve every workload and group the results into routes.
///
/// This is the answer to "what is actually running, and on what evidence" — the
/// analog of CAKE's Figure 9, which breaks a dispatcher-backed family into its
/// per-route speedups. That breakdown is the point: its Flash-KMeans routes span
/// 1.037× to 3.757× under an overall 1.803×, so reporting only the aggregate
/// would hide both the 3.76× win and the routes contributing nothing. It is also
/// the report that makes a bad *key* obvious — a bucket collecting shapes that
/// should not share a schedule shows up here as one route with a suspiciously
/// large `shapes` count.
pub fn explain(arch: &GpuArch, workloads: &[Workload]) -> Vec<Route> {
    let table = table()
        .overrides
        .lock()
        .expect("gpu dispatch table poisoned");
    let mut rows: Vec<Route> = Vec::new();
    for w in workloads {
        let key = w.key(arch);
        let (choice, source) = match table.get(&key) {
            Some(r) => (
                r.choice,
                RouteSource::Measured {
                    speedup: r.speedup,
                    holdouts: r.holdouts,
                },
            ),
            None => (default_choice(arch, w), RouteSource::Default),
        };
        match rows
            .iter_mut()
            .find(|r| r.bucket == key.bucket && r.op == key.op && r.choice == choice)
        {
            Some(existing) => existing.shapes += 1,
            None => rows.push(Route {
                bucket: key.bucket,
                op: key.op,
                choice,
                source,
                shapes: 1,
            }),
        }
    }
    rows.sort_by_key(|r| (r.op, r.bucket));
    rows
}

/// Render [`explain`] as a table.
///
/// Deliberately reports routes that are on the *default* too. A report that
/// listed only the tuned routes would read as "everything is tuned" — the same
/// silent-coverage failure as a tuner that prints only its winners.
pub fn render_routes(arch: &GpuArch, routes: &[Route]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "dispatch routes on {}", arch.as_str());
    if routes.is_empty() {
        s.push_str("  (no workloads reported)\n");
        return s;
    }
    s.push_str(
        "  op         bucket        shapes  route                               \
         source     speedup  holdouts\n",
    );
    let mut measured = 0usize;
    let mut unvalidated = 0usize;
    for r in routes {
        let (sp, hold) = match r.source {
            RouteSource::Measured { speedup, holdouts } => {
                measured += 1;
                if holdouts == 0 {
                    unvalidated += 1;
                }
                (
                    speedup
                        .map(|v| format!("{v:.2}x"))
                        .unwrap_or_else(|| "?".into()),
                    holdouts.to_string(),
                )
            }
            RouteSource::Default => ("-".to_string(), "-".to_string()),
        };
        let _ = writeln!(
            s,
            "  {:<10} {:<12} {:>6}  {:<34} {:<9} {:>8}  {}",
            r.op.as_str(),
            r.bucket.encode(),
            r.shapes,
            r.choice.encode(),
            r.source.label(),
            sp,
            hold
        );
    }
    let _ = writeln!(
        s,
        "  {} route(s): {measured} measured, {} on the compile-time default",
        routes.len(),
        routes.len() - measured
    );
    if unvalidated > 0 {
        // Not a warning about performance — a warning about *evidence*.
        let _ = writeln!(
            s,
            "  NOTE: {unvalidated} measured route(s) have no held-out confirmation; \
             they were tuned on one shape and applied to the whole bucket."
        );
    }
    s
}

// ── Retained decisions (audit journal) ──────────────────────────────────────

/// One line of the tuning decision log.
///
/// The tuning cache holds *current state*: which schedule each bucket is on right
/// now. That is what the runtime needs and nothing more. It cannot answer "when
/// did this bucket change, and on what evidence?" — so a bad override is
/// indistinguishable from a good one after the fact, and a regression traced to a
/// route has no history to bisect.
///
/// CAKE §4: *"retained results make decisions auditable and recurring findings
/// reusable."* This is the append-only half of that. It is deliberately separate
/// from the cache: the cache is read on every process start and must stay small,
/// while the journal only grows and is only read by a human.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRecord {
    /// Seconds since the Unix epoch, supplied by the caller — this crate does not
    /// read the clock (it has no dependencies and stays deterministic under test).
    pub at_unix_s: u64,
    pub arch: GpuArch,
    pub op: GpuOp,
    pub bucket: ShapeBucket,
    /// What was decided.
    pub outcome: DecisionOutcome,
    /// The exact shape that was measured, for reproducibility.
    pub tuned_shape: String,
}

/// What a tuning run concluded for one bucket.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionOutcome {
    /// A winner was installed.
    Installed {
        choice: Choice,
        speedup: f32,
        holdouts: u16,
    },
    /// A candidate led the search but failed the held-out check. Recorded because
    /// a *rejection* is the more informative event — it says the bucket is
    /// heterogeneous, which is a fact about the KEY, not about the candidate.
    RejectedByHoldout {
        choice: Choice,
        speedup: f32,
        detail: String,
    },
    /// Nothing beat the compile-time default by the required margin.
    DefaultHeld { best_speedup: f32 },
}

impl DecisionOutcome {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Installed { .. } => "installed",
            Self::RejectedByHoldout { .. } => "rejected-holdout",
            Self::DefaultHeld { .. } => "default-held",
        }
    }
}

impl DecisionRecord {
    /// One tab-separated line. Same format discipline as the tuning cache:
    /// diffable, greppable, no serde dependency.
    pub fn encode(&self) -> String {
        let (choice, sp, extra) = match &self.outcome {
            DecisionOutcome::Installed {
                choice,
                speedup,
                holdouts,
            } => (
                choice.encode(),
                format!("{speedup:.4}"),
                format!("holdouts={holdouts}"),
            ),
            DecisionOutcome::RejectedByHoldout {
                choice,
                speedup,
                detail,
            } => (
                choice.encode(),
                format!("{speedup:.4}"),
                detail.replace(['\t', '\n'], " "),
            ),
            DecisionOutcome::DefaultHeld { best_speedup } => {
                ("-".to_string(), format!("{best_speedup:.4}"), String::new())
            }
        };
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}|{}",
            self.at_unix_s,
            self.arch.as_str(),
            self.op.as_str(),
            self.bucket.encode(),
            self.outcome.label(),
            choice,
            sp,
            self.tuned_shape,
            extra,
        )
    }
}

/// Header for a fresh journal file.
pub fn decision_journal_header() -> &'static str {
    "# rlx gpu dispatch decision journal v1 (append-only)\n\
     # unix_s\tarch\top\tbucket\toutcome\tchoice\tspeedup\tshape|detail\n"
}

/// Render records as journal lines, ready to append.
pub fn render_decisions(records: &[DecisionRecord]) -> String {
    let mut s = String::new();
    for r in records {
        s.push_str(&r.encode());
        s.push('\n');
    }
    s
}
