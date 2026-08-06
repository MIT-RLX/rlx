// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed kernel planning and launch configuration for Metal kernel families.
//!
//! This module keeps policy and legality logic in plain Rust while using
//! declarative macros only for repetitive variant metadata.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelFamily {
    SdpaDecode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    F64,
    F32,
    BF16,
    F16,
    F8E4M3,
    F8E5M2,
    I8,
    I4,
    Q8,
    Q4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConvertPolicy {
    Native,
    Cast,
    DequantOnLoad,
    StochasticRound,
    DoubleSingleEmu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RolePrecision {
    pub storage: ScalarType,
    pub compute: ScalarType,
    pub convert: ConvertPolicy,
}

impl RolePrecision {
    pub const fn native(storage: ScalarType, compute: ScalarType) -> Self {
        Self {
            storage,
            compute,
            convert: ConvertPolicy::Native,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HybridPrecisionSpec {
    pub activations: RolePrecision,
    pub weights: RolePrecision,
    pub accum: RolePrecision,
    pub kv_cache: RolePrecision,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrecisionProfile {
    UltraLowLatency,
    Balanced,
    HighFidelity,
    Deterministic,
}

impl PrecisionProfile {
    pub fn default_spec(self) -> HybridPrecisionSpec {
        match self {
            Self::UltraLowLatency => HybridPrecisionSpec {
                activations: RolePrecision::native(ScalarType::F16, ScalarType::F16),
                weights: RolePrecision::native(ScalarType::F16, ScalarType::F16),
                accum: RolePrecision::native(ScalarType::F32, ScalarType::F32),
                kv_cache: RolePrecision::native(ScalarType::F16, ScalarType::F16),
            },
            Self::Balanced => HybridPrecisionSpec {
                activations: RolePrecision::native(ScalarType::BF16, ScalarType::F16),
                weights: RolePrecision::native(ScalarType::BF16, ScalarType::F16),
                accum: RolePrecision::native(ScalarType::F32, ScalarType::F32),
                kv_cache: RolePrecision::native(ScalarType::BF16, ScalarType::F16),
            },
            Self::HighFidelity => HybridPrecisionSpec {
                activations: RolePrecision::native(ScalarType::F32, ScalarType::F32),
                weights: RolePrecision::native(ScalarType::F32, ScalarType::F32),
                accum: RolePrecision::native(ScalarType::F32, ScalarType::F32),
                kv_cache: RolePrecision::native(ScalarType::F32, ScalarType::F32),
            },
            Self::Deterministic => HybridPrecisionSpec {
                activations: RolePrecision::native(ScalarType::F32, ScalarType::F32),
                weights: RolePrecision::native(ScalarType::F32, ScalarType::F32),
                accum: RolePrecision::native(ScalarType::F32, ScalarType::F32),
                kv_cache: RolePrecision::native(ScalarType::F32, ScalarType::F32),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadPolicy {
    pub max_threads_per_tg: u32,
    pub prefer_simdgroups: bool,
}

impl Default for ThreadPolicy {
    fn default() -> Self {
        Self {
            max_threads_per_tg: 256,
            prefer_simdgroups: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryPolicy {
    ArenaOnly,
    SharedIfFits { max_tg_bytes: u32 },
    ForceShared,
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self::SharedIfFits {
            max_tg_bytes: 32 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TunePolicy {
    pub enabled: bool,
    pub warmup_iters: u32,
    pub measure_iters: u32,
    pub cache_load: bool,
    pub persist_cache: bool,
    pub cache_max_entries: usize,
    pub cache_eviction: TuneCacheEviction,
    pub deterministic_mode: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuneCacheEviction {
    KeepLowBuckets,
    KeepHighBuckets,
    Lru,
}

impl TunePolicy {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            warmup_iters: 0,
            measure_iters: 0,
            cache_load: true,
            persist_cache: false,
            cache_max_entries: 256,
            cache_eviction: TuneCacheEviction::KeepLowBuckets,
            deterministic_mode: true,
        }
    }

    pub const fn balanced_defaults() -> Self {
        Self {
            enabled: true,
            warmup_iters: 2,
            measure_iters: 8,
            cache_load: true,
            persist_cache: true,
            cache_max_entries: 256,
            cache_eviction: TuneCacheEviction::KeepLowBuckets,
            deterministic_mode: false,
        }
    }
}

impl Default for TunePolicy {
    fn default() -> Self {
        Self::disabled()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenBucketStrategy {
    pub edges: &'static [u32],
}

impl TokenBucketStrategy {
    pub const fn decode_default() -> Self {
        Self {
            edges: &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        }
    }

    pub fn bucket_of(self, tokens: u32) -> u16 {
        for (i, edge) in self.edges.iter().enumerate() {
            if tokens <= *edge {
                return i as u16;
            }
        }
        self.edges.len() as u16
    }
}

impl Default for TokenBucketStrategy {
    fn default() -> Self {
        Self::decode_default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SdpaDecodeCandidate {
    pub split_k: u32,
    pub tg_size: u32,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub pad_q: u32,
    pub pad_kv: u32,
    pub use_f16_kv: bool,
    pub partial_reduce: bool,
}

impl SdpaDecodeCandidate {
    /// Stable text form for persistence and logs.
    pub fn tag(self) -> String {
        let kv = if self.use_f16_kv { "f16kv" } else { "f32kv" };
        let red = if self.partial_reduce {
            "partial"
        } else {
            "full"
        };
        format!(
            "skp{}_tg{}_tm{}_tn{}_tk{}_pq{}_pk{}_{}_{}",
            self.split_k,
            self.tg_size,
            self.tile_m,
            self.tile_n,
            self.tile_k,
            self.pad_q,
            self.pad_kv,
            kv,
            red
        )
    }

    /// Parse [`Self::tag`] text form.
    pub fn from_tag(tag: &str) -> Option<Self> {
        let mut split_k: Option<u32> = None;
        let mut tg_size: Option<u32> = None;
        let mut tile_m: Option<u32> = None;
        let mut tile_n: Option<u32> = None;
        let mut tile_k: Option<u32> = None;
        let mut pad_q: Option<u32> = None;
        let mut pad_kv: Option<u32> = None;
        let mut use_f16_kv: Option<bool> = None;
        let mut partial_reduce: Option<bool> = None;

        for part in tag.split('_') {
            if let Some(v) = part.strip_prefix("skp") {
                split_k = v.parse::<u32>().ok();
            } else if let Some(v) = part.strip_prefix("tg") {
                tg_size = v.parse::<u32>().ok();
            } else if let Some(v) = part.strip_prefix("tm") {
                tile_m = v.parse::<u32>().ok();
            } else if let Some(v) = part.strip_prefix("tn") {
                tile_n = v.parse::<u32>().ok();
            } else if let Some(v) = part.strip_prefix("tk") {
                tile_k = v.parse::<u32>().ok();
            } else if let Some(v) = part.strip_prefix("pq") {
                pad_q = v.parse::<u32>().ok();
            } else if let Some(v) = part.strip_prefix("pk") {
                pad_kv = v.parse::<u32>().ok();
            } else if part == "f16kv" {
                use_f16_kv = Some(true);
            } else if part == "f32kv" {
                use_f16_kv = Some(false);
            } else if part == "partial" {
                partial_reduce = Some(true);
            } else if part == "full" {
                partial_reduce = Some(false);
            }
        }

        Some(Self {
            split_k: split_k?,
            tg_size: tg_size?,
            // Backward-compat with older persisted rows that predate tile fields.
            tile_m: tile_m.unwrap_or(1),
            tile_n: tile_n.unwrap_or(1),
            tile_k: tile_k.unwrap_or(1),
            pad_q: pad_q.unwrap_or(1),
            pad_kv: pad_kv.unwrap_or(1),
            use_f16_kv: use_f16_kv?,
            partial_reduce: partial_reduce?,
        })
    }
}

macro_rules! define_sdpa_decode_variants {
    ($( $variant:ident => { split_k: $split_k:expr, tg_size: $tg_size:expr, tile_m: $tile_m:expr, tile_n: $tile_n:expr, tile_k: $tile_k:expr, pad_q: $pad_q:expr, pad_kv: $pad_kv:expr, use_f16_kv: $f16:expr, partial_reduce: $partial:expr } ),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum SdpaDecodeVariant {
            $( $variant ),+
        }

        impl SdpaDecodeVariant {
            pub const ALL: &'static [Self] = &[
                $( Self::$variant ),+
            ];

            pub const fn candidate(self) -> SdpaDecodeCandidate {
                match self {
                    $(
                        Self::$variant => SdpaDecodeCandidate {
                            split_k: $split_k,
                            tg_size: $tg_size,
                            tile_m: $tile_m,
                            tile_n: $tile_n,
                            tile_k: $tile_k,
                            pad_q: $pad_q,
                            pad_kv: $pad_kv,
                            use_f16_kv: $f16,
                            partial_reduce: $partial,
                        },
                    )+
                }
            }
        }
    };
}

define_sdpa_decode_variants! {
    BaseF32 => { split_k: 1, tg_size: 128, tile_m: 1, tile_n: 64, tile_k: 64, pad_q: 1, pad_kv: 1, use_f16_kv: false, partial_reduce: false },
    BaseF16Kv => { split_k: 1, tg_size: 128, tile_m: 1, tile_n: 64, tile_k: 64, pad_q: 1, pad_kv: 1, use_f16_kv: true, partial_reduce: false },
    SplitK4F16Kv => { split_k: 4, tg_size: 256, tile_m: 1, tile_n: 128, tile_k: 128, pad_q: 1, pad_kv: 1, use_f16_kv: true, partial_reduce: false },
    PartialF16Kv => { split_k: 4, tg_size: 256, tile_m: 1, tile_n: 128, tile_k: 128, pad_q: 1, pad_kv: 1, use_f16_kv: true, partial_reduce: true },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectiveWeights {
    pub latency: f32,
    pub error: f32,
    pub memory: f32,
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self {
            latency: 0.6,
            error: 0.3,
            memory: 0.1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelPlan {
    pub family: KernelFamily,
    pub precision_profile: PrecisionProfile,
    pub hybrid: HybridPrecisionSpec,
    pub thread_policy: ThreadPolicy,
    pub memory_policy: MemoryPolicy,
    pub tune_policy: TunePolicy,
    pub objective_weights: ObjectiveWeights,
    pub token_buckets: TokenBucketStrategy,
    pub candidates: Vec<SdpaDecodeCandidate>,
    pub default_candidate: SdpaDecodeCandidate,
    pub schema_version: u16,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LaunchConfig {
    pub family: KernelFamily,
    pub m: u32,
    pub tokens: u32,
    pub token_bucket: u16,
    pub batch: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub seq_q: u32,
    pub seq_kv: u32,
    pub candidate: SdpaDecodeCandidate,
    /// Deterministic, human-readable kernel label that reflects the generated
    /// launch configuration and selected candidate.
    pub kernel_name: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShapeBucket {
    pub m_bucket: u16,
    pub batch_bucket: u16,
    pub heads_bucket: u16,
    pub head_dim_bucket: u16,
    pub seq_q_bucket: u16,
    pub seq_kv_bucket: u16,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TuneCacheKey {
    pub schema_version: u16,
    pub family: KernelFamily,
    pub precision_profile: PrecisionProfileTag,
    pub token_bucket: u16,
    pub shape_bucket: ShapeBucket,
    pub device_fingerprint: String,
    pub env_mask: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrecisionProfileTag {
    UltraLowLatency,
    Balanced,
    HighFidelity,
    Deterministic,
    Custom,
}

impl From<PrecisionProfile> for PrecisionProfileTag {
    fn from(value: PrecisionProfile) -> Self {
        match value {
            PrecisionProfile::UltraLowLatency => Self::UltraLowLatency,
            PrecisionProfile::Balanced => Self::Balanced,
            PrecisionProfile::HighFidelity => Self::HighFidelity,
            PrecisionProfile::Deterministic => Self::Deterministic,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    EmptyCandidates,
    UnsupportedFamily(KernelFamily),
    InvalidCandidate(&'static str),
    InvalidPrecision(&'static str),
    InvalidObjectiveWeights,
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidates => write!(f, "kernel plan needs at least one candidate"),
            Self::UnsupportedFamily(family) => {
                write!(f, "unsupported kernel family in plan: {family:?}")
            }
            Self::InvalidCandidate(msg) => write!(f, "invalid candidate: {msg}"),
            Self::InvalidPrecision(msg) => write!(f, "invalid precision configuration: {msg}"),
            Self::InvalidObjectiveWeights => write!(
                f,
                "objective weights must be finite, non-negative, and sum to 1.0 (+/- 1e-3)"
            ),
        }
    }
}

impl std::error::Error for PlanError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchError {
    MissingField(&'static str),
    InvalidField(&'static str),
    CandidateNotPresent,
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(name) => write!(f, "missing launch field: {name}"),
            Self::InvalidField(name) => write!(f, "invalid launch field: {name}"),
            Self::CandidateNotPresent => write!(f, "selected candidate is not in the plan"),
        }
    }
}

impl std::error::Error for LaunchError {}

#[derive(Clone, Debug)]
pub struct KernelPlanBuilder {
    family: KernelFamily,
    precision_profile: PrecisionProfile,
    hybrid_override: Option<HybridPrecisionSpec>,
    allow_emulation: bool,
    thread_policy: ThreadPolicy,
    memory_policy: MemoryPolicy,
    tune_policy: TunePolicy,
    objective_weights: ObjectiveWeights,
    token_buckets: TokenBucketStrategy,
    candidates: Vec<SdpaDecodeCandidate>,
    schema_version: u16,
}

impl KernelPlanBuilder {
    pub fn new(family: KernelFamily) -> Self {
        let candidates = match family {
            KernelFamily::SdpaDecode => SdpaDecodeVariant::ALL
                .iter()
                .map(|v| v.candidate())
                .collect(),
        };
        Self {
            family,
            precision_profile: PrecisionProfile::Balanced,
            hybrid_override: None,
            allow_emulation: false,
            thread_policy: ThreadPolicy::default(),
            memory_policy: MemoryPolicy::default(),
            tune_policy: TunePolicy::default(),
            objective_weights: ObjectiveWeights::default(),
            token_buckets: TokenBucketStrategy::default(),
            candidates,
            schema_version: 1,
        }
    }

    pub fn precision_profile(mut self, profile: PrecisionProfile) -> Self {
        self.precision_profile = profile;
        self
    }

    pub fn hybrid_profile(mut self, profile: HybridPrecisionSpec) -> Self {
        self.hybrid_override = Some(profile);
        self
    }

    pub fn allow_emulation(mut self, enable: bool) -> Self {
        self.allow_emulation = enable;
        self
    }

    pub fn threads(mut self, policy: ThreadPolicy) -> Self {
        self.thread_policy = policy;
        self
    }

    pub fn memory(mut self, policy: MemoryPolicy) -> Self {
        self.memory_policy = policy;
        self
    }

    pub fn autotune(mut self, policy: TunePolicy) -> Self {
        self.tune_policy = policy;
        self
    }

    /// Configure persisted winner-table behavior for tune cache.
    pub fn tune_cache(mut self, load: bool, persist: bool, max_entries: usize) -> Self {
        self.tune_policy.cache_load = load;
        self.tune_policy.persist_cache = persist;
        self.tune_policy.cache_max_entries = max_entries.max(1);
        self
    }

    pub fn tune_cache_eviction(mut self, eviction: TuneCacheEviction) -> Self {
        self.tune_policy.cache_eviction = eviction;
        self
    }

    pub fn objective_weights(mut self, weights: ObjectiveWeights) -> Self {
        self.objective_weights = weights;
        self
    }

    pub fn token_buckets(mut self, strategy: TokenBucketStrategy) -> Self {
        self.token_buckets = strategy;
        self
    }

    pub fn candidates(mut self, candidates: Vec<SdpaDecodeCandidate>) -> Self {
        self.candidates = candidates;
        self
    }

    /// Override tile sizes across all current candidates in this plan.
    pub fn tile_sizes(mut self, tile_m: u32, tile_n: u32, tile_k: u32) -> Self {
        for c in &mut self.candidates {
            c.tile_m = tile_m;
            c.tile_n = tile_n;
            c.tile_k = tile_k;
        }
        self
    }

    /// Override sequence padding factors across all current candidates.
    pub fn padding(mut self, pad_q: u32, pad_kv: u32) -> Self {
        for c in &mut self.candidates {
            c.pad_q = pad_q;
            c.pad_kv = pad_kv;
        }
        self
    }

    pub fn schema_version(mut self, schema_version: u16) -> Self {
        self.schema_version = schema_version;
        self
    }

    pub fn build_plan(self) -> Result<KernelPlan, PlanError> {
        let hybrid = self
            .hybrid_override
            .unwrap_or_else(|| self.precision_profile.default_spec());

        validate_objective_weights(self.objective_weights)?;
        validate_tune_policy(self.tune_policy)?;
        validate_precision(self.family, hybrid, self.allow_emulation)?;

        if self.candidates.is_empty() {
            return Err(PlanError::EmptyCandidates);
        }

        for c in &self.candidates {
            validate_sdpa_candidate(*c, self.thread_policy, self.memory_policy)?;
        }

        let default_candidate =
            choose_default_candidate(self.precision_profile, self.thread_policy, &self.candidates);

        Ok(KernelPlan {
            family: self.family,
            precision_profile: self.precision_profile,
            hybrid,
            thread_policy: self.thread_policy,
            memory_policy: self.memory_policy,
            tune_policy: self.tune_policy,
            objective_weights: self.objective_weights,
            token_buckets: self.token_buckets,
            candidates: self.candidates,
            default_candidate,
            schema_version: self.schema_version,
        })
    }
}

#[derive(Clone, Debug)]
pub struct LaunchBuilder<'a> {
    plan: &'a KernelPlan,
    m: Option<u32>,
    tokens: Option<u32>,
    batch: Option<u32>,
    num_heads: Option<u32>,
    head_dim: Option<u32>,
    seq_q: Option<u32>,
    seq_kv: Option<u32>,
    candidate_override: Option<SdpaDecodeCandidate>,
}

impl<'a> LaunchBuilder<'a> {
    pub fn from_plan(plan: &'a KernelPlan) -> Self {
        Self {
            plan,
            m: None,
            tokens: None,
            batch: None,
            num_heads: None,
            head_dim: None,
            seq_q: None,
            seq_kv: None,
            candidate_override: None,
        }
    }

    pub fn m(mut self, m: u32) -> Self {
        self.m = Some(m);
        self
    }

    pub fn tokens(mut self, tokens: u32) -> Self {
        self.tokens = Some(tokens);
        self
    }

    pub fn batch(mut self, batch: u32) -> Self {
        self.batch = Some(batch);
        self
    }

    pub fn num_heads(mut self, num_heads: u32) -> Self {
        self.num_heads = Some(num_heads);
        self
    }

    pub fn head_dim(mut self, head_dim: u32) -> Self {
        self.head_dim = Some(head_dim);
        self
    }

    pub fn seq_q(mut self, seq_q: u32) -> Self {
        self.seq_q = Some(seq_q);
        self
    }

    pub fn seq_kv(mut self, seq_kv: u32) -> Self {
        self.seq_kv = Some(seq_kv);
        self
    }

    pub fn candidate(mut self, candidate: SdpaDecodeCandidate) -> Self {
        self.candidate_override = Some(candidate);
        self
    }

    pub fn build(self) -> Result<LaunchConfig, LaunchError> {
        let m = self.m.ok_or(LaunchError::MissingField("m"))?;
        let tokens = self.tokens.ok_or(LaunchError::MissingField("tokens"))?;
        let batch = self.batch.ok_or(LaunchError::MissingField("batch"))?;
        let num_heads = self
            .num_heads
            .ok_or(LaunchError::MissingField("num_heads"))?;
        let head_dim = self.head_dim.ok_or(LaunchError::MissingField("head_dim"))?;
        let seq_q = self.seq_q.ok_or(LaunchError::MissingField("seq_q"))?;
        let seq_kv = self.seq_kv.ok_or(LaunchError::MissingField("seq_kv"))?;

        if m == 0 {
            return Err(LaunchError::InvalidField("m"));
        }
        if tokens == 0 {
            return Err(LaunchError::InvalidField("tokens"));
        }
        if batch == 0 {
            return Err(LaunchError::InvalidField("batch"));
        }
        if num_heads == 0 {
            return Err(LaunchError::InvalidField("num_heads"));
        }
        if head_dim == 0 || head_dim > 512 {
            return Err(LaunchError::InvalidField("head_dim"));
        }
        if seq_q == 0 || seq_kv == 0 {
            return Err(LaunchError::InvalidField("seq_q/seq_kv"));
        }

        let selected_candidate = self
            .candidate_override
            .unwrap_or_else(|| choose_launch_candidate(self.plan, tokens, seq_kv));

        if !self.plan.candidates.contains(&selected_candidate) {
            return Err(LaunchError::CandidateNotPresent);
        }

        Ok(LaunchConfig {
            family: self.plan.family,
            m,
            tokens,
            token_bucket: self.plan.token_buckets.bucket_of(tokens),
            batch,
            num_heads,
            head_dim,
            seq_q,
            seq_kv,
            candidate: selected_candidate,
            kernel_name: render_kernel_name(
                self.plan,
                m,
                tokens,
                batch,
                num_heads,
                head_dim,
                seq_q,
                seq_kv,
                selected_candidate,
            ),
        })
    }
}

pub fn shape_bucket(launch: &LaunchConfig) -> ShapeBucket {
    ShapeBucket {
        m_bucket: bucket_pow2(launch.m),
        batch_bucket: bucket_pow2(launch.batch),
        heads_bucket: bucket_pow2(launch.num_heads),
        head_dim_bucket: bucket_pow2(launch.head_dim),
        seq_q_bucket: bucket_pow2(launch.seq_q),
        seq_kv_bucket: bucket_pow2(launch.seq_kv),
    }
}

pub fn tune_cache_key(
    plan: &KernelPlan,
    launch: &LaunchConfig,
    device_fingerprint: impl Into<String>,
    env_mask: u64,
) -> TuneCacheKey {
    TuneCacheKey {
        schema_version: plan.schema_version,
        family: plan.family,
        precision_profile: plan.precision_profile.into(),
        token_bucket: launch.token_bucket,
        shape_bucket: shape_bucket(launch),
        device_fingerprint: device_fingerprint.into(),
        env_mask,
    }
}

fn choose_default_candidate(
    profile: PrecisionProfile,
    thread_policy: ThreadPolicy,
    candidates: &[SdpaDecodeCandidate],
) -> SdpaDecodeCandidate {
    let preferred_f16 = matches!(
        profile,
        PrecisionProfile::UltraLowLatency | PrecisionProfile::Balanced
    );
    let preferred_partial = matches!(profile, PrecisionProfile::UltraLowLatency);

    candidates
        .iter()
        .copied()
        .filter(|c| c.tg_size <= thread_policy.max_threads_per_tg)
        .find(|c| c.use_f16_kv == preferred_f16 && c.partial_reduce == preferred_partial)
        .or_else(|| {
            candidates
                .iter()
                .copied()
                .find(|c| c.use_f16_kv == preferred_f16)
        })
        .unwrap_or(candidates[0])
}

fn choose_launch_candidate(plan: &KernelPlan, tokens: u32, seq_kv: u32) -> SdpaDecodeCandidate {
    // Conservative heuristic until an online tuner plugs in.
    let prefer_partial = tokens <= 8 || seq_kv > 1024;
    if prefer_partial {
        if let Some(c) = plan.candidates.iter().copied().find(|c| c.partial_reduce) {
            return c;
        }
    }
    plan.default_candidate
}

fn validate_objective_weights(weights: ObjectiveWeights) -> Result<(), PlanError> {
    let sum = weights.latency + weights.error + weights.memory;
    if !weights.latency.is_finite()
        || !weights.error.is_finite()
        || !weights.memory.is_finite()
        || weights.latency < 0.0
        || weights.error < 0.0
        || weights.memory < 0.0
        || (sum - 1.0).abs() > 1e-3
    {
        return Err(PlanError::InvalidObjectiveWeights);
    }
    Ok(())
}

fn validate_tune_policy(policy: TunePolicy) -> Result<(), PlanError> {
    if policy.cache_max_entries == 0 {
        return Err(PlanError::InvalidCandidate(
            "cache_max_entries must be at least 1",
        ));
    }
    match policy.cache_eviction {
        TuneCacheEviction::KeepLowBuckets
        | TuneCacheEviction::KeepHighBuckets
        | TuneCacheEviction::Lru => {}
    }
    Ok(())
}

fn validate_precision(
    family: KernelFamily,
    hybrid: HybridPrecisionSpec,
    allow_emulation: bool,
) -> Result<(), PlanError> {
    if matches!(
        hybrid.accum.compute,
        ScalarType::F16 | ScalarType::F8E4M3 | ScalarType::F8E5M2
    ) {
        return Err(PlanError::InvalidPrecision(
            "accumulator compute precision too low for stable reductions",
        ));
    }

    if !allow_emulation {
        let emulated = [
            hybrid.activations,
            hybrid.weights,
            hybrid.accum,
            hybrid.kv_cache,
        ]
        .iter()
        .any(|r| {
            matches!(
                r.convert,
                ConvertPolicy::StochasticRound
                    | ConvertPolicy::DoubleSingleEmu
                    | ConvertPolicy::DequantOnLoad
            )
        });
        if emulated {
            return Err(PlanError::InvalidPrecision(
                "emulation policies are disabled in this plan",
            ));
        }
    }

    match family {
        KernelFamily::SdpaDecode => {
            let kv_ok = matches!(
                hybrid.kv_cache.storage,
                ScalarType::F16 | ScalarType::BF16 | ScalarType::F32
            );
            if !kv_ok {
                return Err(PlanError::InvalidPrecision(
                    "sdpa decode requires kv-cache storage in F16/BF16/F32",
                ));
            }
            Ok(())
        }
    }
}

fn validate_sdpa_candidate(
    c: SdpaDecodeCandidate,
    thread_policy: ThreadPolicy,
    memory_policy: MemoryPolicy,
) -> Result<(), PlanError> {
    if c.split_k == 0 || c.split_k > 32 || !c.split_k.is_power_of_two() {
        return Err(PlanError::InvalidCandidate(
            "split_k must be a power-of-two in [1, 32]",
        ));
    }
    if !matches!(c.tg_size, 64 | 128 | 256 | 512) {
        return Err(PlanError::InvalidCandidate(
            "tg_size must be one of 64, 128, 256, 512",
        ));
    }
    if c.tile_m == 0 || c.tile_n == 0 || c.tile_k == 0 {
        return Err(PlanError::InvalidCandidate("tile sizes must be non-zero"));
    }
    if c.pad_q == 0 || c.pad_kv == 0 {
        return Err(PlanError::InvalidCandidate(
            "padding factors must be non-zero",
        ));
    }
    if c.tile_m > 1024 || c.tile_n > 1024 || c.tile_k > 4096 {
        return Err(PlanError::InvalidCandidate(
            "tile sizes exceed supported planner bounds",
        ));
    }
    if c.pad_q > 4096 || c.pad_kv > 4096 {
        return Err(PlanError::InvalidCandidate(
            "padding factors exceed supported planner bounds",
        ));
    }
    if c.tg_size > thread_policy.max_threads_per_tg {
        return Err(PlanError::InvalidCandidate(
            "tg_size exceeds thread policy max_threads_per_tg",
        ));
    }

    if matches!(memory_policy, MemoryPolicy::ArenaOnly) && c.partial_reduce {
        return Err(PlanError::InvalidCandidate(
            "partial_reduce requires shared memory staging",
        ));
    }

    if let MemoryPolicy::SharedIfFits { max_tg_bytes } = memory_policy {
        // Approximate staging footprint for decode partial path.
        let staging_bytes = if c.partial_reduce {
            16 * 1024
        } else {
            4 * 1024
        };
        if staging_bytes > max_tg_bytes {
            return Err(PlanError::InvalidCandidate(
                "candidate staging bytes exceed memory policy limit",
            ));
        }
    }

    Ok(())
}

fn bucket_pow2(v: u32) -> u16 {
    if v <= 1 {
        return 0;
    }
    (32 - (v - 1).leading_zeros()) as u16
}

fn render_kernel_name(
    plan: &KernelPlan,
    m: u32,
    tokens: u32,
    batch: u32,
    num_heads: u32,
    head_dim: u32,
    seq_q: u32,
    seq_kv: u32,
    candidate: SdpaDecodeCandidate,
) -> String {
    let family = match plan.family {
        KernelFamily::SdpaDecode => "sdpa_decode",
    };
    let profile = match plan.precision_profile {
        PrecisionProfile::UltraLowLatency => "ull",
        PrecisionProfile::Balanced => "bal",
        PrecisionProfile::HighFidelity => "hf",
        PrecisionProfile::Deterministic => "det",
    };
    let kv = if candidate.use_f16_kv {
        "f16kv"
    } else {
        "f32kv"
    };
    let red = if candidate.partial_reduce {
        "partial"
    } else {
        "full"
    };
    format!(
        "{family}_m{m}_tok{tokens}_b{batch}_h{num_heads}_d{head_dim}_sq{seq_q}_sk{seq_kv}_skp{}_tg{}_tm{}_tn{}_tk{}_pq{}_pk{}_{}_{}_{}",
        candidate.split_k,
        candidate.tg_size,
        candidate.tile_m,
        candidate.tile_n,
        candidate.tile_k,
        candidate.pad_q,
        candidate.pad_kv,
        kv,
        red,
        profile
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_builder_works_with_defaults() {
        let plan = KernelPlanBuilder::new(KernelFamily::SdpaDecode)
            .autotune(TunePolicy::balanced_defaults())
            .build_plan()
            .expect("plan should build");
        assert_eq!(plan.family, KernelFamily::SdpaDecode);
        assert!(!plan.candidates.is_empty());
    }

    #[test]
    fn launch_builder_validates_fields() {
        let plan = KernelPlanBuilder::new(KernelFamily::SdpaDecode)
            .build_plan()
            .expect("plan should build");

        let err = LaunchBuilder::from_plan(&plan)
            .m(1)
            .tokens(32)
            .batch(1)
            .num_heads(16)
            .head_dim(128)
            .seq_q(1)
            .build()
            .expect_err("missing seq_kv should fail");
        assert!(matches!(err, LaunchError::MissingField("seq_kv")));
    }

    #[test]
    fn tune_cache_key_has_shape_and_token_bucket() {
        let plan = KernelPlanBuilder::new(KernelFamily::SdpaDecode)
            .build_plan()
            .expect("plan should build");
        let launch = LaunchBuilder::from_plan(&plan)
            .m(1)
            .tokens(192)
            .batch(2)
            .num_heads(32)
            .head_dim(128)
            .seq_q(1)
            .seq_kv(192)
            .build()
            .expect("launch should build");

        let key = tune_cache_key(&plan, &launch, "m4pro-gpu", 0x10);
        assert_eq!(key.family, KernelFamily::SdpaDecode);
        assert_eq!(key.device_fingerprint, "m4pro-gpu");
        assert_eq!(key.env_mask, 0x10);
        assert_eq!(key.token_bucket, plan.token_buckets.bucket_of(192));
        assert!(launch.kernel_name.contains("sdpa_decode_m1_tok192"));
        assert!(launch.kernel_name.contains("_skp"));
        assert!(launch.kernel_name.contains("_tm"));
    }

    #[test]
    fn custom_tile_sizes_apply_to_candidates() {
        let plan = KernelPlanBuilder::new(KernelFamily::SdpaDecode)
            .tile_sizes(2, 96, 192)
            .build_plan()
            .expect("plan should build");
        assert!(
            plan.candidates
                .iter()
                .all(|c| c.tile_m == 2 && c.tile_n == 96 && c.tile_k == 192)
        );
    }

    #[test]
    fn custom_padding_applies_to_candidates() {
        let plan = KernelPlanBuilder::new(KernelFamily::SdpaDecode)
            .padding(2, 64)
            .build_plan()
            .expect("plan should build");
        assert!(
            plan.candidates
                .iter()
                .all(|c| c.pad_q == 2 && c.pad_kv == 64)
        );
    }

    #[test]
    fn tune_cache_config_applies_to_plan() {
        let plan = KernelPlanBuilder::new(KernelFamily::SdpaDecode)
            .tune_cache(true, false, 32)
            .tune_cache_eviction(TuneCacheEviction::Lru)
            .build_plan()
            .expect("plan should build");
        assert!(plan.tune_policy.cache_load);
        assert!(!plan.tune_policy.persist_cache);
        assert_eq!(plan.tune_policy.cache_max_entries, 32);
        assert_eq!(plan.tune_policy.cache_eviction, TuneCacheEviction::Lru);
    }
}
