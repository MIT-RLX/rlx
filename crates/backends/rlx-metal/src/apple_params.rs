// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The Apple kernel knobs, in one place.**
//!
//! ```no_run
//! use rlx_metal::apple_params::{AppleKernelParams, Precision, SyncScope};
//! use rlx_metal::occupancy::AppleGpuFamily;
//!
//! let p = AppleKernelParams::default()      // measured-best defaults
//!     .stages(2)
//!     .sync(SyncScope::Simdgroup)
//!     .precision(Precision::F16Storage);
//!
//! p.predict(AppleGpuFamily::M4);            // worth measuring?
//! p.defines();                              // the MSL prelude it implies
//! ```
//!
//! # One knob per bottleneck, and each one has a number behind it
//!
//! Every parameter here targets a limiter this project has actually measured on
//! Apple silicon. A knob without evidence is a knob nobody can reason about, so
//! there are deliberately only five:
//!
//! | parameter | bottleneck | measured |
//! |---|---|---|
//! | [`stages`](crate::apple_params::AppleKernelParams::stages) | occupancy | −9.5% throughput per doubling of threadgroup memory |
//! | [`sync`](crate::apple_params::AppleKernelParams::sync) | barrier cost | 259 `threadgroup_barrier` sites, 0 `simdgroup_barrier` |
//! | [`precision`](crate::apple_params::AppleKernelParams::precision) | bytes moved | f16 weights took decode 24 → 89 tps |
//! | [`tile`](crate::apple_params::AppleKernelParams::tile) | coalescing / padding | tiled-transpose coalescing 2× |
//! | [`encode`](crate::apple_params::AppleKernelParams::encode) | encode overhead | ~5–20 µs objc bridging per dispatch |
//!
//! # One environment variable, not five
//!
//! This tree already registers ~577 `RLX_*` variables, and five more would be
//! five more things to discover, document and keep in sync. The typed API above
//! is the interface; the environment gets a **single** variable holding a
//! compact spec:
//!
//! ```text
//! RLX_METAL_PARAMS="stages=2,sync=simd,precision=f16"
//! ```
//!
//! Unset means [`AppleKernelParams::default`](crate::apple_params::AppleKernelParams::default). An unparseable key or value is
//! **reported and ignored**, never silently defaulted — a typo that quietly
//! restores the default makes an A/B measure one path twice, which is the
//! `metal-variant-typo-silent` defect this tree has already shipped.

use crate::occupancy::{self, AppleGpuFamily, Verdict};

/// Which barrier scope a cross-thread handoff needs.
///
/// Apple's `simdgroup_barrier` orders 32 threads; `threadgroup_barrier` orders
/// the whole group and costs more. A kernel whose producer and consumer are the
/// *same* simdgroup — the `hgemm_simd_4x4` shape, where `sg_row`/`sg_col`
/// partition the work — only needs the cheap one.
///
/// This is a declaration about the schedule, not a preference: choosing
/// `Simdgroup` for a handoff that actually crosses simdgroups is a race.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncScope {
    /// Whole threadgroup. Always correct; the current behaviour everywhere.
    #[default]
    Threadgroup,
    /// One simdgroup. Cheaper, and only valid when the handoff does not leave it.
    Simdgroup,
}

/// Storage precision of the staged operands.
///
/// The single biggest measured Apple win in this tree came from halving weight
/// bytes, not from arithmetic: decode went 24 → 89 tps. Apple GPUs are
/// bytes-bound far more often than they are FLOP-bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// f32 staging and f32 accumulate. Bit-exact against the CPU reference.
    #[default]
    F32,
    /// f16 staging, f32 accumulate. Halves staged bytes; **not** bit-exact.
    F16Storage,
}

/// How dispatches reach the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Encode {
    /// One encoder call per dispatch.
    #[default]
    PerDispatch,
    /// Batch through an Indirect Command Buffer (`crate::icb`).
    ///
    /// Apple's answer to per-dispatch cost is fusing *encodes*, not kernels —
    /// which is why this is the Apple-shaped analogue of CAKE's megakernel and
    /// why `icb.rs` already exists.
    Indirect,
}

/// Apple kernel parameters.
///
/// `Default` is the measured-best configuration: one stage (deeper loses on
/// Apple), threadgroup-scope barriers (always correct), f32 (bit-exact), the
/// shipping tile, one encode per dispatch. Every non-default value is a
/// deliberate trade with a number attached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppleKernelParams {
    stages: usize,
    sync: SyncScope,
    precision: Precision,
    tile: usize,
    encode: Encode,
}

impl Default for AppleKernelParams {
    fn default() -> Self {
        Self {
            stages: 1,
            sync: SyncScope::Threadgroup,
            precision: Precision::F32,
            tile: 16,
            encode: Encode::PerDispatch,
        }
    }
}

impl AppleKernelParams {
    /// Rotation depth of the staged tiles. `1` disables rotation.
    ///
    /// Above 1 this costs threadgroup memory, and on Apple that costs
    /// occupancy — see [`Self::predict`] before spending device time.
    pub fn stages(mut self, n: usize) -> Self {
        self.stages = n.max(1);
        self
    }

    /// Barrier scope for cross-thread handoffs.
    pub fn sync(mut self, s: SyncScope) -> Self {
        self.sync = s;
        self
    }

    /// Staged-operand precision.
    pub fn precision(mut self, p: Precision) -> Self {
        self.precision = p;
        self
    }

    /// Threadgroup tile edge.
    pub fn tile(mut self, edge: usize) -> Self {
        self.tile = edge.max(1);
        self
    }

    /// Dispatch encoding strategy.
    pub fn encode(mut self, e: Encode) -> Self {
        self.encode = e;
        self
    }

    pub fn stages_value(&self) -> usize {
        self.stages
    }
    pub fn sync_value(&self) -> SyncScope {
        self.sync
    }
    pub fn precision_value(&self) -> Precision {
        self.precision
    }
    pub fn tile_value(&self) -> usize {
        self.tile
    }
    pub fn encode_value(&self) -> Encode {
        self.encode
    }

    /// Bytes of threadgroup memory this configuration stages.
    ///
    /// Two tiles of `tile x tile`, `stages` deep, at the staged precision. This
    /// is the quantity Apple occupancy is priced in.
    pub fn threadgroup_bytes(&self) -> usize {
        let elem = match self.precision {
            Precision::F32 => 4,
            Precision::F16Storage => 2,
        };
        2 * self.tile * self.tile * elem * self.stages
    }

    /// Would this beat the default configuration on `chip`?
    ///
    /// Delegates to [`crate::occupancy::predict`], so an unmeasured chip
    /// returns [`Verdict::Unknown`] rather than a guess.
    pub fn predict(&self, chip: AppleGpuFamily) -> Verdict {
        occupancy::predict(
            chip,
            Self::default().tile(self.tile).threadgroup_bytes(),
            self.threadgroup_bytes(),
        )
    }

    /// The MSL prelude these parameters imply.
    ///
    /// This is how a parameter reaches the kernel: the emitter prepends it, so
    /// there is exactly one place a value is written down and no opportunity
    /// for the Rust-side and shader-side views to disagree.
    pub fn defines(&self) -> String {
        format!(
            "#define RLX_TS {}\n\
             #define RLX_STAGES {}\n\
             #define RLX_STAGE_T {}\n\
             #define RLX_BARRIER() {}\n",
            self.tile,
            self.stages,
            match self.precision {
                Precision::F32 => "float",
                Precision::F16Storage => "half",
            },
            match self.sync {
                SyncScope::Threadgroup => "threadgroup_barrier(mem_flags::mem_threadgroup)",
                SyncScope::Simdgroup => "simdgroup_barrier(mem_flags::mem_threadgroup)",
            }
        )
    }

    /// Parse a compact `k=v,k=v` spec. Unknown keys/values are an `Err`.
    ///
    /// Accepted: `stages=N`, `sync=threadgroup|simd`, `precision=f32|f16`,
    /// `tile=N`, `encode=dispatch|icb`.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut p = Self::default();
        for field in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (k, v) = field
                .split_once('=')
                .ok_or_else(|| format!("`{field}` is not `key=value`"))?;
            let v = v.trim();
            match k.trim() {
                "stages" => p.stages = v.parse().map_err(|_| format!("stages={v:?}"))?,
                "tile" => p.tile = v.parse().map_err(|_| format!("tile={v:?}"))?,
                "sync" => {
                    p.sync = match v {
                        "threadgroup" | "tg" => SyncScope::Threadgroup,
                        "simd" | "simdgroup" => SyncScope::Simdgroup,
                        _ => return Err(format!("sync={v:?} (threadgroup|simd)")),
                    }
                }
                "precision" => {
                    p.precision = match v {
                        "f32" => Precision::F32,
                        "f16" => Precision::F16Storage,
                        _ => return Err(format!("precision={v:?} (f32|f16)")),
                    }
                }
                "encode" => {
                    p.encode = match v {
                        "dispatch" | "per-dispatch" => Encode::PerDispatch,
                        "icb" | "indirect" => Encode::Indirect,
                        _ => return Err(format!("encode={v:?} (dispatch|icb)")),
                    }
                }
                other => return Err(format!("unknown key `{other}`")),
            }
        }
        if p.stages == 0 || p.tile == 0 {
            return Err("stages and tile must be >= 1".into());
        }
        Ok(p)
    }

    /// The measured-best configuration for a given GEMM shape.
    ///
    /// **Tile edge is shape-dependent on Apple, and strongly so.** Measured on
    /// an M4 Pro against the shipping `sgemm_tiled` (`tile=16`), every arm
    /// bit-exact, `serial` control at 0.998x:
    ///
    /// | shape class | `tile=8` | `tile=32` |
    /// |---|---|---|
    /// | m < 32 (decode, tiny batch) | **1.55 - 1.82x** | 0.52 - 0.57x |
    /// | m >= 32 (batch, prefill) | 0.96 - 0.99x | **1.08 - 1.10x** |
    ///
    /// The mechanism is the trade the occupancy model alone cannot see: a small
    /// tile launches more threadgroups and keeps the GPU fed when `m` is tiny,
    /// while a large tile reuses each staged element more and wins once there
    /// is enough work to fill the machine anyway. At `m = 1` a 32x32 tile masks
    /// off 31 of its 32 rows.
    ///
    /// # Caveat
    ///
    /// One run, on a machine at load 71, with a worst within-arm spread of
    /// 3.48x. The large effects (1.5x+, 0.5x) are far outside that; the
    /// `tile=32` prefill win of ~1.09x is consistent across eight shapes but is
    /// closer to the floor and deserves a quiet-machine confirmation before it
    /// is trusted to two digits.
    pub fn for_shape(m: usize, _k: usize, _n: usize) -> Self {
        let tile = if m < 32 { 8 } else { 32 };
        Self::default().tile(tile)
    }

    /// Every key, with its accepted values. The single source of truth for the
    /// spec, the CLI and `--help`, so the three cannot describe different sets.
    pub const KEYS: &'static [(&'static str, &'static str)] = &[
        ("stages", "N (rotation depth; 1 disables)"),
        ("sync", "threadgroup|simd"),
        ("precision", "f32|f16"),
        ("tile", "N (threadgroup tile edge)"),
        ("encode", "dispatch|icb"),
    ];

    /// Usage text, generated from [`Self::KEYS`].
    pub fn help() -> String {
        let mut out = String::from(
            "Apple kernel parameters — the same keys in all three front doors:\n\
             \x20 builder : AppleKernelParams::default().stages(2).sync(SyncScope::Simdgroup)\n\
             \x20 config  : RLX_METAL_PARAMS=\"stages=2,sync=simd\"\n\
             \x20 cli     : --metal-params stages=2,sync=simd   OR   --stages 2 --sync simd\n\n",
        );
        for (k, v) in Self::KEYS {
            out.push_str(&format!("  {k:<10} {v}\n"));
        }
        out.push_str("\nprecedence: CLI > config > default\n");
        out
    }

    /// Collect CLI arguments into a spec string.
    ///
    /// Accepts `--metal-params "k=v,k=v"` and the individual `--key value`
    /// forms, and **lowers both to the same spec** that [`Self::parse`] reads.
    /// One parser, three front doors — the alternative is three parsers that
    /// are supposed to agree and eventually will not.
    ///
    /// Returns `Ok(None)` when the arguments mention none of the keys.
    pub fn spec_from_args<S: AsRef<str>>(args: &[S]) -> Result<Option<String>, String> {
        let a: Vec<&str> = args.iter().map(|s| s.as_ref()).collect();
        let mut parts: Vec<String> = Vec::new();
        let mut i = 0;
        while i < a.len() {
            let tok = a[i];
            if tok == "--metal-params" {
                let v = a
                    .get(i + 1)
                    .ok_or("--metal-params needs a value".to_string())?;
                parts.push((*v).to_string());
                i += 2;
                continue;
            }
            if let Some(key) = tok.strip_prefix("--")
                && Self::KEYS.iter().any(|(k, _)| *k == key)
            {
                let v = a
                    .get(i + 1)
                    .ok_or_else(|| format!("--{key} needs a value"))?;
                parts.push(format!("{key}={v}"));
                i += 2;
                continue;
            }
            i += 1;
        }
        Ok((!parts.is_empty()).then(|| parts.join(",")))
    }

    /// Parse from CLI arguments. `Ok(None)` when unspecified.
    pub fn from_args<S: AsRef<str>>(args: &[S]) -> Result<Option<Self>, String> {
        match Self::spec_from_args(args)? {
            None => Ok(None),
            Some(spec) => Self::parse(&spec).map(Some),
        }
    }

    /// Read `RLX_METAL_PARAMS`, or the default when unset.
    ///
    /// A malformed spec is reported on stderr and the default is used. It is
    /// **not** silently accepted: a typo that quietly restores the default makes
    /// an A/B measure the same path twice and report it as parity.
    pub fn from_env() -> Self {
        match rlx_ir::env::var("RLX_METAL_PARAMS") {
            None => Self::default(),
            Some(spec) => Self::parse(&spec).unwrap_or_else(|why| {
                eprintln!(
                    "rlx-metal: RLX_METAL_PARAMS {why} — using defaults.\n{}",
                    Self::help()
                );
                Self::default()
            }),
        }
    }

    /// The one call an application should make: **CLI > config > default**.
    ///
    /// A bad CLI spec is reported and falls through to the config layer rather
    /// than aborting, matching how `from_env` treats a bad variable — the
    /// program still runs, and it says which layer it ended up using.
    pub fn resolve<S: AsRef<str>>(args: &[S]) -> Self {
        match Self::from_args(args) {
            Ok(Some(p)) => p,
            Ok(None) => Self::from_env(),
            Err(why) => {
                eprintln!(
                    "rlx-metal: bad --metal-params {why} — falling back to config.\n{}",
                    Self::help()
                );
                Self::from_env()
            }
        }
    }

    /// Round-trip spelling, so a configuration can be logged and pasted back.
    pub fn spec(&self) -> String {
        format!(
            "stages={},sync={},precision={},tile={},encode={}",
            self.stages,
            match self.sync {
                SyncScope::Threadgroup => "threadgroup",
                SyncScope::Simdgroup => "simd",
            },
            match self.precision {
                Precision::F32 => "f32",
                Precision::F16Storage => "f16",
            },
            self.tile,
            match self.encode {
                Encode::PerDispatch => "dispatch",
                Encode::Indirect => "icb",
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default must be the measured-best configuration, not an arbitrary
    /// one. Every Apple run in this tree says one stage wins.
    #[test]
    fn the_default_is_the_configuration_that_measured_best() {
        let d = AppleKernelParams::default();
        assert_eq!(d.stages_value(), 1);
        assert_eq!(d.sync_value(), SyncScope::Threadgroup);
        assert_eq!(d.precision_value(), Precision::F32);
        // 2 tiles x 16x16 x 4 B = 2048, which is what the MSL sgemm stages.
        assert_eq!(d.threadgroup_bytes(), 2048);
    }

    /// Every parameter must reach the shader. A knob the kernel cannot see is
    /// a knob that silently does nothing.
    #[test]
    fn every_parameter_reaches_the_msl() {
        let d = AppleKernelParams::default()
            .stages(3)
            .tile(8)
            .precision(Precision::F16Storage)
            .sync(SyncScope::Simdgroup)
            .defines();
        assert!(d.contains("#define RLX_STAGES 3"));
        assert!(d.contains("#define RLX_TS 8"));
        assert!(d.contains("#define RLX_STAGE_T half"));
        assert!(d.contains("simdgroup_barrier"));
        assert!(!d.contains("threadgroup_barrier(mem"));
    }

    /// f16 staging halves the bytes — the lever behind 24 -> 89 tps.
    #[test]
    fn f16_staging_halves_the_threadgroup_footprint() {
        let f32p = AppleKernelParams::default();
        let f16p = AppleKernelParams::default().precision(Precision::F16Storage);
        assert_eq!(f16p.threadgroup_bytes() * 2, f32p.threadgroup_bytes());
    }

    /// The knob is wired to the cost model, so a configuration can be judged
    /// before it is run. This is the CAKE §3.1 filtering step.
    #[test]
    fn a_deeper_rotation_is_predicted_slower_before_running() {
        let p = AppleKernelParams::default().stages(3);
        let v = p.predict(AppleGpuFamily::M4);
        assert!(matches!(v, Verdict::Slower(_)), "got {v:?}");
        assert!(!v.worth_measuring());
    }

    /// f16 *reduces* the footprint, so occupancy should not object — the
    /// opposite direction from adding stages, and worth pinning because the two
    /// levers move the same quantity in opposite ways.
    #[test]
    fn f16_staging_is_not_penalised_by_occupancy() {
        let p = AppleKernelParams::default().precision(Precision::F16Storage);
        assert!(p.predict(AppleGpuFamily::M4).worth_measuring());
    }

    /// The shape-keyed default must reproduce the measured winners, and the
    /// boundary must be where the measurement put it.
    #[test]
    fn for_shape_picks_the_measured_winner() {
        // Decode and tiny batch measured 1.55-1.82x at tile=8.
        for m in [1usize, 4, 16, 31] {
            assert_eq!(AppleKernelParams::for_shape(m, 4096, 4096).tile_value(), 8);
        }
        // Batch and prefill measured 1.08-1.10x at tile=32.
        for m in [32usize, 128, 512, 4096] {
            assert_eq!(AppleKernelParams::for_shape(m, 4096, 4096).tile_value(), 32);
        }
    }

    /// It must not quietly turn on anything that is not bit-exact. `f16`
    /// staging is a deliberate trade with a tolerance attached, never a default.
    #[test]
    fn for_shape_never_silently_costs_precision() {
        for m in [1usize, 64, 4096] {
            let p = AppleKernelParams::for_shape(m, 2048, 2048);
            assert_eq!(p.precision_value(), Precision::F32);
            assert_eq!(p.stages_value(), 1);
            assert_eq!(p.sync_value(), SyncScope::Threadgroup);
        }
    }

    #[test]
    fn a_spec_round_trips() {
        let p = AppleKernelParams::default()
            .stages(2)
            .sync(SyncScope::Simdgroup)
            .precision(Precision::F16Storage)
            .tile(32)
            .encode(Encode::Indirect);
        assert_eq!(AppleKernelParams::parse(&p.spec()), Ok(p));
    }

    /// A typo must be an error, not a silent default.
    ///
    /// `metal-variant-typo-silent`: an unrecognised variant name once fell
    /// through to the default in silence, so an A/B measured one path twice.
    #[test]
    fn a_typo_is_an_error_rather_than_a_silent_default() {
        for bad in [
            "stages=two",
            "sync=warp",
            "precision=bf16",
            "encode=graph",
            "stagez=2",
            "stages",
        ] {
            assert!(
                AppleKernelParams::parse(bad).is_err(),
                "`{bad}` should not parse"
            );
        }
    }

    #[test]
    fn an_empty_spec_is_the_default() {
        assert_eq!(
            AppleKernelParams::parse("").unwrap(),
            AppleKernelParams::default()
        );
    }

    /// **The equality guarantee.** Builder, config spec and both CLI spellings
    /// must produce the identical value — not merely similar behaviour.
    ///
    /// They do because there is one parser: the CLI forms lower to a spec
    /// string and `parse` reads it. Three parsers that are supposed to agree
    /// eventually do not.
    #[test]
    fn builder_config_and_cli_are_the_same_thing() {
        let builder = AppleKernelParams::default()
            .stages(2)
            .sync(SyncScope::Simdgroup)
            .precision(Precision::F16Storage);

        let config = AppleKernelParams::parse("stages=2,sync=simd,precision=f16").unwrap();

        let cli_compact =
            AppleKernelParams::from_args(&["--metal-params", "stages=2,sync=simd,precision=f16"])
                .unwrap()
                .unwrap();

        let cli_flags = AppleKernelParams::from_args(&[
            "prog",
            "--stages",
            "2",
            "--sync",
            "simd",
            "--precision",
            "f16",
        ])
        .unwrap()
        .unwrap();

        assert_eq!(builder, config, "builder != config");
        assert_eq!(config, cli_compact, "config != --metal-params");
        assert_eq!(cli_compact, cli_flags, "--metal-params != individual flags");
    }

    /// Unrelated arguments must not be mistaken for parameters.
    #[test]
    fn arguments_that_mention_no_key_are_not_a_configuration() {
        let none = AppleKernelParams::from_args(&["prog", "--verbose", "--out", "x.json"]).unwrap();
        assert_eq!(none, None, "unrelated flags must not fabricate a config");
    }

    /// A CLI typo is an error, exactly like a config typo.
    #[test]
    fn a_cli_typo_is_an_error_too() {
        assert!(AppleKernelParams::from_args(&["--sync", "warp"]).is_err());
        assert!(AppleKernelParams::from_args(&["--stages"]).is_err());
    }

    /// `--help` text is generated from `KEYS`, so it cannot list a key the
    /// parser does not accept, or omit one it does.
    #[test]
    fn help_lists_exactly_the_keys_the_parser_accepts() {
        let help = AppleKernelParams::help();
        for (k, _) in AppleKernelParams::KEYS {
            assert!(help.contains(k), "help omits `{k}`");
            // And the key really parses.
            let probe = match *k {
                "stages" | "tile" => format!("{k}=2"),
                "sync" => "sync=simd".into(),
                "precision" => "precision=f16".into(),
                "encode" => "encode=icb".into(),
                other => panic!("KEYS gained `{other}` with no probe"),
            };
            assert!(
                AppleKernelParams::parse(&probe).is_ok(),
                "`{probe}` rejected"
            );
        }
    }

    /// One variable, not five. If this ever needs a second, the compact spec
    /// should grow a key instead.
    #[test]
    fn the_whole_surface_is_one_environment_variable() {
        // Scan only the non-test half: this test mentions the pattern it looks
        // for, and a self-matching scan is how `pkill -f` kills its own shell.
        let src = include_str!("apple_params.rs");
        let code = src.split("#[cfg(test)]").next().expect("module body");
        let count = code.matches("rlx_ir::env::var(").count();
        assert_eq!(count, 1, "expected exactly one env read, found {count}");
        assert!(code.contains("RLX_METAL_PARAMS"));
    }
}
