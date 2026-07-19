// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Unified compile options.
//!
//! Replaces the historical mix of `compile()`, `compile_with_precision()`,
//! `compile_with_options()` with a single `Backend::compile(graph, &options)`.
//! New compile-time knobs can be added to `CompileOptions` without
//! changing the trait — backends just read what they care about.
//!
//! Builder-pattern API for ergonomics:
//!
//! ```rust,ignore
//! let opts = CompileOptions::new()
//!     .precision(Precision::F16)
//!     .policy(PrecisionPolicy::AutoMixed)
//!     .with_dce(true)
//!     .with_constant_folding(true);
//! ```

use crate::Precision;
use rlx_ir::OpKind;
use rlx_ir::logical_kernel::{KernelDispatchConfig, KernelDispatchPolicy};
use rlx_opt::rlx_compile::scaled_quant_insert::ScaledQuantConfig;
use rlx_opt::{FusionOptions, FusionTarget, PrecisionPolicy};
use std::collections::HashMap;

/// All knobs the compile pipeline understands.
/// Add new fields here rather than introducing new compile entry points.
#[derive(Debug, Clone)]
pub struct CompileOptions {
    /// Target numeric precision for execution. Default: F32.
    pub precision: Precision,
    /// Optional per-op precision policy (mixed precision rewrite).
    pub policy: Option<PrecisionPolicy>,
    /// Opt-in native low-precision GEMM. When set, every 2-D `Op::MatMul` is
    /// rewritten to a `ScaledMatMul` whose operands are dynamically quantized to
    /// this element format + scale layout — any [`ScaledFormat`], including a
    /// parameterized `Custom` (e.g. `f4e3m0`). Off by default; changes numerics.
    pub scaled_quant: Option<ScaledQuantConfig>,
    /// RNG policy for in-graph [`Op::RngNormal`] / [`Op::RngUniform`] nodes.
    pub rng: rlx_ir::RngOptions,
    /// Run dead-code elimination as part of compile. Default: true.
    pub dce: bool,
    /// Run constant folding. Default: true (cheap, only helps).
    pub constant_folding: bool,
    /// Verbose pass logging. Equivalent to `RLX_VERBOSE=1` or
    /// [`rlx_ir::env::set("RLX_VERBOSE", "1")`].
    pub verbose: bool,
    /// Override fusion pipeline target (default: inferred from device).
    pub fusion_target: Option<FusionTarget>,
    /// Per-target fusion toggles (Metal env overrides, skip fusion, …).
    pub fusion_opts: FusionOptions,
    /// Arena alignment for buffer planning. Default: 64.
    pub arena_alignment: usize,
    /// Panic at compile time if fusion diagnostics report missed patterns.
    pub assert_fusion_clean: bool,
    /// Backend op claim set for backend-aware fusion + post-fusion
    /// legalization. Set by [`Backend::compile`] implementations.
    pub supported_ops: Option<&'static [OpKind]>,
    /// When set, specialize symbolic dims before backend lowering.
    pub dim_binding: Option<rlx_ir::DimBinding>,
    /// Bake fixed param tensors into constants before DCE / constant folding.
    pub param_bindings: Option<HashMap<String, Vec<f32>>>,
    /// Packed U8 GGUF weights for `Op::Param` nodes in `DequantMatMul` (TPU HLO bake).
    pub quant_param_bindings: Option<HashMap<String, Vec<u8>>>,
    /// Native vs common IR lowering ([`KernelDispatchConfig`], `RLX_KERNEL_DISPATCH=common`).
    pub kernel_dispatch: KernelDispatchConfig,
    /// Prevent Metal from lowering the graph through MPSGraph.
    ///
    /// This keeps the regular Metal thunk schedule while retaining all other
    /// compile options. Off by default.
    pub disable_mpsgraph: bool,
    /// On backends that list `OpKind::Scan` (host-fallback), Scans with
    /// `length <= scan_unroll_max_length` are IR-unrolled into body replicas so
    /// they run as ordinary device ops. Longer Scans keep `ScanHost`.
    /// Default: **64**. `0` disables unroll on claiming backends; `u32::MAX`
    /// prefers unroll for every eligible Scan.
    pub scan_unroll_max_length: u32,
    /// Hoist the param-invariant subgraph (compute that depends only on weights,
    /// never on `Op::Input`) into a separate "prepare" graph run ONCE, feeding
    /// its outputs into the main graph across forwards. Complements
    /// `param_bindings` (which folds such compute away at compile when weight
    /// VALUES are known); this handles run-time-only weights. Opt-in; default
    /// off (also settable via `RLX_CACHE_PARAM_INVARIANT=1`).
    pub cache_param_invariant: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            precision: Precision::F32,
            policy: None,
            scaled_quant: None,
            rng: rlx_ir::RngOptions::default(),
            dce: true,
            constant_folding: true,
            verbose: false,
            fusion_target: None,
            fusion_opts: FusionOptions::default(),
            arena_alignment: 64,
            assert_fusion_clean: false,
            supported_ops: None,
            dim_binding: None,
            param_bindings: None,
            quant_param_bindings: None,
            kernel_dispatch: KernelDispatchConfig::from_env(),
            disable_mpsgraph: false,
            scan_unroll_max_length: 64,
            cache_param_invariant: rlx_ir::env::flag("RLX_CACHE_PARAM_INVARIANT"),
        }
    }
}

impl CompileOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn precision(mut self, p: Precision) -> Self {
        self.precision = p;
        self
    }
    pub fn policy(mut self, p: PrecisionPolicy) -> Self {
        self.policy = Some(p);
        self
    }
    /// Enable native low-precision GEMM for every 2-D matmul in the graph, in
    /// the given element format + scale layout. Accepts any [`ScaledFormat`] via
    /// [`ScaledQuantConfig`] — e.g. `ScaledQuantConfig::fp8_e4m3()`, or a
    /// parameterized `ScaledFormat::custom(3, 0)` (`f4e3m0`).
    pub fn scaled_quant(mut self, cfg: ScaledQuantConfig) -> Self {
        self.scaled_quant = Some(cfg);
        self
    }
    pub fn rng(mut self, rng: rlx_ir::RngOptions) -> Self {
        self.rng = rng;
        self
    }
    pub fn rng_backend(mut self, backend: rlx_ir::RngBackend) -> Self {
        self.rng.backend = backend;
        self
    }
    pub fn rng_seed(mut self, seed: u64) -> Self {
        self.rng.seed = seed;
        self
    }
    pub fn no_policy(mut self) -> Self {
        self.policy = None;
        self
    }
    pub fn with_dce(mut self, on: bool) -> Self {
        self.dce = on;
        self
    }
    pub fn with_constant_folding(mut self, on: bool) -> Self {
        self.constant_folding = on;
        self
    }
    pub fn with_verbose(mut self, on: bool) -> Self {
        self.verbose = on;
        self
    }
    pub fn fusion_target(mut self, target: FusionTarget) -> Self {
        self.fusion_target = Some(target);
        self
    }
    pub fn fusion_opts(mut self, opts: FusionOptions) -> Self {
        self.fusion_opts = opts;
        self
    }
    pub fn arena_alignment(mut self, bytes: usize) -> Self {
        self.arena_alignment = bytes;
        self
    }
    pub fn supported_ops(mut self, ops: &'static [OpKind]) -> Self {
        self.supported_ops = Some(ops);
        self
    }
    pub fn assert_fusion_clean(mut self, on: bool) -> Self {
        self.assert_fusion_clean = on;
        self
    }
    pub fn dim_binding(mut self, binding: rlx_ir::DimBinding) -> Self {
        self.dim_binding = Some(binding);
        self
    }
    pub fn param_bindings(mut self, bindings: HashMap<String, Vec<f32>>) -> Self {
        self.param_bindings = Some(bindings);
        self
    }
    /// Packed U8 weights for TPU GGUF `DequantMatMul` param bake at compile time.
    pub fn quant_param_bindings(mut self, bindings: HashMap<String, Vec<u8>>) -> Self {
        self.quant_param_bindings = Some(bindings);
        self
    }
    pub fn kernel_dispatch(mut self, policy: KernelDispatchPolicy) -> Self {
        self.kernel_dispatch.policy = policy;
        self
    }

    pub fn kernel_dispatch_config(mut self, config: KernelDispatchConfig) -> Self {
        self.kernel_dispatch = config;
        self
    }

    /// Prefer IR-unrolling Scans with `length <= max` on Scan-claiming backends.
    pub fn scan_unroll_max_length(mut self, max: u32) -> Self {
        self.scan_unroll_max_length = max;
        self
    }

    /// Force listed logical kernels to use common IR even when native is in `supported_ops`.
    pub fn force_common_kinds(mut self, kinds: &'static [OpKind]) -> Self {
        self.kernel_dispatch.force_common_kinds = kinds;
        self
    }

    /// Keep listed logical kernels native even under `ForceCommon` / missing from `supported_ops`.
    pub fn force_native_kinds(mut self, kinds: &'static [OpKind]) -> Self {
        self.kernel_dispatch.force_native_kinds = kinds;
        self
    }
}
