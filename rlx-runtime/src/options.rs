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
use rlx_ir::logical_kernel::{KernelDispatchConfig, KernelDispatchPolicy};
use rlx_ir::OpKind;
use rlx_opt::{FusionOptions, FusionTarget, PrecisionPolicy};

/// All knobs the compile pipeline understands.
/// Add new fields here rather than introducing new compile entry points.
#[derive(Debug, Clone)]
pub struct CompileOptions {
    /// Target numeric precision for execution. Default: F32.
    pub precision: Precision,
    /// Optional per-op precision policy (mixed precision rewrite).
    pub policy: Option<PrecisionPolicy>,
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
    /// Native vs common IR lowering ([`KernelDispatchConfig`], `RLX_KERNEL_DISPATCH=common`).
    pub kernel_dispatch: KernelDispatchConfig,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            precision: Precision::F32,
            policy: None,
            dce: true,
            constant_folding: true,
            verbose: false,
            fusion_target: None,
            fusion_opts: FusionOptions::default(),
            arena_alignment: 64,
            assert_fusion_clean: false,
            supported_ops: None,
            dim_binding: None,
            kernel_dispatch: KernelDispatchConfig::from_env(),
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
    pub fn kernel_dispatch(mut self, policy: KernelDispatchPolicy) -> Self {
        self.kernel_dispatch.policy = policy;
        self
    }

    pub fn kernel_dispatch_config(mut self, config: KernelDispatchConfig) -> Self {
        self.kernel_dispatch = config;
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
