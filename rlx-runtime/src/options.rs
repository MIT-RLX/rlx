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
use rlx_opt::PrecisionPolicy;

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
    /// Verbose pass logging. Equivalent to RLX_VERBOSE=1.
    pub verbose: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            precision: Precision::F32,
            policy: None,
            dce: true,
            constant_folding: true,
            verbose: false,
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
}
