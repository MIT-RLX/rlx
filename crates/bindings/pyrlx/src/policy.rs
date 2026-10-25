// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-op-kind precision policies for `Session(policy=…)`,
//! `Session.compile_with(…, policy=…)` and `Trainer(policy=…)`.
//!
//! A string names a preset; a dict gives an explicit precision per op kind:
//!
//! ```python
//! rlx.Session("metal", policy="mixed")
//! rlx.Session("metal", policy={"compute": "f16", "reduction": "f32"})
//! ```
//!
//! Op kinds a dict does not name stay f32 rather than following the lowest
//! precision present: a partial policy must not silently downcast what it did
//! not mention — `boundary` above all, since inputs and outputs that became
//! f16 would not round-trip.

use pyo3::prelude::*;
use rlx_bind::runtime::policy;
use rlx_runtime::PrecisionPolicy;

use crate::bind::{opt_json, to_py};

/// A policy from a preset name, a dict of op kinds, or `None`.
pub(crate) fn parse_policy(v: Option<&Bound<'_, PyAny>>) -> PyResult<Option<PrecisionPolicy>> {
    policy::parse(&opt_json(v)?).map_err(to_py)
}
