// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-op ONNX lowering registry and grouped lowerers.

pub mod registry;

pub use registry::{
    CategoryCoverage, LowerStrategy, OP_REGISTRY, OpCategory, OpEntry, bundle_coverage_by_category,
    coverage_histogram, format_bundle_category_report, format_registry_dashboard, lowered_ops,
    op_is_registered, ops_in_category, registry_by_category, registry_lookup, rewritten_ops,
};
