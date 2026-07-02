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

//! Per-op ONNX lowering registry and grouped lowerers.

pub mod registry;

pub use registry::{
    CategoryCoverage, LowerStrategy, OP_REGISTRY, OpCategory, OpEntry, bundle_coverage_by_category,
    coverage_histogram, format_bundle_category_report, format_registry_dashboard, lowered_ops,
    op_is_registered, ops_in_category, registry_by_category, registry_lookup, rewritten_ops,
};
