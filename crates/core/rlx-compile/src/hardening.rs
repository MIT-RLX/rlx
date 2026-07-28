// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Debug-build verification hooks at compiler stage boundaries.
//!
//! Use [`debug_assert_graph!`] at pipeline stage boundaries. The macro
//! is compiled out entirely in release builds.

/// Stage-boundary IR check. **Debug builds only** — compiled out in release.
#[macro_export]
macro_rules! debug_assert_graph {
    ($graph:expr, $stage:expr) => {
        rlx_ir::debug_assert_valid!($graph, $stage);
    };
}
