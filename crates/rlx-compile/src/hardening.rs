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
