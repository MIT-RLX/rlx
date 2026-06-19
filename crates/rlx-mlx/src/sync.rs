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

//! Serialize access to the MLX C++ runtime.
//!
//! MLX's device context and `mlx::compile` trace builder are not safe
//! under concurrent use from multiple Rust threads. Integration tests
//! run in parallel by default; without this lock, compiled-mode conv
//! repro tests can exit with SIGTRAP.

use std::sync::{Mutex, MutexGuard, OnceLock};

static MLX_RUNTIME_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Hold for the duration of any MLX FFI that builds or executes graphs.
pub(crate) fn runtime_guard() -> MutexGuard<'static, ()> {
    MLX_RUNTIME_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("mlx runtime lock poisoned")
}
