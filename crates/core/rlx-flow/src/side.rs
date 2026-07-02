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

//! Side-output collection (KV cache taps, auxiliary heads, …).

use std::sync::{Arc, Mutex};

use rlx_ir::HirNodeId;

/// Collects extra graph outputs emitted by side-effect stages (e.g. KV taps).
#[derive(Debug, Clone, Default)]
pub struct SideOutputs {
    inner: Arc<Mutex<Vec<HirNodeId>>>,
}

impl SideOutputs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Shared handle for side-effect stages (KV taps, …).
    pub fn inner(&self) -> Arc<Mutex<Vec<HirNodeId>>> {
        Arc::clone(&self.inner)
    }

    pub fn drain(&self) -> Vec<HirNodeId> {
        self.inner.lock().expect("side outputs").clone()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().expect("side outputs").is_empty()
    }
}
