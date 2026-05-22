// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

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
