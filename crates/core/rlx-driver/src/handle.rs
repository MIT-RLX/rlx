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

//! Persistent buffer handles — state that survives across forward passes.
//!
//! For inference of stateful models (KV-cache, beam search) and for
//! training (gradient accumulators, optimizer state), the runtime needs
//! buffers that persist beyond a single `compiled.run()`. The arena
//! is rebuilt every compile, so it can't carry state.
//!
//! `BufferHandle` is an opaque, stable identifier the user creates once
//! and binds at compile time. The backend allocates a separate "handles"
//! region (independent of the arena) and routes reads/writes there.
//!
//! Workflow:
//!
//! ```rust,ignore
//! let kv_cache = BufferHandle::new("kv", &[batch, max_seq, num_heads, head_dim]);
//! let session = Session::new(Device::Metal);
//! let mut compiled = session.compile_with(graph, &CompileOptions::new()
//!     .bind_handle(&kv_cache));
//!
//! for token in tokens {
//!     compiled.bind_handle("kv", &kv_cache_data); // initial value
//!     let logits = compiled.run(&[("token", &[token])]);
//!     kv_cache_data = compiled.read_handle("kv").unwrap();
//! }
//! ```

use rlx_ir::{DType, Shape};

/// External, persistent buffer reference. Created once, bound at compile,
/// carried across many `compiled.run()` invocations.
#[derive(Debug, Clone)]
pub struct BufferHandle {
    pub name: String,
    pub shape: Shape,
}

impl BufferHandle {
    pub fn new(name: impl Into<String>, dims: &[usize], dtype: DType) -> Self {
        Self {
            name: name.into(),
            shape: Shape::new(dims, dtype),
        }
    }

    /// Total byte size of this handle.
    pub fn byte_size(&self) -> usize {
        self.shape.size_bytes().unwrap_or(0)
    }

    /// Number of elements (for slicing as &[f32] etc.).
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements().unwrap_or(0)
    }
}
