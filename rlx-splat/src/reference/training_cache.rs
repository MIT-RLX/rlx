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
//! Thread-local stash so [`super::packed_backward`] can reuse a training forward trace
//! (avoids re-project/raster inside [`rlx_ir::Op::GaussianSplatRenderBackward`] on the same thread).

use std::cell::RefCell;

use super::training::TrainingForward;

thread_local! {
    static CACHE: RefCell<Option<*const TrainingForward>> = const { RefCell::new(None) };
}

/// Pin `forward` for the next arena/host backward on this thread. Call [`clear_training_forward_cache`] after.
pub fn set_training_forward_cache(forward: &TrainingForward) {
    CACHE.with(|c| *c.borrow_mut() = Some(std::ptr::from_ref(forward)));
}

/// Drop any stashed forward (required before the next training step on this thread).
pub fn clear_training_forward_cache() {
    CACHE.with(|c| *c.borrow_mut() = None);
}

pub(crate) fn cached_training_forward<'a>() -> Option<&'a TrainingForward> {
    CACHE.with(|c| c.borrow().map(|p| unsafe { &*p }))
}
