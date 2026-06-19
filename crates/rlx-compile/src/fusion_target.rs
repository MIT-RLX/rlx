// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Thread-local [`FusionTarget`] for IO-gated fusion passes during compile.

use std::cell::Cell;

use crate::fusion_pipeline::FusionTarget;

thread_local! {
    static ACTIVE_TARGET: Cell<Option<FusionTarget>> = const { Cell::new(None) };
}

/// Fusion target for the current compile (set by [`CompilePipeline`]).
pub fn active_fusion_target() -> Option<FusionTarget> {
    ACTIVE_TARGET.with(|c| c.get())
}

/// Run `f` with `target` installed for IO-gated passes (single-threaded compile).
pub fn with_fusion_target<T>(target: FusionTarget, f: impl FnOnce() -> T) -> T {
    ACTIVE_TARGET.with(|c| {
        let prev = c.get();
        c.set(Some(target));
        let out = f();
        c.set(prev);
        out
    })
}
