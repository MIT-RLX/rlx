// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Thread-local [`FusionTarget`] / [`FusionOptions`] for IO-gated fusion passes.

use std::cell::Cell;

use crate::fusion_pipeline::{FusionOptions, FusionTarget};

thread_local! {
    static ACTIVE_TARGET: Cell<Option<FusionTarget>> = const { Cell::new(None) };
    static ACTIVE_OPTS: Cell<Option<FusionOptions>> = const { Cell::new(None) };
}

/// Fusion target for the current compile (set by [`CompilePipeline`]).
pub fn active_fusion_target() -> Option<FusionTarget> {
    ACTIVE_TARGET.with(|c| c.get())
}

/// Fusion options for the current compile (IO-gated passes).
pub fn active_fusion_options() -> Option<FusionOptions> {
    ACTIVE_OPTS.with(|c| c.get())
}

/// Run `f` with `target` installed for IO-gated passes (single-threaded compile).
pub fn with_fusion_target<T>(target: FusionTarget, f: impl FnOnce() -> T) -> T {
    with_fusion_context(target, FusionOptions::default(), f)
}

/// Run `f` with target + options installed for IO-gated passes.
pub fn with_fusion_context<T>(
    target: FusionTarget,
    opts: FusionOptions,
    f: impl FnOnce() -> T,
) -> T {
    ACTIVE_TARGET.with(|t| {
        ACTIVE_OPTS.with(|o| {
            let prev_t = t.get();
            let prev_o = o.get();
            t.set(Some(target));
            o.set(Some(opts));
            let out = f();
            t.set(prev_t);
            o.set(prev_o);
            out
        })
    })
}
