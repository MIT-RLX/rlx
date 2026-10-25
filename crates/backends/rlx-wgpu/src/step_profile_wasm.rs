// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`crate::step_profile`] on wasm32: always inactive. The native profiler keeps
//! wgpu buffers in a process-wide `static`, which the browser's `Rc`-based
//! WebGPU handles cannot be (`!Send + !Sync`); there are no env vars to request it
//! in a tab either.

pub fn requested() -> bool {
    false
}

pub(crate) fn required_feature(_adapter: &wgpu::Adapter) -> wgpu::Features {
    wgpu::Features::empty()
}

pub(crate) fn active() -> bool {
    false
}

pub(crate) fn begin_run() {}

pub(crate) fn pass_writes() -> Option<wgpu::ComputePassTimestampWrites<'static>> {
    None
}

/// Only reached when [`active`] (never here).
pub(crate) fn intern(_s: String) -> &'static str {
    ""
}

pub(crate) fn label_last(_name: &'static str) {}

pub(crate) fn resolve(_enc: &mut wgpu::CommandEncoder) {}

pub(crate) fn end_run() {}

pub fn print_summary() {}
