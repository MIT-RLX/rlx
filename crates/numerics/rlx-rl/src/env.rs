// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// RLX — environment interface (no simulator bindings).

use crate::buffer::Transition;

/// Host-side MDP interface. Implement this for your simulator / robot stack.
///
/// RLX only sees [`Transition`] records; physics lives outside the compiler.
pub trait RlEnv {
    /// Initial state after reset (length = `state_dim`).
    fn reset(&mut self) -> Vec<f32>;

    /// Apply `action` (length = `action_dim`), return the transition.
    fn step(&mut self, action: &[f32]) -> Transition;
}
